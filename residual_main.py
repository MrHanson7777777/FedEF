#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 残差联邦学习主程序

# 设置编码
import sys

# 强制设置UTF-8编码
if sys.platform.startswith('win'):
    import os
    os.system('chcp 65001 > nul')  #设置Windows控制台为UTF-8
    
# 确保print输出使用UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import os
import copy
import time
import pickle
import numpy as np
import pandas as pd
import torch
import math
import sys
from tqdm import tqdm
from datetime import timedelta, datetime

# 设置PyTorch和cuDNN选项以优化性能
torch.backends.cudnn.enabled = True #如果设置为 True，PyTorch 会调用 cuDNN 来加速深度学习中的卷积操作和其他相关操作
torch.backends.cudnn.benchmark = True  # 启用benchmark以自动选择最优算法
torch.backends.cudnn.deterministic = False  # 允许非确定性算法获得更好性能

# 设置输出编码
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

from tensorboardX import SummaryWriter

from options import args_parser
from update import test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, CNNCifar100, ResNet18Fed
from model_factory import get_model
from utils import get_dataset, exp_details
from residual_utils import LocalUpdateResidual, average_weights_residual, model_subtract, model_add, apply_residual_compression_fast, adaptive_client_aggregation, calculate_diversity_scores_residual, pack_sparse_residual, unpack_sparse_residual, calculate_communication_cost_dict, print_round_communication_stats, print_final_compression_stats



def main():
    start_time = time.time()
    
    print("[DEBUG] 残差联邦学习程序启动")
    
    # 设置日志路径
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')
    
    print("[DEBUG] 日志设置完成")
    
    # 解析命令行参数
    args = args_parser()
    exp_details(args)
    print("[DEBUG] 参数解析完成")
    
    # 设置设备
    device = setup_device(args)
    
    # 加载数据集
    print("[DEBUG] 开始加载数据集...")
    train_dataset, test_dataset, user_groups = get_dataset(args)
    print(f"[DEBUG] 数据集加载完成: train={len(train_dataset)}, test={len(test_dataset)}")
    
    # 构建模型
    global_model = build_model(args, train_dataset)
    global_model.to(device)
    global_model.train()

    # 设置模型到设备
    global_model.to(device)
    global_model.train()
    
    # 保存全局模型的权重
    global_weights = global_model.state_dict()
    
    # 初始化每个客户端的残差状态,后面用于上传
    client_residuals = {}
    print(f"[DEBUG] 初始化 {args.num_users} 个客户端的残差状态...")
    
    # 将残差保存在CPU上，只在需要时移动到GPU
    for i in range(args.num_users):
        client_residuals[i] = {key: torch.zeros_like(param).cpu() 
                              for key, param in global_weights.items()}
    
    print(f"[DEBUG] 上行残差状态初始化完成,保存在CPU内存中")
    
    # <--- 初始化服务器端的下放残差误差反馈状态 --->
    server_downlink_error = {key: torch.zeros_like(param).cpu()
                             for key, param in global_weights.items()}
    print(f"[DEBUG] 服务器下行误差反馈状态初始化完成")
    
    # <--- 初始化客户端同步模型状态 (所有客户端从零模型开始) --->
    zero_weights = {key: torch.zeros_like(param).cpu()
                    for key, param in global_weights.items()}
    client_synced_models = {i: copy.deepcopy(zero_weights) for i in range(args.num_users)}
    print(f"[DEBUG] 客户端同步模型初始化为零状态")
    
    # 统计变量
    communication_cost = []
    print_every = 2
    
    # <--- 下行通信的单独计时变量 --->
    total_downlink_pack_time = 0.0    # 服务器打包时间
    total_downlink_unpack_time = 0.0  # 客户端解包时间
    downlink_pack_count = 0
    downlink_unpack_count = 0
    
    # 初始化训练统计变量
    train_loss = []  # 用于记录每轮的平均训练损失
    global_test_accuracy = []  # 用于记录每轮的全局测试准确率
    communication_cost = []  # 用于记录每轮的通信开销（字节数）
    epoch_times = []  # 用于记录每轮的训练耗时（秒）
    improve_streak = 0  # 初始化连续提升计数器
    
    # 早停机制参数
    patience = args.stopping_rounds
    
    #自适应压缩所需的状态变量
    ema_alpha = 0.4  # EMA平滑因子，用于平滑准确率的变化，避免因单轮波动导致误判
    ema_acc = None   # 平滑后的准确率，初始值为None，后续会动态更新
    best_ema_acc = -1.0 # 记录最佳的平滑准确率，用于判断模型性能是否提升
    patience_counter = 0  # 耐心计数器，用于早停机制，记录连续未提升的轮次数
    best_global_weights = None # 确保best_global_weights在这里初始化，用于保存最佳模型权重

    #初始化历史记录,用于后续分析和可视化
    history = {
        'epoch': [],
        'test_accuracy': [],
        'avg_train_loss': [],
        'learning_rate': [],
        'compression_ratio': [],
        'communication_cost': []  # 添加通信开销记录
    }
    
    # 初始化压缩时间统计变量
    total_pack_time = 0.0      # 总打包时间 (毫秒)
    total_unpack_time = 0.0    # 总解包时间 (毫秒)  
    pack_count = 0             # 打包次数
    unpack_count = 0           # 解包次数
    
    # 在主函数中调用封装的打印函数  
    print_training_details(args=args, ema_alpha=ema_alpha, patience=patience)
    
    # 训练循环 - 使用tqdm显示进度
    pbar = tqdm(range(args.epochs), desc="Training Progress", 
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    
    #在训练循环 for epoch in pbar: 之前，初始化一个新的字典来存储平滑损失
    client_ema_losses = {}
    EMA_ALPHA = 0.3 # 平滑因子，可以作为超参数调整
    #---------------------初始化结束---------------------
    
    # <--- Phase 1 - 初始模型冷启动传输 --->
    initial_global_weights = global_model.state_dict()  # 目标G_0,服务器随机初始化得到的初始化模型
    server_side_synced_model = {key: torch.zeros_like(param).cpu() for key, param in initial_global_weights.items()}

    # 判断是否需要冷启动流式传输：有下行压缩且用户未明确禁用
    if args.downlink_compression == 'uniform':
        # 有下行压缩 - 执行冷启动流式传输
        # --- 保留原始的流式传输逻辑 (使用新的下行压缩比参数) ---
        INITIALIZATION_ROUNDS = int(1 / args.downlink_compression_ratio + 1)

        print(f"\n{'='*60}\n🚀 开始初始模型流式传输 (共 {INITIALIZATION_ROUNDS} 轮)\n{'='*60}")
        pbar_init = tqdm(range(INITIALIZATION_ROUNDS), desc="初始模型同步")

        for init_round in pbar_init:
            # 1. 计算当前轮的完整残差：目标模型 - 当前同步模型
            total_residual_to_send = model_subtract(initial_global_weights, server_side_synced_model)
            
            # 2. 对完整残差进行Top-K压缩（使用新的下行压缩比参数）
            compressed_residual = apply_residual_compression_fast(
                total_residual_to_send, args.downlink_compression_ratio
            )
            
            # 压缩模式：打包压缩后的残差用于传输
            pack_result = pack_sparse_residual(compressed_residual, enable_timing=True)
            packed_residual, pack_time = pack_result
            total_downlink_pack_time += pack_time
            downlink_pack_count += 1
                
            print(f"[INIT {init_round+1}/{INITIALIZATION_ROUNDS}] 压缩率: {args.downlink_compression_ratio:.1f}, 打包耗时: {pack_time:.2f}ms")

            # 4. 更新服务器端的客户端模拟模型（使用压缩后的参数）
            server_side_synced_model = model_add(server_side_synced_model, compressed_residual)

            # 5. 所有客户端接收并更新自己的同步模型
            # 每个客户端都接收打包后的数据并独立进行解包
            # 初始化解包时间为0
            max_client_unpack_time = 0.0

            for i in range(args.num_users):
                # 压缩模式：客户端接收打包数据并解包
                client_received_packed = copy.deepcopy(packed_residual)
                    
                # 客户端独立解包
                client_unpacked_params, client_unpack_time = unpack_sparse_residual(
                    client_received_packed, zero_weights, enable_timing=True
                )
                    
                # 更新最大解包时间
                max_client_unpack_time = max(max_client_unpack_time, client_unpack_time)
                
                # 客户端更新自己的同步模型
                client_synced_models[i] = model_add(client_synced_models[i], client_unpacked_params)            # 统计所有客户端的解包时间（并行情况下取最大值）
                total_downlink_unpack_time += max_client_unpack_time
                downlink_unpack_count += 1  # 一轮并行解包算作一次操作

            # 更新进度条
            pbar_init.set_postfix({
                'PackTime': f'{pack_time:.2f}ms', 
                'UnpackTime': f'{max_client_unpack_time:.2f}ms'
            })

        print(f"\n✅ 初始模型流式传输完成.\n{'='*60}")
    else:
        print(f"\n{'='*60}\n🚀 初始模型流式传输: 已跳过 (因下行未压缩,无需冷启动)")
        print(f"   所有 {args.num_users} 个客户端将直接同步完整模型.\n{'='*60}")
        
        # 所有客户端直接获得完整模型
        client_synced_models = {i: copy.deepcopy(initial_global_weights) for i in range(args.num_users)}
        server_side_synced_model = copy.deepcopy(initial_global_weights)
    
    # <--- 在主训练循环前初始化广播变量 --->
    residual_to_broadcast_packed = None  # 持久化存储每轮服务器计算的待广播残差
    
    # 现在，主训练循环开始
    for epoch in pbar:
        epoch_start_time = time.time()
        
        print(f"\n{'='*60}\n🔄 全局轮次 {epoch+1}/{args.epochs}\n{'='*60}")
        
        # 初始化本地残差和损失列表，用于存储每个客户端的训练结果
        local_residuals = [] # local_residuals存储每个客户端要上传的残差字典
        local_losses= [] # local_losses存储每个客户端的训练损失
        
        # 初始化本轮通信开销统计变量
        epoch_comm_cost = 0  # epoch_comm_cost用于记录当前轮次所有客户端的通信开销（字节数）
        
        # 初始化通信量统计变量
        total_nonzero_values = 0   # 压缩后的参数总量
        total_original_params = 0  # 压缩前的参数总量
        
        global_model.train()
        
        # 初始化客户端状态跟踪
        if not hasattr(args, 'client_history'):
            args.client_history = {
                'losses': {},           # 每个客户端的历史损失
                'last_selected': [],    # 所有历史选择的客户端（用于简单频率统计）
                'round_selections': [], # 按轮记录的选择历史（用于更精确的轮级分析）
                'performance_scores': {}
            }
        
        # 定义每轮选择的客户端数量
        m = max(int(args.frac * args.num_users), 1)

        # <--- 对所有客户端执行下行更新 --->
        if residual_to_broadcast_packed is not None:
            print(f"\n[DOWNLINK] 向所有 {args.num_users} 个客户端广播下行更新...")

            # 每个客户端并行接收服务器广播的残差,支持全量和压缩模式
            max_client_unpack_time = 0.0  # 记录最大解包时间（并行操作）
            for i in range(args.num_users):  # 遍历每一个用户
                if args.downlink_compression == 'uniform':
                    # 压缩模式：客户端接收打包数据并解包
                    client_received_packed = copy.deepcopy(residual_to_broadcast_packed)
                    
                    # 客户端独立解包
                    unpacked_downlink_residual, client_unpack_time = unpack_sparse_residual(
                        client_received_packed, zero_weights, enable_timing=True
                    )
                    
                    # 更新最大解包时间（并行操作取最大值）
                    max_client_unpack_time = max(max_client_unpack_time, client_unpack_time)
                else:
                    # 全量模式：直接使用密集张量，跳过解包步骤
                    unpacked_downlink_residual = copy.deepcopy(residual_to_broadcast_packed)
                
                # 客户端更新自己的同步模型
                client_synced_models[i] = model_add(client_synced_models[i], unpacked_downlink_residual)

            # 更新计时统计 (并行情况下取最大值)
            total_downlink_unpack_time += max_client_unpack_time
            downlink_unpack_count += 1  # 一轮并行解包算作一次操作
            if max_client_unpack_time > 0:  # 只有当压缩模式才打印
                print(f"[DOWNLINK] 所有客户端同步模型更新完成，最大解包耗时: {max_client_unpack_time:.2f}ms")
        else:
            print(f"\n[DOWNLINK] 第 {epoch+1} 轮：无下行更新（初始轮）")

        # 统一调用select_clients，由函数内部处理 'random' 和 'smart' 逻辑
        idxs_users = select_clients(epoch, args, user_groups, client_ema_losses, EMA_ALPHA, m)
        
        # 初始化并行上行打包时间记录
        max_uplink_pack_time = 0.0  # 记录所有选中客户端中最大的打包时间
        
        for idx in idxs_users:
            print(f"\n[CLIENT {idx}] 开始本地训练...")
            
            # 初始化客户端本地更新对象
            local_model = LocalUpdateResidual(args=args, dataset=train_dataset,
                                              idxs=user_groups[idx], client_id=idx)
            
            # <--- 客户端从其自己的同步模型开始训练 --->
            current_client_synced_weights = client_synced_models[idx]
            
            # 本地训练 - 使用客户端自己的同步模型
            updated_weights, loss = local_model.update_weights_memory_efficient(
                global_weights=current_client_synced_weights, global_round=epoch, device=device)
            
            # 记录客户端性能用于智能选择
            if idx not in args.client_history['losses']:
                #如果目前这个用户还没有记录,那么先新建一个空列表
                args.client_history['losses'][idx] = []
            args.client_history['losses'][idx].append(loss)
            # 只保留最近10轮的损失记录
            if len(args.client_history['losses'][idx]) > 10:
                args.client_history['losses'][idx] = args.client_history['losses'][idx][-10:]
            
            # <--- 残差相对于客户端自己的起始模型计算 --->
            model_residual = model_subtract(updated_weights, current_client_synced_weights)

            # 1. 应用上行误差反馈 (Error Feedback)
            if args.disable_uplink_ef:
                compensated_residual = model_residual # 跳过EF
                if epoch == 0: print(f"[CLIENT {idx}] 上行误差反馈: 禁用")
            else:
                compensated_residual = model_add(model_residual, client_residuals[idx]) # 默认EF

            residual_to_upload = copy.deepcopy(compensated_residual)

            # 2. 应用上行压缩 (Compression)
            if args.uplink_compression == 'uniform':
                residual_to_upload = apply_residual_compression_fast(
                    residual_to_upload, args.uplink_compression_ratio)
                if epoch == 0: print(f"[CLIENT {idx}] 上行压缩: uniform (ratio={args.uplink_compression_ratio})")
            else:
                if epoch == 0: print(f"[CLIENT {idx}] 上行压缩: none")

            # 3. 更新上行误差状态 (Error Feedback State)
            if args.disable_uplink_ef:
                # EF被禁用，残差清零
                new_client_residual = {key: torch.zeros_like(param).cpu() 
                                      for key, param in model_residual.items()}
            else:
                # EF启用，计算未发送的部分
                new_client_residual = model_subtract(compensated_residual, residual_to_upload)

            # 更新该客户端的历史误差，为下一轮做准备
            for key, param in new_client_residual.items():
                if key in client_residuals[idx]:
                    client_residuals[idx][key] = param.cpu()

            # 步骤 6: 处理"最终要上传"的残差 - 支持全量和压缩模式
            if args.uplink_compression == 'uniform':
                # 压缩模式：对残差进行稀疏化打包处理
                pack_result = pack_sparse_residual(residual_to_upload, enable_timing=True)
                
                # 初始化最终残差变量
                final_residual = None
                
                # 检查打包结果是否包含计时信息
                if isinstance(pack_result, tuple):
                    # 如果返回的是元组，说明包含了打包后的残差和打包时间
                    final_residual, pack_time = pack_result
                    # 记录最大打包时间（并行操作取最大值）
                    max_uplink_pack_time = max(max_uplink_pack_time, pack_time)
                else:
                    # 如果返回的不是元组，直接将结果赋值为最终残差
                    final_residual = pack_result
                
                if epoch == 0: print(f"[CLIENT {idx}] 上行压缩打包完成，耗时: {pack_time:.2f}ms")
            else:
                # 全量模式：直接传递密集张量，跳过打包步骤
                final_residual = copy.deepcopy(residual_to_upload)
                if epoch == 0: print(f"[CLIENT {idx}] 全量上行传输，跳过打包步骤")
            
            # 步骤7: 记录残差和损失
            local_residuals.append(copy.deepcopy(final_residual))  # 保存当前客户端的最终残差
            local_losses.append(copy.deepcopy(loss))  # 保存当前客户端的训练损失
            
            # 调用封装函数计算并打印通信统计信息
            client_transmitted_bytes = calculate_and_print_client_communication_stats(
                idx=idx,
                final_residual=final_residual,
                global_model=global_model,
                args=args,
                total_original_params=total_original_params,
                total_nonzero_values=total_nonzero_values
            )
            # 累加本轮通信开销
            epoch_comm_cost += client_transmitted_bytes
            
        # 更新并行上行打包时间统计
        if max_uplink_pack_time > 0:
            total_pack_time += max_uplink_pack_time
            pack_count += 1  # 一轮并行打包算作一次操作
            print(f"\n📤 上行并行打包完成，最大打包耗时: {max_uplink_pack_time:.2f}ms")
        
        # 这个模板将告诉聚合函数在CPU上进行解包和计算
        cpu_model_template = {key: param.cpu() for key, param in global_model.state_dict().items()}
            
        # 服务器聚合残差 - 支持全量和压缩模式，支持IID和OOD场景
        # 检查是否使用自适应聚合
        aggregation_method = getattr(args, 'adaptive_aggregation', 'weighted_average')
        use_adaptive_aggregation = aggregation_method == 'diversity_aware'
        
        if use_adaptive_aggregation:
            # 自适应聚合场景（支持IID和OOD）
            client_data_sizes = [len(user_groups[idx]) for idx in idxs_users]
            client_losses_vals = [loss for loss in local_losses]
            
            # 判断数据分布类型并相应调整策略
            data_distribution = "OOD" if (hasattr(args, 'iid') and args.iid == 0) else "IID"
            
            # 如果使用diversity_aware聚合，计算多样性分数
            diversity_scores = None
            if aggregation_method == 'diversity_aware':
                # 将CPU模型模板传入
                diversity_scores = calculate_diversity_scores_residual(
                    local_residuals, 
                    client_data_sizes, 
                    server_model_template=cpu_model_template,  # 使用CPU模板
                    args=args  # 传入args参数以判断是否压缩
                )
                print(f"🧮 计算多样性分数 ({data_distribution}): {[f'{score:.4f}' for score in diversity_scores]}")
            
            # 根据是否启用上行压缩选择聚合方式
            if args.uplink_compression == 'uniform':
                agg_result = adaptive_client_aggregation(
                    local_residuals, 
                    client_data_sizes, 
                    client_losses_vals,
                    server_model_template=cpu_model_template,  # 使用CPU模板
                    aggregation_method=aggregation_method,
                    diversity_scores=diversity_scores,
                    enable_timing=True  # 启用计时
                )
                if isinstance(agg_result, tuple):
                    aggregated_residual, server_unpack_time = agg_result
                    total_unpack_time += server_unpack_time
                    unpack_count += 1  # 服务器端聚合算作一次解包操作
                else:
                    aggregated_residual = agg_result
            else:
                # 全量模式：直接对密集张量进行聚合，无需解包，但仍使用自适应聚合
                agg_result = adaptive_client_aggregation(
                    local_residuals, 
                    client_data_sizes, 
                    client_losses_vals,
                    server_model_template=None,  # 全量模式不需要模板
                    aggregation_method=aggregation_method,
                    diversity_scores=diversity_scores,
                    enable_timing=False # 全量模式无解包计时
                )
                # 确保 aggregated_residual 被正确赋值
                if isinstance(agg_result, tuple):
                    aggregated_residual, _ = agg_result
                else:
                    aggregated_residual = agg_result
            
            print(f"📊 使用自适应聚合策略 ({aggregation_method}) - {data_distribution}场景")
        else:
            # 标准聚合场景（weighted_average）
            data_distribution = "OOD" if (hasattr(args, 'iid') and args.iid == 0) else "IID"
            
            if args.uplink_compression == 'uniform':
                # 压缩模式：需要解包并使用CPU模板
                agg_result = average_weights_residual(local_residuals, 
                                                      template=cpu_model_template,  # 传入CPU模板
                                                      enable_timing=True)
                if isinstance(agg_result, tuple):
                    aggregated_residual, server_unpack_time = agg_result
                    total_unpack_time += server_unpack_time
                    unpack_count += 1  # 服务器端聚合算作一次解包操作
                else:
                    aggregated_residual = agg_result
            else:
                # 全量模式：直接对密集张量进行聚合，无需解包
                aggregated_residual = average_weights_residual(local_residuals, 
                                                              template=None,  # 全量模式不需要模板
                                                              enable_timing=False)  # 全量模式不需要解包计时
            print(f"📊 使用标准聚合策略 (weighted_average) - {data_distribution}场景")
        
        # 更新全局模型 - 确保设备一致性
        # 获取当前全局模型的权重（在正确设备上）
        current_global_weights = global_model.state_dict()
        updated_global_weights = model_add(current_global_weights, aggregated_residual)
        
        # 检查更新后的权重是否包含NaN或Inf
        weights_valid = True
        for key, param in updated_global_weights.items():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"警告: 参数 {key} 包含NaN/Inf值")
                weights_valid = False
        
        if weights_valid:
            print("\n[SERVER] 准备下行广播...")

            # 1. 应用下行误差反馈 (Error Feedback)
            if args.disable_downlink_ef:
                compensated_downlink_residual = aggregated_residual # 跳过EF
                if epoch == 0: print("[SERVER] 下行误差反馈: 禁用")
            else:
                compensated_downlink_residual = model_add(aggregated_residual, server_downlink_error) # 默认EF

            # 2. 应用下行压缩 (Compression)
            if args.downlink_compression == 'uniform':
                residual_to_broadcast_unpacked = apply_residual_compression_fast(
                    compensated_downlink_residual, args.downlink_compression_ratio
                )
                if epoch == 0: print(f"[SERVER] 下行压缩: uniform (ratio={args.downlink_compression_ratio})")
            else:
                residual_to_broadcast_unpacked = compensated_downlink_residual # 无压缩
                if epoch == 0: print("[SERVER] 下行压缩: none")

            # 3. 处理残差用于传输 - 支持全量和压缩模式
            if args.downlink_compression == 'uniform':
                # 压缩模式：打包残差用于传输并计时
                pack_result = pack_sparse_residual(residual_to_broadcast_unpacked, enable_timing=True)
                residual_to_broadcast_packed, pack_time = pack_result
                total_downlink_pack_time += pack_time
                downlink_pack_count += 1
                print(f"[SERVER] 下行残差打包完成，耗时: {pack_time:.2f}ms")
            else:
                # 全量模式：直接传递密集张量，跳过打包步骤
                residual_to_broadcast_packed = copy.deepcopy(residual_to_broadcast_unpacked)
                print(f"[SERVER] 全量下行传输，跳过打包步骤")

            # 4. 更新下行误差状态 (Error Feedback State)
            if args.disable_downlink_ef:
                # EF被禁用，残差清零
                server_downlink_error = {key: torch.zeros_like(param).cpu()
                                         for key, param in aggregated_residual.items()}
            else:
                # EF启用，计算未发送的部分
                server_downlink_error = model_subtract(compensated_downlink_residual, residual_to_broadcast_unpacked)
            
            # 5. 用完整的、未压缩的聚合更新服务器的高保真度全局模型
            # 这对于准确的服务器端测试至关重要
            global_model.load_state_dict(updated_global_weights)
            global_weights = updated_global_weights
            
        else:
            print("⚠️ 模型权重包含无效值，跳过此次更新，使用上一轮权重")
            # 保持原有权重不变，设置空的下行广播
            residual_to_broadcast_packed = None
            global_weights = current_global_weights
        
        # 计算当前轮次耗时
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        
        # 计算平均轮次时间和剩余时间估计
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = args.epochs - (epoch + 1)
        estimated_remaining_time = avg_epoch_time * remaining_epochs
        
        # 评估当前轮次性能
        (current_test_acc, test_loss, loss_avg, ema_acc, best_ema_acc, 
         patience_counter, improve_streak, best_global_weights) = evaluate_epoch_performance(
            args, global_model, test_dataset, local_losses, 
            global_test_accuracy, train_loss, ema_acc, ema_alpha,
            best_ema_acc, patience_counter, improve_streak, patience,
            best_global_weights)

        # 打印轮次总结
        print_epoch_summary(epoch, args, current_test_acc, best_ema_acc, ema_acc, 
                           loss_avg, test_loss, epoch_duration, estimated_remaining_time)
        
        # 打印通信量汇总
        total_round_comm_cost = print_communication_summary(
            epoch, args, idxs_users, global_model, epoch_comm_cost,
            residual_to_broadcast_packed, communication_cost)

        # 计算当前学习率以便记录
        current_lr = calculate_current_learning_rate(args, epoch)
                
        history['epoch'].append(epoch + 1)
        history['test_accuracy'].append(current_test_acc)
        history['avg_train_loss'].append(loss_avg)
        history['learning_rate'].append(current_lr)
        # 记录压缩比例信息 (简化为主要压缩比)
        main_compression_ratio = args.uplink_compression_ratio if args.uplink_compression == 'uniform' else 0.0
        history['compression_ratio'].append(main_compression_ratio)
        history['communication_cost'].append(total_round_comm_cost)  # <-- 新代码：记录双向总和
        
        # 早停检查
        if patience_counter >= patience:
            print(f'🛑 Early stopping triggered after {epoch+1} global rounds')
            if best_global_weights is not None:
                global_model.load_state_dict(best_global_weights)
                print("Loaded best model weights for final testing.")
            break
        
        # 更新tqdm描述信息
        pbar.set_postfix({
            'Acc': f'{current_test_acc*100:.2f}%',
            'Loss': f'{loss_avg:.4f}',
            'Time': f'{epoch_duration:.1f}s',
            'ETA': f'{estimated_remaining_time/60:.1f}min'
        })
    
    # 评估最终模型性能
    test_acc, avg_local_test_acc_best_model = evaluate_final_model_performance(
        args, global_model, test_dataset, train_dataset, user_groups, epoch
    )
    
    # 打印完整训练过程通信量汇总
    print_final_communication_summary(
        args, global_model, communication_cost, epoch,
        total_pack_time, total_unpack_time, pack_count, unpack_count,
        total_downlink_pack_time, total_downlink_unpack_time, 
        downlink_pack_count, downlink_unpack_count, epoch_times
    )

    # 保存实验结果和生成图像
    save_experiment_results(args, history, train_loss, global_test_accuracy, communication_cost)
# 主函数结束

def evaluate_final_model_performance(args, global_model, test_dataset, train_dataset, user_groups, epoch):
    """
    评估最终模型性能，包括全局测试和平均本地测试。
    
    Args:
        args: 实验参数
        global_model: 全局模型
        test_dataset: 测试数据集
        train_dataset: 训练数据集
        user_groups: 用户组索引
        epoch: 当前轮数
        
    Returns:
        tuple: (test_acc, avg_local_test_acc_best_model)
    """
    print("\n评估最终模型性能...")
    # 注意：此时的global_model已经是早停机制加载的最佳模型
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print("使用最佳模型评估平均本地测试性能...")
    # 需要LocalUpdate来评估，我们从update.py导入
    from update import LocalUpdate
    list_acc_best_model = []
    global_model.eval()
    for c in range(args.num_users):
        local_model = LocalUpdate(args=args, dataset=train_dataset,
                                   idxs=user_groups[c], client_id=c)
        acc, _ = local_model.inference(model=global_model)
        list_acc_best_model.append(acc)
    avg_local_test_acc_best_model = sum(list_acc_best_model) / len(list_acc_best_model)

    print(f' \n Results after {epoch+1} global rounds of training:')
    print("|---- Avg Local Test Accuracy (Best Model): {:.2f}%".format(100*avg_local_test_acc_best_model))
    print("|---- Global Test Accuracy (Best Model): {:.2f}%".format(100*test_acc))
    
    return test_acc, avg_local_test_acc_best_model


def print_final_communication_summary(args, global_model, communication_cost, epoch,
                                     total_pack_time, total_unpack_time, pack_count, unpack_count,
                                     total_downlink_pack_time, total_downlink_unpack_time, 
                                     downlink_pack_count, downlink_unpack_count, epoch_times):
    """
    打印完整训练过程的通信量汇总和压缩统计。
    
    Args:
        args: 实验参数
        global_model: 全局模型
        communication_cost: 通信开销列表
        epoch: 当前轮数
        各种时间统计参数
    """
    # 计算完整训练过程的压缩统计
    total_comm_cost = sum(communication_cost)  # communication_cost 现在记录的是字节数
    print(f"|---- 总通信开销: {total_comm_cost:,} 字节")
    
    # ================= 整体训练过程通信量汇总 =================
    total_rounds = epoch + 1  # 实际训练轮数
    single_model_params = sum(torch.numel(param) for _, param in global_model.named_parameters())
    single_model_bytes = single_model_params * 4
    avg_selected_clients = args.frac * args.num_users
    
    print(f"\n🎯 整体训练过程通信量汇总:")
    # 检查是否有任何压缩启用
    has_compression = (args.uplink_compression == 'uniform') or (args.downlink_compression == 'uniform')
    if has_compression:
        # 理论基准应考虑上行和下行
        # 上行基准：avg_clients * rounds * model_size
        theoretical_uplink_baseline_total = int(single_model_bytes * avg_selected_clients * total_rounds)
        # 下行基准：all_clients * rounds * model_size (因为每轮广播给所有客户端)
        theoretical_downlink_baseline_total = int(single_model_bytes * args.num_users * total_rounds)

        theoretical_total_baseline_bytes = theoretical_uplink_baseline_total + theoretical_downlink_baseline_total

        # total_comm_cost 是每轮双向通信的和，是正确的
        overall_compression_effectiveness = (1 - total_comm_cost / theoretical_total_baseline_bytes) * 100 if theoretical_total_baseline_bytes > 0 else 0

        print(f"   • 实际总传输字节数: {int(total_comm_cost):,} B")
        print(f"   • 理论基准总字节数: {theoretical_total_baseline_bytes:,} B")
        print(f"     - 上行基准: {theoretical_uplink_baseline_total:,} B (选中客户端 × {total_rounds} 轮 × {single_model_bytes:,} B)")
        print(f"     - 下行基准: {theoretical_downlink_baseline_total:,} B (所有客户端 × {total_rounds} 轮 × {single_model_bytes:,} B)")
        print(f"   • 整体压缩效果: 减少 {overall_compression_effectiveness:.2f}% 通信量")
        print(f"   • 总训练轮数: {total_rounds}")
        print(f"   • 总客户端数: {args.num_users}")
        print(f"   • 单个模型参数量: {single_model_params:,} ({single_model_bytes:,} B)")
        
        # <--- 双向压缩时间统计（并行操作） --->
        print(f"\n⏱️ 上行压缩时间统计 (并行操作):")
        print(f"   • 总打包时间 (客户端并行): {total_pack_time:.2f}ms (执行 {pack_count} 轮)")
        print(f"   • 总解包时间 (服务器聚合): {total_unpack_time:.2f}ms (执行 {unpack_count} 次)")
            
        print(f"\n⏱️ 下行压缩时间统计 (并行操作):")
        print(f"   • 总打包时间 (服务器): {total_downlink_pack_time:.2f}ms (执行 {downlink_pack_count} 轮)")
        print(f"   • 总解包时间 (客户端并行): {total_downlink_unpack_time:.2f}ms (执行 {downlink_unpack_count} 轮)")
            
        total_compression_time = total_pack_time + total_unpack_time + total_downlink_pack_time + total_downlink_unpack_time
        training_duration_ms = sum(epoch_times) * 1000
        print(f"\n   • 双向压缩总时间: {total_compression_time:.2f}ms")
        print(f"   • 压缩时间占训练总时间比例: {total_compression_time / training_duration_ms * 100:.3f}%")
    else:
        theoretical_total_baseline_bytes = int(single_model_bytes * avg_selected_clients * total_rounds)
        print(f"   • 密集传输总字节数: {total_comm_cost:,} B")
        print(f"   • 理论基准总字节数: {theoretical_total_baseline_bytes:,} B")
        
        # 双向压缩时间统计
        total_all_pack_time = total_pack_time + total_downlink_pack_time
        total_all_unpack_time = total_unpack_time + total_downlink_unpack_time
        total_all_process_time = total_all_pack_time + total_all_unpack_time
        total_training_time_ms = sum(epoch_times) * 1000
        
        print(f"\n⏱️ 双向压缩时间统计 (并行操作):")
        print(f"   📤 上行通信:")
        print(f"      • 客户端打包时间 (并行): {total_pack_time:.2f}ms (执行 {pack_count} 轮)")
        print(f"      • 服务器解包时间: {total_unpack_time:.2f}ms (执行 {unpack_count} 次)")
        
        print(f"   📥 下行通信:")
        print(f"      • 服务器打包时间: {total_downlink_pack_time:.2f}ms (执行 {downlink_pack_count} 轮)")
        print(f"      • 客户端解包时间 (并行): {total_downlink_unpack_time:.2f}ms (执行 {downlink_unpack_count} 轮)")
        
        print(f"   🔄 总计:")
        print(f"      • 总打包时间: {total_all_pack_time:.2f}ms")
        print(f"      • 总解包时间: {total_all_unpack_time:.2f}ms")
        print(f"      • 总处理时间: {total_all_process_time:.2f}ms")
        print(f"      • 处理时间占训练总时间比例: {total_all_process_time / total_training_time_ms * 100:.3f}%")
        print(f"   • 总训练轮数: {total_rounds}")
        print(f"   • 单个模型参数量: {single_model_params:,} ({single_model_bytes:,} B)")


def save_experiment_results(args, history, train_loss, global_test_accuracy, communication_cost):
    """
    保存实验结果，包括CSV历史记录、详情文件、图像生成和pickle文件。
    
    Args:
        args: 实验参数
        history: 训练历史记录
        train_loss: 训练损失
        global_test_accuracy: 全局测试准确率
        communication_cost: 通信开销
    """
    # --- 修改开始：将历史记录保存为CSV文件 ---
    # 创建时间戳命名的文件夹
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('./save/logs', current_time)
    os.makedirs(log_dir, exist_ok=True)

    # CSV文件名与文件夹名保持一致（都使用时间戳）
    log_filename = f'{current_time}.csv'
    log_path = os.path.join(log_dir, log_filename)
    
    # 保存实验详情到同一文件夹
    iid_str = 'iid' if args.iid else 'noniid'
    
    # 生成压缩描述字符串
    uplink_comp = f"uplink_{args.uplink_compression_ratio}" if args.uplink_compression == 'uniform' else "uplink_none"
    downlink_comp = f"downlink_{args.downlink_compression_ratio}" if args.downlink_compression == 'uniform' else "downlink_none"
    comp_str = f"{uplink_comp}_{downlink_comp}"
    
    details_content = f"""实验时间: {current_time}
                          实验类型: Residual Federated Learning (Ablation)
                          数据集: {args.dataset.upper()}
                          模型: {args.model.upper()}
                          训练轮数: {args.epochs}
                          数据分布: {iid_str.upper()}
                          学习率: {args.lr}
                          本地训练轮数: {args.local_ep}
                          参与客户端数: {args.num_users}
                          参与比例: {args.frac}
                          客户端选择: {args.selection_method}
                          上行压缩: {args.uplink_compression} ({args.uplink_compression_ratio if args.uplink_compression == 'uniform' else 'N/A'})
                          下行压缩: {args.downlink_compression} ({args.downlink_compression_ratio if args.downlink_compression == 'uniform' else 'N/A'})
                          上行误差反馈: {'禁用' if args.disable_uplink_ef else '启用'}
                          下行误差反馈: {'禁用' if args.disable_downlink_ef else '启用'}"""
    details_path = os.path.join(log_dir, 'experiment_details.txt')
    with open(details_path, 'w', encoding='utf-8') as f:
        f.write(details_content)
    
    df = pd.DataFrame(history)
    df.to_csv(log_path, index=False)
    print(f"📈 训练历史已保存到: {log_path}")
    print(f"📋 实验详情已保存到: {details_path}")
    
    # 自动生成图像
    try:
        from visualize_results import plot_single_experiment
        plots_dir = './save/plots'
        plot_result = plot_single_experiment(log_path, plots_dir)
        if plot_result:
            print(f"📊 实验图像已自动生成到: {plot_result}")
        else:
            print("⚠️ 图像生成失败")
    except Exception as e:
        print(f"⚠️ 自动生成图像时出错: {e}")
        print("💡 你可以手动运行: python visualize_results.py --single " + log_path)
    # --- 修改结束 ---

    # 保存结果
    save_dir = './save/objects'
    os.makedirs(save_dir, exist_ok=True)

    file_name = './save/objects/residual_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_comp[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs, 'smart' if getattr(args, 'compression', 'none') == 'smart' else 'uniform' if getattr(args, 'compression', 'none') == 'uniform' else 'none')

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, global_test_accuracy, communication_cost], f)

    print(f'训练结果已保存到: {file_name}')


def print_training_details(args, ema_alpha, patience):
    """
    打印残差联邦学习的训练详情和启用的高级特性。
    
    Args:
        args: 命令行参数对象 (包含消融实验参数)
        ema_alpha (float): EMA 平滑因子
        patience (int): 早停机制的耐心值
    """
    print(f"=== 消融实验配置 ===")
    print(f"🎯 客户端选择策略: {args.selection_method}")
    print(f"📤 上行压缩: {args.uplink_compression} ({args.uplink_compression_ratio if args.uplink_compression == 'uniform' else 'N/A'})")
    print(f"📥 下行压缩: {args.downlink_compression} ({args.downlink_compression_ratio if args.downlink_compression == 'uniform' else 'N/A'})")
    print(f"🔄 上行误差反馈: {'禁用' if args.disable_uplink_ef else '启用'}")
    print(f"🔄 下行误差反馈: {'禁用' if args.disable_downlink_ef else '启用'}")
    
    print(f"\n=== 启用的高级特性（残差联邦学习）===")
    print(f"📊 EMA平滑准确率: 启用 (α={ema_alpha})")
    
    # Label Smoothing 检查
    criterion_type = getattr(args, 'criterion', 'crossentropy')
    label_smoothing_enabled = criterion_type == 'label_smoothing'
    smoothing_value = getattr(args, 'smoothing', 0.0)
    print(f"🎯 Label Smoothing: {'启用' if label_smoothing_enabled and smoothing_value > 0 else '禁用'}")
    
    # SWA 检查
    swa_enabled = getattr(args, 'enable_swa', 0) == 1
    swa_start = getattr(args, 'swa_start', 150)
    print(f"🔄 SWA: {'启用' if swa_enabled else '禁用'}")
    
    # CutMix 检查
    cutmix_enabled = getattr(args, 'enable_cutmix', 0) == 1
    cutmix_prob = getattr(args, 'cutmix_prob', 0.0)
    print(f"🔀 CutMix: {'启用' if cutmix_enabled and cutmix_prob > 0 else '禁用'}")
    
    # Mixup 检查
    mixup_enabled = getattr(args, 'enable_mixup', 0) == 1
    mixup_alpha = getattr(args, 'mixup_alpha', 0.0)
    print(f"🎨 Mixup: {'启用' if mixup_enabled and mixup_alpha > 0 else '禁用'}")
    
    # 知识蒸馏检查
    kd_enabled = getattr(args, 'enable_knowledge_distillation', 0) == 1
    print(f"🧠 知识蒸馏: {'启用' if kd_enabled else '禁用'}")
    
    # 学习率调度检查
    lr_scheduler_type = getattr(args, 'lr_scheduler', 'none')
    print(f"📈 学习率调度: {lr_scheduler_type}")
    
    # 聚合策略检查
    aggregation_method = getattr(args, 'adaptive_aggregation', 'standard')
    print(f"🤝 聚合策略: {aggregation_method}")
    
    print(f"========================================")
    
    print(f"\n{'='*70}")
    print(f"🚀 开始残差联邦学习训练")
    print(f"📊 总轮次: {args.epochs}, 客户端: {args.num_users}, 参与比例: {args.frac}")
    print(f"📋 早停机制: 耐心值 = {patience} 轮")
    print(f"{'='*70}")

def select_clients(epoch, args, user_groups, client_ema_losses, EMA_ALPHA, m):
    """
    根据历史表现和多样性选择客户端。

    Args:
        epoch (int): 当前轮次
        args: 命令行参数对象
        user_groups (dict): 每个客户端的数据索引
        client_ema_losses (dict): 每个客户端的EMA损失
        EMA_ALPHA (float): EMA平滑因子
        m (int): 每轮选择的客户端数量

    Returns:
        idxs_users (list): 选中的客户端索引列表
    """
    # 检查是 'smart' 模式且已过预热期
    if args.selection_method == 'smart' and epoch > 2:
        # 基于历史表现的智能客户端选择
        client_weights = []
        
        # 计算一次全局最大数据量，避免在循环中重复计算
        max_data_size = max(len(user_groups[i]) for i in range(args.num_users))
        
        for idx in range(args.num_users):
            # 更新客户端的EMA损失
            last_loss = np.mean(args.client_history['losses'].get(idx, [1.0]))
            if idx not in client_ema_losses:
                client_ema_losses[idx] = last_loss
            else:
                client_ema_losses[idx] = EMA_ALPHA * last_loss + (1 - EMA_ALPHA) * client_ema_losses[idx]

            # 使用EMA损失来计算得分
            current_ema_loss = client_ema_losses[idx]
            loss_score = 1.0 / (1.0 + current_ema_loss)  # 使用平滑后的损失
            
            # 数据量权重
            data_size = len(user_groups[idx])
            data_score = data_size / max_data_size
            '''
            计算当前客户端拥有的数据样本量(data_size)
            并将其与所有客户端中最大的数据量进行比较,得出一个0到1之间的归一化分数
            '''
            
            # 避免过度选择同一客户端 - 更精确的频率惩罚
            # 计算最近几轮中该客户端被选中的次数
            recent_window = min(6 * m, len(args.client_history['last_selected']))  # 最近6轮的选择
            '''
            定义我们要回顾的“历史记录”有多长。6 * m 意味着我们关注最近6轮选择的所有客户端
            在训练刚开始时(比如才第3轮),总共也只选了30个客户端,历史记录没有60条那么长
            min函数确保我们不会试图查看不存在的历史记录
            '''
            
            # args.client_history是一个字典，包含多个键
            '''
            args.client_history = {
            'losses': {},           # 每个客户端的历史损失(字典,键为客户端ID,值为损失列表）
            'last_selected': [],    # 所有历史选择的客户端（列表，用于简单频率统计）
            'round_selections': [], # 按轮记录的选择历史(列表,每轮记录一个客户端ID列表)
            'performance_scores': {} # (可选)每个客户端的性能分数(字典,键为客户端ID,值为分数)
            }
            '''
            recent_selections = args.client_history['last_selected'][-recent_window:] if recent_window > 0 else []
            frequency_penalty = 1.0 - (recent_selections.count(idx) * 0.15)  
            frequency_penalty = max(frequency_penalty, 0.2)  
            # recent_selections.count(idx)：数一下在刚刚那张“小纸条”上，这个客户 idx 的名字出现了几次
            # 最小保持20%权重，为了防止一个性能特别好的客户端因为被频繁选中而导致其分数过低，被完全“封杀”
            '''
            代码会观察一个动态的“最近历史窗口”(大致是最近6轮被选中的所有客户端列表),并检查当前的客户端(idx)在这个窗口中出现了多少次。
            每出现一次,它的“频率惩罚”得分就会降低0.15。如果一个客户端最近频繁被选中，它的这个分数就会很低。
            一个客户端最近被选中的次数越多，它的 frequency_penalty 得分就越低。
            '''
            
            # 综合得分
            combined_score = (0.5 * loss_score + 0.25 * data_score + 0.25 * frequency_penalty)
            client_weights.append(combined_score)
        
        # 概率选择确保多样性
        client_weights = np.array(client_weights)
        client_probs = client_weights / client_weights.sum()
        
        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=client_probs)
        
        # 显示选择详情（仅在verbose模式下）
        if args.verbose:
            selected_scores = [client_weights[idx] for idx in idxs_users]
            print(f'🧠 智能选择客户端: {list(idxs_users)} (权重: {[f"{s:.3f}" for s in selected_scores]})')
        else:
            print(f'🧠 智能选择客户端: {list(idxs_users)}')
    else:
        # 随机选择 (用于 '--selection_method random' 或 'smart' 模式的预热)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        if args.selection_method == 'random':
            print(f'🎲 随机选择客户端: {list(idxs_users)} (用户指定使用随机选择客户端的方法)')
        else:
            print(f'🎲 随机选择客户端: {list(idxs_users)} (用户指定使用智能客户端选择方法,此时正在预热)')
    
    # 记录选择的客户端历史
    args.client_history['last_selected'].extend(idxs_users.tolist())
    args.client_history['round_selections'].append(idxs_users.tolist())  # 按轮记录
    
    # 维护合理的历史窗口大小
    if len(args.client_history['last_selected']) > (m * 8):  # 保持8轮的历史
        args.client_history['last_selected'] = args.client_history['last_selected'][-(m*8):]
    if len(args.client_history['round_selections']) > 8:  # 保持最近8轮的轮级记录
        args.client_history['round_selections'] = args.client_history['round_selections'][-8:]
    '''
    'last_selected': [0, 1, 2, 0, 3, 1, 4, 2]
    表示在过去的几轮中，客户端 0、1、2、3、4 被选择的次数
    'round_selections': [[0, 1, 2], [1, 3, 4], [0, 2, 4]]
    表示：
    第 1 轮选择了客户端 0、1、2
    第 2 轮选择了客户端 1、3、4
    第 3 轮选择了客户端 0、2、4
    '''
    
    print(f"🎯 选中的客户端: {sorted(idxs_users)} (共{len(idxs_users)}个)")
    return idxs_users

def setup_device(args):
    """
    设置设备（GPU或CPU）。
    
    Args:
        args: 命令行参数对象，包含GPU设置。
    
    Returns:
        device: torch.device 对象，表示使用的设备。
    """
    if args.gpu is not None and torch.cuda.is_available():
        try:
            # 确保args.gpu是有效的整数
            gpu_id = int(args.gpu)
            if gpu_id >= 0 and gpu_id < torch.cuda.device_count():
                torch.cuda.set_device(gpu_id)
                device = torch.device(f'cuda:{gpu_id}')
                print(f"使用GPU设备: {device}")
                
                # 测试GPU是否正常工作
                test_tensor = torch.randn(1, 1, 28, 28).to(device)
                print(f"GPU测试成功，可用内存: {torch.cuda.get_device_properties(device).total_memory / 1024**2:.0f}MB")
                del test_tensor
                torch.cuda.empty_cache()
            else:
                print(f"GPU ID {gpu_id} 无效，可用GPU数量: {torch.cuda.device_count()}, 切换到CPU")
                device = torch.device('cpu')
        except Exception as e:
            print(f"GPU {args.gpu} 初始化失败: {str(e)}, 切换到CPU")
            device = torch.device('cpu')
    else:
        if args.gpu is not None:
            print(f"请求使用GPU {args.gpu}，但CUDA不可用，切换到CPU")
        device = torch.device('cpu')
        print("使用CPU设备")
    
    print(f"[DEBUG] 设备设置完成: {device}")
    return device

def build_model(args, train_dataset):
    """
    构建全局模型。
    
    Args:
        args: 命令行参数对象，包含模型和数据集设置。
        train_dataset: 训练数据集，用于确定输入维度（如MLP模型）。
    
    Returns:
        global_model: 构建好的全局模型。
    """
    print(f"[DEBUG] 开始构建模型: {args.model} for dataset: {args.dataset}")
    print(f"正在构建模型: {args.model} for dataset: {args.dataset}")
    
    try:
        print("[DEBUG] 尝试加载优化模型...")
        if args.model in ['cnn']:
            print(f"[DEBUG] 标准CNN模型 - 数据集: {args.dataset}")
            if args.dataset == 'mnist':
                global_model = get_model('mnist', 'cnn')
            elif args.dataset == 'cifar':
                global_model = get_model('cifar10', 'cnn')
            else:
                raise ValueError(f"标准CNN不支持数据集: {args.dataset}")
                
        elif args.model in ['optimized', 'cnn_optimized', 'cnn_opt', 'cnn_enhanced']:
            print(f"[DEBUG] 优化CNN模型 - 数据集: {args.dataset}")
            if args.dataset == 'mnist':
                global_model = get_model('mnist', 'optimized')
            elif args.dataset == 'cifar':
                global_model = get_model('cifar10', 'cnn')
            else:
                raise ValueError(f"优化CNN目前仅支持MNIST和CIFAR数据集")
                
        elif args.model in ['optimized_gn', 'cnn_optimized_gn', 'groupnorm']:
            print(f"[DEBUG] GroupNorm优化CNN模型 - 数据集: {args.dataset}")
            if args.dataset == 'mnist':
                global_model = get_model('mnist', 'optimized_gn', num_groups=getattr(args, 'num_groups', 8))
            elif args.dataset == 'cifar':
                global_model = get_model('cifar10', 'resnet18_fed', use_groupnorm=True, num_groups=getattr(args, 'num_groups', 8))
            elif args.dataset == 'cifar100':
                use_groupnorm = getattr(args, 'use_groupnorm', True)
                num_groups = getattr(args, 'num_groups', 8)
                global_model = get_model('cifar100', 'resnet18_fed', 
                                        use_groupnorm=use_groupnorm, 
                                        num_groups=num_groups)
            else:
                raise ValueError(f"GroupNorm优化CNN目前支持MNIST、CIFAR-10和CIFAR-100数据集")
                
        elif args.model in ['resnet18', 'resnet18_fed', 'resnet', 'resnet_mini', 'resnet18_gn']:
            if args.dataset == 'mnist':
                global_model = get_model('mnist', 'optimized')
            elif args.dataset == 'cifar':
                if args.model == 'resnet18_gn':
                    global_model = get_model('cifar10', 'resnet18_fed', use_groupnorm=True, num_groups=getattr(args, 'num_groups', 8))
                else:
                    global_model = get_model('cifar10', 'resnet18_fed')
            elif args.dataset == 'cifar100':
                if args.model == 'resnet18_gn':
                    use_groupnorm = getattr(args, 'use_groupnorm', True)
                    num_groups = getattr(args, 'num_groups', 8)
                    global_model = get_model('cifar100', 'resnet18_fed', 
                                            use_groupnorm=use_groupnorm, 
                                            num_groups=num_groups)
                else:
                    global_model = get_model('cifar100', 'resnet18_fed')
            else:
                raise ValueError(f"ResNet18不支持数据集: {args.dataset}")
                
        elif args.model in ['efficientnet', 'efficient']:
            if args.dataset == 'cifar':
                global_model = get_model('cifar10', 'efficientnet')
            elif args.dataset == 'cifar100':
                global_model = get_model('cifar100', 'efficientnet')
            else:
                raise ValueError(f"EfficientNet不支持数据集: {args.dataset}")
                
        elif args.model == 'densenet':
            if args.dataset == 'cifar100':
                use_attention = getattr(args, 'use_attention', True)
                use_groupnorm = getattr(args, 'use_groupnorm', True)
                global_model = get_model('cifar100', 'densenet', 
                                        use_attention=use_attention, 
                                        use_groupnorm=use_groupnorm)
            else:
                raise ValueError(f"DenseNet不支持数据集: {args.dataset}")
        else:
            raise ValueError("尝试原有模型")
            
        print(f"[SUCCESS] 成功加载优化模型: {global_model.__class__.__name__}")
        
    except Exception as e:
        print(f"[WARNING] 优化模型加载失败: {e}")
        print(f"[INFO] 回退到原有模型...")
        
        if args.model == 'cnn':
            if args.dataset == 'mnist':
                global_model = CNNMnist(args=args)
            elif args.dataset == 'fmnist':
                global_model = CNNFashion_Mnist(args=args)
            elif args.dataset == 'cifar':
                global_model = CNNCifar(args=args)
            elif args.dataset == 'cifar100':
                global_model = CNNCifar100(args=args)
        elif args.model == 'resnet':
            if args.dataset == 'cifar':
                global_model = ResNet18Fed(num_classes=args.num_classes)
            elif args.dataset == 'cifar100':
                global_model = ResNet18Fed(num_classes=100)
            else:
                print(f"ResNet not implemented for dataset {args.dataset}, using CNN instead")
                if args.dataset == 'mnist':
                    global_model = CNNMnist(args=args)
                elif args.dataset == 'fmnist':
                    global_model = CNNFashion_Mnist(args=args)
        elif args.model == 'mlp':
            img_size = train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
        else:
            exit('Error: unrecognized model')
    
    return global_model

def calculate_and_print_client_communication_stats(
    idx, final_residual, global_model, args, 
    total_original_params=None, total_nonzero_values=None
):
    """
    计算并打印每个客户端的通信统计信息，同时支持全量和压缩模式。

    Args:
        idx (int): 客户端索引。
        final_residual (dict): 客户端上传的最终残差。
        global_model (torch.nn.Module): 全局模型，用于计算未压缩时的基准字节数。
        args: 命令行参数对象，包含压缩相关设置。
        total_original_params (int, optional): 全局统计中压缩前的参数量累加器。
        total_nonzero_values (int, optional): 全局统计中实际传输的字节数累加器。
    
    Returns:
        int: 客户端实际传输的字节数
    """
    # 计算每个客户端的通信量
    if args.uplink_compression == 'uniform':
        # 压缩模式：调用打包格式的通信成本计算
        comm_cost, client_transmitted_bytes, layer_details = calculate_communication_cost_dict(final_residual)
    else:
        # 全量模式：直接计算密集张量的字节数
        client_transmitted_bytes = sum(torch.numel(tensor) * 4 for tensor in final_residual.values())  # float32 = 4字节
        layer_details = []  # 全量模式不需要层详细信息
        comm_cost = client_transmitted_bytes / 4.0  # 为了兼容性保持这个变量

    # 打印每个客户端的通信统计信息
    print(f"[CLIENT {idx}] 通信量统计:")
    print(f"   • 传递层数: {len(final_residual)}")
    print(f"   • 实际传输字节数: {client_transmitted_bytes:,} B")  # 直接打印字节数

    # 打印每层的详细信息
    for detail in layer_details:
        total_params = detail['total_params']

        if 'mask_bytes' in detail and 'values_bytes' in detail:
            # 显示掩码和值的字节分解
            mask_bytes = detail['mask_bytes']

            # 检查是否使用了位级优化
            if detail.get('bit_packed', False):
                # 位级优化格式
                # 计算如果不使用位级优化需要多少字节
                traditional_mask_bytes = total_params * 1  # 每个布尔值1字节
                mask_savings = traditional_mask_bytes - mask_bytes

    # 显示压缩统计信息（仅在启用上行压缩时）
    if args.uplink_compression == 'uniform':
        client_total_params = sum(torch.numel(param) for _, param in global_model.named_parameters())
        baseline_bytes = client_total_params * 4  # 未压缩时的基准字节数

        compression_effectiveness = (1 - client_transmitted_bytes / baseline_bytes) * 100 if baseline_bytes > 0 else 0

        print(f"   • 模型总参数量: {client_total_params:,}")
        print(f"   • 未压缩时基准字节数: {baseline_bytes:,} B")
        print(f"   • 压缩效果: 减少 {compression_effectiveness:.1f}% 通信量")

        # 累加到全局统计
        if total_original_params is not None:
            total_original_params += client_total_params  # 压缩前参数量
        if total_nonzero_values is not None:
            total_nonzero_values += client_transmitted_bytes  # 实际传输的字节数
    else:
        # 如果未启用压缩，显示密集传输信息
        print(f"   • 密集传输字节数: {client_transmitted_bytes:,} B")

    # 返回客户端实际传输的字节数，由调用者累加到本轮总通信量
    return client_transmitted_bytes

def evaluate_epoch_performance(args, global_model, test_dataset, local_losses, 
                               global_test_accuracy, train_loss, ema_acc, ema_alpha,
                               best_ema_acc, patience_counter, improve_streak, patience,
                               best_global_weights):
    """
    评估当前轮次的性能并更新相关统计信息。
    
    Args:
        args: 命令行参数对象
        global_model: 全局模型
        test_dataset: 测试数据集
        local_losses: 本轮客户端训练损失列表
        global_test_accuracy: 全局测试准确率历史列表
        train_loss: 训练损失历史列表
        ema_acc: 当前EMA平滑准确率
        ema_alpha: EMA平滑因子
        best_ema_acc: 最佳EMA准确率
        patience_counter: 耐心计数器
        improve_streak: 连续提升计数器
        patience: 早停耐心值
        best_global_weights: 最佳模型权重
    
    Returns:
        tuple: (current_test_acc, test_loss, loss_avg, ema_acc, best_ema_acc, 
                patience_counter, improve_streak, best_global_weights)
    """
    # 1. 评估当前轮次性能
    current_test_acc, test_loss = test_inference(args, global_model, test_dataset)
    global_test_accuracy.append(current_test_acc)
    loss_avg = sum(local_losses) / len(local_losses)
    train_loss.append(loss_avg)

    # 2. 更新EMA平滑准确率
    if ema_acc is None:
        ema_acc = current_test_acc
    else:
        ema_acc = ema_alpha * current_test_acc + (1 - ema_alpha) * ema_acc

    # 3. 判断性能是否提升 (基于平滑后的准确率)
    eps = 1e-4  # 小阈值防止抖动
    if ema_acc > best_ema_acc + eps:
        best_ema_acc = ema_acc
        patience_counter = 0
        improve_streak += 1
        print(f'✅ 新的最佳平滑准确率: {100*best_ema_acc:.2f}% (连续提升 {improve_streak} 轮)')
        best_global_weights = copy.deepcopy(global_model.state_dict())
    else:
        patience_counter += 1
        improve_streak = 0
        print(f'⚠️ 平滑准确率未改善. 耐心值: {patience_counter}/{patience}')
    
    return (current_test_acc, test_loss, loss_avg, ema_acc, best_ema_acc, 
            patience_counter, improve_streak, best_global_weights)

def print_epoch_summary(epoch, args, current_test_acc, best_ema_acc, ema_acc, 
                       loss_avg, test_loss, epoch_duration, estimated_remaining_time):
    """
    打印轮次总结信息。
    
    Args:
        epoch: 当前轮次
        args: 命令行参数对象
        current_test_acc: 当前测试准确率
        best_ema_acc: 最佳EMA准确率
        ema_acc: 当前EMA准确率
        loss_avg: 平均训练损失
        test_loss: 测试损失
        epoch_duration: 轮次耗时
        estimated_remaining_time: 预计剩余时间
    """
    print(f"\n{'='*60}")
    print(f"📊 轮次 {epoch+1} 总结:")
    print(f"   💯 全局测试准确率: {current_test_acc*100:.2f}% (最佳: {best_ema_acc*100:.2f}%)")
    print(f"   📈 平滑准确率 (用于决策): {ema_acc*100:.2f}%")
    uplink_ratio = args.uplink_compression_ratio if args.uplink_compression == 'uniform' else 'N/A'
    downlink_ratio = args.downlink_compression_ratio if args.downlink_compression == 'uniform' else 'N/A'
    print(f"   🧬 压缩率 - 上行: {uplink_ratio}, 下行: {downlink_ratio}")
    print(f"   📉 平均训练损失: {loss_avg:.6f}")
    print(f"   📉 测试损失: {test_loss:.6f}")
    print(f"   ⏱️ 轮次耗时: {timedelta(seconds=int(epoch_duration))}")
    print(f"   ⏳ 预计剩余: {timedelta(seconds=int(estimated_remaining_time))}")

def print_communication_summary(epoch, args, idxs_users, global_model, epoch_comm_cost,
                               residual_to_broadcast_packed, communication_cost):
    """
    打印轮次通信量汇总信息。
    
    Args:
        epoch: 当前轮次
        args: 命令行参数对象
        idxs_users: 选中的客户端列表
        global_model: 全局模型
        epoch_comm_cost: 本轮上行通信开销
        residual_to_broadcast_packed: 下行广播残差
        communication_cost: 通信开销历史列表
    
    Returns:
        int: 本轮双向总通信量
    """
    print(f"\n📡 轮次 {epoch+1} 通信量汇总:")

    # --- 上行通信统计 ---
    num_selected_clients = len(idxs_users)
    single_model_bytes = sum(p.numel() for p in global_model.parameters()) * 4

    print(f"   ⬆️ 上行通信 (来自 {num_selected_clients} 个客户端):")
    # epoch_comm_cost 已经是所有选中客户端上传字节数的总和
    theoretical_uplink_baseline = single_model_bytes * num_selected_clients
    uplink_effectiveness = (1 - epoch_comm_cost / theoretical_uplink_baseline) * 100 if theoretical_uplink_baseline > 0 else 0
    print(f"      • 实际传输总字节: {int(epoch_comm_cost):,} B")
    print(f"      • 理论基准 (未压缩): {theoretical_uplink_baseline:,} B")
    print(f"      • 本轮上行压缩效果: 减少 {uplink_effectiveness:.1f}%")

    # --- 下行通信统计 ---
    print(f"   ⬇️ 下行通信 (服务器广播给所有 {args.num_users} 个客户端):")
    downlink_bytes = 0  # 初始化为0
    if residual_to_broadcast_packed is not None:
        if args.downlink_compression == 'uniform':
            # 压缩模式：调用工具函数计算下行广播的字节数
            _, single_broadcast_bytes, _ = calculate_communication_cost_dict(residual_to_broadcast_packed)
        else:
            # 全量模式：直接计算密集张量的字节数
            single_broadcast_bytes = sum(torch.numel(tensor) * 4 for tensor in residual_to_broadcast_packed.values())
        
        # 下行总量 = (所有客户端数量) × (服务器广播字典大小)
        downlink_bytes = single_broadcast_bytes * args.num_users

        # 下行基准是所有客户端都收到未压缩模型的字节数
        theoretical_downlink_baseline = single_model_bytes * args.num_users
        downlink_effectiveness = (1 - downlink_bytes / theoretical_downlink_baseline) * 100 if theoretical_downlink_baseline > 0 else 0

        print(f"      • 实际传输字节: {int(downlink_bytes):,} B ({int(single_broadcast_bytes):,} B × {args.num_users} 客户端)")
        print(f"      • 理论基准 (未压缩): {theoretical_downlink_baseline:,} B")
        print(f"      • 本轮下行压缩效果: 减少 {downlink_effectiveness:.1f}%")
    else:
        print("      • 无下行传输 (初始轮)")

    # --- 双向总通信量 ---
    total_round_comm_cost = epoch_comm_cost + downlink_bytes
    communication_cost.append(total_round_comm_cost)
    print(f"   🔄 本轮双向总通信量: {int(total_round_comm_cost):,} B")
    print(f"{'='*60}")
    
    return total_round_comm_cost

def calculate_current_learning_rate(args, epoch):
    """
    计算当前学习率。
    
    Args:
        args: 命令行参数对象
        epoch: 当前轮次
    
    Returns:
        float: 当前学习率
    """
    current_lr = args.lr
    if args.lr_scheduler == 'cosine':
        total_rounds = args.epochs
        min_lr = args.lr * 0.05
        warmup_rounds = min(5, total_rounds // 10)
        if epoch < warmup_rounds:
            current_lr = args.lr * (epoch + 1) / warmup_rounds
        else:
            effective_round = epoch - warmup_rounds
            effective_total = total_rounds - warmup_rounds
            cosine_factor = 0.5 * (1 + math.cos(math.pi * effective_round / effective_total))
            current_lr = min_lr + (args.lr - min_lr) * cosine_factor
    
    return current_lr

if __name__ == '__main__':
    main()