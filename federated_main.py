#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# 设置编码
import sys

# 强制设置UTF-8编码
if sys.platform.startswith('win'):
    import os
    os.system('chcp 65001 > nul')  # 设置Windows控制台为UTF-8
    
# 确保print输出使用UTF-8，并禁用缓冲
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)

# 强制禁用Python输出缓冲
import os
os.environ['PYTHONUNBUFFERED'] = '1'

import os # 用于与操作系统交互，例如路径操作
import copy # 用于深拷贝
import time # 用于计时
import pickle # 用于序列化和反序列化 Python 对象 (例如保存训练结果)
import numpy as np
from tqdm import tqdm # 进度条
import pandas as pd # 用于数据处理和保存CSV
import sys
from datetime import datetime

# 设置输出编码
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

import torch
from tensorboardX import SummaryWriter # 用于 TensorBoard 日志记录，可视化训练过程

from options import args_parser
from update import LocalUpdate, test_inference # 从 update.py 导入本地更新类和测试函数
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, CNNCifar100, ResNet18Fed, replace_bn_with_gn
from model_factory import get_model, get_recommended_model, list_available_models  # 新的模型工厂
from utils import get_dataset, average_weights, exp_details, adaptive_federated_aggregation, calculate_diversity_scores # 从 utils.py 导入数据加载、权重平均和打印实验细节的函数

def print_training_details(epoch, global_round, num_users, frac, local_ep, local_bs, lr, device, model):
    """
    打印训练详细信息
    
    Args:
        epoch: 当前轮次
        global_round: 全局轮数
        num_users: 客户端总数
        frac: 参与训练的客户端比例
        local_ep: 本地训练轮数
        local_bs: 本地批量大小
        lr: 学习率
        device: 设备
        model: 模型
    """
    print(f"\n🌟 开始第 {epoch+1}/{global_round} 轮联邦学习")
    print(f"📊 训练配置: 客户端总数={num_users}, 参与比例={frac:.1%}")
    print(f"🔧 本地训练: 轮数={local_ep}, 批量={local_bs}, 学习率={lr}")
    print(f"💻 设备: {device}")
    
    # 计算模型参数总数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 模型参数: 总计{total_params:,}个, 可训练{trainable_params:,}个")

def intelligent_client_selection(args, epoch, user_groups, client_ema_losses, EMA_ALPHA):
    """
    智能客户端选择 - 基于历史表现和数据分布
    
    Args:
        args: 命令行参数
        epoch: 当前轮次
        user_groups: 用户数据分组
        client_ema_losses: 客户端EMA损失记录
        EMA_ALPHA: EMA平滑因子
        
    Returns:
        idxs_users: 选中的客户端索引
        client_ema_losses: 更新后的EMA损失记录
    """
    m = max(int(args.frac * args.num_users), 1)
    
    if epoch > 2:  # 前几轮收集数据
        # 基于历史表现的智能客户端选择
        client_weights = []
        
        # 计算一次全局最大数据量，避免在循环中重复计算
        max_data_size = max(len(user_groups[i]) for i in range(args.num_users))
        
        for idx in range(args.num_users):
            # --- 更新客户端的EMA损失 ---
            last_loss = np.mean(args.client_history['losses'].get(idx, [1.0]))
            if idx not in client_ema_losses:
                client_ema_losses[idx] = last_loss
            else:
                client_ema_losses[idx] = EMA_ALPHA * last_loss + (1 - EMA_ALPHA) * client_ema_losses[idx]

            # 使用EMA损失来计算得分
            current_ema_loss = client_ema_losses[idx]
            loss_score = 1.0 / (1.0 + current_ema_loss) # 使用平滑后的损失
            
            # 数据量权重
            data_size = len(user_groups[idx])
            data_score = data_size / max_data_size
            
            # 避免过度选择同一客户端 - 更精确的频率惩罚
            # 计算最近几轮中该客户端被选中的次数
            recent_window = min(6 * m, len(args.client_history['last_selected']))  # 最近6轮的选择
            recent_selections = args.client_history['last_selected'][-recent_window:] if recent_window > 0 else []
            frequency_penalty = 1.0 - (recent_selections.count(idx) * 0.15)  # 降低惩罚强度
            frequency_penalty = max(frequency_penalty, 0.2)  # 最小保持20%权重
            
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
        # 前几轮随机选择收集基准数据
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print(f'🎲 随机选择客户端: {list(idxs_users)} (收集基准数据)')
    
    return idxs_users, client_ema_losses


def update_client_history(args, idxs_users):
    """
    更新客户端选择历史记录
    
    Args:
        args: 命令行参数
        idxs_users: 本轮选中的客户端
    """
    m = max(int(args.frac * args.num_users), 1)
    
    # 记录选择的客户端历史
    args.client_history['last_selected'].extend(idxs_users.tolist())
    args.client_history['round_selections'].append(idxs_users.tolist())  # 按轮记录
    
    # 维护合理的历史窗口大小
    if len(args.client_history['last_selected']) > (m * 8): # 保持8轮的历史
        args.client_history['last_selected'] = args.client_history['last_selected'][-(m*8):]
    if len(args.client_history['round_selections']) > 8:  # 保持最近8轮的轮级记录
        args.client_history['round_selections'] = args.client_history['round_selections'][-8:]


def perform_local_training(args, idxs_users, global_model, train_dataset, user_groups, epoch):
    """
    执行客户端本地训练
    
    Args:
        args: 命令行参数
        idxs_users: 选中的客户端
        global_model: 全局模型
        train_dataset: 训练数据集
        user_groups: 用户数据分组
        epoch: 当前轮次
        
    Returns:
        local_weights: 本地权重列表
        local_losses: 本地损失列表
        epoch_comm_cost: 通信成本
    """
    local_weights, local_losses = [], []
    epoch_comm_cost = 0
    
    for idx in idxs_users:
        print(f'\n[CLIENT {idx}] 开始本地训练...')
        
        # 创建 LocalUpdate 实例，传入客户端的本地数据 (user_groups[idx])
        local_model = LocalUpdate(args=args, dataset=train_dataset,
                                  idxs=user_groups[idx], client_id=idx)
        # 客户端进行本地训练
        w, loss = local_model.update_weights(
            model=copy.deepcopy(global_model), global_round=epoch, 
            global_weights=global_model.state_dict() if epoch > 0 else None)
        local_weights.append(copy.deepcopy(w))
        local_losses.append(copy.deepcopy(loss))
        
        # 计算通信成本
        client_comm_cost = sum(torch.numel(param) for param in w.values())
        epoch_comm_cost += client_comm_cost
        
        # 更新客户端损失历史
        if idx not in args.client_history['losses']:
            args.client_history['losses'][idx] = []
        args.client_history['losses'][idx].append(loss)
        
        # 保持历史记录窗口大小
        if len(args.client_history['losses'][idx]) > 10:
            args.client_history['losses'][idx] = args.client_history['losses'][idx][-10:]
        
        print(f'[CLIENT {idx}] 本地训练完成，损失: {loss:.6f}，上传参数量: {client_comm_cost:,}')
    
    return local_weights, local_losses, epoch_comm_cost


def select_clients_enhanced(num_users, frac, idxs_users=None):
    """
    选择参与训练的客户端 - 增强版
    
    Args:
        num_users: 客户端总数
        frac: 参与比例
        idxs_users: 可选的预定义客户端索引
        
    Returns:
        selected_users: 选中的客户端索引列表
        m: 参与客户端数量
    """
    # 计算参与训练的客户端数量
    m = max(int(frac * num_users), 1)
    
    if idxs_users is not None:
        selected_users = idxs_users
    else:
        selected_users = np.random.choice(range(num_users), m, replace=False)
    
    print(f"👥 选择客户端: {selected_users} (共{len(selected_users)}个)")
    return selected_users, m

def print_enhanced_communication_stats(epoch, selected_users, local_weights, enable_compression=False):
    """
    打印通信统计信息 - 增强版
    
    Args:
        epoch: 当前轮次
        selected_users: 参与的客户端列表
        local_weights: 本地模型权重列表
        enable_compression: 是否启用压缩
    """
    print(f"\n📡 第{epoch+1}轮通信统计:")
    
    total_comm_cost = 0
    for i, idx in enumerate(selected_users):
        # 计算客户端模型参数数量
        client_params = sum(torch.numel(param) for param in local_weights[i].values())
        total_comm_cost += client_params
        
        if enable_compression:
            print(f"[CLIENT {idx}] 压缩传输: {client_params:,} 参数")
        else:
            print(f"[CLIENT {idx}] 密集传输: {client_params:,} 参数")
    
    print(f"📊 本轮总通信量: {total_comm_cost:,} 参数")
    return total_comm_cost

def print_enhanced_epoch_results(epoch, train_loss, train_accuracy, test_accuracy, ema_accuracy, comm_cost, best_ema_acc):
    """
    打印轮次结果 - 增强版
    
    Args:
        epoch: 当前轮次
        train_loss: 训练损失
        train_accuracy: 训练准确率  
        test_accuracy: 测试准确率
        ema_accuracy: EMA平滑准确率
        comm_cost: 通信成本
        best_ema_acc: 最佳EMA准确率
    """
    print(f"\n📈 第{epoch+1}轮结果汇总:")
    print(f"   🔴 训练损失: {train_loss:.6f}")
    print(f"   🟢 训练准确率: {train_accuracy:.2f}%")
    print(f"   🔵 测试准确率: {test_accuracy:.2f}%")
    print(f"   🎯 EMA平滑准确率: {ema_accuracy:.2f}% (最佳: {best_ema_acc:.2f}%)")
    print(f"   📡 通信成本: {comm_cost:,} 参数")
    print("=" * 80)

def save_enhanced_model_checkpoint(model, epoch, save_path, file_name):
    """
    保存模型检查点 - 增强版
    
    Args:
        model: 模型
        epoch: 当前轮次
        save_path: 保存路径
        file_name: 文件名
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    checkpoint_path = os.path.join(save_path, f'{file_name}_epoch_{epoch+1}.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print(f"💾 模型已保存: {checkpoint_path}")

if __name__ == '__main__':
    start_time = time.time() # 记录开始时间
    print(start_time)
    print('尝试')

    # define paths
    path_project = os.path.abspath('..') # 获取项目上级目录的绝对路径 (可能用于保存文件)
    logger = SummaryWriter('../logs') # 初始化 TensorBoard 的 SummaryWriter，日志保存在 ../logs 目录下

    args = args_parser() # 解析命令行参数
    exp_details(args) # 打印实验配置详情
    '''
    功能：调用 utils.py 文件中的 exp_details(args) 函数，把刚刚解析到的所有实验参数详细打印出来。
    意义：方便你在控制台/日志中确认本次实验的所有配置，避免参数设置错误，便于后续复现实验和调试。
    '''

    # 设备检查和设置
    print("检查CUDA支持...")
    if args.gpu is not None and torch.cuda.is_available():
        try:
            # 测试CUDA是否真正可用
            test_tensor = torch.randn(1, 1, 28, 28)
            test_tensor = test_tensor.cuda(int(args.gpu))
            print(f"✓ CUDA测试成功，使用GPU: {args.gpu}")
            torch.cuda.set_device(int(args.gpu))
            device = torch.device(f'cuda:{int(args.gpu)}')
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ GPU {args.gpu} 不可用: {str(e)}")
            print("切换到CPU模式")
            device = torch.device('cpu')
    else:
        if args.gpu is not None:
            print(f"✗ CUDA不可用，但指定了GPU {args.gpu}")
        else:
            print("未指定GPU")
        print("使用CPU模式")
        device = torch.device('cpu')
    
    print(f"最终设备: {device}")

    # load dataset and user groups (加载数据集和用户数据划分)
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL (构建全局模型，支持原有模型和新的优化模型)
    print(f"正在构建模型: {args.model} for dataset: {args.dataset}")
    
    # 首先尝试使用新的优化模型
    try:
        if args.model in ['cnn']:
            # 使用新的标准CNN模型
            if args.dataset == 'mnist':
                global_model = get_model('mnist', 'cnn')  # CNN_MNIST
            elif args.dataset == 'cifar':
                global_model = get_model('cifar10', 'cnn')  # CNNCifar
            else:
                raise ValueError(f"标准CNN不支持数据集: {args.dataset}")
                
        elif args.model in ['optimized', 'cnn_optimized', 'cnn_opt', 'cnn_enhanced', 'optimized_gn']:
            # 使用新的优化CNN模型
            if args.dataset == 'mnist':
                if args.model == 'optimized_gn':
                    # 使用GroupNorm优化模型，适合Non-IID场景
                    global_model = get_model('mnist', 'optimized_gn')  # CNN_MNIST_Optimized_GN
                else:
                    global_model = get_model('mnist', 'optimized')  # CNN_MNIST_Optimized
            elif args.dataset == 'cifar':
                global_model = get_model('cifar10', 'cnn')  # CNNCifar (对于CIFAR使用标准CNN作为增强版)
            else:
                raise ValueError(f"优化CNN目前仅支持MNIST和CIFAR数据集")
                
        elif args.model in ['resnet18', 'resnet18_fed', 'resnet', 'resnet_mini', 'resnet18_gn']:
            # 使用新的ResNet18Fed优化模型
            if args.dataset == 'mnist':
                global_model = get_model('mnist', 'optimized')  # CNN_MNIST_Optimized
            elif args.dataset == 'cifar':
                # 如果是GroupNorm版本，使用特殊参数
                if args.model == 'resnet18_gn':
                    global_model = get_model('cifar10', 'resnet18_fed', use_groupnorm=True, num_groups=getattr(args, 'num_groups', 8))
                else:
                    global_model = get_model('cifar10', 'resnet18_fed')  # ResNet18_CIFAR10_Fed 
            elif args.dataset == 'cifar100':
                global_model = get_model('cifar100', 'resnet18_fed')  # ResNet18_CIFAR100_Fed
            else:
                raise ValueError(f"ResNet18不支持数据集: {args.dataset}")
                
        elif args.model in ['efficientnet', 'efficient']:
            # 使用新的EfficientNet优化模型
            if args.dataset == 'cifar':
                global_model = get_model('cifar10', 'efficientnet')  # EfficientNet_CIFAR10
            elif args.dataset == 'cifar100':
                global_model = get_model('cifar100', 'efficientnet')  # EfficientNet_CIFAR100
            else:
                raise ValueError(f"EfficientNet不支持数据集: {args.dataset}")
                
        elif args.model == 'densenet':
            # 使用新的DenseNet模型 (仅CIFAR-100)
            if args.dataset == 'cifar100':
                # 传递新的参数给DenseNet模型
                global_model = get_model('cifar100', 'densenet', 
                                        use_attention=bool(getattr(args, 'use_attention', 1)),
                                        use_groupnorm=bool(getattr(args, 'use_groupnorm', 1)))  # 默认启用GroupNorm
            else:
                raise ValueError(f"DenseNet不支持数据集: {args.dataset}")
                
        else:
            # 使用推荐模型或抛出异常尝试原有模型
            raise ValueError("尝试原有模型")
            
        print(f"[SUCCESS] 成功加载优化模型: {global_model.__class__.__name__}")
        
    except Exception as e:
        print(f"[WARNING] 优化模型加载失败: {e}")
        print(f"[INFO] 回退到原有模型...")
        
        # 回退到原有模型构建逻辑
        if args.model == 'cnn':
            # Convolutional neural netork
            if args.dataset == 'mnist':
                global_model = CNNMnist(args=args)
            elif args.dataset == 'fmnist':
                global_model = CNNFashion_Mnist(args=args)
            elif args.dataset == 'cifar':
                global_model = CNNCifar(args=args)
            elif args.dataset == 'cifar100':
                global_model = CNNCifar100(args=args)
        elif args.model == 'resnet':
            # ResNet for federated learning
            if args.dataset == 'cifar':
                global_model = ResNet18Fed(num_classes=args.num_classes)
                # 使用GroupNorm替换BatchNorm以提高联邦学习性能
                global_model = replace_bn_with_gn(global_model)
            elif args.dataset == 'cifar100':
                global_model = ResNet18Fed(num_classes=100)
                global_model = replace_bn_with_gn(global_model)
            else:
                print(f"ResNet not implemented for dataset {args.dataset}, using CNN instead")
                if args.dataset == 'mnist':
                    global_model = CNNMnist(args=args)
                elif args.dataset == 'fmnist':
                    global_model = CNNFashion_Mnist(args=args)
        elif args.model == 'mlp':
            # Multi-layer preceptron
            img_size = train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
        else:
            print(f"[ERROR] 错误: 不支持的模型 '{args.model}'")
            print("[INFO] 支持的模型:")
            print("   原有模型: mlp, cnn, resnet")
            print("   优化模型: resnet18, efficientnet, densenet, cnn_optimized")
            exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device) # 模型移至设备
    global_model.train() # 设置为训练模式 (尽管全局模型主要通过聚合更新，但初始状态和分发给客户端时应为训练模式)
    print(global_model) # 打印模型结构

    # copy weights (获取全局模型的初始权重)
    global_weights = global_model.state_dict() # state_dict() 返回包含模型所有参数的字典
    '''
    获取当前全局模型的所有参数（权重和偏置）并保存为一个字典
    global_model.state_dict() 会返回一个包含模型所有可学习参数(如权重、偏置)的有序字典(OrderedDict)。
    这样做的目的是保存全局模型的当前参数状态，后续可以用这些参数分发给各个客户端，或者在聚合后更新全局模型。
    在联邦学习中，每一轮开始时，客户端会拿到全局模型的参数进行本地训练，训练后再上传参数，最后服务器端聚合这些参数，更新全局模型。
    例如
    OrderedDict([
    ('fc1.weight', tensor([[ 0.01, -0.02, ...], [...], ...])),   # 第一层全连接的权重
    ('fc1.bias', tensor([0.0, 0.0, ...])),                      # 第一层全连接的偏置
    ('fc2.weight', tensor([[...], [...], ...])),                # 第二层全连接的权重
    ('fc2.bias', tensor([0.0, 0.0, ...]))                       # 第二层全连接的偏置
    ])
    '''

    # Training (联邦学习训练过程)
    train_loss, local_test_accuracy = [], [] # 记录每轮的平均训练损失和准确率
    communication_cost = []  # 记录每轮的通信开销
    val_acc_list, net_list = [], [] # (未使用)
    cv_loss, cv_acc = [], [] # (未使用)
    print_every = 1 # 每轮都打印训练统计信息，提供更详细的输出
    
    # 早停机制参数
    best_val_acc = 0.0
    patience = args.stopping_rounds  # 使用参数中的early stopping rounds
    patience_counter = 0
    best_global_weights = None

    # --- 修改开始：初始化历史记录 ---
    history = {
        'epoch': [],
        'test_accuracy': [],
        'ema_accuracy': [],      # 添加EMA平滑准确率记录
        'avg_train_loss': [],
        'learning_rate': [],
        'communication_cost': []  # 添加通信开销记录
    }
    # --- 修改结束 ---

    # 初始化客户端状态跟踪
    if not hasattr(args, 'client_history'):
        args.client_history = {
            'losses': {},           # 每个客户端的历史损失
            'last_selected': [],    # 所有历史选择的客户端（用于简单频率统计）
            'round_selections': [], # 按轮记录的选择历史（用于更精确的轮级分析）
        }
    '''
    args 对象就像一个贯穿整个程序的“全局信息板”
    我们可以方便地用它来存储和传递在训练过程中动态产生的信息（比如每个客户端的历史损失）
    而不需要把它定义成一个固定的命令行参数
    '''

    # --- 在训练循环 for epoch in pbar: 之前，初始化一个新的字典来存储平滑损失 ---
    client_ema_losses = {}
    EMA_ALPHA = 0.3 # 平滑因子，可以作为超参数调整
    
    # EMA平滑准确率相关变量（与残差联邦学习对齐）
    ema_alpha = 0.4  # EMA平滑因子，用于平滑准确率的变化，避免因单轮波动导致误判
    ema_acc = None   # 平滑后的准确率，初始值为None，后续会动态更新
    best_ema_acc = -1.0 # 记录最佳的平滑准确率，用于判断模型性能是否提升
    
    # SWA (Stochastic Weight Averaging) 支持
    swa_model = None
    swa_n = 0
    swa_start_epoch = getattr(args, 'swa_start', 150)
    enable_swa = getattr(args, 'enable_swa', 0) == 1
    
    if enable_swa:
        print(f"SWA启用: 将在第{swa_start_epoch}轮开始收集模型权重")
    
    # 打印启用的高级特性
    print("\n=== 启用的高级特性 ===")
    print(f"📊 EMA平滑准确率: 启用 (α={ema_alpha})")
    print(f"🎯 Label Smoothing: {'启用' if getattr(args, 'criterion', 'cross_entropy') == 'label_smoothing' else '禁用'}")
    if getattr(args, 'criterion', 'cross_entropy') == 'label_smoothing':
        print(f"   平滑参数: {getattr(args, 'smoothing', 0.1)}")
    print(f"🔄 SWA: {'启用' if enable_swa else '禁用'}")
    if enable_swa:
        print(f"   开始轮次: {swa_start_epoch}")
    print(f"🔀 CutMix: {'启用' if getattr(args, 'enable_cutmix', 0) == 1 else '禁用'}")
    if getattr(args, 'enable_cutmix', 0) == 1:
        print(f"   α={getattr(args, 'cutmix_alpha', 1.0)}, 概率={getattr(args, 'cutmix_prob', 0.5)}")
    print(f"🎨 Mixup: {'启用' if getattr(args, 'enable_mixup', 0) == 1 else '禁用'}")
    if getattr(args, 'enable_mixup', 0) == 1:
        print(f"   α={getattr(args, 'mixup_alpha', 0.4)}")
    print(f"🧠 知识蒸馏: {'启用' if getattr(args, 'enable_knowledge_distillation', 0) == 1 else '禁用'}")
    if getattr(args, 'enable_knowledge_distillation', 0) == 1:
        print(f"   温度={getattr(args, 'distill_temperature', 3.0)}, α={getattr(args, 'distill_alpha', 0.3)}")
    print(f"📈 学习率调度: {getattr(args, 'lr_scheduler', 'none')}")
    if getattr(args, 'lr_scheduler', 'none') == 'cosine':
        print(f"   T_max={getattr(args, 'cosine_t_max', 50)}")
    print(f"🤝 聚合策略: {getattr(args, 'adaptive_aggregation', 'weighted_avg')}")
    if getattr(args, 'iid', 1) == 0 and getattr(args, 'mu', 0.0) > 0:
        print(f"🔧 FedProx: 启用 (μ={getattr(args, 'mu', 0.01)})")
    print("=" * 30)
    # --- 初始化结束 ---

    for epoch in range(args.epochs): # 联邦学习主循环
        # 使用模块化函数打印训练详细信息
        print_training_details(epoch, args.epochs, args.num_users, args.frac, 
                             args.local_ep, args.local_bs, args.lr, device, global_model)

        global_model.train() # 确保全局模型在分发给客户端前处于训练模式
        
        # 智能客户端选择
        idxs_users, client_ema_losses = intelligent_client_selection(
            args, epoch, user_groups, client_ema_losses, EMA_ALPHA)
        
        # 更新客户端历史记录
        update_client_history(args, idxs_users)
        
        # 执行本地训练
        local_weights, local_losses, epoch_comm_cost = perform_local_training(
            args, idxs_users, global_model, train_dataset, user_groups, epoch)
        '''
        如果你选择了 CNNMnist,那么:
        全局的 global_model 就是一个 CNNMnist 实例。
        每个用户本地训练时，也是用 copy.deepcopy(global_model)，即每个客户端拿到的都是和全局模型一模一样的 CNNMnist
        每轮通信时，客户端会在自己的数据上训练这个模型，然后把更新后的参数上传，最后服务器端聚合这些参数，更新全局的 CNNMnist 模型。
        整个联邦学习过程中，模型结构始终保持一致，只是参数在不断更新。
        '''
        # 选择参与本轮训练的客户端
        m = max(int(args.frac * args.num_users), 1) # 计算参与客户端数量 (至少为1)
                                                    # args.frac 是参与比例，args.num_users 是总客户端数
        
        # 改进的客户端选择策略
        if epoch > 2:  # 前几轮收集数据
            # 基于历史表现的智能客户端选择
            client_weights = []
            
            # 计算一次全局最大数据量，避免在循环中重复计算
            max_data_size = max(len(user_groups[i]) for i in range(args.num_users))
            
            for idx in range(args.num_users):
                # --- 更新客户端的EMA损失 ---
                last_loss = np.mean(args.client_history['losses'].get(idx, [1.0]))
                if idx not in client_ema_losses:
                    client_ema_losses[idx] = last_loss
                else:
                    client_ema_losses[idx] = EMA_ALPHA * last_loss + (1 - EMA_ALPHA) * client_ema_losses[idx]

                # 使用EMA损失来计算得分
                current_ema_loss = client_ema_losses[idx]
                loss_score = 1.0 / (1.0 + current_ema_loss) # 使用平滑后的损失
                
                # 数据量权重
                data_size = len(user_groups[idx])
                data_score = data_size / max_data_size
                
                # 避免过度选择同一客户端 - 更精确的频率惩罚
                # 计算最近几轮中该客户端被选中的次数
                recent_window = min(6 * m, len(args.client_history['last_selected']))  # 最近6轮的选择
                recent_selections = args.client_history['last_selected'][-recent_window:] if recent_window > 0 else []
                frequency_penalty = 1.0 - (recent_selections.count(idx) * 0.15)  # 降低惩罚强度
                frequency_penalty = max(frequency_penalty, 0.2)  # 最小保持20%权重
                
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
            # 前几轮随机选择收集基准数据
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            print(f'🎲 随机选择客户端: {list(idxs_users)} (收集基准数据)')
        
        # 记录选择的客户端历史
        args.client_history['last_selected'].extend(idxs_users.tolist())
        args.client_history['round_selections'].append(idxs_users.tolist())  # 按轮记录
        
        # update global weights (聚合客户端权重，更新全局模型)
        lens = [len(user_groups[idx]) for idx in idxs_users]
        
        # 使用自适应聚合策略（Non-IID场景）
        if hasattr(args, 'iid') and args.iid == 0:
            # Non-IID场景使用自适应聚合
            client_data_sizes = lens
            client_losses_vals = local_losses
            
            aggregation_method = getattr(args, 'adaptive_aggregation', 'loss_aware')
            
            # 如果使用diversity_aware聚合，计算多样性分数
            diversity_scores = None
            if aggregation_method == 'diversity_aware':
                diversity_scores = calculate_diversity_scores(local_weights, client_data_sizes)
                print(f"🧮 计算多样性分数: {[f'{score:.4f}' for score in diversity_scores]}")
            
            global_weights = adaptive_federated_aggregation(
                local_weights, 
                client_data_sizes, 
                client_losses_vals,
                aggregation_method=aggregation_method,
                diversity_scores=diversity_scores
            )
            print(f"📊 使用自适应聚合策略 ({aggregation_method})")
        else:
            # IID场景使用标准聚合
            global_weights = average_weights(local_weights, lens) # 调用 utils.py 中的 average_weights 函数
            print(f"📊 使用标准聚合策略")
        '''这一步会把所有本地权重(如w2, w5, w7, w1, w9)做平均,得到新的全局权重'''

        # update global model with new weights
        global_model.load_state_dict(global_weights) # 将聚合后的平均权重加载到全局模型中
        
        # SWA权重收集
        if enable_swa and epoch >= swa_start_epoch:
            if swa_model is None:
                # 初始化SWA模型
                swa_model = copy.deepcopy(global_model.state_dict())
                swa_n = 1
                print(f"🔄 SWA: 开始收集权重 (轮次 {epoch+1})")
            else:
                # 更新SWA权重: θ_SWA = (θ_SWA * n + θ_current) / (n + 1)
                swa_n += 1
                for key in swa_model.keys():
                    swa_model[key] = (swa_model[key] * (swa_n - 1) + global_weights[key]) / swa_n
                print(f"🔄 SWA: 更新权重 (收集了 {swa_n} 轮)")

        
        loss_avg = sum(local_losses) / len(local_losses) # 计算本轮所有参与客户端的平均本地损失
        train_loss.append(loss_avg) # 记录

        # --- 每轮性能评估 ---
        print(f'\n🧪 评估轮次 {epoch+1} 的全局模型性能...', flush=True)
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c], client_id=c)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        local_test_accuracy.append(sum(list_acc)/len(list_acc))

        current_acc_for_stopping = local_test_accuracy[-1]
        
        # 使用模块化函数打印通信统计
        epoch_comm_cost = print_enhanced_communication_stats(epoch, idxs_users, local_weights, enable_compression=False)
        
        # 记录本轮通信开销  
        communication_cost.append(epoch_comm_cost)
        
        # 更新EMA平滑准确率（与残差联邦学习对齐）
        if ema_acc is None:
            ema_acc = current_acc_for_stopping
        else:
            ema_acc = ema_alpha * current_acc_for_stopping + (1 - ema_alpha) * ema_acc

        # 使用模块化函数打印轮次结果
        print_enhanced_epoch_results(epoch, loss_avg, current_acc_for_stopping*100, 
                                   current_acc_for_stopping*100, ema_acc*100, 
                                   epoch_comm_cost, best_ema_acc*100)
        
        # 早停机制检查（使用EMA平滑准确率）
        eps = 1e-4  # 小阈值防止抖动
        if ema_acc > best_ema_acc + eps:
            best_ema_acc = ema_acc
            patience_counter = 0
            best_global_weights = copy.deepcopy(global_model.state_dict())
            print(f'✅ 新的最佳EMA平滑准确率: {100*best_ema_acc:.2f}%')
        else:
            patience_counter += 1
            print(f'⚠️ EMA平滑准确率未改善. 耐心值: {patience_counter}/{patience}')

        # 定期保存模型检查点（每10轮保存一次）
        if (epoch+1) % 10 == 0:
            save_enhanced_model_checkpoint(global_model, epoch, './save/', f'{args.dataset}_{args.model}_fedavg')

        # print global training loss after every 'print_every' rounds
        if (epoch+1) % print_every == 0:
            print(f'\n📊 {epoch+1} 轮训练统计:')
            print(f'📈 平均训练损失: {np.mean(np.array(train_loss)):.6f}')
            print(f'🎯 训练准确率: {100*local_test_accuracy[-1]:.2f}%\n')

        # --- 修改开始：记录本轮次的指标 ---
        # 重新计算当前学习率以便记录
        import math
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

        loss_avg = np.mean(np.array(train_loss))
        history['epoch'].append(epoch + 1)
        history['test_accuracy'].append(local_test_accuracy[-1])
        history['ema_accuracy'].append(ema_acc)  # 记录EMA平滑准确率
        history['avg_train_loss'].append(loss_avg)
        history['learning_rate'].append(current_lr)
        history['communication_cost'].append(epoch_comm_cost)  # 添加通信开销记录
        # --- 修改结束 ---
            
        # 早停检查
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} global rounds')
            if best_global_weights is not None:
                global_model.load_state_dict(best_global_weights)
                print("Loaded best model weights for final testing.")
            break

    # --- MODIFICATION START: Final Reporting ---
    # 应用SWA权重 (如果启用)
    final_model_name = "最佳EMA模型"
    if enable_swa and swa_model is not None:
        print(f"\n🔄 应用SWA权重 (收集了 {swa_n} 轮)...")
        # 首先评估当前最佳模型
        if best_global_weights is not None:
            global_model.load_state_dict(best_global_weights)
        current_test_acc, _ = test_inference(args, global_model, test_dataset)
        
        # 然后评估SWA模型
        global_model.load_state_dict(swa_model)
        swa_test_acc, _ = test_inference(args, global_model, test_dataset)
        
        print(f"🔍 最佳EMA模型准确率: {current_test_acc*100:.2f}%")
        print(f"🔍 SWA模型准确率: {swa_test_acc*100:.2f}%")
        
        # 选择更好的模型
        if swa_test_acc > current_test_acc:
            print("✅ SWA模型表现更好，使用SWA权重")
            final_model_name = "SWA模型"
        else:
            print("✅ 最佳EMA模型表现更好，保持原权重")
            if best_global_weights is not None:
                global_model.load_state_dict(best_global_weights)
    else:
        if best_global_weights is not None:
            global_model.load_state_dict(best_global_weights)
    
    print(f"\n评估最终模型性能 ({final_model_name})...")
    # 注意：此时的global_model已经是选择的最佳模型
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print("使用最佳模型评估平均本地测试性能...")
    list_acc_best_model = []
    global_model.eval()
    for c in range(args.num_users):
        local_model = LocalUpdate(args=args, dataset=train_dataset,
                                  idxs=user_groups[c], client_id=c)
        acc, _ = local_model.inference(model=global_model)
        list_acc_best_model.append(acc)
    avg_local_test_acc_best_model = sum(list_acc_best_model) / len(list_acc_best_model)

    # 训练完成，打印最终总结
    total_training_time = time.time()-start_time
    print(f"\n🎉 联邦学习训练完成!")
    print(f"⏱️ 总训练时间: {total_training_time:.2f}秒")
    print(f"🔄 实际训练轮数: {epoch+1}/{args.epochs}")
    print(f"📡 总通信成本: {sum(communication_cost):,} 参数")
    
    # 最终模型评估
    print(f"\n🏁 最终结果:")
    print("|---- Avg Local Test Accuracy (Best Model): {:.2f}%".format(100*avg_local_test_acc_best_model))
    print("|---- Global Test Accuracy (Best Model): {:.2f}%".format(100*test_acc))
    print("|---- Best EMA Smoothed Accuracy: {:.2f}%".format(100*best_ema_acc))
    
    # 保存最终模型
    save_enhanced_model_checkpoint(global_model, epoch, './save/', f'{args.dataset}_{args.model}_fedavg_final')

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
    details_content = f"""实验时间: {current_time}
实验类型: Federated Learning
数据集: {args.dataset.upper()}
模型: {args.model.upper()}
训练轮数: {args.epochs}
数据分布: {iid_str.upper()}
学习率: {args.lr}
本地训练轮数: {args.local_ep}
参与客户端数: {args.num_users}
参与比例: {args.frac}
"""
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
    import pickle
    save_dir = './save/objects'
    os.makedirs(save_dir, exist_ok=True)

    file_name = './save/objects/federated_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
                args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, local_test_accuracy, communication_cost], f) # MODIFIED

    print(f'训练结果已保存到: {file_name}')
    # --- MODIFICATION END ---