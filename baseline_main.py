#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6  # 指定Python解释器及编码格式（兼容中文注释）

# 设置编码
import sys
import locale
import os

# 强制设置UTF-8编码
if sys.platform.startswith('win'):
    os.system('chcp 65001 > nul')  # 设置Windows控制台为UTF-8
    
# 确保print输出使用UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# 导入依赖库
from tqdm import tqdm       # 进度条显示工具
import matplotlib.pyplot as plt  # 可视化绘图库
import pandas as pd         # 数据处理和保存CSV
import torch
import sys
from datetime import datetime

# 设置输出编码
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

from torch.utils.data import DataLoader  # 数据批量加载工具

# 导入自定义模块
from utils import get_dataset    # 数据集加载工具函数
from options import args_parser # 命令行参数解析器
from update import test_inference  # 测试推理函数
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, CNNCifar100, ResNet18Fed, replace_bn_with_gn  # 自定义模型定义
from model_factory import get_model  # 新的模型工厂

'''
该文件实现了一个传统的、非联邦的（中心化）机器学习模型训练流程。它通常用作联邦学习性能的基准 (baseline)。
代码会加载数据，构建模型，在整个训练集上进行训练，并在测试集上评估模型。
'''

if __name__ == '__main__':
    # 参数解析与设备配置
    args = args_parser()  # 解析命令行参数（如模型类型、数据集、epoch数等）
    if args.gpu:          # 若启用GPU加速
        torch.cuda.set_device(int(args.gpu))  # 指定GPU设备编号
        #加了int之后就不用像之前一样在命令行输入gpu=cuda:0了,只用写gpu=0
    device = 'cuda' if args.gpu else 'cpu'  # 确定计算设备（GPU/CPU）

    # 加载数据集
    train_dataset, test_dataset, _ = get_dataset(args)  # 获取训练集、测试集及可能的额外信息
    
    # 构建模型
    print(f"正在构建基线模型: {args.model} for dataset: {args.dataset}")
    
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
                
        elif args.model in ['optimized', 'cnn_optimized']:
            # 使用新的优化CNN模型
            if args.dataset == 'mnist':
                global_model = get_model('mnist', 'optimized')  # CNN_MNIST_Optimized
            else:
                raise ValueError(f"优化CNN目前仅支持MNIST数据集")
                
        elif args.model in ['resnet18', 'resnet']:
            # 使用ResNet模型
            if args.dataset == 'cifar':
                global_model = get_model('cifar10', 'resnet18_fed')  # ResNet18_CIFAR10_Fed 
            elif args.dataset == 'cifar100':
                global_model = get_model('cifar100', 'resnet18_fed')  # ResNet18_CIFAR100_Fed
            else:
                raise ValueError(f"ResNet18不支持数据集: {args.dataset}")
                
        elif args.model in ['efficientnet']:
            # 使用EfficientNet模型
            if args.dataset == 'cifar':
                global_model = get_model('cifar10', 'efficientnet')  # EfficientNet_CIFAR10
            elif args.dataset == 'cifar100':
                global_model = get_model('cifar100', 'efficientnet')  # EfficientNet_CIFAR100
            else:
                raise ValueError(f"EfficientNet不支持数据集: {args.dataset}")
                
        elif args.model == 'densenet':
            # 使用DenseNet模型
            if args.dataset == 'cifar100':
                global_model = get_model('cifar100', 'densenet')  # DenseNet_CIFAR100
            else:
                raise ValueError(f"DenseNet不支持数据集: {args.dataset}")
        else:
            raise ValueError("尝试原有模型")
            
        print(f"[SUCCESS] 成功加载优化模型: {global_model.__class__.__name__}")
        
    except Exception as e:
        print(f"[WARNING] 优化模型加载失败: {e}")
        print(f"[INFO] 回退到原有模型...")
        
        # 回退到原有模型构建逻辑
        if args.model == 'cnn':  # 卷积神经网络模型分支
            # 根据数据集选择不同CNN结构
            if args.dataset == 'mnist':      # MNIST手写数字识别
                global_model = CNNMnist(args=args)  # 28x28灰度图输入的网络
            elif args.dataset == 'fmnist':   # Fashion-MNIST服装分类
                global_model = CNNFashion_Mnist(args=args)
            elif args.dataset == 'cifar':    # CIFAR-10图像分类
                global_model = CNNCifar(args=args)  # 适用于32x32彩色图的CNN
            elif args.dataset == 'cifar100': # CIFAR-100图像分类
                global_model = CNNCifar100(args=args)  # 适用于32x32彩色图的深度CNN
        elif args.model == 'resnet':  # ResNet18Fed模型分支
            # ResNet for better performance
            if args.dataset == 'cifar':
                global_model = ResNet18Fed(num_classes=args.num_classes)
                global_model = replace_bn_with_gn(global_model)  # 使用GroupNorm
            elif args.dataset == 'cifar100':
                global_model = ResNet18Fed(num_classes=100)
                global_model = replace_bn_with_gn(global_model)
            else:
                print(f"ResNet not implemented for dataset {args.dataset}, using CNN instead")
                if args.dataset == 'mnist':
                    global_model = CNNMnist(args=args)
                elif args.dataset == 'fmnist':
                    global_model = CNNFashion_Mnist(args=args)
        elif args.model == 'mlp':  # 多层感知机模型分支
            img_size = train_dataset[0][0].shape  # 获取输入图像尺寸
            len_in = 1
            for x in img_size:  # 计算输入层维度（展平后的像素总数）
                len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
        else:
            exit('Error: unrecognized model')  # 模型类型错误处理
    
    # 模型配置
    global_model.to(device)    # 将模型移动到指定设备（GPU/CPU）
    global_model.train()       # 设置为训练模式（启用BN/Dropout等层）
    print(global_model)        # 打印模型结构
    
    # 训练配置
    # 优化器选择（支持SGD、Adam、AdamW）
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), 
                                  lr=args.lr, 
                                  momentum=args.momentum, 
                                  weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), 
                                   lr=args.lr, 
                                   betas=(args.adam_beta1, args.adam_beta2),
                                   eps=args.adam_eps,
                                   weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(global_model.parameters(), 
                                    lr=args.lr, 
                                    betas=(args.adam_beta1, args.adam_beta2),
                                    eps=args.adam_eps,
                                    weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    # 学习率调度器
    scheduler = None
    if args.lr_scheduler in ['none', 'fixed']:
        scheduler = None  # 固定学习率，不使用调度器
    elif args.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                  step_size=args.lr_step_size, 
                                                  gamma=args.lr_gamma)
    elif args.lr_scheduler == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                                                         gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                             T_max=args.cosine_t_max,
                                                             eta_min=getattr(args, 'eta_min', 1e-6))
    else:
        print(f"警告: 未知的学习率调度器 '{args.lr_scheduler}'，将使用固定学习率")
        scheduler = None
    
    # 数据加载器（批处理+随机打乱）
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # 验证集数据加载器 - 用于早停
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    trainloader = DataLoader(train_subset, batch_size=64, shuffle=True)
    valloader = DataLoader(val_subset, batch_size=64, shuffle=False)
    
    # 损失函数：交叉熵损失（适用于原始logits输出）
    criterion = torch.nn.CrossEntropyLoss().to(device)
    epoch_loss = []  # 记录每轮平均损失
    val_losses = []  # 记录验证损失
    
    # --- 修改开始：初始化历史记录 ---
    history = {
        'epoch': [],
        'test_accuracy': [],
        'train_loss': []
    }
    # --- 修改结束 ---
    
    # 早停机制参数
    best_val_loss = float('inf')
    patience = 5  # 连续多少个epoch验证损失不下降就停止
    patience_counter = 0

    # 训练循环[6,8](@ref)
    for epoch in range(args.epochs):  # 移除tqdm，使用详细输出
        batch_loss = []
        
        # 训练阶段
        global_model.train()
        # 遍历训练数据批次
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)  # 数据送设备
            
            # 前向传播
            optimizer.zero_grad()       # 清空梯度（避免累积）
            outputs = global_model(images)  # 模型推理
            loss = criterion(outputs, labels)  # 计算损失
            
            # 反向传播与优化
            loss.backward()             # 反向传播计算梯度
            optimizer.step()            # 更新模型参数
            
            # 每50个批次打印训练状态 - 使用您要求的详细格式
            if batch_idx % 50 == 0:
                print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, 1, batch_idx * len(images), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
            batch_loss.append(loss.item())  # 记录当前批次损失
        
        # 计算并记录当前epoch平均训练损失
        loss_avg = sum(batch_loss) / len(batch_loss)
        print(f'\nTrain loss: {loss_avg}')
        epoch_loss.append(loss_avg)
        
        # 更新学习率调度器
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f'Learning rate updated to: {current_lr:.6f}')
        
        # 验证阶段
        global_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                outputs = global_model(images)
                val_loss += criterion(outputs, labels).item()
        
        val_loss /= len(valloader)
        val_losses.append(val_loss)
        print(f'Validation loss: {val_loss:.6f}')
        
        # 计算测试准确率用于记录
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        
        # --- 修改开始：记录本轮次的指标 ---
        history['epoch'].append(epoch + 1)
        history['test_accuracy'].append(test_acc)
        history['train_loss'].append(loss_avg)
        # --- 修改结束 ---
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f'New best validation loss: {val_loss:.6f}')
        else:
            patience_counter += 1
            print(f'Validation loss did not improve. Patience: {patience_counter}/{patience}')
            
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break

    # 可视化训练损失曲线
    print(f'\nTraining completed after {len(epoch_loss)} epochs')
    print(f'Final training loss: {epoch_loss[-1]:.6f}')
    print(f'Final validation loss: {val_losses[-1]:.6f}')
    print(f'Best validation loss: {best_val_loss:.6f}')
    
    # 模型测试评估
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print(f'Test on {len(test_dataset)} samples')
    print(f"Test Accuracy: {100*test_acc:.2f}%")  # 输出测试准确率（百分比形式）

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
实验类型: Baseline Centralized Learning
数据集: {args.dataset.upper()}
模型: {args.model.upper()}
训练轮数: {args.epochs}
数据分布: {iid_str.upper()}
学习率: {args.lr}
批次大小: {args.local_bs}
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

