#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

'''
该文件主要用于定义和解析命令行参数。这些参数允许用户在运行脚本时配置联邦学习实验的各种超参数,例如学习率、epoch数量、客户端数量、模型类型等。
'''

def args_parser():
    parser = argparse.ArgumentParser() # 创建一个 ArgumentParser 对象

    # federated arguments (Notation for the arguments followed from paper)
    # 定义联邦学习相关的参数
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training") # 全局训练的轮数
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K") # 参与联邦学习的客户端总数
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C') # 每轮选择参与训练的客户端比例
    parser.add_argument('--local_ep', type=int, default=5,
                        help="the number of local epochs: E") # 将默认本地轮数降低到5
    parser.add_argument('--local_bs', type=int, default=32,
                        help="local batch size: B") # 客户端本地训练的批量大小，增加到32
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate') # 学习率，针对不同数据集和模型调整
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)') # SGD 优化器的动量，提高到0.9
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay (L2 regularization)') # 权重衰减，增加到5e-4
    parser.add_argument('--lr_scheduler', type=str, default='none',
                        help='learning rate scheduler: none, fixed, step, exp, cosine') # 学习率调度策略
    parser.add_argument('--lr_step_size', type=int, default=20,
                        help='step size for StepLR scheduler') # StepLR调度器的步长
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='gamma for StepLR and ExponentialLR scheduler') # 学习率衰减因子
    parser.add_argument('--cosine_t_max', type=int, default=50,
                        help='T_max for CosineAnnealingLR scheduler') # 余弦退火调度器的最大周期
    parser.add_argument('--adam_beta1', type=float, default=0.9,
                        help='beta1 for Adam and AdamW optimizer') # Adam优化器的beta1参数
    parser.add_argument('--adam_beta2', type=float, default=0.999,
                        help='beta2 for Adam and AdamW optimizer') # Adam优化器的beta2参数
    parser.add_argument('--adam_eps', type=float, default=1e-8,
                        help='eps for Adam and AdamW optimizer') # Adam优化器的eps参数

    # model arguments
    # 定义模型相关的参数
    parser.add_argument('--model', type=str, default='cnn', 
                       help='model name: MNIST(cnn, optimized, optimized_gn) | CIFAR(cnn, resnet18_fed, efficientnet) | CIFAR100(cnn, resnet18_fed, efficientnet, densenet)') # 使用的模型名称，支持新的优化模型
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel') # (特定于某些CNN模型) 每种卷积核的数量
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution') # (特定于某些CNN模型) 卷积核大小，逗号分隔
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs") # 图像的通道数 (例如 MNIST 为 1, CIFAR-10 为 3)
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None") # 归一化类型
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.") # (特定于某些CNN模型) 卷积核数量/滤波器数量
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions") # 是否使用最大池化

    # other arguments
    # 定义其他参数
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset") # 使用的数据集名称 (例如 'mnist', 'fmnist', 'cifar', 'cifar100')
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes") # 数据集的类别数量 (MNIST/CIFAR-10: 10, CIFAR-100: 100)
    parser.add_argument('--gpu', type=int, default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.") # 指定使用的 GPU ID，默认为 CPU
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer: sgd, adam, adamw") # 优化器类型
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.') # 数据是否独立同分布 (IID)。1 表示 IID，0 表示 Non-IID
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)') # 在 Non-IID 设置下，数据划分是否不均衡
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet distribution concentration parameter for non-IID data partitioning (smaller values create more non-IID data)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping') # 早停机制的轮数
    parser.add_argument('--verbose', type=int, default=1, help='verbose') # 是否打印详细日志
    parser.add_argument('--seed', type=int, default=1, help='random seed') # 随机种子，用于实验复现
    
    # 残差联邦学习专用参数
    # 简化版：只需要compression_ratio参数，自动启用uniform压缩，上行和下行使用相同压缩率
    parser.add_argument('--compression_ratio', type=float, default=None,
                        help='双向压缩率 - 上行和下行统一使用uniform压缩 (0~1)')
    parser.add_argument('--init_rounds', type=int, default=10,
                        help='初始模型流式传输轮数')
     
    # Non-IID优化策略参数
    parser.add_argument('--enable_knowledge_distillation', type=int, default=0,
                        help='enable knowledge distillation for Non-IID scenarios (1 for True, 0 for False)')
    parser.add_argument('--distill_temperature', type=float, default=3.0,
                        help='temperature parameter for knowledge distillation')
    parser.add_argument('--distill_alpha', type=float, default=0.3,
                        help='weight for distillation loss')
    parser.add_argument('--distill_warmup_rounds', type=int, default=3,
                        help='Rounds to wait before starting knowledge distillation')
    parser.add_argument('--enable_enhanced_augmentation', type=int, default=1,
                        help='enable enhanced data augmentation for Non-IID (1 for True, 0 for False)')
    parser.add_argument('--personalized_lr', type=int, default=0,
                        help='enable personalized learning rate adjustment (1 for True, 0 for False)') # 默认关闭有风险的个性化学习率
    parser.add_argument('--adaptive_aggregation', type=str, default='diversity_aware',
                        choices=['weighted_avg', 'diversity_aware'],
                        help='aggregation strategy for Non-IID scenarios') # 默认使用更稳健的策略
    parser.add_argument('--mu', type=float, default=0.01,
                        help='mu parameter for FedProx regularizer (0 for standard FedAvg)')
    
    # 添加报告中使用的参数别名支持
    parser.add_argument('--step_size', type=int, default=None, help='alias for lr_step_size')
    parser.add_argument('--gamma', type=float, default=None, help='alias for lr_gamma')
    parser.add_argument('--T_max', type=int, default=None, help='alias for cosine_t_max')
    parser.add_argument('--eta_min', type=float, default=1e-6, help='minimum learning rate for cosine annealing')
    parser.add_argument('--beta1', type=float, default=None, help='alias for adam_beta1')
    parser.add_argument('--beta2', type=float, default=None, help='alias for adam_beta2')
    parser.add_argument('--num_groups', type=int, default=8, help='number of groups for GroupNorm')
    
    # 添加CIFAR-100高级优化参数
    parser.add_argument('--criterion', type=str, default='cross_entropy', 
                        choices=['cross_entropy', 'label_smoothing'],
                        help='loss function type')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='label smoothing parameter (epsilon)')
    parser.add_argument('--enable_cutmix', type=int, default=0,
                        help='enable CutMix data augmentation (1 for True, 0 for False)')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0,
                        help='CutMix alpha parameter for Beta distribution')
    parser.add_argument('--cutmix_prob', type=float, default=0.5,
                        help='probability of applying CutMix')
    parser.add_argument('--enable_mixup', type=int, default=0,
                        help='enable Mixup data augmentation (1 for True, 0 for False)')
    parser.add_argument('--mixup_alpha', type=float, default=0.4,
                        help='Mixup alpha parameter for Beta distribution')
    parser.add_argument('--enable_swa', type=int, default=0,
                        help='enable Stochastic Weight Averaging (1 for True, 0 for False)')
    parser.add_argument('--swa_start', type=int, default=200,
                        help='epoch to start SWA')
    parser.add_argument('--use_attention', type=int, default=1,
                        help='use attention modules in DenseNet (1 for True, 0 for False)')
    parser.add_argument('--use_groupnorm', type=int, default=1,
                        help='use GroupNorm instead of BatchNorm in DenseNet (1 for True, 0 for False)')
    
    # CIFAR-10 Non-IID 增强参数 (仅当数据集为 cifar10 且 iid=0 时使用)
    parser.add_argument('--pretrain_epochs', type=int, default=20,
                        help='central warmup pretraining epochs for CIFAR-10 non-IID')
    parser.add_argument('--pretrain_lr', type=float, default=0.1,
                        help='learning rate for central pretraining')
    parser.add_argument('--shared_size', type=int, default=5000,
                        help='size of shared public dataset for CIFAR-10 non-IID')
    parser.add_argument('--shared_lambda', type=float, default=0.1,
                        help='weight for shared public data loss')
    parser.add_argument('--enable_cifar10_enhancements', type=int, default=1,
                        help='enable CIFAR-10 non-IID enhancements (1 for True, 0 for False)')
    
    # --- 消融实验控制参数 ---
    
    # 1. 客户端选择
    parser.add_argument('--selection_method', type=str, default='smart',
                        choices=['smart', 'random'],
                        help='客户端选择策略: smart(智能选择), random(随机选择)')

    # 2. 上行(客户端->服务器)控制
    parser.add_argument('--uplink_compression', type=str, default='none',
                        choices=['none', 'uniform'],
                        help='上行压缩类型: none(无压缩), uniform(Top-K压缩)')
    parser.add_argument('--uplink_compression_ratio', type=float, default=0.1,
                        help='上行压缩比例 (默认: 0.1)')
    parser.add_argument('--disable_uplink_ef', action='store_true',
                        help='禁用上行(客户端)误差反馈')

    # 3. 下行(服务器->客户端)控制
    parser.add_argument('--downlink_compression', type=str, default='none',
                        choices=['none', 'uniform'],
                        help='下行压缩类型: none(无压缩), uniform(Top-K压缩)')
    parser.add_argument('--downlink_compression_ratio', type=float, default=0.1,
                        help='下行压缩比例 (默认: 0.1)')
    parser.add_argument('--disable_downlink_ef', action='store_true',
                        help='禁用下行(服务器)误差反馈')

    args = parser.parse_args() # 解析命令行传入的参数
    
    # 处理参数别名映射
    if args.step_size is not None:
        args.lr_step_size = args.step_size
    if args.gamma is not None:
        args.lr_gamma = args.gamma
    if args.T_max is not None:
        args.cosine_t_max = args.T_max
    if args.beta1 is not None:
        args.adam_beta1 = args.beta1
    if args.beta2 is not None:
        args.adam_beta2 = args.beta2
    
    return args # 返回解析后的参数对象

'''
关键函数和语法:
●	import argparse: 导入Python标准库中的argparse模块,它是专门用来处理命令行参数的。
●	parser = argparse.ArgumentParser(): 创建一个ArgumentParser对象。这个对象包含了所有必要的参数信息,并能从sys.argv中解析出这些参数。
●	parser.add_argument(): 这个方法用于向ArgumentParser对象添加一个你期望程序接受的命令行选项。
○	'--epochs': 参数的名称，通常以--开头表示可选参数。
○	type=int: 指定参数的类型。argparse会尝试将输入转换为这个类型。
○	default=10: 如果命令行中没有提供这个参数，则使用这个默认值。
○	help="...": 参数的描述信息，当用户使用-h或--help选项时会显示。
●	args = parser.parse_args(): 这个方法会检查命令行,将每个参数转换成适当的类型,然后调用相应的动作。它返回一个包含所有参数及其值的对象（通常是一个Namespace对象）。
●	return args: 函数返回这个包含所有已解析参数的对象，以便在其他脚本中使用。

'''