#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy # 用于深拷贝对象，特别是模型权重
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
# 从 sampling.py 导入数据划分函数
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid


class CutMix:
    """
    CutMix数据增强
    ==============
    
    CutMix是一种强大的数据增强技术，通过剪切和粘贴图像区域来生成新的训练样本。
    相比于Mixup，CutMix保持了图像的空间结构，在CIFAR-100等精细分类任务上效果更好。
    
    原理：
    - 从一张图像中剪切一个矩形区域
    - 用另一张图像的对应区域填充
    - 标签按照区域比例混合
    """
    
    def __init__(self, alpha=1.0, prob=0.5):
        self.alpha = alpha  # Beta分布参数，控制混合比例
        self.prob = prob    # 应用CutMix的概率
    
    def __call__(self, batch):
        if np.random.rand() > self.prob:
            return batch
        
        images, labels = batch
        batch_size = images.size(0)
        
        # 生成混合比例
        lam = np.random.beta(self.alpha, self.alpha)
        
        # 随机选择混合的样本对
        index = torch.randperm(batch_size)
        
        # 计算剪切区域
        W, H = images.size(3), images.size(2)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # 随机选择剪切位置
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # 执行剪切和粘贴
        images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        
        # 调整混合比例
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        return images, (labels, labels[index], lam)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    标签平滑交叉熵损失
    ==================
    
    标签平滑是一种正则化技术，通过"软化"硬标签来防止模型过拟合。
    特别适用于类别数较多的数据集（如CIFAR-100）。
    
    原理：
    - 将硬标签 [0, 0, 1, 0, ...] 转换为软标签 [ε/K, ε/K, 1-ε+ε/K, ε/K, ...]
    - ε是平滑参数，K是类别数
    - 鼓励模型不要过于自信于单一类别
    """
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        """
        pred: [N, C] 预测logits
        target: [N] 真实标签
        """
        N, C = pred.size()
        
        # 转换为one-hot编码
        target_one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        
        # 应用标签平滑
        target_smooth = target_one_hot * (1 - self.smoothing) + self.smoothing / C
        
        # 计算交叉熵损失
        log_pred = F.log_softmax(pred, dim=1)
        loss = -(target_smooth * log_pred).sum(dim=1).mean()
        
        return loss


def mixup_data(x, y, alpha=0.4):
    """
    Mixup数据增强
    =============
    
    通过线性插值混合两个样本和对应标签。
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup损失计算"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def check_dataset_exists(dataset_name, data_dir):
    """检查数据集是否已经下载"""
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'mnist':
        check_files = ['MNIST/raw/train-images-idx3-ubyte', 'MNIST/raw/t10k-images-idx3-ubyte']
    elif dataset_name == 'cifar10':
        check_files = ['cifar-10-batches-py/data_batch_1', 'cifar-10-batches-py/test_batch']
    elif dataset_name == 'cifar100':
        check_files = ['cifar-100-python/train', 'cifar-100-python/test']
    elif dataset_name == 'fmnist':
        check_files = ['FashionMNIST/raw/train-images-idx3-ubyte', 'FashionMNIST/raw/t10k-images-idx3-ubyte']
    else:
        return False
    
    # 检查关键文件是否存在
    for file_path in check_files:
        if not os.path.exists(os.path.join(data_dir, file_path)):
            return False
    return True


def get_dataset(args): # 根据参数加载并划分数据集
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    返回:
        train_dataset: 原始的完整训练数据集
        test_dataset: 原始的完整测试数据集
        user_groups: 一个字典,键是用户ID,值是分配给该用户的数据索引
    """

    if args.dataset == 'cifar': # 如果是 CIFAR-10 数据集
        data_dir = '../data/cifar/' # 数据存储目录
        
        # 检查数据集是否存在
        dataset_exists = check_dataset_exists('cifar10', data_dir)
        if dataset_exists:
            print("[INFO] 找到已存在的 CIFAR-10 数据集")
        else:
            print("[INFO] CIFAR-10 数据集不存在，将自动下载")
        
        # 定义 CIFAR-10 的强化数据增强策略（训练时）
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 随机裁剪，边缘填充4像素
            transforms.RandomHorizontalFlip(p=0.5),  # 50%概率水平翻转
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3)),  # 随机擦除
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) # CIFAR-10标准化参数

        # 测试时不使用数据增强
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        # 加载 CIFAR-10 训练集（使用增强变换）
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=not dataset_exists,
                                       transform=train_transform)
        # 加载 CIFAR-10 测试集（不使用增强变换）
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=not dataset_exists,
                                      transform=test_transform)

        # sample training data amongst users (在用户间划分训练数据)
        if args.iid: # 如果是 IID 设置
            # Sample IID user data from Cifar
            user_groups = cifar_iid(train_dataset, args.num_users)
        else: # 如果是 Non-IID 设置
            # Sample Non-IID user data from Cifar
            if args.unequal: # 如果数据划分不均衡
                # Chose uneuqal splits for every user
                raise NotImplementedError() # 此处代码表示不均衡的 CIFAR Non-IID 划分未实现
            else: # 如果数据划分均衡
                # 根据是否提供 alpha 参数选择划分方法
                if hasattr(args, 'alpha') and args.alpha is not None:
                    # 使用 Dirichlet 分布进行 Non-IID 划分
                    from sampling import cifar_noniid_dirichlet
                    user_groups = cifar_noniid_dirichlet(train_dataset, args.num_users, args.alpha)
                    print(f"[INFO] 使用 Dirichlet 分布划分 (alpha={args.alpha})")
                else:
                    # 使用传统的分片方法进行 Non-IID 划分
                    user_groups = cifar_noniid(train_dataset, args.num_users)
                    print("[INFO] 使用传统分片方法划分 (每客户端2个类别)")
                
    elif args.dataset == 'cifar100': # 如果是 CIFAR-100 数据集
        data_dir = '../data/cifar100/' # 数据存储目录
        
        # 检查数据集是否存在
        dataset_exists = check_dataset_exists('cifar100', data_dir)
        if dataset_exists:
            print("[INFO] 找到已存在的 CIFAR-100 数据集")
        else:
            print("[INFO] CIFAR-100 数据集不存在，将自动下载")
        
        # 定义 CIFAR-100 的强化数据增强操作
        apply_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),  # 使用AutoAugment
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),  # 随机擦除
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]) # CIFAR-100标准化参数
        
        # 测试时不用数据增强
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        # 加载 CIFAR-100 训练集
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=not dataset_exists,
                                        transform=apply_transform)
        # 加载 CIFAR-100 测试集
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=not dataset_exists,
                                       transform=test_transform)

        # sample training data amongst users (在用户间划分训练数据)
        if args.iid: # 如果是 IID 设置
            # Sample IID user data from Cifar100
            user_groups = cifar_iid(train_dataset, args.num_users)
        else: # 如果是 Non-IID 设置
            # Sample Non-IID user data from Cifar100
            if args.unequal: # 如果数据划分不均衡
                # Chose uneuqal splits for every user
                raise NotImplementedError() # 此处代码表示不均衡的 CIFAR-100 Non-IID 划分未实现
            else: # 如果数据划分均衡
                # 根据是否提供 alpha 参数选择划分方法
                if hasattr(args, 'alpha') and args.alpha is not None:
                    # 使用 Dirichlet 分布进行 Non-IID 划分 (CIFAR-100 有100个类别)
                    from sampling import cifar_noniid_dirichlet
                    # 为 CIFAR-100 创建专门的函数或修改现有函数
                    user_groups = cifar_noniid_dirichlet(train_dataset, args.num_users, args.alpha)
                    print(f"[INFO] 使用 Dirichlet 分布划分 CIFAR-100 (alpha={args.alpha})")
                else:
                    # 使用传统的分片方法进行 Non-IID 划分
                    user_groups = cifar_noniid(train_dataset, args.num_users)
                    print("[INFO] 使用传统分片方法划分 CIFAR-100")

    elif args.dataset == 'mnist' or args.dataset == 'fmnist': # 如果是 MNIST 或 Fashion-MNIST 数据集
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
            dataset_type = 'mnist'
        else: # args.dataset == 'fmnist'
            data_dir = '../data/fmnist/'
            dataset_type = 'fmnist'

        # 检查数据集是否存在
        dataset_exists = check_dataset_exists(dataset_type, data_dir)
        if dataset_exists:
            print(f"[INFO] 找到已存在的 {dataset_type.upper()} 数据集")
        else:
            print(f"[INFO] {dataset_type.upper()} 数据集不存在，将自动下载")

        # 定义 MNIST/Fashion-MNIST 的图像预处理操作
        # 为Non-IID场景提供更强的数据增强
        if hasattr(args, 'iid') and args.iid == 0 and getattr(args, 'enable_enhanced_augmentation', 1) == 1:  # Non-IID场景且启用增强
            apply_transform = transforms.Compose([
                transforms.ToPILImage() if not isinstance(datasets.MNIST(data_dir, train=True, download=False, transform=transforms.ToTensor())[0][0], torch.Tensor) else transforms.Lambda(lambda x: x),
                transforms.RandomRotation(degrees=10),  # 随机旋转 ±10度
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 的均值和标准差
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.15))  # 随机擦除
            ])
            print(f"🎨 Non-IID {args.dataset.upper()}: 启用增强数据变换")
        else:  # IID场景或关闭增强时使用标准变换
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))]) # MNIST 的均值和标准差 (单通道)
            if hasattr(args, 'iid') and args.iid == 0:
                print(f"🎨 Non-IID {args.dataset.upper()}: 使用标准数据变换（增强已关闭）")

        # 修复：根据数据集类型加载对应的数据
        if args.dataset == 'mnist':
            train_dataset = datasets.MNIST(data_dir, train=True, download=not dataset_exists,
                                           transform=apply_transform)
            test_dataset = datasets.MNIST(data_dir, train=False, download=not dataset_exists,
                                          transform=apply_transform)
        else: # Fashion-MNIST
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=not dataset_exists,
                                                transform=apply_transform)
            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=not dataset_exists,
                                               transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)
    else: # 如果数据集名称无法识别
        exit(f"Error: unrecognized dataset {args.dataset}")


    return train_dataset, test_dataset, user_groups


def average_weights(w,lens): # 计算模型权重的平均值 (FedAvg 算法的核心)
    """
    Returns the average of the weights.
    :param w: 一个列表，其中每个元素是一个客户端的模型权重 (state_dict)
    :return: 平均后的模型权重 (state_dict)
    """
    total = sum(lens)
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w[0][key] * (lens[0] / total)
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * (lens[i] / total)
    return w_avg


import torch
import copy
import math # 确保导入 math 模块

def calculate_diversity_scores(local_weights, client_data_sizes):
    """
    计算客户端模型的多样性分数 - [RFL Aligned Version]
    基于模型权重差异和数据分布不平衡程度
    """
    num_clients = len(local_weights)
    diversity_scores = []

    # 计算每个客户端相对于其他客户端的模型权重差异
    for i in range(num_clients):
        total_distance = 0.0
        weight_count = 0

        for j in range(num_clients):
            if i != j:
                # 计算两个模型权重之间的余弦相似度
                distance = 0.0
                for key in local_weights[i].keys():
                    if key in local_weights[j]:
                        w1 = local_weights[i][key].flatten().float()  # 确保是浮点型
                        w2 = local_weights[j][key].flatten().float()  # 确保是浮点型

                        # 计算余弦距离 (1 - cosine_similarity)
                        norm1 = torch.norm(w1)
                        norm2 = torch.norm(w2)

                        if norm1 > 0 and norm2 > 0:
                            cosine_sim = torch.dot(w1, w2) / (norm1 * norm2)
                            cosine_distance = 1.0 - cosine_sim.item()
                            distance += cosine_distance
                            weight_count += 1

                if weight_count > 0:
                    total_distance += distance / weight_count

        # 归一化距离分数
        avg_distance = total_distance / max(1, num_clients - 1)

        # 结合数据量不平衡因子
        total_samples = sum(client_data_sizes)
        data_imbalance = abs(client_data_sizes[i] / total_samples - 1.0 / num_clients)

        # 综合多样性分数 (权重差异 + 数据不平衡)
        diversity_score = 0.7 * avg_distance + 0.3 * data_imbalance
        diversity_scores.append(diversity_score)

    return diversity_scores

def adaptive_federated_aggregation(local_weights, client_data_sizes, client_losses, 
                               diversity_scores=None, aggregation_method='weighted_avg'):
    """
    自适应联邦聚合策略 - 普通联邦学习版本 [RFL Aligned Version]
    根据数据量、损失和多样性动态调整聚合权重
    """
    if not local_weights:
        return {}

    num_clients = len(local_weights)

    # 计算基础权重（数据量）
    total_samples = sum(client_data_sizes)
    data_weights = [size / total_samples for size in client_data_sizes]

    # 处理NaN或无穷损失值
    safe_losses = []
    for loss in client_losses:
        # Create a tensor to check for isnan or isinf
        loss_tensor = torch.tensor(loss)
        if torch.isnan(loss_tensor) or torch.isinf(loss_tensor):
            # If loss is invalid, append a default high loss value (e.g., 1.0)
            safe_losses.append(1.0)
        else:
            safe_losses.append(float(loss))

    if aggregation_method == 'weighted_avg':
        # 标准加权平均
        weights = data_weights
    elif aggregation_method == 'loss_aware':
        # 基于损失的权重调整
        loss_weights = [1.0 / (1.0 + loss) for loss in safe_losses]
        total_loss_weight = sum(loss_weights)
        if total_loss_weight > 0:
            loss_weights = [w / total_loss_weight for w in loss_weights]
        else:
            loss_weights = [1.0 / num_clients] * num_clients

        # 结合数据量和损失权重
        weights = [0.6 * dw + 0.4 * lw for dw, lw in zip(data_weights, loss_weights)]
    elif aggregation_method == 'diversity_aware' and diversity_scores is not None:
        # 基于多样性的权重调整 (RFL's robust logic)
        # 1. 先计算 loss_weights
        loss_weights = [1.0 / (1.0 + loss) for loss in safe_losses]
        total_loss_weight = sum(loss_weights)
        if total_loss_weight > 0:
            loss_weights = [w / total_loss_weight for w in loss_weights]
        else:
            loss_weights = [1.0 / num_clients] * num_clients

        # 2. 计算 diversity_weights 调整因子
        div_weights = [min(1.5, 1.0 + 0.3 * score) for score in diversity_scores]

        # 3. 三重权重结合 (核心修改)
        loss_aware_weights = [0.6 * dw + 0.4 * lw for dw, lw in zip(data_weights, loss_weights)]
        weights = [law * divw for law, divw in zip(loss_aware_weights, div_weights)]
    else:
        weights = data_weights

    # 归一化权重并确保数值稳定性
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    else:
        weights = [1.0 / num_clients] * num_clients

    # 执行加权聚合
    aggregated_weights = copy.deepcopy(local_weights[0])
    first_weight = local_weights[0]

    for key in first_weight.keys():
        aggregated_weights[key] = torch.zeros_like(first_weight[key])
        for i, weight_dict in enumerate(local_weights):
            aggregated_weights[key] += weights[i] * weight_dict[key]

    return aggregated_weights


def create_personalization_layer(model, client_id, device):
    """
    为Non-IID场景创建轻量级个性化层 - 普通联邦学习版本
    """
    # 获取模型的最后一层特征维度
    if hasattr(model, 'classifier') and hasattr(model.classifier, 'in_features'):
        # 对于有classifier的模型
        feature_dim = model.classifier.in_features
    elif hasattr(model, 'fc') and hasattr(model.fc, 'in_features'):
        # 对于有fc的模型
        feature_dim = model.fc.in_features
    else:
        # 默认维度
        feature_dim = 512
    
    # 创建轻量级个性化层（仅包含少量参数）
    personalization_layer = torch.nn.Sequential(
        torch.nn.Linear(feature_dim, feature_dim // 4),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(feature_dim // 4, feature_dim)
    ).to(device)
    
    return personalization_layer


def apply_personalization(features, personalization_layer, alpha=0.2):
    """
    应用个性化变换 - 普通联邦学习版本
    features: 模型提取的特征
    personalization_layer: 个性化层
    alpha: 个性化程度权重
    """
    if personalization_layer is None:
        return features
    
    # 应用个性化变换
    personalized_features = personalization_layer(features)
    # 混合原始特征和个性化特征
    mixed_features = (1 - alpha) * features + alpha * personalized_features
    
    return mixed_features


'''
 average_weights 实现是简单平均（每个客户端权重相同），而标准的 FedAvg 算法应该是加权平均，每个客户端的权重应与其本地数据量成正比
'''


def exp_details(args): # 打印实验的配置参数
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return