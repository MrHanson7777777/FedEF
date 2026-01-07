#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MNIST数据集专用优化模型
============================

本文件包含针对MNIST手写数字识别数据集（28x28像素，单通道灰度图像，10个类别）
优化设计的深度学习模型，包括：

1. CNN_MNIST - 标准卷积神经网络模型 (推荐)
2. CNN_MNIST_Optimized - 基于ResNet思想的优化CNN模型

所有模型都针对MNIST数据集的特点进行了优化，目标是在保持高精度的同时
降低计算复杂度，适用于联邦学习场景。
s
数据集特点：
- 图像尺寸: 28x28
- 通道数: 1 (灰度图像)
- 类别数: 10 (数字0-9)
- 训练样本: 60,000
- 测试样本: 10,000
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CNN_MNIST(nn.Module):
    """
    CNN_MNIST - 用于MNIST数据集的增强卷积神经网络 (GroupNorm版本)
    ===========================================================
    
    专门为MNIST数据集和Non-IID联邦学习场景优化的卷积神经网络。
    采用现代CNN架构设计，针对联邦学习的特殊需求进行了关键改进。
    
    网络架构：
    - 3个卷积块，每块包含卷积层、GroupNorm、ReLU激活
    - 逐步增加通道数：32 -> 64 -> 128
    - 全连接分类器，带Dropout防止过拟合
    - 针对MNIST 28x28灰度图像优化
    
    关键改进（针对Non-IID联邦学习）：
    1. BatchNorm -> GroupNorm: 不依赖批次大小，在Non-IID环境下更稳定
    2. 5x5 -> 3x3卷积核: 更现代、高效的卷积核设计
    3. 增加网络深度: 3个卷积块提升特征提取能力
    4. 优化的权重初始化: 针对GroupNorm的Kaiming初始化
    
    Args:
        num_classes (int): 分类类别数，默认10（MNIST）
        dropout_rate (float): Dropout比率，默认0.5
        num_groups (int): GroupNorm的组数，默认8
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.5, num_groups=8):
        super(CNN_MNIST, self).__init__()
        
        # 第一个卷积块: 1 -> 32通道
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)      # 保持28x28
        self.gn1 = nn.GroupNorm(num_groups, 32)
        self.pool1 = nn.MaxPool2d(2, 2)                              # 28x28 -> 14x14
        
        # 第二个卷积块: 32 -> 64通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)     # 保持14x14
        self.gn2 = nn.GroupNorm(num_groups, 64)
        self.pool2 = nn.MaxPool2d(2, 2)                              # 14x14 -> 7x7

        # 第三个卷积块: 64 -> 128通道 (新增层，提升特征提取能力)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)    # 保持7x7
        self.gn3 = nn.GroupNorm(num_groups, 128)
        
        # 计算展平后的特征维度
        self.feature_size = 128 * 7 * 7  # 128通道 * 7x7空间维度
        
        # 分类器 (增加神经元数量以匹配更丰富的特征)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.feature_size, 256)                 # 增加到256个神经元
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)                       # 输出层
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化 - 针对GroupNorm优化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层使用Kaiming初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                # GroupNorm层权重初始化为1，偏置初始化为0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 全连接层使用正态分布初始化
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, 1, 28, 28]
        
        Returns:
            torch.Tensor: 输出张量，形状为 [batch_size, num_classes]
        """
        features = self.forward_features(x)
        x = self.classifier(features)
        return x
    
    def forward_features(self, x):
        """
        特征提取前向传播，用于个性化联邦学习
        
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, 1, 28, 28]
        
        Returns:
            torch.Tensor: 特征张量，形状为 [batch_size, 256]
        """
        # 第一个卷积块
        x = self.pool1(F.relu(self.gn1(self.conv1(x))))  # [B, 1, 28, 28] -> [B, 32, 14, 14]
        
        # 第二个卷积块
        x = self.pool2(F.relu(self.gn2(self.conv2(x))))  # [B, 32, 14, 14] -> [B, 64, 7, 7]
        
        # 第三个卷积块（无池化，保持7x7分辨率）
        x = F.relu(self.gn3(self.conv3(x)))               # [B, 64, 7, 7] -> [B, 128, 7, 7]
        
        # 展平特征
        x = x.view(x.size(0), -1)                         # [B, 128, 7, 7] -> [B, 6272]
        
        # 中间特征层
        x = self.dropout1(x)                              # Dropout正规化
        x = F.relu(self.fc1(x))                           # [B, 6272] -> [B, 256]
        
        return x
    
    def classifier(self, features):
        """
        分类器，用于个性化联邦学习
        
        Args:
            features (torch.Tensor): 特征张量，形状为 [batch_size, 256]
        
        Returns:
            torch.Tensor: 输出张量，形状为 [batch_size, num_classes]
        """
        x = self.dropout2(features)                       # Dropout正规化
        x = self.fc2(x)                                   # [B, 256] -> [B, num_classes]
        return x


class ResBlock_MNIST(nn.Module):
    """
    MNIST专用残差块
    ===============
    
    针对MNIST数据集优化的残差连接模块，使用较小的卷积核和适中的通道数。
    残差连接有助于训练更深的网络，提高模型性能。
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 快捷连接：当输入输出维度不同时需要调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)  # 保存输入用于残差连接
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity  # 残差连接
        out = F.relu(out)
        
        return out


class ResBlock_MNIST_GN(nn.Module):
    """
    使用GroupNorm的MNIST专用残差块
    =============================
    
    针对联邦学习Non-IID场景优化的残差连接模块，使用GroupNorm替换BatchNorm。
    GroupNorm不依赖批次统计，在数据分布异构的联邦学习中表现更稳定。
    """
    
    def __init__(self, in_channels, out_channels, stride=1, num_groups=8):
        super(ResBlock_MNIST_GN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        # 确保num_groups可以整除通道数
        groups1 = num_groups if out_channels % num_groups == 0 else max(1, out_channels // 4)
        self.gn1 = nn.GroupNorm(groups1, out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        groups2 = num_groups if out_channels % num_groups == 0 else max(1, out_channels // 4)
        self.gn2 = nn.GroupNorm(groups2, out_channels)
        
        # 快捷连接：当输入输出维度不同时需要调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            groups_shortcut = num_groups if out_channels % num_groups == 0 else max(1, out_channels // 4)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.GroupNorm(groups_shortcut, out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)  # 保存输入用于残差连接
        
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        
        out += identity  # 残差连接
        out = F.relu(out)
        
        return out


class CNN_MNIST_Optimized(nn.Module):
    """
    MNIST数据集优化CNN模型
    ======================
    
    基于ResNet思想设计的MNIST专用CNN模型，目标精度99.5%+。
    使用残差连接和批量归一化，在保持高精度的同时控制模型复杂度。
    
    网络架构：
    - 输入：28x28x1的灰度图像
    - 特征提取：4个残差层，逐步增加通道数
    - 分类：全局平均池化 + 全连接层
    
    优化特点：
    1. 使用残差连接缓解梯度消失问题
    2. 批量归一化加速训练收敛
    3. 自适应池化适应不同输入尺寸
    4. Dropout防止过拟合
    
    适用场景：
    - 联邦学习中的高性能模型
    - 需要高精度的MNIST分类任务
    - 计算资源相对充足的场景
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(CNN_MNIST_Optimized, self).__init__()
        
        # 初始卷积层：1通道 -> 64通道
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 4个残差层，逐步增加特征图深度
        self.layer1 = self._make_layer(64, 64, 2, stride=1)    # 28x28保持
        self.layer2 = self._make_layer(64, 128, 2, stride=2)   # 28x28 -> 14x14
        self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 14x14 -> 7x7
        self.layer4 = self._make_layer(256, 512, 2, stride=2)  # 7x7 -> 4x4
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)
        
        # 权重初始化
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """构建残差层"""
        layers = []
        # 第一个块可能包含下采样
        layers.append(ResBlock_MNIST(in_channels, out_channels, stride))
        # 后续块保持维度
        for _ in range(1, blocks):
            layers.append(ResBlock_MNIST(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        
        # 通过残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 分类
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        # 注意：返回原始logits，避免重复softmax操作
        return x
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def get_mnist_model(model_type='optimized', **kwargs):
    """
    MNIST模型工厂函数
    =================
    
    根据指定类型返回相应的MNIST模型实例。
    
    参数:
        model_type (str): 模型类型
            - 'cnn': 标准卷积神经网络
            - 'optimized': 优化CNN模型（默认，推荐）
        **kwargs: 模型特定参数
    
    返回:
        nn.Module: 对应的模型实例
    """
    if model_type == 'cnn':
        return CNN_MNIST(**kwargs)
    elif model_type == 'optimized':
        return CNN_MNIST_Optimized(**kwargs)
    elif model_type == 'optimized_gn':
        return CNN_MNIST_Optimized_GN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Available: ['cnn', 'optimized', 'optimized_gn']")


# 模型配置预设
MNIST_MODEL_CONFIGS = {
    'cnn_default': {
        'model_type': 'cnn',
        'dropout_rate': 0.5         # 标准Dropout率
    },
    'cnn_light_dropout': {
        'model_type': 'cnn',
        'dropout_rate': 0.3         # 轻量Dropout，适合小数据集
    },
    'cnn_optimized': {
        'model_type': 'optimized',
        'dropout_rate': 0.5
    }
}


if __name__ == "__main__":
    """测试模型实例化和前向传播"""
    import torch
    
    # 创建测试输入
    test_input = torch.randn(4, 1, 28, 28)  # batch_size=4
    
    print("MNIST模型测试")
    print("=" * 50)
    
    # 测试所有模型
    for config_name, config in MNIST_MODEL_CONFIGS.items():
        print(f"\n测试配置: {config_name}")
        model = get_mnist_model(**config)
        
        # 计算参数量
        param_count = sum(p.numel() for p in model.parameters())
        
        # 前向传播测试
        with torch.no_grad():
            output = model(test_input)
        
        print(f"  参数量: {param_count:,}")
        print(f"  输出形状: {output.shape}")
        print(f"  输出范围: [{output.min():.4f}, {output.max():.4f}]")


class CNN_MNIST_Optimized_GN(nn.Module):
    """
    基于GroupNorm的MNIST数据集优化CNN模型
    ===================================
    
    专为联邦学习Non-IID场景优化的MNIST模型，使用GroupNorm替换BatchNorm。
    GroupNorm不依赖批次统计，在数据分布异构的联邦学习中表现更稳定。
    
    网络架构：
    - 输入：28x28x1的灰度图像
    - 特征提取：4个残差层，逐步增加通道数
    - 分类：全局平均池化 + 全连接层
    
    Non-IID优化特点：
    1. 使用GroupNorm替换BatchNorm，避免批次统计偏移
    2. 残差连接缓解梯度消失问题
    3. 自适应池化适应不同输入尺寸
    4. 输出原始logits，避免重复softmax操作
    
    适用场景：
    - Non-IID联邦学习场景
    - 数据分布异构的环境
    - 需要高精度的MNIST分类任务
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.5, num_groups=8):
        super(CNN_MNIST_Optimized_GN, self).__init__()
        self.num_groups = num_groups
        
        # 初始卷积层：1通道 -> 64通道
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(self.num_groups, 64)  # 64通道使用8组
        
        # 4个残差层，逐步增加特征图深度
        self.layer1 = self._make_layer(64, 64, 2, stride=1)    # 28x28保持
        self.layer2 = self._make_layer(64, 128, 2, stride=2)   # 28x28 -> 14x14
        self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 14x14 -> 7x7
        self.layer4 = self._make_layer(256, 512, 2, stride=2)  # 7x7 -> 4x4
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)
        
        # 权重初始化
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """构建残差层"""
        layers = []
        # 第一个块可能包含下采样
        layers.append(ResBlock_MNIST_GN(in_channels, out_channels, stride, num_groups=self.num_groups))
        # 后续块保持维度
        for _ in range(1, blocks):
            layers.append(ResBlock_MNIST_GN(out_channels, out_channels, 1, num_groups=self.num_groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始特征提取
        x = F.relu(self.gn1(self.conv1(x)))
        
        # 通过残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 分类
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        # 注意：返回原始logits，不使用log_softmax
        return x
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):  # 修改为GroupNorm
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)