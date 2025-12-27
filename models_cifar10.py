#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CIFAR-10数据集专用优化模型
==========================

本文件包含针对CIFAR-10自然图像分类数据集（32x32像素，3通道彩色图像，10个类别）
优化设计的深度学习模型，包括：

1. CNN_CIFAR10 - 多层卷积神经网络，具有BatchNorm和Dropout正规化
2. ResNet18联邦学习版本 - 使用GroupNorm替代BatchNorm，适用于联邦学习场景
3. EfficientNet风格模型 - 使用MobileNet倒置残差块

所有模型都针对CIFAR-10数据集的特点进行了优化，在联邦学习场景下
具有良好的收敛性和泛化能力。

数据集特点：
- 图像尺寸: 32x32
- 通道数: 3 (RGB彩色图像)
- 类别数: 10 (飞机、汽车、鸟类、猫、鹿、狗、青蛙、马、船、卡车)
- 训练样本: 50,000
- 测试样本: 10,000
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CNNCifar(nn.Module):
    """
    CNN_CIFAR10 - 用于CIFAR-10数据集的标准卷积神经网络
    =================================================
    
    专门为CIFAR-10数据集设计的多层卷积神经网络，具有良好的性能和适中的复杂度。
    采用现代深度学习最佳实践，包括批标准化、Dropout正规化等技术。
    
    网络架构：
    - 3个卷积块，每块包含卷积层、BatchNorm、ReLU、MaxPooling
    - 逐步增加通道数：32 -> 64 -> 128
    - 自适应平均池化减少空间维度
    - 全连接分类器，带Dropout防止过拟合
    
    特点：
    1. 适度的模型复杂度，平衡性能和计算效率
    2. BatchNorm加速收敛和提高稳定性
    3. Dropout防止过拟合
    4. 自适应池化增强模型鲁棒性
    
    Args:
        num_classes (int): 分类类别数，默认10（CIFAR-10）
        dropout_rate (float): Dropout比率，默认0.5
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(CNNCifar, self).__init__()
        
        # 第一个卷积块：3 -> 32通道
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)     # 保持32x32
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)                             # 32x32 -> 16x16
        
        # 第二个卷积块：32 -> 64通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)    # 保持16x16
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)                             # 16x16 -> 8x8
        
        # 第三个卷积块：64 -> 128通道
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)   # 保持8x8
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)                             # 8x8 -> 4x4
        
        # 自适应平均池化：4x4 -> 1x1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128, 128)                              # 特征映射层
        self.fc2 = nn.Linear(128, num_classes)                      # 输出层
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层使用Kaiming初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm层权重初始化为1，偏置初始化为0
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
            x (torch.Tensor): 输入张量，形状为 [batch_size, 3, 32, 32]
        
        Returns:
            torch.Tensor: 输出张量，形状为 [batch_size, num_classes]
        """
        # 第一个卷积块
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # [B, 3, 32, 32] -> [B, 32, 16, 16]
        
        # 第二个卷积块
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # [B, 32, 16, 16] -> [B, 64, 8, 8]
        
        # 第三个卷积块
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # [B, 64, 8, 8] -> [B, 128, 4, 4]
        
        # 自适应平均池化
        x = self.adaptive_pool(x)                         # [B, 128, 4, 4] -> [B, 128, 1, 1]
        
        # 展平特征
        x = x.view(x.size(0), -1)                         # [B, 128, 1, 1] -> [B, 128]
        
        # 分类器
        x = self.dropout(x)                               # Dropout正规化
        x = F.relu(self.fc1(x))                           # [B, 128] -> [B, 128]
        x = self.dropout(x)                               # Dropout正规化
        x = self.fc2(x)                                   # [B, 128] -> [B, num_classes]
        
        return x                                                      


class BasicBlock_CIFAR10(nn.Module):
    """
    CIFAR-10专用ResNet基础块
    ========================
    
    针对32x32图像优化的ResNet基础残差块。
    使用较小的卷积核和适中的通道数，适合CIFAR-10的图像尺寸。
    
    Args:
        in_planes (int): 输入通道数
        planes (int): 输出通道数
        stride (int): 卷积步长，默认为1
    """
    expansion = 1  # 基础块的扩展因子，始终为1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_CIFAR10, self).__init__()
        
        # 第一个3x3卷积层，可能包含下采样
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 第二个3x3卷积层，保持特征图尺寸
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # 短路连接：当输入输出维度不匹配时，使用1x1卷积调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        """前向传播"""
        # 第一个卷积-BN-ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        # 第二个卷积-BN
        out = self.bn2(self.conv2(out))
        # 添加残差连接
        out += self.shortcut(x)
        # 最终ReLU激活
        out = F.relu(out)
        return out


class ResNet18_CIFAR10_Fed(nn.Module):
    """
    CIFAR-10专用联邦学习ResNet18模型
    ================================
    
    专门为联邦学习场景设计的ResNet18变体，针对CIFAR-10数据集优化。
    使用GroupNorm替代BatchNorm，提高在非独立同分布数据上的性能。
    
    网络架构：
    - 输入：32x32x3的RGB图像
    - 4个残差层：[2, 2, 2, 2]个基础块
    - 输出：10个类别的概率分布
    
    联邦学习优化：
    1. 使用GroupNorm替代BatchNorm
    2. 适当的Dropout防止过拟合
    3. 自适应池化减少参数敏感性
    
    Args:
        num_classes (int): 分类类别数，默认10
        use_groupnorm (bool): 是否使用GroupNorm替代BatchNorm
        num_groups (int): GroupNorm的组数，默认8
    """
    
    def __init__(self, num_classes=10, use_groupnorm=True, num_groups=8):
        super(ResNet18_CIFAR10_Fed, self).__init__()
        self.in_planes = 64  # 当前通道数追踪
        self.use_groupnorm = use_groupnorm
        self.num_groups = num_groups
        
        # 初始卷积层：3x3卷积，不降采样以保持CIFAR-10的小尺寸
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 4个残差层，通道数依次为[64, 128, 256, 512]
        self.layer1 = self._make_layer(64, 2, stride=1)   # 32x32 -> 32x32
        self.layer2 = self._make_layer(128, 2, stride=2)  # 32x32 -> 16x16
        self.layer3 = self._make_layer(256, 2, stride=2)  # 16x16 -> 8x8
        self.layer4 = self._make_layer(512, 2, stride=2)  # 8x8 -> 4x4
        
        # 分类头：自适应平均池化 + 全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 4x4 -> 1x1
        self.fc = nn.Linear(512, num_classes)
        
        # 如果启用GroupNorm，替换所有BatchNorm
        if use_groupnorm:
            self._replace_bn_with_gn()
        
        # 初始化模型权重
        self._initialize_weights()

    def _make_layer(self, planes, blocks, stride):
        """
        构建残差层
        
        Args:
            planes (int): 该层的输出通道数
            blocks (int): 该层包含的基础块数量
            stride (int): 第一个块的步长
        """
        strides = [stride] + [1] * (blocks - 1)  # 只有第一个块可能有下采样
        layers = []
        for s in strides:
            layers.append(BasicBlock_CIFAR10(self.in_planes, planes, s))
            self.in_planes = planes  # 更新当前通道数
        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播"""
        # 初始特征提取
        out = F.relu(self.bn1(self.conv1(x)))  # [B, 3, 32, 32] -> [B, 64, 32, 32]
        
        # 四个残差层
        out = self.layer1(out)  # [B, 64, 32, 32] -> [B, 64, 32, 32]
        out = self.layer2(out)  # [B, 64, 32, 32] -> [B, 128, 16, 16]
        out = self.layer3(out)  # [B, 128, 16, 16] -> [B, 256, 8, 8]
        out = self.layer4(out)  # [B, 256, 8, 8] -> [B, 512, 4, 4]
        
        # 全局平均池化和分类
        out = self.avgpool(out)     # [B, 512, 4, 4] -> [B, 512, 1, 1]
        out = torch.flatten(out, 1) # [B, 512, 1, 1] -> [B, 512]
        out = self.fc(out)          # [B, 512] -> [B, num_classes]
        
        # 返回对数概率用于NLLLoss
        return F.log_softmax(out, dim=1)
    
    def _replace_bn_with_gn(self):
        """
        将BatchNorm替换为GroupNorm，提高联邦学习性能
        
        GroupNorm不依赖批次统计，更适合联邦学习中的非IID数据分布
        """
        def replace_bn(module):
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    num_channels = child.num_features
                    # 确保组数不超过通道数
                    groups = min(self.num_groups, num_channels)
                    gn = nn.GroupNorm(groups, num_channels)
                    setattr(module, name, gn)
                else:
                    # 递归处理子模块
                    replace_bn(child)
        
        replace_bn(self)
    
    def _initialize_weights(self):
        """
        权重初始化
        
        使用He初始化用于ReLU网络，提高训练稳定性
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He初始化，适用于ReLU激活函数
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # BN/GN参数初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 全连接层小随机初始化
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class MBConvBlock_CIFAR10(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, se_ratio=0.5, use_groupnorm=True, num_groups=8):
        super(MBConvBlock_CIFAR10, self).__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        if expand_ratio != 1:
            expanded_channels = in_channels * expand_ratio
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.GroupNorm(num_groups, expanded_channels),
                nn.ReLU(inplace=True)
            )
            self.has_expansion = True
        else:
            expanded_channels = in_channels
            self.expand_conv = nn.Identity()
            self.has_expansion = False
        self.expanded_channels = expanded_channels
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, 3, stride, 1, groups=expanded_channels, bias=False),
            nn.GroupNorm(num_groups, expanded_channels),
            nn.ReLU(inplace=True)
        )
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(expanded_channels, se_channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(se_channels, expanded_channels, 1),
                nn.Sigmoid()
            )
        else:
            self.se = nn.Identity()
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.GroupNorm(num_groups, out_channels)
        )
    def forward(self, x):
        identity = x
        if self.has_expansion:
            x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        se_weight = self.se(x)
        x = x * se_weight
        x = self.project_conv(x)
        if self.use_residual:
            x = x + identity
        return x

class EfficientNet_CIFAR10(nn.Module):
    def _initialize_weights(self):
        """
        权重初始化，适用于GroupNorm和卷积/全连接层
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def __init__(self, num_classes=10, num_groups=8):
        super(EfficientNet_CIFAR10, self).__init__()
        # 改进的stem层，增加初始通道数
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1, bias=False),  # 增加到48通道
            nn.GroupNorm(num_groups, 48),
            nn.ReLU(inplace=True)
        )
        # 减少stage数量和通道数，优化针对CIFAR-10的架构
        self.stage1 = self._make_stage(48, 32, 2, 1, expand_ratio=1, num_groups=num_groups)  # stride=1
        self.stage2 = self._make_stage(32, 48, 2, 2, expand_ratio=4, num_groups=num_groups)  # stride=2, 32x32->16x16
        self.stage3 = self._make_stage(48, 64, 3, 2, expand_ratio=4, num_groups=num_groups)  # stride=2, 16x16->8x8
        self.stage4 = self._make_stage(64, 96, 2, 1, expand_ratio=4, num_groups=num_groups)  # stride=1, 保持8x8
        self.stage5 = self._make_stage(96, 128, 2, 1, expand_ratio=4, num_groups=num_groups)  # stride=1, 保持8x8
        
        # 简化head部分
        self.head_conv = nn.Conv2d(128, 256, kernel_size=1, bias=False)  # 减少到256通道
        self.head_gn = nn.GroupNorm(num_groups, 256)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 调整Dropout率
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),  # 增加Dropout率到0.4
            nn.Linear(256, num_classes)
        )
        self._initialize_weights()
    def _make_stage(self, in_channels, out_channels, num_blocks, stride, expand_ratio, num_groups):
        layers = []
        layers.append(MBConvBlock_CIFAR10(in_channels, out_channels, stride, expand_ratio, se_ratio=0.5, use_groupnorm=True, num_groups=num_groups))
        for _ in range(1, num_blocks):
            layers.append(MBConvBlock_CIFAR10(out_channels, out_channels, 1, expand_ratio, se_ratio=0.5, use_groupnorm=True, num_groups=num_groups))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        # 移除stage6和stage7，简化模型
        x = F.relu(self.head_gn(self.head_conv(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


def replace_bn_with_gn(model, num_groups=8):
    """
    将模型中的BatchNorm2d替换为GroupNorm
    ====================================
    
    在联邦学习场景中，GroupNorm通常比BatchNorm表现更好，
    因为它不依赖于批次统计信息，更适合非独立同分布的数据。
    
    GroupNorm将通道分组进行归一化，不受批次大小影响，
    在联邦学习的小批次训练中更加稳定。
    
    参数:
        model (nn.Module): 要修改的模型
        num_groups (int): GroupNorm的组数，默认8
    
    返回:
        nn.Module: 修改后的模型
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_channels = module.num_features
            # 确保组数不超过通道数，避免分组错误
            if num_channels < num_groups:
                groups = num_channels
            else:
                groups = num_groups
            # 创建对应的GroupNorm层
            gn = nn.GroupNorm(groups, num_channels)
            setattr(model, name, gn)
        else:
            # 递归处理子模块
            replace_bn_with_gn(module, num_groups)
    return model


def get_cifar10_model(model_type='resnet18_fed', **kwargs):
    """
    CIFAR-10模型工厂函数
    ====================
    
    根据指定类型返回相应的CIFAR-10模型实例。
    提供统一的模型创建接口，便于实验和部署。
    
    参数:
        model_type (str): 模型类型
            - 'resnet18_fed': ResNet18联邦学习版本（默认）
            - 'efficientnet': EfficientNet风格模型
        **kwargs: 模型特定参数，会传递给对应的模型构造函数
    
    返回:
        nn.Module: 对应的模型实例
    
    异常:
        ValueError: 当指定的模型类型不存在时抛出
    """
    if model_type == 'resnet18_fed':
        return ResNet18_CIFAR10_Fed(**kwargs)
    elif model_type == 'efficientnet':
        return EfficientNet_CIFAR10(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available types: ['resnet18_fed', 'efficientnet']")


# 模型配置预设
# ============
# 预定义的模型配置，便于快速实验和部署
CIFAR10_MODEL_CONFIGS = {
    'resnet18_fed_default': {
        'model_type': 'resnet18_fed',
        'use_groupnorm': True,      # 使用GroupNorm，适合联邦学习
        'num_groups': 8             # GroupNorm分组数
    },
    'resnet18_fed_bn': {
        'model_type': 'resnet18_fed',
        'use_groupnorm': False      # 使用BatchNorm，适合传统训练
    },
    'efficientnet_default': {
        'model_type': 'efficientnet'
    }
}


if __name__ == "__main__":
    """
    模型测试和验证
    ==============
    
    测试所有模型的实例化和前向传播，验证模型正确性
    """
    import torch
    
    # 创建测试输入：模拟CIFAR-10的批次数据
    test_input = torch.randn(4, 3, 32, 32)  # batch_size=4, channels=3, height=32, width=32
    
    print("CIFAR-10模型测试")
    print("=" * 50)
    
    # 测试所有预定义的模型配置
    for config_name, config in CIFAR10_MODEL_CONFIGS.items():
        print(f"\n测试配置: {config_name}")
        print("-" * 30)
        
        # 创建模型实例
        model = get_cifar10_model(**config)
        
        # 计算模型参数量
        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 前向传播测试
        model.eval()  # 设置为评估模式
        with torch.no_grad():
            output = model(test_input)
        
        # 打印模型信息
        print(f"  模型类型: {config['model_type']}")
        print(f"  总参数量: {param_count:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  输出形状: {output.shape}")
        print(f"  输出范围: [{output.min():.4f}, {output.max():.4f}]")
        
        # 验证输出形状正确性
        assert output.shape == (4, 10), f"输出形状错误: {output.shape}"
        print("  ✓ 前向传播测试通过")
    
    # 测试GroupNorm替换功能
    print(f"\n测试GroupNorm替换功能")
    print("-" * 30)
    
    # 创建使用BatchNorm的模型
    model_bn = ResNet18_CIFAR10_Fed(use_groupnorm=False)
    print(f"  替换前BatchNorm层数: {len([m for m in model_bn.modules() if isinstance(m, nn.BatchNorm2d)])}")
    
    # 替换为GroupNorm
    model_gn = replace_bn_with_gn(model_bn, num_groups=8)
    bn_count = len([m for m in model_gn.modules() if isinstance(m, nn.BatchNorm2d)])
    gn_count = len([m for m in model_gn.modules() if isinstance(m, nn.GroupNorm)])
    
    print(f"  替换后BatchNorm层数: {bn_count}")
    print(f"  替换后GroupNorm层数: {gn_count}")
    print("  ✓ GroupNorm替换功能测试完成")
    
    print(f"\n所有测试完成！")