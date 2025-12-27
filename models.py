#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

"""
联邦学习模型集合 - 向后兼容版本
========================================

本文件保持与原有代码的兼容性，同时提供新的优化模型访问方式。

推荐使用新的专用模型文件：
- models_mnist.py: MNIST数据集专用模型
- models_cifar10.py: CIFAR-10数据集专用模型  
- models_cifar100.py: CIFAR-100数据集专用模型
- model_factory.py: 统一模型工厂接口

新的使用方式：
```python
from model_factory import get_model

# 获取MNIST优化模型
model = get_model('mnist', 'optimized')

# 获取CIFAR-10联邦学习ResNet18
model = get_model('cifar10', 'resnet18_fed', use_groupnorm=True)
```
"""

from torch import nn # 从 PyTorch 中导入 nn 模块，它是构建神经网络的核心
import torch.nn.functional as F # 导入 PyTorch 中的函数式 API，包含激活函数、池化等
import torch
import math

# 尝试导入新的专用模型（可选）
try:
    from model_factory import get_model, list_available_models
    HAS_NEW_MODELS = True
except ImportError:
    HAS_NEW_MODELS = False

class MLP(nn.Module): # 定义一个多层感知机模型，继承自 nn.Module
    def __init__(self, dim_in, dim_hidden, dim_out): # 构造函数
        super(MLP, self).__init__() # 调用父类的构造函数
        self.layer_input = nn.Linear(dim_in, dim_hidden) # 定义输入层到隐藏层的全连接层
        self.relu = nn.ReLU() # 定义 ReLU 激活函数
        self.dropout = nn.Dropout() # 定义 Dropout 层，用于防止过拟合
        self.layer_hidden = nn.Linear(dim_hidden, dim_out) # 定义隐藏层到输出层的全连接层
        self.log_softmax = nn.LogSoftmax(dim=1) # 定义 LogSoftmax 层，用于输出概率分布，dim=1 表示对每一行进行 softmax 操作

    def forward(self, x): # 定义模型的前向传播逻辑
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1]) # 将输入 x 展平 (flatten)
                                                          # -1 表示该维度大小由其他维度和总元素数自动推断
                                                          # x.shape[1]*x.shape[-2]*x.shape[-1] 计算图像的总像素数
        x = self.layer_input(x) # 通过输入层
        x = self.dropout(x) # 应用 Dropout
        x = self.relu(x) # 应用 ReLU 激活
        x = self.layer_hidden(x) # 通过隐藏层
        return self.log_softmax(x) # 返回 Softmax 输出


# ResNet基础模块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, input_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out, dim=1)


# 联邦学习专用ResNet18模型
class ResNet18Fed(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18Fed, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride):
        strides = [stride] + [1]*(blocks-1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)


# BatchNorm替换为GroupNorm的函数（适用于联邦学习）
def replace_bn_with_gn(model, num_groups=8):
    """将模型中的BatchNorm2d替换为GroupNorm，提高联邦学习性能"""
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_channels = module.num_features
            if num_channels < num_groups:
                groups = num_channels
            else:
                groups = num_groups
            gn = nn.GroupNorm(groups, num_channels)
            setattr(model, name, gn)
        else:
            replace_bn_with_gn(module, num_groups)
    return model


class CNNMnist(nn.Module): # 基于ResNet思想的MNIST模型 - 目标准确率99.5%+
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        # 使用ResNet-18的简化版本，适配28x28的MNIST图像
        self.conv1 = nn.Conv2d(args.num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet blocks - 使用残差连接
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # 分类层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, args.num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        # 第一个block可能需要下采样
        layers.append(ResBlock(in_channels, out_channels, stride))
        # 后续block保持尺寸
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)


class ResBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 快捷连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity
        out = F.relu(out)
        
        return out


class CNNFashion_Mnist(nn.Module): # 改进的 Fashion-MNIST CNN 模型
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        # 使用更深的网络架构，类似EfficientNet的思想
        # 初始卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # 使用深度可分离卷积减少参数量
        self.layer1 = self._make_layer(32, 64, 2, stride=1)    # 28x28 -> 28x28
        self.layer2 = self._make_layer(64, 128, 2, stride=2)   # 28x28 -> 14x14
        self.layer3 = self._make_layer(128, 256, 3, stride=2)  # 14x14 -> 7x7
        self.layer4 = self._make_layer(256, 512, 3, stride=2)  # 7x7 -> 4x4 (Fashion-MNIST稍微复杂)
        
        # Squeeze-and-Excitation注意力机制
        self.se = SEBlock(512)
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.se(x)  # 注意力机制
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=1)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),  # 降维
            nn.ReLU(inplace=True),  # 激活
            nn.Linear(channels // reduction, channels, bias=False),  # 升维
            nn.Sigmoid()  # 权重限制在 [0, 1]
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze：全局平均池化
        y = self.avg_pool(x).view(b, c)
        # Excitation：生成权重
        y = self.fc(y).view(b, c, 1, 1)
        # Scale：加权
        return x * y.expand_as(x)

class CNNCifar(nn.Module): # 改进的 CIFAR-10 CNN 模型 - 基于EfficientNet思想
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        # 基于EfficientNet-B0的简化版本，专门为CIFAR-10优化
        
        # Stem层
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # EfficientNet风格的MBConv块
        # Stage 1: 32x32 -> 32x32
        self.stage1 = self._make_stage(32, 16, 1, 1, expand_ratio=1)
        # Stage 2: 32x32 -> 16x16  
        self.stage2 = self._make_stage(16, 24, 2, 2, expand_ratio=6)
        # Stage 3: 16x16 -> 8x8
        self.stage3 = self._make_stage(24, 40, 2, 2, expand_ratio=6)
        # Stage 4: 8x8 -> 4x4
        self.stage4 = self._make_stage(40, 80, 3, 2, expand_ratio=6)
        # Stage 5: 4x4 -> 4x4
        self.stage5 = self._make_stage(80, 112, 3, 1, expand_ratio=6)
        # Stage 6: 4x4 -> 2x2
        self.stage6 = self._make_stage(112, 192, 4, 2, expand_ratio=6)
        # Stage 7: 2x2 -> 2x2
        self.stage7 = self._make_stage(192, 320, 1, 1, expand_ratio=6)
        
        # Head层
        self.head_conv = nn.Conv2d(320, 1280, kernel_size=1, bias=False)
        self.head_bn = nn.BatchNorm2d(1280)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, args.num_classes)
        )
        
    def _make_stage(self, in_channels, out_channels, num_blocks, stride, expand_ratio):
        layers = []
        layers.append(MBConvBlock(in_channels, out_channels, stride, expand_ratio))
        for _ in range(1, num_blocks):
            layers.append(MBConvBlock(out_channels, out_channels, 1, expand_ratio))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        
        x = F.relu(self.head_bn(self.head_conv(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=1)


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution Block"""
    def __init__(self, in_channels, out_channels, stride, expand_ratio, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Expansion
        if expand_ratio != 1:
            expanded_channels = in_channels * expand_ratio
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.ReLU(inplace=True)
            )
            self.has_expansion = True
        else:
            expanded_channels = in_channels  # 修复：当没有扩展时，使用输入通道数
            self.expand_conv = nn.Identity()
            self.has_expansion = False
            
        self.expanded_channels = expanded_channels  # 保存为实例变量
            
        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, 3, stride, 1, 
                     groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU(inplace=True)
        )
        
        # Squeeze-and-Excitation
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
            
        # Pointwise convolution
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        identity = x
        
        # Expansion
        if self.has_expansion:
            x = self.expand_conv(x)
        
        # Depthwise
        x = self.depthwise_conv(x)
        
        # SE
        if hasattr(self.se, 'weight'):
            se_weight = self.se(x)
            x = x * se_weight
            
        # Projection
        x = self.project_conv(x)
        
        # Residual connection
        if self.use_residual:
            x = x + identity
            
        return x

# 注意：下面的 modelC 类定义中 super(AllConvNet, self).__init__() 似乎是一个笔误，
# 应该是 super(modelC, self).__init__()。
# 并且 AllConvNet 这个名字也没有在类名中使用。
class modelC(nn.Module): # 定义一个更复杂的 CNN 模型 (可能参考了 AllConvNet)
    def __init__(self, input_size, n_classes=10, **kwargs): # input_size 应该是输入通道数
        super(modelC, self).__init__() # 修正：应该是 super(modelC, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2) # stride=2 实现下采样
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1) # 1x1 卷积

        self.class_conv = nn.Conv2d(192, n_classes, 1) # 使用 1x1 卷积作为分类层

    def forward(self, x):
        x_drop = F.dropout(x, .2) # 输入层 Dropout
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5) # 中间层 Dropout
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5) # 中间层 Dropout
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out)) # 分类卷积层
        pool_out = F.adaptive_avg_pool2d(class_out, 1) # 自适应平均池化到 1x1
        pool_out.squeeze_(-1) # 移除大小为 1 的维度
        pool_out.squeeze_(-1) # 再次移除大小为 1 的维度，得到 (batch_size, n_classes) 的形状
        return pool_out # 返回 logits


# 为CIFAR-100专门设计的深度模型 - 基于EfficientNet-B3
class CNNCifar100(nn.Module):
    def __init__(self, args):
        super(CNNCifar100, self).__init__()
        # 更深的EfficientNet架构，专门为CIFAR-100的100个类别设计
        
        # Stem层
        self.stem = nn.Sequential(
            nn.Conv2d(3, 40, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True)
        )
        
        # 更深的EfficientNet风格架构
        # Stage 1: 32x32 -> 32x32
        self.stage1 = self._make_stage(40, 24, 1, 1, expand_ratio=1)
        # Stage 2: 32x32 -> 16x16  
        self.stage2 = self._make_stage(24, 32, 2, 2, expand_ratio=6)
        # Stage 3: 16x16 -> 8x8
        self.stage3 = self._make_stage(32, 48, 3, 2, expand_ratio=6)
        # Stage 4: 8x8 -> 4x4
        self.stage4 = self._make_stage(48, 96, 4, 2, expand_ratio=6)
        # Stage 5: 4x4 -> 4x4
        self.stage5 = self._make_stage(96, 136, 4, 1, expand_ratio=6)
        # Stage 6: 4x4 -> 2x2
        self.stage6 = self._make_stage(136, 232, 5, 2, expand_ratio=6)
        # Stage 7: 2x2 -> 2x2  
        self.stage7 = self._make_stage(232, 384, 2, 1, expand_ratio=6)
        
        # Head层 - 为100个类别优化
        self.head_conv = nn.Conv2d(384, 1536, kernel_size=1, bias=False)
        self.head_bn = nn.BatchNorm2d(1536)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # 更复杂的分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1536, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 100)  # CIFAR-100 has 100 classes
        )
        
    def _make_stage(self, in_channels, out_channels, num_blocks, stride, expand_ratio):
        layers = []
        layers.append(MBConvBlock(in_channels, out_channels, stride, expand_ratio))
        for _ in range(1, num_blocks):
            layers.append(MBConvBlock(out_channels, out_channels, 1, expand_ratio))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        
        x = F.relu(self.head_bn(self.head_conv(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=1)