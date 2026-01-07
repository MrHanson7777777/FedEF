#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
联邦学习优化模型统一入口
========================

本文件提供了所有数据集专用模型的统一访问接口，包括：

1. MNIST数据集模型 (models_mnist.py)
   - CNN_MNIST: 标准卷积神经网络 (推荐)
   - CNN_MNIST_Optimized: 优化CNN模型
   - CNN_MNIST_Optimized_GN: GroupNorm优化CNN模型 (Non-IID联邦学习推荐)

2. CIFAR-10数据集模型 (models_cifar10.py)
   - CNNCifar: 标准卷积神经网络 (推荐)
   - ResNet18_CIFAR10_Fed: 联邦学习ResNet18
   - EfficientNet_CIFAR10: EfficientNet风格模型

3. CIFAR-100数据集模型 (models_cifar100.py)
   - ResNet18_CIFAR100_Fed: 联邦学习ResNet18
   - EfficientNet_CIFAR100: EfficientNet-B3风格模型
   - DenseNet_CIFAR100: DenseNet模型

使用方法：
```python
from model_factory import get_model, list_available_models

# 获取MNIST标准CNN模型
model = get_model('mnist', 'cnn', dropout_rate=0.3)

# 获取MNIST优化CNN模型
model = get_model('mnist', 'optimized')

# 获取MNIST GroupNorm优化CNN模型 (Non-IID场景推荐)
model = get_model('mnist', 'optimized_gn', num_groups=8)

# 获取CIFAR-10标准CNN模型
model = get_model('cifar10', 'cnn', dropout_rate=0.3)

# 获取CIFAR-10联邦学习ResNet18
model = get_model('cifar10', 'resnet18_fed', use_groupnorm=True)

# 列出所有可用模型
list_available_models()
```
"""

import torch
import torch.nn as nn

# 尝试使用相对导入方式导入各数据集的模型模块
try:
    from .models_mnist import get_mnist_model, MNIST_MODEL_CONFIGS
    from .models_cifar10 import get_cifar10_model, CIFAR10_MODEL_CONFIGS, replace_bn_with_gn
    from .models_cifar100 import get_cifar100_model, CIFAR100_MODEL_CONFIGS
except ImportError:
    # 如果相对导入失败，尝试使用绝对导入（处理直接运行或不同导入路径的情况）
    try:
        from models_mnist import get_mnist_model, MNIST_MODEL_CONFIGS
        from models_cifar10 import get_cifar10_model, CIFAR10_MODEL_CONFIGS, replace_bn_with_gn
        from models_cifar100 import get_cifar100_model, CIFAR100_MODEL_CONFIGS
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保模型文件在正确的路径下")


def get_model(dataset, model_type, **kwargs):
    """
    统一模型获取接口
    ================
    
    根据数据集和模型类型返回相应的模型实例。
    这是模型工厂的核心功能，提供统一的模型访问接口。
    
    参数:
        dataset (str): 数据集名称
            - 'mnist': MNIST手写数字识别数据集
            - 'cifar10': CIFAR-10自然图像分类数据集（10个类别）
            - 'cifar100': CIFAR-100自然图像分类数据集（100个类别）
        model_type (str): 模型类型（取决于具体数据集，每个数据集支持不同的模型）
        **kwargs: 模型特定参数，会传递给对应的模型构造函数
    
    返回:
        nn.Module: 对应的PyTorch模型实例
    
    抛出异常:
        ValueError: 当指定的数据集不被支持时
    
    示例:
        >>> # 获取MNIST数据集的优化CNN模型，设置dropout率为0.5
        >>> model = get_model('mnist', 'optimized', dropout_rate=0.5)
        >>> # 获取CIFAR-10数据集的ResNet18联邦学习版本，使用GroupNorm替代BatchNorm
        >>> model = get_model('cifar10', 'resnet18_fed', use_groupnorm=True)
        >>> # 获取CIFAR-100数据集的EfficientNet模型，设置dropout率为0.3
        >>> model = get_model('cifar100', 'efficientnet', dropout_rate=0.3)
    """
    # 将数据集名称转换为小写，确保大小写不敏感
    dataset = dataset.lower()
    
    # 根据数据集类型调用对应的模型获取函数
    if dataset == 'mnist':
        return get_mnist_model(model_type, **kwargs)
    elif dataset in ['cifar', 'cifar10']:  # 支持 cifar 和 cifar10 两种写法
        return get_cifar10_model(model_type, **kwargs)
    elif dataset == 'cifar100':
        return get_cifar100_model(model_type, **kwargs)
    else:
        # 如果数据集不被支持，抛出异常并提示支持的数据集列表
        raise ValueError(f"不支持的数据集: {dataset}. 支持的数据集: ['mnist', 'cifar', 'cifar10', 'cifar100']")


def get_model_by_config(dataset, config_name):
    """
    根据预设配置获取模型
    ====================
    
    使用预定义的配置快速获取模型实例。
    这种方式简化了模型创建过程，用户只需指定配置名称即可获得经过调优的模型。
    
    参数:
        dataset (str): 数据集名称
        config_name (str): 预设配置名称，每个数据集都有相应的配置选项
    
    返回:
        nn.Module: 对应的模型实例
    
    抛出异常:
        ValueError: 当指定的数据集或配置不被支持时
    """
    # 将数据集名称转换为小写
    dataset = dataset.lower()
    
    # 根据数据集类型获取对应的预设配置
    if dataset == 'mnist':
        # 检查配置是否存在于MNIST配置字典中
        if config_name not in MNIST_MODEL_CONFIGS:
            raise ValueError(f"不支持的MNIST配置: {config_name}")
        # 获取配置参数并创建模型
        config = MNIST_MODEL_CONFIGS[config_name]
        return get_mnist_model(**config)
    elif dataset == 'cifar10':
        # 检查配置是否存在于CIFAR-10配置字典中
        if config_name not in CIFAR10_MODEL_CONFIGS:
            raise ValueError(f"不支持的CIFAR-10配置: {config_name}")
        # 获取配置参数并创建模型
        config = CIFAR10_MODEL_CONFIGS[config_name]
        return get_cifar10_model(**config)
    elif dataset == 'cifar100':
        # 检查配置是否存在于CIFAR-100配置字典中
        if config_name not in CIFAR100_MODEL_CONFIGS:
            raise ValueError(f"不支持的CIFAR-100配置: {config_name}")
        # 获取配置参数并创建模型
        config = CIFAR100_MODEL_CONFIGS[config_name]
        return get_cifar100_model(**config)
    else:
        # 如果数据集不被支持，抛出异常
        raise ValueError(f"不支持的数据集: {dataset}")


def list_available_models():
    """
    列出所有可用的模型和配置
    ========================
    
    打印所有数据集的可用模型类型和预设配置。
    这个函数帮助用户了解当前系统支持的所有模型选项。
    """
    print("可用模型概览")
    print("=" * 80)
    
    # 显示MNIST数据集相关模型信息
    print("\n📊 MNIST数据集模型:")
    print("   模型类型:")
    print("   - 'cnn': 标准卷积神经网络（推荐用于一般任务）")
    print("   - 'optimized': 优化CNN模型（ResNet风格，性能更好）")
    print("   预设配置:")
    # 遍历并显示所有MNIST预设配置
    for config in MNIST_MODEL_CONFIGS.keys():
        print(f"   - {config}")
    
    # 显示CIFAR-10数据集相关模型信息
    print("\n🌅 CIFAR-10数据集模型:")
    print("   模型类型:")
    print("   - 'cnn': 标准卷积神经网络（推荐用于一般任务）")
    print("   - 'resnet18_fed': ResNet18联邦学习版本（适合分布式训练）")
    print("   - 'efficientnet': EfficientNet风格模型（高效且准确）")
    print("   预设配置:")
    # 遍历并显示所有CIFAR-10预设配置
    for config in CIFAR10_MODEL_CONFIGS.keys():
        print(f"   - {config}")
    
    # 显示CIFAR-100数据集相关模型信息
    print("\n🎯 CIFAR-100数据集模型:")
    print("   模型类型:")
    print("   - 'resnet18_fed': ResNet18联邦学习版本（适合分布式训练）")
    print("   - 'efficientnet': EfficientNet-B3风格模型（高精度）")
    print("   - 'densenet': DenseNet模型（密集连接网络）")
    print("   预设配置:")
    # 遍历并显示所有CIFAR-100预设配置
    for config in CIFAR100_MODEL_CONFIGS.keys():
        print(f"   - {config}")


def get_model_info(model):
    """
    获取模型信息
    ============
    
    返回模型的基本信息，包括参数量、模型大小等统计数据。
    这些信息对于模型选择和性能评估非常有用。
    
    参数:
        model (nn.Module): 需要分析的PyTorch模型实例
    
    返回:
        dict: 包含以下键值的模型信息字典:
            - total_params: 总参数数量
            - trainable_params: 可训练参数数量
            - model_size_mb: 模型大小（MB）
            - model_class: 模型类名
    """
    # 计算模型总参数数量
    total_params = sum(p.numel() for p in model.parameters())
    # 计算可训练参数数量（requires_grad=True的参数）
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 估算模型大小（以MB为单位）
    # 计算参数占用的字节数
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    # 计算缓冲区占用的字节数（如BatchNorm的running_mean等）
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    # 转换为MB
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    # 返回包含所有信息的字典
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb,
        'model_class': model.__class__.__name__
    }


def compare_models(models_dict, input_shape):
    """
    比较多个模型的性能指标
    ======================
    
    对比多个模型的参数量、推理时间等关键性能指标。
    这个函数帮助用户在多个模型之间做出选择。
    
    参数:
        models_dict (dict): {模型名称: 模型实例} 的字典
        input_shape (tuple): 输入张量形状 (batch_size, channels, height, width)
    
    返回:
        dict: 包含每个模型比较结果的字典，每个模型包含参数量、大小、推理时间等信息
    """
    import time
    
    results = {}
    # 创建测试用的随机输入张量
    test_input = torch.randn(*input_shape)
    
    # 遍历每个需要比较的模型
    for name, model in models_dict.items():
        # 设置模型为评估模式（关闭dropout等训练时的操作）
        model.eval()
        # 获取模型基本信息
        info = get_model_info(model)
        
        # 测试推理时间
        with torch.no_grad():  # 禁用梯度计算以加速推理
            start_time = time.time()
            # 运行100次推理取平均值，提高时间测量的准确性
            for _ in range(100):
                _ = model(test_input)
            # 计算平均推理时间
            avg_time = (time.time() - start_time) / 100
        
        # 将模型信息和推理时间合并到结果中
        results[name] = {
            **info,  # 展开基本模型信息
            'inference_time_ms': avg_time * 1000  # 转换为毫秒
        }
    
    return results


def print_model_comparison(results):
    """
    打印模型比较结果
    ================
    
    以表格形式美观地展示模型比较结果。
    方便用户直观地比较不同模型的性能特征。
    
    参数:
        results (dict): compare_models函数返回的比较结果字典
    """
    print("\n模型性能对比")
    print("=" * 100)
    # 打印表格头部
    print(f"{'模型名称':<20} {'参数量':<12} {'大小(MB)':<10} {'推理时间(ms)':<15} {'模型类型':<20}")
    print("-" * 100)
    
    # 遍历每个模型的结果并打印
    for name, info in results.items():
        print(f"{name:<20} {info['total_params']:<12,} {info['model_size_mb']:<10.2f} "
              f"{info['inference_time_ms']:<15.2f} {info['model_class']:<20}")


# 推荐配置字典 - 针对不同使用场景提供最佳模型配置建议
RECOMMENDED_CONFIGS = {
    'mnist': {
        'high_accuracy': 'cnn_optimized',    # 高精度场景：使用优化的CNN
        'fast_training': 'mlp_large'         # 快速训练场景：使用大型MLP
    },
    'cifar10': {
        'federated_learning': 'resnet18_fed_default',  # 联邦学习场景：ResNet18联邦版本
        'high_accuracy': 'efficientnet_default'        # 高精度场景：EfficientNet
    },
    'cifar100': {
        'federated_learning': 'resnet18_fed_default',  # 联邦学习场景：ResNet18联邦版本
        'high_accuracy': 'efficientnet_default',       # 高精度场景：EfficientNet
        'research': 'densenet_default'                  # 研究场景：DenseNet
    }
}


def get_recommended_model(dataset, scenario):
    """
    获取推荐模型配置
    ================
    
    根据使用场景返回经过优化和验证的推荐模型配置。
    这简化了模型选择过程，用户只需指定应用场景即可获得合适的模型。
    
    参数:
        dataset (str): 数据集名称
        scenario (str): 使用场景
            - 'high_accuracy': 高精度场景（追求最佳准确率）
            - 'federated_learning': 联邦学习场景（适合分布式训练）
            - 'fast_training': 快速训练场景（追求训练速度）
            - 'research': 研究场景（用于实验和研究）
    
    返回:
        nn.Module: 推荐的模型实例
    
    抛出异常:
        ValueError: 当指定的数据集或场景不被支持时
    """
    # 将数据集名称转换为小写
    dataset = dataset.lower()
    
    # 检查数据集是否在推荐配置中
    if dataset not in RECOMMENDED_CONFIGS:
        raise ValueError(f"不支持的数据集: {dataset}")
    
    # 检查场景是否在该数据集的推荐配置中
    if scenario not in RECOMMENDED_CONFIGS[dataset]:
        available_scenarios = list(RECOMMENDED_CONFIGS[dataset].keys())
        raise ValueError(f"数据集 {dataset} 不支持场景 {scenario}. 可用场景: {available_scenarios}")
    
    # 获取推荐的配置名称
    config_name = RECOMMENDED_CONFIGS[dataset][scenario]
    # 根据配置名称获取模型实例
    return get_model_by_config(dataset, config_name)


if __name__ == "__main__":
    """
    演示模型工厂的使用
    ==================
    
    当直接运行此文件时执行的演示代码，展示模型工厂的各种功能。
    """
    print("联邦学习模型工厂演示")
    print("=" * 50)
    
    # 显示所有可用的模型和配置信息
    list_available_models()
    
    # 定义要测试的数据集列表
    datasets = ['mnist', 'cifar10', 'cifar100']
    # 定义每个数据集对应的输入张量形状
    input_shapes = {
        'mnist': (1, 1, 28, 28),      # MNIST: 1通道, 28x28像素
        'cifar10': (1, 3, 32, 32),    # CIFAR-10: 3通道, 32x32像素
        'cifar100': (1, 3, 32, 32)    # CIFAR-100: 3通道, 32x32像素
    }
    
    # 遍历每个数据集进行测试
    for dataset in datasets:
        print(f"\n测试 {dataset.upper()} 数据集模型:")
        
        # 获取联邦学习场景的推荐模型
        try:
            # 尝试获取联邦学习推荐模型
            model = get_recommended_model(dataset, 'federated_learning')
            # 获取模型详细信息
            info = get_model_info(model)
            print(f"  联邦学习推荐模型: {info['model_class']}")
            print(f"  参数量: {info['total_params']:,}")  # 使用千分位分隔符格式化数字
            print(f"  模型大小: {info['model_size_mb']:.2f} MB")
            
            # 测试模型的前向传播功能
            test_input = torch.randn(*input_shapes[dataset])
            with torch.no_grad():  # 禁用梯度计算
                output = model(test_input)
            print(f"  输出形状: {output.shape}")
            
        except Exception as e:
            # 如果测试过程中出现任何错误，打印错误信息
            print(f"  错误: {e}")
    
    print(f"\n模型工厂演示完成！")