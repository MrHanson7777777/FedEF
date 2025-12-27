# 

> 这是我在机器学习课程大作业中实现的联邦学习实验项目。我在这个项目中尝试了不同的联邦学习方法，包括基线对比实验、经典联邦学习和残差联邦学习的实现。

## 项目简介

本项目实现了三种不同的机器学习训练方法：

- **基线实验 (Baseline)**：集中式训练，作为对比基准
- **联邦学习 (Federated Learning)**：经典的FedAvg算法实现  
- **残差联邦学习 (Residual Federated Learning)**：我们提出的改进方法

通过对比这三种方法在不同数据集上的表现，探索联邦学习的优势和局限性。

## 项目结构

``` bash
Code/
├── 主程序文件
│   ├── baseline_main.py          # 基线实验主程序
│   ├── federated_main.py         # 联邦学习主程序  
│   ├── residual_main.py          # 残差联邦学习主程序
│   └── experiment_manager.py     # 统一实验管理器
│
├── 模型定义
│   ├── models.py                 # 兼容性模型集合
│   ├── models_mnist.py           # MNIST专用模型
│   ├── models_cifar10.py         # CIFAR-10专用模型
│   ├── models_cifar100.py        # CIFAR-100专用模型
│   └── model_factory.py          # 模型工厂类
│
├── 核心组件
│   ├── options.py                # 命令行参数配置
│   ├── utils.py                  # 工具函数和数据增强
│   ├── update.py                 # 本地更新和测试函数
│   ├── sampling.py               # 数据采样和分布模拟
│   └── residual_utils.py         # 残差学习专用工具
│
├── 辅助工具
│   ├── download.py               # 数据集下载工具
│   └── visualize_results.py      # 实验结果可视化
│
└── 输出目录
    ├── data/                     # 数据集存储
    ├── logs/                     # 训练日志
    └── save/                     # 保存的模型和结果
        ├── logs/                 # 实验日志记录
        ├── objects/              # 序列化对象
        └── plots/                # 实验图表
```

## 快速开始

### 环境要求

- Python 3.6+
- PyTorch >= 1.0.0
- torchvision
- numpy
- matplotlib
- pandas
- tqdm
- tensorboardX

### 安装依赖

```bash
pip install torch torchvision numpy matplotlib pandas tqdm tensorboardX
```

### 使用统一实验管理器（推荐）

```bash
# 查看帮助信息
python experiment_manager.py --help

# 运行MNIST基线实验
python experiment_manager.py --dataset mnist --method baseline --model cnn --epochs 50

# 运行CIFAR-10联邦学习实验
python experiment_manager.py --dataset cifar --method federated --model resnet18 --epochs 100

# 运行CIFAR-100残差联邦学习实验
python experiment_manager.py --dataset cifar100 --method residual --model efficientnet --epochs 150
```

### 单独运行实验

#### 1. 基线实验

```bash
python baseline_main.py --dataset mnist --model cnn --epochs 50 --lr 0.01
python baseline_main.py --dataset cifar --model resnet18 --epochs 100 --lr 0.001
```

#### 2. 联邦学习实验  

```bash
python federated_main.py --dataset mnist --model cnn --epochs 50 --num_users 100 --frac 0.1
python federated_main.py --dataset cifar --model resnet18 --epochs 100 --num_users 50 --frac 0.2
```

#### 3. 残差联邦学习实验

```bash
python residual_main.py --dataset mnist --model cnn --epochs 50 --pretrain_epochs 10
python residual_main.py --dataset cifar100 --model efficientnet --epochs 150 --residual_type adaptive
```

## 支持的数据集和模型

### 数据集

- **MNIST**: 手写数字识别 (28×28灰度图像)
- **Fashion-MNIST**: 服装分类 (28×28灰度图像)  
- **CIFAR-10**: 自然图像分类 (32×32彩色图像，10类)
- **CIFAR-100**: 细粒度图像分类 (32×32彩色图像，100类)

### 模型架构

- **MNIST**: CNN, 优化CNN, 优化CNN+GroupNorm, MLP
- **CIFAR-10**: CNN, ResNet18, ResNet18-Fed, EfficientNet
- **CIFAR-100**: CNN, ResNet18-Fed, EfficientNet, DenseNet
- **Fashion-MNIST**: CNN

## 重要参数说明

### 联邦学习核心参数

- `--epochs`: 全局训练轮数
- `--num_users`: 参与训练的客户端总数
- `--frac`: 每轮选择的客户端比例
- `--local_ep`: 客户端本地训练轮数
- `--local_bs`: 本地批量大小
- `--lr`: 学习率

### 数据分布设置

- `--iid`: IID数据分布（默认：False）
- `--unequal`: 非等量数据分布（默认：False）

### 残差学习特有参数

- `--pretrain_epochs`: 预训练轮数
- `--residual_type`: 残差类型 (basic/adaptive)
- `--residual_weight`: 残差权重

## 实验结果可视化

运行实验后，使用可视化工具分析结果：

```bash
# 可视化单个实验结果
python visualize_results.py path/to/log_file.csv --output_dir ./save/plots

# 对比多个实验结果  
python visualize_results.py --comparison --output_dir ./save/plots
```

生成的图表包括：

- 训练损失曲线
- 测试准确率曲线
- 客户端性能分布
- 方法对比分析

## 实验设计说明

### 消融研究思路

作为学生，我设计了以下消融研究：

1. **基线对比**: 首先运行集中式训练获得理想性能上界
2. **联邦学习**: 实现经典FedAvg算法，观察去中心化带来的性能损失
3. **残差学习**: 提出改进方法，尝试缓解联邦学习中的问题

### 关键创新点

- **智能客户端选择**: 基于历史表现动态选择参与训练的客户端
- **自适应聚合**: 根据客户端数据质量调整聚合权重
- **残差学习机制**: 通过残差连接改善收敛性能
- **数据增强策略**: 实现CutMix、Mixup等增强技术

训练过程中会自动保存详细日志：

```save/logs/20251227_143000/
├── experiment_config.json    # 实验配置
├── training_log.txt          # 详细训练日志  
├── results_summary.csv       # 结果汇总
└── tensorboard_logs/         # TensorBoard日志
```

这是我在机器学习课程中的学习成果，希望能为联邦学习的研究贡献一份微薄之力



