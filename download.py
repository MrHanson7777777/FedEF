#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
from torchvision import datasets, transforms

# 创建保存数据的目录结构
os.makedirs('../data/mnist', exist_ok=True)
os.makedirs('../data/fmnist', exist_ok=True)
os.makedirs('../data/cifar', exist_ok=True)

# 定义MNIST数据集的变换和下载
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 下载MNIST训练集和测试集
print("正在下载MNIST数据集...")
datasets.MNIST(
    root='../data/mnist',
    train=True,
    download=True,
    transform=mnist_transform
)

datasets.MNIST(
    root='../data/mnist',
    train=False,
    download=True,
    transform=mnist_transform
)
print("MNIST数据集下载完成！")

# 定义Fashion-MNIST数据集的变换和下载
fmnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 下载Fashion-MNIST训练集和测试集
print("正在下载Fashion-MNIST数据集...")
datasets.FashionMNIST(
    root='../data/fmnist',
    train=True,
    download=True,
    transform=fmnist_transform
)

datasets.FashionMNIST(
    root='../data/fmnist',
    train=False,
    download=True,
    transform=fmnist_transform
)
print("Fashion-MNIST数据集下载完成！")

# 定义CIFAR-10数据集的变换和下载
cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 下载CIFAR-10训练集和测试集
print("正在下载CIFAR-10数据集...")
datasets.CIFAR10(
    root='../data/cifar',
    train=True,
    download=True,
    transform=cifar_transform
)

datasets.CIFAR10(
    root='../data/cifar',
    train=False,
    download=True,
    transform=cifar_transform
)
print("CIFAR-10数据集下载完成！")

print("\n所有数据集已成功下载并保存在正确的目录中。")
print("您现在可以运行原始代码，无需担心数据下载问题。")    