#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

'''
这个文件负责将数据集划分给不同的客户端，模拟联邦学习中的数据分布情况。
它包含了为 MNIST 和 CIFAR-10 数据集生成独立同分布 (IID) 和非独立同分布 (Non-IID) 数据的函数。
Non-IID 的情况进一步区分为数据量均衡和不均衡的场景。
'''

import numpy as np # 导入 NumPy 用于数值计算，特别是数组操作
from torchvision import datasets, transforms # 从 torchvision 导入数据集和图像变换工具
import torch
from torch.utils.data import Dataset

class DatasetSplit(Dataset):
    """数据集分割类"""
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # 使用clone().detach()代替torch.tensor()来避免警告
        if isinstance(image, torch.Tensor):
            image = image.clone().detach()
        else:
            image = torch.tensor(image)
            
        if isinstance(label, torch.Tensor):
            label = label.clone().detach()
        else:
            label = torch.tensor(label)
            
        return image, label

def mnist_iid(dataset, num_users): # 为 MNIST 数据集进行 IID 划分
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset: 整个训练数据集
    :param num_users: 客户端数量
    :return: dict of image index, 字典,键是用户ID,值是该用户拥有的数据样本索引集合
    """
    num_items = int(len(dataset)/num_users) # 计算每个用户平均分配到的数据项数量
    dict_users, all_idxs = {}, [i for i in range(len(dataset))] # 初始化用户字典和所有数据样本的索引列表
    for i in range(num_users):
        # np.random.choice 从 all_idxs 中不重复地随机选择 num_items 个索引
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        #replace表示随机选择时不允许重复选择。换句话说，同一个索引不能被同一个用户选择两次。这确保了分配给每个用户的索引是唯一的。
        # 从 all_idxs 中移除已经分配给用户的索引
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users
'''
最终是一个字典，其结构如下：
键(key):用户编号int,0 ~ num_users-1
值(value):该用户拥有的数据样本索引的集合(set,里面是 int)
可能是
{
    0: {2, 5, 8},
    1: {0, 3, 7},
    2: {1, 4, 6}
}
'''


def mnist_noniid(dataset, num_users): # 为 MNIST 数据集进行 Non-IID 划分 (每个用户固定分片数)
    """
    Sample non-I.I.D client data from MNIST dataset (Robust Version)
    
    使用"洗牌后轮流分发"策略，确保分片分配的公平性和唯一性。
    
    :param dataset: MNIST数据集
    :param num_users: 客户端数量
    :return: 客户端数据索引字典
    """
    # MNIST 训练集 60,000 张图片 --> 假设每个分片 300 张图片，共 200 个分片
    # "shard" 指的是将数据库数据分割成多个片段或子集。
    # 每片图片的标签是相邻的（即同一片大概率属于同一类别），这样每个客户端的数据分布就不是IID，而是偏向某几个类别
    num_shards, num_imgs = 200, 300
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)  # 所有数据样本的原始索引 (0 到 59999)
    
    # 获取训练数据的标签 (兼容不同PyTorch版本)
    # 优先使用 .targets 属性 (新版torchvision)，否则回退到 .train_labels
    labels = np.array(dataset.targets if hasattr(dataset, 'targets') else dataset.train_labels)
    
    # sort labels (关键步骤：按标签排序数据，以创建 Non-IID 分布)
    idxs_labels = np.vstack((idxs, labels))  # 将索引和标签垂直堆叠
    # argsort() 返回排序后的索引，这里按第二行 (标签) 排序
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # 按标签排序，保证同一类别的图片索引排在一起
    idxs = idxs_labels[0, :]  # 获取排序后的数据样本索引

    # --- BEGIN: Recommended Change ---
    
    # 创建所有分片的索引列表
    shard_idxs = list(range(num_shards))
    
    # 随机打乱分片索引
    np.random.shuffle(shard_idxs)
    
    # 使用轮流分发 (round-robin) 的方式将打乱后的分片分配给客户端
    for i in range(num_users):
        # 计算当前用户应该分配到的分片
        assigned_shards = shard_idxs[i::num_users]  # 从第i个开始，每隔num_users个取一个
        
        # 将这些分片中的数据索引分配给用户
        for shard_id in assigned_shards:
            start_idx = shard_id * num_imgs
            end_idx = start_idx + num_imgs
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[start_idx:end_idx]), axis=0)

    # 打印分配信息
    shards_per_client_list = [len(shard_idxs[i::num_users]) for i in range(num_users)]
    print(f"📊 数据分配信息: {num_users}个客户端，"
          f"每客户端分配 {min(shards_per_client_list)}-{max(shards_per_client_list)} 个分片")
          
    # --- END: Recommended Change ---
    
    return dict_users
'''
假设 num_users = 3,每个用户分配到的图片索引如下
{
    0: array([   0.,    1.,    2., ...,  899.,  900., 1199.]),  # 用户0拥有第0~1199号图片
    1: array([1200., 1201., ..., 1799.]),                       # 用户1拥有第1200~1799号图片
    2: array([1800., 1801., ..., 2399.])                        # 用户2拥有第1800~2399号图片
}
'''

def mnist_noniid_unequal(dataset, num_users): # 为 MNIST 数据集进行 Non-IID 且数据量不均衡的划分
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # MNIST 训练集 60,000 张图片 --> 假设每个分片 50 张图片，共 1200 个分片
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    
    # 获取训练数据的标签 (兼容不同PyTorch版本)
    # 优先使用 .targets 属性 (新版torchvision)，否则回退到 .train_labels
    labels = np.array(dataset.targets if hasattr(dataset, 'targets') else dataset.train_labels)

    # sort labels (同上，按标签排序)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1 # 每个客户端最少分配的分片数
    max_shard = 30 # 每个客户端最多分配的分片数

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    # 为每个用户随机生成一个介于 min_shard 和 max_shard 之间的分片数量
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    '''比如 np.random.randint(1, 31, size=4) 得到 [10, 25, 5, 20],其实作为一个占比,后面与分片总数相乘'''
    # 将这些随机生成的分片数量进行归一化，使其总和约等于总分片数 num_shards
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int) # 转换为整数
    '''归一化后，假设得到 [200, 400, 100, 500],即4个用户分别分配200、400、100、500个分片(总和约等于1200)。'''

    # Assign the shards randomly to each client
    # 处理随机分配后分片总数可能略大于或小于 num_shards 的情况
    if sum(random_shard_size) > num_shards:
        # 如果分配的总分片数超过了实际拥有的分片数
        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data (先给每个用户分配一个基础分片)
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1 # 减去已分配的基础分片
        '''先分基础分片是为了防止有客户端分不到数据，后面这段代码是把剩余分片按归一化后的目标数量继续分配，直到分片分完。'''
        # Next, randomly assign the remaining shards (再分配剩余的)
        for i in range(num_users):
            if len(idx_shard) == 0: # 如果没有剩余分片了，则跳过
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard): # 如果期望分配数大于剩余数，则取剩余数
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else: # 如果分配的总分片数小于等于实际拥有的分片数
        '''
        如果分配完每个用户的目标分片数后还有剩余的分片(len(idx_shard) > 0),就把这些剩余的分片全部分配给当前拥有数据最少的那个用户。
        '''
        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0: # 如果还有剩余的分片
            # Add the leftover shards to the client with minimum images:
            # 将剩余的分片分配给当前拥有数据最少的客户端
            shard_size = len(idx_shard)
            # 找到拥有数据最少的客户端 k
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    return dict_users


def cifar_iid(dataset, num_users): # 为 CIFAR-10 数据集进行 IID 划分
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    # 逻辑与 mnist_iid 完全相同
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users): # 为 CIFAR-10 数据集进行 Non-IID 划分
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # CIFAR-10 训练集 50,000 张图片 --> 假设每个分片 250 张图片，共 200 个分片
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy() # torchvision 0.9.1 之前版本
    labels = np.array(dataset.targets) # torchvision 0.9.1 及之后版本，CIFAR10 的标签属性名为 targets

    # sort labels (同上)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign (每个客户端分配 2 个主要类别的分片)
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def cifar_noniid_dirichlet(dataset, num_users, alpha=0.5):
    """
    使用 Dirichlet 分布进行 CIFAR 数据集的 Non-IID 划分
    
    Args:
        dataset: CIFAR 数据集 (支持 CIFAR-10 和 CIFAR-100)
        num_users: 客户端数量
        alpha: Dirichlet 分布的浓度参数，越小数据越不均匀
        
    Returns:
        dict_users: 字典，键为客户端ID，值为该客户端的数据索引数组
    """
    # 自动检测类别数量
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))  # 动态检测类别数量
    dict_users = {i: np.array([]) for i in range(num_users)}
    
    # 按类别组织数据索引
    labels = np.array(dataset.targets)
    label_distribution = [[] for _ in range(num_classes)]
    for idx, label in enumerate(labels):
        label_distribution[label].append(idx)
    
    # 为每个客户端生成类别分布
    for i in range(num_users):
        # 从 Dirichlet 分布采样类别权重
        proportions = np.random.dirichlet(np.repeat(alpha, num_classes))
        
        # 将权重转换为每个类别的样本数量
        proportions = np.array([p * len(label_distribution[j]) 
                               for j, p in enumerate(proportions)])
        proportions = proportions.astype(int)
        
        # 确保每个客户端至少有一些数据
        if proportions.sum() == 0:
            proportions[np.random.randint(0, num_classes)] = 1
            
        # 为当前客户端分配数据
        client_data = []
        for j in range(num_classes):
            if proportions[j] > 0:
                # 从该类别中随机选择相应数量的样本
                available_samples = len(label_distribution[j])
                take_samples = min(proportions[j], available_samples)
                
                if take_samples > 0:
                    selected = np.random.choice(
                        label_distribution[j], 
                        take_samples, 
                        replace=False
                    )
                    client_data.extend(selected)
                    
                    # 从可用样本中移除已选择的样本
                    label_distribution[j] = list(
                        set(label_distribution[j]) - set(selected)
                    )
        
        dict_users[i] = np.array(client_data)
    
    # 打印数据分布信息（用于调试）
    print(f"[INFO] Dirichlet 划分完成 (alpha={alpha}):")
    total_samples = 0
    for i in range(min(5, num_users)):  # 只显示前5个客户端的分布
        client_labels = [labels[idx] for idx in dict_users[i]]
        label_counts = np.bincount(client_labels, minlength=num_classes)
        total_samples += len(dict_users[i])
        print(f"  客户端 {i}: {len(dict_users[i])} 样本, 类别分布: {label_counts}")
    print(f"  总样本数: {total_samples}")
    
    return dict_users


if __name__ == '__main__': # 测试代码块
    # 加载 MNIST 训练数据集
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=False,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(), # 将 PIL Image 或 numpy.ndarray 转换为 FloatTensor，并将图像的像素范围从 [0, 255] 归一化到 [0, 1]
                                       transforms.Normalize((0.1307,), (0.3081,)) # 用均值和标准差对张量图像进行标准化
                                   ]))
    num = 100 # 假设有 100 个用户
    d = mnist_noniid(dataset_train, num) # 测试 Non-IID 划分函数