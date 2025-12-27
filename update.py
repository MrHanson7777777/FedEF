#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import copy
import math
from torch import nn # 导入 PyTorch 神经网络模块
from torch.utils.data import DataLoader, Dataset # 导入数据加载工具
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR # 导入学习率调度器

class DatasetSplit(Dataset): # 自定义数据集类，用于从大数据集中抽取一部分作为某个客户端的数据
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, idxs): # 构造函数
        self.dataset = dataset # 原始的完整数据集
        self.idxs = [int(i) for i in idxs] # 分配给该客户端的数据样本的索引列表

    def __len__(self): # 返回该客户端拥有的数据样本数量
        return len(self.idxs)

    def __getitem__(self, item): # 根据索引获取单个数据样本
        image, label = self.dataset[self.idxs[item]] # 从原始数据集中获取图像和标签
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

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, client_id=-1):
        self.args = args # 命令行参数
        self.client_id = client_id # 客户端唯一标识
        # 调用 train_val_test 方法，将分配给该客户端的数据 (由 idxs 指定) 划分为本地训练集、验证集和测试集
        # 并创建对应的 DataLoader
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        # 健壮的GPU设备检查
        try:
            gpu_id = int(args.gpu) if args.gpu is not None else -1
            if gpu_id >= 0 and torch.cuda.is_available():
                # 测试CUDA是否真正可用
                try:
                    test_tensor = torch.randn(1).cuda(gpu_id)
                    self.device = torch.device(f'cuda:{gpu_id}')
                    del test_tensor
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"CUDA设备{gpu_id}不可用: {str(e)}, 使用CPU")
                    self.device = torch.device('cpu')
            else:
                self.device = torch.device('cpu')
        except (ValueError, TypeError):
            self.device = torch.device('cpu')
        # Default criterion set to CrossEntropy loss function (默认损失函数为交叉熵损失，适用于原始logits)
        self.criterion = nn.CrossEntropyLoss()
        self.client_id = client_id

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        为给定的数据集和用户索引返回训练、验证和测试的 DataLoader。
        """
        # split indexes for train, validation, and test (80, 10, 10)
        # 将客户端数据按 80% 训练，10% 验证，10% 测试的比例划分
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        # 使用 DatasetSplit 和 DataLoader 创建对应的数据加载器
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True) # 本地训练批量大小，打乱数据
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False) # 验证集批量大小，不打乱
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False) # 测试集批量大小，不打乱
        return trainloader, validloader, testloader

    def _initialize_personalization_layer(self, model):
        """初始化个性化层"""
        if self.use_personalization and self.personalization_layer is None:
            from residual_utils import create_personalization_layer
            self.personalization_layer = create_personalization_layer(model, self.client_id, self.device)
    
    def _get_model_features(self, model, x):
        """从模型中提取特征，支持不同的模型结构"""
        if hasattr(model, 'forward_features') and hasattr(model, 'classifier'):
            # 新的模型结构，支持特征提取
            features = model.forward_features(x)
            return features, model.classifier
        elif hasattr(model, 'features') and hasattr(model, 'classifier'):
            # 类似VGG结构，有features和classifier
            features = model.features(x)
            features = features.view(features.size(0), -1)  # flatten
            return features, model.classifier
        elif hasattr(model, 'conv_layers') and hasattr(model, 'fc'):
            # 自定义CNN结构
            features = x
            for layer in model.conv_layers:
                features = layer(features)
            # 假设有avgpool
            if hasattr(model, 'avgpool'):
                features = model.avgpool(features)
            features = features.view(features.size(0), -1)  # flatten
            return features, model.fc
        else:
            # 简单结构，尝试找到最后的全连接层
            if hasattr(model, 'fc'):
                # 通过forward到fc层之前获取特征
                features = model.forward_features(x) if hasattr(model, 'forward_features') else None
                if features is None:
                    # 如果没有forward_features方法，使用完整的forward
                    return None, None
                return features, model.fc
            elif hasattr(model, 'classifier'):
                features = model.forward_features(x) if hasattr(model, 'forward_features') else None
                if features is None:
                    return None, None
                return features, model.classifier
            else:
                return None, None

    def update_weights(self, model, global_round, global_weights=None):
        """
        执行本地模型更新 (训练) - [RFL Aligned Version]
        """
        # 确保模型在正确的设备上
        model = model.to(self.device)
        model.train()
        epoch_loss = []

        # --- 新增代码：在这里一次性创建好教师模型 ---
        teacher_model = None
        distill_warmup_rounds = getattr(self.args, 'distill_warmup_rounds', 3) 

        if (hasattr(self.args, 'iid') and self.args.iid == 0 
            and getattr(self.args, 'enable_knowledge_distillation', 1) == 1
            and global_weights is not None
            and global_round > distill_warmup_rounds):

            print(f"[CLIENT {self.client_id}] 启用知识蒸馏 (轮次 > {distill_warmup_rounds})")
            from model_factory import get_model
            teacher_model = get_model(self.args.dataset, self.args.model)

            temp_weights = {k: v.to(self.device) for k, v in global_weights.items()}
            teacher_model.load_state_dict(temp_weights)
            teacher_model = teacher_model.to(self.device)
            teacher_model.eval()

            for param in teacher_model.parameters():
                param.requires_grad = False
        # --- 新增结束 ---

        # 设置优化器
        optimizer = self._get_optimizer(model)

        # 设置学习率
        lr_scheduler_type = getattr(self.args, 'lr_scheduler', 'none')
        current_lr = self.args.lr

        if lr_scheduler_type == 'cosine':
            total_rounds = getattr(self.args, 'epochs', 50)
            min_lr = self.args.lr * 0.05
            warmup_rounds = min(5, total_rounds // 10)
            if global_round < warmup_rounds:
                current_lr = self.args.lr * (global_round + 1) / warmup_rounds
            else:
                effective_round = global_round - warmup_rounds
                effective_total = total_rounds - warmup_rounds
                cosine_factor = 0.5 * (1 + math.cos(math.pi * effective_round / effective_total))
                current_lr = min_lr + (self.args.lr - min_lr) * cosine_factor

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        print(f"[CLIENT {self.client_id}] 轮次 {global_round}: 学习率 = {current_lr:.6f}")

        # 本地训练循环
        for iter_epoch in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                outputs = model(images)

                if torch.isnan(outputs).any():
                    print(f"警告: 模型输出包含NaN值，跳过此批次")
                    continue

                ce_loss = self.criterion(outputs, labels)

                if torch.isnan(ce_loss):
                    print(f"警告: 交叉熵损失为NaN，跳过此批次")
                    continue

                total_loss = ce_loss
                if teacher_model is not None:
                    try:
                        with torch.no_grad():
                            teacher_logits = teacher_model(images)

                        if not torch.isnan(teacher_logits).any() and not torch.isinf(teacher_logits).any():
                            T = getattr(self.args, 'distill_temperature', 3.0)
                            alpha = getattr(self.args, 'distill_alpha', 0.3)
                            student_soft = torch.log_softmax(outputs / T, dim=1)
                            teacher_soft = torch.softmax(teacher_logits / T, dim=1)
                            distill_loss = torch.nn.functional.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T * T)

                            if not torch.isnan(distill_loss) and not torch.isinf(distill_loss):
                                total_loss = (1 - alpha) * ce_loss + alpha * distill_loss
                    except Exception as e:
                        print(f"知识蒸馏计算出错，使用标准损失: {str(e)}")
                        total_loss = ce_loss

                loss = total_loss

                if torch.isnan(loss):
                    print(f"警告: 最终损失为NaN，跳过此批次")
                    continue

                if getattr(self.args, 'iid', 1) == 0 and getattr(self.args, 'mu', 0.0) > 0 and global_weights is not None:
                    prox_term = 0.0
                    for name, param in model.named_parameters():
                        if name in global_weights:
                            global_param_tensor = global_weights[name].detach().to(param.device)
                            prox_term += torch.sum(torch.pow(param - global_param_tensor, 2))
                    loss += (self.args.mu / 2) * prox_term

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    # 计算训练进度
                    total_batches = len(self.trainloader)
                    progress_percent = 100.0 * (batch_idx + 1) / total_batches
                    print(f'| Global Round : {global_round} | Local Epoch : {iter_epoch} | [{batch_idx + 1}/{total_batches} ({progress_percent:.0f}%)] Loss: {loss.item():.6f}', flush=True)

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def _get_optimizer(self, model):
        """
        根据参数选择并创建优化器
        """
        if self.args.optimizer.lower() == 'sgd':
            # 针对CIFAR数据集优化SGD参数
            if hasattr(self.args, 'dataset') and self.args.dataset in ['cifar', 'cifar100']:
                # CIFAR-10/CIFAR-100专用SGD配置
                return torch.optim.SGD(
                    model.parameters(), 
                    lr=self.args.lr,
                    momentum=0.9,  # 固定使用0.9动量
                    weight_decay=5e-4,  # 增强正则化
                    nesterov=True  # 使用Nesterov动量
                )
            else:
                # 其他数据集使用标准SGD配置
                return torch.optim.SGD(
                    model.parameters(), 
                    lr=self.args.lr,
                    momentum=self.args.momentum,
                    weight_decay=self.args.weight_decay
                )
        elif self.args.optimizer.lower() == 'adam':
            return torch.optim.Adam(
                model.parameters(), 
                lr=self.args.lr,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_eps,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer.lower() == 'adamw':
            return torch.optim.AdamW(
                model.parameters(), 
                lr=self.args.lr,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_eps,
                weight_decay=self.args.weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器: {self.args.optimizer}. 支持的优化器: ['sgd', 'adam', 'adamw']")
    
    def _get_scheduler(self, optimizer):
        """
        根据参数选择并创建学习率调度器
        """
        if self.args.lr_scheduler.lower() in ['none', 'fixed']:
            return None
        elif self.args.lr_scheduler.lower() == 'step':
            return StepLR(
                optimizer, 
                step_size=self.args.lr_step_size, 
                gamma=self.args.lr_gamma
            )
        elif self.args.lr_scheduler.lower() == 'exp':
            return ExponentialLR(
                optimizer, 
                gamma=self.args.lr_gamma
            )
        elif self.args.lr_scheduler.lower() == 'cosine':
            return CosineAnnealingLR(
                optimizer, 
                T_max=self.args.cosine_t_max,
                eta_min=getattr(self.args, 'eta_min', 1e-6)
            )
        else:
            raise ValueError(f"不支持的学习率调度器: {self.args.lr_scheduler}. 支持的调度器: ['none', 'fixed', 'step', 'exp', 'cosine']")

    def inference(self, model): # 在本地测试集上进行模型推断 (评估)
        """ Returns the inference accuracy and loss.
        """
        '''
        本地评估
        '''
        # 确保模型在正确的设备上
        model = model.to(self.device)
        model.eval() # 将模型设置为评估模式 (这会禁用 Dropout, BatchNorm 等层的训练行为)
        loss, total, correct = 0.0, 0.0, 0.0 # 初始化损失、总样本数、正确预测数
        valid_batches = 0

        for batch_idx, (images, labels) in enumerate(self.testloader): # 遍历本地测试数据加载器
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images) # 模型前向传播
            
            # 检查模型输出是否包含NaN或Inf
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print(f"警告: 本地测试时模型输出包含NaN/Inf值，跳过此批次")
                continue
            
            batch_loss = self.criterion(outputs, labels) # 计算当前 batch 的损失
            
            # 检查损失是否为NaN或Inf
            if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                print(f"警告: 本地测试损失为NaN/Inf，跳过此批次")
                continue
            
            loss += batch_loss.item() # 累加损失
            valid_batches += 1

            # Prediction
            _, pred_labels = torch.max(outputs, 1) # 获取预测概率最高的类别索引
            pred_labels = pred_labels.view(-1) # 展平预测标签
            correct += torch.sum(torch.eq(pred_labels, labels)).item() # 计算正确预测的数量
            total += len(labels) # 累加总样本数

        if total > 0:
            accuracy = correct/total # 计算准确率
        else:
            accuracy = 0.0
            
        # 计算平均损失，避免除零
        if valid_batches > 0:
            avg_loss = loss / valid_batches
        else:
            avg_loss = float('inf')  # 如果没有有效批次，返回无穷大
            print("警告: 本地测试过程中没有有效的批次")
            
        return accuracy, avg_loss # 返回准确率和平均损失


def test_inference(args, model, test_dataset): # 在全局测试集上进行模型推断 (评估全局模型性能)
    """ Returns the test accuracy and loss.
    """
    '''
    server评估
    '''
    # 健壮的GPU设备检查
    try:
        gpu_id = int(args.gpu) if args.gpu is not None else -1
        if gpu_id >= 0 and torch.cuda.is_available():
            # 测试CUDA是否真正可用
            try:
                test_tensor = torch.randn(1).cuda(gpu_id)
                device = torch.device(f'cuda:{gpu_id}')
                del test_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"CUDA设备{gpu_id}不可用: {str(e)}, 使用CPU")
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
    except (ValueError, TypeError):
        device = torch.device('cpu')
    
    # 确保模型在正确的设备上
    model = model.to(device)
    model.eval() # 设置为评估模式
    loss, total, correct = 0.0, 0.0, 0.0
    valid_batches = 0

    criterion = nn.CrossEntropyLoss().to(device)
    # 使用完整的测试数据集创建 DataLoader
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    with torch.no_grad(): # 在评估时，不需要计算梯度，可以节省内存和计算
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(images)
            
            # 检查模型输出是否包含NaN或Inf
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print(f"警告: 测试时模型输出包含NaN/Inf值，跳过此批次")
                continue
            
            batch_loss = criterion(outputs, labels)
            
            # 检查损失是否为NaN或Inf
            if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                print(f"警告: 测试损失为NaN/Inf，跳过此批次")
                continue
            
            loss += batch_loss.item()
            valid_batches += 1

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

    if total > 0:
        accuracy = correct/total
    else:
        accuracy = 0.0
    
    # 计算平均损失，避免除零
    if valid_batches > 0:
        avg_loss = loss / valid_batches
    else:
        avg_loss = float('inf')  # 如果没有有效批次，返回无穷大
        print("警告: 测试过程中没有有效的批次")
    
    return accuracy, avg_loss # 返回在整个测试集上的准确率和平均损失