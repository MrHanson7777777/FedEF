#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 残差联邦学习工具函数 - 简化版

import torch
import copy
import time
import math
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sampling import DatasetSplit

# 设置cuDNN选项以解决算法选择问题
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True  # 启用benchmark以自动选择最优算法
torch.backends.cudnn.deterministic = False  # 允许非确定性算法获得更好性能


def calculate_diversity_scores_residual(local_residuals, client_data_sizes, server_model_template, args=None):
    """
    计算客户端残差的多样性分数 - 残差联邦学习版本
    基于残差权重差异和数据分布不平衡程度
    支持打包格式和密集格式的残差数据
    """
    num_clients = len(local_residuals)
    diversity_scores = []
    #num_clients 存储参与本轮的客户端总数
    #diversity_scores 是一个空列表，后续会用来存放我们为每个客户端计算出的最终多样性分数

    # 检查上行链路是否压缩
    is_compressed = (args is not None and args.uplink_compression != 'none')

    # 计算每个客户端相对于其他客户端的残差权重差异
    for i in range(num_clients):
        total_distance = 0.0
        #在为客户端 i 计算分数之前，先将它的“总距离”(total_distance)清零。
        #这个变量将用来累加它与其他所有客户端的差异程度。
        weight_count = 0
        
        for j in range(num_clients):
            if i != j:
                # 计算两个残差权重之间的余弦相似度
                distance = 0.0
                
                # 根据压缩状态决定是否解包
                if is_compressed and server_model_template is not None:
                    # 压缩格式，需要解包
                    dense_residual_i = unpack_sparse_residual(local_residuals[i], template=server_model_template)
                    dense_residual_j = unpack_sparse_residual(local_residuals[j], template=server_model_template)
                else:
                    # 未压缩格式，local_residuals 已经是密集的
                    dense_residual_i = local_residuals[i]
                    dense_residual_j = local_residuals[j]
                
                for key in dense_residual_i.keys():
                    if key in dense_residual_j:
                        w1 = dense_residual_i[key].flatten().float()  # 确保是浮点型
                        w2 = dense_residual_j[key].flatten().float()  # 确保是浮点型
                        '''
                        .flatten() 将某一层的所有参数（无论原来是矩阵还是更高维的张量）“压平”成一个长长的一维向量
                        .float() 确保数据类型是浮点数，便于后续计算
                        现在 w1 和 w2 分别代表了客户端 i 和 j 在同一层上的更新向量
                        '''

                        # 计算余弦距离 (1 - cosine_similarity)
                        norm1 = torch.norm(w1)
                        norm2 = torch.norm(w2)
                        '''
                        norm1 和 norm2 是向量的长度（模）
                        '''
                        
                        if norm1 > 0 and norm2 > 0:
                            cosine_sim = torch.dot(w1, w2) / (norm1 * norm2)#torch.dot(w1, w2) 是向量的点积
                            cosine_distance = 1.0 - cosine_sim.item()
                            '''
                            1.0 - cosine_sim 将相似度转换成了距离
                            方向越一致（相似度接近1），距离就越接近0
                            方向越相反（相似度接近-1），距离就越接近2。
                            '''
                            distance += cosine_distance
                            weight_count += 1
                            #当 for key ... 这个最内层循环全部执行完毕后，weight_count 的值已经不再是0了
                            #它等于客户端 i 和 j 之间共同拥有的、被成功比较的总层数
                
                if weight_count > 0:
                    total_distance += distance / weight_count
                    #total_distance 就代表了客户端 i 与其他所有客户端的平均距离之和
        
        #计算平均距离
        avg_distance = total_distance / max(1, num_clients - 1)
        
        # 结合数据量不平衡因子
        total_samples = sum(client_data_sizes)
        data_imbalance = abs(client_data_sizes[i] / total_samples - 1.0 / num_clients)
        '''
        1.0 / num_clients 是在数据完全均匀分布时，每个客户端应占的数据比例
        client_data_sizes[i] / total_samples 是客户端 i 实际的数据占比
        这两者之差的绝对值，就衡量了客户端 i 的数据量偏离“平均水平”的程度
        数据量特别多或特别少的客户端，这个值都会比较大
        '''
        
        # 综合多样性分数 (残差差异 + 数据不平衡)
        diversity_score = 0.7 * avg_distance + 0.3 * data_imbalance
        diversity_scores.append(diversity_score)
    
    return diversity_scores

class DatasetSplit(Dataset):
    """将数据集分割给不同客户端的类"""
    
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

class LocalUpdateResidual(object):
    """残差联邦学习的客户端本地更新类"""
    
    def __init__(self, args, dataset, idxs, client_id=None):
        self.args = args
        self.client_id = client_id if client_id is not None else "Unknown"
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        
        # 设备设置
        try:
            gpu_id = int(args.gpu) if args.gpu is not None else -1
            self.device = 'cuda' if gpu_id >= 0 and torch.cuda.is_available() else 'cpu'
        except (ValueError, TypeError):
            self.device = 'cpu'
            
        self.criterion = nn.CrossEntropyLoss().to(self.device)
    
    def train_val_test(self, dataset, idxs):
        """将数据集分割为训练、验证和测试集"""
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

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
        """更新模型权重"""
        # 设置模型为训练模式
        model.train()
        epoch_loss = []

        # 设置优化器
        trainable_params = list(model.parameters())

        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(trainable_params, lr=self.args.lr,
                                        momentum=getattr(self.args, 'momentum', 0.5), 
                                        weight_decay=getattr(self.args, 'weight_decay', 1e-4))
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(trainable_params, lr=self.args.lr,
                                         betas=(getattr(self.args, 'adam_beta1', 0.9), 
                                               getattr(self.args, 'adam_beta2', 0.999)),
                                         eps=getattr(self.args, 'adam_eps', 1e-8),
                                         weight_decay=getattr(self.args, 'weight_decay', 1e-4))
        elif self.args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr,
                                          betas=(getattr(self.args, 'adam_beta1', 0.9), 
                                                getattr(self.args, 'adam_beta2', 0.999)),
                                          eps=getattr(self.args, 'adam_eps', 1e-8),
                                          weight_decay=getattr(self.args, 'weight_decay', 0.01))
        else:
            # 默认使用SGD
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5, weight_decay=1e-4)

        # 学习率调度
        lr_scheduler_type = getattr(self.args, 'lr_scheduler', 'none')
        
        if lr_scheduler_type == 'cosine':
            # 改进的余弦退火学习率调整
            total_rounds = getattr(self.args, 'epochs', 50)
            min_lr = self.args.lr * 0.05  # 降低最小学习率，增强后期微调
            
            # 使用warmup + cosine策略
            warmup_rounds = min(5, total_rounds // 10)  # 前10%轮次进行warmup
            if global_round < warmup_rounds:
                # Warmup阶段：线性增长到目标学习率
                current_lr = self.args.lr * (global_round + 1) / warmup_rounds
            else:
                # Cosine退火阶段
                effective_round = global_round - warmup_rounds
                effective_total = total_rounds - warmup_rounds
                cosine_factor = 0.5 * (1 + math.cos(math.pi * effective_round / effective_total))
                current_lr = min_lr + (self.args.lr - min_lr) * cosine_factor
        elif lr_scheduler_type == 'step':
            # 步长调度器
            step_size = getattr(self.args, 'lr_step_size', 20)
            gamma = getattr(self.args, 'lr_gamma', 0.1)
            current_lr = self.args.lr * (gamma ** (global_round // step_size))
        elif lr_scheduler_type == 'exp':
            # 指数衰减调度器
            gamma = getattr(self.args, 'lr_gamma', 0.95)
            current_lr = self.args.lr * (gamma ** global_round)
        else:
            # 固定学习率或无调度器
            current_lr = self.args.lr
        
        # 应用学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        print(f"[CLIENT {self.client_id}] 轮次 {global_round}: 学习率 = {current_lr:.6f}")

        for iter_epoch in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                if self.personalization_layer is not None:
                    self.personalization_layer.zero_grad()
                
                log_probs = model(images)
                
                # 检查输出是否包含NaN
                if torch.isnan(log_probs).any():
                    print(f"警告: 模型输出包含NaN值，跳过此批次")
                    continue
                
                # 计算标准交叉熵损失
                ce_loss = self.criterion(log_probs, labels)
                
                # 检查损失是否为NaN
                if torch.isnan(ce_loss):
                    print(f"警告: 交叉熵损失为NaN，跳过此批次")
                    continue
                
                # Non-IID场景下添加知识蒸馏策略
                total_loss = ce_loss
                if (hasattr(self.args, 'iid') and self.args.iid == 0 and global_round > 0 
                    and getattr(self.args, 'enable_knowledge_distillation', 1) == 1
                    and global_weights is not None):
                    try:
                        # --- 优化：直接使用全局模型作为教师模型（eval模式） ---
                        # 避免重复创建教师模型，提高效率
                        with torch.no_grad():
                            # 创建教师模型的简化版本，复用当前model结构
                            teacher_logits = None
                            
                            # 临时保存当前模型状态
                            current_state = {k: v.clone() for k, v in model.state_dict().items()}
                            
                            # 临时加载全局权重到当前模型
                            temp_global_weights = {k: v.to(self.device) for k, v in global_weights.items()}
                            model.load_state_dict(temp_global_weights)
                            model.eval()
                            
                            # 获取教师输出
                            teacher_logits = model(images)
                            
                            # 恢复学生模型状态
                            model.load_state_dict(current_state)
                            model.train()
                        
                        # 检查teacher输出是否有效
                        if teacher_logits is not None and not torch.isnan(teacher_logits).any() and not torch.isinf(teacher_logits).any():
                            # 计算知识蒸馏损失
                            T = getattr(self.args, 'distill_temperature', 3.0)
                            alpha = getattr(self.args, 'distill_alpha', 0.3)
                            
                            student_soft = torch.log_softmax(log_probs / T, dim=1)
                            teacher_soft = torch.softmax(teacher_logits / T, dim=1)
                            distill_loss = torch.nn.functional.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T * T)
                            
                            # 检查蒸馏损失是否有效
                            if not torch.isnan(distill_loss) and not torch.isinf(distill_loss):
                                total_loss = (1 - alpha) * ce_loss + alpha * distill_loss
                    except Exception as e:
                        print(f"知识蒸馏计算出错，使用标准损失: {str(e)}")
                        total_loss = ce_loss
                
                loss = total_loss
                
                # 最终检查损失是否为NaN
                if torch.isnan(loss):
                    print(f"警告: 最终损失为NaN，跳过此批次")
                    continue
                
                # --- MODIFICATION START: Conditionally apply FedProx ---
                # 只有在Non-IID场景下，并且mu>0时，才应用FedProx近端项
                if getattr(self.args, 'iid', 1) == 0 and getattr(self.args, 'mu', 0.0) > 0 and global_weights is not None:
                    prox_term = 0.0
                    # global_weights 是本轮开始时的全局模型权重
                    for name, param in model.named_parameters():
                        if name in global_weights:
                            # 确保全局参数张量在正确的设备上
                            # global_weights[name] 是全局模型的参数
                            # param 是当前客户端模型的参数
                            # 这里将全局参数移动到当前客户端参数所在的设备，并确保数据类型一致
                            global_param_tensor = global_weights[name].detach().to(param.device, dtype=param.dtype)
                            
                            # 计算当前客户端参数与全局参数之间的平方差
                            # torch.pow(param - global_param_tensor, 2) 计算每个参数的平方差
                            # torch.sum(...) 对所有参数的平方差求和，得到一个标量
                            prox_term += torch.sum(torch.pow(param - global_param_tensor, 2))
                    
                    # 将FedProx的近端项加入到总损失中
                    # self.args.mu 是FedProx的正则化强度超参数
                    # prox_term 是所有参数平方差的总和
                    # (self.args.mu / 2) * prox_term 是FedProx的正则化项
                    loss += (self.args.mu / 2) * prox_term
                # --- MODIFICATION END ---
                    
                loss.backward()
                
                # 添加梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    current_samples = (batch_idx + 1) * len(images)  # 已处理的样本数
                    total_samples = len(self.trainloader.dataset)
                    progress_percent = 100. * current_samples / total_samples  # 修复：基于样本数计算进度
                    print('| Global Round : {} | Client : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, self.client_id, iter_epoch, 
                        min(current_samples, total_samples),  # 确保不超过总样本数
                        total_samples,
                        min(progress_percent, 100.0),  # 确保不超过100%
                        loss.item()))
                # self.logger.add_scalar('loss', loss.item())  # 注释掉logger调用
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """推理函数，用于评估模型性能"""
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.testloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # 推理
                outputs = model(images)
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()

                # 预测
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

        accuracy = correct/total
        return accuracy, loss/len(self.testloader)

    def update_weights_memory_efficient(self, global_weights, global_round, device):
        """内存高效的权重更新方法"""
        from model_factory import get_model
        
        #创建模型
        model = get_model(self.args.dataset, self.args.model)  # 根据数据集和模型名称创建模型实例
        model.load_state_dict(global_weights)  # 加载全局模型权重
        model = model.to(device)  # 将模型移动到指定设备（如GPU或CPU）
        model.train()  # 设置模型为训练模式
        
        #在这里一次性创建好教师模型（优化后的实现）
        teacher_model = None  # 初始化教师模型为None
        # 增加一个 distill_warmup_rounds 参数，默认为3，用于控制知识蒸馏的预热轮次
        distill_warmup_rounds = getattr(self.args, 'distill_warmup_rounds', 3) 

        # 检查是否满足启用知识蒸馏的条件
        if (hasattr(self.args, 'iid') and self.args.iid == 0  # 确保是Non-IID场景
            and getattr(self.args, 'enable_knowledge_distillation', 1) == 1  # 确保启用了知识蒸馏
            and global_weights is not None  # 确保全局权重可用
            and global_round > distill_warmup_rounds):  # 确保当前轮次超过预热轮次

            print(f"[CLIENT {self.client_id}] 启用知识蒸馏 (轮次 > {distill_warmup_rounds})")
            
            # --- 优化：复用全局模型而非重新创建 ---
            # 创建教师模型的优化版本，直接基于当前全局权重
            teacher_model = get_model(self.args.dataset, self.args.model)  # 创建教师模型实例

            # 将全局权重移动到当前设备再加载，确保权重和模型在同一设备上
            temp_weights = {k: v.to(device) for k, v in global_weights.items()} #temp_weights是一个新的字典，存储了移动到目标设备上的全局权重
            teacher_model.load_state_dict(temp_weights)  # 加载全局权重到教师模型
            teacher_model = teacher_model.to(device)  # 将教师模型移动到指定设备
            teacher_model.eval()  # 设置教师模型为评估模式

            # 禁用梯度计算，节省显存和计算资源
            for param in teacher_model.parameters():
              param.requires_grad = False
        
        epoch_loss = []  # 初始化用于存储每轮训练损失的列表

        # 设置优化器
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=getattr(self.args, 'momentum', 0.5), 
                                        weight_decay=getattr(self.args, 'weight_decay', 1e-4))
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         betas=(getattr(self.args, 'adam_beta1', 0.9), 
                                               getattr(self.args, 'adam_beta2', 0.999)),
                                         eps=getattr(self.args, 'adam_eps', 1e-8),
                                         weight_decay=getattr(self.args, 'weight_decay', 1e-4))
        elif self.args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr,
                                          betas=(getattr(self.args, 'adam_beta1', 0.9), 
                                                getattr(self.args, 'adam_beta2', 0.999)),
                                          eps=getattr(self.args, 'adam_eps', 1e-8),
                                          weight_decay=getattr(self.args, 'weight_decay', 1e-4))

        # 优化的学习率调度策略
        lr_scheduler_type = getattr(self.args, 'lr_scheduler', 'none')
        current_lr = self.args.lr
        
        if lr_scheduler_type == 'cosine':
            # 改进的余弦退火学习率调整
            total_rounds = getattr(self.args, 'epochs', 50)  # 获取总训练轮次，默认为50
            min_lr = self.args.lr * 0.05  # 设置最小学习率为初始学习率的5%，用于后期微调
            
            # 使用warmup + cosine策略
            warmup_rounds = min(5, total_rounds // 10)  # 前10%的轮次用于warmup，最多5轮
            if global_round < warmup_rounds:
            # Warmup阶段：学习率从0线性增长到目标学习率
              current_lr = self.args.lr * (global_round + 1) / warmup_rounds
            else:
            # Cosine退火阶段：学习率按照余弦函数逐渐减小
              effective_round = global_round - warmup_rounds  # 当前轮次减去warmup轮次
              effective_total = total_rounds - warmup_rounds  # 总轮次减去warmup轮次
              cosine_factor = 0.5 * (1 + math.cos(math.pi * effective_round / effective_total))  # 计算余弦因子
              current_lr = min_lr + (self.args.lr - min_lr) * cosine_factor  # 根据余弦因子调整学习率
            
            # 针对Non-IID和GroupNorm的适应性调整
            # --- 核心修改：完全注释掉额外衰减，让cosine调度器全权负责 ---
            # if hasattr(self.args, 'iid') and self.args.iid == 0:
            #     # 方案 (推荐): 让衰减更温和
            #     current_lr *= 0.9  # <--- 将之前的衰减因子 (例如0.7) 调整为更平缓的 0.9
            # if hasattr(self.args, 'model') and 'gn' in str(self.args.model).lower():
            #     current_lr *= 0.9  # GroupNorm模型微调
            
            # 将计算出的学习率应用到优化器的参数组中
            for param_group in optimizer.param_groups:
              param_group['lr'] = current_lr
                
        print(f"[CLIENT {self.client_id}] 轮次 {global_round}: 学习率 = {current_lr:.6f}")

        # 本地训练
        for iter_epoch in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(device), labels.to(device)

                model.zero_grad()
                log_probs = model(images)
                
                # 检查输出是否包含NaN
                if torch.isnan(log_probs).any():
                    print(f"警告: 模型输出包含NaN值，跳过此批次")
                    continue
                
                # 计算标准交叉熵损失
                ce_loss = self.criterion(log_probs, labels)
                
                # 检查损失是否为NaN
                if torch.isnan(ce_loss):
                    print(f"警告: 交叉熵损失为NaN，跳过此批次")
                    continue
                
                # Non-IID场景下添加知识蒸馏策略
                total_loss = ce_loss
                if teacher_model is not None: # <--- 直接判断教师模型是否存在
                    try:
                        # 使用预先创建的教师模型进行知识蒸馏
                        with torch.no_grad():
                            teacher_logits = teacher_model(images) # <--- 直接复用，不再重新创建
                        
                        # 检查teacher输出是否有效
                        if not torch.isnan(teacher_logits).any() and not torch.isinf(teacher_logits).any():
                            # 计算知识蒸馏损失
                            T = getattr(self.args, 'distill_temperature', 3.0)
                            alpha = getattr(self.args, 'distill_alpha', 0.3)
                            
                            student_soft = torch.log_softmax(log_probs / T, dim=1)
                            teacher_soft = torch.softmax(teacher_logits / T, dim=1)
                            distill_loss = torch.nn.functional.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T * T)
                            
                            # 检查蒸馏损失是否有效
                            if not torch.isnan(distill_loss) and not torch.isinf(distill_loss):
                                total_loss = (1 - alpha) * ce_loss + alpha * distill_loss
                    except Exception as e:
                        print(f"知识蒸馏计算出错，使用标准损失: {str(e)}")
                        total_loss = ce_loss
                
                loss = total_loss
                
                # 最终检查损失是否为NaN
                if torch.isnan(loss):
                    print(f"警告: 最终损失为NaN,跳过此批次")
                    continue
                
                # 只有在Non-IID场景下，并且mu>0时，才应用FedProx近端项
                if getattr(self.args, 'iid', 1) == 0 and getattr(self.args, 'mu', 0.0) > 0 and global_weights is not None:
                    prox_term = 0.0
                    # global_weights 是本轮开始时的全局模型权重
                    for name, param in model.named_parameters():
                        if name in global_weights:
                            # 确保全局参数张量在正确的设备上
                            global_param_tensor = global_weights[name].detach().to(param.device)
                            prox_term += torch.sum(torch.pow(param - global_param_tensor, 2))
                    
                    loss += (self.args.mu / 2) * prox_term
                
                loss.backward()
                
                # 添加梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    current_samples = (batch_idx + 1) * len(images)  # 已处理的样本数
                    total_samples = len(self.trainloader.dataset)
                    progress_percent = 100. * current_samples / total_samples  # 修复：基于样本数计算进度
                    print('| Global Round : {} | Client : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, self.client_id, iter_epoch, 
                        min(current_samples, total_samples),  # 确保不超过总样本数
                        total_samples,
                        min(progress_percent, 100.0),  # 确保不超过100%
                        loss.item()))
                # self.logger.add_scalar('loss', loss.item())  # 注释掉logger调用
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # 获取更新后的权重（在CPU上）
        updated_weights = {k: v.cpu() for k, v in model.state_dict().items()}
            
        return updated_weights, sum(epoch_loss) / len(epoch_loss)

def model_subtract(dict1, dict2):
    """计算两个模型参数字典的差值（残差），确保设备一致性
    
    Args:
        dict1: 被减数参数字典（通常是更新后的本地权重）
        dict2: 减数参数字典（通常是全局模型权重）
        
    Returns:
        result: 差值字典（残差）
    """
    result = {}
    for key in dict1.keys():
        # 确保两个字典都包含该键
        if key not in dict2:
            continue  # 跳过dict2中不存在的键
            
        # 确保两个张量在同一设备上
        tensor1 = dict1[key]
        tensor2 = dict2[key]
        
        # 如果设备不同，将tensor2移动到tensor1的设备
        if tensor1.device != tensor2.device:
            tensor2 = tensor2.to(tensor1.device)
        
        result[key] = tensor1 - tensor2
    return result

def model_add(dict1, dict2):
    """将两个模型参数字典相加，确保设备一致性
    
    Args:
        dict1: 基础模型参数字典（通常是全局模型权重）
        dict2: 要添加的参数字典（通常是残差）
        
    Returns:
        result: 相加后的参数字典
    """
    result = {}
    for key in dict1.keys():
        # 如果残差字典中不包含该键（比如被过滤掉的归一化层参数），
        # 则保持原值不变
        if key not in dict2:
            result[key] = dict1[key].clone()
            continue
            
        # 确保两个张量在同一设备上
        tensor1 = dict1[key]
        tensor2 = dict2[key]
        
        # 如果设备不同，将tensor2移动到tensor1的设备
        if tensor1.device != tensor2.device:
            tensor2 = tensor2.to(tensor1.device)
        
        result[key] = tensor1 + tensor2
    return result

def average_weights_residual(w, template=None, enable_timing=False):
    """
    平均多个残差字典 (支持掩码+值格式和密集格式)。
    
    Args:
        w: 残差字典列表，可以是打包后的 (mask, values) 格式或密集张量格式。
        template (dict): 模板字典，提供目标形状、设备和数据类型（仅压缩模式需要）。
        enable_timing (bool): 是否启用时间统计
        
    Returns:
        w_avg: 平均后的残差字典（密集Tensor格式）。
        或 (w_avg, total_unpack_time): 如果启用时间统计，返回元组
    """
    if not w or len(w) == 0:
        return {}
    
    # 检查第一个残差字典中第一个元素的格式
    first_residual = w[0]
    if not first_residual:
        return {}
    
    first_key = next(iter(first_residual))
    first_value = first_residual[first_key]
    
    # 判断是新格式（packed）还是旧格式（dense）
    # 需要检查位打包格式或传统格式
    is_packed_format = (isinstance(first_value, dict) and 
                       (('mask' in first_value and 'values' in first_value) or 
                        ('mask_data' in first_value and 'is_bit_packed' in first_value)))
    
    if is_packed_format and template is not None:
        # 新格式：处理打包的 (mask, values) 数据，支持位级和传统格式
        w_avg = {}
        total_unpack_time = 0.0  # 统计解包时间
        unpack_count = 0
        
        # --- 新增代码：直接从模板解包所有残差 ---
        dense_residuals = []
        for res in w:
            unpack_result = unpack_sparse_residual(res, template=template, enable_timing=enable_timing)
            if enable_timing and isinstance(unpack_result, tuple):
                dense_res, unpack_time = unpack_result
                total_unpack_time += unpack_time
            else:
                dense_res = unpack_result
            dense_residuals.append(dense_res)
        # ----------------------------------------

        # --- 修改代码：使用解包后的 dense_residuals 进行平均 ---
        if not dense_residuals:
            return {}
            
        w_avg = copy.deepcopy(dense_residuals[0])
        for key in w_avg.keys():
            # 从第二个残差开始累加
            for i in range(1, len(dense_residuals)):
                # 因为全部在CPU上，无需检查设备
                w_avg[key] += dense_residuals[i][key]
            
            # 除以客户端数量得到平均值
            w_avg[key] = torch.div(w_avg[key], len(dense_residuals))
        # ----------------------------------------------------
        
        # 如果启用了时间统计，打印并返回时间信息
        if enable_timing and total_unpack_time > 0:
            print(f"   📦 服务器标准聚合解包耗时: {total_unpack_time:.2f}ms")
            return w_avg, total_unpack_time
            
        return w_avg
        

    elif template is None:
        # 全量模式：直接处理密集张量数据，无需解包
        print("   🚀 全量模式聚合：直接处理密集张量，跳过解包步骤")
        
        # 直接使用密集张量进行平均聚合
        w_avg = copy.deepcopy(w[0])
        
        for key in w_avg.keys():
            # 从第二个残差开始累加
            for i in range(1, len(w)):
                # 确保张量在同一设备上再相加
                if w[i][key].device != w_avg[key].device:
                    w[i][key] = w[i][key].to(w_avg[key].device)
                w_avg[key] = torch.add(w_avg[key], w[i][key])
            
            # 除以客户端数量得到平均值
            w_avg[key] = torch.div(w_avg[key], len(w))
        
        return w_avg
        
    else:
        # 旧格式：处理密集张量数据（向后兼容）
        dense_residuals = []
        for residual_dict in w:
            dense_w = {}
            for key, tensor in residual_dict.items():
                if hasattr(tensor, 'is_sparse') and tensor.is_sparse:
                    # PyTorch稀疏张量转密集
                    dense_w[key] = tensor.to_dense()
                else:
                    # 已经是密集张量
                    dense_w[key] = tensor
            dense_residuals.append(dense_w)
            
        # 使用密集张量进行平均聚合
        w_avg = copy.deepcopy(dense_residuals[0])
        
        for key in w_avg.keys():
            # 从第二个残差开始累加
            for i in range(1, len(dense_residuals)):
                # 确保设备一致性
                if w_avg[key].device != dense_residuals[i][key].device:
                    dense_residuals[i][key] = dense_residuals[i][key].to(w_avg[key].device)
                
                # 密集张量的加法
                w_avg[key] += dense_residuals[i][key]
            
            # 除以客户端数量得到平均值
            w_avg[key] = torch.div(w_avg[key], len(dense_residuals))
        
        return w_avg

def apply_residual_compression_fast(residual, compression_ratio=0.1):
    """
    优化的残差压缩函数，使用Top-K稀疏化减少通信开销
    
    Args:
        residual: 残差参数字典，包含每一层的参数更新值
        compression_ratio: 压缩比例 (0~1)，表示保留的参数比例，值越小压缩越强
    
    Returns:
        compressed_residual: 压缩后的残差字典，保留重要参数，其余置零
    """
    compressed_residual = {}  # 存储压缩后的残差

    for key, param in residual.items():
        # 如果参数不是浮点类型（例如整数类型），直接克隆保存，不进行压缩
        if not param.dtype.is_floating_point:
            compressed_residual[key] = param.clone()
            continue

        # 使用torch.no_grad()避免计算图的构建，减少内存和计算开销
        with torch.no_grad():
            # 调用优化的Top-K压缩函数，对当前层的参数进行稀疏化
            compressed_residual[key] = fast_topk_compression(param, compression_ratio)

    return compressed_residual

def fast_topk_compression(param, compression_ratio):
    """
    优化的Top-K压缩实现，用于稀疏化张量以减少通信开销。
    
    Args:
        param: 输入的张量（通常是模型的参数或残差）。
        compression_ratio: 压缩比例 (0~1)，表示保留的参数比例，值越小压缩越强。
    
    Returns:
        压缩后的张量，仅保留重要参数，其余置零。
    """
    # 将张量展平为一维向量，使用view代替flatten以提高效率
    flat_param = param.view(-1)
    
    # 计算需要保留的参数数量 k
    k = max(1, int(len(flat_param) * compression_ratio))
    
    # 如果保留的参数数量大于等于总参数数量，则无需压缩，直接返回原张量的副本
    if k >= len(flat_param):
        return param.clone()
    
    # 使用 kthvalue 方法计算阈值，比 topk 方法更高效
    try:
        # 找到第 (len(flat_param) - k) 小的绝对值作为阈值
        threshold = torch.kthvalue(torch.abs(flat_param), len(flat_param) - k)[0]
        
        # 创建掩码，标记大于等于阈值的元素
        mask = torch.abs(flat_param) >= threshold
        
        # 创建一个与原张量形状相同的零张量
        compressed = torch.zeros_like(flat_param)
        
        # 使用掩码保留重要参数，其余置零
        compressed[mask] = flat_param[mask]
        
        # 将压缩后的张量恢复为原始形状并返回
        return compressed.view(param.shape)
    except:
        # 如果 kthvalue 方法失败（例如在某些特殊情况下），回退到标准的 topk 方法
        # 使用 topk 找到绝对值最大的 k 个元素的索引
        _, top_k_indices = torch.topk(torch.abs(flat_param), k)
        
        # 创建一个与原张量形状相同的零张量
        compressed = torch.zeros_like(flat_param)
        
        # 根据索引保留重要参数，其余置零
        compressed[top_k_indices] = flat_param[top_k_indices]
        
        # 将压缩后的张量恢复为原始形状并返回
        return compressed.view(param.shape)

def adaptive_client_aggregation(local_residuals, client_data_sizes, client_losses, 
                               server_model_template, diversity_scores=None, 
                               aggregation_method='weighted_avg', enable_timing=False):
    #FedAC(Federated Averaging with Adaptive Client Weighting)
    """
    自适应客户端聚合策略
    根据数据量、损失和多样性动态调整聚合权重
    """
    if not local_residuals:
        return {}
    
    num_clients = len(local_residuals)
    '''
    关于local_residuals的示例:
    local_residuals = [
    {"layer1": torch.tensor([1.0, 2.0]), "layer2": torch.tensor([3.0, 4.0])},
    {"layer1": torch.tensor([0.5, 1.5]), "layer2": torch.tensor([2.5, 3.5])}
    ]
    '''
    
    # 计算基础权重（数据量）
    total_samples = sum(client_data_sizes)
    data_weights = [size / total_samples for size in client_data_sizes]
    
    # 处理NaN或无穷损失值
    safe_losses = []
    for loss in client_losses:
        #每个客户端本地训练后计算得到的平均损失
        if torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)):
            safe_losses.append(1.0)  # 使用默认损失值
        else:
            safe_losses.append(float(loss))
    
    if aggregation_method == 'weighted_avg':
        # 标准加权平均
        weights = data_weights
    elif aggregation_method == 'diversity_aware' and diversity_scores is not None:
        # 基于多样性的权重调整
        # 多样性越高，权重适当增加（但不过度）
        
        # 1. 先计算损失权重
        loss_weights = [1.0 / (1.0 + loss) for loss in safe_losses]
        total_loss_weight = sum(loss_weights)
        if total_loss_weight > 0:
            # 归一化损失权重
            loss_weights = [w / total_loss_weight for w in loss_weights]
        else:
            # 如果总损失权重为0，均匀分配权重
            loss_weights = [1.0 / num_clients] * num_clients

        # 2. 计算多样性权重
        # 多样性越高，权重适当增加（但不过度）
        div_weights = [min(1.5, 1.0 + 0.3 * score) for score in diversity_scores]
        # 注意：这里的 div_weights 是一个"调整因子"，而不是一个归一化的权重

        # 3. 结合数据量、损失和多样性权重
        # 基础权重结合数据量和损失，多样性作为乘法调整项
        base_weights = [0.6 * dw + 0.4 * lw for dw, lw in zip(data_weights, loss_weights)]

        # 用多样性因子来调整基础权重
        weights = [bw * divw for bw, divw in zip(base_weights, div_weights)]
    else:
        # 默认使用数据量权重
        weights = data_weights
    
    # 归一化权重并确保数值稳定性
    total_weight = sum(weights)
    if total_weight > 0:
        # 归一化权重
        weights = [w / total_weight for w in weights]
    else:
        # 如果总权重为0，均匀分配权重
        weights = [1.0 / num_clients] * num_clients
    
    # 确保权重范围合理
    weights = [max(min(w, 1.0), 0.0) for w in weights]
    
    # 再次归一化权重
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    else:
        weights = [1.0 / num_clients] * num_clients
    '''
    虽然第一次归一化理论上应该能确保权重在合理范围内
    但在复杂的计算中，可能会因为浮点数精度问题或某些极端情况（例如某个权重因多样性奖励乘数变得略大于1）
    导致个别权重值出现微小的偏差，比如变成 -0.0000001 或者 1.0000001
    这一步 max(min(w, 1.0), 0.0) 会非常严格地将所有这些意外值“裁剪”回 [0.0, 1.0] 的区间内
    '''

    # 执行加权聚合
    aggregated_residual = {}  # 初始化聚合后的残差字典
    first_residual = local_residuals[0]  # 获取第一个客户端的残差作为模板
    total_unpack_time = 0.0  # 初始化计时器，用于统计解包时间
    
    # 检查是否为打包格式的数据
    first_value = next(iter(first_residual.values())) if first_residual else None
    is_packed_format = (isinstance(first_value, dict) and 
                       (('mask' in first_value and 'values' in first_value) or 
                        ('mask_data' in first_value and 'is_bit_packed' in first_value)))
    
    if is_packed_format and server_model_template is not None:
        # isinstance检查第一个值是字典还是张量,对应着是打包残差字典还是没有打包的张量残差字典
        # 打包格式：直接使用传入的模板进行解包
        
        # 解包所有客户端的残差（使用CPU模板）
        dense_residuals = []  # 用于存储解包后的密集格式残差
        for residual in local_residuals:
            unpack_result = unpack_sparse_residual(residual, 
                                                   template=server_model_template, 
                                                   enable_timing=enable_timing)
            
            if enable_timing and isinstance(unpack_result, tuple):
                dense_residual, unpack_time = unpack_result
                total_unpack_time += unpack_time
            else:
                dense_residual = unpack_result
            
            dense_residuals.append(dense_residual)
        
        # 使用密集格式进行加权聚合（在CPU上完成）
        first_tensor_dict = dense_residuals[0]  # 获取第一个客户端的密集残差作为模板
        for key in first_tensor_dict.keys():
            first_tensor = first_tensor_dict[key]
            
            # 初始化当前层的聚合张量（在CPU上）
            aggregated_residual[key] = torch.zeros_like(first_tensor, 
                                                       dtype=first_tensor.dtype,
                                                       device=first_tensor.device)  # 应该在CPU上
            
            # 遍历每个客户端的残差，进行加权累加
            for i, residual in enumerate(dense_residuals):
                # 检查当前客户端的残差是否包含NaN或Inf值
                if torch.isnan(residual[key]).any() or torch.isinf(residual[key]).any():
                    print(f"警告: 客户端 {i} 的残差包含NaN/Inf值，跳过此客户端")
                    continue
                
                # 将权重转换为张量，确保类型和设备一致（CPU上）
                weight_scalar = torch.tensor(float(weights[i]), dtype=torch.float32, 
                                           device=residual[key].device)  # 应该在CPU上
                
                if residual[key].dtype != torch.float32:
                    weight_scalar = weight_scalar.to(residual[key].dtype)
                
                # 加权累加残差（CPU上的计算）
                aggregated_residual[key] += weight_scalar * residual[key]
    
    else:
        # --- 新增代码块：处理全量(dense)模式 ---
        # local_residuals 已经是 [dense_dict_1, dense_dict_2, ...]
        
        first_tensor_dict = local_residuals[0]
        for key in first_tensor_dict.keys():
            
            first_tensor = first_tensor_dict[key]
            
            # 初始化当前层的聚合张量
            aggregated_residual[key] = torch.zeros_like(first_tensor, 
                                                       dtype=first_tensor.dtype,
                                                       device=first_tensor.device)
            
            # 遍历每个客户端的残差，进行加权累加
            for i, residual in enumerate(local_residuals):
                # 检查当前客户端的残差是否包含NaN或Inf值
                if key not in residual or torch.isnan(residual[key]).any() or torch.isinf(residual[key]).any():
                    print(f"警告: 客户端 {i} 的残差 {key} 包含无效值，跳过此客户端")
                    continue
                
                # 确保设备一致
                if residual[key].device != aggregated_residual[key].device:
                    residual[key] = residual[key].to(aggregated_residual[key].device)
                
                # 将权重转换为张量，确保类型和设备一致
                weight_scalar = torch.tensor(float(weights[i]), dtype=torch.float32, 
                                           device=residual[key].device)
                
                if residual[key].dtype != torch.float32:
                    weight_scalar = weight_scalar.to(residual[key].dtype)
                
                # 加权累加残差
                aggregated_residual[key] += weight_scalar * residual[key]
        # --- 新增代码块结束 ---

    # 在函数末尾返回计时结果
    if enable_timing:
        print(f"   📦 服务器自适应聚合解包耗时: {total_unpack_time:.2f}ms")
        return aggregated_residual, total_unpack_time
    else:
        return aggregated_residual

def pack_bool_mask_to_bits(bool_mask):
    """
    位级打包函数
    将布尔掩码打包为位级存储,8个布尔值打包为1个字节,节省87.5%存储空间
    
    Args:
        bool_mask: torch.BoolTensor,布尔掩码
        
    Returns:
        tuple: (packed_bytes, original_shape, num_bits)
            - packed_bytes: np.ndarray (uint8)，打包后的字节数组  
            - original_shape: tuple,原始掩码的形状
            - num_bits: int,原始掩码的总位数,用于解压缩时截取有效位
    """
    # 获取布尔掩码的原始形状，用于解压缩时恢复形状
    original_shape = bool_mask.shape
    
    # 将布尔掩码展平为一维数组，并转换为NumPy数组
    # .flatten() 将张量展平为一维
    # .cpu() 将张量移动到CPU上（如果在GPU上）
    # .numpy() 转换为NumPy数组
    # .astype(np.uint8) 将布尔值转换为无符号8位整数（0或1）
    flat_mask = bool_mask.flatten().cpu().numpy().astype(np.uint8)
    
    # 计算布尔掩码的总位数（即展平后的一维数组长度）
    num_bits = len(flat_mask)
    
    # 使用NumPy的packbits函数将布尔值打包为字节
    # 每8个布尔值（0或1）打包为1个字节，实现8:1的压缩
    packed_bytes = np.packbits(flat_mask)
    
    # 返回打包后的字节数组、原始形状和总位数
    return packed_bytes, original_shape, num_bits

def unpack_bits_to_bool_mask(packed_bytes, original_shape, num_bits, device='cpu'):
    """
    位级解包函数
    
    Args:
        packed_bytes: np.ndarray (uint8)，打包后的字节数组
        original_shape: tuple,目标掩码形状
        num_bits: int,有效位数
        device: str,目标设备
        
    Returns:
        torch.BoolTensor,恢复的布尔掩码
    """
    # 解包字节为位
    unpacked_bits = np.unpackbits(packed_bytes)
    
    # 截取实际需要的位数（因为最后一个字节可能有填充）
    unpacked_bits = unpacked_bits[:num_bits]
    
    # 转换为布尔张量并恢复形状
    bool_mask = torch.from_numpy(unpacked_bits.astype(bool)).reshape(original_shape)
    
    return bool_mask.to(device)

def pack_sparse_residual(compressed_residual, enable_timing=True, use_bit_packing=True):
    """
    将稀疏残差Tensor打包成 (mask, values) 的格式，以便高效传输。
    支持位级掩码打包

    Args:
        compressed_residual (dict): 经过Top-K压缩的残差字典,值是包含大量0的Tensor。
        enable_timing (bool): 是否启用计时功能
        use_bit_packing (bool): 是否使用位级掩码打包(默认启用)

    Returns:
        dict: 打包后的残差字典。
              格式为: { 'layer_name': {'mask': packed_data, 'values': torch.FloatTensor}, ... }
    """
    import time
    start_time = time.time() if enable_timing else None
    
    packed_residual = {}
    
    for key, sparse_tensor in compressed_residual.items():
        # 只处理浮点类型的参数
        if not sparse_tensor.dtype.is_floating_point:
            continue
        
        # 1. 创建 mask,形状与 sparse_tensor 完全相同,每个位置的值表示 sparse_tensor 中对应位置是否为非零
        mask = (sparse_tensor != 0)
        
        # 2. 提取非零 values
        values = sparse_tensor[mask]
        #这里去除所有的0值，values是一个一维张量，包含所有非零元素
        
        # 3. 如果确实有非零值，则打包
        if values.numel() > 0:
            #values.numel()返回张量中元素的总数
            if use_bit_packing:
                #使用位级打包
                packed_mask, original_shape, num_bits = pack_bool_mask_to_bits(mask)
                
                packed_residual[key] = {
                    'mask_data': packed_mask,           # 打包的字节数组
                    'mask_shape': original_shape,       # 原始掩码形状
                    'mask_bits': num_bits,              # 有效位数
                    'values': values.cpu(),             # 非零值数组
                    'is_bit_packed': True               # 标识使用了位打包
                }
            else:
                # 传统方式，每个布尔值占1字节
                packed_residual[key] = {
                    'mask': mask.cpu(),                 # 布尔掩码
                    'values': values.cpu(),             # 非零值数组
                    'is_bit_packed': False              # 标识未使用位打包
                }
    
    if enable_timing and start_time is not None:
        pack_time = (time.time() - start_time) * 1000  # 转换为毫秒
        total_params = sum(torch.numel(tensor) for tensor in compressed_residual.values())
        
        return packed_residual, pack_time
    
    return packed_residual

def unpack_sparse_residual(packed_residual, template, enable_timing=False):
    """
    将打包后的稀疏残差数据解包成原来的格式。
    支持位级掩码解包，兼容传统布尔掩码格式。

    Args:
        packed_residual (dict): 已打包的残差字典。支持位级和传统格式
        template (dict): 模板字典，提供原始张量的形状和设备信息
        enable_timing (bool): 是否启用计时功能

    Returns:
        dict: 解包后的残差字典，格式与原始残差相同。
    """
    import time
    start_time = time.time() if enable_timing else None
    
    unpacked_residual = {}
    #首先确保所有参数都被初始化为零
    for key, template_tensor in template.items():
        unpacked_residual[key] = torch.zeros_like(template_tensor)
    
    # 然后只更新那些在 packed_residual 中存在的参数
    for key, packed_data in packed_residual.items():
        if key in template:
            target_device = template[key].device
            
            # 首先检查 packed_data 是否是字典类型
            if isinstance(packed_data, dict):
                # 检查是否使用了位级打包
                if packed_data.get('is_bit_packed', False):
                    # 位级解包
                    mask = unpack_bits_to_bool_mask(
                        packed_data['mask_data'],
                        packed_data['mask_shape'],
                        packed_data['mask_bits'],
                        device=target_device
                    )
                    values = packed_data['values'].to(target_device)
                else:
                    # 传统解包（向后兼容）
                    if 'mask' in packed_data:
                        mask = packed_data['mask'].to(target_device)
                        values = packed_data['values'].to(target_device)
                    else:
                        # 兼容更旧的格式
                        continue
                
                # 重构原始张量
                unpacked_residual[key][mask] = values
            else:
                # packed_data 是 Tensor，说明是密集格式，直接复制
                # 这种情况实际上不应该出现在 unpack_sparse_residual 中
                # 但为了安全起见，我们直接复制值
                unpacked_residual[key] = packed_data.to(target_device)
    
    if enable_timing and start_time is not None:
        unpack_time = (time.time() - start_time) * 1000  # 转换为毫秒
        total_params = sum(torch.numel(tensor) for tensor in unpacked_residual.values())
        print(f"   📦 解包耗时: {unpack_time:.2f}ms (重建 {total_params:,} 参数)")
        return unpacked_residual, unpack_time
    
    return unpacked_residual

def calculate_communication_cost_dict(packed_residual):
    """
    计算 (mask, values) 方式的通信成本，支持位级掩码优化。

    Args:
        packed_residual (dict): 打包后的残差字典，支持位级和传统格式
        
    Returns:
        tuple: (comm_cost, client_transmitted_bytes, layer_details)
    """
    client_transmitted_bytes = 0
    layer_details = []
    
    # 处理 (mask, values) 数据
    for key, data in packed_residual.items():
            # 检查是否使用位级打包
            if data.get('is_bit_packed', False):
                # 位级打包格式
                mask_data = data['mask_data']
                mask_shape = data['mask_shape']
                mask_bits = data['mask_bits']
                values = data['values']
                
                total_params = mask_bits  # 总参数数量等于位数
                nonzero_params = torch.numel(values)
                
                # 位级掩码的字节数（已经是实际传输的字节数）
                mask_bytes = len(mask_data)  # numpy数组的实际字节数
                
                # 根据实际数据类型计算非零值的字节数
                dtype_size = values.element_size()
                values_bytes = nonzero_params * dtype_size
                
                layer_bytes = mask_bytes + values_bytes
                client_transmitted_bytes += layer_bytes
                
                sparsity = (total_params - nonzero_params) / total_params * 100 if total_params > 0 else 0
                
                layer_details.append({
                    'layer_name': key,
                    'total_params': total_params,
                    'nonzero_params': nonzero_params,
                    'sparsity': sparsity,
                    'transmitted_bytes': layer_bytes,
                    'mask_bytes': mask_bytes,
                    'values_bytes': values_bytes,
                    'bit_packed': True  # 标记使用了位级优化
                })
                
            elif 'mask' in data and 'values' in data:
                # 传统格式（向后兼容）
                mask = data['mask']
                values = data['values']
                
                total_params = torch.numel(mask)
                nonzero_params = torch.numel(values)
                
                # 传统方式：每个布尔值1字节
                mask_bytes = total_params * mask.element_size()  # 通常是1字节
                
                # 根据实际数据类型计算非零值的字节数
                dtype_size = values.element_size()
                values_bytes = nonzero_params * dtype_size
                
                layer_bytes = mask_bytes + values_bytes
                client_transmitted_bytes += layer_bytes
                
                sparsity = (total_params - nonzero_params) / total_params * 100 if total_params > 0 else 0
                
                layer_details.append({
                    'layer_name': key,
                    'total_params': total_params,
                    'nonzero_params': nonzero_params,
                    'sparsity': sparsity,
                    'transmitted_bytes': layer_bytes,
                    'mask_bytes': mask_bytes,
                    'values_bytes': values_bytes,
                    'bit_packed': False  # 标记未使用位级优化
                })
    
    # 为了与之前代码兼容，我们返回一个等效的"参数量"
    # 这里我们定义1个"参数单位"= 4字节 (float32)
    comm_cost_in_params = client_transmitted_bytes / 4.0
    
    return comm_cost_in_params, client_transmitted_bytes, layer_details

def print_round_communication_stats(enable_compression, epoch_comm_cost, total_nonzero_values, 
                                   total_original_params, num_selected_clients, 
                                   single_model_params, current_compression_ratio):
    """
    打印轮次通信量统计信息 - 简化版
    
    Args:
        enable_compression (bool): 是否启用压缩
        epoch_comm_cost (int): 轮次通信成本
        total_nonzero_values (int): 总非零参数数量
        total_original_params (int): 总原始参数数量
        num_selected_clients (int): 选中的客户端数量
        single_model_params (int): 单个模型参数数量
        current_compression_ratio (float): 当前压缩率
    """
    if enable_compression:
        # 计算压缩效果统计
        total_baseline_params = single_model_params * num_selected_clients
        compression_effectiveness = (1 - epoch_comm_cost / total_baseline_params) * 100
        
        # 字节级统计
        total_param_bytes = total_nonzero_values * 4     # float32参数值字节数
        baseline_bytes = total_baseline_params * 4  # 基准字节数
        bytes_compression_effectiveness = (1 - total_param_bytes / baseline_bytes) * 100
        
        print(f"   📡 双向uniform压缩通信量:")
        print(f"      • 传递参数总数: {total_nonzero_values:,} 参数 ({total_param_bytes:,} 字节)")
        print(f"   🗜️ 压缩效果: 减少 {compression_effectiveness:.1f}% 通信量")
        print(f"   📊 基准通信量: {total_baseline_params:,} 参数 = {baseline_bytes:,} 字节")
        
        # 实际压缩率分析
        actual_compression_ratio = total_nonzero_values / total_baseline_params
        print(f"   🔍 实际压缩率: {actual_compression_ratio:.3f} (设定: {current_compression_ratio:.3f})")
    else:
        dense_bytes = epoch_comm_cost * 4  # 密集传输
        print(f"   📡 密集通信量: {epoch_comm_cost:,} 参数 = {dense_bytes:,} 字节")


def print_final_compression_stats(enable_compression, total_comm_cost, total_rounds, 
                                   single_model_params, avg_selected_clients, compression_ratio):
    """
    打印最终的双向uniform压缩统计信息 - 简化版
    """
    print("\n📊 最终压缩统计:")
    if enable_compression:
        theoretical_baseline_total = int(single_model_params * avg_selected_clients * total_rounds)
        compression_effectiveness = (1 - total_comm_cost / theoretical_baseline_total) * 100
        actual_compression_ratio = total_comm_cost / theoretical_baseline_total

        print(f"   • 理论基准通信量: {theoretical_baseline_total:,} 参数")
        print(f"   • 实际总通信量: {total_comm_cost:,} 参数")
        print(f"   • 双向uniform压缩效果: 减少 {compression_effectiveness:.2f}% 通信量")
        print(f"   • 实际平均压缩率: {actual_compression_ratio:.3f}")
        print(f"   • 设定双向压缩率: {compression_ratio:.3f}")
    else:
        print(f"   • 总通信量: {total_comm_cost:,} 参数 (未启用压缩)")