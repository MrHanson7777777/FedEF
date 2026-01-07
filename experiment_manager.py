#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 实验管理器 - 统一管理基线、联邦学习和残差联邦学习实验

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime


# 设置输出编码
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


class ExperimentManager:
    """实验管理器类"""
    def __init__(self):
        self.available_datasets = ['mnist', 'cifar', 'cifar100', 'fmnist']
        self.available_methods = ['baseline', 'federated', 'residual']
        self.available_models = {
            'mnist': ['cnn', 'optimized', 'optimized_gn', 'mlp'],
            'cifar': ['cnn', 'resnet18', 'resnet18_fed', 'efficientnet'],
            'cifar100': ['cnn', 'resnet18_fed', 'efficientnet', 'densenet'],
            'fmnist': ['cnn']
        }
        self.method_files = {
            'baseline': 'baseline_main.py',
            'federated': 'federated_main.py', 
            'residual': 'residual_main.py'
        }
    
    def parse_args(self):
        """解析命令行参数"""
        parser = argparse.ArgumentParser(description='联邦学习实验管理器')
        
        # 基本参数
        parser.add_argument('--dataset', type=str, choices=self.available_datasets,
                          help='数据集选择: mnist, cifar, cifar100, fmnist')
        parser.add_argument('--method', type=str, choices=self.available_methods,
                          help='方法选择: baseline(基线), federated(普通联邦学习), residual(残差联邦学习)')
        parser.add_argument('--model', type=str,
                          help='模型选择，根据数据集而定')
        parser.add_argument('--epochs', type=int, default=50,
                          help='训练轮数 (默认: 50)')
        parser.add_argument('--pretrain_epochs', type=int, default=10,
                          help='预训练轮数 (默认: 10)')
        
        # 联邦学习参数
        parser.add_argument('--num_users', type=int, default=100,
                          help='客户端数量 (默认: 100)')
        parser.add_argument('--frac', type=float, default=0.1,
                          help='每轮参与的客户端比例 (默认: 0.1)')
        parser.add_argument('--local_ep', type=int, default=10,
                          help='本地训练轮数 (默认: 10)')
        parser.add_argument('--local_bs', type=int, default=10,
                          help='本地批量大小 (默认: 10)')
        parser.add_argument('--lr', type=float, default=0.01,
                          help='学习率 (默认: 0.01)')
        
        # 数据分布参数
        parser.add_argument('--iid', type=int, default=1,
                          help='数据分布: 1=IID, 0=Non-IID (默认: 1)')
        parser.add_argument('--unequal', type=int, default=0,
                          help='数据分布不均衡: 1=不均衡, 0=均衡 (默认: 0)')
        parser.add_argument('--alpha', type=float, default=0.5,
                          help='Dirichlet分布参数: 控制Non-IID程度，越小越不均衡 (默认: 0.5)')
        
        # 其他参数
        parser.add_argument('--gpu', type=str, default=None,
                          help='GPU设备ID (例如: 0)')
        parser.add_argument('--optimizer', type=str, default='sgd',
                          choices=['sgd', 'adam', 'adamw'], help='优化器选择')
        parser.add_argument('--momentum', type=float, default=0.9,
                          help='SGD动量参数 (默认: 0.9)')
        parser.add_argument('--lr_scheduler', type=str, default='none',
                          choices=['none', 'fixed', 'step', 'exp', 'cosine'],
                          help='学习率调度器选择')
        parser.add_argument('--cosine_t_max', type=int, default=200,
                          help='余弦退火调度器的最大迭代数 (默认: 200)')
        parser.add_argument('--weight_decay', type=float, default=1e-4,
                          help='权重衰减 (L2正则化)')
        parser.add_argument('--verbose', type=int, default=1,
                          help='详细输出 (默认: 1)')
        
        # 早停机制参数
        parser.add_argument('--stopping_rounds', type=int, default=10,
                          help='早停机制的耐心轮数 (默认: 10)')
        
        # 智能压缩参数
        parser.add_argument('--smart_compression', type=int, default=0,
                          help='启用智能分层压缩: 1=启用, 0=禁用 (默认: 0)')
        
        # 交互模式
        parser.add_argument('--interactive', action='store_true',
                          help='启用交互式配置模式')
        
        # 批量运行
        parser.add_argument('--batch_run', action='store_true',
                          help='批量运行多个配置')
        
        # 压缩相关参数 (仅残差联邦学习使用)
        parser.add_argument('--compression', type=str, default='none',
                          choices=['none', 'smart', 'uniform'],
                          help='压缩类型: none(无压缩), smart(智能分层压缩), uniform(统一压缩)')
        parser.add_argument('--compression_ratio', type=float, default=0.1,
                          help='压缩比例 (默认: 0.1)')
        parser.add_argument('--disable_compression', action='store_true',
                          help='禁用压缩 (仅残差联邦学习)')
        
        # CIFAR-100优化参数
        parser.add_argument('--use_attention', type=int, default=1,
                          help='使用注意力模块 (DenseNet): 1=启用, 0=禁用 (默认: 1)')
        parser.add_argument('--use_groupnorm', type=int, default=1,
                          help='使用GroupNorm替代BatchNorm: 1=启用, 0=禁用 (默认: 1)')
        parser.add_argument('--num_groups', type=int, default=8,
                          help='GroupNorm分组数 (默认: 8)')
        parser.add_argument('--criterion', type=str, default='cross_entropy',
                          choices=['cross_entropy', 'label_smoothing'],
                          help='损失函数类型 (默认: cross_entropy)')
        parser.add_argument('--smoothing', type=float, default=0.1,
                          help='标签平滑参数 (默认: 0.1)')
        parser.add_argument('--enable_cutmix', type=int, default=0,
                          help='启用CutMix数据增强: 1=启用, 0=禁用 (默认: 0)')
        parser.add_argument('--cutmix_alpha', type=float, default=1.0,
                          help='CutMix的alpha参数 (默认: 1.0)')
        parser.add_argument('--cutmix_prob', type=float, default=0.5,
                          help='CutMix应用概率 (默认: 0.5)')
        parser.add_argument('--enable_mixup', type=int, default=0,
                          help='启用Mixup数据增强: 1=启用, 0=禁用 (默认: 0)')
        parser.add_argument('--mixup_alpha', type=float, default=0.4,
                          help='Mixup的alpha参数 (默认: 0.4)')
        parser.add_argument('--enable_swa', type=int, default=0,
                          help='启用随机权重平均: 1=启用, 0=禁用 (默认: 0)')
        parser.add_argument('--swa_start', type=int, default=200,
                          help='SWA开始的轮数 (默认: 200)')
        
        # Non-IID优化策略参数
        parser.add_argument('--enable_knowledge_distillation', type=int, default=1,
                          help='启用知识蒸馏: 1=启用, 0=禁用 (默认: 1)')
        parser.add_argument('--distill_temperature', type=float, default=3.0,
                          help='知识蒸馏温度参数 (默认: 3.0)')
        parser.add_argument('--distill_alpha', type=float, default=0.3,
                          help='蒸馏损失权重 (默认: 0.3)')
        parser.add_argument('--adaptive_aggregation', type=str, default='loss_aware',
                          choices=['weighted_avg', 'loss_aware', 'diversity_aware'],
                          help='Non-IID场景下的聚合策略 (默认: loss_aware)')
        parser.add_argument('--mu', type=float, default=0.01,
                          help='FedProx正则化参数 (0表示标准FedAvg) (默认: 0.01)')
        
        # 显示选项
        parser.add_argument('--realtime', action='store_true',
                          help='显示实时训练过程输出')
        parser.add_argument('--silent', action='store_true',
                          help='静默模式，只显示最终结果')
        
        return parser.parse_args()
    
    def interactive_config(self):
        """交互式配置模式"""
        print("=== 联邦学习实验交互式配置 ===\n")
        
        # 选择数据集
        print("可用数据集:")
        for i, dataset in enumerate(self.available_datasets):
            print(f"  {i+1}. {dataset}")
        while True:
            try:
                choice = int(input("请选择数据集 (输入数字): ")) - 1
                if 0 <= choice < len(self.available_datasets):
                    dataset = self.available_datasets[choice]
                    break
                else:
                    print("无效选择，请重新输入")
            except ValueError:
                print("请输入有效数字")
        
        # 选择方法
        print(f"\n可用方法:")
        method_descriptions = {
            'baseline': '基线方法(中心化训练)',
            'federated': '标准联邦学习(FedAvg)',
            'residual': '残差联邦学习(Residual FL)'
        }
        for i, method in enumerate(self.available_methods):
            print(f"  {i+1}. {method} - {method_descriptions[method]}")
        while True:
            try:
                choice = int(input("请选择方法 (输入数字): ")) - 1
                if 0 <= choice < len(self.available_methods):
                    method = self.available_methods[choice]
                    break
                else:
                    print("无效选择，请重新输入")
            except ValueError:
                print("请输入有效数字")
        
        # 选择模型
        available_models = self.available_models.get(dataset, ['cnn'])
        print(f"\n{dataset}数据集可用模型:")
        model_descriptions = {
            'cnn': '标准CNN模型',
            'optimized': '优化CNN模型', 
            'resnet18': 'ResNet18模型',
            'resnet18_fed': '联邦优化ResNet18',
            'efficientnet': 'EfficientNet模型',
            'densenet': 'DenseNet模型',
            'mlp': '多层感知机'
        }
        for i, model in enumerate(available_models):
            desc = model_descriptions.get(model, '自定义模型')
            print(f"  {i+1}. {model} - {desc}")
        while True:
            try:
                choice = int(input("请选择模型 (输入数字): ")) - 1
                if 0 <= choice < len(available_models):
                    model = available_models[choice]
                    break
                else:
                    print("无效选择，请重新输入")
            except ValueError:
                print("请输入有效数字")
        
        # 其他参数
        epochs = int(input(f"\n训练轮数 (默认50): ") or "50")
        
        if method != 'baseline':
            num_users = int(input("客户端数量 (默认100): ") or "100")
            frac = float(input("参与比例 (默认0.1): ") or "0.1")
            local_ep = int(input("本地训练轮数 (默认10): ") or "10")
            local_bs = int(input("本地批量大小 (默认10): ") or "10")
            iid_choice = input("数据分布 IID? (y/n, 默认y): ").lower()
            iid = 1 if iid_choice != 'n' else 0
        else:
            num_users = 1
            frac = 1.0
            local_ep = epochs
            local_bs = 32
            iid = 1
        
        lr = float(input("学习率 (默认0.01): ") or "0.01")
        gpu = input("GPU设备ID (留空使用CPU): ") or None
        
        return {
            'dataset': dataset,
            'method': method, 
            'model': model,
            'epochs': epochs,
            'num_users': num_users,
            'frac': frac,
            'local_ep': local_ep,
            'local_bs': local_bs,
            'lr': lr,
            'iid': iid,
            'gpu': gpu
        }
    
    def _get_log_path(self, config):
        """根据实验配置生成预期的日志文件路径"""
        log_dir = './save/logs'
        method = config['method']
        iid_str = 'iid' if config.get('iid') else 'noniid'

        # 为 baseline, federated, residual 分别构建文件名
        if method == 'baseline':
            filename = f"log_baseline_{config['dataset']}_{config['model']}_{config['epochs']}ep_{iid_str}.csv"
        
        elif method == 'federated':
            filename = f"log_federated_{config['dataset']}_{config['model']}_{config['epochs']}ep_{iid_str}.csv"
        
        elif method == 'residual':
            compression_type = config.get('compression', 'none')
            enable_compression = compression_type != 'none' or config.get('enable_compression', False)
            
            comp_str = 'none'
            if enable_compression:
                if compression_type == 'smart' or config.get('smart_compression') == 1:
                    comp_str = 'smart'
                else:
                    comp_str = 'uniform'
            
            filename = f"log_residual_{config['dataset']}_{config['model']}_{config['epochs']}ep_{iid_str}_comp_{comp_str}.csv"
        
        else:
            return None # 如果方法未知，则返回None

        return os.path.join(log_dir, filename)
    
    def build_command(self, config):
        """构建执行命令"""
        script_file = self.method_files[config['method']]
        
        cmd = [sys.executable, script_file]
        cmd.extend(['--dataset', config['dataset']])
        cmd.extend(['--model', config['model']])
        cmd.extend(['--epochs', str(config['epochs'])])
        if config.get('pretrain_epochs'):
            cmd.extend(['--pretrain_epochs', str(config['pretrain_epochs'])])
        cmd.extend(['--num_users', str(config['num_users'])])
        cmd.extend(['--frac', str(config['frac'])])
        cmd.extend(['--local_ep', str(config['local_ep'])])
        cmd.extend(['--local_bs', str(config['local_bs'])])
        cmd.extend(['--lr', str(config['lr'])])
        cmd.extend(['--iid', str(config['iid'])])
        if config.get('alpha') is not None:
            cmd.extend(['--alpha', str(config['alpha'])])
        
        if config.get('gpu'):
            cmd.extend(['--gpu', str(config['gpu'])])
        
        cmd.extend(['--verbose', '1'])
        cmd.extend(['--optimizer', config.get('optimizer', 'sgd')])
        
        # 添加学习率调度器和优化器参数
        if config.get('lr_scheduler', 'none') != 'none':
            cmd.extend(['--lr_scheduler', config['lr_scheduler']])
        if config.get('momentum'):
            cmd.extend(['--momentum', str(config['momentum'])])
        if config.get('cosine_t_max'):
            cmd.extend(['--cosine_t_max', str(config['cosine_t_max'])])
        if config.get('weight_decay'):
            cmd.extend(['--weight_decay', str(config['weight_decay'])])
        
        # 添加CIFAR-100优化参数
        if config.get('use_attention') is not None:
            cmd.extend(['--use_attention', str(config['use_attention'])])
        if config.get('use_groupnorm') is not None:
            cmd.extend(['--use_groupnorm', str(config['use_groupnorm'])])
        if config.get('num_groups'):
            cmd.extend(['--num_groups', str(config['num_groups'])])
        if config.get('criterion'):
            cmd.extend(['--criterion', config['criterion']])
        if config.get('smoothing'):
            cmd.extend(['--smoothing', str(config['smoothing'])])
        if config.get('enable_cutmix') is not None:
            cmd.extend(['--enable_cutmix', str(config['enable_cutmix'])])
        if config.get('cutmix_alpha'):
            cmd.extend(['--cutmix_alpha', str(config['cutmix_alpha'])])
        if config.get('cutmix_prob'):
            cmd.extend(['--cutmix_prob', str(config['cutmix_prob'])])
        if config.get('enable_mixup') is not None:
            cmd.extend(['--enable_mixup', str(config['enable_mixup'])])
        if config.get('mixup_alpha'):
            cmd.extend(['--mixup_alpha', str(config['mixup_alpha'])])
        if config.get('enable_swa') is not None:
            cmd.extend(['--enable_swa', str(config['enable_swa'])])
        if config.get('swa_start'):
            cmd.extend(['--swa_start', str(config['swa_start'])])
        
        # 添加早停机制参数
        if config.get('stopping_rounds'):
            cmd.extend(['--stopping_rounds', str(config['stopping_rounds'])])
        
        # 为残差联邦学习添加专属参数
        if config['method'] == 'residual':
            # --- 推荐的修改逻辑 ---

            # 1. 传递压缩类型字符串 (核心！)
            # 这解决了文件名 comp[none] 的问题
            cmd.extend(['--compression', config.get('compression', 'none')])

            # 2. 传递是否禁用压缩的标志
            if config.get('disable_compression', False):
                cmd.extend(['--disable_compression'])

            # 3. 传递是否启用智能压缩的标志
            # (注意：这里的逻辑要和第1步匹配)
            if config.get('compression') == 'smart':
                cmd.extend(['--smart_compression', '1'])
            else:
                cmd.extend(['--smart_compression', '0'])

            # 4. 传递是否启用压缩的标志
            if config.get('compression') in ['smart', 'uniform']:
                cmd.extend(['--enable_compression'])

            # 5. 传递其他所有 residual 专属参数
            if config.get('compression_ratio') is not None:
                cmd.extend(['--compression_ratio', str(config.get('compression_ratio'))])

            if config.get('enable_knowledge_distillation') is not None:
                cmd.extend(['--enable_knowledge_distillation', str(config['enable_knowledge_distillation'])])
            if config.get('distill_temperature') is not None:
                cmd.extend(['--distill_temperature', str(config['distill_temperature'])])
            if config.get('distill_alpha') is not None:
                cmd.extend(['--distill_alpha', str(config['distill_alpha'])])
            if config.get('adaptive_aggregation') is not None:
                cmd.extend(['--adaptive_aggregation', config['adaptive_aggregation']])
        
        # 添加FedProx参数
        if config.get('mu') is not None:
            cmd.extend(['--mu', str(config['mu'])])
        
        return cmd
    
    def run_experiment(self, config, realtime=None, silent=False):
        """运行单个实验"""
        print(f"\n=== 开始实验 ===")
        print(f"数据集: {config['dataset']}")
        print(f"方法: {config['method']}")
        print(f"模型: {config['model']}")
        print(f"训练轮数: {config['epochs']}")
        if config['method'] != 'baseline':
            print(f"客户端数量: {config['num_users']}")
            print(f"参与比例: {config['frac']}")
            print(f"数据分布: {'IID' if config['iid'] else 'Non-IID'}")
        print(f"学习率: {config['lr']}")
        
        cmd = self.build_command(config)
        print(f"\n执行命令: {' '.join(cmd)}")
        
        # 永远使用实时输出，除非明确指定静默模式
        if silent:
            show_realtime = 'n'  # 只有静默模式才不显示实时输出
        else:
            show_realtime = 'y'  # 默认总是实时输出，不再询问用户
        
        start_time = time.time()
        
        # 设置环境变量以确保正确的编码
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONLEGACYWINDOWSSTDIO'] = '1'
        env['PYTHONUTF8'] = '1'
        
        # 在Windows上设置控制台编码
        if os.name == 'nt':
            os.system('chcp 65001 > nul')
        
        try:
            if show_realtime == 'y':
                # 实时显示输出 - 直接继承stdio，不捕获输出
                print("\n=== 开始实时训练 ===")
                print("您将看到完整的训练过程...")
                print("-" * 50)
                
                # 直接运行，完全继承当前进程的stdio
                result = subprocess.run(cmd, env=env, cwd='.')
                return_code = result.returncode
                
                end_time = time.time()
                print("-" * 50)
                print(f"实验完成! 用时: {end_time - start_time:.2f}秒")
                
                if return_code == 0:
                    print("[SUCCESS] 实验成功完成")
                    
                    # 默认开启自动可视化
                    print("\n[INFO] 正在启动自动可视化...")
                    log_path = self._get_log_path(config)
                    
                    if log_path and os.path.exists(log_path):
                        try:
                            vis_cmd = [sys.executable, 'visualize_results.py', log_path]
                            print(f"执行可视化命令: {' '.join(vis_cmd)}")
                            # 使用 subprocess.run 来确保可视化窗口关闭后主程序才继续
                            subprocess.run(vis_cmd)
                        except Exception as vis_e:
                            print(f"[ERROR] 启动可视化脚本失败: {vis_e}")
                    else:
                        print(f"[WARNING] 找不到预期的日志文件: {log_path}，跳过可视化。")
                else:
                    print("[ERROR] 实验执行出错")
            else:
                # 静默模式 - 后台运行
                print("\n[INFO] 静默模式运行中...")
                result = subprocess.run(cmd, capture_output=True, text=True, cwd='.', 
                                      encoding='utf-8', errors='replace', env=env)
                end_time = time.time()
                
                print(f"\n实验完成! 用时: {end_time - start_time:.2f}秒")
                
                if result.returncode == 0:
                    print("[SUCCESS] 实验成功完成")
                    if result.stdout:
                        print("\n--- 训练结果摘要 ---")
                        lines = result.stdout.split('\n')
                        for line in lines:
                            if any(keyword in line for keyword in ['测试准确率', '训练准确率', '总运行时间', '保存到']):
                                print(line.strip())
                    
                    # 自动可视化
                    print("\n[INFO] 正在启动自动可视化...")
                    log_path = self._get_log_path(config)
                    
                    if log_path and os.path.exists(log_path):
                        try:
                            vis_cmd = [sys.executable, 'visualize_results.py', log_path]
                            print(f"执行可视化命令: {' '.join(vis_cmd)}")
                            subprocess.run(vis_cmd)
                        except Exception as vis_e:
                            print(f"[ERROR] 启动可视化脚本失败: {vis_e}")
                    else:
                        print(f"[WARNING] 找不到预期的日志文件: {log_path}，跳过可视化。")
                else:
                    print("[ERROR] 实验执行出错")
                    if result.stderr:
                        print("--- 错误信息 ---")
                        print(result.stderr)
                
        except KeyboardInterrupt:
            print(f"\n[INFO] 用户中断实验")
        except Exception as e:
            print(f"[ERROR] 实验执行异常: {e}")
    
    def batch_run_experiments(self):
        """批量运行实验"""
        print("=== 批量运行模式 ===")
        print("将运行预定义的实验配置...")
        
        # 预定义实验配置
        batch_configs = [
            # MNIST实验
            {'dataset': 'mnist', 'method': 'baseline', 'model': 'cnn', 'epochs': 20},
            {'dataset': 'mnist', 'method': 'federated', 'model': 'cnn', 'epochs': 20},
            {'dataset': 'mnist', 'method': 'residual', 'model': 'cnn', 'epochs': 20},
            {'dataset': 'mnist', 'method': 'federated', 'model': 'optimized', 'epochs': 20},
            {'dataset': 'mnist', 'method': 'residual', 'model': 'optimized', 'epochs': 20},
            
            # CIFAR-10实验
            {'dataset': 'cifar10', 'method': 'baseline', 'model': 'cnn', 'epochs': 50},
            {'dataset': 'cifar10', 'method': 'federated', 'model': 'cnn', 'epochs': 50},
            {'dataset': 'cifar10', 'method': 'residual', 'model': 'cnn', 'epochs': 50},
            {'dataset': 'cifar10', 'method': 'federated', 'model': 'resnet18_fed', 'epochs': 50},
            {'dataset': 'cifar10', 'method': 'residual', 'model': 'resnet18_fed', 'epochs': 50},
        ]
        
        # 默认参数
        default_params = {
            'num_users': 100,
            'frac': 0.1,
            'local_ep': 10,
            'local_bs': 10,
            'lr': 0.01,
            'iid': 1,
            'gpu': None
        }
        
        total_experiments = len(batch_configs)
        for i, config in enumerate(batch_configs, 1):
            print(f"\n{'='*50}")
            print(f"实验 {i}/{total_experiments}")
            
            # 合并默认参数
            full_config = {**default_params, **config}
            
            # 为基线方法调整参数
            if config['method'] == 'baseline':
                full_config.update({
                    'num_users': 1,
                    'frac': 1.0,
                    'local_ep': config['epochs'],
                })
            
            self.run_experiment(full_config, realtime=False, silent=True)
            
        print(f"\n[SUCCESS] 批量实验完成! 共运行了 {total_experiments} 个实验")
    
    def print_help(self):
        """打印使用帮助"""
        print("=== 联邦学习实验管理器使用指南 ===\n")
        
        print("1. 实时训练模式 (推荐):")
        print("   python realtime_train.py")
        print("   快速选择配置，总是显示实时训练过程\n")
        
        print("2. 交互式模式:")
        print("   python experiment_manager.py --interactive")
        print("   按提示选择配置参数\n")
        
        print("3. 命令行模式 (默认实时输出):")
        print("   python experiment_manager.py --dataset mnist --method federated --model cnn --epochs 20")
        print("   直接指定所有参数，默认显示训练过程\n")
        
        print("4. 批量运行模式:")
        print("   python experiment_manager.py --batch_run")
        print("   运行预定义的多个实验配置\n")
        
        print("5. 强制静默模式:")
        print("   python experiment_manager.py --dataset mnist --method federated --model cnn --epochs 20 --silent")
        print("   只显示最终结果，不显示训练过程\n")
        
        print("可用参数:")
        print("  --dataset: 数据集 (mnist, cifar10, cifar100, fmnist)")
        print("  --method: 方法 (baseline, federated, residual)")
        print("  --model: 模型 (根据数据集不同)")
        print("  --epochs: 训练轮数")
        print("  --num_users: 客户端数量")
        print("  --frac: 参与比例")
        print("  --local_ep: 本地训练轮数")
        print("  --local_bs: 本地批量大小") 
        print("  --lr: 学习率")
        print("  --iid: 数据分布 (1=IID, 0=Non-IID)")
        print("  --gpu: GPU设备ID")
        print("  --optimizer: 优化器 (sgd/adam/adamw)")
        print("  --lr_scheduler: 学习率调度器 (none/fixed/step/exp/cosine)")
        print("  --weight_decay: 权重衰减")
        print("  --compression: 压缩类型 (none/smart/uniform, 仅残差联邦学习)")
        print("  --compression_ratio: 压缩比例 (仅残差联邦学习)")
        print("  --disable_compression: 禁用压缩 (仅残差联邦学习)")
        print("  --realtime: 显示实时训练过程")
        print("  --silent: 静默模式，只显示最终结果")
    
    def validate_config(self, config):
        """验证配置参数"""
        if config['dataset'] not in self.available_datasets:
            raise ValueError(f"不支持的数据集: {config['dataset']}")
        
        if config['method'] not in self.available_methods:
            raise ValueError(f"不支持的方法: {config['method']}")
        
        available_models = self.available_models.get(config['dataset'], [])
        if config['model'] not in available_models:
            print(f"警告: 模型 {config['model']} 可能不支持数据集 {config['dataset']}")
            print(f"推荐模型: {', '.join(available_models)}")
        
        return True
    
    def run(self):
        """主运行函数"""
        args = self.parse_args()
        
        if args.interactive:
            config = self.interactive_config()
            self.validate_config(config)
            self.run_experiment(config, realtime=args.realtime)
            
        elif args.batch_run:
            self.batch_run_experiments()
            
        elif args.dataset and args.method and args.model:
            config = {
                'dataset': args.dataset,
                'method': args.method,
                'model': args.model,
                'epochs': args.epochs,
                'pretrain_epochs': args.pretrain_epochs,
                'num_users': args.num_users,
                'frac': args.frac,
                'local_ep': args.local_ep,
                'local_bs': args.local_bs,
                'lr': args.lr,
                'iid': args.iid,
                'alpha': args.alpha,
                'gpu': args.gpu,
                'optimizer': args.optimizer,
                'lr_scheduler': args.lr_scheduler,
                'weight_decay': args.weight_decay,
                'compression': args.compression,
                'compression_ratio': args.compression_ratio,
                'disable_compression': args.disable_compression,
                'stopping_rounds': args.stopping_rounds,
                'smart_compression': args.smart_compression,
                'enable_knowledge_distillation': args.enable_knowledge_distillation,
                'distill_temperature': args.distill_temperature,
                'distill_alpha': args.distill_alpha,
                'adaptive_aggregation': args.adaptive_aggregation,
                'mu': args.mu
            }
            
            # 为基线方法调整参数
            if config['method'] == 'baseline':
                config.update({
                    'num_users': 1,
                    'frac': 1.0,
                    'local_ep': config['epochs'],
                })
            
            self.validate_config(config)
            # 如果没有指定realtime或silent，默认使用实时输出
            realtime_mode = args.realtime if hasattr(args, 'realtime') and args.realtime is not None else True
            silent_mode = args.silent if hasattr(args, 'silent') and args.silent else False
            self.run_experiment(config, realtime=realtime_mode, silent=silent_mode)
            
        else:
            self.print_help()


if __name__ == '__main__':
    manager = ExperimentManager()
    manager.run()