import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import glob
from datetime import datetime
import re
import shutil

'''
单独使用这个文件的命令行是
python visualize_results.py path/to/csv/file.csv --output_dir ./save/plots
python visualize_results.py --comparison --output_dir ./save/plots
'''

def extract_experiment_info(csv_files):
    """
    从CSV文件名中提取实验信息
    """
    info_list = []
    for file in csv_files:
        filename = os.path.basename(file)
        info = f"文件: {filename}\n"
        
        # 尝试从文件名中提取信息
        # 例如: log_federated_mnist_cnn_100ep_noniid.csv
        if 'federated' in filename:
            info += "  方法: Federated Learning\n"
        if 'mnist' in filename:
            info += "  数据集: MNIST\n"
        elif 'cifar' in filename:
            info += "  数据集: CIFAR\n"
        if 'cnn' in filename:
            info += "  模型: CNN\n"
        elif 'mlp' in filename:
            info += "  模型: MLP\n"
        if 'iid' in filename:
            info += "  数据分布: IID\n"
        elif 'noniid' in filename:
            info += "  数据分布: Non-IID\n"
            
        # 提取epoch数
        epoch_match = re.search(r'(\d+)ep', filename)
        if epoch_match:
            info += f"  训练轮数: {epoch_match.group(1)}\n"
            
        info_list.append(info)
    
    return '\n'.join(info_list)

def extract_experiment_label_from_details(experiment_dir):
    """
    从experiment_details.txt文件中提取实验标签
    """
    details_path = os.path.join(experiment_dir, 'experiment_details.txt')
    if not os.path.exists(details_path):
        return os.path.basename(experiment_dir)
    
    try:
        with open(details_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取关键信息构建标签
        method = "Unknown"
        dataset = "Unknown"
        model = "Unknown"
        distribution = "Unknown"
        
        if "Federated Learning" in content:
            method = "Federated"
        elif "Residual" in content or "residual" in content:
            method = "Residual"
        elif "Baseline" in content or "baseline" in content:
            method = "Baseline"
            
        if "MNIST" in content:
            dataset = "MNIST"
        elif "CIFAR" in content:
            dataset = "CIFAR"
            
        if "CNN" in content:
            model = "CNN"
        elif "MLP" in content:
            model = "MLP"
            
        if "IID" in content and "Non-IID" not in content:
            distribution = "IID"
        elif "Non-IID" in content:
            distribution = "Non-IID"
        
        return f"{method}_{dataset}_{model}_{distribution}"
    except:
        return os.path.basename(experiment_dir)

def extract_experiment_label_from_filename(filename):
    """
    从CSV文件名中提取实验标签
    例如: log_residual_mnist_cnn_100ep_noniid_comp_smart.csv
    """
    # 移除文件扩展名和前缀
    base_name = filename.replace('log_', '').replace('.csv', '')
    
    # 提取关键信息
    method = "Unknown"
    dataset = "Unknown"
    model = "Unknown"
    distribution = "Unknown"
    compression = ""
    
    if 'residual' in base_name:
        method = "Residual"
    elif 'federated' in base_name:
        method = "Federated"
    elif 'baseline' in base_name:
        method = "Baseline"
        
    if 'mnist' in base_name:
        dataset = "MNIST"
    elif 'cifar10' in base_name:
        dataset = "CIFAR10"
    elif 'cifar100' in base_name:
        dataset = "CIFAR100"
    elif 'cifar' in base_name:
        dataset = "CIFAR"
        
    if 'cnn' in base_name:
        model = "CNN"
    elif 'mlp' in base_name:
        model = "MLP"
        
    if 'noniid' in base_name:
        distribution = "Non-IID"
    elif 'iid' in base_name:
        distribution = "IID"
    
    # 提取压缩信息
    if 'comp_smart' in base_name:
        compression = "_SmartComp"
    elif 'comp_none' in base_name:
        compression = "_NoComp"
    elif 'comp_' in base_name:
        # 提取压缩比例
        comp_match = re.search(r'comp_([0-9.]+)', base_name)
        if comp_match:
            compression = f"_Comp{comp_match.group(1)}"
    
    return f"{method}_{dataset}_{model}_{distribution}{compression}"

def create_comparison_plots(plots_dir, output_dir, custom_input_dir=None):
    """
    根据plots文件夹中的多个实验结果创建对比图
    
    Args:
        plots_dir: 默认的plots目录
        output_dir: 输出目录
        custom_input_dir: 自定义输入目录，如果提供则直接从该目录读取CSV文件
    """
    if custom_input_dir and os.path.exists(custom_input_dir):
        # 如果指定了自定义输入目录，直接从该目录读取CSV文件
        print(f"使用自定义输入目录: {custom_input_dir}")
        csv_files = glob.glob(os.path.join(custom_input_dir, '*.csv'))
        
        if len(csv_files) < 2:
            print(f"自定义目录中找到的CSV文件少于2个，无法进行对比。找到的文件: {csv_files}")
            return
        
        print(f"在自定义目录中找到 {len(csv_files)} 个CSV文件，开始创建对比图...")
        
        # 收集所有实验的CSV数据
        all_experiment_data = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # 从CSV文件对应的logs目录中查找experiment_details.txt
                filename = os.path.basename(csv_file)
                timestamp = filename.replace('.csv', '')
                
                # 在logs目录中查找对应的实验详情
                logs_dir = os.path.join('./save/logs', timestamp)
                if os.path.exists(logs_dir):
                    label = extract_experiment_label_from_details(logs_dir)
                else:
                    # 如果找不到详情文件，使用时间戳作为标签
                    label = timestamp
                
                all_experiment_data.append({
                    'df': df,
                    'label': label,
                    'exp_dir': filename
                })
            except Exception as e:
                print(f"无法读取CSV文件 {csv_file}: {e}")
        
    else:
        # 原有逻辑：从logs目录寻找所有实验文件夹
        logs_dir = './save/logs'
        if not os.path.exists(logs_dir):
            print(f"日志目录 {logs_dir} 不存在")
            return
            
        experiment_dirs = [d for d in os.listdir(logs_dir) 
                          if os.path.isdir(os.path.join(logs_dir, d)) and 
                          re.match(r'\d{8}_\d{6}', d)]
        
        if len(experiment_dirs) < 2:
            print(f"找到的实验文件夹少于2个，无法进行对比。找到的文件夹: {experiment_dirs}")
            return
        
        print(f"找到 {len(experiment_dirs)} 个实验文件夹，开始创建对比图...")
        
        # 收集所有实验的CSV数据
        all_experiment_data = []
        for exp_dir in experiment_dirs:
            exp_path = os.path.join(logs_dir, exp_dir)
            # CSV文件名现在是时间戳.csv
            csv_file = os.path.join(exp_path, f'{exp_dir}.csv')
            
            if os.path.exists(csv_file):
                try:
                    df = pd.read_csv(csv_file)
                    label = extract_experiment_label_from_details(exp_path)
                    all_experiment_data.append({
                        'df': df,
                        'label': label,
                        'exp_dir': exp_dir
                    })
                except Exception as e:
                    print(f"无法读取实验 {exp_dir} 的CSV文件: {e}")
            else:
                print(f"实验文件夹 {exp_dir} 中未找到对应的CSV文件: {csv_file}")
    
    if len(all_experiment_data) < 2:
        print("有效的实验数据少于2个，无法进行对比。")
        return
    
    # 创建对比图的输出文件夹
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    comparison_dir = os.path.join(output_dir, f"comparison_{current_time}")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # 标题和标签
    labels = {
        'epoch': 'Epoch',
        'accuracy': 'Accuracy',
        'loss': 'Loss',
        'lr': 'Learning Rate',
        'compression': 'Compression Ratio'
    }

    titles = {
        'accuracy': 'Test Accuracy Comparison',
        'loss': 'Average Training Loss Comparison',
        'lr': 'Learning Rate Comparison',
        'compression': 'Compression Ratio Comparison'
    }
    
    # 1. 对比测试准确率
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    for data in all_experiment_data:
        if 'test_accuracy' in data['df'].columns:
            ax1.plot(data['df']['epoch'], data['df']['test_accuracy'], 
                    marker='o', linestyle='-', markersize=4, label=data['label'])
    ax1.set_title(titles['accuracy'], fontsize=16)
    ax1.set_xlabel(labels['epoch'], fontsize=12)
    ax1.set_ylabel(labels['accuracy'], fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True)
    fig1.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'comparison_test_accuracy.png'))
    print(f"已保存对比准确率图到 {comparison_dir}")
    
    # 2. 对比平均训练损失
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    for data in all_experiment_data:
        if 'avg_train_loss' in data['df'].columns:
            ax2.plot(data['df']['epoch'], data['df']['avg_train_loss'], 
                    marker='x', linestyle='--', markersize=4, label=data['label'])
    ax2.set_title(titles['loss'], fontsize=16)
    ax2.set_xlabel(labels['epoch'], fontsize=12)
    ax2.set_ylabel(labels['loss'], fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True)
    fig2.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'comparison_train_loss.png'))
    print(f"已保存对比损失图到 {comparison_dir}")
    
    # 3. 对比学习率
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    for data in all_experiment_data:
        if 'learning_rate' in data['df'].columns:
            ax3.plot(data['df']['epoch'], data['df']['learning_rate'], 
                    marker='.', linestyle=':', label=data['label'])
    ax3.set_title(titles['lr'], fontsize=16)
    ax3.set_xlabel(labels['epoch'], fontsize=12)
    ax3.set_ylabel(labels['lr'], fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True)
    fig3.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'comparison_learning_rate.png'))
    print(f"已保存对比学习率图到 {comparison_dir}")
    
    # 4. 对比压缩率（如果有的话）
    residual_data = [data for data in all_experiment_data if 'compression_ratio' in data['df'].columns]
    if residual_data:
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        for data in residual_data:
            ax4.plot(data['df']['epoch'], data['df']['compression_ratio'], 
                    marker='s', linestyle='-.', markersize=4, label=data['label'])
        ax4.set_title(titles['compression'], fontsize=16)
        ax4.set_xlabel(labels['epoch'], fontsize=12)
        ax4.set_ylabel(labels['compression'], fontsize=12)
        ax4.legend(fontsize=10)
        ax4.grid(True)
        fig4.tight_layout()
        plt.savefig(os.path.join(comparison_dir, 'comparison_compression_ratio.png'))
        print(f"已保存对比压缩率图到 {comparison_dir}")
    
    # 保存对比实验详情
    experiment_details = f"""=== 对比实验详情 ===
时间戳: {current_time}
对比的实验数量: {len(all_experiment_data)}

=== 包含的实验 ===
"""
    for data in all_experiment_data:
        experiment_details += f"- {data['label']} (来自文件夹: {data['exp_dir']})\n"
    
    experiment_details += f"""

=== 生成的对比图像文件 ===
- comparison_test_accuracy.png: 测试准确率对比图
- comparison_train_loss.png: 训练损失对比图  
- comparison_learning_rate.png: 学习率对比图
{('- comparison_compression_ratio.png: 压缩率对比图' if residual_data else '')}
"""

    with open(os.path.join(comparison_dir, 'comparison_details.txt'), 'w', encoding='utf-8') as f:
        f.write(experiment_details)
    
    plt.show()
    return comparison_dir

def merge_same_type_plots(plots_dir, output_dir):
    """
    合并相同类型的图表到一个文件中
    plots_dir: 包含实验结果文件夹的目录
    output_dir: 输出合并图表的目录
    """
    try:
        # 查找所有实验文件夹
        experiment_dirs = [d for d in os.listdir(plots_dir) 
                          if os.path.isdir(os.path.join(plots_dir, d)) and 
                          re.match(r'\d{8}_\d{6}', d)]
        
        if len(experiment_dirs) < 2:
            print(f"找到的实验文件夹少于2个，无需合并。找到的文件夹: {experiment_dirs}")
            return
        
        experiment_dirs.sort()  # 按时间排序
        print(f"找到 {len(experiment_dirs)} 个实验文件夹，开始合并同类型图表...")
        
        # 创建输出目录
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        merge_dir = os.path.join(output_dir, f"merged_{current_time}")
        os.makedirs(merge_dir, exist_ok=True)
        
        # 图表类型和对应的文件名
        plot_types = {
            '1_training_loss.png': 'Training Loss Comparison',
            '2_test_accuracy.png': 'Test Accuracy Comparison',
            '3_communication_cost.png': 'Communication Cost Comparison'
        }
        
        # 为每种类型创建合并图
        for plot_file, plot_title in plot_types.items():
            plt.figure(figsize=(12, 8))
            
            valid_experiments = []
            for exp_dir in experiment_dirs:
                exp_path = os.path.join(plots_dir, exp_dir)
                plot_path = os.path.join(exp_path, plot_file)
                
                if os.path.exists(plot_path):
                    # 读取对应的CSV数据
                    csv_path = os.path.join(exp_path, f'{exp_dir}.csv')
                    if os.path.exists(csv_path):
                        try:
                            df = pd.read_csv(csv_path)
                            label = extract_experiment_label_from_details(exp_path)
                            valid_experiments.append({
                                'df': df,
                                'label': label,
                                'exp_dir': exp_dir
                            })
                        except Exception as e:
                            print(f"读取实验 {exp_dir} 的CSV文件失败: {e}")
            
            if not valid_experiments:
                print(f"⚠️ 没有找到有效的数据用于 {plot_title}")
                plt.close()
                continue
            
            # 根据图表类型绘制数据
            if '1_training_loss' in plot_file:
                for exp_data in valid_experiments:
                    df = exp_data['df']
                    label = exp_data['label']
                    # 查找轮次和训练损失列
                    round_col = find_column(df, ['round', 'epoch'])
                    loss_col = find_column(df, ['train_loss', 'training_loss', 'loss'])
                    if round_col and loss_col:
                        plt.plot(df[round_col], df[loss_col], marker='o', label=label, linewidth=2)
                plt.ylabel('Loss')
                
            elif '2_test_accuracy' in plot_file:
                for exp_data in valid_experiments:
                    df = exp_data['df']
                    label = exp_data['label']
                    round_col = find_column(df, ['round', 'epoch'])
                    acc_col = find_column(df, ['test_accuracy', 'testing_accuracy'])
                    if round_col and acc_col:
                        acc_data = df[acc_col]
                        if acc_data.max() <= 1.0:
                            acc_data = acc_data * 100
                        plt.plot(df[round_col], acc_data, marker='^', label=label, linewidth=2)
                plt.ylabel('Accuracy (%)')
                
            elif '3_communication_cost' in plot_file:
                for exp_data in valid_experiments:
                    df = exp_data['df']
                    label = exp_data['label']
                    round_col = find_column(df, ['round', 'epoch'])
                    comm_col = find_column(df, ['communication_cost', 'comm_cost', 'communication', 'comm'])
                    if round_col and comm_col:
                        plt.plot(df[round_col], df[comm_col], marker='d', label=label, linewidth=2)
                plt.ylabel('Parameters Transmitted')
            
            plt.title(plot_title, fontsize=16)
            plt.xlabel('Communication Round', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 保存合并图表
            merged_path = os.path.join(merge_dir, plot_file.replace('.png', '_merged.png'))
            plt.savefig(merged_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ 保存合并图表: {merged_path}")
        
        print(f"📊 所有合并图表已保存到: {merge_dir}")
        return merge_dir
        
    except Exception as e:
        print(f"合并图表失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def find_column(df, possible_names):
    """在DataFrame中查找匹配的列名"""
    for col in df.columns:
        col_lower = col.lower()
        for name in possible_names:
            if name.lower() in col_lower:
                return col
    return None

def plot_single_experiment(csv_file_path, plots_dir):
    """
    绘制单个实验的结果图，生成4个独立的PNG文件
    csv_file_path: CSV文件路径（来自./save/logs/时间戳/时间戳.csv）
    plots_dir: plots输出目录
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file_path)
        print(f"读取CSV文件: {csv_file_path}")
        print(f"CSV列名: {list(df.columns)}")
        print(f"CSV数据形状: {df.shape}")
        
        # 从文件路径中提取时间戳
        csv_dir = os.path.dirname(csv_file_path)
        timestamp = os.path.basename(csv_dir)
        
        # 在plots目录中创建对应的时间戳文件夹
        plot_dir = os.path.join(plots_dir, timestamp)
        os.makedirs(plot_dir, exist_ok=True)
        
        # 复制CSV文件到plots目录
        dest_csv = os.path.join(plot_dir, f'{timestamp}.csv')
        import shutil
        shutil.copy2(csv_file_path, dest_csv)
        
        # 复制experiment_details.txt文件（如果存在）
        details_src = os.path.join(csv_dir, 'experiment_details.txt')
        if os.path.exists(details_src):
            details_dest = os.path.join(plot_dir, 'experiment_details.txt')
            shutil.copy2(details_src, details_dest)
        
        # 设置绘图样式
        plt.style.use('default')
        
        # 检查数据列并映射到正确的列名
        round_col = None
        train_loss_col = None
        test_acc_col = None
        comm_cost_col = None
        compression_ratio_col = None  # 添加压缩比列
        
        # 尝试不同的列名模式
        for col in df.columns:
            col_lower = col.lower()
            if 'round' in col_lower or 'epoch' in col_lower:
                round_col = col
            elif 'train' in col_lower and 'loss' in col_lower:
                train_loss_col = col
            elif 'test' in col_lower and ('acc' in col_lower or 'accuracy' in col_lower):
                test_acc_col = col
            elif 'communication' in col_lower or 'comm' in col_lower:
                comm_cost_col = col
            elif 'compression' in col_lower and 'ratio' in col_lower:
                compression_ratio_col = col
        
        print(f"检测到的列映射:")
        print(f"  轮次列: {round_col}")
        print(f"  训练损失列: {train_loss_col}")
        print(f"  测试准确率列: {test_acc_col}")
        print(f"  通信开销列: {comm_cost_col}")
        print(f"  压缩比列: {compression_ratio_col}")
        
        # 1. 训练损失图
        if round_col and train_loss_col:
            plt.figure(figsize=(10, 6))
            plt.plot(df[round_col], df[train_loss_col], marker='o', color='red', linewidth=2, markersize=6)
            plt.title(f'Training Loss - {timestamp}', fontsize=14)
            plt.xlabel('Communication Round', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            loss_path = os.path.join(plot_dir, '1_training_loss.png')
            plt.savefig(loss_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ 保存训练损失图: {loss_path}")
        else:
            print("⚠️ 无法生成训练损失图 - 缺少必要数据列")
        
        # 2. 测试准确率图
        if round_col and test_acc_col:
            plt.figure(figsize=(10, 6))
            # 如果数据是小数形式，转换为百分比
            test_acc_data = df[test_acc_col]
            if test_acc_data.max() <= 1.0:
                test_acc_data = test_acc_data * 100
            plt.plot(df[round_col], test_acc_data, marker='^', color='green', linewidth=2, markersize=6)
            plt.title(f'Test Accuracy - {timestamp}', fontsize=14)
            plt.xlabel('Communication Round', fontsize=12)
            plt.ylabel('Accuracy (%)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            test_acc_path = os.path.join(plot_dir, '2_test_accuracy.png')
            plt.savefig(test_acc_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ 保存测试准确率图: {test_acc_path}")
        else:
            print("⚠️ 无法生成测试准确率图 - 缺少必要数据列")
        
        # 3. 通信开销图
        if round_col and comm_cost_col:
            plt.figure(figsize=(10, 6))
            plt.plot(df[round_col], df[comm_cost_col], marker='d', color='orange', linewidth=2, markersize=6)
            plt.title(f'Communication Cost - {timestamp}', fontsize=14)
            plt.xlabel('Communication Round', fontsize=12)
            plt.ylabel('Parameters Transmitted', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            comm_path = os.path.join(plot_dir, '3_communication_cost.png')
            plt.savefig(comm_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ 保存通信开销图: {comm_path}")
        else:
            # 创建空的通信开销图
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No Communication Cost Data Available', 
                    horizontalalignment='center', verticalalignment='center', 
                    fontsize=16, transform=plt.gca().transAxes)
            plt.title(f'Communication Cost - {timestamp}', fontsize=14)
            plt.xlabel('Communication Round', fontsize=12)
            plt.ylabel('Parameters Transmitted', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            comm_path = os.path.join(plot_dir, '3_communication_cost.png')
            plt.savefig(comm_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ 保存通信开销图（无数据）: {comm_path}")
        
        # 4. 压缩比图（如果有压缩比数据）
        if round_col and compression_ratio_col:
            plt.figure(figsize=(10, 6))
            plt.plot(df[round_col], df[compression_ratio_col], marker='s', color='purple', linewidth=2, markersize=6)
            plt.title(f'Compression Ratio - {timestamp}', fontsize=14)
            plt.xlabel('Communication Round', fontsize=12)
            plt.ylabel('Compression Ratio', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)  # 压缩比通常在0-1之间
            plt.tight_layout()
            comp_path = os.path.join(plot_dir, '4_compression_ratio.png')
            plt.savefig(comp_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ 保存压缩比图: {comp_path}")
        else:
            print("ℹ️ 未检测到压缩比数据，跳过压缩比图生成")
            plt.title(f'Communication Cost - {timestamp}', fontsize=14)
            plt.xlabel('Communication Round', fontsize=12)
            plt.ylabel('Parameters Transmitted', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            comm_path = os.path.join(plot_dir, '3_communication_cost.png')
            plt.savefig(comm_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ 保存通信开销图（无数据）: {comm_path}")
        
        print(f"📊 实验图表已保存到: {plot_dir}")
        print(f"📋 CSV文件已复制到: {dest_csv}")
        print("生成的图片文件顺序:")
        print("  1_training_loss.png")
        print("  2_test_accuracy.png")
        print("  3_communication_cost.png")
        
        return plot_dir
        
    except Exception as e:
        print(f"绘图失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_results(csv_files, output_dir):
    """
    从一个或多个CSV日志文件中读取数据并生成对比图。
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 强制使用英文标题和标签，避免中文字体问题
    use_english = True

    # 英文标题和标签
    labels = {
        'epoch': 'Epoch',
        'accuracy': 'Accuracy',
        'loss': 'Loss',
        'lr': 'Learning Rate',
        'compression': 'Compression Ratio'
    }

    titles = {
        'accuracy': 'Test Accuracy Comparison',
        'loss': 'Average Training Loss Comparison',
        'lr': 'Learning Rate Trend',
        'compression': 'Compression Ratio Comparison'
    }
    
    all_data = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # 从文件名中提取标签用于图例
            label = os.path.basename(file).replace('log_', '').replace('.csv', '')
            all_data.append({'df': df, 'label': label, 'path': file})
        except Exception as e:
            print(f"无法读取或处理文件 {file}: {e}")
            continue

    if not all_data:
        print("没有找到有效的数据进行绘图。")
        return

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取当前时间戳
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 创建实验文件夹
    experiment_dir = os.path.join(output_dir, current_time)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 1. 绘制测试准确率 (Test Accuracy)
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    for data in all_data:
        ax1.plot(data['df']['epoch'], data['df']['test_accuracy'], marker='o', linestyle='-', markersize=4, label=data['label'])
    ax1.set_title(titles['accuracy'], fontsize=16)
    ax1.set_xlabel(labels['epoch'], fontsize=12)
    ax1.set_ylabel(labels['accuracy'], fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True)
    fig1.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'plot_test_accuracy.png'))
    print(f"已保存准确率对比图到 {experiment_dir}")

    # 2. 绘制平均训练损失 (Average Training Loss)
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    for data in all_data:
        ax2.plot(data['df']['epoch'], data['df']['avg_train_loss'], marker='x', linestyle='--', markersize=4, label=data['label'])
    ax2.set_title(titles['loss'], fontsize=16)
    ax2.set_xlabel(labels['epoch'], fontsize=12)
    ax2.set_ylabel(labels['loss'], fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True)
    fig2.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'plot_train_loss.png'))
    print(f"已保存损失对比图到 {experiment_dir}")
    
    # 3. 绘制学习率 (Learning Rate)
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    for data in all_data:
        ax3.plot(data['df']['epoch'], data['df']['learning_rate'], marker='.', linestyle=':', label=data['label'])
    ax3.set_title(titles['lr'], fontsize=16)
    ax3.set_xlabel(labels['epoch'], fontsize=12)
    ax3.set_ylabel(labels['lr'], fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True)
    fig3.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'plot_learning_rate.png'))
    print(f"已保存学习率对比图到 {experiment_dir}")

    # 4. 绘制压缩比例 (Compression Ratio) - 仅限包含此列的数据
    residual_data = [data for data in all_data if 'compression_ratio' in data['df'].columns]
    if residual_data:
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        for data in residual_data:
            ax4.plot(data['df']['epoch'], data['df']['compression_ratio'], marker='s', linestyle='-.', markersize=4, label=data['label'])
        ax4.set_title(titles['compression'], fontsize=16)
        ax4.set_xlabel(labels['epoch'], fontsize=12)
        ax4.set_ylabel(labels['compression'], fontsize=12)
        ax4.legend(fontsize=10)
        ax4.grid(True)
        fig4.tight_layout()
        plt.savefig(os.path.join(experiment_dir, 'plot_compression_ratio.png'))
        print(f"已保存压缩率对比图到 {experiment_dir}")

    # 保存实验详情到文本文件
    # 从文件名中提取实验信息
    experiment_info = extract_experiment_info(csv_files)
    
    experiment_details = f"""=== 实验详情 ===
时间戳: {current_time}
输入文件: {', '.join([os.path.basename(f) for f in csv_files])}
输出目录: {experiment_dir}

=== 从文件名提取的实验信息 ===
{experiment_info}

=== 生成的图像文件 ===
- plot_test_accuracy.png: 测试准确率对比图
- plot_train_loss.png: 训练损失对比图  
- plot_learning_rate.png: 学习率变化曲线
{('- plot_compression_ratio.png: 压缩率对比图' if residual_data else '')}
"""

    with open(os.path.join(experiment_dir, 'experiment_details.txt'), 'w', encoding='utf-8') as f:
        f.write(experiment_details)
    
    # 复制CSV文件到实验文件夹
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            dest_path = os.path.join(experiment_dir, os.path.basename(csv_file))
            shutil.copy2(csv_file, dest_path)
            print(f"已复制CSV文件到: {dest_path}")
    
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从CSV日志中可视化训练历史。')
    parser.add_argument('files', nargs='*', help='CSV日志文件的路径。支持通配符，例如 "./save/logs/*/*.csv"。')
    parser.add_argument('--input_dir', type=str, default='./save/logs', help='如果未提供特定文件，则在此目录中搜索CSV文件。')
    parser.add_argument('--output_dir', type=str, default='./save/plots', help='保存图表的目录。')
    parser.add_argument('--comparison', action='store_true', help='创建对比图，对比多个实验结果。')
    parser.add_argument('--custom_input_dir', type=str, help='自定义输入目录，用于对比模式。')
    parser.add_argument('--single', type=str, help='指定单个实验的CSV文件路径进行绘图。')
    parser.add_argument('--merge', action='store_true', help='合并同类型的图表到一个文件中。')
    
    args = parser.parse_args()

    # 单个实验绘图模式
    if args.single:
        if os.path.exists(args.single):
            plot_single_experiment(args.single, args.output_dir)
        else:
            print(f"错误：文件 {args.single} 不存在。")
        exit(0)

    # 合并图表模式
    if args.merge:
        merge_same_type_plots(args.output_dir, args.output_dir)
        exit(0)

    # 对比图模式
    if args.comparison:
        create_comparison_plots(args.output_dir, args.output_dir, args.custom_input_dir)
        exit(0)

    # 原有的多文件可视化模式
    if args.files:
        csv_files = []
        for file_pattern in args.files:
            matched_files = glob.glob(file_pattern)
            if matched_files:
                csv_files.extend(matched_files)
            else:
                print(f"警告：模式 '{file_pattern}' 没有匹配到任何文件。")
    else:
        # 如果没有指定文件，则查找最新的实验
        print(f"未指定文件，将在 '{args.input_dir}' 目录中查找最新的实验...")
        experiment_dirs = [d for d in os.listdir(args.input_dir) 
                          if os.path.isdir(os.path.join(args.input_dir, d)) and 
                          re.match(r'\d{8}_\d{6}', d)]
        
        if experiment_dirs:
            # 找到最新的实验
            latest_exp = sorted(experiment_dirs)[-1]
            latest_csv = os.path.join(args.input_dir, latest_exp, f'{latest_exp}.csv')
            if os.path.exists(latest_csv):
                print(f"找到最新实验: {latest_exp}")
                plot_single_experiment(latest_csv, args.output_dir)
            else:
                print(f"错误：在最新实验文件夹 {latest_exp} 中未找到CSV文件。")
        else:
            print(f"错误：在 {args.input_dir} 中未找到任何实验文件夹。")
        exit(0)
    
    if not csv_files:
        print("错误：找不到任何CSV文件进行可视化。请检查文件路径或输入目录。")
    else:
        print(f"找到以下文件进行可视化: {csv_files}")
        plot_results(csv_files, args.output_dir)