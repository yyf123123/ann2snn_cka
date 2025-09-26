import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional
import os
import matplotlib.pyplot as plt

def evaluate_snn(snn_model: nn.Module, 
                 test_loader: DataLoader, 
                 device: torch.device, 
                 time_steps: int, 
                 save_path_prefix: Optional[str] = None) -> np.ndarray:
    """
    评估脉冲神经网络（SNN）在给定时间步长内的准确性，并可选择保存结果。

    Args:
        snn_model (nn.Module): 需要评估的SNN模型。
        test_loader (DataLoader): 包含测试数据集的DataLoader。
        device (torch.device): 运行评估的设备（例如 'cuda' 或 'cpu'）。
        time_steps (int): 每个输入样本要模拟的总时间步数。
        save_path_prefix (Optional[str], optional): 保存准确率数据和曲线图的文件路径前缀。
                                                    如果为None，则不保存文件。默认为 None。

    Returns:
        np.ndarray: 一个NumPy数组，包含每个时间步的累积准确率。
    """
    snn_model.eval().to(device)
    total_samples = 0.0
    
    # 用于存储每个时间步正确预测数量的数组
    cumulative_corrects = np.zeros(time_steps)

    with torch.no_grad():
        for img, label in tqdm(test_loader, desc="Evaluating SNN"):
            img = img.to(device)
            label = label.to(device)
            
            # 重置模型中所有有状态模块（如神经元）的状态
            for module in snn_model.modules():
                if hasattr(module, 'reset'):
                    module.reset()

            # 初始化输出张量
            out = 0.0
            
            # 在所有时间步上模拟网络
            for t in range(time_steps):
                # 模型的输出随时间累积
                out += snn_model(img)
                
                # 基于累积输出计算正确预测的数量
                preds = out.argmax(dim=1)
                cumulative_corrects[t] += (preds == label).float().sum().item()

            total_samples += img.shape[0]

    # 计算每个时间步的准确率
    # 将准确率转换为百分比形式
    accuracies = (cumulative_corrects / total_samples) * 100 if total_samples > 0 else np.zeros(time_steps)
    
    if total_samples > 0:
        print(f"Final accuracy after {time_steps} steps: {accuracies[-1]:.2f}%")
    else:
        print("Warning: No samples were evaluated.")

    # 如果提供了保存路径，则保存数据和绘图
    if save_path_prefix is not None:
        # 确保输出目录存在
        output_dir = os.path.dirname(save_path_prefix)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        np.save(save_path_prefix + '_accuracies.npy', accuracies)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, time_steps + 1), accuracies, marker='o', linestyle='-', label='Accuracy per Timestep')
        
        if total_samples > 0: # 避免在没有数据时出错
            best_t = np.argmax(accuracies)
            best_acc = accuracies[best_t]
            plt.plot(best_t + 1, best_acc, 'r*', markersize=15, label=f'Best: {best_acc:.2f}% @ T={best_t+1}')
        
        plt.xlabel('Time Step')
        plt.ylabel('Accuracy (%)')
        plt.title(f'SNN Accuracy Curve ({os.path.basename(save_path_prefix)})')
        plt.legend()
        plt.grid(True)
        if total_samples > 0: plt.ylim(0, 100) # 仅在有有效数据时设置Y轴范围
        
        plt.savefig(save_path_prefix + '_accuracy_curve.png')
        plt.close()
        print(f"准确率数据和曲线图已保存至: {save_path_prefix}_*")
        
    return accuracies



def evaluate_ann(ann_model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """
    评估ANN模型的准确率。

    Args:
        ann_model (nn.Module): 待评估的ANN模型。
        test_loader (torch.utils.data.DataLoader): 测试数据加载器。
        device (torch.device): 运行推理的设备。

    Returns:
        float: 模型在测试集上的准确率 (%)。
    """
    ann_model.eval()
    ann_model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="评估ANN", leave=False, ncols=100):
            images = images.to(device)
            labels = labels.to(device)
            outputs = ann_model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


