import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Union, Optional
from tqdm import tqdm
import os
import numpy as np
import copy
# ==================== 新增代码区域 (1/5) ====================
# 导入 matplotlib 用于绘图
import matplotlib.pyplot as plt
# ==========================================================
# 导入您自定义的损失函数和评估函数
from loss_functions import CombinedLoss
# 假设您的 evaluate_snn 函数保存在一个名为 'evaluate.py' 的文件中
from evaluate import evaluate_snn 

class FeatureExtractor:
    """
    一个辅助类，用于通过PyTorch hooks提取模型中间层的特征图。
    优化了SNN的特征提取，以避免在长时间步上出现显存溢出。
    """
    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = layer_names
        self.features: Dict[str, torch.Tensor] = {}
        self._hooks = []

    def _create_hook(self, name: str, is_snn: bool):
        def hook(model, input, output):
            if is_snn:
                # 优化：不存储每个时间步的输出，而是进行累加
                if name not in self.features:
                    # 在第一个时间步，初始化为输出值
                    self.features[name] = output.detach().clone()
                else:
                    # 在后续时间步，将新输出累加到已有张量上
                    self.features[name] += output.detach()
            else:
                # 对于ANN，直接存储特征
                self.features[name] = output
        return hook

    def register_hooks(self, is_snn: bool = False):
        self.remove_hooks()
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                hook = self._create_hook(name, is_snn)
                self._hooks.append(module.register_forward_hook(hook))

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def clear_features(self):
        """清空已捕获的特征。"""
        self.features = {}

def train_snn(
    student_snn: nn.Module,
    teacher_ann: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader, 
    device: torch.device,
    time_steps: int,
    epochs: int,
    w_l_dict: Dict[str, float],
    alpha: float,
    beta: float,
    temperature: float,
    lr: float = 1e-4,
    model_save_path: str = 'best_snn_model.pth',
    # ==================== 新增代码区域 (2/5) ====================
    # 新增用于保存绘图的路径前缀参数
    plot_save_prefix: Optional[str] = None
    # ==========================================================
):
    """
    使用复合损失函数对SNN进行微调训练。
    """
    for param in teacher_ann.parameters():
        param.requires_grad = False
    teacher_ann.eval()

    student_snn.train()

    criterion = CombinedLoss(w_l=w_l_dict, alpha=alpha, beta=beta, temperature=temperature)
    optimizer = torch.optim.Adam(student_snn.parameters(), lr=lr)

    layer_names = list(w_l_dict.keys())
    snn_extractor = FeatureExtractor(student_snn, layer_names)
    ann_extractor = FeatureExtractor(teacher_ann, layer_names)
    
    best_acc = 0.0

    # ==================== 新增代码区域 (3/5) ====================
    # 初始化用于记录历史数据的列表
    history = {
        'total_loss': [],
        'task_loss': [],
        'global_loss': [],
        'local_loss': [],
        'accuracy': []
    }
    # ==========================================================

    for epoch in range(epochs):
        # ==================== 新增代码 ====================
        # 在每个 epoch 开始时，强制清空 PyTorch 的 CUDA 缓存
        # 这有助于减少内存碎片，为新的 epoch 准备干净的显存环境
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        # ===============================================
        student_snn.train() 

        epoch_total_loss, epoch_task_loss, epoch_global_loss, epoch_local_loss = 0.0, 0.0, 0.0, 0.0
        
        for img, label in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()

            ann_extractor.clear_features()
            ann_extractor.register_hooks(is_snn=False)
            with torch.no_grad():
                ann_logits = teacher_ann(img)
            ann_features = ann_extractor.features
            ann_extractor.remove_hooks()

            snn_extractor.clear_features()
            snn_extractor.register_hooks(is_snn=True)
            
            for m in student_snn.modules():
                if hasattr(m, 'reset'):
                    m.reset()
            
            snn_output_accumulator = 0.0
            for _ in range(time_steps):
                snn_output_accumulator += student_snn(img)

            final_snn_output = snn_output_accumulator / time_steps
            
            # 优化：直接从提取器获取特征和，然后计算平均值
            summed_snn_features = snn_extractor.features
            avg_snn_features = {name: feats / time_steps for name, feats in summed_snn_features.items()}
            snn_extractor.remove_hooks()

            loss, l_task, l_global, l_local = criterion(
                snn_output=final_snn_output,
                ann_output=ann_logits,
                labels=label,
                snn_features=avg_snn_features,
                ann_features=ann_features
            )

            loss.backward()
            optimizer.step()

            epoch_total_loss += loss.item()
            epoch_task_loss += l_task.item()
            epoch_global_loss += l_global.item()
            epoch_local_loss += l_local.item()

        num_batches = len(train_loader)
        print(f"Epoch {epoch + 1} finished.")
        print(f"  Avg Total Loss: {epoch_total_loss / num_batches:.4f}")
        print(f"  -> Avg Task Loss: {epoch_task_loss / num_batches:.4f}")
        print(f"  -> Avg Global Loss: {epoch_global_loss / num_batches:.4f}")
        print(f"  -> Avg Local Loss: {epoch_local_loss / num_batches:.4f}")

        print("  Evaluating on test set...")
        accuracies = evaluate_snn(student_snn, test_loader, device, time_steps)
        current_acc = accuracies[-1] 

        # ==================== 新增代码区域 (4/5) ====================
        # 将当前周期的平均损失和准确率存入历史记录
        history['total_loss'].append(epoch_total_loss / num_batches)
        history['task_loss'].append(epoch_task_loss / num_batches)
        history['global_loss'].append(epoch_global_loss / num_batches)
        history['local_loss'].append(epoch_local_loss / num_batches)
        history['accuracy'].append(current_acc)
        # ==========================================================

        if current_acc > best_acc:
            best_acc = current_acc
            print(f"  New best accuracy: {best_acc:.2f}%. Saving model to {model_save_path}...")
            torch.save(student_snn.state_dict(), model_save_path)

    print(f"SNN fine-tuning complete. Best accuracy achieved: {best_acc:.2f}%")

    # ==================== 新增代码区域 (5/5) ====================
    # 在所有训练周期结束后，进行绘图和保存
    if plot_save_prefix and epochs > 0:
        print(f"正在生成训练过程曲线图并保存至: {plot_save_prefix}_*")
        
        # 创建一个包含两个子图的画布
        fig, ax1 = plt.subplots(figsize=(12, 5))
        
        # 绘制准确率曲线 (左Y轴)
        color = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)', color=color)
        ax1.plot(range(1, epochs + 1), history['accuracy'], 'o-', color=color, label='Accuracy')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--')

        # 创建第二个Y轴来绘制损失曲线
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Total Loss', color=color)
        ax2.plot(range(1, epochs + 1), history['total_loss'], 's-', color=color, label='Total Loss')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # 添加图例和标题
        fig.suptitle(f'SNN Fine-Tuning Performance (T={time_steps})', fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96]) # 调整布局为标题留出空间
        
        # 保存图像
        plt.savefig(f"{plot_save_prefix}_training_curve.png")
        plt.close(fig) # 关闭图像以释放内存
        
        # 将历史数据保存为Numpy文件，方便后续分析
        np.save(f"{plot_save_prefix}_history.npy", history)
    # ==========================================================

    print(f"Loading best model from {model_save_path}...")
    student_snn.load_state_dict(torch.load(model_save_path))
    
    return student_snn

