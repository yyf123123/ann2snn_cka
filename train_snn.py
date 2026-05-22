import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Set, Union
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from spikingjelly.activation_based import functional
from loss_functions import CombinedLoss
from evaluate import evaluate_snn
import time

class FeatureExtractor:
    """
    一个高效的辅助类，用于通过PyTorch hooks提取模型中间层的特征图。

    该类为SNN的特征提取做了优化，通过累加多个时间步的特征来避免显存溢出，
    同时也兼容标准ANN的单次特征提取。
    """
    def __init__(self, model: nn.Module, layer_names: List[str]):
        """
        初始化提取器。

        Args:
            model (nn.Module): 需要提取特征的模型。
            layer_names (List[str]): 需要提取特征的目标层的名称列表。
        """
        self.model = model
        self.layer_names: Set[str] = set(layer_names)
        self.features: Dict[str, torch.Tensor] = {}
        self._hooks = []
        
        # 1. 效率优化：在初始化时一次性找到所有目标模块，避免在每次注册时重复搜索。
        self._target_modules = {name: mod for name, mod in self.model.named_modules() if name in self.layer_names}
        
        # 2. 健壮性优化：验证所有指定的层都能在模型中找到，否则抛出错误。
        if len(self._target_modules) != len(self.layer_names):
            found_layers = set(self._target_modules.keys())
            missing_layers = self.layer_names - found_layers
            raise ValueError(f"错误: 在模型中未找到以下层: {missing_layers}")

    def _create_hook(self, name: str, is_snn: bool):
        def hook(model, input, output):
            # 对于ANN，我们只做一次前向传播，所以直接存储即可。
            # .detach()可以安全地将张量从计算图中分离出来。
            if not is_snn:
                self.features[name] = output.detach()
                return

            # 对于SNN，我们需要累加所有时间步的输出。
            if name not in self.features:
                # 在第一个时间步，初始化为输出值的副本。
                self.features[name] = output.detach().clone()
            else:
                # 3. 效率优化：在后续时间步，使用内存高效的原地加法(in-place addition)。
                self.features[name].add_(output.detach())
        return hook

    def register_hooks(self, is_snn: bool = False):
        """
        清空旧特征并为一次新的前向传播注册钩子。

        Args:
            is_snn (bool): 标记是否为SNN模式（累加特征）。默认为 False。
        """
        self.remove_hooks()  # 先移除旧的钩子，防止重复注册。
        self.clear_features() # 清空上一轮捕获的特征。
        
        for name, module in self._target_modules.items():
            hook = self._create_hook(name, is_snn)
            self._hooks.append(module.register_forward_hook(hook))

    def remove_hooks(self):
        """移除所有已注册的钩子，防止内存泄漏。"""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def clear_features(self):
        """清空已捕获的特征字典。"""
        self.features.clear()

def train_snn(
    student_snn: nn.Module,
    teacher_ann: nn.Module,
    train_loader,
    test_loader, 
    device: torch.device,
    time_steps: int,
    epochs: int,
    w_l_dict: Dict[str, float],
    alpha: float,
    beta: float,
    temperature: float,
    lr: float = 1e-4,
    model_save_path: str = 'best_snn_model.pth',
    plot_save_prefix: Optional[str] = None,
    # ==================== 【修改点 1 - 位置 A：新增参数】 ====================
    is_dynamic_cka: bool = False,
    cka_obj = None  # 传入在 main 中实例化的 CKA 对象
    # =====================================================================
):
    """
    使用复合损失函数对SNN进行微调训练。
    支持动态更新 CKA 权重，并统计时间与显存开销。
    """
    for param in teacher_ann.parameters():
        param.requires_grad = False
    teacher_ann.eval()

    student_snn.train()

    # 这里的 CombinedLoss 需要根据你的实际路径导入
    criterion = CombinedLoss(w_l=w_l_dict, alpha=alpha, beta=beta, temperature=temperature)
    optimizer = torch.optim.Adam(student_snn.parameters(), lr=lr)

    layer_names = list(w_l_dict.keys())
    # 注意：这里的 FeatureExtractor 需要是你自己实现的类，原封不动保留
    snn_extractor = FeatureExtractor(student_snn, layer_names)
    ann_extractor = FeatureExtractor(teacher_ann, layer_names)
    
    best_acc = 0.0

    history = {
        'total_loss': [],
        'task_loss': [],
        'global_loss': [],
        'local_loss': [],
        'accuracy': []
    }
    
    # ==================== 【修改点 1 - 新增：性能统计列表】 ====================
    performance_stats = {
        'epoch_time': [],
        'peak_memory_mb': []
    }
    # =====================================================================

    for epoch in range(epochs):
        # ==================== 【修改点 1 - 位置 B：开始统计与动态更新】 ====================
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache() # 为公平对比，每个epoch前清空一下缓存碎片
        
        start_time = time.time()

        if is_dynamic_cka and cka_obj is not None:
            print(f"\n[Dynamic CKA] Recalculating CKA weights for Epoch {epoch + 1}...")
            # 利用验证集或测试集重新计算 CKA 矩阵
            hsic_matrix, model1_names, model2_names = cka_obj.inference(test_loader)
            hsic_matrix = hsic_matrix.cpu().numpy()
            cka_current = np.diag(hsic_matrix)
            
            # 按照论文公式更新权重
            one_minus_cka = 1 - cka_current
            denominator = np.sum(one_minus_cka)
            if denominator == 0:
                num_layers = len(one_minus_cka)
                w_l = np.full(num_layers, 1.0 / num_layers)
            else:
                w_l = one_minus_cka / denominator
            
            new_w_l_dict = {name: weight for name, weight in zip(model1_names, w_l)}
            
            # 直接暴力替换损失函数实例中的权重配置
            criterion.local_loss_fn.w_l = new_w_l_dict
            
            np.set_printoptions(precision=4, suppress=True)
            print(f"[Dynamic CKA] Updated weights: {list(np.round(w_l, 4))}")
        # ===========================================================================

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

        history['total_loss'].append(epoch_total_loss / num_batches)
        history['task_loss'].append(epoch_task_loss / num_batches)
        history['global_loss'].append(epoch_global_loss / num_batches)
        history['local_loss'].append(epoch_local_loss / num_batches)
        history['accuracy'].append(current_acc)

        if current_acc > best_acc:
            best_acc = current_acc
            print(f"  New best accuracy: {best_acc:.2f}%. Saving model to {model_save_path}...")
            torch.save(student_snn.state_dict(), model_save_path)

        # ==================== 【修改点 1 - 位置 C：性能结算与打印】 ====================
        end_time = time.time()
        epoch_time = end_time - start_time
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0.0
        
        performance_stats['epoch_time'].append(epoch_time)
        performance_stats['peak_memory_mb'].append(peak_mem)
        
        print("-" * 50)
        print(f"  [Performance] Epoch Time: {epoch_time:.2f} s | Peak GPU Memory: {peak_mem:.2f} MB")
        print("-" * 50)
        # ===========================================================================

    print(f"SNN fine-tuning complete. Best accuracy achieved: {best_acc:.2f}%")

    if plot_save_prefix and epochs > 0:
        print(f"正在生成训练过程曲线图并保存至: {plot_save_prefix}_*")
        
        fig, ax1 = plt.subplots(figsize=(12, 5))
        color = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)', color=color)
        ax1.plot(range(1, epochs + 1), history['accuracy'], 'o-', color=color, label='Accuracy')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--')

        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Total Loss', color=color)
        ax2.plot(range(1, epochs + 1), history['total_loss'], 's-', color=color, label='Total Loss')
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.suptitle(f'SNN Fine-Tuning Performance (T={time_steps})', fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96]) 
        
        plt.savefig(f"{plot_save_prefix}_training_curve.png")
        plt.close(fig) 
        
        np.save(f"{plot_save_prefix}_history.npy", history)
        # 保存性能数据以备画 Rebuttal 的柱状图
        np.save(f"{plot_save_prefix}_performance_stats.npy", performance_stats)

    print(f"Loading best model from {model_save_path}...")
    student_snn.load_state_dict(torch.load(model_save_path))
    
    return student_snn