import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# 导入您自定义的损失函数和评估函数
from loss_functions import CombinedLoss
from evaluate import evaluate_snn 

class FeatureExtractor:
    """
    一个辅助类，用于通过PyTorch hooks提取模型中间层的特征图。
    这个修正版本可以正确处理SNN的梯度，并用于训练。
    """
    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = layer_names
        self.features: Dict[str, torch.Tensor] = {}
        self._hooks = []

    def _create_hook(self, name: str, is_snn: bool):
        def hook(model, input, output):
            if is_snn:
                # --- THE FIX for Logic Bug ---
                # We NO LONGER detach the output.
                # This allows gradients to flow back through the features for local loss.
                if name not in self.features:
                    self.features[name] = output
                else:
                    self.features[name] += output
            else:
                # For ANN, we are in a no_grad context, so this is fine.
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
        self.features.clear()

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
    plot_save_prefix: Optional[str] = None
):
    teacher_ann.eval()
    for param in teacher_ann.parameters():
        param.requires_grad = False

    criterion = CombinedLoss(w_l=w_l_dict, alpha=alpha, beta=beta, temperature=temperature)
    optimizer = torch.optim.Adam(student_snn.parameters(), lr=lr)

    layer_names = list(w_l_dict.keys())
    snn_extractor = FeatureExtractor(student_snn, layer_names)
    ann_extractor = FeatureExtractor(teacher_ann, layer_names)
    
    best_acc = 0.0
    history = {'total_loss': [], 'task_loss': [], 'global_loss': [], 'local_loss': [], 'accuracy': []}

    for epoch in range(epochs):
        student_snn.train()
        epoch_total_loss, epoch_task_loss, epoch_global_loss, epoch_local_loss = 0.0, 0.0, 0.0, 0.0
        
        for img, label in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad(set_to_none=True) # Use set_to_none=True for slightly better performance

            # 1. Get Teacher ANN features and logits (within no_grad context)
            ann_extractor.clear_features()
            ann_extractor.register_hooks(is_snn=False)
            with torch.no_grad():
                ann_logits = teacher_ann(img)
            ann_features = ann_extractor.features
            ann_extractor.remove_hooks()

            # 2. Get Student SNN features and logits
            snn_extractor.clear_features()
            snn_extractor.register_hooks(is_snn=True)
            
            # Reset SNN states
            for m in student_snn.modules():
                if hasattr(m, 'reset'):
                    m.reset()
            
            snn_output_accumulator = 0.0
            # The forward pass over T steps populates snn_extractor.features
            for _ in range(time_steps):
                snn_output_accumulator += student_snn(img)

            # 3. Average the outputs and features
            final_snn_output = snn_output_accumulator / time_steps
            summed_snn_features = snn_extractor.features
            avg_snn_features = {name: feats / time_steps for name, feats in summed_snn_features.items()}
            snn_extractor.remove_hooks()

            # 4. Calculate loss
            # The avg_snn_features tensor now correctly has a computation graph attached.
            # The CKA function is now numerically stable.
            loss, l_task, l_global, l_local = criterion(
                snn_output=final_snn_output,
                ann_output=ann_logits,
                labels=label,
                snn_features=avg_snn_features,
                ann_features=ann_features
            )

            # --- THE FIX for Memory Leak ---
            # By calculating loss on the averaged features, the massive graph
            # connecting all T time steps is resolved into a single graph node before
            # the complex CKA calculation. This graph is released after optimizer.step().
            loss.backward()
            optimizer.step()

            epoch_total_loss += loss.item()
            epoch_task_loss += l_task.item()
            epoch_global_loss += l_global.item()
            epoch_local_loss += l_local.item()

        # ... (logging, evaluation, and plotting code remains the same) ...
        num_batches = len(train_loader)
        print(f"\nEpoch {epoch + 1} finished.")
        print(f"  Avg Total Loss: {epoch_total_loss / num_batches:.4f}")
        print(f"  -> Avg Task Loss: {epoch_task_loss / num_batches:.4f}")
        print(f"  -> Avg Global Loss: {epoch_global_loss / num_batches:.4f}")
        print(f"  -> Avg Local Loss: {epoch_local_loss / num_batches:.4f}")

        # ==================== 内存泄漏修复 ====================
        # 在评估前，手动清空提取器中持有的、来自最后一个训练批次的
        # 带有计算图的特征张量，从而释放显存。
        snn_extractor.clear_features()
        # ========================================================

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

    print(f"\nSNN fine-tuning complete. Best accuracy achieved: {best_acc:.2f}%")

    if plot_save_prefix and epochs > 0:
        # ... (plotting code) ...
        pass
    
    print(f"Loading best model from {model_save_path}...")
    student_snn.load_state_dict(torch.load(model_save_path))
    
    return student_snn
