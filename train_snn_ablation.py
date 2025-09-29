import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# 导入您自定义的损失函数和评估函数
# 注意：我们只需要GlobalLoss
from loss_functions import GlobalLoss
from evaluate import evaluate_snn 

def train_snn(
    student_snn: nn.Module,
    teacher_ann: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader, 
    device: torch.device,
    time_steps: int,
    epochs: int,
    alpha: float,
    temperature: float,
    # --- 为了API兼容性，保留以下参数但不在函数内部使用 ---
    w_l_dict: Optional[Dict[str, float]] = None,
    beta: Optional[float] = None,
    # ---------------------------------------------------
    lr: float = 1e-4,
    model_save_path: str = 'best_snn_model_no_local_loss.pth',
    plot_save_prefix: Optional[str] = None
):
    """
    SNN的对照实验（消融实验）训练函数。

    此版本与原始train_snn函数签名兼容，但仅使用任务损失(L_task)和全局损失(L_global)，
    完全忽略了与局部损失(L_local)相关的 `w_l_dict` 和 `beta` 参数。
    """
    teacher_ann.eval()
    for param in teacher_ann.parameters():
        param.requires_grad = False

    student_snn.train()

    # --- 损失函数 ---
    task_loss_fn = nn.CrossEntropyLoss()
    global_loss_fn = GlobalLoss(temperature)
    
    optimizer = torch.optim.Adam(student_snn.parameters(), lr=lr)
    
    best_acc = 0.0
    history = {
        'total_loss': [],
        'task_loss': [],
        'global_loss': [],
        'accuracy': []
    }

    for epoch in range(epochs):
        student_snn.train()
        epoch_total_loss, epoch_task_loss, epoch_global_loss = 0.0, 0.0, 0.0
        
        for img, label in tqdm(train_loader, desc=f"Ablation Training Epoch {epoch + 1}/{epochs}"):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                ann_logits = teacher_ann(img)

            for m in student_snn.modules():
                if hasattr(m, 'reset'):
                    m.reset()
            
            snn_output_accumulator = 0.0
            for _ in range(time_steps):
                snn_output_accumulator += student_snn(img)

            final_snn_output = snn_output_accumulator / time_steps
            
            # --- 计算损失 (仅 L_task 和 L_global) ---
            l_task = task_loss_fn(final_snn_output, label)
            l_global = global_loss_fn(final_snn_output, ann_logits)
            
            # 新的、简化的总损失
            loss = (1 - alpha) * l_task + alpha * l_global

            loss.backward()
            optimizer.step()

            epoch_total_loss += loss.item()
            epoch_task_loss += l_task.item()
            epoch_global_loss += l_global.item()

        # --- 日志记录 ---
        num_batches = len(train_loader)
        print(f"\nEpoch {epoch + 1} finished.")
        print(f"  Avg Total Loss: {epoch_total_loss / num_batches:.4f}")
        print(f"  -> Avg Task Loss: {epoch_task_loss / num_batches:.4f}")
        print(f"  -> Avg Global Loss: {epoch_global_loss / num_batches:.4f}")
        # ==================== 内存泄漏修复 ====================
        if 'final_snn_output' in locals():
            del final_snn_output, l_task, l_global, loss
            torch.cuda.empty_cache()
        # --- 评估 ---
        print("  Evaluating on test set...")
        accuracies = evaluate_snn(student_snn, test_loader, device, time_steps)
        current_acc = accuracies[-1] 

        # --- 保存历史数据 ---
        history['total_loss'].append(epoch_total_loss / num_batches)
        history['task_loss'].append(epoch_task_loss / num_batches)
        history['global_loss'].append(epoch_global_loss / num_batches)
        history['accuracy'].append(current_acc)

        # --- 保存最佳模型 ---
        if current_acc > best_acc:
            best_acc = current_acc
            print(f"  New best accuracy: {best_acc:.2f}%. Saving model to {model_save_path}...")
            torch.save(student_snn.state_dict(), model_save_path)

    print(f"\nSNN ablation fine-tuning complete. Best accuracy achieved: {best_acc:.2f}%")

    # --- 绘图 ---
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
        fig.suptitle(f'SNN Ablation Tuning (T={time_steps})', fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"{plot_save_prefix}_training_curve.png")
        plt.close(fig)
        np.save(f"{plot_save_prefix}_history.npy", history)
    
    print(f"Loading best model from {model_save_path}...")
    student_snn.load_state_dict(torch.load(model_save_path))
    
    return student_snn
