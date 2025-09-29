import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from loss_functions import GlobalLoss, LocalLoss, CKA # 引入CKA用于计算
from evaluate import evaluate_snn 


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
    """
    使用“边算边丢”策略，高效地对SNN进行微调训练，修复了内存泄漏和逻辑bug。
    """
    teacher_ann.eval()
    for param in teacher_ann.parameters():
        param.requires_grad = False

    student_snn.train()

    # 初始化各个损失函数
    task_loss_fn = nn.CrossEntropyLoss()
    global_loss_fn = GlobalLoss(temperature)
    local_loss_fn = LocalLoss(w_l_dict)
    
    optimizer = torch.optim.Adam(student_snn.parameters(), lr=lr)
    
    best_acc = 0.0
    history = {'total_loss': [], 'task_loss': [], 'global_loss': [], 'local_loss': [], 'accuracy': []}

    # --- 为特征提取注册钩子 ---
    layer_names = list(w_l_dict.keys())
    snn_features, ann_features = {}, {}

    snn_hooks = []
    def get_snn_hook(name):
        def hook(model, input, output):
            snn_features[name] = output
        return hook
    for name, module in student_snn.named_modules():
        if name in layer_names:
            snn_hooks.append(module.register_forward_hook(get_snn_hook(name)))

    ann_hooks = []
    def get_ann_hook(name):
        def hook(model, input, output):
            ann_features[name] = output
        return hook
    for name, module in teacher_ann.named_modules():
        if name in layer_names:
            ann_hooks.append(module.register_forward_hook(get_ann_hook(name)))
            
    for epoch in range(epochs):
        student_snn.train() 
        epoch_total_loss, epoch_task_loss, epoch_global_loss, epoch_local_loss = 0.0, 0.0, 0.0, 0.0
        
        for img, label in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()

            # 1. 获取一次教师ANN的特征和输出
            ann_features.clear()
            with torch.no_grad():
                ann_logits = teacher_ann(img)
            
            # 2. 重置SNN状态
            for m in student_snn.modules():
                if hasattr(m, 'reset'):
                    m.reset()
            
            # 3. 逐时间步计算和累积
            snn_output_accumulator = 0.0
            local_loss_accumulator = 0.0
            
            for _ in range(time_steps):
                snn_features.clear()
                snn_output_single_step = student_snn(img)
                snn_output_accumulator += snn_output_single_step
                
                # --- 核心修改：在每个时间步计算并累积局部损失 ---
                # 这里的snn_features包含了当前时间步的带梯度输出
                # local_loss_fn内部的CKA计算图在每个时间步结束后就会被丢弃
                local_loss_accumulator += local_loss_fn(snn_features, ann_features)

            # 4. 计算平均值
            final_snn_output = snn_output_accumulator / time_steps
            final_local_loss = local_loss_accumulator / time_steps

            # 5. 计算其他损失并组合
            l_task = task_loss_fn(final_snn_output, label)
            l_global = global_loss_fn(final_snn_output, ann_logits)
            l_local = final_local_loss # 使用累积并平均后的局部损失
            
            loss = (1 - alpha) * l_task + alpha * (beta * l_global + (1 - beta) * l_local)

            loss.backward()
            optimizer.step()

            # 记录损失
            epoch_total_loss += loss.item()
            epoch_task_loss += l_task.item()
            epoch_global_loss += l_global.item()
            epoch_local_loss += l_local.item()

        # ... (后续的日志打印、评估、保存模型和绘图代码保持不变) ...
        num_batches = len(train_loader)
        print(f"Epoch {epoch + 1} finished.")
        print(f"  Avg Total Loss: {epoch_total_loss / num_batches:.4f}")
        # ... (其他打印)

        accuracies = evaluate_snn(student_snn, test_loader, device, time_steps)
        current_acc = accuracies[-1] 
        history['accuracy'].append(current_acc)
        # ... (保存最佳模型)

    # 训练结束后移除钩子，防止内存泄漏
    for hook in snn_hooks + ann_hooks:
        hook.remove()

    # ... (绘图和返回模型的代码) ...
    print(f"Loading best model from {model_save_path}...")
    student_snn.load_state_dict(torch.load(model_save_path))
    return student_snn