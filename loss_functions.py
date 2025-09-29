import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Union
from cka_compare import CKA

class GlobalLoss(nn.Module):
    """
    全局损失 (Global Loss): 计算Teacher ANN和Student SNN输出logits之间的KL散度。
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, snn_logits: torch.Tensor, ann_logits: torch.Tensor) -> torch.Tensor:
        soft_snn_p = F.log_softmax(snn_logits / self.temperature, dim=1)
        soft_ann_p = F.softmax(ann_logits / self.temperature, dim=1)
        
        loss = self.kl_div_loss(soft_snn_p, soft_ann_p) * (self.temperature ** 2)
        return loss

class LocalLoss(nn.Module):
    """
    局部损失 (Local Loss): 计算SNN和ANN在一系列选定层上的加权CKA损失。
    """
    def __init__(self, w_l: Union[Dict[str, float], List[float]]):
        super().__init__()
        self.w_l = w_l

    def forward(self, snn_features: Dict[str, torch.Tensor], 
                      ann_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        
        total_local_loss = torch.tensor(0.0, device=next(iter(snn_features.values())).device)
        
        for layer_name in snn_features.keys():
            # --- THE FIX for Memory Leak ---
            # Call the new memory-efficient cka_train method instead of the standard similarity.
            cka_sim = CKA.cka_train(snn_features[layer_name], ann_features[layer_name])
            loss_l = self.w_l[layer_name] * (1 - cka_sim)
            total_local_loss += loss_l
                
        return total_local_loss

class CombinedLoss(nn.Module):
    """
    复合损失函数 (Composite Loss Function): 结合了任务损失、全局损失和局部损失。
    """
    def __init__(self, w_l: Union[Dict[str, float], List[float]], 
                 alpha: float, beta: float, temperature: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
        self.task_loss_fn = nn.CrossEntropyLoss()
        self.global_loss_fn = GlobalLoss(temperature)
        self.local_loss_fn = LocalLoss(w_l)

    def forward(self, snn_output: torch.Tensor, ann_output: torch.Tensor, 
                labels: torch.Tensor, 
                snn_features: Dict[str, torch.Tensor], 
                ann_features: Dict[str, torch.Tensor]):
        
        l_task = self.task_loss_fn(snn_output, labels)
        l_global = self.global_loss_fn(snn_output, ann_output)
        l_local = self.local_loss_fn(snn_features, ann_features)
        
        l_total = (1 - self.alpha) * l_task + self.alpha * (self.beta * l_global + (1 - self.beta) * l_local)
        
        return l_total, l_task, l_global, l_local

