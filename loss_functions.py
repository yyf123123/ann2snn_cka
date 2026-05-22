import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Union
from cka_compare import CKA

class GlobalLoss(nn.Module):
    """
    全局损失 (Global Loss): 计算Teacher ANN和Student SNN输出logits之间的KL散度。
    对应论文公式 (7).
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, snn_logits: torch.Tensor, ann_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            snn_logits (torch.Tensor): Student SNN 的输出 logits.
            ann_logits (torch.Tensor): Teacher ANN 的输出 logits.
        Returns:
            torch.Tensor: 全局损失值.
        """
        soft_snn_p = F.log_softmax(snn_logits / self.temperature, dim=1)
        soft_ann_p = F.softmax(ann_logits / self.temperature, dim=1)
        
        loss = self.kl_div_loss(soft_snn_p, soft_ann_p) * (self.temperature ** 2)
        return loss

class LocalLoss(nn.Module):
    """
    局部损失 (Local Loss): 计算SNN和ANN在一系列选定层上的加权CKA损失。
    对应论文公式 (8)。
    """
    def __init__(self, w_l: Union[Dict[str, float], List[float]]):
        super().__init__()
        if not isinstance(w_l, (dict, list, tuple)):
            raise TypeError("w_l must be a dictionary, list, or tuple.")
        self.w_l = w_l

    def forward(self, snn_features: Union[Dict[str, torch.Tensor], List[torch.Tensor]], 
                      ann_features: Union[Dict[str, torch.Tensor], List[torch.Tensor]]) -> torch.Tensor:
        """
        Args:
            snn_features (Union[Dict, List]): Student SNN 的中间层特征图.
            ann_features (Union[Dict, List]): Teacher ANN 的中间层特征图.
        Returns:
            torch.Tensor: 局部损失值.
        """
        if type(snn_features) != type(ann_features):
            raise TypeError("snn_features and ann_features must be of the same type (dict or list).")
        
        total_local_loss = torch.tensor(0.0, device=next(iter(snn_features.values() if isinstance(snn_features, dict) else snn_features)).device)
        
        if isinstance(snn_features, dict):
            if not isinstance(self.w_l, dict):
                raise TypeError("If features are dicts, w_l must also be a dict.")
            if snn_features.keys() != ann_features.keys() or snn_features.keys() != self.w_l.keys():
                raise ValueError("Keys in snn_features, ann_features, and w_l must match.")
            
            for layer_name in snn_features.keys():
                cka_sim = CKA.similarity(snn_features[layer_name], ann_features[layer_name])
                loss_l = self.w_l[layer_name] * (1 - cka_sim)
                total_local_loss += loss_l

        else: # list or tuple
            if len(snn_features) != len(ann_features) or len(snn_features) != len(self.w_l):
                raise ValueError("Length of snn_features, ann_features, and w_l lists must be equal.")
            
            for i in range(len(snn_features)):
                cka_sim = CKA.similarity(snn_features[i], ann_features[i])
                loss_l = self.w_l[i] * (1 - cka_sim)
                total_local_loss += loss_l
                
        return total_local_loss

class CombinedLoss(nn.Module):
    """
    复合损失函数 (Composite Loss Function): 结合了任务损失、全局损失和局部损失。
    对应论文公式 (5)。
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
                snn_features: Union[Dict[str, torch.Tensor], List[torch.Tensor]], 
                ann_features: Union[Dict[str, torch.Tensor], List[torch.Tensor]]):
        """
        Args:
            snn_output (torch.Tensor): Student SNN 的最终输出 logits.
            ann_output (torch.Tensor): Teacher ANN 的最终输出 logits.
            labels (torch.Tensor): 真实标签 (Ground-truth labels).
            snn_features (Union[Dict, List]): Student SNN 的中间层特征图.
            ann_features (Union[Dict, List]): Teacher ANN 的中间层特征图.
        Returns:
            Tuple[torch.Tensor, ...]: (总损失, 任务损失, 全局损失, 局部损失).
        """
        l_task = self.task_loss_fn(snn_output, labels)
        l_global = self.global_loss_fn(snn_output, ann_output)
        l_local = self.local_loss_fn(snn_features, ann_features)
        
        l_total = (1 - self.alpha) * l_task + self.alpha * (self.beta * l_global + (1 - self.beta) * l_local)
        
        return l_total, l_task, l_global, l_local

