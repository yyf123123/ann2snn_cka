import torch
import torch.nn as nn
import copy
from tqdm import tqdm



class CKA:
    """
    一个多功能的CKA类，既可以用于评估（通过inference方法），
    也提供了可用于训练的静态方法（similarity）。
    """
    def __init__(self, snn_model: nn.Module = None, ann_model: nn.Module = None, device='cuda', batch_size=1024, repeat=5, T=50):
        self.snn_model = snn_model
        self.ann_model = ann_model
        self.device = device
        self.batch_size = batch_size
        self.repeat = repeat
        self.T = T
        if self.snn_model and self.ann_model:
            self.snn_model.eval()
            self.ann_model.eval()
            self.init_var()

    def init_var(self):
        self.Ks = []
        self.Ls = []
        self.hsic_k = 0
        self.hsic_l = 0
        self.hsic_kl = 0
        self.snn_layer_names = []
        self.ann_layer_names = []
    
    # ==============================================================================
    # 新增的、可用于训练的静态方法 (NEW Static methods for training)
    # ==============================================================================
    @staticmethod
    def _center_gram(gram: torch.Tensor) -> torch.Tensor:
        if gram.size(0) != gram.size(1):
            raise ValueError("Gram matrix must be square.")
        
        mean_rows = gram.mean(dim=1, keepdim=True)
        mean_cols = gram.mean(dim=0, keepdim=True)
        mean_all = gram.mean()
        
        return gram - mean_rows - mean_cols + mean_all

    @staticmethod
    def similarity(X: torch.Tensor, Y: torch.Tensor, kernel: str = 'linear') -> torch.Tensor:
        X = X.view(X.size(0), -1)
        Y = Y.view(Y.size(0), -1)

        if kernel == 'linear':
            gram_X = torch.matmul(X, X.t())
            gram_Y = torch.matmul(Y, Y.t())
        else:
            raise NotImplementedError("RBF kernel not implemented yet.")

        centered_gram_X = CKA._center_gram(gram_X)
        centered_gram_Y = CKA._center_gram(gram_Y)

        hsic_numerator = (centered_gram_X * centered_gram_Y).sum()
        hsic_denom_X = (centered_gram_X * centered_gram_X).sum()
        hsic_denom_Y = (centered_gram_Y * centered_gram_Y).sum()

        cka_val = hsic_numerator / (torch.sqrt(hsic_denom_X * hsic_denom_Y) + 1e-9)
        return cka_val

    # ==============================================================================
    # 原有的、用于评估的方法 (ORIGINAL methods for evaluation)
    # ==============================================================================
    def hook_layer(self, is_key_layer_fn):
        def get_gram_snn(name):
            def hook(model, input, output):
                if not model.training:
                    mask = torch.eye(output.shape[0], device=self.device).bool()
                    flat_output = output.flatten(1)
                    gram = torch.matmul(flat_output, flat_output.T)
                    self.Ks.append(gram.masked_fill_(mask, 0))
                    self.snn_layer_names.append(name)
            return hook

        def get_gram_ann(name):
            def hook(model, input, output):
                if not model.training:
                    mask = torch.eye(output.shape[0], device=self.device).bool()
                    flat_output = output.flatten(1)
                    gram = torch.matmul(flat_output, flat_output.T)
                    self.Ls.append(gram.masked_fill_(mask, 0))
                    self.ann_layer_names.append(name)
            return hook
        
        for name, module in self.snn_model.named_modules():
            if is_key_layer_fn(module):
                module.register_forward_hook(get_gram_snn(name))

        for name, module in self.ann_model.named_modules():
            if is_key_layer_fn(module):
                module.register_forward_hook(get_gram_ann(name))
    
    def get_hsic(self, K, L, n, ones):
        if n < 4: return torch.tensor(0.0).to(K.device)
        # 注意: .item() 会中断梯度流，因此这个方法不适用于训练
        return (torch.trace(K @ L) + (ones.T @ K @ ones * ones.T @ L @ ones) / ((n - 1) * (n - 2)) -
                2 * (ones.T @ K @ L @ ones) / (n - 2)).item() / (n * (n - 3))

    @torch.no_grad()
    def inference(self, loader):
        iter_loader = iter(loader)
        final_snn_names, final_ann_names = [], []
        self.hsic_k, self.hsic_l, self.hsic_kl = 0, 0, 0

        for i in range(self.repeat):
            data, label = next(iter_loader)
            data = data.to(self.device)
            data2 = copy.deepcopy(data)

            self.Ls = []
            self.ann_layer_names = []
            self.ann_model(data2)
            
            for m in self.snn_model.modules():
                if hasattr(m, 'reset'):
                    m.reset()

            k_over_time = [] 
            for t in range(self.T):
                self.Ks = []
                self.snn_layer_names = []
                self.snn_model(data)
                k_over_time.append(self.Ks)
                if t == 0:
                    final_snn_names = self.snn_layer_names.copy()

            num_layers = len(final_snn_names)
            avg_Ks = []
            for l_idx in range(num_layers):
                layer_l_grams = torch.stack([k_over_time[t][l_idx] for t in range(self.T)])
                avg_Ks.append(layer_l_grams.mean(dim=0))
            
            self.Ks = avg_Ks
            
            if i == 0:
                final_ann_names = self.ann_layer_names.copy()
            
            n = self.batch_size
            ones = torch.ones((n, 1), device=self.device)

            hsic_k = [self.get_hsic(K, K, n, ones) for K in self.Ks]
            hsic_l = [self.get_hsic(L, L, n, ones) for L in self.Ls]
            hsic_kl = [[self.get_hsic(K, L, n, ones) for K in self.Ks] for L in self.Ls]

            torch.cuda.empty_cache()

            self.hsic_k += torch.tensor(hsic_k, device=self.device)
            self.hsic_l += torch.tensor(hsic_l, device=self.device)
            self.hsic_kl += torch.tensor(hsic_kl, device=self.device)
        
        self.hsic_k /= self.repeat
        self.hsic_l /= self.repeat
        self.hsic_kl /= self.repeat

        self.hsic_k = torch.sqrt(torch.abs(self.hsic_k))
        self.hsic_l = torch.sqrt(torch.abs(self.hsic_l))

        l_k = self.hsic_k.numel()
        l_l = self.hsic_l.numel()

        denominator = self.hsic_l.reshape(l_l, 1) @ self.hsic_k.reshape(1, l_k)
        hsic_matrix = self.hsic_kl.squeeze() / (denominator + 1e-6)
        
        return hsic_matrix, final_snn_names, final_ann_names
