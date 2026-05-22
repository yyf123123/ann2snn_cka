import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
import spikingjelly.activation_based.ann2snn as ann2snn
import model_cifar10_resnet
from spikingjelly.activation_based import neuron, surrogate, layer
from cka_compare import CKA
# 从新的 models.py 中导入所有需要的组件
from models import SNNBasicBlock, RebuiltSNNResNet, rebuild_snn_resnet18
from evaluate import evaluate_ann, evaluate_snn
from train_snn import train_snn

import torchvision
import torch

def get_data_loaders(batch_size: int, data_dir: str = '/home/lbz/git-hub/datasets'):
    """准备CIFAR-10的数据加载器，包含数据增强。"""
    print("正在准备数据加载器...")

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.AutoAugment(torchvision.transforms.AutoAugmentPolicy.CIFAR10),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return trainloader, testloader

def main():
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = 'cuda:0'
    use_pretrained = True
    dataset_dir = '/home/lbz/git-hub/datasets'
    # 自己从0训练的ANN模型，96.39%的准确率
    weights_path = '/home/lbz/git-hub/pretrained_models/ann_resnet18_cifar10_best.pth'
    snn_save_path = '/home/lbz/git-hub/pretrained_models/MY-cifar10-resnet18_SNN.pth'

    batch_size = 100
    T=2
    fine_tune_epochs = 50
    learning_rate = 1e-5
    alpha = 0.5
    beta = 0.2
    temperature = 2.0
    snnmodel_save_path = f"(0329T={T}_{alpha},{beta})best_snn_model.pth"
    # 数据加载器
    train_data_loader, test_data_loader = get_data_loaders(batch_size, dataset_dir)

    model = model_cifar10_resnet.ResNet18()
    # 如果 use_pretrained = True 就使用model.load_state_dict()。否则使用train函数把权重保存到weights_path地址。
    model.load_state_dict(torch.load(weights_path))

    print('ANN accuracy: ')
    model.to(device)
    acc_ann = evaluate_ann(model, test_data_loader, device)
    # print(f'Validating Accuracy: {acc_ann:.2f}%')

    # print('---------------------------------------------')
    # print('Converting using 99.9% RobustNorm')
    # model_converter = ann2snn.Converter(mode='99.9%', dataloader=train_data_loader)
    # snn_model = model_converter(model)
    # print(f'(T={T})Spikingjelly_snn_model accuracy:  ')
    # acc_snn = evaluate_snn(snn_model, test_data_loader, device, time_steps = T, save_path_prefix = None)

    # rebuilt_snn = rebuild_snn_resnet18(model, snn_model)
    # rebuilt_snn.to(device)

    # torch.save(rebuilt_snn, snn_save_path)
    # print('SNN structure rebuilt successfully.save path:', snn_save_path)


    snn_loaded_model = torch.load(snn_save_path, weights_only=False)
    # print(snn_loaded_model)
    snn_loaded_model.eval()
    print('---------------------------------------------')
    print(f'(T={T})loaded_snn_model accuracy:  ')
    acc_snn = evaluate_snn(snn_loaded_model, test_data_loader, device, time_steps = T, save_path_prefix = None)
    # print(f'Validating Accuracy: {acc_snn[-1]:.2f}%')
    snn_loaded_model.to(device)

    # 第一个参数是snn第二个参数是ann
    cka = CKA(snn_loaded_model, model, device, batch_size, repeat=5, T=T)
    print('Registering hook to model key layers...')
    # 1. 在外部定义用于判断关键层的函数
    #    这个函数就是之前在 hook_layer 内部的 is_key_layer
    def is_resnet_basic_block(m):
        # 根据您的模型结构，判断哪些层是您想要比较的关键层
        return isinstance(m, (model_cifar10_resnet.BasicBlock, SNNBasicBlock))

    # 2. 将这个函数作为参数传入 hook_layer
    cka.hook_layer(is_key_layer_fn=is_resnet_basic_block)

    print('Computing centered kernel alignment for key layers...')
    hsic, model1_names, model2_names = cka.inference(loader=test_data_loader)
    
    hsic = hsic.cpu().numpy()
    print("HSIC matrix between matched layers:")
    # ==================== 新增代码：初始化 w_l 权重 ====================

    # 1. 提取初始CKA相似度 (对角线元素)
    # hsic 是一个 (L, L) 的矩阵，L 是监控的层数
    # 我们假设 model1_names 和 model2_names 中的层是一一对应的
    # cka_initial[l] 对应 ANN 和 SNN 第 l 个匹配层的 CKA 相似度
    cka_initial = np.diag(hsic)

    np.set_printoptions(precision=3, suppress=True)
    print("Initial CKA similarity for matched layers (diagonal of HSIC matrix):")
    print(cka_initial)

    # 2. 根据论文公式(9)计算 1 - CKA_initial(l)
    one_minus_cka = 1 - cka_initial
    # 3. 计算分母：sum(1 - CKA_initial(j))
    denominator = np.sum(one_minus_cka)
    if denominator == 0:
        # 防止除以零，如果所有层都完美对齐，则平均分配权重
        num_layers = len(one_minus_cka)
        w_l = np.full(num_layers, 1.0 / num_layers)
    else:
        # 计算每个层的权重 w_l
        w_l = one_minus_cka / denominator

    # print("\nCalculated weights w_l for LocalLoss:")
    # print(w_l)

    # ====== 可选：统一权重 ======
    use_uniform_w_l = True
    if use_uniform_w_l:
        num_layers = len(w_l)
        w_l = np.full(num_layers, 1.0 / num_layers, dtype=np.float32)
        print("\nUniform weighting w_l used for LocalLoss:")

    # 将权重与层名对应，方便后续使用
    w_l_dict = {name: weight for name, weight in zip(model1_names, w_l)}
    print("\nWeights dictionary (w_l_dict):")
    print(w_l_dict)

    output_dir_cka = 'cka_results'
    os.makedirs(output_dir_cka, exist_ok=True)
    
    cka_results = {
        'cka_matrix': hsic,
        'snn_layer_names': model1_names,
        'ann_layer_names': model2_names
    }
    
    save_path = os.path.join(output_dir_cka, f'(T={T})key_layers_asnn_res18.npy')
    np.save(save_path, cka_results)
    
    print(f"CKA results for key layers saved to: {save_path}")
    print("\nStarting Closed-Loop Fine-Tuning for SNN...")

    # 设置微调的超参数 (移到最开头)

    plot_save_prefix = f"(T={T}_{alpha},{beta})"

    # 打印训练参数
    print("--- Training Parameters ---")
    print(f"Learning Rate: {learning_rate}")
    print(f"Alpha:         {alpha}")
    print(f"Beta:          {beta}")
    print(f"Temperature:   {temperature}")
    print("--------------------------")

# =========================================================================
    # 修改点 1：静态 VS 动态 CKA 训练开销对比 (调换顺序，Epoch=50)
    # =========================================================================
    model.eval()
    for param in model.parameters(): param.requires_grad = False

    print("\n" + "="*50)
    print("🚀 [Ablation] Static vs Dynamic CKA Cost Comparison")
    print("="*50)
    
    ablation_epochs = 50
    finetuned_snn_model = None 
    memory_peaks = {"Static": 0.0, "Dynamic": 0.0} 
    
    # 核心修改：先跑 Static (False)，再跑 Dynamic (True)
    for is_dynamic in [False, True]:
        mode_name = "Dynamic" if is_dynamic else "Static"
        print(f"\n>>> Running {mode_name} CKA mode for {ablation_epochs} epochs...")
        
        # 强制垃圾回收并重置峰值记录，确保两者完全隔离
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        student_model_copy = copy.deepcopy(snn_loaded_model)
        
        current_model = train_snn(
            student_snn=student_model_copy,
            teacher_ann=model,
            train_loader=train_data_loader, 
            test_loader=test_data_loader,
            device=device,
            time_steps=T,
            epochs=ablation_epochs, 
            w_l_dict=w_l_dict,
            alpha=alpha,
            beta=beta,
            temperature=temperature,
            lr=learning_rate,
            model_save_path=f"ablation_{mode_name}_snn.pth",
            plot_save_prefix=f"{plot_save_prefix}_{mode_name}",
            is_dynamic_cka=is_dynamic,
            cka_obj=cka 
        )

        # 记录该模式下的显存真实峰值
        if torch.cuda.is_available():
            memory_peaks[mode_name] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        
        if not is_dynamic:
            finetuned_snn_model = copy.deepcopy(current_model)
            
        # 跑完当前模式后，彻底销毁对象
        del current_model
        del student_model_copy

    print("\nFine-tuning finished. Evaluating the fine-tuned (Static) SNN...")
    evaluate_snn(finetuned_snn_model, test_data_loader, device, T, save_path_prefix = None)
    
    print("\n开始计算最终微调的模型的CKA值")
    cka_final = CKA(finetuned_snn_model, model, device, batch_size, repeat=5, T=T)
    cka_final.hook_layer(is_key_layer_fn=is_resnet_basic_block)
    hsic_final, _, _ = cka_final.inference(loader=test_data_loader)
    hsic_final = hsic_final.cpu().numpy()
    cka_final_diag = np.diag(hsic_final)
  
    cka_results = {
        'cka_matrix': hsic_final,
        'snn_layer_names': model1_names,
        'ann_layer_names': model2_names
    }
    save_path = os.path.join(output_dir_cka, f'(T={T}_{alpha},{beta})key_layers_finetune_res18.npy')
    np.save(save_path, cka_results)
    
    print(f"CKA results for key layers saved to: {save_path}")
    print("打印最终微调的模型的CKA值for matched layers (diagonal of HSIC matrix):")
    print(cka_final_diag)

    # =========================================================================
    # 修改点 2 & 3: 硬件能效评估与 50 Epoch 微调成本悖论回击 (引入 Forward+Backward)
    # =========================================================================
    print("\n" + "="*50)
    print("🔋 [Rebuttal] Hardware Efficiency & Fine-Tuning Cost Evaluation")
    print("="*50)
    
    from spikingjelly.activation_based.monitor import OutputMonitor
    from spikingjelly.activation_based import neuron
    
    print("1. Calculating Average Spike Rate (R)...")
    finetuned_snn_model.eval()
    monitor = OutputMonitor(finetuned_snn_model, neuron.BaseNode)
    monitor.enable()
    monitor.clear_recorded_data()
    
    with torch.no_grad():
        for m in finetuned_snn_model.modules():
            if hasattr(m, 'reset'): m.reset()
        img, _ = next(iter(test_data_loader))
        img = img.to(device)
        for t in range(T):
            finetuned_snn_model(img)
            
    total_spikes = 0
    total_neurons = 0
    for record in monitor.records:
        total_spikes += record.sum().item()
        total_neurons += record.numel()
        
    avg_spike_rate = total_spikes / total_neurons
    print(f"   Average Spike Rate (R): {avg_spike_rate:.4f} ({avg_spike_rate*100:.2f}%)")
    monitor.remove_hooks() 
    
    print("2. Calculating Theoretical Training Energy Cost (Forward + Backward)...")
    try:
        from thop import profile
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        F_ann, _ = profile(model, inputs=(dummy_input, ), verbose=False)
    except ImportError:
        F_ann = 5.56e8
        
    SOPs = F_ann * T * avg_spike_rate
    
    E_mac = 4.6 
    E_add = 0.9 
    N_epochs_STBP = 300
    N_epochs_Ours = 50 
    
    # 核心公式更新：真正的训练包含 Forward 和 Backward (约等于 Forward 的 2 倍)
    # STBP: 300 Epochs * (SNN 前向加法 + 2倍 SNN 反向加法)
    Cost_STBP_Energy = N_epochs_STBP * (SOPs * E_add + 2 * SOPs * E_add) 
    
    # Ours: 50 Epochs * (ANN前向乘加 + SNN前向加法 + 2倍 SNN反向加法) -> ANN 冻结无反向
    Cost_Ours_Energy = N_epochs_Ours * (F_ann * E_mac + SOPs * E_add + 2 * SOPs * E_add)
    ratio = Cost_Ours_Energy / Cost_STBP_Energy

    # =========================================================================
    # 最终数据对比总结版块
    # =========================================================================
    print("\n" + "="*50)
    print("📊 [Data Summary] Key Metrics Comparison")
    print("="*50)
    print("1. GPU Memory Peak (Ablation Phase):")
    print(f"   - Static CKA Mode:  {memory_peaks['Static']:.2f} MB")
    print(f"   - Dynamic CKA Mode: {memory_peaks['Dynamic']:.2f} MB")
    
    print("\n2. Hardware Training Energy Estimations (per image/epoch base):")
    print(f"   - ANN FLOPs (F_ann, Forward): {F_ann / 1e9:.4f} G")
    print(f"   - SNN SOPs  (Forward):        {SOPs / 1e9:.4f} G")
    print(f"   * Note: Backward pass cost is mathematically modeled as 2x Forward pass.")
    print(f"   - STBP Energy ({N_epochs_STBP} Ep) = {Cost_STBP_Energy / 1e9:.4f} mJ")
    print(f"   - Ours Energy ({N_epochs_Ours} Ep) = {Cost_Ours_Energy / 1e9:.4f} mJ")
    print(f"   - Final Training Energy Ratio (Ours/STBP): {ratio*100:.2f}%")
    print("="*50)

if __name__ == '__main__':
    main()