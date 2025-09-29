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

# def rebuild_resnet_structure(converted_snn_model, original_resnet):
#     """
#     This function's only job is now to parse the flat model and pass the
#     resulting dictionaries to the RebuiltSNNResNet class constructor.
#     """
#     snn_modules = dict(converted_snn_model.named_modules())
#     snn_tailor_modules = {}
    
#     # Parsing logic remains the same...
#     for name, module in snn_modules.items():
#         if 'snn tailor' in name:
#             # ... (the parsing code is unchanged)
#             name_parts = name.split('.')
#             if len(name_parts) >= 3:
#                 try:
#                     for part in name_parts:
#                         if part.isdigit():
#                             idx = int(part)
#                             if name_parts[-1].isdigit():
#                                 parent_idx = idx
#                                 sub_idx = int(name_parts[-1])
#                                 key = f"{parent_idx}.{sub_idx}"
#                                 snn_tailor_modules[key] = module
#                             break
#                 except (ValueError, IndexError):
#                     print(f"Warning: Could not parse index from {name}")
#                     continue

#     # ✅ The function no longer defines a local helper function.
#     # It just passes the two prepared dictionaries to the class constructor.
#     return RebuiltSNNResNet(snn_modules=snn_modules,
#                             snn_tailor_modules=snn_tailor_modules)

def main():
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = 'cuda:3'
    use_pretrained = True
    dataset_dir = '/home/lbz/git-hub/datasets'
    weights_path = '/home/lbz/git-hub/pretrained_models/SJ-cifar10-resnet18_model-sample.pth'
    snn_save_path = '/home/lbz/git-hub/pretrained_models/SJ-cifar10-resnet18_SNN.pth'
    batch_size = 100
    T=4
    fine_tune_epochs = 100
    learning_rate = 1e-5
    alpha = 0.5
    beta = 1.0
    temperature = 2.0

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    model = model_cifar10_resnet.ResNet18()
    # 如果 use_pretrained = True 就使用model.load_state_dict()。否则使用train函数把权重保存到weights_path地址。
    model.load_state_dict(torch.load(weights_path))

    train_data_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir,
        train=True,
        transform=transform,
        download=True)
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_data_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False)
    test_data_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir,
        train=False,
        transform=transform,
        download=True)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_data_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False)


    print('ANN accuracy: 94.57%')
    model.to(device)
    # acc_ann = evaluate_ann(model, test_data_loader, device)
    # print(f'Validating Accuracy: {acc_ann:.2f}%')

    # print('---------------------------------------------')
    # print('Converting using 99.9% RobustNorm')
    # model_converter = ann2snn.Converter(mode='99.9%', dataloader=train_data_loader)
    # snn_model = model_converter(model)
    # rebuilt_snn = rebuild_snn_resnet18(snn_model)
    # print(rebuilt_snn)
    # rebuilt_snn.to(device)
    # acc_snn = evaluate_snn(rebuilt_snn, test_data_loader, device, time_steps = T, save_path_prefix = None)
    # 新增保存
    # torch.save(rebuilt_snn, snn_save_path)
    # print('SNN structure rebuilt successfully.save path:', snn_save_path)
    # print(model)
    # print(snn_model)
    
    # 加载
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

    # ==================== 新增代码：初始化 w_l 权重 ====================

    # 1. 提取初始CKA相似度 (对角线元素)
    # hsic 是一个 (L, L) 的矩阵，L 是监控的层数
    # 我们假设 model1_names 和 model2_names 中的层是一一对应的
    # cka_initial[l] 对应 ANN 和 SNN 第 l 个匹配层的 CKA 相似度
    cka_initial = np.diag(hsic)
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

    # 将权重与层名对应，方便后续使用
    w_l_dict = {name: weight for name, weight in zip(model1_names, w_l)}
    # print("\nWeights dictionary (w_l_dict):")
    # print(w_l_dict)

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
    
    snnmodel_save_path = f"(T={T})best_snn_model.pth"
    plot_save_prefix = f"(T={T})"

    # 打印训练参数
    print("--- Training Parameters ---")
    print(f"Learning Rate: {learning_rate}")
    print(f"Alpha:         {alpha}")
    print(f"Beta:          {beta}")
    print(f"Temperature:   {temperature}")
    print("--------------------------")

    # 确保教师模型处于评估模式且参数被冻结
    model.eval()
    for param in model.parameters(): param.requires_grad = False

    finetuned_snn_model = train_snn(
        student_snn=snn_loaded_model,
        teacher_ann=model,
        train_loader=train_data_loader, # 使用训练数据加载器
        test_loader=test_data_loader,
        device=device,
        time_steps=T,
        epochs=fine_tune_epochs,
        w_l_dict=w_l_dict,
        alpha=alpha,
        beta=beta,
        temperature=temperature,
        lr=learning_rate,
        model_save_path = snnmodel_save_path,
        plot_save_prefix = plot_save_prefix
    )
    print("\nFine-tuning finished. Evaluating the fine-tuned SNN...")
    # 使用您已有的评估函数来查看训练后的效果
    evaluate_snn(finetuned_snn_model, test_data_loader, device, T, save_path_prefix = None)

    
    print("\n开始计算最终微调的模型的CKA值")
    cka_final = CKA(finetuned_snn_model, model, device, batch_size, repeat=5, T=T)

    # 2. 将这个函数作为参数传入 hook_layer
    cka_final.hook_layer(is_key_layer_fn=is_resnet_basic_block)
    hsic_final, _, _ = cka_final.inference(loader=test_data_loader)

    hsic_final = hsic_final.cpu().numpy()

    cka_final = np.diag(hsic_final)
    print("打印最终微调的模型的CKA值for matched layers (diagonal of HSIC matrix):")
    print(cka_final)


if __name__ == '__main__':
    main()

