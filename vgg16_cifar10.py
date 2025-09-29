import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import spikingjelly.activation_based.ann2snn as ann2snn

# 导入VGG模型定义
from model_cifar10_vgg import vgg16_bn_cifar10,vgg16_cifar10
# 从更新后的models.py导入VGG的SNN重建函数
from models import rebuild_snn_vgg
from cka_compare import CKA
from evaluate import evaluate_snn,evaluate_ann
from train_snn import train_snn


def get_data_loaders(batch_size: int, data_dir: str = '/home/lbz/git-hub/datasets'):
    """准备CIFAR-10的数据加载器，包含数据增强。"""
    print("正在准备数据加载器...")
    # 为达到最佳性能，我们使用ImageNet的均值和标准差进行归一化
    # 同时应用了 강력한 데이터 증강 (strong data augmentation)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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
    # --- 实验配置 ---
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = 'cuda:1' # 您可以根据需要修改GPU ID
    dataset_dir = '/home/lbz/git-hub/datasets' # 数据集路径
    # SNN 和 微调参数
    batch_size = 100
    T = 20 # 时间步长
    fine_tune_epochs = 20
    learning_rate = 1e-5
    alpha = 0.5
    beta = 0.5
    temperature = 2.0
    # 使用您提供的预训练VGG模型路径
    weights_path = '/home/lbz/git-hub/my_test/VGG_cifar10_cka/pretrained_models/ann_vgg16_woBN_cifar10_best.pth'
    snn_save_path = '/home/lbz/git-hub/spikingjelly_CKAvgg/SNN_models/SJ-cifar10-vgg16_woBNSNN.pth'
    snnmodel_save_path = f"(T={T})best_snn_vgg16_woBN_model.pth"

    # --- 配置结束 ---

    # 数据加载器
    train_data_loader, test_data_loader = get_data_loaders(batch_size, dataset_dir)

    # 1. 加载预训练的ANN (VGG16)
    model = vgg16_cifar10(pretrained=False) # 创建神经网络结构，可以选择是否带有BN层的VGG16
    model.load_state_dict(torch.load(weights_path, map_location=device)) # 加载权重
    model.to(device)
    print(f'成功加载预训练VGG16模型，教师ANN准确率: with_BN = 95.02%  w/oBN = 94.65%')
    # acc_ann = evaluate_ann(model, test_data_loader, device)
    # print(f'Validating Accuracy: {acc_ann:.2f}%')

    # print('---------------------------------------------')
    # print('正在使用 99.9% RobustNorm 进行转换...')
    # converter = ann2snn.Converter(mode='99.9%', dataloader=train_data_loader)
    # snn_model = converter(model)
    # # 4. 评估转换后的初始SNN
    # print(f'评估SJ框架转换后的初始SNN (T={T})...')
    # evaluate_snn(snn_model, test_data_loader, device, time_steps=T)
    # # 3. 重建SNN结构
    # # rebuilt_snn = rebuild_snn_vgg(snn_model)
    # rebuilt_snn = rebuild_snn_vgg(model, snn_model) 
    # rebuilt_snn.to(device)

    # print(f'评估重建后的初始SNN (T={T})...')
    # evaluate_snn(rebuilt_snn, test_data_loader, device, time_steps=T)
    # torch.save(rebuilt_snn, snn_save_path)
    # print('SNN structure rebuilt successfully.save path:', snn_save_path)

    # 加载
    snn_loaded_model = torch.load(snn_save_path, weights_only=False)
    snn_loaded_model.eval()
    print('---------------------------------------------')
    print(f'(T={T})loaded_snn_model accuracy:  ')
    acc_snn = evaluate_snn(snn_loaded_model, test_data_loader, device, time_steps = T, save_path_prefix = None)
    snn_loaded_model.to(device)
    # 5. 计算初始CKA相似度以确定局部损失的权重 w_l
    cka = CKA(snn_loaded_model, model, device, batch_size, repeat=5, T=T)
    print('正在为关键层注册钩子...')

    # 为VGG网络定义关键层：所有卷积层
    def is_vgg_key_layer(m):
        return isinstance(m, nn.Conv2d)

    cka.hook_layer(is_key_layer_fn=is_vgg_key_layer)

    print('正在计算初始CKA...')
    hsic, model1_names, model2_names = cka.inference(loader=test_data_loader)
    hsic = hsic.cpu().numpy()
    
# 根据初始CKA计算w_l (逻辑与ResNet脚本完全相同)
    cka_initial = np.diag(hsic)
    one_minus_cka = 1 - cka_initial
    denominator = np.sum(one_minus_cka)
    w_l = one_minus_cka / denominator if denominator > 0 else np.full_like(one_minus_cka, 1.0 / len(one_minus_cka))
    w_l_dict = {name: weight for name, weight in zip(model1_names, w_l)}
    
    # --- 修改后的打印部分 ---
    # 打印SNN关键层名称和它们对应的初始CKA值
    print("\n--- 初始CKA相似度 ---")
    print("已成功计算初始CKA值，将用于确定局部损失权重。")
    print(f"SNN关键层名称: {model1_names}")
    print(f"对应的初始CKA值: {cka_initial}")
    print("--------------------")

    # 打印训练参数
    print("--- Training Parameters ---")
    print(f"Learning Rate: {learning_rate}")
    print(f"Alpha:         {alpha}")
    print(f"Beta:          {beta}")
    print(f"Temperature:   {temperature}")
    print("--------------------------")

    # 6. 使用闭环方案进行微调
    print("\n开始SNN的闭环微调...")
    plot_save_prefix = f"VGG_(T={T})"
    
    # 确保教师模型处于评估模式
    model.eval()
    for param in model.parameters(): param.requires_grad = False

    finetuned_snn_model = train_snn(
        student_snn=snn_loaded_model,
        teacher_ann=model,
        train_loader=train_data_loader,
        test_loader=test_data_loader,
        device=device,
        time_steps=T,
        epochs=fine_tune_epochs,
        w_l_dict=w_l_dict,
        alpha=alpha,
        beta=beta,
        temperature=temperature,
        lr=learning_rate,
        model_save_path=snnmodel_save_path,
        plot_save_prefix=plot_save_prefix
    )

    # 7. 评估微调后的SNN
    print("\n微调完成。评估最终的SNN模型...")
    evaluate_snn(finetuned_snn_model, test_data_loader, device, T)

    # 8. (可选) 计算微调后的CKA
    print("\n计算微调后模型的最终CKA值...")
    cka_final = CKA(finetuned_snn_model, model, device, batch_size, repeat=5, T=T)
    cka_final.hook_layer(is_key_layer_fn=is_vgg_key_layer)
    hsic_final, _, _ = cka_final.inference(loader=test_data_loader)
    hsic_final = hsic_final.cpu().numpy()
    cka_final_diag = np.diag(hsic_final)
    print("微调后模型的对角线CKA值:")
    print(cka_final_diag)
    print("\n实验流程结束。")

if __name__ == '__main__':
    main()
