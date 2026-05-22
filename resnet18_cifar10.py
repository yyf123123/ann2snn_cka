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
from models import SNNBasicBlock, RebuiltSNNResNet, rebuild_snn_resnet18
from evaluate import evaluate_ann, evaluate_snn
from train_snn import train_snn

import torchvision
import torch


def get_data_loaders(batch_size: int, data_dir: str = '/home/lbz/git-hub/datasets'):
    """Prepare CIFAR-10 data loaders with standard augmentation."""
    print("Preparing data loaders...")

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
    # =========================================================================
    # Experiment Configuration
    # =========================================================================
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = 'cuda:0'
    dataset_dir = '/home/lbz/git-hub/datasets'

    # Paths
    weights_path = '/home/lbz/git-hub/pretrained_models/ann_resnet18_cifar10_best.pth'
    snn_save_path = '/home/lbz/git-hub/pretrained_models/MY-cifar10-resnet18_SNN.pth'

    # SNN and fine-tuning hyperparameters
    batch_size = 100
    T = 20
    fine_tune_epochs = 50
    learning_rate = 1e-5
    alpha = 0.5
    beta = 0.1
    temperature = 2.0

    # =========================================================================
    # Step 1: Load and evaluate the pre-trained ANN (Teacher)
    # =========================================================================
    train_data_loader, test_data_loader = get_data_loaders(batch_size, dataset_dir)

    model = model_cifar10_resnet.ResNet18()
    model.load_state_dict(torch.load(weights_path))
    model.to(device)
    model.eval()

    print('=== ANN Teacher Evaluation ===')
    acc_ann = evaluate_ann(model, test_data_loader, device)
    print(f'ANN accuracy: {acc_ann:.2f}%')

    # =========================================================================
    # Step 2: Load the pre-converted SNN (Student)
    #     The conversion (ANN->SNN via SpikingJelly) and model rebuilding
    #     are performed offline. Uncomment the block below for fresh conversion:
    #
    # converter = ann2snn.Converter(mode='99.9%', dataloader=train_data_loader)
    # snn_model = converter(model)
    # rebuilt_snn = rebuild_snn_resnet18(model, snn_model)
    # rebuilt_snn.to(device)
    # torch.save(rebuilt_snn, snn_save_path)
    # =========================================================================

    snn_loaded_model = torch.load(snn_save_path, weights_only=False)
    snn_loaded_model.eval()

    print(f'\n=== Initial SNN Evaluation (T={T}) ===')
    evaluate_snn(snn_loaded_model, test_data_loader, device, time_steps=T)
    snn_loaded_model.to(device)

    # =========================================================================
    # Step 3: Compute initial CKA and determine layer weights w_l
    #     Following Formula (9) in the paper:
    #     w_l = (1 - CKA_l) / sum_j(1 - CKA_j)
    # =========================================================================
    cka = CKA(snn_loaded_model, model, device, batch_size, repeat=5, T=T)
    print('\nRegistering hooks on key layers (BasicBlock)...')

    def is_resnet_basic_block(m):
        return isinstance(m, (model_cifar10_resnet.BasicBlock, SNNBasicBlock))

    cka.hook_layer(is_key_layer_fn=is_resnet_basic_block)

    print('Computing initial CKA matrix...')
    hsic, model1_names, model2_names = cka.inference(loader=test_data_loader)
    hsic = hsic.cpu().numpy()

    # Extract diagonal CKA values (matched layer pairs)
    cka_initial = np.diag(hsic)

    np.set_printoptions(precision=3, suppress=True)
    print('\nInitial CKA similarity (diagonal):')
    print(cka_initial)

    # Compute w_l via Formula (9)
    one_minus_cka = 1 - cka_initial
    denominator = np.sum(one_minus_cka)
    if denominator == 0:
        num_layers = len(one_minus_cka)
        w_l = np.full(num_layers, 1.0 / num_layers)
    else:
        w_l = one_minus_cka / denominator

    w_l_dict = {name: weight for name, weight in zip(model1_names, w_l)}
    print('\nLayer weights w_l (Formula 9):')
    print(w_l_dict)

    # Save initial CKA results
    output_dir_cka = 'cka_results'
    os.makedirs(output_dir_cka, exist_ok=True)

    cka_results = {
        'cka_matrix': hsic,
        'snn_layer_names': model1_names,
        'ann_layer_names': model2_names
    }
    np.save(os.path.join(output_dir_cka, f'(T={T})initial_cka_resnet18.npy'), cka_results)

    # =========================================================================
    # Step 4: Closed-Loop CKA Distillation (Formula 5)
    #     L_total = (1-alpha)*L_task + alpha*(beta*L_global + (1-beta)*L_local)
    # =========================================================================
    print('\n' + '=' * 50)
    print('Closed-Loop CKA Distillation')
    print('=' * 50)
    print(f'  T = {T}, epochs = {fine_tune_epochs}')
    print(f'  alpha = {alpha}, beta = {beta}, temperature = {temperature}')
    print(f'  lr = {learning_rate}')
    print('=' * 50)

    # Freeze teacher
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    plot_save_prefix = f'(T={T}_alpha={alpha}_beta={beta})'

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
        model_save_path=f'(T={T}_alpha={alpha}_beta={beta})best_snn_model.pth',
        plot_save_prefix=plot_save_prefix
    )

    # =========================================================================
    # Step 5: Evaluate the fine-tuned SNN
    # =========================================================================
    print('\n=== Fine-tuned SNN Evaluation ===')
    evaluate_snn(finetuned_snn_model, test_data_loader, device, T)

    # Compute and save final CKA
    print('\nComputing final CKA...')
    cka_final = CKA(finetuned_snn_model, model, device, batch_size, repeat=5, T=T)
    cka_final.hook_layer(is_key_layer_fn=is_resnet_basic_block)
    hsic_final, _, _ = cka_final.inference(loader=test_data_loader)
    hsic_final = hsic_final.cpu().numpy()
    cka_final_diag = np.diag(hsic_final)

    print('\nFinal CKA similarity (diagonal):')
    print(cka_final_diag)

    cka_results_final = {
        'cka_matrix': hsic_final,
        'snn_layer_names': model1_names,
        'ann_layer_names': model2_names
    }
    np.save(os.path.join(output_dir_cka, f'(T={T}_alpha={alpha}_beta={beta})final_cka_resnet18.npy'),
            cka_results_final)

    print('\nExperiment complete.')


if __name__ == '__main__':
    main()
