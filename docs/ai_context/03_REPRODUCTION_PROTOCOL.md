# Reproduction Protocol

## 1. Environment

- **Python**: 3.8+
- **PyTorch**: 2.7.0
- **torchvision**: 0.22.0
- **SpikingJelly**: 0.0.0.0.14
- **CUDA**: 12.x (recommended)

The file `requirements.txt` is a **frozen environment snapshot**. A lighter `requirements-minimal.txt` may be provided in a future update.

```
pip install -r requirements.txt
```

## 2. Dataset

- **CIFAR-10**: downloaded automatically by `torchvision.datasets.CIFAR10`.

Current scripts use an author-local `dataset_dir` parameter:
- `resnet18_cifar10.py` line 57: `dataset_dir = '/home/lbz/git-hub/datasets'`
- `vgg16_cifar10.py` line 53: `dataset_dir = '/home/lbz/git-hub/datasets'`

These paths are intentionally preserved in the initial cleanup stage. Users must adjust them manually or wait for a future `argparse` update.

## 3. Checkpoints

**This repository does NOT provide model checkpoints.**

Users need to prepare the following checkpoint files themselves:

For `resnet18_cifar10.py`:
- `weights_path`: trained ResNet-18 ANN (`.pth`)
- `snn_save_path`: pre-converted ResNet-18 SNN (`.pth`)

For `vgg16_cifar10.py`:
- `weights_path`: trained VGG-16 ANN (`.pth`)
- `snn_save_path`: pre-converted VGG-16 SNN (`.pth`)

Current scripts refer to author-local absolute paths (e.g., `/home/lbz/git-hub/pretrained_models/...`). These are intentionally preserved for now.

## 4. Main ResNet-18 / CIFAR-10 Path

The primary runnable entry:

```bash
python resnet18_cifar10.py
```

> **TODO**: verify this command in the target server environment after preparing checkpoints.

The script performs the following stages:
1. Load CIFAR-10 data loaders
2. Load pre-trained teacher ANN
3. Evaluate ANN accuracy
4. Load pre-converted student SNN
5. Evaluate initial SNN accuracy over T timesteps
6. Compute initial CKA matrix between ANN and SNN key layers
7. Compute adaptive layer weights `w_l` (Formula 9)
8. Fine-tune SNN with `CombinedLoss` (Formula 5)
9. Evaluate final SNN accuracy
10. Compute and save final CKA matrix

## 5. VGG-16 / CIFAR-10 Path

```bash
python vgg16_cifar10.py
```

Secondary path; retained but not the first cleanup target.

## 6. Fresh ANN-to-SNN Conversion

Both scripts contain commented-out code blocks for fresh conversion using SpikingJelly:

```python
# converter = ann2snn.Converter(mode='99.9%', dataloader=train_data_loader)
# snn_model = converter(model)
# rebuilt_snn = rebuild_snn_resnet18(model, snn_model)
# torch.save(rebuilt_snn, snn_save_path)
```

Uncomment and adjust paths to perform a fresh conversion.

## 7. Time Steps

The paper evaluates SNN accuracy across multiple time steps (commonly T ∈ {2, 4, 8, 16, 32, 64}). The current script uses `T = 20` as default in `resnet18_cifar10.py`. Adjust the `T` variable in `main()` to test other values.

## 8. Expected Outputs

Running the main script generates:
- `cka_results/` — CKA matrices as `.npy` files
- `*_training_curve.png` — accuracy/loss plots
- `*_history.npy` — training history
- `*_best_snn_model.pth` — best checkpoint during fine-tuning

## 9. What Is NOT Included

- Paper PDF (not uploaded)
- Pre-trained model checkpoints
- Full ImageNet reproduction package
- Automatic download of author's private weights
- Verified reproduction logs
