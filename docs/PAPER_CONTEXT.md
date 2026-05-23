# Paper-to-Code Mapping

This repository implements **High-Fidelity ANN-to-SNN Conversion via Closed-Loop CKA Distillation** in two stages:

## Stage 1: ANN-to-SNN Conversion

- A pre-trained ANN is converted to an SNN using SpikingJelly's `ann2snn.Converter`.
- Conversion includes BN folding, ReLU-to-IF neuron replacement, and 99.9% activation-based threshold normalization.
- In `resnet18_cifar10.py`, fresh conversion code is commented out; the default path loads a pre-converted SNN checkpoint.

## Stage 2: Closed-Loop CKA Distillation

- **Teacher ANN** is frozen. **Student SNN** is trainable.
- Initial CKA is computed between ANN and SNN intermediate layers.
- Adaptive layer weights are derived from CKA: `w_l = (1 - CKA_l) / Σ(1 - CKA_j)`.
- The SNN is fine-tuned with a combined loss:

```
L_total = (1-α) · L_task + α · (β · L_global + (1-β) · L_local)
```

## File-to-Method Mapping

| File | Responsibility |
|------|---------------|
| `resnet18_cifar10.py` | Primary workflow: load data, load models, compute CKA, call `train_snn()`, evaluate |
| `vgg16_cifar10.py` | VGG-16 workflow (retained) |
| `cka_compare.py` | `CKA` class: `similarity()`, `hook_layer()`, `inference()`, HSIC computation |
| `train_snn.py` | `train_snn()`: fine-tuning loop with `FeatureExtractor` hooks, temporal averaging, optimizer steps |
| `loss_functions.py` | `GlobalLoss` (KL distillation), `LocalLoss` (weighted CKA loss), `CombinedLoss` (L_total) |
| `models.py` | `SNNBasicBlock`, `RebuiltSNNResNet`, `rebuild_snn_resnet18()`, `rebuild_snn_vgg()` |
| `model_cifar10_resnet.py` | ResNet-18/34/50/101/152 ANN definitions for CIFAR-10 |
| `model_cifar10_vgg.py` | VGG-16 (with/without BN) ANN definitions for CIFAR-10 |
| `evaluate.py` | `evaluate_snn()` (temporal accuracy), `evaluate_ann()` (standard accuracy) |
