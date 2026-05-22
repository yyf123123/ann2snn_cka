# Current Repository Structure

## Layout

The repository currently uses a **flat structure** (no `src/` package):

```
ann2snn_cka/
├── README.md
├── AGENTS.md
├── CLAUDE.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── cka_compare.py              # CKA computation and layer-hook analysis
├── evaluate.py                 # ANN/SNN evaluation functions
├── loss_functions.py           # Task, global KD, local CKA, combined loss
├── model_cifar10_resnet.py     # ResNet ANN model for CIFAR-10
├── model_cifar10_vgg.py        # VGG ANN model for CIFAR-10
├── models.py                   # SNN reconstruction (ResNet + VGG)
├── resnet18_cifar10.py         # MAIN ENTRY: ResNet-18 / CIFAR-10 pipeline
├── train_snn.py                # Closed-loop SNN fine-tuning
├── vgg16_cifar10.py            # VGG-16 / CIFAR-10 pipeline (retained)
└── docs/
    └── ai_context/
        ├── 00_PROJECT_OVERVIEW.md
        ├── 01_METHOD_TO_CODE_MAP.md
        ├── 02_REPOSITORY_STRUCTURE_CURRENT.md
        ├── 03_REPRODUCTION_PROTOCOL.md
        ├── 04_RESULTS_AND_CLAIMS.md
        ├── 05_MANUAL_CHECKLIST.md
        └── 06_CURRENT_REPO_AUDIT.md
```

## File Responsibilities

| File | Responsibility |
|------|---------------|
| `resnet18_cifar10.py` | Main entry: orchestrates the full ResNet-18 pipeline |
| `vgg16_cifar10.py` | Secondary VGG-16 entry, retained |
| `cka_compare.py` | CKA class: similarity, hook registration, HSIC inference |
| `train_snn.py` | `train_snn()`: fine-tuning loop with `CombinedLoss` |
| `loss_functions.py` | `GlobalLoss`, `LocalLoss`, `CombinedLoss` |
| `models.py` | `SNNBasicBlock`, `RebuiltSNNResNet`, `rebuild_snn_resnet18()`, `rebuild_snn_vgg()` |
| `model_cifar10_resnet.py` | `ResNet()`, `BasicBlock`, `ResNet18()` etc. |
| `model_cifar10_vgg.py` | `vgg16_bn_cifar10()`, `vgg16_cifar10()` |
| `evaluate.py` | `evaluate_snn()`, `evaluate_ann()` |
| `requirements.txt` | Frozen pip dependencies |

## Files That Should NOT Be Committed

The following should be excluded via `.gitignore`:

- `*.pth`, `*.pt`, `*.ckpt` — model checkpoints
- `checkpoints/`, `pretrained_models/` — model weight directories
- `data/`, `datasets/` — dataset directories
- `outputs/`, `logs/`, `wandb/`, `runs/` — experiment outputs
- `cka_results/`, `cka_results_vgg/` — generated CKA matrices
- `__pycache__/` — Python bytecode
- `*.npy`, `*.npz` — NumPy arrays
- `*.png`, `*.jpg`, `*.jpeg` — generated figures
- `*.pdf` — including the paper PDF
- `.vscode/`, `.idea/` — IDE config

Note: the paper PDF must **not** be added to the repository.
Model weights and generated result files are not included in this public release.
