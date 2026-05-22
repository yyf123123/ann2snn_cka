# ANN2SNN-CKA: Closed-Loop CKA Distillation for ANN-to-SNN Conversion

A research codebase for **high-fidelity ANN-to-SNN conversion** using **Closed-Loop CKA Distillation**.

This repository implements the method described in:

> **High-Fidelity ANN-to-SNN Conversion via Closed-Loop CKA Distillation**

The core idea: after converting a pre-trained ANN to an SNN via SpikingJelly, we improve conversion fidelity by fine-tuning the SNN with a combined loss that includes a **CKA-based feature alignment term** with adaptively weighted layers.

---

## Method Summary

### Stage 1: ANN-to-SNN Conversion

- Pre-trained ANN serves as the **teacher** (frozen).
- SpikingJelly `ann2snn.Converter` performs: BN fusion, ReLU→IF neuron replacement, threshold normalization (99.9% activation-based calibration).

### Stage 2: Closed-Loop CKA Distillation

- **Teacher ANN** is frozen.
- **Student SNN** is fine-tuned with a combined loss:

```
L_total = (1 - α) · L_task + α · (β · L_global + (1 - β) · L_local)
```

| Term | Description |
|------|-------------|
| `L_task` | Cross-entropy loss on ground-truth labels |
| `L_global` | KL divergence between SNN and ANN output logits |
| `L_local` | Weighted CKA feature alignment loss across layers |

- Layer weights are adaptive, computed from initial CKA similarity:

```
w_l = (1 - CKA_l) / Σ_j(1 - CKA_j)
```

---

## Current Release Scope

| Included | Status |
|----------|--------|
| ResNet-18 / CIFAR-10 pipeline | **Primary entry** |
| VGG-16 / CIFAR-10 pipeline | Retained |
| CKA computation module | Included |
| Combined loss functions | Included |
| SNN fine-tuning loop | Included |
| Model checkpoints | **Not provided** |
| Paper PDF | **Not included** |
| ImageNet pipeline | Not included in this release |

---

## Repository Structure

```
ann2snn_cka/
├── README.md
├── LICENSE                        # MIT
├── .gitignore
├── requirements.txt
├── resnet18_cifar10.py            # MAIN ENTRY: ResNet-18 / CIFAR-10
├── vgg16_cifar10.py               # VGG-16 / CIFAR-10 (retained)
├── cka_compare.py                 # CKA computation and layer analysis
├── train_snn.py                   # Closed-loop SNN fine-tuning
├── loss_functions.py              # Task, global KD, local CKA, combined loss
├── models.py                      # SNN model reconstruction
├── model_cifar10_resnet.py        # ResNet ANN model definition
├── model_cifar10_vgg.py           # VGG ANN model definition
├── evaluate.py                    # ANN/SNN evaluation utilities
└── docs/
    └── ai_context/                # AI agent context files
```

---

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies:
- Python 3.8+
- PyTorch 2.7.0
- torchvision 0.22.0
- SpikingJelly 0.0.0.0.14
- CUDA 12.x (recommended)

> **Note**: `requirements.txt` is a full frozen environment. A minimal version may be provided later.

---

## Usage

### ResNet-18 / CIFAR-10 (Primary)

```bash
python resnet18_cifar10.py
```

This script performs the full closed-loop CKA distillation pipeline:

1. Load CIFAR-10 data
2. Load pre-trained ResNet-18 teacher ANN
3. Evaluate ANN accuracy
4. Load pre-converted student SNN
5. Evaluate initial SNN accuracy
6. Compute initial CKA matrix
7. Compute adaptive layer weights `w_l`
8. Fine-tune SNN with combined loss
9. Evaluate final SNN accuracy and CKA

### VGG-16 / CIFAR-10

```bash
python vgg16_cifar10.py
```

Secondary pipeline; retained but not the primary focus.

### Fresh ANN-to-SNN Conversion

Both scripts contain commented-out blocks for fresh conversion via SpikingJelly:

```python
# converter = ann2snn.Converter(mode='99.9%', dataloader=train_data_loader)
# snn_model = converter(model)
# rebuilt_snn = rebuild_snn_resnet18(model, snn_model)
# torch.save(rebuilt_snn, snn_save_path)
```

---

## Checkpoints and Data

**This repository does NOT provide model checkpoints.**

Users must prepare the following files:

| File | Purpose |
|------|---------|
| Pre-trained ANN weights (`.pth`) | Teacher model |
| Pre-converted SNN weights (`.pth`) | Student model before fine-tuning |

Current scripts contain author-local absolute paths (e.g., `/home/lbz/git-hub/...`). These are intentionally preserved in the initial release. Users should adjust paths manually or wait for a future `argparse` update.

CIFAR-10 is downloaded automatically by `torchvision`.

---

## Results

### Paper-Reported Results

The paper reports that closed-loop CKA distillation substantially improves ANN-to-SNN conversion fidelity, achieving near-lossless accuracy at moderate-to-high time steps (e.g., T=32) and improved CKA similarity across layers.

### Repository-Reproduced Results

**Status**: To be reproduced.

This repository provides code infrastructure. Verified reproduction logs are not yet included. Users should run the pipeline in their own environment after preparing the required checkpoints and verify results independently.

---

## Documentation for AI Agents

The `docs/ai_context/` directory contains structured documentation for AI coding agents (Claude Code, Codex, etc.):

| File | Content |
|------|---------|
| `00_PROJECT_OVERVIEW.md` | Short summary for agents |
| `01_METHOD_TO_CODE_MAP.md` | Paper method → source file mapping |
| `02_REPOSITORY_STRUCTURE_CURRENT.md` | File layout and gitignore policy |
| `03_REPRODUCTION_PROTOCOL.md` | How to run and reproduce |
| `04_RESULTS_AND_CLAIMS.md` | Paper vs. repository results policy |
| `05_MANUAL_CHECKLIST.md` | What is confirmed vs. pending |
| `06_CURRENT_REPO_AUDIT.md` | Known issues and risk items |

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Citation

If you use this code in your research, please cite the associated paper (metadata to be finalized):

```bibtex
@article{ann2snn_cka,
  title   = {High-Fidelity ANN-to-SNN Conversion via Closed-Loop CKA Distillation},
  author  = {},
  journal = {},
  year    = {}
}
```

Also cite SpikingJelly:

```bibtex
@article{fang2020spikingjelly,
  title   = {SpikingJelly: An open-source learning framework for spiking neural networks},
  author  = {Fang, Wei and Chen, Yanqi and others},
  journal = {arXiv preprint arXiv:2212.10805},
  year    = {2022}
}
```

---

## Acknowledgments

- [SpikingJelly](https://github.com/fangwei123456/spikingjelly) for the ANN-to-SNN conversion framework.
- CKA methodology based on [Kornblith et al. (ICML 2019)](https://arxiv.org/abs/1905.05172).
