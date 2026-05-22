# Project Overview

## What This Repository Is

This repository implements an **ANN-to-SNN conversion and closed-loop CKA distillation** workflow associated with the paper:

> **High-Fidelity ANN-to-SNN Conversion via Closed-Loop CKA Distillation**

The method converts a pre-trained Artificial Neural Network (ANN) into a Spiking Neural Network (SNN), then improves conversion fidelity through a closed-loop fine-tuning stage guided by **Centered Kernel Alignment (CKA)** between ANN and SNN intermediate representations.

## Current Scope

- **Primary entry**: `resnet18_cifar10.py` — ResNet-18 on CIFAR-10, the main focus for readability and reproduction.
- **Secondary entry**: `vgg16_cifar10.py` — VGG-16 on CIFAR-10, retained but not the first priority.
- **Not included**: paper PDF, model checkpoints, full ImageNet pipeline, generated result files (`.npy`, `.png`).

## Core Files

| File | Role |
|------|------|
| `resnet18_cifar10.py` | Main workflow: load teacher ANN, load/convert SNN, compute CKA, fine-tune, evaluate |
| `vgg16_cifar10.py` | VGG-16 workflow, retained |
| `cka_compare.py` | CKA computation: similarity, hook-based layer analysis, inference |
| `train_snn.py` | Closed-loop SNN fine-tuning with combined loss |
| `loss_functions.py` | Task loss (CE), global KD loss (KL), local CKA loss, combined loss |
| `models.py` | SNN model reconstruction (ResNet and VGG) |
| `model_cifar10_resnet.py` | CIFAR-10 ResNet ANN model definition |
| `model_cifar10_vgg.py` | CIFAR-10 VGG ANN model definition |
| `evaluate.py` | ANN/SNN evaluation utilities |

## Key Method

1. **Stage 1 — ANN-to-SNN Conversion**: BN fusion, ReLU→IF neuron replacement, threshold normalization via SpikingJelly `ann2snn.Converter`.
2. **Stage 2 — Closed-Loop CKA Distillation**: Freeze teacher ANN, compute initial CKA, derive adaptive layer weights `w_l`, fine-tune SNN with `L_total = (1-α)·L_task + α·(β·L_global + (1-β)·L_local)`.

## README Policy

- The README distinguishes **paper-reported results** from **repository-reproduced results**.
- The README does not claim that all paper numbers are reproduced by this repository.
- Checkpoints are **not provided**.

## For AI Agents

Read the following files before making changes:
- `AGENTS.md` (editing rules)
- `docs/ai_context/01_METHOD_TO_CODE_MAP.md`
- `docs/ai_context/06_CURRENT_REPO_AUDIT.md`
