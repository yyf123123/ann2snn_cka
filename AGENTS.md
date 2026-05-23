# Repository Instructions for AI Coding Agents

## Project Scope

This repository implements the code for:

> **High-Fidelity ANN-to-SNN Conversion via Closed-Loop CKA Distillation**

The primary runnable entry is `resnet18_cifar10.py`. The secondary entry `vgg16_cifar10.py` is retained.

## Before Editing

Read these files first:
- `README.md`
- `docs/PAPER_CONTEXT.md`

## Editing Rules

- Do **not** delete existing source files or move them.
- Do **not** remove VGG-16 or ImageNet-related code or notes.
- Do **not** upload model weights (`.pth`, `.pt`, `.ckpt`), datasets, or the paper PDF.
- Preserve existing author-local absolute paths during initial cleanup.
- Keep **paper-reported results** strictly separate from **repository-reproduced results**.
- Do **not** claim reproduction unless verified logs or results are present.
- Do **not** run `git commit` unless explicitly requested.

## Method Summary

1. Load teacher ANN (frozen).
2. Load or convert student SNN (trainable).
3. Compute initial CKA → adaptive layer weights `w_l = (1 - CKA_l) / Σ(1 - CKA_j)`.
4. Fine-tune SNN: `L_total = (1-α)·L_task + α·(β·L_global + (1-β)·L_local)`.
5. Evaluate final SNN accuracy and final CKA.

## Safety

If tokens, passwords, or API keys are found, do **not** print them. Record only the file name and approximate location.
