# Repository Instructions for AI Coding Agents

## Project Scope

This repository implements an ANN-to-SNN conversion and closed-loop CKA distillation workflow associated with the paper:

> **High-Fidelity ANN-to-SNN Conversion via Closed-Loop CKA Distillation**

The current public repository is organized as a flat Python project. The primary runnable entry is:

- `resnet18_cifar10.py`

The secondary retained entry is:

- `vgg16_cifar10.py`

## AI Context Files

Before making non-trivial changes, read the files in `docs/ai_context/` in this order:

1. `docs/ai_context/00_PROJECT_OVERVIEW.md` — short summary, scope, and policy
2. `docs/ai_context/01_METHOD_TO_CODE_MAP.md` — paper method mapped to each source file
3. `docs/ai_context/02_REPOSITORY_STRUCTURE_CURRENT.md` — current file layout and responsibilities
4. `docs/ai_context/03_REPRODUCTION_PROTOCOL.md` — how to run and what is needed
5. `docs/ai_context/06_CURRENT_REPO_AUDIT.md` — known issues, hard-coded paths, risk items

## Current Source Files

| File | Role |
|------|------|
| `resnet18_cifar10.py` | Main entry: ResNet-18 / CIFAR-10 pipeline |
| `vgg16_cifar10.py` | Secondary entry: VGG-16 / CIFAR-10 (retained) |
| `cka_compare.py` | CKA computation and layer-hook analysis |
| `train_snn.py` | Closed-loop SNN fine-tuning with combined loss |
| `loss_functions.py` | Task loss, global KD loss, local CKA loss, combined loss |
| `models.py` | SNN model reconstruction (ResNet and VGG) |
| `model_cifar10_resnet.py` | ResNet ANN model for CIFAR-10 |
| `model_cifar10_vgg.py` | VGG ANN model for CIFAR-10 |
| `evaluate.py` | ANN/SNN evaluation utilities |
| `requirements.txt` | Frozen pip dependencies |

## Editing Rules

- Do **not** delete existing runnable scripts.
- Do **not** remove VGG-16-related code or ImageNet-related notes.
- Do **not** upload or add paper PDF files to the repository.
- Do **not** add checkpoints (`.pth`, `.pt`, `.ckpt`), datasets, logs, wandb outputs, or generated result files.
- Preserve existing author-local absolute paths during the initial cleanup stage.
- Prefer documentation and small verifiable changes before large refactoring.
- Keep paper-reported results strictly separate from repository-reproduced results.
- Do **not** claim reproduction unless logs/results are present and verified.
- Do **not** perform `git commit` unless explicitly requested by the user.

## Method Rules

The intended method pipeline is:

1. Load teacher ANN (frozen).
2. Load or convert student SNN (trainable).
3. Evaluate ANN and initial SNN.
4. Compute initial CKA similarity between ANN and SNN key layers.
5. Compute adaptive local loss weights:
   ```
   w_l = (1 - CKA_l) / Σ_j(1 - CKA_j)
   ```
6. Fine-tune student SNN using combined loss:
   ```
   L_total = (1 - α) · L_task + α · (β · L_global + (1 - β) · L_local)
   ```
   - `L_task`: cross-entropy on ground-truth labels
   - `L_global`: KL divergence between SNN and ANN logits
   - `L_local`: weighted CKA feature alignment loss
7. Evaluate final SNN accuracy and final CKA.

## Safety

If sensitive values (tokens, passwords, API keys) are found in any file, do **not** print them. Record only the file name and approximate location in `docs/ai_context/05_MANUAL_CHECKLIST.md`.
