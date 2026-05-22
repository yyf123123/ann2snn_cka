# Results and Claims

## 1. Paper-Reported Results

The following claims originate from the paper and are **not** automatically reproduced by running this repository:

- The closed-loop CKA distillation framework substantially improves low-timestep ANN-to-SNN conversion fidelity compared to standard post-conversion calibration.
- For ResNet-18 on CIFAR-10, the method achieves near-lossless conversion at moderate-to-high time steps (e.g., T=32 approaches the source ANN accuracy).
- The method improves layer-wise CKA similarity between ANN and SNN representations.
- The adaptive weighting scheme `w_l = (1 - CKA_l) / Σ(1 - CKA_j)` outperforms uniform weighting.
- The combined loss `L_total = (1-α)·L_task + α·(β·L_global + (1-β)·L_local)` provides better accuracy than task loss alone.

## 2. Repository-Reproduced Results

**Status**: To be reproduced.

This repository provides the code infrastructure but does **not** include verified reproduction logs, result tables, or pre-computed metrics. Users should:

1. Prepare their own ANN and SNN checkpoints.
2. Run `resnet18_cifar10.py` in their own environment.
3. Verify results independently.

If verified results become available (e.g., from the author's server logs), they should be added in a separate `results/` directory with clear provenance notes.

## 3. README Claim Policy

The `README.md` MUST use the following language:

| Correct | Incorrect |
|---------|-----------|
| "Paper-reported" | "This repository achieves" |
| "Expected workflow" | "Guaranteed result" |
| "To be reproduced" | "Fully reproduced" |
| "Checkpoint not provided" | "Ready to run out of the box" |

The README should:

- Describe the method and its expected behavior.
- State clearly that checkpoints are not included.
- Note that hard-coded local paths exist and may need adjustment.

The README should **not**:

- Claim the repository reproduces specific paper accuracy numbers.
- Promise that the code alone produces paper-reported results.
- Suggest that pre-trained weights are available for download.
