# Current Repository Audit

Audit date: 2026-05-22

## 1. Existing Python Files

| File | Lines | Role |
|------|-------|------|
| `resnet18_cifar10.py` | 198 | Main entry: ResNet-18 / CIFAR-10 pipeline |
| `vgg16_cifar10.py` | 212 | VGG-16 / CIFAR-10 pipeline (retained) |
| `cka_compare.py` | 173 | CKA class: similarity, hooks, HSIC inference |
| `train_snn.py` | 289 | `train_snn()`: fine-tuning loop + `FeatureExtractor` |
| `loss_functions.py` | 115 | `GlobalLoss`, `LocalLoss`, `CombinedLoss` |
| `models.py` | 180 | `SNNBasicBlock`, `RebuiltSNNResNet`, `rebuild_snn_resnet18()` |
| `model_cifar10_resnet.py` | 139 | ResNet ANN models for CIFAR-10 |
| `model_cifar10_vgg.py` | 91 | VGG-16 ANN models for CIFAR-10 |
| `evaluate.py` | 127 | `evaluate_snn()`, `evaluate_ann()` |
| `requirements.txt` | 55 | Frozen pip dependencies |

## 2. Main Runnable Entry

- **Primary**: `resnet18_cifar10.py`
- **Secondary**: `vgg16_cifar10.py`

Both are standalone scripts. No `argparse` — parameters are hard-coded in `main()`.

## 3. Hard-Coded Paths

### `resnet18_cifar10.py`

| Line | Variable | Value |
|------|----------|-------|
| 20 | `data_dir` (default) | `/home/lbz/git-hub/datasets` |
| 57 | `dataset_dir` | `/home/lbz/git-hub/datasets` |
| 60 | `weights_path` | `/home/lbz/git-hub/pretrained_models/ann_resnet18_cifar10_best.pth` |
| 61 | `snn_save_path` | `/home/lbz/git-hub/pretrained_models/MY-cifar10-resnet18_SNN.pth` |

### `vgg16_cifar10.py`

| Line | Variable | Value |
|------|----------|-------|
| 20 | `data_dir` (default) | `/home/lbz/git-hub/datasets` |
| 53 | `dataset_dir` | `/home/lbz/git-hub/datasets` |
| 65 | `weights_path` | `/home/lbz/git-hub/pretrained_models/ann_vgg16_withBN_cifar10_best.pth` |
| 66 | `snn_save_path` | `/home/lbz/git-hub/spikingjelly_CKAvgg/SNN_models/SJ-cifar10-vgg16_withBNSNN.pth` |

**Policy**: intentionally preserved in the initial cleanup. Users must adjust manually.

## 4. Generated Outputs

Running the scripts may produce:

- `cka_results/` — CKA matrices (`.npy`)
- `cka_results_vgg/` — VGG CKA matrices (`.npy`)
- `*_training_curve.png` — training plots
- `*_history.npy` — training history arrays
- `*_performance_stats.npy` — performance statistics
- `*_best_snn_model.pth` — best fine-tuned checkpoint
- `*_accuracies.npy` — accuracy-per-timestep arrays
- `*_accuracy_curve.png` — accuracy curve plots

All should be excluded by `.gitignore`.

## 5. Checkpoint Assumptions

- Scripts assume pre-existing ANN and SNN checkpoint files on disk.
- No checkpoints are provided in the repository.
- No automatic download mechanism exists.

## 6. Dependencies

- `requirements.txt` is a full frozen environment (55 packages).
- Key packages: `torch==2.7.0`, `torchvision==0.22.0`, `spikingjelly==0.0.0.0.14`.
- No `setup.py`, `pyproject.toml`, or `conda` environment file exists.

## 7. Risk Items

1. **Hard-coded paths**: all critical paths are absolute and author-local.
2. **No argparse**: all hyperparameters are in-code constants.
3. **No minimal demo command**: users must prepare checkpoints before running anything.
4. **Heavy requirements**: full frozen environment may be difficult to replicate.
5. **No verified public checkpoint**: reproduction depends on user-provided weights.
6. **Flat structure**: no `src/` package; may become unwieldy as code grows.
7. **VGG `models.py` rebuild**: `rebuild_snn_vgg` is imported by `vgg16_cifar10.py` but the function's presence in `models.py` should be verified by the author.
8. **Missing `.gitignore`**: currently no ignore rules; generated outputs could be accidentally committed.

## 8. Recommended Next Cleanup Tasks (Not for This Round)

These are suggestions only. Do not implement in the current cleanup round:

1. ~~Add `.gitignore`~~ → Done in current round.
2. ~~Add `LICENSE`~~ → Done in current round.
3. ~~Rewrite `README.md`~~ → Done in current round.
4. ~~Add `docs/ai_context/`~~ → Done in current round.
5. ~~Add `AGENTS.md` and `CLAUDE.md`~~ → Done in current round.
6. Add `argparse` to `resnet18_cifar10.py` (future).
7. Add `scripts/run_resnet18_cifar10.sh` (future).
8. Split into `src/` package (future, after main path verified).
9. Add `requirements-minimal.txt` (future).
10. Verify VGG-16 path end-to-end (future).
11. Add CI (GitHub Actions) for import checks (future).
