# Method-to-Code Map

Maps each component of the paper's method to the corresponding source file.

## 1. Main Entry — ResNet-18 / CIFAR-10

**File**: `resnet18_cifar10.py`

Responsibilities:
- Load CIFAR-10 data loaders with augmentation (`get_data_loaders`)
- Load pre-trained ResNet-18 teacher ANN (`model_cifar10_resnet.ResNet18()`)
- Load pre-converted student SNN from disk (or optionally convert fresh via SpikingJelly)
- Evaluate ANN accuracy (`evaluate_ann`)
- Evaluate initial SNN accuracy over T timesteps (`evaluate_snn`)
- Compute initial CKA matrix between ANN and SNN key layers
- Compute adaptive layer weights: `w_l = (1 - CKA_l) / Σ_j(1 - CKA_j)` (Formula 9)
- Call `train_snn()` for closed-loop fine-tuning
- Compute and save final CKA matrix
- Save CKA matrices to `cka_results/`

## 2. VGG Entry — VGG-16 / CIFAR-10

**File**: `vgg16_cifar10.py`

Retained secondary entry. Same pipeline structure as ResNet-18, using VGG-16 with BN or without BN. Not the first cleanup target.

## 3. CKA Computation

**File**: `cka_compare.py`

Class `CKA`:
- `similarity(X, Y, kernel)` — static method, computes CKA between two feature matrices
- `hook_layer(is_key_layer_fn)` — registers forward hooks on ANN and SNN key layers to capture Gram matrices
- `inference(loader)` — evaluates CKA over multiple repeats, returns HSIC matrix + layer names
- `_center_gram(gram)` — centers a Gram matrix
- `get_hsic(K, L, n, ones)` — computes unbiased HSIC estimator

## 4. Loss Functions

**File**: `loss_functions.py`

Three loss classes implementing the combined objective:

### GlobalLoss (KL Distillation)
```
L_global = T² · KL(softmax(snn_logits / T) || softmax(ann_logits / T))
```

### LocalLoss (CKA Feature Alignment)
```
L_local = Σ_l w_l · (1 - CKA(snn_features_l, ann_features_l))
```

### CombinedLoss
```
L_total = (1 - α) · L_task + α · (β · L_global + (1 - β) · L_local)
```
- `L_task`: Cross-entropy loss
- `α`: balances task loss vs distillation loss
- `β`: balances global (logit-level) vs local (feature-level) distillation

## 5. Training

**File**: `train_snn.py`

`train_snn()` function:
- Freezes teacher ANN parameters
- Uses `FeatureExtractor` to capture intermediate layer features via hooks
- Runs temporal SNN forward pass over `time_steps`
- Averages SNN logits and features across time steps
- Computes `CombinedLoss` and backpropagates
- Evaluates SNN on test set each epoch (`evaluate_snn`)
- Saves best model checkpoint
- Generates training curve plots and history `.npy` files

## 6. SNN Model Reconstruction

**File**: `models.py`

- `SNNBasicBlock` — SNN version of ResNet BasicBlock (Conv + IF neuron, no BatchNorm)
- `RebuiltSNNResNet` — reconstructed SNN ResNet with structured layer hierarchy
- `rebuild_snn_resnet18(model, snn_model)` — parses flat SpikingJelly SNN into `RebuiltSNNResNet`
- `rebuild_snn_vgg(ann_model, converted_snn_model)` — reconstructs VGG SNN structure
- `SNNVGGReconstructed` — container for rebuilt VGG SNN

## 7. ANN Model Definitions

- `model_cifar10_resnet.py` — ResNet-18/34/50/101/152 for CIFAR-10 (10 classes, 32×32 input)
- `model_cifar10_vgg.py` — VGG-16 (with and without BN) for CIFAR-10

## 8. Evaluation

**File**: `evaluate.py`

- `evaluate_snn(model, test_loader, device, time_steps)` — cumulative accuracy over T timesteps, optional plot saving
- `evaluate_ann(model, test_loader, device)` — standard ANN accuracy

## 9. Dependencies

**File**: `requirements.txt`

Frozen environment. Key packages: `torch==2.7.0`, `torchvision==0.22.0`, `spikingjelly==0.0.0.0.14`.
