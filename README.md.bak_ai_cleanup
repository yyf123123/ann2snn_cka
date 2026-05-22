# ANN2SNN with CKA Analysis

A comprehensive framework for converting Artificial Neural Networks (ANNs) to Spiking Neural Networks (SNNs) and analyzing their representational similarity using Centered Kernel Alignment (CKA).

## Overview

This project implements an ANN-to-SNN conversion pipeline with built-in performance analysis tools. It leverages the [SpikingJelly](https://github.com/fangwei123456/spikingjelly) library for efficient SNN conversion and uses CKA metrics to quantify the similarity between ANN and SNN representations across network layers.

### Key Features

- **ANN-to-SNN Conversion**: Seamless conversion of pre-trained ANNs to SNNs using SpikingJelly's ann2snn converter
- **CKA-based Analysis**: Compute Centered Kernel Alignment to measure representational similarity between ANN and SNN layers
- **Multiple Architectures**: Support for ResNet-18 and VGG-16 models
- **CIFAR-10 Dataset**: Pre-configured data loading with augmentation
- **Memory-Efficient Training**: Optimized CKA computation for SNN training without memory leaks
- **Comprehensive Evaluation**: Tools for evaluating both ANN and SNN model performance

## Project Structure

```
ann2snn_cka/
├── cka_compare.py                 # CKA similarity computation and analysis
├── models.py                       # SNN model reconstruction utilities
├── evaluate.py                     # Model evaluation functions
├── loss_functions.py               # Custom loss functions
├── train_snn.py                    # SNN training pipeline
├── resnet18_cifar10.py            # ResNet-18 conversion and analysis
├── vgg16_cifar10.py               # VGG-16 conversion and analysis
├── model_cifar10_resnet.py        # ResNet-18 ANN model definition
├── model_cifar10_vgg.py           # VGG-16 ANN model definition
└── requirements.txt                # Python dependencies
```

## Core Components

### 1. CKA Analysis (`cka_compare.py`)

Implements the CKA metric for comparing feature representations:

- **`CKA` class**: Main interface for similarity computation
  - `similarity()`: Compute CKA for inference evaluation
  - `cka_train()`: Memory-efficient variant for training with gradient flow
  - `inference()`: Evaluate CKA across multiple layers with temporal averaging
  - `hook_layer()`: Register hooks to capture intermediate layer outputs

**Key Parameters:**
- `snn_model`, `ann_model`: Models to compare
- `batch_size`: Batch size for computations (default: 1024)
- `repeat`: Number of evaluation repeats (default: 5)
- `T`: Number of SNN timesteps (default: 50)

### 2. Model Reconstruction (`models.py`)

Rebuilds converted SNN models into structured architectures:

- **`rebuild_snn_resnet18()`**: Convert flattened ResNet-18 SNN to hierarchical structure
- **`rebuild_snn_vgg()`**: Reconstruct VGG-16 SNN with sequential features and classifier
- **`SNNBasicBlock`**: Spiking version of ResNet BasicBlock
- **`SNNVGGReconstructed`**: Spiking VGG model container

### 3. Training and Evaluation

- **`evaluate.py`**: Functions to evaluate model accuracy on test sets
- **`train_snn.py`**: SNN training loop with CKA-based objectives
- **`loss_functions.py`**: Custom loss functions for SNN training

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.7.0
- torchvision 0.22.0
- SpikingJelly 0.0.0.0.14
- CUDA 12.x (recommended for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd ann2snn_cka

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic ANN-to-SNN Conversion and Analysis

```python
import torch
from resnet18_cifar10 import main as run_resnet18
from vgg16_cifar10 import main as run_vgg16

# Convert ResNet-18 and analyze
run_resnet18()

# Convert VGG-16 and analyze
run_vgg16()
```

### Manual CKA Computation

```python
from cka_compare import CKA
from models import rebuild_snn_resnet18

# Initialize CKA analyzer
cka = CKA(snn_model=snn_model, ann_model=ann_model, device='cuda')

# Register hooks for layer analysis
cka.hook_layer(lambda m: isinstance(m, torch.nn.Conv2d))

# Compute similarity matrix
similarity_matrix, snn_names, ann_names = cka.inference(test_loader)

# Results: similarity_matrix[i, j] shows CKA between SNN layer i and ANN layer j
```

### Training with CKA Loss

```python
from cka_compare import CKA

# Compute CKA for training (memory-efficient)
cka_loss = CKA.cka_train(snn_features, ann_features, kernel='linear')

# Use in your training loop
loss = cka_loss
loss.backward()
optimizer.step()
```

## Technical Details

### CKA Metric

Centered Kernel Alignment measures the similarity between two sets of representations:

$$\text{CKA}(X, Y) = \frac{\text{HSIC}(\mathbf{K}, \mathbf{L})}{\sqrt{\text{HSIC}(\mathbf{K}, \mathbf{K}) \cdot \text{HSIC}(\mathbf{L}, \mathbf{L})}}$$

Where K and L are centered Gram matrices of X and Y respectively.

### SNN Conversion Process

1. Train ANN model on CIFAR-10
2. Convert ANN to SNN using SpikingJelly's ann2snn converter
3. Rebuild flattened SNN structure to match original architecture
4. Evaluate temporal dynamics over T timesteps
5. Analyze layer-wise similarity using CKA

### Memory Optimization

The `cka_train()` method optimizes memory usage by:
- Computing Gram matrices in `no_grad` context
- Reintroducing gradients only through HSIC values
- Preventing intermediate tensors from holding computation graphs

## Experimental Results

The framework enables:
- **Accuracy Comparison**: ANN vs SNN classification performance on CIFAR-10
- **Representational Analysis**: Layer-by-layer CKA similarity scores
- **Temporal Dynamics**: How SNN representations evolve across timesteps
- **Training Optimization**: Using CKA as a regularization objective

## Citation

If you use this code in your research, please cite:

```bibtex
@project{ann2snn_cka,
  title={ANN2SNN with CKA Analysis},
  year={2024}
}
```

Also cite SpikingJelly:
```bibtex
@article{fang2020spikingjelly,
  title={SpikingJelly: An open-source learning framework for spiking neural networks},
  author={Fang, Wei and Chen, Yanqi and others},
  journal={arXiv preprint arXiv:2212.10805},
  year={2022}
}
```

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## References

- [SpikingJelly](https://github.com/fangwei123456/spikingjelly) - SNN Library
- [CKA: An unbiased measure of feature importance](https://arxiv.org/abs/1905.05172)
- [ANN-to-SNN Conversion Survey](https://arxiv.org/abs/2303.13778)

## Contact

For questions or suggestions, please open an issue on GitHub.
