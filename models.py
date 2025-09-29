import torch
import torch.nn as nn
from typing import Dict, List, Optional, Type, Union

# =============================================================================
#                            SNN Block Definitions
# =============================================================================

class SNNBasicBlock(nn.Module):
    """SNN version of BasicBlock (for ResNet-18/34), with BN layers removed."""
    expansion = 1
    def __init__(self, conv1, conv2, shortcut=None, 
                 snn_activation1=None, snn_activation2=None):
        super().__init__()
        self.conv1 = conv1
        self.conv2 = conv2
        self.shortcut = shortcut if shortcut is not None else nn.Sequential()
        self.snn_activation1 = snn_activation1
        self.snn_activation2 = snn_activation2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        if self.snn_activation1:
            out = self.snn_activation1(out)

        out = self.conv2(out)
        out += identity
        
        if self.snn_activation2:
            out = self.snn_activation2(out)
            
        return out

class SNNBottleneck(nn.Module):
    """SNN version of Bottleneck block (for ResNet-50), with BN layers removed."""
    expansion = 4
    def __init__(self, conv1, conv2, conv3, shortcut=None,
                 snn_activation1=None, snn_activation2=None, snn_activation3=None):
        super().__init__()
        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3
        self.shortcut = shortcut if shortcut is not None else nn.Sequential()
        self.snn_activation1 = snn_activation1
        self.snn_activation2 = snn_activation2
        self.snn_activation3 = snn_activation3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        if self.snn_activation1:
            out = self.snn_activation1(out)

        out = self.conv2(out)
        if self.snn_activation2:
            out = self.snn_activation2(out)

        out = self.conv3(out)
        out += identity
        
        if self.snn_activation3:
            out = self.snn_activation3(out)
            
        return out

# =============================================================================
#                         Rebuilt SNN ResNet Architectures
# =============================================================================

class RebuiltSNNResNet(nn.Module):
    """A general, reconstructable SNN ResNet base class."""
    def __init__(self, snn_modules: Dict, snn_tailor_modules: Dict, 
                 block: Type[Union[SNNBasicBlock, SNNBottleneck]], 
                 layers: List[int]):
        super().__init__()
        self.snn_tailor_modules = snn_tailor_modules
        self._snn_idx_counter = 1  # Global counter for activation layers

        # --- Base Layers ---
        self.conv1 = snn_modules['conv1']
        self.snn_activation_conv1 = self._create_snn_activation(0)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --- Stacked ResNet Layers ---
        self.layer1 = self._make_layer(snn_modules, block, layers[0], layer_name='layer1')
        self.layer2 = self._make_layer(snn_modules, block, layers[1], layer_name='layer2')
        self.layer3 = self._make_layer(snn_modules, block, layers[2], layer_name='layer3')
        self.layer4 = self._make_layer(snn_modules, block, layers[3], layer_name='layer4')

        # --- Classifier Head ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = snn_modules['linear']

    def _create_snn_activation(self, tailor_idx: int) -> Optional[nn.Module]:
        keys = [f"{tailor_idx}.0", f"{tailor_idx}.1", f"{tailor_idx}.2"]
        if all(key in self.snn_tailor_modules for key in keys):
            return nn.Sequential(
                self.snn_tailor_modules[keys[0]],
                self.snn_tailor_modules[keys[1]],
                self.snn_tailor_modules[keys[2]]
            )
        return None

    def _make_layer(self, modules: Dict, block: Type[Union[SNNBasicBlock, SNNBottleneck]], num_blocks: int, layer_name: str = ''):
        layers = []
        
        # --- First block (might have downsampling) ---
        shortcut_conv_name = f"{layer_name}.0.shortcut.0"
        shortcut = nn.Sequential(modules[shortcut_conv_name]) if shortcut_conv_name in modules else None

        if block is SNNBasicBlock:
            block_instance = SNNBasicBlock(
                conv1=modules[f'{layer_name}.0.conv1'],
                conv2=modules[f'{layer_name}.0.conv2'],
                shortcut=shortcut,
                snn_activation1=self._create_snn_activation(self._snn_idx_counter),
                snn_activation2=self._create_snn_activation(self._snn_idx_counter + 1)
            )
            self._snn_idx_counter += 2
        elif block is SNNBottleneck:
            block_instance = SNNBottleneck(
                conv1=modules[f'{layer_name}.0.conv1'],
                conv2=modules[f'{layer_name}.0.conv2'],
                conv3=modules[f'{layer_name}.0.conv3'],
                shortcut=shortcut,
                snn_activation1=self._create_snn_activation(self._snn_idx_counter),
                snn_activation2=self._create_snn_activation(self._snn_idx_counter + 1),
                snn_activation3=self._create_snn_activation(self._snn_idx_counter + 2)
            )
            self._snn_idx_counter += 3
        else:
            raise ValueError(f"Unsupported block type: {block}")
        layers.append(block_instance)

        # --- Remaining blocks ---
        for i in range(1, num_blocks):
            if block is SNNBasicBlock:
                block_instance = SNNBasicBlock(
                    conv1=modules[f'{layer_name}.{i}.conv1'],
                    conv2=modules[f'{layer_name}.{i}.conv2'],
                    snn_activation1=self._create_snn_activation(self._snn_idx_counter),
                    snn_activation2=self._create_snn_activation(self._snn_idx_counter + 1)
                )
                self._snn_idx_counter += 2
            elif block is SNNBottleneck:
                block_instance = SNNBottleneck(
                    conv1=modules[f'{layer_name}.{i}.conv1'],
                    conv2=modules[f'{layer_name}.{i}.conv2'],
                    conv3=modules[f'{layer_name}.{i}.conv3'],
                    snn_activation1=self._create_snn_activation(self._snn_idx_counter),
                    snn_activation2=self._create_snn_activation(self._snn_idx_counter + 1),
                    snn_activation3=self._create_snn_activation(self._snn_idx_counter + 2)
                )
                self._snn_idx_counter += 3
            layers.append(block_instance)
        
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        if self.snn_activation_conv1:
            out = self.snn_activation_conv1(out)
        
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out

# =============================================================================
#                         Public Rebuild Functions
# =============================================================================

def _parse_snn_modules(converted_snn_model: nn.Module) -> (Dict, Dict):
    """Internal helper function to parse the flattened SNN model."""
    snn_modules = {name: mod for name, mod in converted_snn_model.named_modules()}
    snn_tailor_modules = {}
    for name, module in snn_modules.items():
        if 'snn tailor' in name:
            name_parts = name.split('.')
            if len(name_parts) >= 3 and name_parts[-2].isdigit() and name_parts[-1].isdigit():
                parent_idx, sub_idx = int(name_parts[-2]), int(name_parts[-1])
                key = f"{parent_idx}.{sub_idx}"
                snn_tailor_modules[key] = module
    return snn_modules, snn_tailor_modules

def rebuild_snn_resnet18(converted_snn_model: nn.Module) -> RebuiltSNNResNet:
    """Reconstructs an SNN ResNet-18 model."""
    snn_modules, snn_tailor_modules = _parse_snn_modules(converted_snn_model)
    return RebuiltSNNResNet(snn_modules, snn_tailor_modules, SNNBasicBlock, [2, 2, 2, 2])

def rebuild_snn_resnet50(converted_snn_model: nn.Module) -> RebuiltSNNResNet:
    """Reconstructs an SNN ResNet-50 model."""
    snn_modules, snn_tailor_modules = _parse_snn_modules(converted_snn_model)
    return RebuiltSNNResNet(snn_modules, snn_tailor_modules, SNNBottleneck, [3, 4, 6, 3])
# =============================================================================
#                         NEW & CORRECTED: Rebuild Function (VGG)
# =============================================================================

# =============================================================================
#                  FINAL ROBUST VGG RECONSTRUCTION
# =============================================================================

class SNNVGGReconstructed(nn.Module):
    """
    A dynamically reconstructed SNN VGG model that holds the new sequential
    features and classifier.
    """
    def __init__(self, features, avgpool, classifier):
        super().__init__()
        self.features = features
        self.avgpool = avgpool
        self.classifier = classifier
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def rebuild_snn_vgg(ann_model: nn.Module, converted_snn_model: nn.Module) -> nn.Module:
    """
    Dynamically rebuilds a sequential SNN VGG model by inspecting the original ANN's architecture.
    This function is robust and works for VGG models with or without BatchNorm layers.

    Args:
        ann_model (nn.Module): The original, pre-trained ANN VGG model.
        converted_snn_model (nn.Module): The SNN model after conversion by spikingjelly.

    Returns:
        nn.Module: A new, properly structured, and sequential SNN model.
    """
    snn_features_list = []
    snn_classifier_list = []
    
    snn_tailor_module = getattr(converted_snn_model, 'snn tailor')
    tailor_idx = 0

    # --- Rebuild the features section ---
    for idx, layer in enumerate(ann_model.features):
        layer_name = str(idx)
        
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
            # If this layer exists in the converted model, add it.
            if hasattr(converted_snn_model.features, layer_name):
                snn_features_list.append(getattr(converted_snn_model.features, layer_name))

        elif isinstance(layer, nn.ReLU):
            # A ReLU is replaced by a tailor block.
            if tailor_idx < len(list(snn_tailor_module.children())):
                tailor_block_container = getattr(snn_tailor_module, str(tailor_idx))
                # Create a callable sequential module from the container's parts
                callable_tailor_block = nn.Sequential(
                    getattr(tailor_block_container, '0'),
                    getattr(tailor_block_container, '1'),
                    getattr(tailor_block_container, '2')
                )
                snn_features_list.append(callable_tailor_block)
                tailor_idx += 1
        
        # BatchNorm2d layers are intentionally skipped as they are absorbed during conversion.

    # --- Rebuild the classifier section ---
    for idx, layer in enumerate(ann_model.classifier):
        layer_name = str(idx)

        if isinstance(layer, (nn.Linear, nn.Dropout)):
            if hasattr(converted_snn_model.classifier, layer_name):
                snn_classifier_list.append(getattr(converted_snn_model.classifier, layer_name))
        
        elif isinstance(layer, nn.ReLU):
            if tailor_idx < len(list(snn_tailor_module.children())):
                tailor_block_container = getattr(snn_tailor_module, str(tailor_idx))
                callable_tailor_block = nn.Sequential(
                    getattr(tailor_block_container, '0'),
                    getattr(tailor_block_container, '1'),
                    getattr(tailor_block_container, '2')
                )
                snn_classifier_list.append(callable_tailor_block)
                tailor_idx += 1

    # Create new sequential modules
    rebuilt_features = nn.Sequential(*snn_features_list)
    rebuilt_classifier = nn.Sequential(*snn_classifier_list)
    
    # Create the final, fully-functional SNN model
    rebuilt_snn = SNNVGGReconstructed(
        features=rebuilt_features,
        avgpool=converted_snn_model.avgpool,
        classifier=rebuilt_classifier
    )
    
    return rebuilt_snn






