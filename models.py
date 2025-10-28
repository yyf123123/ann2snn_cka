import torch
import torch.nn as nn
from typing import Dict, List, Type, Union
import torch
import torch.nn as nn
from typing import Dict, List, Type, Union

class SNNBasicBlock(nn.Module):
    """
    SNN版本的BasicBlock（适用于ResNet-18/34）。
    这个块的结构与原始ANN的BasicBlock类似，但用脉冲激活函数替换了ReLU，
    并且不包含BatchNorm层，因为BN层的参数通常在ANN-SNN转换时被融合到卷积层中。
    """
    expansion = 1

    def __init__(self, conv1, conv2, shortcut=None,
                 snn_activation1=None, snn_activation2=None):
        super().__init__()
        self.conv1 = conv1
        self.snn_activation1 = snn_activation1
        self.conv2 = conv2
        self.shortcut = shortcut if shortcut is not None else nn.Sequential()
        self.snn_activation2 = snn_activation2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播逻辑遵循标准的ResNet BasicBlock流程：
        x -> conv1 -> snn_act1 -> conv2 -> + shortcut -> snn_act2 -> out
        """
        identity = self.shortcut(x)

        out = self.conv1(x)
        if self.snn_activation1:
            out = self.snn_activation1(out)

        out = self.conv2(out)
        out += identity

        if self.snn_activation2:
            out = self.snn_activation2(out)

        return out


class RebuiltSNNResNet(nn.Module):
    """
    一个通用的、重建后的SNN ResNet模型。
    这个类旨在将spikingjelly转换器生成的扁平化SNN结构，
    重新组织成一个类似于原始PyTorch ResNet的分层结构，以便于后续的微调和分析。
    """
    def __init__(
        self,
        ann_model: nn.Module,
        snn_model: nn.Module,
        snn_tailor_modules: Dict,
        block: Type[SNNBasicBlock],
        layers: List[int],
    ):
        super().__init__()
        self.inplanes = 64
        
        # 从转换后的snn_model中提取第一层卷积
        self.conv1 = snn_model.conv1
        
        # 提取第一个脉冲激活层（在conv1之后）
        self.snn_activation_conv1 = snn_tailor_modules['0']
        
        # 关键修正：CIFAR-10的ResNet-18没有初始的MaxPool层，将其移除。
        # 这是导致精度下降的主要原因之一。
        
        # 使用一个计数器来正确地从snn_tailor_modules中提取激活层
        block_idx_counter = 0

        # 依次重建ResNet的四个主要层
        self.layer1, block_idx_counter = self._make_layer(ann_model.layer1, snn_model.layer1, snn_tailor_modules, block, 64, layers[0], block_idx_counter)
        self.layer2, block_idx_counter = self._make_layer(ann_model.layer2, snn_model.layer2, snn_tailor_modules, block, 128, layers[1], block_idx_counter, stride=2)
        self.layer3, block_idx_counter = self._make_layer(ann_model.layer3, snn_model.layer3, snn_tailor_modules, block, 256, layers[2], block_idx_counter, stride=2)
        self.layer4, block_idx_counter = self._make_layer(ann_model.layer4, snn_model.layer4, snn_tailor_modules, block, 512, layers[3], block_idx_counter, stride=2)

        # 关键修正：使用与原始模型匹配的AvgPool2d(4)而不是AdaptiveAvgPool2d
        self.avgpool = nn.AvgPool2d(4, stride=4) # 匹配 F.avg_pool2d(x, 4)

        # 从snn_model中提取Flatten和Linear层
        self.flatten = snn_model.flatten
        self.linear = snn_model.linear

    def _make_layer(
        self,
        ann_layer: nn.Sequential,
        snn_layer: nn.Module,
        snn_tailor_modules: Dict,
        block: Type[SNNBasicBlock],
        planes: int,
        blocks: int,
        block_idx_counter: int,
        stride: int = 1,
    ) -> (nn.Sequential, int):
        """
        一个辅助函数，用于构建ResNet中的每一个layer（例如layer1, layer2等）。
        """
        layers = []
        for i in range(blocks):
            # 从原始ANN和转换后的SNN中获取对应的模块
            ann_block = ann_layer[i]
            snn_block_module = getattr(snn_layer, str(i))

            # 提取卷积层
            conv1 = snn_block_module.conv1
            conv2 = snn_block_module.conv2
            
            # 提取shortcut连接
            shortcut = getattr(snn_block_module, 'shortcut', None)
            if shortcut is not None:
                # 修正：SNN中的shortcut是nn.Module容器，需要包装成Sequential以支持forward调用
                shortcut = nn.Sequential(getattr(shortcut, '0'))
            else:
                # 如果SNN中没有shortcut，就从ANN中继承一个空的Sequential
                shortcut = ann_block.shortcut

            # 根据block计数器，精确地映射snn_tailor中的激活层
            # 每个BasicBlock有两个激活层
            snn_activation1 = snn_tailor_modules[str(1 + 2 * block_idx_counter)]
            snn_activation2 = snn_tailor_modules[str(2 + 2 * block_idx_counter)]
            
            # 创建SNN BasicBlock
            layers.append(block(conv1, conv2, shortcut, snn_activation1, snn_activation2))
            block_idx_counter += 1

        return nn.Sequential(*layers), block_idx_counter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.snn_activation_conv1(x)
        
        # 移除了错误的maxpool层
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


def rebuild_snn_resnet18(model: nn.Module, snn_model: nn.Module) -> RebuiltSNNResNet:
    """
    重建函数，用于将spikingjelly转换的ResNet-18 SNN模型重建为更结构化的形式。
    
    Args:
        model (nn.Module): 原始的ANN模型，用于参考结构。
        snn_model (nn.Module): 经过spikingjelly.ann2snn.Converter转换后的SNN模型。

    Returns:
        RebuiltSNNResNet: 一个结构清晰、与原始ANN结构更相似的SNN模型。
    """
    # 从扁平化的snn_model中提取'snn tailor'模块（包含所有激活函数）
    snn_tailor_main_module = getattr(snn_model, 'snn tailor')
    
    # 关键修正：将snn tailor中的每个子模块（例如'0', '1'等）
    # 重新包装成一个nn.Sequential。这些子模块本身没有forward方法，
    # 但将它们的子层（scaler, IFNode, scaler）放入Sequential后，整体就可调用了。
    snn_tailor_modules = {}
    for block_name, block_module in snn_tailor_main_module.named_children():
        # block_module包含 '0', '1', '2'。我们将它们放入一个序列中。
        layers = [mod for _, mod in block_module.named_children()]
        snn_tailor_modules[block_name] = nn.Sequential(*layers)
    
    # 创建并返回重建后的模型
    rebuilt_model = RebuiltSNNResNet(model, snn_model, snn_tailor_modules, SNNBasicBlock, [2, 2, 2, 2])
    return rebuilt_model

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

