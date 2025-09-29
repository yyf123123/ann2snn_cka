import torch
import torch.nn as nn
from torchvision import models

def vgg16_bn_cifar10(pretrained: bool = True, num_classes: int = 10) -> models.VGG:
    """
    构造一个适用于CIFAR-10数据集的VGG16-BN模型 (带批量归一化)。

    该函数会加载在ImageNet上预训练的VGG16-BN模型，并将其分类器部分
    替换为一个更适合CIFAR-10（32x32图像，10个类别）的新分类器。

    Args:
        pretrained (bool): 如果为True，则加载ImageNet预训练权重。默认为True。
        num_classes (int): 数据集的类别数量。默认为10。

    Returns:
        torchvision.models.VGG: 修改并配置好的VGG16-BN模型。
    """
    # 1. 加载预训练的VGG16_BN模型
    # 修正：通过 'models' 模块来访问权重枚举
    weights = models.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vgg16_bn(weights=weights)

    # 2. 修改分类器以适应CIFAR-10
    # VGG16的原始分类器输入维度是为224x224图像设计的 (512*7*7)
    # 对于CIFAR-10的32x32图像，特征提取后的维度是 512*1*1
    # 我们添加一个自适应平均池化层来确保输入分类器的维度始终是 512*1*1
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    # 替换为一个新的、复杂度更低的分类器，并添加Dropout来防止过拟合
    model.classifier = nn.Sequential(
        nn.Linear(512 * 1 * 1, 512),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        nn.Linear(512, 256),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        nn.Linear(256, num_classes),
    )
    
    # 初始化新分类器的权重
    for m in model.classifier.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    return model

# ==================== 新增代码区域 ====================

def vgg16_cifar10(pretrained: bool = True, num_classes: int = 10) -> models.VGG:
    """
    构造一个适用于CIFAR-10数据集的标准VGG16模型 (不带批量归一化)。

    该函数会加载在ImageNet上预训练的标准VGG16模型，并将其分类器部分
    替换为一个更适合CIFAR-10的新分类器。

    Args:
        pretrained (bool): 如果为True，则加载ImageNet预训练权重。默认为True。
        num_classes (int): 数据集的类别数量。默认为10。

    Returns:
        torchvision.models.VGG: 修改并配置好的标准VGG16模型。
    """
    # 1. 加载预训练的标准VGG16模型 (不含BN层)
    # 修正：通过 'models' 模块来访问权重枚举
    weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vgg16(weights=weights)

    # 2. 修改分类器以适应CIFAR-10 (逻辑与BN版本完全相同)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    model.classifier = nn.Sequential(
        nn.Linear(512 * 1 * 1, 512),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        nn.Linear(512, 256),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        nn.Linear(256, num_classes),
    )
    
    # 初始化新分类器的权重
    for m in model.classifier.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    return model

