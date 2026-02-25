import torch
from functools import partial
import torch.nn as nn
from torchvision import models, transforms
from typing import List, Tuple

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

def setup(model: str,
          pretrained: bool,
          class_names: List[str],
          device: torch.device,
          size: str = 'base',
          resnet_layers: int = 50,
          resnet_version: int = 1,
          swin_transformer_version: int = 2) -> Tuple[nn.Module, transforms.Compose]:
    
    if model == 'resnet':
        if resnet_version == 1:
            return setup_resnet(layers=resnet_layers,
                                pretrained=pretrained,
                                class_names=class_names,
                                device=device)
        elif resnet_version == 2:
            return setup_resnet_v2(layers=resnet_layers,
                                   pretrained=pretrained,
                                   class_names=class_names,
                                   device=device)
    
    if model == 'swin_transformer':
        return setup_swin_transformer(version=swin_transformer_version,
                                      size=size,
                                      pretrained=pretrained,
                                      class_names=class_names,
                                      device=device)
    
    if model == 'convnext':
        return setup_convnext(size=size,
                              pretrained=pretrained,
                              class_names=class_names,
                              device=device)

def setup_resnet(layers: int,
                 pretrained: bool,
                 class_names: List[str],
                 device: str) -> Tuple[nn.Module, transforms.Compose]:
    
    if layers == 18:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights) if pretrained else models.resnet18()

    elif layers == 34:
        weights = models.ResNet34_Weights.DEFAULT
        model = models.resnet34(weights) if pretrained else models.resnet34()

    elif layers == 50:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights) if pretrained else models.resnet50()

    elif layers == 101:
        weights = models.ResNet101_Weights.DEFAULT
        model = models.resnet101(weights) if pretrained else models.resnet101()

    elif layers == 152:
        weights = models.ResNet152_Weights.DEFAULT
        model = models.resnet152(weights) if pretrained else models.resnet152()
    
    model.to(device)
    preprocess = weights.transforms()
    
    if pretrained:
        freeze_parameters(model)

    model.fc = nn.Linear(model.fc.in_features, len(class_names)).to(device)

    return model, preprocess

def setup_resnet_v2(layers: int,
                    pretrained: bool,
                    class_names: List[str],
                    device: str) -> Tuple[nn.Module, transforms.Compose]:
    
    if layers == 18:
        model = timm.models.resnetv2.resnetv2_18(pretrained=pretrained)

    elif layers == 34:
        model = timm.models.resnetv2.resnetv2_34(pretrained=pretrained)

    elif layers == 50:
        model = timm.models.resnetv2.resnetv2_50(pretrained=pretrained)

    elif layers == 101:
        model = timm.models.resnetv2.resnetv2_101(pretrained=pretrained)

    elif layers == 152:
        model = timm.models.resnetv2.resnetv2_152(pretrained=pretrained)
    
    preprocess = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    if pretrained:
        freeze_parameters(model)

    model.reset_classifier(num_classes=len(class_names))

    model.to(device)

    return model, preprocess

def setup_swin_transformer(version: int,
                           size: str,
                           pretrained: bool,
                           class_names: List[str],
                           device: str) -> Tuple[nn.Module, transforms.Compose]:
    
    if version == 1:

        if size == 'tiny':
            weights = models.Swin_T_Weights.DEFAULT
            model = models.swin_t(weights) if pretrained else models.swin_t()

        elif size == 'small':
            weights = models.Swin_S_Weights.DEFAULT
            model = models.swin_s(weights) if pretrained else models.swin_s()

        elif size == 'base':
            weights = models.Swin_B_Weights.DEFAULT
            model = models.swin_b(weights) if pretrained else models.swin_b()

    elif version == 2:

        if size == 'tiny':
            weights = models.Swin_V2_T_Weights.DEFAULT
            model = models.swin_v2_t(weights) if pretrained else models.swin_v2_t()

        elif size == 'small':
            weights = models.Swin_V2_S_Weights.DEFAULT
            model = models.swin_v2_s(weights) if pretrained else models.swin_v2_s()

        elif size == 'base':
            weights = models.Swin_V2_B_Weights.DEFAULT
            model = models.swin_v2_b(weights) if pretrained else models.swin_v2_b()

    model.to(device)
    preprocess = weights.transforms()

    if pretrained:
        freeze_parameters(model)

    features = model.head.in_features
    model.head = nn.Linear(features, len(class_names)).to(device)

    return model, preprocess

def setup_convnext(size: str,
                   pretrained: bool,
                   class_names: List[str],
                   device: str) -> Tuple[nn.Module, transforms.Compose]:
    
    if size == 'tiny':
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        model = models.convnext_tiny(weights) if pretrained else models.convnext_tiny()

    elif size == 'small':
        weights = models.ConvNeXt_Small_Weights.DEFAULT
        model = models.convnext_small(weights) if pretrained else models.convnext_small()

    elif size == 'base':
        weights = models.ConvNeXt_Base_Weights.DEFAULT
        model = models.convnext_base(weights) if pretrained else models.convnext_base()

    elif size == 'large':
        weights = models.ConvNeXt_Large_Weights.DEFAULT
        model = models.convnext_large(weights) if pretrained else models.convnext_large()

    model.to(device)
    preprocess = weights.transforms()

    if pretrained:
        freeze_parameters(model)

    lastconv_output_channels = 1024
    norm_layer = partial(models.convnext.LayerNorm2d, eps=1e-6)

    model.classifier = nn.Sequential(
        norm_layer(lastconv_output_channels),
        nn.Flatten(1),
        nn.Linear(lastconv_output_channels, len(class_names))
    ).to(device)

    return model, preprocess

def freeze_parameters(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False