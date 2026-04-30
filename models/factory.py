"""
models/factory.py — Pretrained backbone factory.
Each model is returned with its final classification head replaced
to match num_classes. Supports freeze/unfreeze for staged fine-tuning.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    MobileNet_V3_Small_Weights,
)

SUPPORTED_BACKBONES = ("efficientnet_b0", "resnet50", "mobilenet_v3_small")


def build_model(backbone: str, num_classes: int) -> nn.Module:
    """
    Build a pretrained model with a custom classification head.

    Args:
        backbone:    One of SUPPORTED_BACKBONES.
        num_classes: Number of output classes.

    Returns:
        nn.Module with backbone weights initialised from ImageNet.
    """
    # FIX: validate early — a broken Linear won't error until forward pass
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")

    backbone = backbone.lower()

    if backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),          # FIX: removed inplace=True — inconsistent
                                        # with ResNet and can cause gradient issues
            nn.Linear(in_features, num_classes),
        )

    elif backbone == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    elif backbone == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        # FIX: replace the full classifier instead of only swapping the last layer.
        # Previously MobileNetV3 kept its built-in Dropout(0.2) + hidden Linear(576,1024)
        # while the other two got a clean Dropout(0.3) → Linear head — inconsistent
        # regularisation across the ensemble. Now all three have the same head structure.
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    else:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose from {SUPPORTED_BACKBONES}.")

    return model


def freeze_backbone(model: nn.Module, backbone: str) -> None:
    backbone = backbone.lower()
    head_attr = _head_attr(backbone)

    for name, param in model.named_parameters():
        if not name.startswith(head_attr):
            param.requires_grad = False

    # Freeze BN stats during frozen phase so the pretrained running mean/var
    # aren't corrupted by the new dataset distribution before the backbone adapts.
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m.eval()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[freeze] Trainable params: {trainable:,}")


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True

    # Restore BN to training mode so running stats update during fine-tuning.
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = True
            m.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[unfreeze] Trainable params: {trainable:,}")


def get_param_groups(model: nn.Module, backbone: str, lr_backbone: float, lr_head: float):
    """
    Return discriminative learning rate param groups:
    - backbone parameters get lr_backbone
    - head parameters get lr_head
    """
    head_attr = _head_attr(backbone.lower())
    backbone_params, head_params = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(head_attr):
            head_params.append(param)
        else:
            backbone_params.append(param)

    # Warn if either group is empty — would silently produce a broken optimizer
    if not backbone_params:
        print("[WARN] get_param_groups: backbone_params is empty — is the model still frozen?")
    if not head_params:
        print("[WARN] get_param_groups: head_params is empty — check _head_attr mapping.")

    return [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params,     "lr": lr_head},
    ]


def _head_attr(backbone: str) -> str:
    """Returns the attribute prefix for the classification head."""
    mapping = {
        "efficientnet_b0":     "classifier",
        "resnet50":            "fc",
        "mobilenet_v3_small":  "classifier",
    }
    return mapping[backbone]