from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int, pretrained: bool = False):
    if pretrained:
        torch.hub.set_dir(str(Path.cwd() / ".torch"))
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
