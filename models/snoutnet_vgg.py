
# models/snoutnet_vgg.py
import torch
import torch.nn as nn
from torchvision import models

class SnoutNetVGG16(nn.Module):
    """
    VGG16 backbone (pretrained) adapted for 2D regression.
    Replaces the final classifier with Linear(... -> 1024 -> 2).
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        self.backbone = models.vgg16(weights=weights)
        in_features = self.backbone.classifier[-1].in_features  # 4096
        self.backbone.classifier[-1] = nn.Identity()
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2),
        )
        if weights is not None:
            meta = weights.meta
            self.mean = meta.get("mean", (0.485, 0.456, 0.406))
            self.std = meta.get("std", (0.229, 0.224, 0.225))
        else:
            self.mean = (0.5, 0.5, 0.5)
            self.std = (0.5, 0.5, 0.5)

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.classifier[:-1](x)
        x = self.regressor(x)
        return x
