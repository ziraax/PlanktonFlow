import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    DenseNet121_Weights, DenseNet169_Weights,
    DenseNet201_Weights, DenseNet161_Weights,
)


class DenseNetClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        variant: str = "121",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # ----- Choose variant ----------------------------------------------
        weight_map = {
            "121": DenseNet121_Weights.IMAGENET1K_V1,
            "169": DenseNet169_Weights.IMAGENET1K_V1,
            "201": DenseNet201_Weights.IMAGENET1K_V1,
            "161": DenseNet161_Weights.IMAGENET1K_V1,
        }
        if variant not in weight_map:
            raise ValueError("variant must be one of 121/169/201/161")

        weights = weight_map[variant] if pretrained else None
        self.net = getattr(models, f"densenet{variant}")(weights=weights)

        # Replace classifier
        in_f = self.net.classifier.in_features
        self.net.classifier = nn.Linear(in_f, num_classes)

        # Freeze backbone if asked
        if freeze_backbone:
            for p in self.net.features.parameters():
                p.requires_grad = False
            for p in self.net.classifier.parameters():
                p.requires_grad = True

    def forward(self, x): return self.net(x)


    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)


