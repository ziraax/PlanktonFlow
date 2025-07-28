import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    ResNet101_Weights, ResNet152_Weights,
)


class ResNetClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        variant: str = "50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        model_map = {
            "18": (resnet18, ResNet18_Weights),
            "34": (resnet34, ResNet34_Weights),
            "50": (resnet50, ResNet50_Weights),
            "101": (resnet101, ResNet101_Weights),
            "152": (resnet152, ResNet152_Weights),
        }
        if variant not in model_map:
            raise ValueError("variant must be one of: 18, 34, 50, 101, 152")

        fn, weight_enum = model_map[variant]
        weights = weight_enum.DEFAULT if pretrained else None
        self.net = fn(weights=weights)

        in_f = self.net.fc.in_features
        self.net.fc = nn.Linear(in_f, num_classes)

        if freeze_backbone:
            for p in self.net.parameters():
                p.requires_grad = False
            for p in self.net.fc.parameters():
                p.requires_grad = True

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)
