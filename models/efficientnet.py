import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
    efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7,
    EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights,
    EfficientNet_B3_Weights, EfficientNet_B4_Weights, EfficientNet_B5_Weights,
    EfficientNet_B6_Weights, EfficientNet_B7_Weights,
)

class EfficientNetClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        variant: str = "b0",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        model_map = {
            "b0": (efficientnet_b0, EfficientNet_B0_Weights),
            "b1": (efficientnet_b1, EfficientNet_B1_Weights),
            "b2": (efficientnet_b2, EfficientNet_B2_Weights),
            "b3": (efficientnet_b3, EfficientNet_B3_Weights),
            "b4": (efficientnet_b4, EfficientNet_B4_Weights),
            "b5": (efficientnet_b5, EfficientNet_B5_Weights),
            "b6": (efficientnet_b6, EfficientNet_B6_Weights),
            "b7": (efficientnet_b7, EfficientNet_B7_Weights),
        }
        if variant not in model_map:
            raise ValueError("variant must be b0‑b7")

        fn, weight_enum = model_map[variant]
        weights = weight_enum.DEFAULT if pretrained else None
        self.net = fn(weights=weights)

        # Swap classifier
        in_f = self.net.classifier[1].in_features
        self.net.classifier[1] = nn.Linear(in_f, num_classes)

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
