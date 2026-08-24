import torch
import torch.nn as nn
import timm


class CompressionStateEstimator(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0
        )

        self.regressor = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        feat = self.backbone(x)

        quality = self.regressor(feat)

        return quality
