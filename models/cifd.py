import torch
import torch.nn as nn


class CIFD(nn.Module):

    def __init__(
        self,
        feature_dim=1280
    ):
        super().__init__()

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, features):

        proj = self.projector(features)

        proj = nn.functional.normalize(
            proj,
            dim=-1
        )

        return proj
