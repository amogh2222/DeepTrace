import torch
import torch.nn as nn


class CompressionConditionedThreshold(nn.Module):

    def __init__(
        self,
        base_threshold=0.5,
        max_shift=0.15
    ):
        super().__init__()

        self.base_threshold = base_threshold
        self.max_shift = max_shift

    def forward(self, logits, compression_quality):

        probs = torch.sigmoid(logits)

        shift = (
            (0.5 - compression_quality)
            * self.max_shift
        )

        adaptive_threshold = (
            self.base_threshold + shift
        )

        preds = (
            probs > adaptive_threshold
        ).float()

        return {
            "probs": probs,
            "threshold": adaptive_threshold,
            "predictions": preds,
            "threshold_shift": shift
        }
