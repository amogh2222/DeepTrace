import torch
import torch.nn as nn

from models.compression_state_estimator import (
    CompressionStateEstimator
)

from models.ccta import (
    CompressionConditionedThreshold
)

from models.cifd import CIFD


class CATMDWrapper(nn.Module):

    def __init__(

        self,
        base_model,
        cse_checkpoint="cse_trained.pt",
        base_threshold=0.162,
        freq_feature_dim=1280,
        freeze_base=True

    ):

        super().__init__()

        self.base_model = base_model

        # ---------------------------------
        # Compression estimator
        # ---------------------------------

        self.cse = CompressionStateEstimator()

        try:

            self.cse.load_state_dict(
                torch.load(
                    cse_checkpoint,
                    map_location="cpu"
                )
            )

            print(
                "[CATMD] Loaded trained "
                "CSE checkpoint"
            )

        except Exception as e:

            print(
                "[CATMD] Warning:",
                e
            )

        # ---------------------------------
        # Adaptive thresholding
        # ---------------------------------

        self.ccta = CompressionConditionedThreshold(
            base_threshold=base_threshold
        )

        # ---------------------------------
        # Compression invariant features
        # ---------------------------------

        self.cifd = CIFD(
            feature_dim=freq_feature_dim
        )

        if freeze_base:

            for p in self.base_model.parameters():

                p.requires_grad_(False)

    def forward(

        self,
        images,
        mode="image",
        **kwargs

    ):

        quality_score = self.cse(images)

        base_out = self.base_model(
            images,
            mode=mode,
            **kwargs
        )

        logits = base_out["binary_logit"]

        probs = torch.sigmoid(logits)

        ccta_out = self.ccta(
            logits,
            quality_score
        )

        q = quality_score.mean().item()

        if q < 0.30:

            compression_label = (
                "heavily_compressed"
            )

        elif q < 0.65:

            compression_label = (
                "moderately_compressed"
            )

        else:

            compression_label = (
                "near_pristine"
            )

        result = dict(base_out)

        result.update({

            "fake_probability":
                probs.mean().item(),

            "is_fake":
                ccta_out["predictions"]
                .bool()
                .any()
                .item(),

            "threshold_used":
                ccta_out["threshold"]
                .mean()
                .item(),

            "threshold_shift":
                ccta_out["threshold_shift"]
                .mean()
                .item(),

            "compression_quality":
                q,

            "compression_label":
                compression_label,

            "adaptive_prediction":
                "FAKE"
                if ccta_out["predictions"]
                .bool()
                .any()
                .item()
                else "REAL"
        })

        return result
