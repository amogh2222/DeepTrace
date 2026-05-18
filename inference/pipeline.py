"""
End-to-end inference pipeline.
Input: image or video file → Output: prediction JSON + heatmap + forensic report.
"""

import torch
import cv2
import numpy as np
import json
from pathlib import Path
import logging
import yaml

from models.detector import DeepfakeDetector
from models.catmd import CATMDWrapper

from datasets.transforms import (
    get_val_transforms,
    apply_dct_transform
)

from explainability.gradcam import GradCAM
from explainability.forensic_report import generate_forensic_report

from calibration import (
    ModelWithTemperature,
    load_calibration_dict,
    resolve_calibration_path
)

from utils.device import get_device
from utils.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)

MANIPULATION_LABELS = [
    "real",
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures"
]


class InferencePipeline:
    """Full inference pipeline for deepfake detection."""

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = "configs/config.yaml",
        model_config_path: str = "configs/model_config.yaml",
        device: str = "auto",
        calibration_path: str = None,
        temperature_path: str = None,
    ):

        # Load configs
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)

        # Device
        if device == "auto":
            self.device = get_device()
        else:
            self.device = torch.device(device)

        self.use_amp = (
            self.device.type == "cuda"
            and self.config["hardware"].get(
                "mixed_precision",
                True
            )
        )

        # Image size
        self.image_size = self.config["data"]["image_size"]

        self.num_frames = self.config["data"]["num_frames"]

        self.transform = get_val_transforms(
            self.image_size
        )

        # -------------------------------
        # BASE MODEL
        # -------------------------------

        self.base_model = DeepfakeDetector(
            config=model_config
        )

        load_checkpoint(
            checkpoint_path,
            self.base_model,
            device=self.device
        )

        self.base_model.to(self.device)
        self.base_model.eval()

        # -------------------------------
        # CATMD WRAPPER
        # -------------------------------

        self.catmd_model = CATMDWrapper(
            base_model=self.base_model,
            base_threshold=0.162,
            freq_feature_dim=1280
        )

        self.catmd_model.to(self.device)
        self.catmd_model.eval()

        # -------------------------------
        # TEMPERATURE CALIBRATION
        # -------------------------------

        self.model = ModelWithTemperature(
            self.catmd_model
        )

        self.model.to(self.device)
        self.model.eval()

        # Decision threshold
        self.threshold: float = (
            self.config
            .get("evaluation", {})
            .get("threshold", 0.5)
        )

        calibration_override = (
            calibration_path
            or temperature_path
        )

        self.calibration_path = (
            resolve_calibration_path(
                checkpoint_path,
                calibration_override
            )
        )

        self.calibration_loaded = False

        if self.calibration_path.exists():

            calib = load_calibration_dict(
                str(self.calibration_path)
            )

            self.model.set_temperature_value(
                calib["temperature"]
            )

            thr = (
                calib.get("threshold")
                or calib.get("optimal_threshold")
            )

            if thr is not None:
                self.threshold = float(thr)

            self.calibration_loaded = True

            logger.info(
                "Loaded calibration: "
                "T=%.6f threshold=%.4f from %s",
                self.model.temperature_value,
                self.threshold,
                self.calibration_path,
            )

        else:

            logger.info(
                "No calibration.json found, "
                "using T=1.0 threshold=%.4f",
                self.threshold
            )

        # -------------------------------
        # GRADCAM
        # -------------------------------

        self.gradcam = GradCAM(
            self.base_model
        )

        logger.info(
            f"Inference pipeline ready "
            f"(device={self.device})"
        )
