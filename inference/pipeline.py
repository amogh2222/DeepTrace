"""
End-to-end inference pipeline.
"""

import torch
import yaml
import logging

from models.detector import DeepfakeDetector
from models.catmd import CATMDWrapper

from datasets.transforms import (
    get_val_transforms
)

from explainability.gradcam import GradCAM
from explainability.forensic_report import (
    generate_forensic_report
)

from calibration import (
    ModelWithTemperature,
    load_calibration_dict,
    resolve_calibration_path
)

from utils.device import get_device
from utils.checkpoint import load_checkpoint


logger = logging.getLogger(__name__)


class InferencePipeline:

    def __init__(

        self,
        checkpoint_path,
        config_path="configs/config.yaml",
        model_config_path="configs/model_config.yaml",
        device="auto"

    ):

        with open(config_path, "r") as f:

            self.config = yaml.safe_load(f)

        with open(model_config_path, "r") as f:

            model_config = yaml.safe_load(f)

        if device == "auto":

            self.device = get_device()

        else:

            self.device = torch.device(device)

        self.transform = get_val_transforms(
            self.config["data"]["image_size"]
        )

        # -----------------------------------
        # BASE MODEL
        # -----------------------------------

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

        # -----------------------------------
        # CATMD
        # -----------------------------------

        self.model = CATMDWrapper(
            base_model=self.base_model,
            base_threshold=0.162,
            freq_feature_dim=1280
        )

        self.model.to(self.device)

        self.model.eval()

        # -----------------------------------
        # Calibration
        # -----------------------------------

        self.temperature_model = (
            ModelWithTemperature(
                self.model
            )
        )

        self.temperature_model.to(
            self.device
        )

        self.temperature_model.eval()

        self.gradcam = GradCAM(
            self.base_model
        )

        logger.info(
            f"Pipeline initialized "
            f"on {self.device}"
        )
