import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT))

import torch
from PIL import Image
import torchvision.transforms as T

from models.compression_state_estimator import (
    CompressionStateEstimator
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = CompressionStateEstimator()

model.load_state_dict(
    torch.load(
        "cse_trained.pt",
        map_location=device
    )
)

model.to(device)

model.eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

image_path = "data/sample_images/test.jpg"

img = Image.open(
    image_path
).convert("RGB")

img = transform(img)

img = img.unsqueeze(0).to(device)

with torch.no_grad():

    quality = model(img)

print(
    "Compression Quality Score:",
    quality.item()
)
