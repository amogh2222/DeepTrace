import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT))

from inference.pipeline import (
    InferencePipeline
)

pipeline = InferencePipeline(
    checkpoint_path="checkpoints/best_model.pth"
)

result = pipeline.predict_image(
    "data/sample_images/test.jpg"
)

print("\n========== RESULT ==========\n")

for k, v in result.items():

    print(f"{k}: {v}")
