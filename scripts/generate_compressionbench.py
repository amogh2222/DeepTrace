from PIL import Image
from pathlib import Path


INPUT_DIR = "data/sample_images"
OUTPUT_DIR = "benchmarks/compressionbench/jpeg"

Path(OUTPUT_DIR).mkdir(
    parents=True,
    exist_ok=True
)

qualities = [10, 20, 40, 60]

for img_path in Path(INPUT_DIR).glob("*"):

    try:

        img = Image.open(img_path).convert("RGB")

        for q in qualities:

            out_path = (
                Path(OUTPUT_DIR)
                / f"{img_path.stem}_q{q}.jpg"
            )

            img.save(
                out_path,
                quality=q
            )

            print(f"Saved: {out_path}")

    except Exception as e:

        print(e)

print("CompressionBench generation complete")
