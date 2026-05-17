<div align="center">

# 🔍 DeepTrace
### Multimodal Deepfake Detection & Forensics Framework

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)](https://python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗%20Live%20Demo-Spaces-FFD21F?style=flat-square)](https://huggingface.co/spaces/amogh2/deeptrace)
[![Paper](https://img.shields.io/badge/Paper-Under%20Review%20(IEEE)-orange?style=flat-square)](#citation)

**0.90 Accuracy &nbsp;·&nbsp; 0.9496 AUC &nbsp;·&nbsp; Runs on 6GB VRAM**

[**🤗 Try Live Demo**](https://huggingface.co/spaces/amogh2/deeptrace) &nbsp;·&nbsp; [**⚙️ Architecture**](#architecture) &nbsp;·&nbsp; [**📊 Results**](#results)

</div>

---

## The Problem

Deepfake detectors built on RGB spatial features alone fail in the real world. A single JPEG compression cycle — the kind that happens every time content is uploaded to social media — strips the subtle pixel-level artifacts that standard CNNs rely on. Detection rates collapse on content that has been re-encoded even once.

DeepTrace addresses this by fusing three forensic signals that are independently resilient to different classes of degradation, making detection robust under real-world media conditions.

---

## Results

<div align="center">

| Metric | Value |
|--------|-------|
| Accuracy | **0.90** |
| ROC AUC | **0.9496** |
| Calibrated Temperature | 4.397 |
| Optimal Decision Threshold | **0.162** (vs default 0.5) |

*Evaluated on a strictly held-out split of the Kaggle Real-vs-Fake dataset.*

</div>

> The threshold shift from 0.5 → 0.162 via post-hoc temperature scaling significantly reduces false negatives — critical for forensic use cases where missing a fake is worse than a false alarm.

### Baseline Comparison

| Model | AUC | Weakness |
|-------|-----|----------|
| MesoNet | ~0.84 | Spatial only — collapses under compression |
| XceptionNet | ~0.91 | Strong on FF++ but large cross-dataset gap |
| **DeepTrace V1 (ours)** | **0.9496** | Frequency + CLIP fusion, calibrated output |

---

## Architecture

Three parallel branches, fused via cross-attention.

```
Input Image / Video Frame
         │
         ▼
   MTCNN Face Detection + Crop (224×224)
         │
   ┌─────┴──────────────┬──────────────────┐
   ▼                    ▼                  ▼
Spatial Branch     Frequency Branch    Semantic Branch
EfficientNet-B0    EfficientNet-B0     CLIP ViT-B/32
(RGB face crop)    (8×8 DCT, YCrCb)   (frozen encoder)
   └─────┬──────────────┴──────────────────┘
         ▼
   Cross-Attention Fusion
   (learns per-input branch weighting)
         │
   ┌─────┴─────────────────┐
   ▼                       ▼
Real/Fake + Manip Type   GradCAM Heatmap
+ Calibrated Confidence  (explainability)
```

**Why each branch:**

- **Spatial (EfficientNet-B0):** Catches texture inconsistencies, blending boundaries, geometric distortions. Best signal on uncompressed or lightly compressed media.
- **Frequency (DCT on YCrCb):** Deepfake generators leave characteristic patterns in frequency space that survive JPEG re-encoding. This branch keeps working after social media compression.
- **Semantic (CLIP ViT-B/32, frozen):** Grounds representations in a rich visual-semantic space, improving generalization to manipulation techniques unseen during training.

---

## Explainability

GradCAM heatmaps show exactly what the model attends to — not just a score.

- **Fake images:** Activations concentrate on blending boundaries, unnatural skin texture zones, eye/mouth regions where synthesis artifacts cluster.
- **Real images:** Attention remains diffuse — confirming the model isn't latching onto background features.

---

## Quickstart

```bash
git clone https://github.com/amogh2222/DeepTrace.git
cd DeepTrace
pip install -r requirements.txt
```

**Single image inference:**
```bash
python inference/predict.py --input path/to/image.jpg
```

**Gradio UI:**
```bash
# With model checkpoint:
python ui/app.py --checkpoint checkpoints/kaggle_realfake/best.pt

# Demo mode (no checkpoint needed):
python ui/app.py --demo
```

**Evaluate:**
```bash
python evaluation/evaluate.py --checkpoint checkpoints/kaggle_realfake/best.pt
```

**Train from scratch:**
```bash
python training/train.py --config configs/v1.yaml
```

---

## Training Details

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| LR Schedule | Cosine with warmup |
| Early stopping | Validation loss |
| Augmentation | JPEG compression, Gaussian noise, color jitter, coarse dropout |
| Mixed precision | AMP (`torch.cuda.amp`) |
| Memory optimization | Gradient checkpointing |
| Hardware | RTX 4050 · 6GB VRAM |

**Multi-task loss:**
```
L = BCE(real/fake) + CrossEntropy(manip type) + CLIP alignment loss + confidence consistency loss
```

---

## Dataset Setup

Supports FaceForensics++, Celeb-DF, DFDC, and Kaggle Real-vs-Fake. See [`DATASET_SETUP.md`](DATASET_SETUP.md).

```
data/
├── train/
│   ├── real/
│   └── fake/
├── val/
└── test/
```

---

## Repo Structure

```
DeepTrace/
├── checkpoints/        trained model weights
├── configs/            training + model config YAMLs
├── evaluation/         metrics, ROC, calibration scripts
├── explainability/     GradCAM implementation
├── inference/          single image + batch inference pipeline
├── models/             DeepfakeDetector architecture
├── scripts/            dataset preprocessing utilities
├── training/           train loop, losses, schedulers
├── ui/                 Gradio web app
├── utils/              shared helpers
├── calibration.py      temperature scaling
└── requirements.txt
```

---

## Roadmap

- [x] V1: EfficientNet-B0 spatial + DCT frequency + CLIP alignment
- [x] Cross-attention fusion
- [x] GradCAM explainability
- [x] Temperature-scaled confidence calibration
- [x] Gradio UI + HuggingFace Spaces demo
- [ ] V2: Video Swin Transformer temporal pipeline
- [ ] Audio forgery detection (lip-sync inconsistency)
- [ ] FAISS-based retrieval-augmented detection
- [ ] FFT + wavelet multi-spectral analysis
- [ ] HuggingFace Hub model weights

---

## Citation

```bibtex
@misc{deeptrace2026,
  title   = {DeepTrace: Multimodal Deepfake Detection via Hybrid Spatial-Frequency Learning},
  author  = {Srivastava, Amogh},
  year    = {2026},
  url     = {https://github.com/amogh2222/DeepTrace}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
