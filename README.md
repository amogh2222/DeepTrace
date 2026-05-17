<div align="center">

# 🔍 DeepTrace
### Multimodal Deepfake Detection & Forensics Framework

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)](https://python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗%20Live%20Demo-Spaces-FFD21F?style=flat-square)](https://huggingface.co/spaces/amogh2222/deeptrace)
[![Paper](https://img.shields.io/badge/Paper-Under%20Review%20(IEEE)-orange?style=flat-square)](https://github.com/amogh2222/DeepTrace)

**0.90 Accuracy &nbsp;·&nbsp; 0.9496 AUC &nbsp;·&nbsp; Runs on 6GB VRAM**

[**🤗 Try Live Demo**](https://huggingface.co/spaces/amogh2222/deeptrace) &nbsp;·&nbsp; [**📄 Report**](docs/report.pdf) &nbsp;·&nbsp; [**⚙️ Architecture**](#architecture)

</div>

---

## The Problem

Standard deepfake detectors are trained to spot RGB-level artifacts — subtle pixel inconsistencies left by GAN or diffusion generators. The problem: **a single JPEG compression cycle wipes most of these out.**

Upload a deepfake to Twitter, download it, run a standard detector — detection rate collapses. This is the real-world failure mode that makes existing single-modality models impractical for forensics.

DeepTrace addresses this by fusing three forensic signals that are independently resilient to different classes of degradation.

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

### vs. Baselines

| Model | AUC | Limitation |
|-------|-----|------------|
| MesoNet | ~0.84 | Spatial only, degrades under compression |
| XceptionNet | ~0.91 | Strong on FF++ but large cross-dataset gap |
| **DeepTrace V1 (ours)** | **0.9496** | Frequency + CLIP fusion, calibrated output |

---

## Architecture

Three parallel branches, one fusion layer.

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
   ┌─────┴─────────────┐
   ▼                   ▼
Real/Fake Class    GradCAM Heatmap
+ Manip Type       (explainability)
+ Confidence Score
  (temperature-scaled)
```

**Why each branch exists:**

- **Spatial (EfficientNet-B0):** Catches texture inconsistencies, blending boundary artifacts, geometric distortions. Best signal on uncompressed media.
- **Frequency (DCT on YCrCb):** Deepfake generators leave characteristic patterns in the frequency domain that survive JPEG compression. This branch keeps working after social media re-encoding.
- **Semantic (CLIP ViT-B/32, frozen):** Grounds representations in a rich visual-semantic space. Improves generalization to manipulation techniques the model has never seen in training.

---

## Explainability

GradCAM heatmaps show exactly what the model is looking at — not just a score.

For **fake images**: activations concentrate on blending boundaries, unnatural skin texture zones, and eye/mouth regions where synthesis artifacts cluster.

For **real images**: attention is diffuse — confirming the model isn't latching onto spurious background features.

```
docs/assets/
├── fake_input.jpg      ← original fake face
├── fake_gradcam.jpg    ← GradCAM activation map
├── fake_overlay.jpg    ← overlay (blending artifacts highlighted)
├── real_input.jpg
├── real_gradcam.jpg
└── real_overlay.jpg
```

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

**Gradio UI (local):**
```bash
python ui/app.py
# or demo mode (no checkpoint needed):
python ui/app.py --demo
```

**Evaluate on test set:**
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
| Mixed precision | AMP (torch.cuda.amp) |
| Memory | Gradient checkpointing |
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
├── models/             DeepfakeDetector architecture definition
├── scripts/            dataset preprocessing utilities
├── training/           train loop, loss functions, schedulers
├── ui/                 Gradio web app
├── utils/              shared helpers
├── calibration.py      post-hoc temperature scaling
├── requirements.txt
└── DATASET_SETUP.md
```

---

## Roadmap

- [x] V1: EfficientNet-B0 spatial + DCT frequency + CLIP alignment
- [x] Cross-attention fusion
- [x] GradCAM explainability
- [x] Temperature-scaled confidence calibration
- [x] Gradio UI + HuggingFace Spaces demo
- [ ] V2: Video Swin Transformer temporal pipeline
- [ ] Audio forgery detection (lip-sync inconsistency analysis)
- [ ] FAISS-based retrieval-augmented detection
- [ ] FFT + wavelet multi-spectral analysis
- [ ] HuggingFace Hub model weights upload

---

## Citation

```bibtex
@misc{srivastava2026deeptrace,
  title   = {DeepTrace: Multimodal Deepfake Detection via Hybrid Spatial-Frequency Learning},
  author  = {Srivastava, Amogh and Rohit and Gaba, Udit},
  year    = {2026},
  institution = {Manipal University Jaipur},
  url     = {https://github.com/amogh2222/DeepTrace}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

<div align="center">

Built at Manipal University Jaipur &nbsp;·&nbsp; Department of Data Science & Engineering &nbsp;·&nbsp; 2026

</div>
