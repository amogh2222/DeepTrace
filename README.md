# 🧠 DeepTrace — Multimodal Deepfake Detection & Forensics Framework

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-DeepLearning-red?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/Computer-Vision-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Multimodal-AI-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Explainable-AI-green?style=for-the-badge" />
</p>

---

## 📌 Overview

DeepTrace is an advanced multimodal deepfake detection and forensic analysis framework designed to identify synthetic and manipulated media using hybrid spatial-frequency learning pipelines.

The project is being extended toward full-scale audiovisual deepfake forensics research with ongoing work focused on temporal video modeling and audio-based manipulation detection.

A research paper based on DeepTrace is currently under preparation for IEEE publication.

The system combines:

* Spatial artifact learning
* Frequency-domain analysis
* CLIP-aligned feature representations
* Cross-attention fusion
* Explainable AI techniques

Unlike traditional RGB-only deepfake detectors, DeepTrace integrates multiple forensic signals to improve robustness against compression artifacts, unseen generators, and real-world media degradation.

---

# 🚀 Key Features

> Research-oriented multimodal AI framework for robust synthetic media forensics

✅ EfficientNet-B0 spatial encoder for facial artifact detection
✅ DCT-based frequency branch for compression-resistant forensic cues
✅ CLIP-aligned multimodal feature learning
✅ Cross-attention fusion mechanism
✅ GradCAM explainability visualization
✅ Temperature-scaled confidence calibration
✅ Modular PyTorch training pipeline
✅ Hardware-efficient architecture optimized for consumer GPUs

---

# 🏗️ System Architecture

```text
Video/Image Input
        │
        ▼
 Face Detection (MTCNN)
        │
        ▼
 ┌─────────────────────┐
 │ Spatial Encoder     │──► EfficientNet-B0
 └─────────────────────┘
        │
        ▼
 ┌─────────────────────┐
 │ Frequency Encoder   │──► DCT Feature Extraction
 └─────────────────────┘
        │
        ▼
 ┌─────────────────────┐
 │ CLIP Alignment      │──► Visual Representation Learning
 └─────────────────────┘
        │
        ▼
 Cross-Attention Fusion
        │
        ▼
 Real / Fake Classification
        │
        ▼
 GradCAM Explainability
```

---

# 🧠 Core Technologies

| Component                | Technology              |
| ------------------------ | ----------------------- |
| Deep Learning Framework  | PyTorch                 |
| Spatial Feature Learning | EfficientNet-B0         |
| Frequency Analysis       | DCT                     |
| Face Detection           | MTCNN                   |
| Explainability           | GradCAM                 |
| Data Processing          | OpenCV + Albumentations |
| Calibration              | Temperature Scaling     |
| UI                       | Gradio                  |

---

# 📂 Repository Structure

```bash
DeepTrace/
│
├── checkpoints/
├── configs/
├── evaluation/
├── explainability/
├── inference/
├── models/
├── scripts/
├── training/
├── ui/
├── utils/
│
├── README.md
├── requirements.txt
├── calibration.py
├── verify_pipeline.py
└── walkthrough.md
```

---

# 📊 Experimental Performance

| Metric                 | Value    |
| ---------------------- | -------- |
| Accuracy               | 0.90     |
| ROC AUC                | 0.9496   |
| Calibrated Temperature | 4.396576 |
| Optimal Threshold      | 0.162    |

The multimodal fusion strategy significantly improves robustness under compressed and degraded media conditions while maintaining strong generalization performance across manipulation artifacts and real-world distortions.

---

# 🔬 Explainable AI

DeepTrace integrates GradCAM-based explainability to visualize regions influencing model predictions.

The system highlights:

* Facial blending boundaries
* Texture inconsistencies
* Synthetic generation artifacts
* Manipulated facial regions

This improves interpretability and forensic reliability.

---

# ⚡ Hardware Optimization

The training pipeline is optimized for consumer-grade GPUs using:

* Automatic Mixed Precision (AMP)
* Gradient checkpointing
* EfficientNet lightweight backbones
* Memory-aware batching

This enables training on GPUs with approximately 6GB VRAM.

---

# 📦 Dataset Setup

This project supports:

* FaceForensics++
* Celeb-DF
* DFDC
* Kaggle Real-vs-Fake Dataset

Create the following structure:

```bash
data/
├── train/
├── val/
└── test/
```

Datasets are not included in this repository due to size and licensing constraints.

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/amogh2222/DeepTrace.git
cd DeepTrace
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running Inference

```bash
python run_test.py
```

---

# 🏋️ Training

```bash
python training/train.py
```

---

# 🧪 Evaluation

```bash
python evaluation/evaluate.py
```

---

# 🎥 Ongoing Research & Future Improvements

* Full audiovisual deepfake detection pipeline
* Temporal video modeling using Video Swin Transformers
* Audio forgery detection and synchronization analysis
* Lip-sync inconsistency analysis
* Cross-modal audio-video fusion
* Real-time streaming forensic inference
* Video Swin Transformers
* Retrieval-augmented detection
* FFT & wavelet-based analysis
* Identity consistency modeling
* Real-time streaming inference
* Distributed inference deployment

---

# 📈 Applications

* Digital media forensics
* AI-generated media verification
* Social media authenticity checks
* Fraud detection systems
* Synthetic media moderation
* Trust & safety pipelines

---

# 🤝 Contributing

Contributions are welcome.

If you'd like to improve the project:

* Fork the repository
* Create a feature branch
* Submit a pull request

---

# 📜 License

This project is intended for educational and research purposes.

---

---

# 📄 Research Direction

DeepTrace is evolving into a comprehensive multimodal forensic framework targeting:

* Image deepfake detection
* Video manipulation analysis
* Audio spoofing detection
* Cross-modal forensic reasoning
* Explainable AI for synthetic media
* Robustness against adversarial attacks

The project is intended to bridge research and real-world deployment by combining multimodal learning with scalable AI engineering.
