# CheXVision

**Large-Scale Chest X-Ray Pathology Detection** — Deep Learning & Big Data Project

[![CI](https://github.com/arudaev/chexvision/actions/workflows/ci.yml/badge.svg)](https://github.com/arudaev/chexvision/actions/workflows/ci.yml)
[![Dataset](https://img.shields.io/badge/HF-Dataset-blue?logo=huggingface)](https://huggingface.co/datasets/HlexNC/chest-xray-14)
[![Demo](https://img.shields.io/badge/HF-Demo-orange?logo=huggingface)](https://huggingface.co/spaces/HlexNC/chexvision-demo)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

CheXVision tackles **automated chest X-ray pathology detection** at scale using the [NIH Chest X-ray14](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community) dataset — 112,120 frontal-view X-ray images labeled with 14 pathological conditions.

We implement and compare two deep learning approaches on two distinct tasks:

| | Task 1: Multi-Label Classification | Task 2: Binary Classification |
|---|---|---|
| **What** | Detect 14 pathologies simultaneously | Normal vs. Abnormal screening |
| **Model 1** | Custom ResNet-style CNN (from scratch) | Same architecture, binary head |
| **Model 2** | DenseNet-121 transfer learning (CheXNet) | Same architecture, binary head |

### Why This Matters

- **112,120 images** — true big-data scale requiring efficient data pipelines
- **Medical AI** — directly applicable to clinical decision support
- **CheXNet** (Rajpurkar et al., 2017) — landmark paper achieving radiologist-level performance
- **Two tasks, two models** — rigorous comparison of from-scratch vs. transfer learning

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    CheXVision                        │
├─────────────────┬───────────────────────────────────┤
│  Model 1        │  Model 2                          │
│  Custom CNN     │  DenseNet-121 (pretrained)        │
│  (from scratch) │  Fine-tuned on Chest X-ray14      │
├─────────────────┴───────────────────────────────────┤
│              Shared Data Pipeline                    │
│  NIH Chest X-ray14 → Transforms → DataLoader        │
├─────────────────────────────────────────────────────┤
│  Task A: Multi-Label (14 classes, BCE loss)          │
│  Task B: Binary (Normal/Abnormal, BCE loss)          │
├─────────────────────────────────────────────────────┤
│  Evaluation: AUC-ROC, F1, Precision, Recall          │
│  Visualization: Grad-CAM, ROC curves, Confusion Mtx  │
└─────────────────────────────────────────────────────┘
```

## Project Structure

```
chexvision/
├── .github/workflows/     # CI/CD pipelines
├── src/
│   ├── data/              # Dataset, transforms, download utilities
│   ├── models/            # Model 1 (scratch) & Model 2 (transfer)
│   ├── training/          # Training loop, metrics, evaluation
│   └── utils/             # Visualization (Grad-CAM, plots)
├── app/                   # Streamlit demo application
├── configs/               # Hyperparameter configurations
├── notebooks/             # EDA & training notebooks
├── tests/                 # Unit tests
├── Dockerfile             # Container for HF Space deployment
├── requirements.txt       # Dependencies
└── pyproject.toml         # Project metadata
```

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/arudaev/chexvision.git
cd chexvision
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Download Dataset

```bash
python -m src.data.download --output-dir data/
```

This downloads the NIH Chest X-ray14 dataset from HuggingFace and prepares train/val/test splits.

### 3. Train Models

```bash
# Model 1: Custom CNN from scratch
python -m src.training.trainer --config configs/scratch.yaml

# Model 2: DenseNet-121 transfer learning
python -m src.training.trainer --config configs/transfer.yaml
```

### 4. Evaluate & Compare

```bash
python -m src.training.evaluate --model-dir checkpoints/ --compare
```

### 5. Run Streamlit Demo

```bash
streamlit run app/app.py
```

## Dataset

**NIH Chest X-ray14** (National Institutes of Health Clinical Center):

- **112,120** frontal-view chest X-ray images
- **14 pathology labels**: Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia
- **Multi-label**: Each image can have 0 or more conditions
- **Binary split**: "No Finding" → Normal, any label → Abnormal

| Split | Images | Source |
|-------|--------|--------|
| Train | ~78,468 (70%) | Official split |
| Validation | ~11,210 (10%) | Official split |
| Test | ~22,442 (20%) | Official split |

## Models

### Model 1: Custom CNN (From Scratch)

A ResNet-inspired architecture built entirely from scratch:

- **Residual blocks** with skip connections for gradient flow
- **Batch normalization** after each convolutional layer
- **Global average pooling** to reduce spatial dimensions
- **Dual classification heads**: 14-unit sigmoid (multi-label) + 1-unit sigmoid (binary)
- **Loss**: Binary cross-entropy (per-label)
- **Optimizer**: AdamW with cosine annealing learning rate schedule

### Model 2: DenseNet-121 Transfer Learning

Following the CheXNet approach (Rajpurkar et al., 2017):

- **Backbone**: DenseNet-121 pretrained on ImageNet
- **Fine-tuning strategy**: Freeze early layers, unfreeze later dense blocks
- **Custom classifier head**: Adapted for 14-label + binary outputs
- **Loss**: Weighted binary cross-entropy (handling class imbalance)
- **Optimizer**: AdamW with warm-up + cosine annealing

## Evaluation Metrics

- **AUC-ROC** (per-class and macro-averaged) — primary metric
- **F1 Score** (per-class and macro-averaged)
- **Precision & Recall**
- **Confusion matrices** (for binary task)
- **Grad-CAM visualizations** — model interpretability

## Infrastructure

| Component | Platform | Purpose |
|-----------|----------|---------|
| Source code | [GitHub](https://github.com/arudaev/chexvision) | Version control, CI/CD |
| Dataset | [HuggingFace Dataset](https://huggingface.co/datasets/HlexNC/chest-xray-14) | Processed dataset hosting |
| Models | [HF: Scratch](https://huggingface.co/HlexNC/chexvision-scratch), [HF: DenseNet](https://huggingface.co/HlexNC/chexvision-densenet) | Trained model hosting |
| Demo | [HF Space](https://huggingface.co/spaces/HlexNC/chexvision-demo) | Streamlit interactive demo |
| CI/CD | GitHub Actions | Linting, testing, deployment |

## Team

**BIG D(ATA)** — Deep Learning & Big Data, AIN

## References

1. Wang, X. et al. (2017). "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks." CVPR.
2. Rajpurkar, P. et al. (2017). "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning." arXiv:1711.05225.
3. Huang, G. et al. (2017). "Densely Connected Convolutional Networks." CVPR.
4. He, K. et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.

## License

MIT License — see [LICENSE](LICENSE) for details.
