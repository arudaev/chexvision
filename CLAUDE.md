# CheXVision — Claude Development Guide

## Project Overview

CheXVision is a deep learning project for chest X-ray pathology detection using the NIH Chest X-ray14 dataset (112,120 images, 14 disease labels). We implement two models (custom CNN from scratch + DenseNet-121 transfer learning) on two tasks (multi-label classification + binary classification).

## Tech Stack

- **Language**: Python 3.10+
- **Framework**: PyTorch (+ torchvision)
- **Data**: HuggingFace `datasets` library, PIL/Pillow
- **Training**: PyTorch training loop (no high-level wrappers like Lightning — we want full control for the report)
- **Visualization**: matplotlib, seaborn, Grad-CAM
- **Demo**: Streamlit
- **CI**: GitHub Actions (ruff linting, pytest)
- **Hosting**: HuggingFace Hub (datasets, models, Spaces)

## Directory Structure

```
src/                          # Core library
├── data/
│   ├── dataset.py            # ChestXrayDataset (PyTorch Dataset class)
│   ├── transforms.py         # Data augmentation pipelines
│   └── download.py           # Download from HuggingFace Hub
├── models/
│   ├── scratch_cnn.py        # Model 1: Custom ResNet-style CNN
│   └── densenet_transfer.py  # Model 2: DenseNet-121 fine-tuning
├── training/
│   ├── trainer.py            # Training loop (cloud-only guard enforced)
│   ├── metrics.py            # AUC-ROC, F1, confusion matrix
│   └── evaluate.py           # Model evaluation and comparison
└── utils/
    └── visualization.py      # Grad-CAM, ROC curves, training plots

scripts/                      # CLI tools (run locally)
├── eda.py                    # EDA — streams from HF, generates plots
├── dispatch.py               # Dispatch training to Kaggle GPU
└── push_models.py            # Upload checkpoints to HF Hub

kaggle/                       # Kaggle kernel configs (run on cloud GPU)
├── train_scratch/            # Custom CNN training kernel
└── train_transfer/           # DenseNet-121 training kernel
```

## Coding Conventions

- Use **type hints** throughout
- Follow **PEP 8** — enforced by `ruff`
- Use **pathlib.Path** for all file paths
- Config via **YAML files** in `configs/` — no hardcoded hyperparameters in source
- All training runs must be **reproducible** — seed everything (torch, numpy, random, CUDA)
- Use `torch.no_grad()` contexts for evaluation
- Docstrings on public classes/functions only (Google style)

## Key Design Decisions

- **No PyTorch Lightning**: We use raw PyTorch training loops so every design decision is explicit and can be justified in the report
- **Dual heads**: Both models share a backbone but branch into two classification heads (multi-label 14-class sigmoid + binary sigmoid)
- **Weighted BCE loss**: Handles severe class imbalance in chest X-ray labels
- **Grad-CAM**: Used for model interpretability — shows which regions the model focuses on
- **HuggingFace Hub integration**: Dataset and models are pushed to HF for reproducibility and demo access

## Running the Project

```bash
# Install (local)
pip install -e ".[dev]"

# EDA — runs locally, streams metadata from HF (lightweight)
python scripts/eda.py --num-samples 5000 --output-dir results/eda

# Lint & test — runs locally
ruff check src/ tests/ scripts/
CHEXVISION_ALLOW_LOCAL=1 pytest tests/ -v

# Dispatch training to Kaggle GPU (requires `pip install kaggle` + API key)
python scripts/dispatch.py kaggle push scratch
python scripts/dispatch.py kaggle push transfer
python scripts/dispatch.py kaggle status scratch

# Upload trained models to HF Hub
python scripts/push_models.py --checkpoint checkpoints/CheXVision-ResNet_best.pth

# Demo — runs locally or on HF Space
streamlit run app/app.py
```

**Never run `src.training.trainer` directly on a local machine.** The cloud guard will block it.

## Configuration

All hyperparameters live in `configs/*.yaml`. Key parameters:

- `model.type`: "scratch" or "densenet"
- `training.epochs`, `training.batch_size`, `training.lr`
- `training.optimizer`: AdamW config
- `training.scheduler`: Cosine annealing config
- `data.image_size`: 224 (standard for both models)
- `data.augmentation`: RandomHorizontalFlip, RandomRotation, ColorJitter

## HuggingFace Resources

- Dataset: `HlexNC/chest-xray-14` (preprocessed 224x224 parquet shards)
- Model (scratch): `HlexNC/chexvision-scratch`
- Model (transfer): `HlexNC/chexvision-densenet`
- Demo Space: `HlexNC/chexvision-demo`
- Data Pipeline Space: `HlexNC/chexvision-data-pipeline` (one-time repackaging job)

## API Access

Tokens are stored in `.env` at the project root. Load them with:

```python
from dotenv import load_dotenv; load_dotenv()
```

Or read them directly from `.env` for shell/curl usage.

### GitHub API

- **Token env var**: `GITHUB_TOKEN`
- **Repo**: `arudaev/chexvision`
- Use for: creating issues, managing PRs, triggering workflows, setting secrets, reading CI status, managing releases
- Example: `curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/repos/arudaev/chexvision/actions/runs`

### HuggingFace API

- **Token env var**: `HF_TOKEN`
- **Owner**: `HlexNC`
- Use for: uploading datasets, pushing models, managing the Space, creating/managing repos, querying dataset/model info
- Repos:
  - Dataset: `HlexNC/chest-xray-14`
  - Model (scratch): `HlexNC/chexvision-scratch`
  - Model (transfer): `HlexNC/chexvision-densenet`
  - Space (Streamlit demo): `HlexNC/chexvision-demo`
- Example: `curl -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/models/HlexNC/chexvision-scratch`

You have full read/write access to both platforms. Use the APIs freely for any automation — CI/CD, dataset uploads, model pushes, repo management, etc.

## Cloud-Only Compute Policy

**All heavy compute (training, dataset processing, large-scale evaluation) MUST run in the cloud. Never on local dev machines.**

| Task | Where to Run | How |
|------|-------------|-----|
| Dataset repackaging | HF Space (`chexvision-data-pipeline`) | One-time Gradio job on HF CPU |
| Training | Kaggle GPU kernels | `python scripts/dispatch.py kaggle push scratch` |
| Large-scale evaluation | Kaggle GPU | Same kernel, or separate eval kernel |
| Streamlit demo | HF Space (`chexvision-demo`) | Auto-deployed via GitHub Actions |
| Linting, unit tests | GitHub Actions CI | Automated on every push/PR |
| EDA | Local (lightweight) | `python scripts/eda.py` — streams metadata only |
| Local dev | Code editing, small tests only | `CHEXVISION_ALLOW_LOCAL=1 pytest tests/ -v` |

The training script (`src/training/trainer.py`) enforces this: it will **exit with an error** if run on a local CPU machine without GPU. Override with `CHEXVISION_ALLOW_LOCAL=1` for unit tests only.

### Training on Kaggle

Kernel configs live in `kaggle/`. Each has a `kernel-metadata.json` (GPU enabled, internet enabled) and a `script.py`. The script clones the repo, downloads data, trains, and uploads the checkpoint to HF Hub.

```bash
pip install kaggle  # one-time
# Set up ~/.kaggle/kaggle.json with your API key

python scripts/dispatch.py kaggle push scratch    # dispatches to Kaggle GPU
python scripts/dispatch.py kaggle status scratch   # check progress
python scripts/dispatch.py kaggle output scratch   # download output
```

### Deploying to HF Space

Push to `main` and the GitHub Action (`.github/workflows/deploy-space.yml`) auto-deploys to the HF Space.

## Important Notes

- The NIH Chest X-ray14 dataset is ~45GB raw. Our HF dataset repo stores preprocessed 224x224 parquet shards (~5GB).
- The data pipeline Space (`chexvision-data-pipeline`) handles repackaging — run it once, then the dataset is ready.
- Always set random seeds for reproducibility.
- The report must justify every architectural decision — keep code comments explaining "why" not "what".
