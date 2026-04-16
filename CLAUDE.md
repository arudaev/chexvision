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
src/
├── data/           # Dataset loading, transforms, download scripts
│   ├── dataset.py  # ChestXrayDataset (PyTorch Dataset class)
│   ├── transforms.py # Data augmentation pipelines
│   └── download.py # Download from HuggingFace Hub
├── models/
│   ├── scratch_cnn.py     # Model 1: Custom ResNet-style CNN
│   └── densenet_transfer.py # Model 2: DenseNet-121 fine-tuning
├── training/
│   ├── trainer.py   # Main training loop with logging
│   ├── metrics.py   # AUC-ROC, F1, confusion matrix computation
│   └── evaluate.py  # Model evaluation and comparison script
└── utils/
    └── visualization.py  # Grad-CAM, ROC curves, training plots
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
# Install
pip install -e ".[dev]"

# Download data
python -m src.data.download --output-dir data/

# Train
python -m src.training.trainer --config configs/scratch.yaml
python -m src.training.trainer --config configs/transfer.yaml

# Evaluate
python -m src.training.evaluate --model-dir checkpoints/ --compare

# Lint & test
ruff check src/ tests/
pytest tests/ -v

# Demo
streamlit run app/app.py
```

## Configuration

All hyperparameters live in `configs/*.yaml`. Key parameters:

- `model.type`: "scratch" or "densenet"
- `training.epochs`, `training.batch_size`, `training.lr`
- `training.optimizer`: AdamW config
- `training.scheduler`: Cosine annealing config
- `data.image_size`: 224 (standard for both models)
- `data.augmentation`: RandomHorizontalFlip, RandomRotation, ColorJitter

## HuggingFace Resources

- Dataset: `HlexNC/chest-xray-14`
- Model (scratch): `HlexNC/chexvision-scratch`
- Model (transfer): `HlexNC/chexvision-densenet`
- Demo Space: `HlexNC/chexvision-demo`

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

| Task | Where to Run | Why |
|------|-------------|-----|
| Training | Google Colab (free T4 GPU) | 112k images + deep CNNs need GPU; keeps runs reproducible via shared notebooks |
| Dataset download/processing | Google Colab or HF Space | 45GB dataset, not practical for local storage |
| Large-scale evaluation | Google Colab | GPU-accelerated inference on full test set |
| Streamlit demo | HF Space (`HlexNC/chexvision-demo`) | Always-on public demo, auto-deployed via GitHub Actions |
| Linting, unit tests | GitHub Actions CI | Automated on every push/PR |
| Local dev | Code editing, small tests only | `CHEXVISION_ALLOW_LOCAL=1 pytest tests/ -v` |

The training script (`src/training/trainer.py`) enforces this: it will **exit with an error** if run on a local CPU machine. Override with `CHEXVISION_ALLOW_LOCAL=1` only for unit tests.

### Running on Google Colab

All notebooks in `notebooks/` have Colab badges and setup cells. Open them directly from GitHub:
- `01_eda.ipynb` — Exploratory data analysis
- `02_train_scratch.ipynb` — Model 1 training (custom CNN)
- `03_train_transfer.ipynb` — Model 2 training (DenseNet-121)

### Deploying to HF Space

Push to `main` and the GitHub Action (`.github/workflows/deploy-space.yml`) auto-deploys to the HF Space.

## Important Notes

- The NIH Chest X-ray14 dataset is ~45GB raw. Use HuggingFace streaming mode or the Colab-cached version.
- Always set random seeds for reproducibility.
- The report must justify every architectural decision — keep code comments explaining "why" not "what".
