# CheXVision — Claude Development Guide

## Project Overview

CheXVision is a university deep-learning project for chest X-ray pathology detection using the NIH Chest X-ray14 dataset (112,120 images, 14 disease labels). We implement two PyTorch models (custom ResNet CNN from scratch + DenseNet-121 transfer learning) on two tasks (14-class multi-label classification + binary normal/abnormal). Deadline: **June 23, 2026**. Course: Deep Learning & Big Data, AIN program, "BIG D(ATA)" team.

---

## Repository Layout

```
src/
├── data/
│   ├── dataset.py              # ChestXrayDataset, PATHOLOGY_LABELS, NUM_CLASSES
│   ├── transforms.py           # get_train_transforms() / get_eval_transforms()
│   ├── download.py             # download_dataset() — snapshot_download wrapper
│   └── resize_320_pipeline.py  # Kaggle kernel: raw NIH ZIPs → 320×320 parquet shards
├── models/
│   ├── scratch_cnn.py          # CheXVisionScratch — ResNet-50-depth CNN + SE, dual heads
│   └── densenet_transfer.py    # CheXVisionDenseNet — DenseNet-121 fine-tuning
├── training/
│   ├── trainer.py              # train() — main entry point, config merging, history
│   ├── metrics.py              # compute_multilabel_metrics(), compute_binary_metrics()
│   └── evaluate.py             # post-training evaluation, model comparison
└── utils/
    ├── hub.py                  # load_hf_token(), configure_hf_runtime(),
    │                           # upload_model_artifacts(), render_model_card()
    └── visualization.py        # Grad-CAM, ROC curves, training history plots

scripts/
├── eda.py                      # EDA — streams metadata from HF, saves plots
├── dispatch.py                 # Kaggle kernel dispatch (build bundle, push, status)
├── push_models.py              # Manual HF Hub upload (recovery path)
└── generate_diagram_pngs.py    # Render Mermaid source → PNG via Playwright + Pillow,
                                # then push PNGs to HF Hub model cards

kaggle/
├── train_scratch/
│   ├── kernel-metadata.json    # id, enable_gpu, enable_internet, dataset_sources
│   └── script.py               # Self-contained training script (template with placeholder)
└── train_transfer/
    ├── kernel-metadata.json
    └── script.py

configs/
├── default.yaml                # Base config — all kernels start from this
├── scratch.yaml                # Overrides: model.type=scratch, SE attention, 100 epochs
└── transfer.yaml               # Overrides: model.type=densenet, freeze_epochs, 60 epochs

app/
└── app.py                      # Streamlit demo — hf_hub_download for checkpoints

tests/                          # pytest unit tests (51 tests across 9 modules)
├── test_dataset.py
├── test_metrics.py
├── test_transforms.py
├── test_models.py
├── test_hub.py
├── test_download.py
├── test_dispatch.py
├── test_resize_320_pipeline.py
└── test_app_bootstrap.py
```

---

## Tech Stack

- **Language**: Python 3.10+
- **ML Framework**: PyTorch + torchvision (raw training loops — no Lightning)
- **Data**: HuggingFace `datasets` + `huggingface_hub`, PIL/Pillow
- **Demo**: Streamlit
- **CI**: GitHub Actions — ruff lint, pytest, mypy (on every push/PR)
- **Hosting**: HuggingFace Hub (dataset, models, Spaces)
- **Training compute**: Kaggle GPU kernels (free T4, dispatched via `scripts/dispatch.py`)

---

## Coding Conventions

- **Type hints** throughout; mypy runs in CI (`--ignore-missing-imports --no-strict-optional`)
- **PEP 8** enforced by ruff (`E`, `F`, `I`, `N`, `W`, `UP` rules, line length 120)
- **pathlib.Path** for all file operations — never raw string paths
- **YAML configs** for all hyperparameters — nothing hardcoded in source
- **Reproducibility**: seed `torch`, `numpy`, `random`, and CUDA in every training run
- **Docstrings**: Google style on public classes/functions only
- **No notebooks** — all logic lives in `src/` or `scripts/`
- **No Co-Authored-By** — never add `Co-Authored-By: Claude` trailers to commit messages or PR descriptions

---

## Key Design Decisions

- **Raw PyTorch loops**: every design choice is explicit and defensible in the course report
- **Dual heads on a shared backbone**: one forward pass produces both a 14-class multilabel output and a binary output; trained with combined BCE loss
- **Weighted BCE**: class-level `pos_weight` tensors handle the severe label imbalance in this dataset
- **YAML config inheritance**: configs declare `_defaults_: default.yaml`; `_load_config()` deep-merges before training
- **History JSON**: trainer saves per-epoch metrics to `{model}_history.json` alongside checkpoints for report figures
- **`src.utils.hub`**: centralises all HF token resolution, upload logic, and model card rendering — Kaggle scripts import from here
- **SE attention (scratch model)**: Squeeze-Excitation blocks after each residual stage for channel-wise recalibration — justified in the report as particularly suited to multi-label pathology detection
- **320×320 training resolution**: dataset `arudaev/chest-xray-14-320` — higher resolution than the 224px baseline for better small-lesion detection; batch size reduced to 24 (from 32) to stay within T4 VRAM
- **CLAHE preprocessing** (`data.clahe: true`): Contrast Limited Adaptive Histogram Equalisation applied in LAB colour space before the network sees the image; enhances local contrast for low-contrast findings (Nodule, Infiltration, Pneumonia) without global brightness shifts; implemented as `CLAHETransform` in `src/data/transforms.py` with a lazy `cv2` import
- **Label smoothing** (`training.label_smoothing: 0.1`): regularises against noisy NIH patient-level annotations; positive targets become `1 − ε`, negative targets become `ε / 2`; applied per-batch in `train_one_epoch` before BCE loss, so it is orthogonal to the `pos_weight` class balancing
- **TTA at inference** (`use_tta=True` in `app.py` and `evaluate.py`): averages sigmoid probabilities over 4 views (original, h-flip, rotate ±7°) — reduces prediction variance with no training cost; implemented in `predict_with_tta()` using `torchvision.transforms.functional` (aliased as `transforms_functional` per ruff N812)
- **Ensemble inference**: `predict_ensemble()` in `evaluate.py` and the Ensemble row in `app.py` average TTA probabilities from both models; the two architectures have different inductive biases and fail on different examples, so averaging improves macro AUC

---

## Running Locally

```bash
# Install (editable, with dev extras)
pip install -e ".[dev]"

# Lint
ruff check src/ tests/ scripts/ app/

# Tests — CHEXVISION_ALLOW_LOCAL=1 bypasses the cloud-only guard
CHEXVISION_ALLOW_LOCAL=1 pytest tests/ -v

# Type check
mypy src/ --ignore-missing-imports

# EDA (lightweight — streams metadata only, no full download)
python scripts/eda.py --num-samples 5000 --output-dir results/eda

# Regenerate model card diagram PNGs (requires Playwright + Chromium)
python scripts/generate_diagram_pngs.py

# Streamlit demo (loads models from HF Hub)
streamlit run app/app.py
```

**Never run `src.training.trainer` directly on a local machine** — the cloud guard will raise unless `CHEXVISION_ALLOW_LOCAL=1` is set.

---

## Kaggle Training Dispatch

### How it works

`scripts/dispatch.py` bundles `src/` and `configs/` into a base64 zip payload, injects it into the kernel script template (replacing the `__CHEXVISION_PROJECT_BUNDLE_B64__` sentinel), writes a `kernel-metadata.json` that forces `enable_gpu=true` and `enable_internet=true`, and calls `kaggle kernels push`.

On Kaggle, the script unpacks the bundle, downloads the dataset via `src.data.download`, trains, and uploads artifacts to HF Hub using `src.utils.hub.upload_model_artifacts`.

### Credentials

`KAGGLE_API_TOKEN` in `.env` must be a `KGAT_...` token (Kaggle CLI >= 1.8.0 / kaggle 2.x format). The `dispatch.py` CLI reads `.env` via `python-dotenv`.

HF token injection into the Kaggle runtime:

1. **Preferred (automated)**: private dataset `hlexnc/chexvision-secrets` containing `hf_token.txt`. Both `kernel-metadata.json` files declare `"dataset_sources": ["hlexnc/chexvision-secrets"]`; the token file is mounted at `/kaggle/input/chexvision-secrets/hf_token.txt` in every API-pushed kernel. `load_hf_token()` checks this path automatically.
2. **Fallback**: `UserSecretsClient().get_secret("HF_TOKEN")` — works only in interactive Kaggle sessions; not reliable for automated pushes.

### Commands

```bash
python scripts/dispatch.py kaggle push scratch    # build bundle, push, trigger run
python scripts/dispatch.py kaggle push transfer

python scripts/dispatch.py kaggle status scratch  # check run status
python scripts/dispatch.py kaggle output scratch  # download output files
```

---

## HuggingFace Resources

| Resource | ID |
|----------|----|
| Dataset (320×320) | `arudaev/chest-xray-14-320` |
| Model — scratch CNN | `arudaev/chexvision-scratch` |
| Model — DenseNet | `arudaev/chexvision-densenet` |
| Demo Space (Streamlit) | `arudaev/chexvision-demo` |
| Data pipeline Space (one-time) | `arudaev/chexvision-data-pipeline` |

The dataset is pinned to a specific commit hash stored in `src/utils/hub.py` (`HF_DATASET_REVISION`). Update this constant when a new dataset version is intentionally published.

---

## API Access

Tokens live in `.env` at the project root (gitignored). Load in Python with:

```python
from dotenv import load_dotenv; load_dotenv()
```

| Token | Env var | Scope |
|-------|---------|-------|
| HuggingFace | `HF_TOKEN` | Read + write to arudaev repos |
| Kaggle | `KAGGLE_API_TOKEN` | Push kernels, read status/output |
| GitHub | `GITHUB_TOKEN` | CI status, workflow triggers, secrets management |

GitHub repo: `arudaev/chexvision`
HF owner: `arudaev`

---

## Cloud-Only Compute Policy

| Task | Where | How |
|------|-------|-----|
| Model training | Kaggle GPU kernels | `dispatch.py kaggle push` |
| Large-scale evaluation | Kaggle GPU | Same or separate kernel |
| Streamlit demo | HF Space `chexvision-demo` | Auto-deployed by GitHub Actions on push to `main` |
| Linting / unit tests | GitHub Actions CI | Every push / PR |
| EDA | Local (lightweight) | `scripts/eda.py` — metadata only |
| Code editing, small tests | Local | `CHEXVISION_ALLOW_LOCAL=1 pytest` |

No notebooks. No local GPU training. No Colab.

---

## CI/CD

- **`ci.yml`**: runs on every push/PR to `main` — three parallel jobs: Lint (ruff), Test (pytest), Type Check (mypy)
- **`deploy-space.yml`**: runs on push to `main` — pushes the full repo to `arudaev/chexvision-demo` HF Space

The HF Space uses `requirements.txt` (not the Dockerfile) on Streamlit Community Cloud. The Dockerfile is kept for completeness and potential future Docker-based HF Spaces deployment.

---

## Configuration Reference

All hyperparameters are in `configs/*.yaml`. The `_defaults_` key triggers deep merge with the base config.

Key fields:
- `model.type`: `"scratch"` or `"densenet"`
- `model.architecture.use_se`: `true` (scratch only — SE channel attention)
- `model.architecture.block_config`: `[3, 4, 6, 3]` (scratch — ResNet-50 depth)
- `training.epochs`: 100 (scratch) / 60 (transfer)
- `training.batch_size`: `24` (320×320 images; reduced from 32 to fit T4 VRAM)
- `training.optimizer`: AdamW settings
- `training.scheduler`: cosine annealing settings
- `training.freeze_epochs`: (DenseNet only) epochs to train with frozen backbone
- `training.grad_accum_steps`: `4` — effective batch = 24 × 4 = 96
- `training.label_smoothing`: `0.1` — positive targets → 0.9, negative targets → 0.05
- `data.image_size`: `320`
- `data.clahe`: `true` — CLAHE contrast enhancement in LAB colour space (clip 2.0, tile 8×8)
- `data.dataset_name`: `"arudaev/chest-xray-14-320"`
- `data.augmentation`: RandomHorizontalFlip, RandomRotation, ColorJitter, RandomErasing
- `logging.checkpoint_dir`: where to save `.pth` files

---

## Important Notes

- The raw NIH dataset is ~45 GB. `arudaev/chest-xray-14-320` stores pre-resized 320×320 parquet shards (~7.97 GB, 36 shards).
- Always pin `HF_DATASET_REVISION` in `src/utils/hub.py` to a specific commit hash for reproducible training runs.
- The report must justify every architectural decision — keep code comments explaining *why*, not *what*.
- Grad-CAM and ROC curve figures for the report are generated by `src/utils/visualization.py`.
- Model card PNG diagrams are generated by `scripts/generate_diagram_pngs.py` (Playwright + Mermaid.js CDN) and pushed directly to the HF Hub model repos. Re-run the script if Mermaid source changes in `src/utils/hub.py`.
