"""Download and prepare the NIH Chest X-ray14 dataset from Hugging Face."""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.utils.hub import HF_DATASET_REPO, HF_DATASET_REVISION, configure_hf_runtime

logger = logging.getLogger(__name__)

DATASET_ALLOW_PATTERNS = [
    "README.md",
    "load_dataset.py",
    "*.json",
    "**/*.json",
    "data/*.parquet",
]
SNAPSHOT_DIR_ENV = "CHEXVISION_HF_SNAPSHOT_DIR"

# The 320px shards are large (hundreds of MB) and Xet-backed; the HTTP bridge
# occasionally closes a stream mid-file ("peer closed connection without sending
# complete message body"). snapshot_download keeps a partial .incomplete file and
# resumes via HTTP Range, so retrying makes forward progress instead of failing
# the whole training run on a single transient drop.
SNAPSHOT_MAX_ATTEMPTS = 8
SNAPSHOT_RETRY_BACKOFF_SECONDS = 5

# Substrings of exception type names that indicate a transient transport failure
# worth retrying (vs. auth/not-found errors, which are re-raised immediately).
_RETRYABLE_ERROR_NAMES = (
    "RemoteProtocolError",
    "ProtocolError",
    "ChunkedEncodingError",
    "IncompleteRead",
    "ConnectionError",
    "ConnectionResetError",
    "ReadTimeout",
    "ConnectTimeout",
    "Timeout",
    "ReadError",
    "WriteError",
    "PoolTimeout",
)


def _is_retryable_download_error(exc: BaseException) -> bool:
    """Return True when *exc* (or any error in its chain) is a transient drop."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if any(name in type(current).__name__ for name in _RETRYABLE_ERROR_NAMES):
            return True
        # "peer closed connection" surfaces from several layers with varying types.
        if "peer closed connection" in str(current).lower():
            return True
        current = current.__cause__ or current.__context__
    return False


def _dataset_features():
    """Build the dataset feature schema after the HF runtime is configured."""
    from datasets import Features, Image, Value

    return Features(
        {
            "image": Image(),
            "labels": Value("string"),
            # The live parquet shards include this column even though older card
            # metadata omitted it, so we declare it explicitly for stable loads.
            "filename": Value("string"),
        }
    )


def _build_local_parquet_files(snapshot_dir: Path) -> dict[str, str]:
    """Return the local parquet shard globs for each split."""
    data_dir = Path(snapshot_dir) / "data"
    return {
        "train": str(data_dir / "train-*.parquet"),
        "validation": str(data_dir / "validation-*.parquet"),
        "test": str(data_dir / "test-*.parquet"),
    }


def _snapshot_dataset_repo(
    repo_id: str,
    revision: str,
    snapshot_dir: Path,
    token: str | None,
) -> Path:
    """Download a pinned dataset snapshot from the HF Hub."""
    from huggingface_hub import snapshot_download

    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    last_exc: BaseException | None = None
    for attempt in range(1, SNAPSHOT_MAX_ATTEMPTS + 1):
        try:
            # snapshot_download is idempotent: completed shards are skipped and
            # partial ones resume from their .incomplete file, so each retry
            # continues the download rather than restarting from zero.
            return Path(
                snapshot_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    revision=revision,
                    local_dir=str(snapshot_dir),
                    allow_patterns=DATASET_ALLOW_PATTERNS,
                    token=token,
                )
            )
        except Exception as exc:  # noqa: BLE001 - re-raised below when not transient
            last_exc = exc
            if attempt >= SNAPSHOT_MAX_ATTEMPTS or not _is_retryable_download_error(exc):
                raise
            wait_seconds = SNAPSHOT_RETRY_BACKOFF_SECONDS * attempt
            logger.warning(
                "Dataset download interrupted (attempt %d/%d): %s. "
                "Resuming in %ds ...",
                attempt,
                SNAPSHOT_MAX_ATTEMPTS,
                exc,
                wait_seconds,
            )
            time.sleep(wait_seconds)

    # Unreachable: the loop either returns or raises, but keep mypy/readers happy.
    raise RuntimeError("Dataset snapshot download failed") from last_exc


def _load_streaming_dataset(snapshot_dir: Path):
    """Load the parquet shards from a local HF snapshot in streaming mode."""
    from datasets import load_dataset

    return load_dataset(
        "parquet",
        data_files=_build_local_parquet_files(snapshot_dir),
        streaming=True,
        features=_dataset_features(),
    )


def _default_snapshot_dir(output_dir: Path) -> Path:
    """Choose a stable local snapshot directory for the current environment."""
    configured_dir = os.environ.get(SNAPSHOT_DIR_ENV, "").strip()
    if configured_dir:
        return Path(configured_dir)

    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return Path("/kaggle/working/hf_datasets/chest_xray_14")

    return Path(output_dir) / "_hf_snapshot"


def download_dataset(
    output_dir: Path,
    repo_id: str = HF_DATASET_REPO,
    revision: str = HF_DATASET_REVISION,
    snapshot_dir: Path | None = None,
) -> None:
    """Download the dataset snapshot and materialize train/val/test PNG files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    token = configure_hf_runtime()
    local_snapshot_dir = Path(snapshot_dir) if snapshot_dir else _default_snapshot_dir(output_dir)

    if token:
        logger.info(
            "Downloading dataset snapshot from %s at revision %s with authenticated HF access ...",
            repo_id,
            revision,
        )
    else:
        logger.info("Downloading dataset snapshot from %s at revision %s ...", repo_id, revision)

    dataset_dir = _snapshot_dataset_repo(repo_id, revision, local_snapshot_dir, token)
    logger.info("Loading parquet shards from %s", dataset_dir)
    dataset = _load_streaming_dataset(dataset_dir)

    records = []
    for split_name in dataset:
        split_data = dataset[split_name]
        logger.info("Processing split: %s", split_name)

        for idx, sample in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
            image = sample["image"]
            filename = f"{split_name}_{idx:06d}.png"
            image_path = images_dir / filename
            if not image_path.exists():
                image.save(image_path)

            split_mapped = "val" if split_name in ("validation", "valid") else split_name
            records.append(
                {
                    "image_path": f"images/{filename}",
                    "labels": sample.get("labels", sample.get("label", "No Finding")),
                    "split": split_mapped,
                }
            )

    df = pd.DataFrame(records)
    csv_path = output_dir / "labels.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved labels to %s (%d total samples)", csv_path, len(df))
    logger.info("Dataset ready at %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NIH Chest X-ray14 dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="Output directory")
    parser.add_argument("--repo-id", type=str, default=HF_DATASET_REPO, help="Hugging Face dataset repo")
    parser.add_argument(
        "--revision",
        type=str,
        default=HF_DATASET_REVISION,
        help="Pinned Hugging Face dataset revision to download.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Optional local directory for the downloaded dataset snapshot.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    download_dataset(args.output_dir, args.repo_id, args.revision, args.snapshot_dir)


if __name__ == "__main__":
    main()
