"""Download and prepare the NIH Chest X-ray14 dataset from HuggingFace Hub."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

DATASET_REPO = "HlexNC/chest-xray-14"

# Fallback: original NIH dataset on HuggingFace
FALLBACK_REPO = "alkzar90/NIH-Chest-X-ray-dataset"


def download_dataset(output_dir: Path, repo_id: str = DATASET_REPO) -> None:
    """Download chest X-ray dataset and prepare train/val/test splits.

    Args:
        output_dir: Directory to save images and labels CSV.
        repo_id: HuggingFace dataset repository ID.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    logger.info("Loading dataset from %s ...", repo_id)

    try:
        dataset = load_dataset(repo_id)
    except Exception:
        logger.warning("Could not load %s, falling back to %s", repo_id, FALLBACK_REPO)
        dataset = load_dataset(FALLBACK_REPO)

    # Process each split
    records = []
    for split_name in dataset:
        split_data = dataset[split_name]
        logger.info("Processing split: %s (%d samples)", split_name, len(split_data))

        for idx, sample in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
            # Save image
            image = sample["image"]
            filename = f"{split_name}_{idx:06d}.png"
            image_path = images_dir / filename
            if not image_path.exists():
                image.save(image_path)

            # Map split names
            split_mapped = split_name
            if split_name in ("validation", "valid"):
                split_mapped = "val"

            records.append({
                "image_path": f"images/{filename}",
                "labels": sample.get("labels", sample.get("label", "No Finding")),
                "split": split_mapped,
            })

    # Save labels CSV
    df = pd.DataFrame(records)
    csv_path = output_dir / "labels.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved labels to %s (%d total samples)", csv_path, len(df))
    logger.info("Dataset ready at %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NIH Chest X-ray14 dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="Output directory")
    parser.add_argument("--repo-id", type=str, default=DATASET_REPO, help="HuggingFace dataset repo")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    download_dataset(args.output_dir, args.repo_id)


if __name__ == "__main__":
    main()
