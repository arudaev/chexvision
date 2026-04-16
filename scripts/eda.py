#!/usr/bin/env python3
"""Exploratory Data Analysis for NIH Chest X-ray14 dataset.

Streams metadata from HuggingFace (no heavy downloads) and generates
summary statistics and plots.

Usage:
    python scripts/eda.py
    python scripts/eda.py --num-samples 10000 --output-dir results/eda
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset

# Standalone copy — no import from src so this script is self-contained.
PATHOLOGY_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

DATASET_REPO = "alkzar90/NIH-Chest-X-ray-dataset"


def stream_samples(num_samples: int) -> list[dict]:
    """Stream *num_samples* rows from the HF dataset (metadata + images)."""
    print(f"Streaming {num_samples:,} samples from {DATASET_REPO} ...")
    ds = load_dataset(DATASET_REPO, streaming=True)
    samples = list(ds["train"].take(num_samples))
    print(f"Loaded {len(samples):,} samples.")
    return samples


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_label_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Horizontal bar chart of label counts."""
    label_counts: Counter[str] = Counter()
    for labels_str in df["labels"]:
        if pd.isna(labels_str) or labels_str == "No Finding":
            label_counts["No Finding"] += 1
        else:
            for label in str(labels_str).split("|"):
                label_counts[label.strip()] += 1

    sorted_items = sorted(label_counts.items(), key=lambda x: -x[1])
    labels, counts = zip(*sorted_items)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(labels, counts, color="steelblue")
    ax.set_xlabel("Count")
    ax.set_title("Label Distribution — NIH Chest X-ray14 (EDA sample)")
    plt.tight_layout()
    fig.savefig(output_dir / "label_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Saved label_distribution.png")


def plot_binary_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Pie chart of Normal vs Abnormal."""
    is_abnormal = df["labels"].apply(
        lambda x: 0 if pd.isna(x) or x == "No Finding" else 1
    )
    normal_count = int((is_abnormal == 0).sum())
    abnormal_count = int((is_abnormal == 1).sum())

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(
        [normal_count, abnormal_count],
        labels=["Normal", "Abnormal"],
        autopct="%1.1f%%",
        colors=["#66b3ff", "#ff6666"],
        startangle=90,
    )
    ax.set_title("Normal vs Abnormal Distribution")
    plt.tight_layout()
    fig.savefig(output_dir / "binary_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Saved binary_distribution.png")


def plot_cooccurrence_matrix(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap of pathology co-occurrences."""
    n = len(PATHOLOGY_LABELS)
    cooccurrence = np.zeros((n, n), dtype=int)

    for labels_str in df["labels"]:
        if pd.isna(labels_str) or labels_str == "No Finding":
            continue
        present = [
            PATHOLOGY_LABELS.index(l.strip())
            for l in str(labels_str).split("|")
            if l.strip() in PATHOLOGY_LABELS
        ]
        for i in present:
            for j in present:
                cooccurrence[i][j] += 1

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cooccurrence,
        xticklabels=PATHOLOGY_LABELS,
        yticklabels=PATHOLOGY_LABELS,
        cmap="YlOrRd",
        annot=True,
        fmt="d",
        ax=ax,
    )
    ax.set_title("Pathology Co-occurrence Matrix")
    plt.tight_layout()
    fig.savefig(output_dir / "cooccurrence_matrix.png", dpi=150)
    plt.close(fig)
    print(f"  Saved cooccurrence_matrix.png")


def plot_sample_images(samples: list[dict], output_dir: Path) -> None:
    """2x5 grid of sample X-ray images (streamed from HF)."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for ax, sample in zip(axes.flat, samples[:10]):
        img = sample["image"]
        ax.imshow(img, cmap="gray")
        label_text = str(sample.get("labels", sample.get("label", "?")))[:30]
        ax.set_title(label_text, fontsize=8)
        ax.axis("off")

    fig.suptitle("Sample Chest X-ray Images (streamed from HuggingFace)", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "sample_images.png", dpi=150)
    plt.close(fig)
    print(f"  Saved sample_images.png")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    """Print key statistics to stdout."""
    total = len(df)
    is_abnormal = df["labels"].apply(
        lambda x: 0 if pd.isna(x) or x == "No Finding" else 1
    )
    normal_count = int((is_abnormal == 0).sum())
    abnormal_count = int((is_abnormal == 1).sum())

    label_counts: Counter[str] = Counter()
    multi_label_count = 0
    for labels_str in df["labels"]:
        if pd.isna(labels_str) or labels_str == "No Finding":
            continue
        individual = [l.strip() for l in str(labels_str).split("|")]
        if len(individual) > 1:
            multi_label_count += 1
        for label in individual:
            label_counts[label] += 1

    print("\n" + "=" * 60)
    print("  CheXVision — EDA Summary Statistics")
    print("=" * 60)
    print(f"  Total samples analysed : {total:,}")
    print(f"  Normal (No Finding)    : {normal_count:,} ({normal_count / total:.1%})")
    print(f"  Abnormal               : {abnormal_count:,} ({abnormal_count / total:.1%})")
    print(f"  Multi-label samples    : {multi_label_count:,} ({multi_label_count / total:.1%})")
    print("-" * 60)
    print("  Per-pathology counts:")
    for label in PATHOLOGY_LABELS:
        cnt = label_counts.get(label, 0)
        print(f"    {label:<22s} {cnt:>6,}  ({cnt / total:.2%})")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lightweight EDA for NIH Chest X-ray14 (streams from HuggingFace)."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5000,
        help="Number of samples to stream (default: 5000).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/eda"),
        help="Directory for output plots (default: results/eda).",
    )
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Stream samples
    samples = stream_samples(args.num_samples)

    # Build a lightweight DataFrame (labels only — no images)
    records = []
    for s in samples:
        labels = s.get("labels", s.get("label", "No Finding"))
        records.append({"labels": labels})
    df = pd.DataFrame(records)

    # Generate plots
    print("Generating plots ...")
    plot_label_distribution(df, args.output_dir)
    plot_binary_distribution(df, args.output_dir)
    plot_cooccurrence_matrix(df, args.output_dir)
    plot_sample_images(samples, args.output_dir)

    # Print summary
    print_summary(df)

    print(f"All plots saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
