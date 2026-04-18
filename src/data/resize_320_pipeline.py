"""Build the 320x320 CheXVision dataset from the raw NIH source repo."""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.hub import configure_hf_runtime

SOURCE_REPO = "alkzar90/NIH-Chest-X-ray-dataset"
TARGET_REPO = "HlexNC/chest-xray-14-320"
NUM_ZIPS = 12
TARGET_SIZE = (320, 320)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
EXPECTED_SPLIT_COUNTS = {
    "train": 77_967,
    "validation": 8_557,
    "test": 25_596,
}
MAX_UNREADABLE_SAMPLES = 25
MAX_UNKNOWN_SAMPLES = 20


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration for the resize-and-publish pipeline."""

    source_repo: str = SOURCE_REPO
    target_repo: str = TARGET_REPO
    num_zips: int = NUM_ZIPS
    target_size: tuple[int, int] = TARGET_SIZE
    max_zips: int | None = None
    max_images_per_zip: int | None = None
    skip_upload: bool = False
    is_kaggle: bool = False
    work_dir: Path = Path(".codex_tmp/resize_320")


@dataclass
class PipelineStats:
    """Accumulated counters and diagnostics for the run."""

    split_counts: dict[str, int] = field(
        default_factory=lambda: {"train": 0, "validation": 0, "test": 0}
    )
    total_images: int = 0
    unreadable_total: int = 0
    unknown_total: int = 0
    unreadable_samples: list[tuple[str, str, str]] = field(default_factory=list)
    unknown_samples: list[str] = field(default_factory=list)
    parquet_bytes: int = 0
    written_parquets: list[str] = field(default_factory=list)


def _parse_bool_env(name: str) -> bool:
    """Parse a boolean environment variable with common truthy forms."""
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _parse_int_env(name: str) -> int | None:
    """Parse a positive integer environment variable when present."""
    value = os.environ.get(name, "").strip()
    if not value:
        return None

    parsed = int(value)
    if parsed <= 0:
        raise RuntimeError(f"{name} must be a positive integer, got {value!r}.")
    return parsed


def _default_work_dir(is_kaggle: bool) -> Path:
    """Choose a stable work directory for Kaggle runs and local smoke tests."""
    if is_kaggle:
        return Path("/kaggle/working/chexvision_resize_320")
    return Path(".codex_tmp/resize_320")


def build_config() -> PipelineConfig:
    """Construct runtime configuration from environment variables."""
    is_kaggle = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
    max_zips = _parse_int_env("CHEXVISION_MAX_ZIPS")
    max_images_per_zip = _parse_int_env("CHEXVISION_MAX_IMAGES_PER_ZIP")
    skip_upload = _parse_bool_env("CHEXVISION_SKIP_UPLOAD")

    if (max_zips or max_images_per_zip) and not skip_upload:
        raise RuntimeError(
            "Debug limits require CHEXVISION_SKIP_UPLOAD=1 to avoid publishing a "
            "partial dataset."
        )

    return PipelineConfig(
        max_zips=max_zips,
        max_images_per_zip=max_images_per_zip,
        skip_upload=skip_upload,
        is_kaggle=is_kaggle,
        work_dir=_default_work_dir(is_kaggle),
    )


def dataset_features():
    """Return the feature schema for the uploaded data-only dataset."""
    from datasets import Features, Image, Value

    return Features(
        {
            "image": Image(),
            "labels": Value("string"),
            "filename": Value("string"),
        }
    )


def stable_bucket(filename: str) -> int:
    """Deterministically bucket a filename for the train/validation split."""
    digest = hashlib.sha256(filename.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 10


def split_for(filename: str, train_val_files: set[str], test_files: set[str]) -> str:
    """Map a filename to train, validation, test, or unknown."""
    if filename in test_files:
        return "test"
    if filename in train_val_files:
        return "train" if stable_bucket(filename) < 9 else "validation"
    return "unknown"


def normalize_label(raw_label: object) -> str:
    """Convert label values into the canonical pipe-delimited string form."""
    if not isinstance(raw_label, str):
        return "No Finding"

    label = raw_label.strip()
    return label or "No Finding"


def verify_split_contract(train_val_files: set[str], test_files: set[str]) -> dict[str, int]:
    """Validate the split contract used by the live 224 dataset."""
    counts = {"train": 0, "validation": 0, "test": len(test_files)}
    for filename in train_val_files:
        counts["train" if stable_bucket(filename) < 9 else "validation"] += 1

    if counts != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(
            "Source split contract changed unexpectedly. "
            f"Expected {EXPECTED_SPLIT_COUNTS}, got {counts}."
        )
    return counts


def prepare_local_work_dir(config: PipelineConfig) -> None:
    """Reset local staging paths before a new run or smoke test."""
    if config.work_dir.exists():
        shutil.rmtree(config.work_dir)
    (config.work_dir / "data").mkdir(parents=True, exist_ok=True)


def download_source_metadata(token: str):
    """Fetch source labels and split manifests from the HF Hub."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    csv_path = hf_hub_download(
        repo_id=SOURCE_REPO,
        filename="data/Data_Entry_2017_v2020.csv",
        repo_type="dataset",
        token=token,
    )
    labels_df = pd.read_csv(csv_path)
    label_map = {
        row["Image Index"]: normalize_label(row["Finding Labels"])
        for _, row in labels_df.iterrows()
    }

    train_val_path = hf_hub_download(
        repo_id=SOURCE_REPO,
        filename="data/train_val_list.txt",
        repo_type="dataset",
        token=token,
    )
    with open(train_val_path, encoding="utf-8") as handle:
        train_val_files = {line.strip() for line in handle if line.strip()}

    test_path = hf_hub_download(
        repo_id=SOURCE_REPO,
        filename="data/test_list.txt",
        repo_type="dataset",
        token=token,
    )
    with open(test_path, encoding="utf-8") as handle:
        test_files = {line.strip() for line in handle if line.strip()}

    return label_map, train_val_files, test_files


def ensure_clean_target_repo(api, config: PipelineConfig) -> None:
    """Create the target repo if needed and remove stale data-only artifacts."""
    from huggingface_hub import CommitOperationDelete

    api.create_repo(
        repo_id=config.target_repo,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )

    existing_files = api.list_repo_files(
        repo_id=config.target_repo,
        repo_type="dataset",
    )
    delete_targets = [
        path
        for path in existing_files
        if path == "README.md" or path == "load_dataset.py" or path.startswith("data/")
    ]
    if not delete_targets:
        print(f"[resize] Target repo {config.target_repo} is already clean.")
        return

    print(
        f"[resize] Removing {len(delete_targets)} stale file(s) from "
        f"{config.target_repo} ..."
    )
    operations = [CommitOperationDelete(path_in_repo=path) for path in delete_targets]
    api.create_commit(
        repo_id=config.target_repo,
        repo_type="dataset",
        operations=operations,
        commit_message="Reset data-only parquet artifacts before rebuild",
    )


def is_supported_member(info: zipfile.ZipInfo) -> bool:
    """Return True when a ZIP member is an actual source image."""
    if info.is_dir():
        return False

    member_name = info.filename.replace("\\", "/")
    if member_name.startswith("__MACOSX/"):
        return False

    basename = Path(member_name).name
    if not basename or basename.startswith("._"):
        return False

    return basename.lower().endswith(IMAGE_EXTENSIONS)


def resize_image_payload(data: bytes, target_size: tuple[int, int]) -> dict[str, bytes | None]:
    """Decode, convert, resize, and re-encode one image as PNG bytes."""
    from PIL import Image as PILImage

    resampling = getattr(PILImage, "Resampling", PILImage)
    with PILImage.open(io.BytesIO(data)) as raw_image:
        image = raw_image.convert("RGB")
    image = image.resize(target_size, resampling.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return {"bytes": buffer.getvalue(), "path": None}


def write_parquet_shards(
    buckets: dict[str, dict[str, list]],
    output_dir: Path,
    shard_index: int,
    shard_count: int,
    stats: PipelineStats,
) -> None:
    """Write train/validation/test Parquet shards for one source ZIP."""
    from datasets import Dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    features = dataset_features()

    for split_name, data_dict in buckets.items():
        if not data_dict["filename"]:
            continue

        parquet_name = f"{split_name}-{shard_index:05d}-of-{shard_count:05d}.parquet"
        parquet_path = output_dir / parquet_name
        dataset = Dataset.from_dict(data_dict, features=features)
        dataset.to_parquet(str(parquet_path))

        stats.parquet_bytes += parquet_path.stat().st_size
        stats.written_parquets.append(parquet_name)


def upload_parquet_shards(api, config: PipelineConfig, parquet_dir: Path, zip_name: str) -> None:
    """Upload the generated Parquet shards for one source ZIP."""
    api.upload_folder(
        folder_path=str(parquet_dir),
        path_in_repo="data",
        repo_id=config.target_repo,
        repo_type="dataset",
        commit_message=f"Add 320x320 parquet shards for {zip_name}",
    )


def maybe_cleanup_downloaded_zip(config: PipelineConfig, zip_path: Path) -> None:
    """Delete the cached source ZIP in Kaggle runs to keep disk usage stable."""
    if not config.is_kaggle:
        return

    try:
        zip_path.unlink()
        print(f"[resize] Deleted cached ZIP {zip_path.name}")
    except OSError:
        print(f"[resize] Could not delete cached ZIP {zip_path.name}; continuing.")


def process_zip(
    zip_index: int,
    config: PipelineConfig,
    token: str,
    label_map: dict[str, str],
    train_val_files: set[str],
    test_files: set[str],
    stats: PipelineStats,
    api=None,
) -> None:
    """Process one source ZIP into deterministic Parquet shard files."""
    from huggingface_hub import hf_hub_download
    from PIL import ImageFile, UnidentifiedImageError

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    zip_name = f"images_{zip_index:03d}.zip"
    remote_path = f"data/images/{zip_name}"
    shard_index = zip_index - 1

    print(f"[resize] Processing {zip_name} ({zip_index}/{config.num_zips}) ...")
    zip_path = Path(
        hf_hub_download(
            repo_id=config.source_repo,
            filename=remote_path,
            repo_type="dataset",
            token=token,
        )
    )
    print(f"[resize] Downloaded {remote_path}")

    buckets = {
        "train": {"image": [], "labels": [], "filename": []},
        "validation": {"image": [], "labels": [], "filename": []},
        "test": {"image": [], "labels": [], "filename": []},
    }

    zip_unreadable = 0
    zip_unknown = 0

    with zipfile.ZipFile(zip_path, "r") as archive:
        members = [info for info in archive.infolist() if is_supported_member(info)]
        if config.max_images_per_zip is not None:
            members = members[: config.max_images_per_zip]

        print(f"[resize] Found {len(members):,} candidate image entries in {zip_name}")
        for member_index, info in enumerate(members, start=1):
            filename = Path(info.filename).name
            split = split_for(filename, train_val_files, test_files)
            if split == "unknown":
                zip_unknown += 1
                stats.unknown_total += 1
                if len(stats.unknown_samples) < MAX_UNKNOWN_SAMPLES:
                    stats.unknown_samples.append(f"{zip_name}:{info.filename}")
                continue

            try:
                payload = archive.read(info)
                if not payload:
                    raise ValueError("empty file")
                image_payload = resize_image_payload(payload, config.target_size)
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                zip_unreadable += 1
                stats.unreadable_total += 1
                if len(stats.unreadable_samples) < MAX_UNREADABLE_SAMPLES:
                    stats.unreadable_samples.append(
                        (zip_name, info.filename, f"{type(exc).__name__}: {exc}")
                    )
                continue

            buckets[split]["image"].append(image_payload)
            buckets[split]["labels"].append(label_map.get(filename, "No Finding"))
            buckets[split]["filename"].append(filename)
            stats.split_counts[split] += 1
            stats.total_images += 1

            if member_index % 2000 == 0:
                print(
                    f"[resize]   scanned {member_index:,}/{len(members):,} entries "
                    f"in {zip_name}"
                )

    print(
        "[resize] Zip summary: "
        f"train={len(buckets['train']['filename']):,}, "
        f"validation={len(buckets['validation']['filename']):,}, "
        f"test={len(buckets['test']['filename']):,}, "
        f"skipped_unreadable={zip_unreadable:,}, "
        f"skipped_unknown={zip_unknown:,}"
    )

    if config.skip_upload:
        parquet_dir = config.work_dir / "data"
        write_parquet_shards(
            buckets=buckets,
            output_dir=parquet_dir,
            shard_index=shard_index,
            shard_count=config.num_zips,
            stats=stats,
        )
    else:
        temp_dir = config.work_dir / f"tmp_{zip_name}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            write_parquet_shards(
                buckets=buckets,
                output_dir=temp_dir,
                shard_index=shard_index,
                shard_count=config.num_zips,
                stats=stats,
            )
            upload_parquet_shards(api, config, temp_dir, zip_name)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    maybe_cleanup_downloaded_zip(config, zip_path)


def format_bytes(num_bytes: int) -> str:
    """Render bytes with a compact binary unit suffix."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def render_readme(stats: PipelineStats, config: PipelineConfig) -> str:
    """Render a data-only dataset card with accurate split counts and schema."""
    limit_note = ""
    if config.max_zips or config.max_images_per_zip:
        limit_note = (
            "This card reflects a debug smoke-test run because one or more "
            "CHEXVISION_MAX_* limits were enabled."
        )

    front_matter = "\n".join(
        [
            "---",
            "dataset_info:",
            "  features:",
            "    - name: image",
            "      dtype: image",
            "    - name: labels",
            "      dtype: string",
            "    - name: filename",
            "      dtype: string",
            "  splits:",
            "    - name: train",
            f"      num_examples: {stats.split_counts['train']}",
            "    - name: validation",
            f"      num_examples: {stats.split_counts['validation']}",
            "    - name: test",
            f"      num_examples: {stats.split_counts['test']}",
            "configs:",
            "  - config_name: default",
            "    data_files:",
            "      - split: train",
            "        path: data/train-*.parquet",
            "      - split: validation",
            "        path: data/validation-*.parquet",
            "      - split: test",
            "        path: data/test-*.parquet",
            "task_categories:",
            "  - image-classification",
            "task_ids:",
            "  - multi-label-image-classification",
            "language:",
            "  - en",
            "tags:",
            "  - medical",
            "  - chest-xray",
            "  - radiology",
            "  - deep-learning",
            "  - big-data",
            "  - parquet",
            "  - data-only",
            "license: unknown",
            "size_categories:",
            "  - 100K<n<1M",
            "pretty_name: NIH Chest X-ray14 (320x320, Processed for CheXVision)",
            "---",
        ]
    )

    body_lines = [
        "# NIH Chest X-ray14 - 320x320 Processed for CheXVision",
        "",
        "This dataset repackages the raw NIH Chest X-ray14 source dataset from",
        f"[{config.source_repo}](https://huggingface.co/datasets/{config.source_repo})",
        "into a data-only Parquet dataset for the CheXVision project.",
        "",
        "## Dataset Summary",
        "",
        "- Source format: 12 ZIP archives of original chest X-ray images plus CSV manifests",
        "- Output format: data-only Parquet shards under `data/`",
        "- Resolution: 320x320 RGB",
        "- Columns: `image`, `labels`, `filename`",
        "- Split contract: `train`, `validation`, `test`",
        (
            "- Approximate local Parquet size produced in this run: "
            f"`{format_bytes(stats.parquet_bytes)}`"
        ),
    ]
    if limit_note:
        body_lines.extend(["", limit_note, ""])
    else:
        body_lines.append("")

    body_lines.extend(
        [
            "## Splits",
            "",
            "| Split | Images |",
            "|-------|-------:|",
            f"| Train | {stats.split_counts['train']:,} |",
            f"| Validation | {stats.split_counts['validation']:,} |",
            f"| Test | {stats.split_counts['test']:,} |",
            "",
            "## Schema",
            "",
            "- `image`: 320x320 RGB image payload",
            "- `labels`: pipe-delimited pathology labels, or `No Finding`",
            "- `filename`: original NIH image filename",
            "",
            "## Processing Notes",
            "",
            "- Source split manifests come from `train_val_list.txt` and `test_list.txt`",
            "- Validation membership uses the same stable hash-bucket logic as the live",
            "  `HlexNC/chest-xray-14` dataset",
            "- Hidden `__MACOSX` ZIP entries and non-image members are ignored",
            "- Truncated-but-readable images are kept; truly unreadable files are skipped",
            "- This repo intentionally ships no `load_dataset.py` script so it remains a",
            "  data-only dataset that works with the modern HF dataset viewer",
            "",
            "## Usage",
            "",
            "```python",
            "from datasets import load_dataset",
            "",
            f'dataset = load_dataset("{config.target_repo}")',
            "print(dataset)",
            "```",
            "",
            "## Provenance",
            "",
            "Built by the Kaggle kernel `hlexnc/chexvision-resize-320` for the",
            "[CheXVision](https://github.com/arudaev/chexvision) project.",
        ]
    )
    body = "\n".join(body_lines)

    return f"{front_matter}\n\n{body}\n"


def write_or_upload_readme(api, config: PipelineConfig, readme_text: str) -> None:
    """Persist the generated dataset card locally or upload it to the HF repo."""
    if config.skip_upload:
        readme_path = config.work_dir / "README.md"
        readme_path.write_text(readme_text, encoding="utf-8")
        print(f"[resize] Wrote local README to {readme_path}")
        return

    api.upload_file(
        path_or_fileobj=io.BytesIO(readme_text.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=config.target_repo,
        repo_type="dataset",
        commit_message="Update dataset card for 320x320 data-only parquet release",
    )


def validate_final_counts(stats: PipelineStats, config: PipelineConfig) -> None:
    """Enforce the expected full-run split counts in production mode."""
    if config.max_zips or config.max_images_per_zip:
        print("[resize] Debug limits enabled; skipping final full-dataset count check.")
        return

    if stats.split_counts != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(
            "Final split counts do not match the expected live 224 contract. "
            f"Expected {EXPECTED_SPLIT_COUNTS}, got {stats.split_counts}."
        )


def run_resize_320_pipeline(config: PipelineConfig | None = None) -> PipelineStats:
    """Execute the full resize-and-publish pipeline."""
    from huggingface_hub import HfApi

    config = config or build_config()
    hf_token = configure_hf_runtime(required_token=True, check_dns=True)
    if hf_token is None:
        raise RuntimeError("HF token resolution failed unexpectedly.")

    print(
        "[resize] Config: "
        f"target_repo={config.target_repo}, "
        f"target_size={config.target_size}, "
        f"max_zips={config.max_zips}, "
        f"max_images_per_zip={config.max_images_per_zip}, "
        f"skip_upload={config.skip_upload}"
    )
    print("[resize] Downloading source metadata ...")
    label_map, train_val_files, test_files = download_source_metadata(hf_token)
    manifest_counts = verify_split_contract(train_val_files, test_files)
    print(
        "[resize] Verified split contract: "
        f"train={manifest_counts['train']:,}, "
        f"validation={manifest_counts['validation']:,}, "
        f"test={manifest_counts['test']:,}"
    )
    print(f"[resize] Loaded {len(label_map):,} label rows.")

    prepare_local_work_dir(config)

    api = None
    if config.skip_upload:
        print("[resize] CHEXVISION_SKIP_UPLOAD=1 - local smoke-test mode enabled.")
    else:
        api = HfApi(token=hf_token)
        ensure_clean_target_repo(api, config)

    stats = PipelineStats()
    zip_limit = min(config.max_zips or config.num_zips, config.num_zips)

    for zip_index in range(1, zip_limit + 1):
        process_zip(
            zip_index=zip_index,
            config=config,
            token=hf_token,
            label_map=label_map,
            train_val_files=train_val_files,
            test_files=test_files,
            stats=stats,
            api=api,
        )

    validate_final_counts(stats, config)

    readme_text = render_readme(stats, config)
    write_or_upload_readme(api, config, readme_text)

    print("[resize] Run complete.")
    print(
        "[resize] Final counts: "
        f"train={stats.split_counts['train']:,}, "
        f"validation={stats.split_counts['validation']:,}, "
        f"test={stats.split_counts['test']:,}"
    )
    print(f"[resize] Images kept: {stats.total_images:,}")
    print(f"[resize] Unreadable images skipped: {stats.unreadable_total:,}")
    print(f"[resize] Unknown split entries skipped: {stats.unknown_total:,}")
    print(f"[resize] Local parquet footprint: {format_bytes(stats.parquet_bytes)}")

    if not config.skip_upload:
        print(f"[resize] Dataset repo: https://huggingface.co/datasets/{config.target_repo}")

    if stats.unreadable_samples:
        print("[resize] Sample unreadable entries:")
        for zip_name, member, reason in stats.unreadable_samples:
            print(f"[resize]   {zip_name}:{member} -> {reason}")

    if stats.unknown_samples:
        print("[resize] Sample unknown split entries:")
        for sample in stats.unknown_samples:
            print(f"[resize]   {sample}")

    return stats


def main() -> None:
    """CLI entry point for local smoke tests and debugging."""
    run_resize_320_pipeline()


if __name__ == "__main__":
    main()
