#!/usr/bin/env python3
"""Dispatch CheXVision training kernels to Kaggle and manage their lifecycle.

Usage:
    python scripts/dispatch.py kaggle scratch          # push & run the scratch kernel
    python scripts/dispatch.py kaggle transfer         # push & run the transfer kernel
    python scripts/dispatch.py kaggle status scratch   # check kernel status
    python scripts/dispatch.py kaggle output scratch   # download kernel output

Requires the Kaggle CLI: pip install kaggle
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from shutil import rmtree, which

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_ROOT = PROJECT_ROOT / ".codex_tmp" / "kaggle"
BUNDLE_PATHS = (Path("src"), Path("configs"))
BUNDLE_SENTINEL = "__CHEXVISION_PROJECT_BUNDLE_B64__"
EXCLUDED_BUNDLE_DIRS = {"__pycache__", ".pytest_cache"}
EXCLUDED_BUNDLE_SUFFIXES = {".pyc", ".pyo"}

# Map short model names to kernel directory paths (relative to repo root).
KERNEL_DIRS = {
    "scratch": Path("kaggle/train_scratch"),
    "transfer": Path("kaggle/train_transfer"),
}

# Kaggle kernel slugs (must match the "id" in kernel-metadata.json).
KERNEL_SLUGS = {
    "scratch": "hlexnc/chexvision-train-scratch-cnn",
    "transfer": "hlexnc/chexvision-train-densenet-transfer",
}


def _get_kaggle_version() -> tuple[int, ...] | None:
    """Return the installed Kaggle CLI version as a tuple when available."""
    if which("kaggle") is None:
        return None

    result = subprocess.run(
        ["kaggle", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None

    output = (result.stdout or result.stderr).strip()
    prefix = "Kaggle API "
    if output.startswith(prefix):
        output = output[len(prefix):]

    try:
        return tuple(int(part) for part in output.split("."))
    except ValueError:
        return None


def _load_env() -> None:
    """Load environment variables from the project .env when available."""
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        return


def _build_kaggle_bundle(model: str) -> Path:
    """Create a self-contained bundle for Kaggle to run remotely.

    Kaggle script pushes only keep the main code file, so we render a temporary
    script with the project source embedded as a base64 zip payload.
    """
    kernel_dir = PROJECT_ROOT / KERNEL_DIRS[model]
    if not kernel_dir.exists():
        print(f"ERROR: Kernel directory not found: {kernel_dir}", file=sys.stderr)
        sys.exit(1)

    bundle_dir = BUNDLE_ROOT / model
    if bundle_dir.exists():
        rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel_path in BUNDLE_PATHS:
            source_root = PROJECT_ROOT / rel_path
            for path in source_root.rglob("*"):
                if path.is_file() and _should_bundle_path(path):
                    archive.write(path, arcname=path.relative_to(PROJECT_ROOT).as_posix())

    script_template = (kernel_dir / "script.py").read_text(encoding="utf-8")
    if BUNDLE_SENTINEL not in script_template:
        print(
            f"ERROR: Missing {BUNDLE_SENTINEL} placeholder in {kernel_dir / 'script.py'}",
            file=sys.stderr,
        )
        sys.exit(1)

    bundle_b64 = base64.b64encode(archive_buffer.getvalue()).decode("ascii")
    rendered_script = script_template.replace(BUNDLE_SENTINEL, bundle_b64, 1)
    (bundle_dir / "script.py").write_text(rendered_script, encoding="utf-8")
    metadata = _render_kernel_metadata(model)
    (bundle_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return bundle_dir


def _render_kernel_metadata(model: str) -> dict[str, object]:
    """Render the bundle metadata with the required Kaggle runtime flags."""
    metadata_path = PROJECT_ROOT / KERNEL_DIRS[model] / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["id"] = KERNEL_SLUGS[model]
    metadata["code_file"] = "script.py"
    metadata["language"] = "python"
    metadata["kernel_type"] = "script"
    metadata["enable_gpu"] = True
    metadata["enable_internet"] = True
    return metadata


def _should_bundle_path(path: Path) -> bool:
    """Filter out local cache/build artefacts from the Kaggle source bundle."""
    if path.suffix in EXCLUDED_BUNDLE_SUFFIXES:
        return False
    for part in path.parts:
        if part in EXCLUDED_BUNDLE_DIRS or part.endswith(".egg-info"):
            return False
    return True


def _ensure_kaggle_auth(model: str) -> None:
    """Map repo-local Kaggle credentials into the variables the CLI expects."""
    _load_env()

    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return

    api_token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if not api_token:
        print(
            "ERROR: Kaggle credentials not found. Set KAGGLE_USERNAME/KAGGLE_KEY "
            "or provide KAGGLE_API_TOKEN in .env.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Newer Kaggle personal access tokens look like KGAT_... and are handled
    # directly by newer Kaggle CLI releases without a username split.
    if api_token.startswith("KGAT_"):
        version = _get_kaggle_version()
        if version is not None and version < (1, 8, 0):
            print(
                "ERROR: Detected a newer Kaggle API token (KGAT_...), but the "
                f"installed Kaggle CLI is {'.'.join(map(str, version))}. "
                "Upgrade Kaggle CLI to >= 1.8.0 or use kagglehub >= 0.4.1.",
                file=sys.stderr,
            )
            sys.exit(1)
        return

    if ":" in api_token:
        username, key = api_token.split(":", 1)
        os.environ.setdefault("KAGGLE_USERNAME", username)
        os.environ.setdefault("KAGGLE_KEY", key)
        return

    owner = KERNEL_SLUGS[model].split("/", 1)[0]
    os.environ.setdefault("KAGGLE_USERNAME", owner)
    os.environ.setdefault("KAGGLE_KEY", api_token)


def _run(cmd: list[str]) -> None:
    """Run a subprocess and stream its output."""
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"Command exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def cmd_push(model: str) -> None:
    """Push a kernel folder to Kaggle (triggers a new run)."""
    _ensure_kaggle_auth(model)
    bundle_dir = _build_kaggle_bundle(model)
    print(
        "NOTE: Kaggle runs remotely and will not inherit local .env values. "
        "Add HF_TOKEN in Kaggle Secrets for authenticated HF dataset access "
        "and automatic model uploads. Dispatch bundles always force Kaggle "
        "internet and GPU on for training kernels."
    )
    _run(["kaggle", "kernels", "push", "-p", str(bundle_dir)])


def cmd_status(model: str) -> None:
    """Check the current status of a Kaggle kernel."""
    _ensure_kaggle_auth(model)
    slug = KERNEL_SLUGS[model]
    _run(["kaggle", "kernels", "status", slug])


def cmd_output(model: str) -> None:
    """Download the output files of a completed Kaggle kernel."""
    _ensure_kaggle_auth(model)
    slug = KERNEL_SLUGS[model]
    out_dir = Path(f"kaggle_output/{model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    _run(["kaggle", "kernels", "output", slug, "-p", str(out_dir)])
    print(f"Output saved to {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dispatch CheXVision training to Kaggle."
    )
    subparsers = parser.add_subparsers(dest="platform", help="Target platform")

    # --- kaggle sub-command ---------------------------------------------------
    kaggle_parser = subparsers.add_parser("kaggle", help="Kaggle kernel operations")
    kaggle_sub = kaggle_parser.add_subparsers(dest="action", help="Action to perform")

    # kaggle push (default when just model name given)
    push_parser = kaggle_sub.add_parser("push", help="Push kernel to Kaggle")
    push_parser.add_argument("model", choices=["scratch", "transfer"])

    # kaggle status
    status_parser = kaggle_sub.add_parser("status", help="Check kernel status")
    status_parser.add_argument("model", choices=["scratch", "transfer"])

    # kaggle output
    output_parser = kaggle_sub.add_parser("output", help="Download kernel output")
    output_parser.add_argument("model", choices=["scratch", "transfer"])

    args = parser.parse_args()

    if args.platform is None:
        parser.print_help()
        sys.exit(1)

    if args.platform == "kaggle":
        # Allow shorthand: `dispatch.py kaggle scratch` == `dispatch.py kaggle push scratch`
        if args.action is None:
            parser.print_help()
            sys.exit(1)

        if args.action == "push":
            cmd_push(args.model)
        elif args.action == "status":
            cmd_status(args.model)
        elif args.action == "output":
            cmd_output(args.model)
        else:
            # Handle the shorthand case where action IS the model name
            kaggle_parser.print_help()
            sys.exit(1)


# ---------------------------------------------------------------------------
# Support the shorthand syntax from the docstring:
#   python scripts/dispatch.py kaggle scratch
#   python scripts/dispatch.py kaggle status scratch
#
# argparse subcommands alone can't handle both forms, so we do a small
# pre-processing step on sys.argv before parsing.
# ---------------------------------------------------------------------------

def _preprocess_argv() -> None:
    """Rewrite argv so that `kaggle <model>` becomes `kaggle push <model>`."""
    model_names = {"scratch", "transfer"}
    # Pattern: script kaggle <model>  (3 args after script name, 2nd is kaggle, 3rd is model)
    if len(sys.argv) >= 3 and sys.argv[1] == "kaggle" and sys.argv[2] in model_names:
        # Insert "push" between "kaggle" and the model name
        sys.argv.insert(2, "push")


if __name__ == "__main__":
    _preprocess_argv()
    main()
