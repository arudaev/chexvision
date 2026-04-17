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
import os
import subprocess
import sys
from pathlib import Path

# Map short model names to kernel directory paths (relative to repo root).
KERNEL_DIRS = {
    "scratch": Path("kaggle/train_scratch"),
    "transfer": Path("kaggle/train_transfer"),
}

# Kaggle kernel slugs (must match the "id" in kernel-metadata.json).
KERNEL_SLUGS = {
    "scratch": "HlexNC/chexvision-train-scratch",
    "transfer": "HlexNC/chexvision-train-transfer",
}


def _load_env() -> None:
    """Load environment variables from the project .env when available."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        return


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
    kernel_dir = KERNEL_DIRS[model]
    if not kernel_dir.exists():
        print(f"ERROR: Kernel directory not found: {kernel_dir}", file=sys.stderr)
        sys.exit(1)
    _run(["kaggle", "kernels", "push", "-p", str(kernel_dir)])


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
