#!/usr/bin/env python3
"""Utility for downloading the Habitat YCB dataset from Hugging Face."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from huggingface_hub import snapshot_download as _snapshot_download_type


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download YCB assets via huggingface_hub.")
    parser.add_argument(
        "--repo-id",
        default="ai-habitat/ycb",
        help="Hugging Face dataset repo id to download (default: ai-habitat/ycb)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional git revision (tag/commit) to pin. Defaults to latest.",
    )
    parser.add_argument(
        "--dest",
        default="assets/ycb",
        help="Destination directory for extracted assets (default: assets/ycb)",
    )
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=None,
        help="Optional glob patterns to limit downloaded files (e.g. '*/mesh*').",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional huggingface cache directory override.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face access token. Falls back to HF_TOKEN env var if not provided.",
    )
    parser.add_argument(
        "--no-symlinks",
        action="store_true",
        help="Disable use of symlinks inside the destination directory.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - import guard only
        raise SystemExit("huggingface_hub is required; install with pip install huggingface-hub") from exc

    args = parse_args(argv)
    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    token = args.token or os.environ.get("HF_TOKEN")

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        allow_patterns=args.patterns,
        local_dir=dest,
        local_dir_use_symlinks=not args.no_symlinks,
        cache_dir=args.cache_dir,
        token=token,
    )

    return dest


if __name__ == "__main__":
    main()
