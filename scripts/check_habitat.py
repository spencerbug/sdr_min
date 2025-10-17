#!/usr/bin/env python3
"""Quick diagnostic for Habitat-Sim / Habitat-Lab availability."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from types import ModuleType
from typing import Any, Dict


def import_optional(name: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"module": name, "available": False, "version": None, "error": None}
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - diagnostics only
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    result["available"] = True
    result["version"] = getattr(module, "__version__", None)
    if name == "habitat_sim":
        result.update(check_sim_backend(module))
    return result


def check_sim_backend(habitat_sim: ModuleType) -> Dict[str, Any]:
    backend: Dict[str, Any] = {"gpu_available": None, "backend": None}
    try:
        import torch  # type: ignore

        backend["gpu_available"] = bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - diagnostics only
        backend["gpu_available"] = None
    try:
        egl = getattr(habitat_sim, "with_egl").__bool__() if hasattr(habitat_sim, "with_egl") else None
        cuda = getattr(habitat_sim, "with_cuda").__bool__() if hasattr(habitat_sim, "with_cuda") else None
        backend["backend"] = {
            "egl": egl,
            "cuda": cuda,
        }
    except Exception:  # pragma: no cover - diagnostics only
        backend["backend"] = None
    return backend


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Habitat imports and GPU availability.")
    parser.add_argument("--require-gpu", action="store_true", help="Fail if CUDA is unavailable.")
    args = parser.parse_args()

    report: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "modules": {
            "habitat_sim": import_optional("habitat_sim"),
            "habitat": import_optional("habitat"),
        },
    }

    sim_info = report["modules"]["habitat_sim"]
    success = bool(sim_info.get("available"))
    if args.require_gpu:
        success = success and bool(sim_info.get("gpu_available"))

    print(json.dumps(report, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
