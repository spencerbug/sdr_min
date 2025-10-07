"""Smoke test for the SDR loop wiring."""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.loop import LoopConfig, run_loop


def test_loop_runs_for_ten_steps() -> None:
    config = LoopConfig(steps=10, seed=42)
    summary = run_loop(config)
    assert math.isfinite(summary["entropy_mean"])
    assert math.isfinite(summary["peakiness_mean"])
    assert summary["steps"] == float(config.steps)
