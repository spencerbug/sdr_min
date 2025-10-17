"""Minimal Examiner runner demo.

Run this module to execute the single-column Examiner loop for a short horizon
and print the summary statistics. The script mirrors the defaults documented in
``docs/design.md`` and the examiner issue tracker.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# Allow running the script directly without setting PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.column import ColumnConfig
from src.core.consensus import ConsensusConfig
from src.core.encoders import ContextConfig
from src.core.env_ycb import AssetConfig, EnvConfig, OrbitConfig, SensorConfig
from src.core.facet import FacetConfig
from src.core.fusion import FusionConfig
from src.core.loop import LoopConfig, run_loop
from src.core.policy import PolicyConfig


def build_examiner_config(steps: int = 120, seed: int = 7, backend: str = "stub") -> LoopConfig:
    """Construct a loop configuration tailored to the Examiner scenario."""

    return LoopConfig(
        steps=steps,
        seed=seed,
        env=EnvConfig(
            backend=backend,
            columns=("col0",),
            context_length=1024,
            sensor=SensorConfig(resolution=(64, 64), modalities=("rgb", "depth")),
            orbit=OrbitConfig(radius=0.6, min_elevation=-0.6, max_elevation=0.6),
            assets=AssetConfig(
                examiner_scene="examiner/void_black.glb",
                explorer_scene="mp3d/17DRP5sb8fy/17DRP5sb8fy.glb",
            ),
        ),
        context=ContextConfig(length=1024, sources=("metronome", "switch", "intent")),
        column=ColumnConfig(
            phase_dim=2048,
            feature_dim=4096,
            topk_phase=32,
            topk_feature=64,
            fusion=FusionConfig(alpha=0.6, beta_feature=0.3, beta_context=0.1, clip_min=-5.0, clip_max=5.0),
            consensus=ConsensusConfig(temperature=1.5, shared_topk=32),
            facet=FacetConfig(facet_shape=(16, 16), topk_shared=3),
        ),
        policy=PolicyConfig(step_scale=0.1),
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Run the Examiner minimal loop demo.")
    parser.add_argument("--steps", type=int, default=120, help="Number of steps to simulate (default: 120)")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed for reproducibility (default: 7)")
    parser.add_argument(
        "--backend",
        choices=("stub", "habitat"),
        default="stub",
        help="Environment backend to use (default: stub)",
    )
    args = parser.parse_args(argv)

    config = build_examiner_config(steps=args.steps, seed=args.seed, backend=args.backend)
    summary = run_loop(config)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
