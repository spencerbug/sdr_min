"""CLI entry point for running the SDR minimal loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from src.core.loop import LoopConfig, run_loop
from src.core.env_ycb import EnvConfig
from src.core.context import ContextConfig
from src.core.column import ColumnConfig
from src.core.policy import PolicyConfig


def _as_tuple(sequence: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(sequence)


def _load_config(path: Path) -> LoopConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return LoopConfig(
        steps=int(payload.get("steps", 10)),
        seed=int(payload.get("seed", 0)),
        env=EnvConfig(
            patch_shape=_as_tuple(payload.get("env", {}).get("patch_shape", (64, 64, 4))),
            context_length=int(payload.get("env", {}).get("context_length", 1024)),
            dt=float(payload.get("env", {}).get("dt", 0.05)),
            columns=_as_tuple(payload.get("env", {}).get("columns", ("col0",))),
        ),
        context=ContextConfig(
            length=int(payload.get("context", {}).get("length", 1024)),
            sources=_as_tuple(payload.get("context", {}).get("sources", ("metronome", "switch", "intent"))),
        ),
        column=ColumnConfig(
            phase_dim=int(payload.get("column", {}).get("phase_dim", 2048)),
            feature_dim=int(payload.get("column", {}).get("feature_dim", 4096)),
            topk_phase=int(payload.get("column", {}).get("topk_phase", 32)),
            topk_feature=int(payload.get("column", {}).get("topk_feature", 64)),
        ),
        policy=PolicyConfig(
            step_scale=float(payload.get("policy", {}).get("step_scale", 0.1)),
        ),
    )


def main(argv: Sequence[str] | None = None) -> Dict[str, float]:
    parser = argparse.ArgumentParser(description="Run the SDR minimal loop experiment.")
    parser.add_argument("config", nargs="?", help="Path to a JSON config file.")
    args = parser.parse_args(argv)

    if args.config:
        config = _load_config(Path(args.config))
    else:
        config = LoopConfig()

    summary = run_loop(config)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
