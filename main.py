"""CLI entry point for running the SDR minimal loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from src.core.loop import LoopConfig, run_loop
from src.core.env_ycb import EnvConfig
from src.core.encoders import ContextConfig
from src.core.column import ColumnConfig
from src.core.policy import PolicyConfig
from src.core.fusion import FusionConfig
from src.core.consensus import ConsensusConfig
from src.core.facet import FacetConfig


def _as_tuple(sequence: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(sequence)


def _load_config(path: Path) -> LoopConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    column_payload = payload.get("column", {})
    fusion_payload = column_payload.get("fusion", {})
    consensus_payload = column_payload.get("consensus", {})
    facet_payload = column_payload.get("facet", {})

    return LoopConfig(
        steps=int(payload.get("steps", 10)),
        seed=int(payload.get("seed", 0)),
        env=EnvConfig(
            patch_shape=_as_tuple(payload.get("env", {}).get("patch_shape", (64, 64, 4))),
            context_length=int(payload.get("env", {}).get("context_length", 1024)),
            dt=float(payload.get("env", {}).get("dt", 0.05)),
            columns=_as_tuple(payload.get("env", {}).get("columns", ("col0",))),
            objects=_as_tuple(payload.get("env", {}).get("objects", ("stub_object", "alt_object"))),
        ),
        context=ContextConfig(
            length=int(payload.get("context", {}).get("length", 1024)),
            sources=_as_tuple(payload.get("context", {}).get("sources", ("metronome", "switch", "intent"))),
        ),
        column=ColumnConfig(
            phase_dim=int(column_payload.get("phase_dim", 2048)),
            feature_dim=int(column_payload.get("feature_dim", 4096)),
            topk_phase=int(column_payload.get("topk_phase", 32)),
            topk_feature=int(column_payload.get("topk_feature", 64)),
            fusion=FusionConfig(
                alpha=float(fusion_payload.get("alpha", 0.6)),
                beta_feature=float(fusion_payload.get("beta_feature", 0.3)),
                beta_context=float(fusion_payload.get("beta_context", 0.1)),
                clip_min=float(fusion_payload.get("clip_min", -5.0)),
                clip_max=float(fusion_payload.get("clip_max", 5.0)),
            ),
            consensus=ConsensusConfig(
                temperature=float(consensus_payload.get("temperature", 1.0)),
                shared_topk=int(consensus_payload.get("shared_topk", 32)),
            ),
            facet=FacetConfig(
                facet_shape=_as_tuple(facet_payload.get("facet_shape", (32, 32))),
                topk_shared=int(facet_payload.get("topk_shared", 3)),
            ),
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
