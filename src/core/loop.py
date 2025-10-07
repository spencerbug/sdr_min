"""Top-level loop wiring the stub components together."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from ..utils import PacketValidator
from .column import ColumnConfig, ColumnStepResult, ColumnSystem
from .encoders import ContextConfig, ContextEncoder
from .env_ycb import EnvConfig, YCBHabitatAdapter
from .policy import PolicyConfig, RandomPolicy

LOGGER = logging.getLogger(__name__)


@dataclass
class LoopConfig:
    steps: int = 10
    seed: int = 0
    env: EnvConfig = field(default_factory=EnvConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    column: ColumnConfig = field(default_factory=ColumnConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)


def run_loop(config: LoopConfig) -> Dict[str, float]:
    """Run the SDR loop for ``config.steps`` ticks and return summary stats."""

    rng = np.random.default_rng(config.seed)
    validator = PacketValidator()
    env = YCBHabitatAdapter(config.env, validator, rng)
    context_encoder = ContextEncoder(config.context, validator)
    columns = ColumnSystem(config.column, config.context, validator, rng)
    policy = RandomPolicy(config.policy, validator, rng)

    observation, pose, raw_context = env.reset()
    entropies = []
    peakiness_values = []
    facet_losses = []

    for step in range(config.steps):
        context_packet = context_encoder.encode(raw_context)
        column_result: ColumnStepResult = columns.step(observation, pose, context_packet)
        belief_packet = column_result.belief_packet
        entropies.append(belief_packet["entropy"])
        peakiness_values.append(belief_packet["peakiness"])
        for record in column_result.facet_records:
            if "L1" in record["losses"]:
                facet_losses.append(record["losses"]["L1"])
        action_packet = policy.act(belief_packet)
        LOGGER.debug("step=%d action=%s entropy=%.3f", step, action_packet["action_type"], belief_packet["entropy"])
        observation, pose, raw_context = env.step(action_packet)

    summary = {
        "entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
        "peakiness_mean": float(np.mean(peakiness_values)) if peakiness_values else 0.0,
        "facet_L1_mean": float(np.mean(facet_losses)) if facet_losses else 0.0,
        "steps": float(config.steps),
    }
    return summary
