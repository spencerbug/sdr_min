"""Motor / policy stub that emits ActionMessage packets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ..utils import PacketValidator


@dataclass
class PolicyConfig:
    step_scale: float = 0.1


class RandomPolicy:
    """Sample random movements within the allowed action space."""

    def __init__(self, config: PolicyConfig, validator: PacketValidator, rng: np.random.Generator) -> None:
        self._config = config
        self._validator = validator
        self._rng = rng

    def act(self, belief_packet: Dict[str, object]) -> Dict[str, object]:
        entropy = float(belief_packet.get("entropy", 0.0))
        roll = self._rng.random()
        if entropy > 1.5 and roll > 0.6:
            action_type = "switch_object"
            params = {}
        elif roll < 0.15:
            action_type = "jump_to"
            params = {"u": float(self._rng.random()), "v": float(self._rng.random())}
        elif roll < 0.7:
            action_type = "move"
            params = {
                "dx": float(self._rng.uniform(-self._config.step_scale, self._config.step_scale)),
                "dy": float(self._rng.uniform(-self._config.step_scale, self._config.step_scale)),
            }
        else:
            action_type = "noop"
            params = {}
        packet = {
            "type": "action.v1",
            "action_type": action_type,
            "params": params,
        }
        intent_bits = self._intent_bits(belief_packet)
        if intent_bits["indices"]:
            packet["intent_bits"] = intent_bits
        self._validator.validate(packet)
        return packet

    def _intent_bits(self, belief_packet: Dict[str, object]) -> Dict[str, object]:
        shared = belief_packet.get("g_star_sdr", {})
        indices: List[int] = [int(idx) % 128 for idx in shared.get("indices", [])[:2]]
        unique_sorted = sorted(set(indices))
        return {"indices": unique_sorted, "length": 128}
