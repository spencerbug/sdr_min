"""Motor / policy stub that emits ActionMessage packets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

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

    def act(self, _belief_packet: Dict[str, object]) -> Dict[str, object]:
        action_type = "move" if self._rng.random() > 0.1 else "noop"
        if action_type == "move":
            params = {
                "dx": float(self._rng.uniform(-self._config.step_scale, self._config.step_scale)),
                "dy": float(self._rng.uniform(-self._config.step_scale, self._config.step_scale)),
            }
        else:
            params = {}
        packet = {"type": "action.v1", "action_type": action_type, "params": params}
        self._validator.validate(packet)
        return packet
