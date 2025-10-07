"""Minimal YCB-Habitat adapter stub.

The real implementation will call Habitat-Sim. For now, the adapter
creates structurally correct packets so the loop, validators, and tests
exercise the contracts end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..utils import PacketValidator


@dataclass
class EnvConfig:
    """Configuration subset required by the stub environment."""

    patch_shape: Tuple[int, int, int] = (64, 64, 4)
    context_length: int = 1024
    dt: float = 0.05
    columns: Tuple[str, ...] = ("col0",)


class YCBHabitatAdapter:
    """Generates observation + pose streams matching the packet schemas."""

    def __init__(self, config: EnvConfig, validator: PacketValidator, rng: np.random.Generator) -> None:
        self._config = config
        self._validator = validator
        self._rng = rng
        self._tick = 0
        self._pose: Dict[str, Tuple[float, float]] = {
            col: (float(self._rng.random()), float(self._rng.random())) for col in config.columns
        }

    def reset(self) -> Tuple[Dict[str, object], Dict[str, object], List[int]]:
        """Reset the environment state and return the first packets."""

        self._tick = 0
        return self._emit_packets()

    def step(self, _action_packet: Dict[str, object]) -> Tuple[Dict[str, object], Dict[str, object], List[int]]:
        """Advance the stub environment one tick."""

        self._tick += 1
        return self._emit_packets()

    def _emit_packets(self) -> Tuple[Dict[str, object], Dict[str, object], List[int]]:
        observation = self._make_observation_packet()
        pose = self._make_pose_packet()
        context_indices = self._sample_context_indices()
        # Validate packets immediately to enforce contracts.
        self._validator.validate(observation)
        self._validator.validate(pose)
        return observation, pose, context_indices

    def _make_observation_packet(self) -> Dict[str, object]:
        columns_payload = []
        for column_id in self._config.columns:
            u_prev, v_prev = self._pose[column_id]
            delta = self._rng.normal(scale=0.01, size=2)
            u = float((u_prev + delta[0]) % 1.0)
            v = float((v_prev + delta[1]) % 1.0)
            self._pose[column_id] = (u, v)
            columns_payload.append(
                {
                    "column_id": column_id,
                    "view_id": f"{column_id}:sim",
                    "patch": {
                        "dtype": "uint8",
                        "shape": list(self._config.patch_shape),
                        "storage": f"shm://obs/{column_id}/{self._tick:06d}"
                    },
                    "channels": ["rgb", "depth"],
                    "egopose": {
                        "u": u,
                        "v": v,
                        "u_prev": u_prev,
                        "v_prev": v_prev,
                    },
                }
            )
        observation_packet = {
            "type": "observation.v1",
            "columns": columns_payload,
            "global_meta": {
                "object_id": "stub_object",
                "tick": self._tick,
                "camera_intr": [575.8, 575.8, 320.0, 240.0],
            },
        }
        return observation_packet

    def _make_pose_packet(self) -> Dict[str, object]:
        pose_entries = []
        for column_id, (u, v) in self._pose.items():
            pose_entries.append(
                {
                    "column_id": column_id,
                    "pose_t": {"u": u, "v": v},
                    "pose_tm1": {"u": u, "v": v},
                }
            )
        return {
            "type": "pose.v1",
            "per_column": pose_entries,
            "dt": self._config.dt,
        }

    def _sample_context_indices(self) -> List[int]:
        active_k = max(1, int(0.02 * self._config.context_length))
        return sorted(
            self._rng.choice(self._config.context_length, size=active_k, replace=False).astype(int).tolist()
        )
