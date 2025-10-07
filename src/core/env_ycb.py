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
    objects: Tuple[str, ...] = ("stub_object", "alt_object")


class YCBHabitatAdapter:
    """Generates observation + pose streams matching the packet schemas."""

    def __init__(self, config: EnvConfig, validator: PacketValidator, rng: np.random.Generator) -> None:
        self._config = config
        self._validator = validator
        self._rng = rng
        self._tick = 0
        self._current_object_idx = 0
        self._last_switch = False
        self._pose: Dict[str, Tuple[float, float]] = {
            col: (float(self._rng.random()), float(self._rng.random())) for col in config.columns
        }
        self._pose_prev: Dict[str, Tuple[float, float]] = dict(self._pose)

    def reset(self) -> Tuple[Dict[str, object], Dict[str, object], List[int]]:
        """Reset the environment state and return the first packets."""

        self._tick = 0
        self._current_object_idx = int(self._rng.integers(len(self._config.objects)))
        self._last_switch = False
        self._pose_prev = dict(self._pose)
        return self._emit_packets()

    def step(self, _action_packet: Dict[str, object]) -> Tuple[Dict[str, object], Dict[str, object], List[int]]:
        """Advance the stub environment one tick."""

        action_type = _action_packet.get("action_type", "noop")
        params = _action_packet.get("params", {}) if isinstance(_action_packet, dict) else {}
        self._apply_action(action_type, params)
        self._tick += 1
        self._last_switch = action_type == "switch_object"
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
            self._pose_prev[column_id] = (u_prev, v_prev)
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
                "object_id": self._config.objects[self._current_object_idx],
                "tick": self._tick,
                "camera_intr": [575.8, 575.8, 320.0, 240.0],
            },
        }
        return observation_packet

    def _make_pose_packet(self) -> Dict[str, object]:
        pose_entries = []
        for column_id in self._pose:
            u, v = self._pose[column_id]
            u_prev, v_prev = self._pose_prev[column_id]
            pose_entries.append(
                {
                    "column_id": column_id,
                    "pose_t": {"u": u, "v": v},
                    "pose_tm1": {"u": u_prev, "v": v_prev},
                }
            )
        return {
            "type": "pose.v1",
            "per_column": pose_entries,
            "dt": self._config.dt,
        }

    def _sample_context_indices(self) -> List[int]:
        metronome_even_bit = 0
        metronome_odd_bit = 2
        switch_bit = 1
        indices = {metronome_even_bit if self._tick % 2 == 0 else metronome_odd_bit}
        if self._last_switch:
            indices.add(switch_bit)
        active_k = max(1, int(0.02 * self._config.context_length))
        extras_needed = max(0, active_k - len(indices))
        if extras_needed > 0:
            candidates = self._rng.choice(self._config.context_length, size=extras_needed, replace=False).astype(int)
            indices.update(int(idx) for idx in candidates)
        return sorted(indices)

    def _apply_action(self, action_type: str, params: Dict[str, object]) -> None:
        if action_type == "move":
            dx = float(params.get("dx", 0.0))
            dy = float(params.get("dy", 0.0))
            for column_id, (u, v) in self._pose.items():
                self._pose_prev[column_id] = (u, v)
                self._pose[column_id] = ((u + dx) % 1.0, (v + dy) % 1.0)
        elif action_type == "jump_to":
            target_u = float(params.get("u", self._rng.random()))
            target_v = float(params.get("v", self._rng.random()))
            for column_id in self._pose:
                self._pose_prev[column_id] = self._pose[column_id]
                self._pose[column_id] = (target_u % 1.0, target_v % 1.0)
        elif action_type == "switch_object":
            self._current_object_idx = (self._current_object_idx + 1) % len(self._config.objects)
        # noop and unknown actions keep state unchanged
