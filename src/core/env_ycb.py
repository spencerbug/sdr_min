"""Minimal YCB-Habitat adapter stub.

The real implementation will call Habitat-Sim. For now, the adapter
creates structurally correct packets so the loop, validators, and tests
exercise the contracts end-to-end.

This module also defines configuration dataclasses that describe the
Habitat-backed environment surface. The schema is consumed by both the
stub adapter and the forthcoming Habitat integration to ensure
consistent defaults and validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ..utils import PacketValidator


def _as_tuple(sequence: Sequence[Any] | Any, *, item_cast=None) -> Tuple[Any, ...]:
    if isinstance(sequence, (str, bytes)):
        sequence = (sequence,)
    items: List[Any] = []
    for value in sequence:  # type: ignore[arg-type]
        if item_cast is not None:
            value = item_cast(value)
        items.append(value)
    return tuple(items)


@dataclass
class SensorConfig:
    """Describes a single pinhole sensor attached to the Habitat agent."""

    resolution: Tuple[int, int] = (64, 64)
    modalities: Tuple[str, ...] = ("rgb", "depth")
    hfov: float = 70.0
    near: float = 0.1
    far: float = 5.0

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SensorConfig:
        return cls(
            resolution=_as_tuple(payload.get("resolution", (64, 64)), item_cast=int),
            modalities=_as_tuple(payload.get("modalities", ("rgb", "depth")), item_cast=str),
            hfov=float(payload.get("hfov", 70.0)),
            near=float(payload.get("near", 0.1)),
            far=float(payload.get("far", 5.0)),
        )

    def validate(self) -> None:
        if len(self.resolution) != 2 or any(dim <= 0 for dim in self.resolution):
            raise ValueError("sensor.resolution must be a (H, W) tuple with positive entries")
        if self.near <= 0 or self.far <= self.near:
            raise ValueError("sensor clipping planes must satisfy 0 < near < far")
        if not self.modalities:
            raise ValueError("sensor.modalities must declare at least one modality")


@dataclass
class OrbitConfig:
    """Defines the orbit used by the Examiner scenario."""

    radius: float = 0.6
    min_elevation: float = -0.55
    max_elevation: float = 0.65
    jitter: float = 0.0
    default_speed: float = 0.05

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OrbitConfig:
        return cls(
            radius=float(payload.get("radius", 0.6)),
            min_elevation=float(payload.get("min_elevation", -0.55)),
            max_elevation=float(payload.get("max_elevation", 0.65)),
            jitter=float(payload.get("jitter", 0.0)),
            default_speed=float(payload.get("default_speed", 0.05)),
        )

    def validate(self) -> None:
        if self.radius <= 0:
            raise ValueError("orbit.radius must be positive")
        if self.max_elevation <= self.min_elevation:
            raise ValueError("orbit.max_elevation must exceed orbit.min_elevation")
        if self.default_speed <= 0:
            raise ValueError("orbit.default_speed must be positive")


@dataclass
class PhysicsConfig:
    """Lightweight toggles for Habitat physics simulation."""

    enable_physics: bool = False
    enable_sliding: bool = True
    lock_object: bool = True

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PhysicsConfig:
        return cls(
            enable_physics=bool(payload.get("enable_physics", False)),
            enable_sliding=bool(payload.get("enable_sliding", True)),
            lock_object=bool(payload.get("lock_object", True)),
        )

    def validate(self) -> None:
        # No numerical relationships to check yet; placeholder for future invariants.
        return


@dataclass
class AssetConfig:
    """Filesystem layout for Habitat assets."""

    ycb_root: Path = Path("assets/ycb")
    scene_root: Path = Path("assets/scenes")
    examiner_scene: str = "examiner/void_black.glb"
    explorer_scene: str = "mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"

    def __post_init__(self) -> None:
        self.ycb_root = Path(self.ycb_root)
        self.scene_root = Path(self.scene_root)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AssetConfig:
        return cls(
            ycb_root=Path(payload.get("ycb_root", "assets/ycb")),
            scene_root=Path(payload.get("scene_root", "assets/scenes")),
            examiner_scene=str(payload.get("examiner_scene", "examiner/void_black.glb")),
            explorer_scene=str(payload.get("explorer_scene", "mp3d/17DRP5sb8fy/17DRP5sb8fy.glb")),
        )

    def validate(self) -> None:
        if not self.examiner_scene:
            raise ValueError("assets.examiner_scene must reference a GLB relative to scene_root")
        if not self.explorer_scene:
            raise ValueError("assets.explorer_scene must reference a GLB relative to scene_root")


DEFAULT_OBJECTS: Tuple[str, ...] = (
    "003_cracker_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
)


@dataclass
class EnvConfig:
    """Shared configuration for the stub and real Habitat adapters."""

    scenario: str = "examiner"
    backend: str = "stub"
    columns: Tuple[str, ...] = ("col0",)
    context_length: int = 1024
    dt: float = 0.05
    objects: Tuple[str, ...] = DEFAULT_OBJECTS
    patch_shape: Tuple[int, int, int] | None = None
    sensor: SensorConfig = field(default_factory=SensorConfig)
    orbit: OrbitConfig = field(default_factory=OrbitConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    assets: AssetConfig = field(default_factory=AssetConfig)

    def __post_init__(self) -> None:
        self.columns = _as_tuple(self.columns, item_cast=str)
        self.objects = _as_tuple(self.objects, item_cast=str)
        if self.patch_shape is None:
            channels = len(self.sensor.modalities)
            self.patch_shape = (
                int(self.sensor.resolution[0]),
                int(self.sensor.resolution[1]),
                int(channels),
            )
        self.validate()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EnvConfig:
        sensor_cfg = SensorConfig.from_dict(payload.get("sensor", {}))
        orbit_cfg = OrbitConfig.from_dict(payload.get("orbit", {}))
        physics_cfg = PhysicsConfig.from_dict(payload.get("physics", {}))
        assets_cfg = AssetConfig.from_dict(payload.get("assets", {}))

        patch_shape_payload = payload.get("patch_shape")
        patch_shape: Tuple[int, int, int] | None
        if patch_shape_payload is None:
            patch_shape = None
        else:
            if isinstance(patch_shape_payload, Sequence):
                patch_shape = _as_tuple(patch_shape_payload, item_cast=int)  # type: ignore[assignment]
            else:
                raise TypeError("env.patch_shape must be a sequence if provided")

        return cls(
            scenario=str(payload.get("scenario", "examiner")),
            backend=str(payload.get("backend", "stub")),
            columns=_as_tuple(payload.get("columns", ("col0",)), item_cast=str),
            context_length=int(payload.get("context_length", 1024)),
            dt=float(payload.get("dt", 0.05)),
            objects=_as_tuple(payload.get("objects", DEFAULT_OBJECTS), item_cast=str),
            patch_shape=patch_shape,
            sensor=sensor_cfg,
            orbit=orbit_cfg,
            physics=physics_cfg,
            assets=assets_cfg,
        )

    def validate(self) -> None:
        if self.scenario not in {"examiner", "explorer"}:
            raise ValueError("env.scenario must be 'examiner' or 'explorer'")
        if self.backend not in {"stub", "habitat"}:
            raise ValueError("env.backend must be 'stub' or 'habitat'")
        if not self.columns:
            raise ValueError("env.columns must list at least one column id")
        if self.context_length <= 0:
            raise ValueError("env.context_length must be positive")
        if self.dt <= 0:
            raise ValueError("env.dt must be positive")
        if not self.objects:
            raise ValueError("env.objects must include at least one YCB object id")
        if self.patch_shape is None:
            raise ValueError("env.patch_shape cannot be None after initialization")
        if len(self.patch_shape) != 3 or any(dim <= 0 for dim in self.patch_shape):
            raise ValueError("env.patch_shape must be a (H, W, C) tuple with positive entries")

        self.sensor.validate()
        self.orbit.validate()
        self.physics.validate()
        self.assets.validate()


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
                    "channels": list(self._config.sensor.modalities),
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
            speed = float(self._config.orbit.default_speed)
            for column_id, (u, v) in self._pose.items():
                self._pose_prev[column_id] = (u, v)
                self._pose[column_id] = ((u + speed * dx) % 1.0, (v + speed * dy) % 1.0)
        elif action_type == "jump_to":
            target_u = float(params.get("u", self._rng.random()))
            target_v = float(params.get("v", self._rng.random()))
            for column_id in self._pose:
                self._pose_prev[column_id] = self._pose[column_id]
                self._pose[column_id] = (target_u % 1.0, target_v % 1.0)
        elif action_type == "switch_object":
            self._current_object_idx = (self._current_object_idx + 1) % len(self._config.objects)
        # noop and unknown actions keep state unchanged
