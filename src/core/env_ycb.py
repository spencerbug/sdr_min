"""YCB-Habitat adapter supporting both stub and Habitat-Sim backends."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ..utils import PacketValidator

try:  # pragma: no cover - optional dependency
    import habitat_sim  # type: ignore
    from habitat_sim import physics as habitat_physics  # type: ignore
    from habitat_sim.utils import common as habitat_utils  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    habitat_sim = None  # type: ignore
    habitat_physics = None  # type: ignore
    habitat_utils = None  # type: ignore


def _as_tuple(sequence: Sequence[Any] | Any, *, item_cast=None) -> Tuple[Any, ...]:
    if isinstance(sequence, (str, bytes)):
        sequence = (sequence,)
    items: List[Any] = []
    for value in sequence:  # type: ignore[arg-type]
        if item_cast is not None:
            value = item_cast(value)
        items.append(value)
    return tuple(items)


def _sample_context_indices(
    rng: np.random.Generator,
    tick: int,
    context_length: int,
    last_switch: bool,
) -> List[int]:
    """Shared helper for generating sparse context indices."""

    metronome_even_bit = 0
    metronome_odd_bit = 2
    switch_bit = 1
    indices = {metronome_even_bit if tick % 2 == 0 else metronome_odd_bit}
    if last_switch:
        indices.add(switch_bit)
    active_k = max(1, int(0.02 * context_length))
    extras_needed = max(0, active_k - len(indices))
    if extras_needed > 0 and context_length > len(indices):
        available = np.setdiff1d(
            np.arange(context_length, dtype=int), np.fromiter(indices, dtype=int, count=len(indices)), assume_unique=True
        )
        if available.size > 0:
            take = min(extras_needed, int(available.size))
            candidates = rng.choice(available, size=take, replace=False).astype(int)
            indices.update(int(idx) for idx in candidates)
    return sorted(indices)


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

    def validate(self) -> None:  # pragma: no cover - placeholder for future invariants
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
    gpu_device_id: int | None = None

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
            gpu_device_id=int(payload["gpu_device_id"]) if "gpu_device_id" in payload and payload["gpu_device_id"] is not None else None,
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
        if self.gpu_device_id is not None and self.gpu_device_id < -1:
            raise ValueError("env.gpu_device_id must be >= -1 when specified")


class YCBHabitatAdapter:
    """Facade that dispatches to the configured environment backend."""

    def __init__(self, config: EnvConfig, validator: PacketValidator, rng: np.random.Generator) -> None:
        if config.backend == "stub":
            self._impl: _AdapterProtocol = _StubAdapter(config, validator, rng)
        elif config.backend == "habitat":
            self._impl = _HabitatAdapter(config, validator, rng)
        else:  # pragma: no cover - validated earlier
            raise ValueError(f"Unsupported backend: {config.backend}")

    def reset(self) -> Tuple[Dict[str, object], Dict[str, object], List[int]]:
        return self._impl.reset()

    def step(self, action_packet: Dict[str, object]) -> Tuple[Dict[str, object], Dict[str, object], List[int]]:
        return self._impl.step(action_packet)

    def close(self) -> None:
        self._impl.close()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass


class _AdapterProtocol:
    """Protocol describing the minimal backend surface."""

    def reset(self) -> Tuple[Dict[str, object], Dict[str, object], List[int]]:  # pragma: no cover - interface only
        raise NotImplementedError

    def step(self, action_packet: Dict[str, object]) -> Tuple[Dict[str, object], Dict[str, object], List[int]]:  # pragma: no cover - interface only
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface only
        raise NotImplementedError


class _StubAdapter(_AdapterProtocol):
    """Contract-focused stub backend used for tests and CI."""

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
        self._tick = 0
        self._current_object_idx = int(self._rng.integers(len(self._config.objects)))
        self._last_switch = False
        self._pose_prev = dict(self._pose)
        return self._emit_packets()

    def step(self, action_packet: Dict[str, object]) -> Tuple[Dict[str, object], Dict[str, object], List[int]]:
        action_type = action_packet.get("action_type", "noop")
        params = action_packet.get("params", {}) if isinstance(action_packet, dict) else {}
        self._apply_action(str(action_type), params if isinstance(params, dict) else {})
        self._tick += 1
        self._last_switch = action_type == "switch_object"
        return self._emit_packets()

    def close(self) -> None:  # pragma: no cover - nothing to release for stub
        return

    def _emit_packets(self) -> Tuple[Dict[str, object], Dict[str, object], List[int]]:
        observation = self._make_observation_packet()
        pose = self._make_pose_packet()
        context_indices = _sample_context_indices(self._rng, self._tick, self._config.context_length, self._last_switch)
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
                    "view_id": f"{column_id}:stub",
                    "patch": {
                        "dtype": "uint8",
                        "shape": list(self._config.patch_shape),
                        "storage": f"shm://obs/{column_id}/{self._tick:06d}",
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


class _HabitatAdapter(_AdapterProtocol):
    """Real Habitat-Sim backed environment."""

    def __init__(self, config: EnvConfig, validator: PacketValidator, rng: np.random.Generator) -> None:
        if habitat_sim is None:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "Habitat backend requested but habitat-sim is not installed. "
                "Install dependencies from environment.yml to enable it."
            )

        if len(config.columns) != 1:
            raise ValueError("Habitat backend currently supports exactly one column for the Examiner scenario.")

        self._config = config
        self._validator = validator
        self._rng = rng
        self._tick = 0
        self._last_switch = False
        self._column_id = config.columns[0]
        initial = (float(self._rng.random()), float(self._rng.random()))
        self._pose: Dict[str, Tuple[float, float]] = {self._column_id: initial}
        self._pose_prev: Dict[str, Tuple[float, float]] = {self._column_id: initial}
        self._observation_cache: Dict[str, Dict[str, np.ndarray]] = {}

        cuda_enabled = bool(getattr(habitat_sim, "cuda_enabled", False))
        if config.gpu_device_id is not None:
            self._gpu_device_id = int(config.gpu_device_id)
        else:
            self._gpu_device_id = 0 if cuda_enabled else -1
        if self._gpu_device_id == -1 and cuda_enabled:
            # Explicit override may request CPU even if CUDA is present.
            pass
        elif self._gpu_device_id != -1 and not cuda_enabled:
            # Informative message to clarify the automatic fallback.
            print(
                "Habitat-Sim reports CUDA disabled; forcing gpu_device_id=-1 (CPU renderer)."
            )
            self._gpu_device_id = -1
        self._sim = self._create_simulator()
        self._rigid_object_mgr = self._sim.get_rigid_object_manager()
        self._agent = self._sim.get_agent(0)
        self._object_templates = self._preload_object_templates()
        self._object_id: int | None = None
        self._current_object_idx = 0

    def reset(self) -> Tuple[Dict[str, object], Dict[str, object], List[int]]:
        self._tick = 0
        self._last_switch = False
        if self._config.objects:
            self._current_object_idx = int(self._rng.integers(len(self._config.objects)))
        self._load_object(self._config.objects[self._current_object_idx])
        start_u = float(self._rng.random())
        start_v = float(self._rng.random())
        self._pose[self._column_id] = (start_u, start_v)
        self._pose_prev[self._column_id] = (start_u, start_v)
        self._apply_camera_state()
        return self._emit_packets()

    def step(self, action_packet: Dict[str, object]) -> Tuple[Dict[str, object], Dict[str, object], List[int]]:
        action_type = str(action_packet.get("action_type", "noop"))
        params = action_packet.get("params", {}) if isinstance(action_packet, dict) else {}
        if not isinstance(params, dict):
            params = {}
        self._apply_action(action_type, params)
        self._tick += 1
        self._last_switch = action_type == "switch_object"
        if self._config.physics.enable_physics:
            self._sim.step_physics(self._config.dt)
        return self._emit_packets()

    def close(self) -> None:
        if getattr(self, "_sim", None) is not None:
            self._sim.close()
            self._sim = None  # type: ignore[assignment]

    def _emit_packets(self) -> Tuple[Dict[str, object], Dict[str, object], List[int]]:
        observation = self._make_observation_packet()
        pose = self._make_pose_packet()
        context_indices = _sample_context_indices(self._rng, self._tick, self._config.context_length, self._last_switch)
        self._validator.validate(observation)
        self._validator.validate(pose)
        return observation, pose, context_indices

    def _create_simulator(self):
        scene_path = (self._config.assets.scene_root / self._config.assets.examiner_scene).expanduser().resolve()
        if not scene_path.exists():
            raise FileNotFoundError(f"Habitat scene not found: {scene_path}")

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = str(scene_path)
        sim_cfg.enable_physics = bool(self._config.physics.enable_physics)
        sim_cfg.gpu_device_id = int(self._gpu_device_id)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        sensor_specs = []
        for modality in self._config.sensor.modalities:
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid = f"{self._column_id}_{modality}"
            spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            spec.resolution = list(self._config.sensor.resolution)
            spec.hfov = self._config.sensor.hfov
            spec.near = self._config.sensor.near
            spec.far = self._config.sensor.far
            spec.position = [0.0, 0.0, 0.0]
            if modality == "rgb":
                spec.sensor_type = habitat_sim.SensorType.COLOR
                if hasattr(spec, "channels"):
                    spec.channels = 4
            elif modality == "depth":
                spec.sensor_type = habitat_sim.SensorType.DEPTH
            elif modality == "semantic":
                spec.sensor_type = habitat_sim.SensorType.SEMANTIC
            else:
                raise ValueError(f"Unsupported sensor modality '{modality}' for Habitat backend.")
            sensor_specs.append(spec)
        agent_cfg.sensor_specifications = sensor_specs

        habitat_cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        return habitat_sim.Simulator(habitat_cfg)

    def _preload_object_templates(self) -> Dict[str, str]:
        template_mgr = self._sim.get_object_template_manager()
        handles: Dict[str, str] = {}
        for object_id in self._config.objects:
            ycb_root = self._config.assets.ycb_root.expanduser().resolve()
            object_dir = (ycb_root / object_id).resolve()
            config_dir = (ycb_root / "configs").resolve()
            candidate_paths: List[Path] = []
            if object_dir.exists():
                candidate_paths.append(object_dir)
            config_path = config_dir / f"{object_id}.object_config.json"
            if config_path.exists():
                candidate_paths.append(config_path)
            if not candidate_paths:
                raise FileNotFoundError(
                    f"YCB asset config not found for object '{object_id}'. Expected directory "
                    f"{object_dir} or config file {config_path}."
                )

            template_handle: str | None = None
            for candidate in candidate_paths:
                loaded_ids = template_mgr.load_configs(str(candidate))
                if loaded_ids:
                    handle = template_mgr.get_template_handle_by_id(loaded_ids[0])
                    if handle:
                        template_handle = handle
                        break
            if template_handle is None:
                locations = ", ".join(str(path) for path in candidate_paths)
                raise FileNotFoundError(
                    f"Habitat failed to load object template for {object_id} from {locations}."
                )
            handles[object_id] = template_handle
        return handles

    def _load_object(self, object_id: str) -> None:
        if self._object_id is not None:
            try:
                self._rigid_object_mgr.remove_object_by_id(self._object_id)
            except Exception:
                pass
            self._object_id = None

        template_handle = self._object_templates[object_id]
        rigid_obj = self._rigid_object_mgr.add_object_by_template_handle(template_handle)
        if rigid_obj is None:
            raise RuntimeError(f"Failed to instantiate object template '{template_handle}'")
        rigid_obj.translation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if habitat_physics is not None and (not self._config.physics.enable_physics or self._config.physics.lock_object):
            rigid_obj.motion_type = habitat_physics.MotionType.STATIC
        self._object_id = rigid_obj.object_id

    def _apply_action(self, action_type: str, params: Dict[str, object]) -> None:
        if action_type == "move":
            dx = float(params.get("dx", 0.0))
            dy = float(params.get("dy", 0.0))
            u, v = self._pose[self._column_id]
            speed = float(self._config.orbit.default_speed)
            self._pose_prev[self._column_id] = (u, v)
            self._pose[self._column_id] = ((u + speed * dx) % 1.0, (v + speed * dy) % 1.0)
            self._apply_camera_state()
        elif action_type == "jump_to":
            target_u = float(params.get("u", self._rng.random()))
            target_v = float(params.get("v", self._rng.random()))
            u, v = self._pose[self._column_id]
            self._pose_prev[self._column_id] = (u, v)
            self._pose[self._column_id] = (target_u % 1.0, target_v % 1.0)
            self._apply_camera_state()
        elif action_type == "switch_object":
            self._current_object_idx = (self._current_object_idx + 1) % len(self._config.objects)
            self._load_object(self._config.objects[self._current_object_idx])
            u, v = self._pose[self._column_id]
            self._pose_prev[self._column_id] = (u, v)
            self._pose[self._column_id] = (float(self._rng.random()), float(self._rng.random()))
            self._apply_camera_state()

    def _apply_camera_state(self) -> None:
        u, v = self._pose[self._column_id]
        azimuth = 2.0 * math.pi * u
        elevation = self._config.orbit.min_elevation + (
            self._config.orbit.max_elevation - self._config.orbit.min_elevation
        ) * v
        radius = self._config.orbit.radius

        cos_elev = math.cos(elevation)
        position = np.array(
            [
                radius * cos_elev * math.cos(azimuth),
                radius * math.sin(elevation),
                radius * cos_elev * math.sin(azimuth),
            ],
            dtype=np.float32,
        )

        target = np.zeros(3, dtype=np.float32)
        rotation = self._compute_camera_quaternion(position, target)
        state = self._agent.get_state()
        state.position = position
        state.rotation = rotation
        self._agent.set_state(state)

    def _compute_camera_quaternion(self, position: np.ndarray, target: np.ndarray):
        if habitat_utils is None:  # pragma: no cover - defensive guard
            raise RuntimeError("Habitat utilities unavailable for quaternion computation")

        direction = target - position
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            return habitat_utils.quaternion_from_coeffs(np.array([1.0, 0.0, 0.0, 0.0]))
        direction /= norm

        default_forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        quat_forward = habitat_utils.quat_from_two_vectors(default_forward, direction)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        rotated_up = habitat_utils.quat_rotate_vector(quat_forward, world_up)
        correction = habitat_utils.quat_from_two_vectors(rotated_up, world_up)
        return correction * quat_forward

    def _make_observation_packet(self) -> Dict[str, object]:
        observations = self._sim.get_sensor_observations()
        cache_entry: Dict[str, np.ndarray] = {}
        columns_payload = []
        for modality in self._config.sensor.modalities:
            sensor_uuid = f"{self._column_id}_{modality}"
            if sensor_uuid in observations:
                cache_entry[modality] = observations[sensor_uuid]
        storage_key = f"habitat://{self._column_id}/{self._tick:06d}"
        if cache_entry:
            self._observation_cache[storage_key] = cache_entry

        u, v = self._pose[self._column_id]
        u_prev, v_prev = self._pose_prev[self._column_id]
        columns_payload.append(
            {
                "column_id": self._column_id,
                "view_id": f"{self._column_id}:habitat",
                "patch": {
                    "dtype": "uint8",
                    "shape": list(self._config.patch_shape),
                    "storage": storage_key,
                },
                "channels": list(self._config.sensor.modalities),
                "egopose": {"u": u, "v": v, "u_prev": u_prev, "v_prev": v_prev},
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
        u, v = self._pose[self._column_id]
        u_prev, v_prev = self._pose_prev[self._column_id]
        pose_packet = {
            "type": "pose.v1",
            "per_column": [
                {
                    "column_id": self._column_id,
                    "pose_t": {"u": u, "v": v},
                    "pose_tm1": {"u": u_prev, "v": v_prev},
                }
            ],
            "dt": self._config.dt,
        }
        return pose_packet

