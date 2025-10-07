"""Grid-module style path integration stub used by the SDR columns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class PhaseConfig:
    phase_dim: int = 2048
    topk: int = 32
    noise_scale: float = 0.05


@dataclass
class _PhaseState:
    logits: np.ndarray
    sdr_indices: List[int]
    pose_uv: Tuple[float, float]


@dataclass
class PhaseOutput:
    prior_logits: np.ndarray
    prev_sdr: List[int]


class PhaseIntegrator:
    """Track allocentric phase per column with simple velocity updates."""

    def __init__(self, config: PhaseConfig, rng: np.random.Generator) -> None:
        self._config = config
        self._rng = rng
        self._states: Dict[str, _PhaseState] = {}

    def reset(self, columns: Iterable[str]) -> None:
        for column_id in columns:
            logits = self._rng.normal(loc=0.0, scale=1.0, size=self._config.phase_dim).astype(np.float32)
            indices = self._topk_indices(logits)
            self._states[column_id] = _PhaseState(logits=logits, sdr_indices=indices, pose_uv=(0.0, 0.0))

    def ensure_columns(self, columns: Iterable[str]) -> None:
        missing = [column for column in columns if column not in self._states]
        if missing:
            self.reset(missing)

    def step(self, pose_packet: Dict[str, object]) -> Dict[str, PhaseOutput]:
        outputs: Dict[str, PhaseOutput] = {}
        for entry in pose_packet.get("per_column", []):
            column_id = str(entry["column_id"])
            pose_t = (float(entry["pose_t"]["u"]), float(entry["pose_t"]["v"]))
            pose_tm1 = (float(entry["pose_tm1"]["u"]), float(entry["pose_tm1"]["v"]))
            state = self._states.get(column_id)
            if state is None:
                logits = self._rng.normal(loc=0.0, scale=1.0, size=self._config.phase_dim).astype(np.float32)
                indices = self._topk_indices(logits)
                state = _PhaseState(logits=logits, sdr_indices=indices, pose_uv=pose_tm1)
            velocity = self._torus_delta(pose_t, pose_tm1)
            drift = velocity[0] + velocity[1]
            noise = self._rng.normal(loc=0.0, scale=self._config.noise_scale, size=self._config.phase_dim).astype(
                np.float32
            )
            logits = state.logits + drift + noise
            outputs[column_id] = PhaseOutput(prior_logits=logits.copy(), prev_sdr=list(state.sdr_indices))
            self._states[column_id] = _PhaseState(logits=logits, sdr_indices=list(state.sdr_indices), pose_uv=pose_t)
        return outputs

    def commit(self, column_id: str, logits: np.ndarray, sdr_indices: List[int]) -> None:
        pose = self._states.get(column_id)
        pose_uv = pose.pose_uv if pose else (0.0, 0.0)
        self._states[column_id] = _PhaseState(logits=logits.astype(np.float32), sdr_indices=list(sdr_indices), pose_uv=pose_uv)

    def _topk_indices(self, logits: np.ndarray) -> List[int]:
        k = max(1, min(self._config.topk, logits.size))
        topk = np.argpartition(logits, -k)[-k:]
        ordered = topk[np.argsort(-logits[topk])]
        return ordered.astype(int).tolist()

    @staticmethod
    def _torus_delta(pose_t: Tuple[float, float], pose_tm1: Tuple[float, float]) -> Tuple[float, float]:
        du = ((pose_t[0] - pose_tm1[0] + 0.5) % 1.0) - 0.5
        dv = ((pose_t[1] - pose_tm1[1] + 0.5) % 1.0) - 0.5
        return du, dv
