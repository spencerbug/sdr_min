"""Column module stub matching the SDR design skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..utils import PacketValidator


@dataclass
class ColumnConfig:
    phase_dim: int = 2048
    feature_dim: int = 4096
    topk_phase: int = 32
    topk_feature: int = 64


class ColumnSystem:
    """Minimal multi-column processor with deterministic validation."""

    def __init__(self, config: ColumnConfig, validator: PacketValidator, rng: np.random.Generator) -> None:
        self._config = config
        self._validator = validator
        self._rng = rng

    def step(
        self,
        observation_packet: Dict[str, object],
        pose_packet: Dict[str, object],
        context_packet: Dict[str, object],
    ) -> Dict[str, object]:
        per_column = {}
        accumulated_logits = []
        for column in observation_packet["columns"]:
            column_id = column["column_id"]
            logits = self._rng.normal(loc=0.0, scale=1.0, size=self._config.phase_dim)
            accumulated_logits.append(logits)
            g_topk = self._topk_indices(logits, self._config.topk_phase)
            f_logits = self._rng.normal(loc=0.0, scale=1.0, size=self._config.feature_dim)
            f_topk = self._topk_indices(f_logits, self._config.topk_feature)
            per_column[column_id] = {
                "g_post_logits": self._handle("g_post", column_id, self._config.phase_dim),
                "g_sdr": {"indices": g_topk, "length": self._config.phase_dim},
                "f_sdr": {"indices": f_topk, "length": self._config.feature_dim},
            }
        shared_logits = np.mean(np.stack(accumulated_logits, axis=0), axis=0)
        g_star_indices = self._topk_indices(shared_logits, self._config.topk_phase)
        entropy, peakiness = self._compute_entropy(shared_logits)
        belief_packet = {
            "type": "belief.v1",
            "g_star_logits": self._handle("g_star", "shared", self._config.phase_dim),
            "g_star_sdr": {"indices": g_star_indices, "length": self._config.phase_dim},
            "entropy": entropy,
            "peakiness": peakiness,
            "per_column": per_column,
            "c_sdr": context_packet["c_bits"],
        }
        self._validator.validate(belief_packet)
        return belief_packet

    def _topk_indices(self, logits: np.ndarray, k: int) -> List[int]:
        k = min(k, logits.size)
        topk = np.argpartition(logits, -k)[-k:]
        return sorted(np.asarray(topk, dtype=int).tolist())

    def _handle(self, tensor_name: str, suffix: str, dim: int) -> Dict[str, object]:
        return {
            "dtype": "float32",
            "shape": [dim],
            "storage": f"shm://belief/{tensor_name}/{suffix}",
        }

    def _compute_entropy(self, logits: np.ndarray) -> Tuple[float, float]:
        shifted = logits - logits.max()
        exp_logits = np.exp(shifted)
        probs = exp_logits / exp_logits.sum()
        entropy = float(-(probs * np.log(probs + 1e-8)).sum())
        h_max = np.log(logits.size)
        peakiness = float((h_max - entropy) / h_max) if h_max > 0 else 0.0
        return entropy, peakiness
