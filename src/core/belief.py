"""Belief packet construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..utils import PacketValidator


@dataclass
class BeliefConfig:
    phase_dim: int
    topk_phase: int


class BeliefBuilder:
    """Create BeliefPacket dictionaries that satisfy the JSON Schema."""

    def __init__(self, config: BeliefConfig, validator: PacketValidator) -> None:
        self._config = config
        self._validator = validator
        self._storage_counter = 0

    def build(
        self,
        shared_logits: np.ndarray,
        per_column_logits: Dict[str, np.ndarray],
        per_column_sdr: Dict[str, Dict[str, object]],
        context_packet: Dict[str, object],
    ) -> Dict[str, object]:
        shared_indices = self._topk_indices(shared_logits)
        entropy, peakiness = self._entropy(shared_logits)
        packet = {
            "type": "belief.v1",
            "g_star_logits": self._handle("g_star"),
            "g_star_sdr": {"indices": shared_indices, "length": self._config.phase_dim},
            "entropy": entropy,
            "peakiness": peakiness,
            "per_column": {},
            "c_sdr": context_packet["c_bits"],
        }
        for column_id, logits in per_column_logits.items():
            packet["per_column"][column_id] = {
                "g_post_logits": self._handle(f"g_post/{column_id}"),
                "g_sdr": per_column_sdr[column_id]["g_sdr"],
                "f_sdr": per_column_sdr[column_id]["f_sdr"],
            }
        self._validator.validate(packet)
        return packet

    def _handle(self, suffix: str) -> Dict[str, object]:
        self._storage_counter += 1
        return {
            "dtype": "float32",
            "shape": [self._config.phase_dim],
            "storage": f"shm://belief/{suffix}/{self._storage_counter:06d}",
        }

    def _topk_indices(self, logits: np.ndarray) -> List[int]:
        k = max(1, min(self._config.topk_phase, logits.size))
        topk = np.argpartition(logits, -k)[-k:]
        ordered = topk[np.argsort(-logits[topk])]
        return ordered.astype(int).tolist()

    def _entropy(self, logits: np.ndarray) -> Tuple[float, float]:
        shifted = logits - logits.max()
        probs = np.exp(shifted)
        probs /= probs.sum()
        entropy = float(-(probs * np.log(probs + 1e-8)).sum())
        h_max = np.log(logits.size)
        peakiness = float((h_max - entropy) / h_max) if h_max > 0 else 0.0
        return entropy, peakiness
