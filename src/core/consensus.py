"""Multi-column consensus using entropy-weighted product of experts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class ConsensusConfig:
    temperature: float = 1.0
    shared_topk: int = 32


class ConsensusSystem:
    """Aggregate per-column logits into a shared hypothesis."""

    def __init__(self, config: ConsensusConfig) -> None:
        self._config = config

    def fuse(self, per_column: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, float]]:
        if not per_column:
            raise ValueError("per_column logits cannot be empty")
        entropies = {column_id: self._entropy(logits) for column_id, logits in per_column.items()}
        weights = self._softmax({cid: -self._config.temperature * H for cid, H in entropies.items()})
        combined = sum(weights[cid] * per_column[cid] for cid in per_column)
        return combined, weights

    @staticmethod
    def _entropy(logits: np.ndarray) -> float:
        shifted = logits - logits.max()
        probs = np.exp(shifted)
        probs /= probs.sum()
        return float(-(probs * np.log(probs + 1e-8)).sum())

    @staticmethod
    def _softmax(weight_map: Dict[str, float]) -> Dict[str, float]:
        values = np.array(list(weight_map.values()), dtype=np.float32)
        shifted = values - values.max()
        exp_values = np.exp(shifted)
        probs = exp_values / exp_values.sum()
        return {key: float(prob) for key, prob in zip(weight_map.keys(), probs)}
