"""Feature and context encoders for the SDR pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np

from ..utils import PacketValidator


@dataclass
class FeatureEncoderConfig:
    feature_dim: int = 4096
    topk: int = 64


class FeatureEncoder:
    """Convert dense feature vectors into sparse SDR packets."""

    def __init__(self, config: FeatureEncoderConfig) -> None:
        self._config = config

    def encode(self, dense: np.ndarray) -> Dict[str, object]:
        if dense.size != self._config.feature_dim:
            raise ValueError(f"Expected dense feature dim {self._config.feature_dim}, got {dense.size}")
        indices = self._topk_indices(dense, self._config.topk)
        return {"indices": indices, "length": self._config.feature_dim}

    def _topk_indices(self, logits: np.ndarray, k: int) -> List[int]:
        k = max(1, min(k, logits.size))
        topk = np.argpartition(logits, -k)[-k:]
        ordered = topk[np.argsort(-logits[topk])]
        return ordered.astype(int).tolist()


@dataclass
class ContextConfig:
    length: int = 1024
    sources: Sequence[str] = ("metronome", "switch", "intent")


class ContextEncoder:
    """Convert raw context indices into a schema-compliant packet."""

    def __init__(self, config: ContextConfig, validator: PacketValidator) -> None:
        self._config = config
        self._validator = validator

    def encode(self, indices: Iterable[int]) -> Dict[str, object]:
        unique_sorted: List[int] = sorted({int(idx) % self._config.length for idx in indices})
        packet = {
            "type": "context.v1",
            "c_bits": {"indices": unique_sorted, "length": self._config.length},
            "sources": list(self._config.sources),
        }
        self._validator.validate(packet)
        return packet
