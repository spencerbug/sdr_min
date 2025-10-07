"""Sensor adapter that maps observation patches to feature vectors.

The stub uses deterministic RNG seeds based on the column id and tick so tests
stay reproducible while keeping the contract-focused design from DESIGN.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class SensorConfig:
    """Hyper-parameters for the observation-to-feature adapter."""

    feature_dim: int = 4096


class SensorAdapter:
    """Produce dense feature vectors for each column observation."""

    def __init__(self, config: SensorConfig) -> None:
        self._config = config

    def extract(self, observation_packet: Dict[str, object]) -> Dict[str, np.ndarray]:
        """Return per-column feature vectors with deterministic randomness."""

        tick = int(observation_packet.get("global_meta", {}).get("tick", 0))
        features: Dict[str, np.ndarray] = {}
        for column in observation_packet.get("columns", []):
            column_id = str(column["column_id"])
            seed = self._seed_from_ids(column_id, tick)
            local_rng = np.random.default_rng(seed)
            features[column_id] = local_rng.normal(
                loc=0.0, scale=1.0, size=self._config.feature_dim
            ).astype(np.float32)
        return features

    @staticmethod
    def _seed_from_ids(column_id: str, tick: int) -> int:
        return (hash((column_id, tick)) & 0xFFFFFFFFFFFF) or 1
