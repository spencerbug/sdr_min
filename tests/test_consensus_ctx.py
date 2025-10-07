"""Consensus fusion tests covering entropy weighting."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.consensus import ConsensusConfig, ConsensusSystem


def test_consensus_entropy_weights() -> None:
    config = ConsensusConfig(temperature=1.0)
    consensus = ConsensusSystem(config)
    low_entropy_logits = np.array([5.0, -5.0, 5.0, -5.0], dtype=np.float32)
    high_entropy_logits = np.zeros(4, dtype=np.float32)
    shared, weights = consensus.fuse({"low": low_entropy_logits, "high": high_entropy_logits})
    assert shared.shape == low_entropy_logits.shape
    assert weights["low"] > weights["high"]
    assert np.isfinite(shared).all()
    assert abs(sum(weights.values()) - 1.0) < 1e-6
