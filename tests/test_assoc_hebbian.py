"""Unit tests for associative Hebbian maps."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.assoc import AssociativeConfig, AssociativeMaps


def test_row_topk_enforced() -> None:
    config = AssociativeConfig(
        phase_dim=8,
        feature_dim=16,
        context_dim=4,
        topk_phase=3,
        topk_feature=3,
        lr=0.5,
    decay=0.0,
    seed=False,
    )
    rng = np.random.default_rng(0)
    maps = AssociativeMaps(config, rng)
    phase_indices = [0]
    maps.update(phase_indices, list(range(6)), context_indices=[])
    weights = maps.phase_feature_weights(0)
    assert len(weights) <= config.topk_feature
    maps.update(phase_indices, [10, 11, 12, 10, 11, 12], context_indices=[])
    weights_after = maps.phase_feature_weights(0)
    assert len(weights_after) <= config.topk_feature
    assert any(idx in weights_after for idx in (10, 11, 12))
    max_new = max(weights_after.get(idx, 0.0) for idx in (10, 11, 12))
    max_old = max(weights_after.get(idx, 0.0) for idx in range(6))
    assert max_new >= max_old


def test_context_projection_shape() -> None:
    config = AssociativeConfig(
        phase_dim=6,
        feature_dim=12,
        context_dim=8,
        topk_phase=2,
        topk_feature=3,
    seed=False,
    )
    rng = np.random.default_rng(1)
    maps = AssociativeMaps(config, rng)
    logits = maps.phase_from_context([0, 5])
    assert logits.shape == (config.phase_dim,)
    assert np.isfinite(logits).all()
