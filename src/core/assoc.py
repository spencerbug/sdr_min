"""Sparse Hebbian associative maps with row-Top-K pruning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np


@dataclass
class AssociativeConfig:
    phase_dim: int
    feature_dim: int
    context_dim: int
    topk_phase: int
    topk_feature: int
    lr: float = 0.1
    decay: float = 0.02
    seed: bool = True


class _SparseHebbianMatrix:
    def __init__(self, rows: int, cols: int, topk: int) -> None:
        self.rows = rows
        self.cols = cols
        self.topk = topk
        self._rows: List[Dict[int, float]] = [dict() for _ in range(rows)]
        self._cols: List[Dict[int, float]] = [dict() for _ in range(cols)]

    def project_rows(self, active_cols: Sequence[int]) -> np.ndarray:
        result = np.zeros(self.rows, dtype=np.float32)
        for col in active_cols:
            if col >= self.cols:
                continue
            for row, weight in self._cols[col].items():
                result[row] += weight
        return result

    def project_cols(self, active_rows: Sequence[int]) -> np.ndarray:
        result = np.zeros(self.cols, dtype=np.float32)
        for row in active_rows:
            if row >= self.rows:
                continue
            for col, weight in self._rows[row].items():
                result[col] += weight
        return result

    def decay_rows(self, rows: Iterable[int], decay: float) -> None:
        for row in rows:
            if row >= self.rows:
                continue
            row_dict = self._rows[row]
            if not row_dict:
                continue
            for col in list(row_dict.keys()):
                new_weight = row_dict[col] * (1.0 - decay)
                if new_weight <= 1e-6:
                    del row_dict[col]
                    self._cols[col].pop(row, None)
                else:
                    row_dict[col] = new_weight
                    self._cols[col][row] = new_weight

    def update(self, rows: Sequence[int], cols: Sequence[int], lr: float, decay: float) -> None:
        self.decay_rows(rows, decay)
        for row in rows:
            if row >= self.rows:
                continue
            row_dict = self._rows[row]
            for col in cols:
                if col >= self.cols:
                    continue
                new_weight = row_dict.get(col, 0.0) + lr
                row_dict[col] = new_weight
                self._cols[col][row] = new_weight
            if len(row_dict) > self.topk:
                sorted_items = sorted(row_dict.items(), key=lambda kv: kv[1], reverse=True)[: self.topk]
                keep = dict(sorted_items)
                removed = set(row_dict.keys()) - set(keep.keys())
                self._rows[row] = keep
                for col in removed:
                    self._cols[col].pop(row, None)

    def seed_random(self, rng: np.random.Generator) -> None:
        for row in range(self.rows):
            if self.topk <= 0:
                continue
            sample_size = min(self.topk, self.cols)
            cols = rng.choice(self.cols, size=sample_size, replace=False)
            weights = rng.uniform(0.0, 1.0, size=sample_size)
            for col, weight in zip(cols, weights):
                self._rows[row][int(col)] = float(weight)
                self._cols[int(col)][row] = float(weight)


class AssociativeMaps:
    """Bundle of Hebbian maps used across the SDR pipeline."""

    def __init__(self, config: AssociativeConfig, rng: np.random.Generator) -> None:
        self._config = config
        self._phase_phase = _SparseHebbianMatrix(config.phase_dim, config.phase_dim, config.topk_phase)
        self._phase_feat = _SparseHebbianMatrix(config.phase_dim, config.feature_dim, config.topk_feature)
        self._phase_ctx = _SparseHebbianMatrix(config.phase_dim, config.context_dim, config.topk_phase)
        self._feat_ctx = _SparseHebbianMatrix(config.feature_dim, config.context_dim, config.topk_feature)
        if config.seed:
            self._phase_phase.seed_random(rng)
            self._phase_feat.seed_random(rng)
            self._phase_ctx.seed_random(rng)
            self._feat_ctx.seed_random(rng)
        self._rng = rng

    def phase_from_prior(self, phase_indices: Sequence[int]) -> np.ndarray:
        return self._phase_phase.project_rows(phase_indices)

    def phase_from_features(self, feature_indices: Sequence[int]) -> np.ndarray:
        return self._phase_feat.project_rows(feature_indices)

    def phase_from_context(self, context_indices: Sequence[int]) -> np.ndarray:
        return self._phase_ctx.project_rows(context_indices)

    def feature_from_phase(self, phase_indices: Sequence[int]) -> np.ndarray:
        return self._phase_feat.project_cols(phase_indices)

    def feature_from_context(self, context_indices: Sequence[int]) -> np.ndarray:
        return self._feat_ctx.project_rows(context_indices)

    def update(self, phase_indices: Sequence[int], feature_indices: Sequence[int], context_indices: Sequence[int]) -> None:
        self._phase_phase.update(phase_indices, phase_indices, self._config.lr, self._config.decay)
        self._phase_feat.update(phase_indices, feature_indices, self._config.lr, self._config.decay)
        self._phase_ctx.update(phase_indices, context_indices, self._config.lr, self._config.decay)
        self._feat_ctx.update(feature_indices, context_indices, self._config.lr / 2.0, self._config.decay)

    # Inspection helpers for tests -------------------------------------------------

    def phase_feature_weights(self, row: int) -> Dict[int, float]:
        if row >= self._config.phase_dim:
            raise IndexError("phase row out of range")
        return dict(self._phase_feat._rows[row])

    def phase_context_weights(self, row: int) -> Dict[int, float]:
        if row >= self._config.phase_dim:
            raise IndexError("phase row out of range")
        return dict(self._phase_ctx._rows[row])
