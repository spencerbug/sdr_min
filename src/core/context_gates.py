"""Context gating helper for SDR columns.

The initial implementation exposes a static all-ones gate so the Examiner
MVP can rely on deterministic context scaling while keeping the API ready
for adaptive gains in later scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass
class ContextGateConfig:
    """Configuration for context gate controller."""

    length: int


class ContextGates:
    """Compute per-context slice gains.

    The launch build keeps the behaviour simple: every slice receives a gain of
    1.0, effectively behaving like an identity transform. The class retains a
    "`update` → gains → scale`" contract so future adaptive gating can drop in
    without touching the call sites.
    """

    def __init__(self, config: ContextGateConfig) -> None:
        self._config = config

    @property
    def length(self) -> int:
        return self._config.length

    def update(
        self,
        context_bits: Mapping[str, Sequence[int]] | Sequence[int],
        entropy: float | None = None,
    ) -> np.ndarray:
        """Return slice gains for the provided context bits.

        Parameters
        ----------
        context_bits:
            Either the ``c_bits`` mapping from the ContextPacket or a bare
            sequence of indices. The fallback implementation ignores the
            indices and simply returns ones.
        entropy:
            Optional belief entropy input; currently unused but part of the
            public signature for future adaptive strategies.
        """

        # Consumed inputs are intentionally unused for the static helper.
        _ = context_bits, entropy
        return np.ones(self._config.length, dtype=np.float32)

    def scale_indices(self, gains: np.ndarray, indices: Iterable[int]) -> list[int]:
        """Filter/transform context indices using the provided gains.

        For the identity helper we simply return the original indices after
        normalising them to the configured length. Once adaptive gating ships we
        can modulate the output here (e.g. drop slices with zero gain).
        """

        if gains.shape[0] != self._config.length:
            raise ValueError("Gate vector length mismatch with configuration.")
        normalised = []
        for raw_idx in indices:
            idx = int(raw_idx) % self._config.length
            if gains[idx] > 0.0:
                normalised.append(idx)
        return normalised

    def sparse_indices(self, gains: np.ndarray, indices: Iterable[int]) -> list[int]:
        """Alias for ``scale_indices`` retained for semantic clarity."""

        return self.scale_indices(gains, indices)
