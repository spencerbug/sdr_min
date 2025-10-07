"""Product-of-experts fusion for combining priors, features, and context."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FusionConfig:
    alpha: float = 0.6
    beta_feature: float = 0.3
    beta_context: float = 0.1
    clip_min: float = -5.0
    clip_max: float = 5.0


class PoEFuser:
    """Blend logits via weighted sum followed by clipping."""

    def __init__(self, config: FusionConfig) -> None:
        self._config = config

    def fuse(self, prior: np.ndarray, feature: np.ndarray, context: np.ndarray) -> np.ndarray:
        logits = (
            self._config.alpha * prior
            + self._config.beta_feature * feature
            + self._config.beta_context * context
        )
        return np.clip(logits, self._config.clip_min, self._config.clip_max)
