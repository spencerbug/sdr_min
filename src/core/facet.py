"""Counterfactual facet reconstruction stub."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from ..utils import PacketValidator


@dataclass
class FacetConfig:
    facet_shape: Sequence[int] = (32, 32)
    topk_shared: int = 3


class FacetSynthesizer:
    """Produce schema-compliant facet records for top shared hypotheses."""

    def __init__(self, config: FacetConfig, validator: PacketValidator) -> None:
        self._config = config
        self._validator = validator
        self._counter = 0

    def predict(self, shared_indices: List[int], observation_packet: Dict[str, object]) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        tick = int(observation_packet.get("global_meta", {}).get("tick", 0))
        for phase_idx in shared_indices[: self._config.topk_shared]:
            coords = self._index_to_uv(phase_idx)
            losses = self._synthetic_losses(phase_idx, tick)
            record = {
                "type": "facet.v1",
                "phase_idx": int(phase_idx),
                "coords_uv": [coords[0], coords[1]],
                "pred": self._handle("pred"),
                "gt": self._handle("gt"),
                "losses": losses,
            }
            self._validator.validate(record)
            records.append(record)
        return records

    def _handle(self, prefix: str) -> Dict[str, object]:
        self._counter += 1
        return {
            "dtype": "float32",
            "shape": list(self._config.facet_shape),
            "storage": f"shm://facet/{prefix}/{self._counter:06d}",
        }

    def _index_to_uv(self, phase_idx: int) -> List[float]:
        base = (phase_idx % 1024) / 1024.0
        return [float(base), float((phase_idx // 1024) % 1024 / 1024.0)]

    @staticmethod
    def _synthetic_losses(phase_idx: int, tick: int) -> Dict[str, float]:
        return {
            "L1": float(((phase_idx % 17) + tick % 5) / 100.0),
            "PSNR": float(25.0 + (phase_idx % 11)),
        }
