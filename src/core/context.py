"""Context SDR encoder stub."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from ..utils import PacketValidator


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
