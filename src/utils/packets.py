"""Helpers for loading and validating packet schemas.

This module centralises JSON Schema loading so every packet emitted
by the loop validates against the canonical contracts declared in
``docs/packets.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Mapping

from jsonschema import Draft7Validator


@dataclass(frozen=True)
class _SchemaRecord:
    """Container for compiled schema validators."""

    name: str
    validator: Draft7Validator


class PacketValidator:
    """Validate packets against the repository JSON Schemas.

    The validator infers the schema file from the packet ``type`` field,
    e.g. ``observation.v1`` → ``contracts/observation.schema.json``.
    """

    def __init__(self) -> None:
        contracts_dir = Path(__file__).resolve().parents[2] / "contracts"
        if not contracts_dir.exists():
            raise FileNotFoundError("Expected contracts/ directory next to repo root.")
        self._validators: Dict[str, _SchemaRecord] = {}
        for schema_path in contracts_dir.glob("*.schema.json"):
            with schema_path.open("r", encoding="utf-8") as handle:
                schema = json.load(handle)
            name = schema_path.stem.replace(".schema", "")
            validator = Draft7Validator(schema)
            self._validators[name] = _SchemaRecord(name=name, validator=validator)

    def validate(self, packet: Mapping[str, object]) -> None:
        """Validate a packet by inspecting its ``type`` discriminator."""

        packet_type = packet.get("type")
        if not packet_type or not isinstance(packet_type, str):
            raise ValueError("Packet missing string 'type' field for schema lookup.")
        logical_name = packet_type.split(".", 1)[0]
        record = self._validators.get(logical_name)
        if record is None:
            raise KeyError(f"No schema registered for packet type '{packet_type}'.")
        record.validator.validate(packet)

    def get_validator(self, logical_name: str) -> Draft7Validator:
        """Return a compiled validator for advanced checks (tests)."""

        record = self._validators.get(logical_name)
        if record is None:
            raise KeyError(f"Unknown schema '{logical_name}'.")
        return record.validator

    @property
    def schema_names(self) -> Dict[str, Draft7Validator]:
        """Expose available schemas (mainly for tests)."""

        return {name: record.validator for name, record in self._validators.items()}
