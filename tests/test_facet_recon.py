"""Facet synthesizer smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.facet import FacetConfig, FacetSynthesizer
from src.utils import PacketValidator


def test_facet_records_validate() -> None:
    validator = PacketValidator()
    synthesizer = FacetSynthesizer(FacetConfig(topk_shared=2, facet_shape=(16, 16)), validator)
    observation_packet = {
        "type": "observation.v1",
        "columns": [
            {
                "column_id": "col0",
                "view_id": "stub",
                "patch": {"dtype": "uint8", "shape": [64, 64, 4], "storage": "shm://obs/col0/000000"},
                "channels": ["rgb"],
                "egopose": {"u": 0.1, "v": 0.2, "u_prev": 0.1, "v_prev": 0.2},
            }
        ],
        "global_meta": {"object_id": "stub_object", "tick": 5, "camera_intr": [1, 1, 1, 1]},
    }
    records = synthesizer.predict([3, 7, 11], observation_packet)
    assert len(records) == 2
    for record in records:
        assert record["type"] == "facet.v1"
        assert record["pred"]["shape"] == [16, 16]
