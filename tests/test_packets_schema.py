"""Schema validation tests mirroring docs/packets.md examples."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import PacketValidator


@pytest.fixture(scope="module")
def validator() -> PacketValidator:
    return PacketValidator()


def _observation_packet() -> dict:
    return {
        "type": "observation.v1",
        "columns": [
            {
                "column_id": "col0",
                "view_id": "obj_008:orbitA",
                "patch": {
                    "dtype": "uint8",
                    "shape": [64, 64, 4],
                    "storage": "shm://obs/col0/000123",
                },
                "channels": ["rgb", "depth"],
                "egopose": {"u": 0.314, "v": 0.812, "u_prev": 0.280, "v_prev": 0.812},
            }
        ],
        "global_meta": {
            "object_id": "obj_008_pudding_box",
            "tick": 123,
            "camera_intr": [575.8, 575.8, 320.0, 240.0],
        },
    }


def _context_packet() -> dict:
    return {
        "type": "context.v1",
        "c_bits": {"indices": [0, 1, 67], "length": 1024},
        "sources": ["metronome", "switch", "intent"],
        "annotations": {"metronome_bit": 0, "switch_bit": 1},
    }


def _pose_packet() -> dict:
    return {
        "type": "pose.v1",
        "per_column": [
            {
                "column_id": "col0",
                "pose_t": {"u": 0.314, "v": 0.812},
                "pose_tm1": {"u": 0.280, "v": 0.812},
            }
        ],
        "dt": 0.05,
    }


def _action_packet() -> dict:
    return {
        "type": "action.v1",
        "action_type": "move",
        "params": {"dx": 1.0, "dy": 0.0},
        "intent_bits": {"indices": [3], "length": 1024},
    }


def _belief_packet() -> dict:
    return {
        "type": "belief.v1",
        "g_star_logits": {"dtype": "float32", "shape": [2048], "storage": "shm://belief/g_star/000123"},
        "g_star_sdr": {"indices": [12, 97, 402, 955, 1311], "length": 2048},
        "entropy": 0.78,
        "peakiness": 0.62,
        "per_column": {
            "col0": {
                "g_post_logits": {
                    "dtype": "float32",
                    "shape": [2048],
                    "storage": "shm://belief/col0/g_post/000123",
                },
                "g_sdr": {"indices": [12, 97, 955], "length": 2048},
                "f_sdr": {"indices": [44, 203, 511, 912], "length": 4096},
            }
        },
        "c_sdr": {"indices": [0, 1, 67], "length": 1024},
    }


def _facet_record() -> dict:
    return {
        "type": "facet.v1",
        "phase_idx": 955,
        "coords_uv": [0.314, 0.812],
        "pred": {"dtype": "float32", "shape": [32, 32], "storage": "shm://facet/pred/000123"},
        "gt": {"dtype": "float32", "shape": [32, 32], "storage": "shm://facet/gt/000123"},
        "losses": {"L1": 0.042, "PSNR": 27.9},
    }


def _eval_record() -> dict:
    return {
        "type": "eval.v1",
        "episode_id": "2025-10-07T13:45:12Z#seed42",
        "metrics": {
            "entropy_mean": 0.81,
            "peakiness_mean": 0.54,
            "facet_L1_mean": 0.051,
            "coverage": 0.37,
        },
        "series": {
            "entropy_ts": {
                "dtype": "float32",
                "shape": [500],
                "storage": "file://runs/exp1/entropy.bin",
            }
        },
    }


@pytest.mark.parametrize(
    "packet_factory",
    [
        _observation_packet,
        _context_packet,
        _pose_packet,
        _action_packet,
        _belief_packet,
        _facet_record,
        _eval_record,
    ],
)
def test_packet_examples_validate(validator: PacketValidator, packet_factory) -> None:
    validator.validate(packet_factory())
