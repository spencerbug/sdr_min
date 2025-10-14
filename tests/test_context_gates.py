import numpy as np
import pytest

from src.core.context_gates import ContextGateConfig, ContextGates


def test_context_gates_update_returns_ones() -> None:
    gates = ContextGates(ContextGateConfig(length=8))
    gains = gates.update({"indices": [0, 3], "length": 8})
    assert gains.shape == (8,)
    assert np.allclose(gains, 1.0)


def test_context_gates_scale_indices_identity() -> None:
    gates = ContextGates(ContextGateConfig(length=8))
    gains = gates.update({"indices": [0, 3], "length": 8})
    scaled = gates.scale_indices(gains, [0, 7, 15])
    # 15 wraps around to 7 under modulo arithmetic
    assert scaled == [0, 7, 7]


def test_context_gates_scale_indices_length_mismatch() -> None:
    gates = ContextGates(ContextGateConfig(length=4))
    gains = np.ones(3, dtype=np.float32)
    with pytest.raises(ValueError):
        gates.scale_indices(gains, [0])
