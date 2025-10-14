"""Column module implementing the intra-column SDR pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from ..utils import PacketValidator
from .assoc import AssociativeConfig, AssociativeMaps
from .belief import BeliefBuilder, BeliefConfig
from .consensus import ConsensusConfig, ConsensusSystem
from .context_gates import ContextGateConfig, ContextGates
from .encoders import ContextConfig, FeatureEncoder, FeatureEncoderConfig
from .facet import FacetConfig, FacetSynthesizer
from .fusion import FusionConfig, PoEFuser
from .phase import PhaseConfig, PhaseIntegrator
from .sensor import SensorAdapter, SensorConfig


@dataclass
class ColumnConfig:
    phase_dim: int = 2048
    feature_dim: int = 4096
    topk_phase: int = 32
    topk_feature: int = 64
    fusion: FusionConfig = field(default_factory=FusionConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    facet: FacetConfig = field(default_factory=FacetConfig)


@dataclass
class ColumnStepResult:
    belief_packet: Dict[str, object]
    facet_records: List[Dict[str, object]]
    consensus_weights: Dict[str, float]


class ColumnSystem:
    """Run feature/context fusion, consensus, and belief building."""

    def __init__(
        self,
        config: ColumnConfig,
        context_config: ContextConfig,
        validator: PacketValidator,
        rng: np.random.Generator,
    ) -> None:
        self._config = config
        self._validator = validator
        self._rng = rng
        self._sensor = SensorAdapter(SensorConfig(feature_dim=config.feature_dim))
        self._feature_encoder = FeatureEncoder(
            FeatureEncoderConfig(feature_dim=config.feature_dim, topk=config.topk_feature)
        )
        self._phase = PhaseIntegrator(PhaseConfig(phase_dim=config.phase_dim, topk=config.topk_phase), rng)
        assoc_config = AssociativeConfig(
            phase_dim=config.phase_dim,
            feature_dim=config.feature_dim,
            context_dim=context_config.length,
            topk_phase=config.topk_phase,
            topk_feature=config.topk_feature,
        )
        self._assoc = AssociativeMaps(assoc_config, rng)
        self._context_gates = ContextGates(ContextGateConfig(length=context_config.length))
        self._fuser = PoEFuser(config.fusion)
        self._consensus = ConsensusSystem(config.consensus)
        belief_topk = min(config.topk_phase, config.consensus.shared_topk)
        self._belief = BeliefBuilder(BeliefConfig(phase_dim=config.phase_dim, topk_phase=belief_topk), validator)
        self._facet = FacetSynthesizer(config.facet, validator)

    def reset(self, columns: List[str]) -> None:
        self._phase.reset(columns)

    def step(
        self,
        observation_packet: Dict[str, object],
        pose_packet: Dict[str, object],
        context_packet: Dict[str, object],
    ) -> ColumnStepResult:
        column_ids = [str(col["column_id"]) for col in observation_packet.get("columns", [])]
        self._phase.ensure_columns(column_ids)

        phase_outputs = self._phase.step(pose_packet)
        feature_vectors = self._sensor.extract(observation_packet)
        c_bits = context_packet.get("c_bits", {})
        context_indices = c_bits.get("indices", [])
        gains = self._context_gates.update(c_bits)
        scaled_context_indices = self._context_gates.scale_indices(gains, context_indices)
        context_logits_phase = self._assoc.phase_from_context(scaled_context_indices)
        per_column_logits: Dict[str, np.ndarray] = {}
        per_column_sdr: Dict[str, Dict[str, Dict[str, object]]] = {}

        for column in observation_packet.get("columns", []):
            column_id = str(column["column_id"])
            feature_dense = feature_vectors[column_id]
            feature_sdr = self._feature_encoder.encode(feature_dense)
            phase_output = phase_outputs.get(column_id)
            if phase_output is None:
                prior_logits = np.zeros(self._config.phase_dim, dtype=np.float32)
            else:
                prior_logits = phase_output.prior_logits
            g_prior = prior_logits
            g_feat = self._assoc.phase_from_features(feature_sdr["indices"])
            g_post = self._fuser.fuse(g_prior, g_feat, context_logits_phase)
            g_indices = self._topk_indices(g_post, self._config.topk_phase)
            per_column_logits[column_id] = g_post
            per_column_sdr[column_id] = {
                "g_sdr": {"indices": g_indices, "length": self._config.phase_dim},
                "f_sdr": feature_sdr,
            }
            self._phase.commit(column_id, g_post, g_indices)
            self._assoc.update(g_indices, feature_sdr["indices"], scaled_context_indices)

        shared_logits, weights = self._consensus.fuse(per_column_logits)
        belief_packet = self._belief.build(shared_logits, per_column_logits, per_column_sdr, context_packet)
        facet_records = self._facet.predict(belief_packet["g_star_sdr"]["indices"], observation_packet)
        return ColumnStepResult(
            belief_packet=belief_packet,
            facet_records=facet_records,
            consensus_weights=weights,
        )

    def _topk_indices(self, logits: np.ndarray, k: int) -> List[int]:
        k = max(1, min(k, logits.size))
        topk = np.argpartition(logits, -k)[-k:]
        ordered = topk[np.argsort(-logits[topk])]
        return ordered.astype(int).tolist()
