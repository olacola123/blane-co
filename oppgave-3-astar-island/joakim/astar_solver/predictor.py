"""Probabilistic map predictor with static/dynamic decomposition."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .config import SolverConfig
from .dynamics import LocalDynamicsRefiner
from .features import FeatureGrid, MapFeatureExtractor
from .latent import RoundLatentInferer
from .observations import RoundObservationStore
from .probability import apply_probability_floor, predictive_entropy, temperature_scale
from .relations import RelationArtifacts, SettlementRelationModule
from .types import RoundLatentState, SeedState


@dataclass(slots=True)
class PredictionArtifacts:
    """Prediction output plus intermediate diagnostics used elsewhere."""

    probabilities: np.ndarray
    base_probabilities: np.ndarray
    entropy_map: np.ndarray
    dynamic_mass: np.ndarray
    observation_weight: np.ndarray
    features: FeatureGrid
    relations: RelationArtifacts
    latent: RoundLatentState


class ProbabilisticMapPredictor:
    """Hybrid heuristic decoder with explicit uncertainty handling."""

    def __init__(self, config: SolverConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.feature_extractor = MapFeatureExtractor()
        self.latent_inferer = RoundLatentInferer(logger=self.logger)
        self.relation_module = SettlementRelationModule(logger=self.logger)
        self.local_refiner = LocalDynamicsRefiner(passes=config.local_dynamics_passes)

    def predict_seed(
        self,
        seed_state: SeedState,
        round_store: RoundObservationStore,
        features: FeatureGrid | None = None,
    ) -> PredictionArtifacts:
        """Predict a full 40x40x6 tensor for one seed."""
        feature_grid = features or self.feature_extractor.extract(seed_state)
        summary = round_store.build_summary()
        latent = self.latent_inferer.infer(summary)
        seed_observation = round_store.get_seed_memory(seed_state.seed_index)
        relations = self.relation_module.build(seed_state, feature_grid, seed_observation)

        static_component = self._build_static_component(feature_grid)
        dynamic_component = self._build_dynamic_component(feature_grid, relations, latent)
        dynamic_mass = self._build_dynamic_mass(feature_grid, relations, latent)
        base = (1.0 - dynamic_mass[..., None]) * static_component + dynamic_mass[..., None] * dynamic_component
        refined = self.local_refiner.refine(base, feature_grid, relations, latent)
        blended, observation_weight = self._blend_observations(refined, seed_observation)
        calibrated = temperature_scale(blended, self.config.probability.temperature)
        final = apply_probability_floor(calibrated, self.config.probability.floor)
        entropy = predictive_entropy(final)

        self.logger.info(
            "Prediction seed=%s coverage=%.3f mean_entropy=%.3f",
            seed_state.seed_index,
            seed_observation.coverage_ratio(),
            float(entropy.mean()),
        )
        return PredictionArtifacts(
            probabilities=final,
            base_probabilities=base,
            entropy_map=entropy,
            dynamic_mass=dynamic_mass,
            observation_weight=observation_weight,
            features=feature_grid,
            relations=relations,
            latent=latent,
        )

    def _build_static_component(self, features: FeatureGrid) -> np.ndarray:
        ocean = features.channel("terrain_ocean")
        plains_like = features.channel("plains_like")
        forest = features.channel("terrain_forest")
        mountain = features.channel("terrain_mountain")

        static = np.full(features.channels.shape[:2] + (6,), 1e-6, dtype=float)
        static[..., 0] = (
            2.6 * ocean
            + 1.8 * plains_like
            + 0.6 * features.channel("distance_settlement_norm")
            + 0.4 * (1.0 - features.channel("coastal_mask"))
        )
        static[..., 4] = (
            1.8 * forest
            + 0.8 * features.channel("forest_adj")
            + 0.4 * features.channel("local_forest_density")
            + 0.3 * (1.0 - features.channel("settlement_proximity"))
        )
        static[..., 5] = 6.0 * mountain + 0.02
        static = static / static.sum(axis=-1, keepdims=True)

        ocean_mask = ocean > 0.5
        mountain_mask = mountain > 0.5
        static[ocean_mask] = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        static[mountain_mask] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float)
        return static

    def _build_dynamic_component(
        self,
        features: FeatureGrid,
        relations: RelationArtifacts,
        latent: RoundLatentState,
    ) -> np.ndarray:
        latent_scale = self.config.latent_strength * max(latent.confidence, 0.25)
        settlement = features.channel("terrain_settlement")
        port = features.channel("terrain_port")
        ruin = features.channel("terrain_ruin")
        forest = features.channel("terrain_forest")

        dynamic = np.full(features.channels.shape[:2] + (6,), 1e-6, dtype=float)
        dynamic[..., 0] = (
            0.6
            + 0.5 * features.channel("plains_like")
            + 0.2 * features.channel("distance_mountain_norm")
            + 0.2 * (1.0 - relations.frontier_pressure)
        )
        dynamic[..., 1] = (
            0.4
            + 1.2 * settlement
            + 0.9 * features.channel("settlement_proximity")
            + 0.7 * relations.settlement_pressure
            + 0.4 * features.channel("local_settlement_density")
            + latent_scale * (0.7 * latent.expansion_pressure - 0.35 * latent.winter_harshness)
        )
        dynamic[..., 2] = (
            0.3
            + 1.2 * port
            + 0.8 * features.channel("coastal_mask")
            + 0.7 * features.channel("port_proximity")
            + 0.7 * relations.trade_access
            + 0.5 * relations.port_pressure
            + latent_scale * 0.6 * latent.trade_strength
        )
        dynamic[..., 3] = (
            0.3
            + 1.0 * ruin
            + 0.7 * features.channel("settlement_proximity")
            + 0.8 * relations.conflict_risk
            + latent_scale * (
                0.6 * latent.raid_intensity
                + 0.6 * latent.winter_harshness
                - 0.35 * latent.rebuild_tendency
            )
        )
        dynamic[..., 4] = (
            0.3
            + 0.8 * forest
            + 0.4 * ruin
            + 0.7 * features.channel("forest_adj")
            + 0.4 * features.channel("local_forest_density")
            + 0.3 * relations.frontier_pressure
            + latent_scale * 0.6 * latent.reclaim_tendency
        )
        dynamic[..., 5] = 3.0 * features.channel("terrain_mountain") + 0.02
        dynamic = dynamic / dynamic.sum(axis=-1, keepdims=True)

        ocean_mask = features.channel("terrain_ocean") > 0.5
        mountain_mask = features.channel("terrain_mountain") > 0.5
        dynamic[ocean_mask] = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        dynamic[mountain_mask] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float)
        return dynamic

    def _build_dynamic_mass(
        self,
        features: FeatureGrid,
        relations: RelationArtifacts,
        latent: RoundLatentState,
    ) -> np.ndarray:
        dynamic_mass = np.clip(
            0.05
            + 0.35 * features.channel("dynamic_zone_prior")
            + 0.15 * relations.frontier_pressure
            + 0.15 * relations.conflict_risk
            + 0.10 * relations.trade_access
            + 0.10 * max(latent.confidence, 0.1)
            + 0.10 * features.channel("terrain_settlement")
            + 0.10 * features.channel("terrain_port")
            + 0.08 * features.channel("terrain_ruin")
            - 0.30 * features.channel("terrain_ocean")
            - 0.35 * features.channel("terrain_mountain"),
            0.02,
            0.92,
        )
        return dynamic_mass

    def _blend_observations(
        self,
        prediction: np.ndarray,
        seed_observation,
    ) -> tuple[np.ndarray, np.ndarray]:
        freqs = seed_observation.frequencies()
        observed = seed_observation.observed.astype(float)
        observation_weight = self.config.observation_blend * np.tanh(observed / 2.0)
        blended = (1.0 - observation_weight[..., None]) * prediction + observation_weight[..., None] * freqs
        return blended, observation_weight
