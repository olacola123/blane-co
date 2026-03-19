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
from .probability import apply_probability_floor, calibrate_probabilities, predictive_entropy, safe_normalize
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
        self.history_class_bias = np.zeros(6, dtype=float)
        self.history_class_temperature = np.ones(6, dtype=float)
        self.history_rounds_used = 0

    def set_history_calibration(
        self,
        class_bias: np.ndarray | None = None,
        class_temperature: np.ndarray | None = None,
        rounds_used: int = 0,
    ) -> None:
        """Install optional calibration learned from historical analyses."""
        if class_bias is not None:
            self.history_class_bias = np.asarray(class_bias, dtype=float)
        if class_temperature is not None:
            self.history_class_temperature = np.asarray(class_temperature, dtype=float)
        self.history_rounds_used = max(int(rounds_used), 0)

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
        shared_prior = round_store.terrain_conditioned_prior(
            seed_state,
            include_shared_observations=self.config.model.enable_shared_round_priors,
        )
        relations = self.relation_module.build(seed_state, feature_grid, seed_observation)

        structured_fields = self._build_structured_fields(feature_grid, relations, latent, shared_prior)
        static_component = self._build_static_component(feature_grid, shared_prior)
        dynamic_component = self._build_dynamic_component(
            feature_grid,
            relations,
            latent,
            shared_prior,
            structured_fields,
        )
        dynamic_mass = self._build_dynamic_mass(
            feature_grid,
            relations,
            latent,
            shared_prior,
            structured_fields,
        )
        base = (1.0 - dynamic_mass[..., None]) * static_component + dynamic_mass[..., None] * dynamic_component
        refined = self.local_refiner.refine(base, feature_grid, relations, latent)
        blended, observation_weight = self._blend_observations(refined, seed_observation)
        calibrated = self._apply_calibration(blended)
        final = apply_probability_floor(calibrated, self.config.probability.floor)
        entropy = predictive_entropy(final)

        self.logger.info(
            "Prediction seed=%s coverage=%.3f mean_entropy=%.3f mean_max_prob=%.3f",
            seed_state.seed_index,
            seed_observation.coverage_ratio(),
            float(entropy.mean()),
            float(final.max(axis=-1).mean()),
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

    def _build_static_component(
        self,
        features: FeatureGrid,
        shared_prior: np.ndarray,
    ) -> np.ndarray:
        ocean = features.channel("terrain_ocean")
        plains_like = features.channel("plains_like")
        forest = features.channel("terrain_forest")
        mountain = features.channel("terrain_mountain")
        coastal = features.channel("coastal_mask")
        settlement_landmass = features.channel("same_landmass_as_initial_settlement")
        shared_strength = self.config.model.shared_round_prior_strength

        static = np.full(features.channels.shape[:2] + (6,), 1e-6, dtype=float)
        static[..., 0] = (
            self.config.model.geography_prior_strength * shared_strength * shared_prior[..., 0]
            + 3.0 * plains_like
            + 0.9 * features.channel("distance_settlement_norm")
            + 0.6 * (1.0 - coastal)
            + 0.4 * (1.0 - settlement_landmass)
        )
        static[..., 1] = (
            0.35 * self.config.model.geography_prior_strength * shared_strength * shared_prior[..., 1]
            + 1.5 * features.channel("terrain_settlement")
            + 0.25 * features.channel("local_settlement_density")
        )
        static[..., 2] = (
            0.30 * self.config.model.geography_prior_strength * shared_strength * shared_prior[..., 2]
            + 0.85 * features.channel("terrain_port")
            + 0.15 * coastal * features.channel("port_proximity")
        )
        static[..., 3] = (
            0.30 * self.config.model.geography_prior_strength * shared_strength * shared_prior[..., 3]
            + 0.90 * features.channel("terrain_ruin")
            + 0.15 * features.channel("local_ruin_density")
        )
        static[..., 4] = (
            0.95 * self.config.model.geography_prior_strength * shared_strength * shared_prior[..., 4]
            + 2.4 * forest
            + 0.9 * features.channel("forest_adj")
            + 0.6 * features.channel("local_forest_density")
            + 0.2 * (1.0 - features.channel("settlement_proximity"))
        )
        static[..., 5] = (
            1.2 * self.config.model.geography_prior_strength * shared_strength * shared_prior[..., 5]
            + 7.5 * mountain
            + 0.02
        )
        static = safe_normalize(static, axis=-1)

        ocean_mask = ocean > 0.5
        mountain_mask = mountain > 0.5
        static[ocean_mask] = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        static[mountain_mask] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float)
        return static

    def _build_structured_fields(
        self,
        features: FeatureGrid,
        relations: RelationArtifacts,
        latent: RoundLatentState,
        shared_prior: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Construct class-specific influence and plausibility fields."""
        ocean = features.channel("terrain_ocean")
        mountain = features.channel("terrain_mountain")
        land_gate = np.clip(1.0 - ocean - mountain, 0.0, 1.0)
        coastal = features.channel("coastal_mask")
        latent_scale = self.config.latent_strength * max(latent.confidence, 0.25)

        if not self.config.model.enable_structured_gates:
            dynamic = np.clip(
                0.55 * features.channel("dynamic_zone_prior")
                + 0.25 * relations.settlement_pressure
                + 0.20 * shared_prior[..., 1],
                0.0,
                1.0,
            )
            forest_gate = land_gate * np.clip(
                0.55 * features.channel("terrain_forest")
                + 0.25 * features.channel("forest_adj")
                + 0.20 * features.channel("local_forest_density"),
                0.0,
                1.0,
            )
            return {
                "land_gate": land_gate,
                "settlement_gate": land_gate * dynamic,
                "port_gate": land_gate * coastal * np.clip(0.60 * dynamic + 0.40 * relations.trade_access, 0.0, 1.0),
                "ruin_gate": land_gate * np.clip(0.60 * dynamic + 0.40 * relations.conflict_risk, 0.0, 1.0),
                "forest_gate": forest_gate,
                "frontier_focus": np.clip(
                    0.45 * relations.frontier_pressure
                    + 0.30 * relations.conflict_risk
                    + 0.25 * relations.trade_access,
                    0.0,
                    1.0,
                ),
            }

        settlement_anchor = np.clip(
            self.config.model.settlement_influence_strength
            * (
                0.28 * features.channel("same_landmass_as_initial_settlement")
                + 0.22 * features.channel("settlement_proximity")
                + 0.22 * relations.settlement_pressure
                + 0.14 * features.channel("local_settlement_density")
                + 0.09 * shared_prior[..., 1]
                + latent_scale * (0.14 * latent.expansion_pressure - 0.06 * latent.winter_harshness)
            ),
            0.0,
            1.0,
        )
        settlement_gate = land_gate * np.power(np.clip(settlement_anchor, 1e-6, 1.0), 1.10)

        coastal_rebuild = coastal * np.clip(
            0.45 * features.channel("terrain_ruin")
            + 0.35 * features.channel("local_ruin_density")
            + 0.20 * latent.rebuild_tendency,
            0.0,
            1.0,
        )
        port_anchor = coastal * np.clip(
            0.35 * settlement_anchor
            + 0.25 * relations.trade_access
            + 0.20 * relations.port_pressure
            + 0.12 * features.channel("port_proximity")
            + 0.08 * shared_prior[..., 2]
            + 0.08 * coastal_rebuild,
            0.0,
            1.0,
        )
        port_gate = land_gate * np.power(np.clip(port_anchor, 1e-6, 1.0), 1.65)

        ruin_anchor = np.clip(
            0.42 * settlement_anchor
            + 0.24 * relations.conflict_risk
            + 0.14 * features.channel("terrain_ruin")
            + 0.12 * features.channel("local_ruin_density")
            + 0.08 * shared_prior[..., 3]
            + latent_scale
            * (
                0.18 * latent.raid_intensity
                + 0.12 * latent.winter_harshness
                - 0.08 * latent.rebuild_tendency
            ),
            0.0,
            1.0,
        )
        ruin_gate = land_gate * np.power(np.clip(ruin_anchor, 1e-6, 1.0), 1.35)

        forest_anchor = np.clip(
            0.48 * features.channel("terrain_forest")
            + 0.24 * features.channel("forest_adj")
            + 0.16 * features.channel("local_forest_density")
            + 0.10 * shared_prior[..., 4]
            + latent_scale * 0.12 * latent.reclaim_tendency,
            0.0,
            1.0,
        )
        forest_gate = land_gate * forest_anchor
        frontier_focus = np.clip(
            0.45 * relations.frontier_pressure
            + 0.30 * relations.conflict_risk
            + 0.25 * relations.trade_access,
            0.0,
            1.0,
        )
        return {
            "land_gate": land_gate,
            "settlement_gate": settlement_gate,
            "port_gate": port_gate,
            "ruin_gate": ruin_gate,
            "forest_gate": forest_gate,
            "frontier_focus": frontier_focus,
        }

    def _build_dynamic_component(
        self,
        features: FeatureGrid,
        relations: RelationArtifacts,
        latent: RoundLatentState,
        shared_prior: np.ndarray,
        structured_fields: dict[str, np.ndarray],
    ) -> np.ndarray:
        latent_scale = self.config.latent_strength * max(latent.confidence, 0.25)
        settlement_gate = structured_fields["settlement_gate"]
        port_gate = structured_fields["port_gate"]
        ruin_gate = structured_fields["ruin_gate"]
        forest_gate = structured_fields["forest_gate"]
        frontier_focus = structured_fields["frontier_focus"]

        dynamic = np.full(features.channels.shape[:2] + (6,), 1e-6, dtype=float)
        dynamic[..., 0] = (
            1.0
            + 1.8 * shared_prior[..., 0]
            + 1.0 * features.channel("plains_like")
            + 0.35 * features.channel("distance_mountain_norm")
            + 0.35 * (1.0 - settlement_gate)
            + 0.30 * (1.0 - port_gate)
            + 0.25 * (1.0 - ruin_gate)
        )
        dynamic[..., 1] = (
            0.04
            + 4.2
            * settlement_gate
            * (
                0.35
                + 0.28 * relations.settlement_pressure
                + 0.18 * features.channel("local_settlement_density")
                + 0.12 * shared_prior[..., 1]
                + latent_scale * (0.22 * latent.expansion_pressure - 0.08 * latent.winter_harshness)
            )
        )
        dynamic[..., 2] = (
            0.02
            + 4.8
            * port_gate
            * (
                0.28
                + 0.32 * relations.trade_access
                + 0.24 * relations.port_pressure
                + 0.12 * shared_prior[..., 2]
                + latent_scale * 0.20 * latent.trade_strength
            )
        )
        dynamic[..., 3] = (
            0.02
            + 4.0
            * ruin_gate
            * (
                0.25
                + 0.34 * relations.conflict_risk
                + 0.12 * shared_prior[..., 3]
                + latent_scale
                * (
                    0.18 * latent.raid_intensity
                    + 0.16 * latent.winter_harshness
                    - 0.10 * latent.rebuild_tendency
                )
            )
        )
        dynamic[..., 4] = (
            0.10
            + 3.0
            * forest_gate
            * (
                0.30
                + 0.24 * features.channel("local_forest_density")
                + 0.12 * frontier_focus
                + 0.12 * shared_prior[..., 4]
                + latent_scale * 0.18 * latent.reclaim_tendency
            )
        )
        dynamic[..., 5] = 3.0 * features.channel("terrain_mountain") + 0.02
        dynamic = safe_normalize(dynamic, axis=-1)

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
        shared_prior: np.ndarray,
        structured_fields: dict[str, np.ndarray],
    ) -> np.ndarray:
        settlement_gate = structured_fields["settlement_gate"]
        port_gate = structured_fields["port_gate"]
        ruin_gate = structured_fields["ruin_gate"]
        forest_gate = structured_fields["forest_gate"]
        frontier_focus = structured_fields["frontier_focus"]
        dynamic_mass = np.clip(
            0.02
            + 0.30 * settlement_gate
            + 0.22 * port_gate
            + 0.20 * ruin_gate
            + 0.12 * forest_gate * max(latent.reclaim_tendency, 0.25)
            + 0.14 * frontier_focus
            + 0.06 * shared_prior[..., 1]
            + 0.06 * shared_prior[..., 2]
            + 0.06 * shared_prior[..., 3]
            - 0.30 * features.channel("terrain_ocean")
            - 0.38 * features.channel("terrain_mountain")
            - 0.32 * shared_prior[..., 0],
            0.015,
            0.78,
        )
        return dynamic_mass

    def _blend_observations(
        self,
        prediction: np.ndarray,
        seed_observation,
    ) -> tuple[np.ndarray, np.ndarray]:
        posterior = seed_observation.posterior(
            prediction,
            prior_strength=self.config.model.observation_prior_strength,
        )
        observed = seed_observation.observed.astype(float)
        observation_weight = observed / (observed + self.config.model.observation_prior_strength)
        return posterior, observation_weight

    def _apply_calibration(self, probabilities: np.ndarray) -> np.ndarray:
        class_bias = np.asarray(self.config.probability.class_bias, dtype=float)
        class_temperature = np.asarray(self.config.probability.class_temperature, dtype=float)
        if (
            self.config.model.history_tuning_enabled
            and self.history_rounds_used > 0
            and self.history_class_bias.shape == class_bias.shape
            and self.history_class_temperature.shape == class_temperature.shape
        ):
            weight = self.config.model.history_tuning_weight
            class_bias = class_bias + weight * self.history_class_bias
            class_temperature = np.clip(
                class_temperature * np.power(np.clip(self.history_class_temperature, 0.6, 1.8), weight),
                0.6,
                1.8,
            )

        return calibrate_probabilities(
            probabilities,
            temperature=self.config.probability.temperature,
            class_temperatures=class_temperature,
            class_bias=class_bias,
            enable_temperature_scaling=self.config.probability.enable_temperature_scaling,
            enable_class_calibration=self.config.probability.enable_class_calibration,
        )
