"""Round-level latent inference from pooled observations."""

from __future__ import annotations

import logging

import numpy as np

from .types import ObservationSummary, RoundLatentState


def _squash(value: float, scale: float) -> float:
    """Map an unbounded positive statistic into a stable [0, 1) range."""
    if scale <= 0:
        return 0.0
    return float(value / (value + scale))


class RoundLatentInferer:
    """Infer a simple shared latent vector from round-wide observations."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def infer(self, summary: ObservationSummary) -> RoundLatentState:
        """Convert pooled observation summaries into named latent factors."""
        normalized_population = _squash(summary.mean_population, 4.0)
        normalized_food = _squash(summary.mean_food, 2.0)
        normalized_wealth = _squash(summary.mean_wealth, 2.0)
        normalized_defense = _squash(summary.mean_defense, 2.0)

        expansion_pressure = np.clip(
            0.35 * summary.settlement_density
            + 0.20 * summary.port_density
            + 0.20 * normalized_population
            + 0.15 * normalized_wealth
            + 0.10 * summary.alive_ratio,
            0.0,
            1.0,
        )
        winter_harshness = np.clip(
            0.35 * summary.ruin_density
            + 0.20 * (1.0 - normalized_food)
            + 0.20 * (1.0 - summary.alive_ratio)
            + 0.15 * (1.0 - normalized_defense)
            + 0.10 * (1.0 - summary.coverage_ratio),
            0.0,
            1.0,
        )
        raid_intensity = np.clip(
            0.35 * summary.ruin_density
            + 0.25 * summary.owner_diversity
            + 0.15 * (1.0 - normalized_food)
            + 0.15 * (1.0 - normalized_defense)
            + 0.10 * summary.port_ratio,
            0.0,
            1.0,
        )
        trade_strength = np.clip(
            0.35 * summary.port_ratio
            + 0.30 * normalized_wealth
            + 0.20 * summary.alive_ratio
            + 0.15 * summary.port_density,
            0.0,
            1.0,
        )
        rebuild_tendency = np.clip(
            0.40 * summary.rebuild_ratio
            + 0.25 * trade_strength
            + 0.20 * summary.alive_ratio
            + 0.15 * expansion_pressure,
            0.0,
            1.0,
        )
        reclaim_tendency = np.clip(
            0.40 * summary.reclaim_ratio
            + 0.25 * summary.forest_density
            + 0.20 * winter_harshness
            + 0.15 * (1.0 - summary.port_density),
            0.0,
            1.0,
        )
        uncertainty = np.clip(
            0.45 * (1.0 - summary.coverage_ratio)
            + 0.25 * summary.ruin_density
            + 0.15 * summary.owner_diversity
            + 0.15 * (1.0 - summary.alive_ratio),
            0.0,
            1.0,
        )
        confidence = np.clip(
            0.65 * summary.coverage_ratio
            + 0.20 * summary.observed_seed_fraction
            + 0.15 * _squash(summary.num_viewports, 10.0),
            0.0,
            1.0,
        )

        forest_boost = max(0.0, summary.forest_density - 0.15) * 2.0
        expansion_pressure = float(np.clip(expansion_pressure - 0.4 * forest_boost, 0, 1))
        reclaim_tendency = float(np.clip(reclaim_tendency + 0.5 * forest_boost, 0, 1))

        latent = RoundLatentState(
            expansion_pressure=float(expansion_pressure),
            winter_harshness=float(winter_harshness),
            raid_intensity=float(raid_intensity),
            trade_strength=float(trade_strength),
            rebuild_tendency=float(rebuild_tendency),
            reclaim_tendency=float(reclaim_tendency),
            uncertainty=float(uncertainty),
            confidence=float(confidence),
        )
        self.logger.info(
            "Latent z_round expansion=%.3f winter=%.3f raid=%.3f trade=%.3f rebuild=%.3f reclaim=%.3f confidence=%.3f",
            latent.expansion_pressure,
            latent.winter_harshness,
            latent.raid_intensity,
            latent.trade_strength,
            latent.rebuild_tendency,
            latent.reclaim_tendency,
            latent.confidence,
        )
        return latent
