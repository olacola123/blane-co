"""Neighborhood-aware local dynamics refinement."""

from __future__ import annotations

import numpy as np

from .constants import CLASS_EMPTY, CLASS_FOREST, CLASS_MOUNTAIN, CLASS_PORT, CLASS_RUIN, CLASS_SETTLEMENT
from .features import FeatureGrid
from .probability import safe_normalize
from .relations import RelationArtifacts
from .types import RoundLatentState


def _neighbor_mean(values: np.ndarray) -> np.ndarray:
    """Average over the 8-neighborhood."""
    padded = np.pad(values.astype(float), 1, mode="edge")
    total = np.zeros_like(values, dtype=float)
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            total += padded[dy : dy + values.shape[0], dx : dx + values.shape[1]]
    return total / 8.0


class LocalDynamicsRefiner:
    """Combine local spatial dynamics with relation maps and round latent state."""

    def __init__(self, passes: int = 2) -> None:
        self.passes = max(int(passes), 0)

    def refine(
        self,
        probabilities: np.ndarray,
        features: FeatureGrid,
        relations: RelationArtifacts,
        latent: RoundLatentState,
    ) -> np.ndarray:
        """Run a small number of fixed local refinement passes."""
        if self.passes == 0:
            return np.array(probabilities, dtype=float, copy=True)

        probs = np.array(probabilities, dtype=float, copy=True)
        ocean_mask = features.channel("terrain_ocean") > 0.5
        mountain_mask = features.channel("terrain_mountain") > 0.5

        for _ in range(self.passes):
            settlement_neighborhood = _neighbor_mean(probs[..., CLASS_SETTLEMENT] + probs[..., CLASS_PORT])
            ruin_neighborhood = _neighbor_mean(probs[..., CLASS_RUIN])
            forest_neighborhood = _neighbor_mean(probs[..., CLASS_FOREST])

            logits = np.log(np.clip(probs, 1e-9, 1.0))
            logits[..., CLASS_EMPTY] += (
                0.15 * (1.0 - relations.frontier_pressure)
                + 0.10 * features.channel("distance_settlement_norm")
            )
            logits[..., CLASS_SETTLEMENT] += (
                0.60 * settlement_neighborhood
                + 0.55 * relations.settlement_pressure
                + 0.25 * features.channel("settlement_proximity")
                + 0.20 * latent.expansion_pressure
                - 0.20 * latent.winter_harshness
            )
            logits[..., CLASS_PORT] += (
                0.55 * relations.trade_access
                + 0.45 * relations.port_pressure
                + 0.25 * features.channel("coastal_mask")
                + 0.15 * settlement_neighborhood
                + 0.20 * latent.trade_strength
            )
            logits[..., CLASS_RUIN] += (
                0.45 * relations.conflict_risk
                + 0.20 * ruin_neighborhood
                + 0.20 * settlement_neighborhood
                + 0.20 * latent.raid_intensity
                + 0.25 * latent.winter_harshness
                - 0.20 * latent.rebuild_tendency
            )
            logits[..., CLASS_FOREST] += (
                0.45 * forest_neighborhood
                + 0.25 * features.channel("local_forest_density")
                + 0.20 * latent.reclaim_tendency
                + 0.10 * relations.frontier_pressure
                - 0.10 * relations.settlement_pressure
            )
            logits[..., CLASS_MOUNTAIN] = np.where(mountain_mask, 9.0, -9.0)

            logits -= logits.max(axis=-1, keepdims=True)
            probs = np.exp(logits)
            probs = safe_normalize(probs, axis=-1)

            probs[ocean_mask] = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
            probs[mountain_mask] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float)

        return probs
