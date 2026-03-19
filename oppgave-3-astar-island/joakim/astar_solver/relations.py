"""Settlement-centric relational features and influence maps."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .features import FeatureGrid
from .observations import AggregatedSettlementStats, SeedObservationGrid
from .types import SeedState, Settlement


def _squash(value: float, scale: float) -> float:
    """Bound an unbounded statistic into [0, 1)."""
    if scale <= 0:
        return 0.0
    return float(value / (value + scale))


@dataclass(slots=True)
class SettlementNode:
    """Relational node representation for a settlement-like entity."""

    x: int
    y: int
    initial_has_port: bool
    observed_port_ratio: float
    alive_ratio: float
    population: float
    food: float
    wealth: float
    defense: float
    owner_diversity: float
    landmass_id: int
    coastline_access: float
    nearby_forest: float
    nearby_mountain: float


@dataclass(slots=True)
class RelationArtifacts:
    """Spatialized outputs from the settlement relation module."""

    settlement_pressure: np.ndarray
    port_pressure: np.ndarray
    trade_access: np.ndarray
    conflict_risk: np.ndarray
    frontier_pressure: np.ndarray
    nodes: tuple[SettlementNode, ...]
    edge_count: int


class SettlementRelationModule:
    """Build graph-like settlement influence maps without a heavy GNN."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def build(
        self,
        seed_state: SeedState,
        features: FeatureGrid,
        seed_observation: SeedObservationGrid,
    ) -> RelationArtifacts:
        """Build relation nodes, edges, and rendered influence maps."""
        nodes = self._build_nodes(seed_state, features, seed_observation)
        edge_count, density, trade_neighbors, conflict_neighbors = self._build_edges(nodes)
        maps = self._render_maps(
            nodes=nodes,
            density=density,
            trade_neighbors=trade_neighbors,
            conflict_neighbors=conflict_neighbors,
            shape=seed_state.grid.shape,
        )
        self.logger.info(
            "Settlement relations for seed %s: nodes=%s edges=%s",
            seed_state.seed_index,
            len(nodes),
            edge_count,
        )
        return RelationArtifacts(
            settlement_pressure=maps["settlement_pressure"],
            port_pressure=maps["port_pressure"],
            trade_access=maps["trade_access"],
            conflict_risk=maps["conflict_risk"],
            frontier_pressure=maps["frontier_pressure"],
            nodes=tuple(nodes),
            edge_count=edge_count,
        )

    def _build_nodes(
        self,
        seed_state: SeedState,
        features: FeatureGrid,
        seed_observation: SeedObservationGrid,
    ) -> list[SettlementNode]:
        known_positions = {(settlement.x, settlement.y) for settlement in seed_state.settlements}
        nodes = [
            self._node_from_context(
                settlement=settlement,
                stats=seed_observation.settlement_at(settlement.x, settlement.y),
                features=features,
            )
            for settlement in seed_state.settlements
        ]

        for (x, y), stats in seed_observation.settlement_stats.items():
            if (x, y) in known_positions:
                continue
            nodes.append(
                self._node_from_context(
                    settlement=Settlement(x=x, y=y, has_port=False),
                    stats=stats,
                    features=features,
                )
            )
        return nodes

    def _node_from_context(
        self,
        settlement: Settlement,
        stats: AggregatedSettlementStats | None,
        features: FeatureGrid,
    ) -> SettlementNode:
        y = settlement.y
        x = settlement.x
        return SettlementNode(
            x=x,
            y=y,
            initial_has_port=settlement.has_port,
            observed_port_ratio=stats.port_ratio if stats else float(settlement.has_port),
            alive_ratio=stats.alive_ratio if stats else 0.75,
            population=_squash(stats.mean_population if stats else 2.0, 4.0),
            food=_squash(stats.mean_food if stats else 1.0, 2.0),
            wealth=_squash(stats.mean_wealth if stats else 0.8, 2.0),
            defense=_squash(stats.mean_defense if stats else 1.0, 2.0),
            owner_diversity=stats.owner_diversity if stats else 0.0,
            landmass_id=int(features.landmass_id[y, x]),
            coastline_access=float(features.channel("coastal_mask")[y, x]),
            nearby_forest=float(features.channel("forest_adj")[y, x]),
            nearby_mountain=float(features.channel("mountain_adj")[y, x]),
        )

    def _build_edges(
        self,
        nodes: list[SettlementNode],
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        count = len(nodes)
        density = np.zeros(count, dtype=float)
        trade_neighbors = np.zeros(count, dtype=float)
        conflict_neighbors = np.zeros(count, dtype=float)
        edge_count = 0

        for left in range(count):
            for right in range(left + 1, count):
                node_a = nodes[left]
                node_b = nodes[right]
                dy = float(node_a.y - node_b.y)
                dx = float(node_a.x - node_b.x)
                distance = float(np.hypot(dx, dy))
                same_landmass = node_a.landmass_id != 0 and node_a.landmass_id == node_b.landmass_id
                coastal_link = node_a.coastline_access > 0.5 and node_b.coastline_access > 0.5
                if not (distance <= 12.0 or (same_landmass and distance <= 18.0) or (coastal_link and distance <= 20.0)):
                    continue

                edge_count += 1
                density[left] += 1.0
                density[right] += 1.0

                trade_strength = np.clip(
                    0.30 * (node_a.observed_port_ratio + node_b.observed_port_ratio)
                    + 0.25 * (node_a.wealth + node_b.wealth)
                    + 0.20 * float(coastal_link)
                    + 0.25 * max(0.0, 1.0 - distance / 20.0),
                    0.0,
                    1.0,
                )
                conflict_strength = np.clip(
                    0.30 * (node_a.owner_diversity + node_b.owner_diversity)
                    + 0.20 * (1.0 - node_a.food + 1.0 - node_b.food)
                    + 0.20 * (1.0 - node_a.defense + 1.0 - node_b.defense)
                    + 0.30 * max(0.0, 1.0 - distance / 12.0),
                    0.0,
                    1.0,
                )
                trade_neighbors[left] += trade_strength
                trade_neighbors[right] += trade_strength
                conflict_neighbors[left] += conflict_strength
                conflict_neighbors[right] += conflict_strength

        return edge_count, density, trade_neighbors, conflict_neighbors

    def _render_maps(
        self,
        nodes: list[SettlementNode],
        density: np.ndarray,
        trade_neighbors: np.ndarray,
        conflict_neighbors: np.ndarray,
        shape: tuple[int, int],
    ) -> dict[str, np.ndarray]:
        yy, xx = np.indices(shape)
        settlement_pressure = np.zeros(shape, dtype=float)
        port_pressure = np.zeros(shape, dtype=float)
        trade_access = np.zeros(shape, dtype=float)
        conflict_risk = np.zeros(shape, dtype=float)
        frontier_pressure = np.zeros(shape, dtype=float)

        for index, node in enumerate(nodes):
            dist = np.hypot(xx - node.x, yy - node.y)
            density_boost = _squash(density[index], 4.0)
            trade_neighbor_boost = _squash(trade_neighbors[index], 3.0)
            conflict_neighbor_boost = _squash(conflict_neighbors[index], 3.0)

            growth_strength = np.clip(
                0.25
                + 0.20 * node.alive_ratio
                + 0.15 * node.population
                + 0.10 * node.food
                + 0.10 * node.coastline_access
                + 0.10 * node.nearby_forest
                + 0.10 * density_boost,
                0.0,
                1.0,
            )
            trade_strength = np.clip(
                0.20
                + 0.20 * node.observed_port_ratio
                + 0.20 * node.wealth
                + 0.15 * node.coastline_access
                + 0.15 * trade_neighbor_boost
                + 0.10 * node.alive_ratio,
                0.0,
                1.0,
            )
            conflict_strength = np.clip(
                0.15
                + 0.20 * conflict_neighbor_boost
                + 0.15 * (1.0 - node.food)
                + 0.15 * (1.0 - node.defense)
                + 0.15 * density_boost
                + 0.10 * node.owner_diversity
                + 0.10 * node.wealth,
                0.0,
                1.0,
            )
            frontier_strength = np.clip(
                0.20
                + 0.25 * growth_strength
                + 0.20 * trade_strength
                + 0.15 * conflict_strength
                + 0.10 * node.nearby_forest
                + 0.10 * (1.0 - node.nearby_mountain),
                0.0,
                1.0,
            )

            settlement_pressure += growth_strength * np.exp(-dist / 5.5)
            port_pressure += max(node.observed_port_ratio, float(node.initial_has_port)) * np.exp(-dist / 5.0)
            trade_access += trade_strength * np.exp(-dist / 8.0)
            conflict_risk += conflict_strength * np.exp(-dist / 6.5)
            frontier_pressure += frontier_strength * np.exp(-dist / 4.5)

        return {
            "settlement_pressure": self._normalize_map(settlement_pressure),
            "port_pressure": self._normalize_map(port_pressure),
            "trade_access": self._normalize_map(trade_access),
            "conflict_risk": self._normalize_map(conflict_risk),
            "frontier_pressure": self._normalize_map(frontier_pressure),
        }

    @staticmethod
    def _normalize_map(values: np.ndarray) -> np.ndarray:
        maximum = float(values.max())
        if maximum <= 0.0:
            return np.zeros_like(values, dtype=float)
        return values / maximum
