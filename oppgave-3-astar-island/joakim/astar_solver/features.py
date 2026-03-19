"""Deterministic map feature extraction for the baseline predictor."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .constants import (
    GRID_SIZE,
    INTERNAL_EMPTY,
    INTERNAL_FOREST,
    INTERNAL_MOUNTAIN,
    INTERNAL_OCEAN,
    INTERNAL_PLAINS,
    INTERNAL_PORT,
    INTERNAL_RUIN,
    INTERNAL_SETTLEMENT,
    TERRAIN_CODES,
)
from .types import SeedState


def _neighbor_sum(mask: np.ndarray) -> np.ndarray:
    """Return the 8-neighborhood sum of a binary mask."""
    padded = np.pad(mask.astype(float), 1, mode="constant")
    total = np.zeros_like(mask, dtype=float)
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            total += padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return total


def _window_mean(values: np.ndarray, radius: int) -> np.ndarray:
    """Compute a local mean using a square window."""
    size = 2 * radius + 1
    padded = np.pad(values.astype(float), radius, mode="edge")
    result = np.zeros_like(values, dtype=float)
    area = float(size * size)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            window = padded[y : y + size, x : x + size]
            result[y, x] = window.sum() / area
    return result


def _distance_to_sources(source_mask: np.ndarray) -> np.ndarray:
    """Manhattan distance to the nearest source cell."""
    height, width = source_mask.shape
    max_distance = height + width
    dist = np.full((height, width), max_distance, dtype=int)
    queue: deque[tuple[int, int]] = deque()

    for y, x in np.argwhere(source_mask):
        dist[y, x] = 0
        queue.append((y, x))

    if not queue:
        return dist.astype(float)

    while queue:
        y, x = queue.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny = y + dy
            nx = x + dx
            if 0 <= ny < height and 0 <= nx < width and dist[ny, nx] > dist[y, x] + 1:
                dist[ny, nx] = dist[y, x] + 1
                queue.append((ny, nx))

    return dist.astype(float)


def _connected_landmasses(passable_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label connected components and expose per-cell component size."""
    labels = np.zeros(passable_mask.shape, dtype=int)
    sizes = np.zeros(passable_mask.shape, dtype=int)
    current_label = 0
    height, width = passable_mask.shape

    for y in range(height):
        for x in range(width):
            if not passable_mask[y, x] or labels[y, x] != 0:
                continue
            current_label += 1
            queue: deque[tuple[int, int]] = deque([(y, x)])
            labels[y, x] = current_label
            cells: list[tuple[int, int]] = []
            while queue:
                cy, cx = queue.popleft()
                cells.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny = cy + dy
                    nx = cx + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and passable_mask[ny, nx]
                        and labels[ny, nx] == 0
                    ):
                        labels[ny, nx] = current_label
                        queue.append((ny, nx))
            size = len(cells)
            for cy, cx in cells:
                sizes[cy, cx] = size

    return labels, sizes


@dataclass(slots=True)
class FeatureGrid:
    """Named per-cell features for the downstream predictor."""

    channels: np.ndarray
    names: tuple[str, ...]
    landmass_id: np.ndarray
    _name_to_index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._name_to_index = {name: idx for idx, name in enumerate(self.names)}

    def channel(self, name: str) -> np.ndarray:
        """Return one named feature channel."""
        return self.channels[..., self._name_to_index[name]]


class MapFeatureExtractor:
    """Constructs a robust set of geography features from the initial map."""

    def extract(self, seed_state: SeedState) -> FeatureGrid:
        """Build feature channels for one seed."""
        grid = seed_state.grid
        terrain_masks = {code: (grid == code).astype(float) for code in TERRAIN_CODES}

        ocean_mask = grid == INTERNAL_OCEAN
        mountain_mask = grid == INTERNAL_MOUNTAIN
        forest_mask = grid == INTERNAL_FOREST
        settlement_mask = grid == INTERNAL_SETTLEMENT
        port_mask = grid == INTERNAL_PORT
        ruin_mask = grid == INTERNAL_RUIN
        plains_like_mask = np.isin(grid, [INTERNAL_EMPTY, INTERNAL_PLAINS])

        ocean_adj = _neighbor_sum(ocean_mask) / 8.0
        forest_adj = _neighbor_sum(forest_mask) / 8.0
        mountain_adj = _neighbor_sum(mountain_mask) / 8.0
        coastal_mask = ((~ocean_mask) & (ocean_adj > 0)).astype(float)

        passable_mask = (~ocean_mask) & (~mountain_mask)
        landmass_id, landmass_size = _connected_landmasses(passable_mask)
        max_landmass_size = max(int(landmass_size.max()), 1)
        max_landmass_id = max(int(landmass_id.max()), 1)

        distance_ocean = _distance_to_sources(ocean_mask)
        distance_coast = _distance_to_sources(coastal_mask.astype(bool))
        distance_mountain = _distance_to_sources(mountain_mask)

        settlement_source = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        port_source = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        for settlement in seed_state.settlements:
            settlement_source[settlement.y, settlement.x] = True
            if settlement.has_port:
                port_source[settlement.y, settlement.x] = True

        distance_settlement = _distance_to_sources(settlement_source)
        distance_port = _distance_to_sources(port_source)

        settlement_proximity = np.exp(-distance_settlement / 4.5)
        port_proximity = np.exp(-distance_port / 4.5)

        local_forest_density = _window_mean(forest_mask.astype(float), radius=2)
        local_mountain_density = _window_mean(mountain_mask.astype(float), radius=2)
        local_settlement_density = _window_mean(
            (settlement_mask | port_mask).astype(float),
            radius=2,
        )
        local_ruin_density = _window_mean(ruin_mask.astype(float), radius=2)

        dynamic_zone_prior = np.clip(
            0.45 * settlement_proximity
            + 0.25 * port_proximity
            + 0.20 * coastal_mask
            + 0.20 * local_settlement_density
            + 0.10 * ruin_mask.astype(float)
            + 0.10 * forest_adj,
            0.0,
            1.0,
        )

        max_distance = float(distance_ocean.max() or 1.0)
        names: list[str] = ["terrain_code_norm"]
        channels: list[np.ndarray] = [grid.astype(float) / 11.0]

        terrain_name_map = {
            INTERNAL_EMPTY: "terrain_empty",
            INTERNAL_OCEAN: "terrain_ocean",
            INTERNAL_PLAINS: "terrain_plains",
            INTERNAL_SETTLEMENT: "terrain_settlement",
            INTERNAL_PORT: "terrain_port",
            INTERNAL_RUIN: "terrain_ruin",
            INTERNAL_FOREST: "terrain_forest",
            INTERNAL_MOUNTAIN: "terrain_mountain",
        }
        for terrain_code in TERRAIN_CODES:
            names.append(terrain_name_map[terrain_code])
            channels.append(terrain_masks[terrain_code])

        names.extend(
            [
                "plains_like",
                "ocean_adj",
                "coastal_mask",
                "forest_adj",
                "mountain_adj",
                "landmass_id_norm",
                "landmass_size_norm",
                "distance_ocean_norm",
                "distance_coast_norm",
                "distance_mountain_norm",
                "distance_settlement_norm",
                "distance_port_norm",
                "settlement_proximity",
                "port_proximity",
                "local_forest_density",
                "local_mountain_density",
                "local_settlement_density",
                "local_ruin_density",
                "dynamic_zone_prior",
            ]
        )
        channels.extend(
            [
                plains_like_mask.astype(float),
                ocean_adj,
                coastal_mask,
                forest_adj,
                mountain_adj,
                landmass_id.astype(float) / max_landmass_id,
                landmass_size.astype(float) / max_landmass_size,
                distance_ocean / max_distance,
                distance_coast / max_distance,
                distance_mountain / max_distance,
                distance_settlement / max_distance,
                distance_port / max_distance,
                settlement_proximity,
                port_proximity,
                local_forest_density,
                local_mountain_density,
                local_settlement_density,
                local_ruin_density,
                dynamic_zone_prior,
            ]
        )

        feature_tensor = np.stack(channels, axis=-1)
        return FeatureGrid(
            channels=feature_tensor,
            names=tuple(names),
            landmass_id=landmass_id,
        )
