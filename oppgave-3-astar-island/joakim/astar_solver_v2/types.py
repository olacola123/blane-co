"""Typed solver data structures."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import GRID_SIZE, INTERNAL_MOUNTAIN, INTERNAL_OCEAN, TERRAIN_TO_CLASS


@dataclass(slots=True)
class Settlement:
    """Settlement as exposed by the initial round state."""

    x: int
    y: int
    has_port: bool = False


@dataclass(slots=True)
class ObservedSettlement:
    """Settlement stats exposed through a simulation viewport."""

    x: int
    y: int
    population: float
    food: float
    wealth: float
    defense: float
    has_port: bool
    alive: bool
    owner_id: int

    @classmethod
    def from_payload(cls, payload: dict) -> "ObservedSettlement":
        """Parse an observation settlement payload."""
        return cls(
            x=int(payload["x"]),
            y=int(payload["y"]),
            population=float(payload.get("population", 0.0)),
            food=float(payload.get("food", 0.0)),
            wealth=float(payload.get("wealth", 0.0)),
            defense=float(payload.get("defense", 0.0)),
            has_port=bool(payload.get("has_port", False)),
            alive=bool(payload.get("alive", False)),
            owner_id=int(payload.get("owner_id", -1)),
        )


@dataclass(slots=True)
class Viewport:
    """A 2D viewport in grid coordinates."""

    x: int
    y: int
    w: int
    h: int

    def as_dict(self) -> dict[str, int]:
        """Convert to API-compatible dictionary."""
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    @classmethod
    def centered(cls, cx: int, cy: int, size: int) -> "Viewport":
        """Build a centered square viewport clamped to the known grid."""
        half = size // 2
        x = max(0, min(GRID_SIZE - size, int(cx) - half))
        y = max(0, min(GRID_SIZE - size, int(cy) - half))
        return cls(x=x, y=y, w=size, h=size)


@dataclass(slots=True)
class ViewportObservation:
    """One stochastic simulation observation."""

    round_id: str
    seed_index: int
    viewport: Viewport
    grid: np.ndarray
    settlements: list[ObservedSettlement] = field(default_factory=list)
    queries_used: int | None = None
    queries_max: int | None = None

    @classmethod
    def from_simulation_result(
        cls,
        round_id: str,
        seed_index: int,
        payload: dict,
    ) -> "ViewportObservation":
        """Parse the raw API simulation response."""
        viewport_payload = payload["viewport"]
        grid = np.array(payload["grid"], dtype=int)
        settlements = [
            ObservedSettlement.from_payload(item)
            for item in payload.get("settlements", [])
        ]
        return cls(
            round_id=round_id,
            seed_index=seed_index,
            viewport=Viewport(
                x=int(viewport_payload["x"]),
                y=int(viewport_payload["y"]),
                w=int(viewport_payload["w"]),
                h=int(viewport_payload["h"]),
            ),
            grid=grid,
            settlements=settlements,
            queries_used=payload.get("queries_used"),
            queries_max=payload.get("queries_max"),
        )


@dataclass(slots=True)
class SeedState:
    """Parsed initial state for one seed."""

    seed_index: int
    grid: np.ndarray
    settlements: list[Settlement] = field(default_factory=list)
    static_mask: np.ndarray = field(
        default_factory=lambda: np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    )

    @classmethod
    def from_round_data(cls, seed_index: int, seed_data: dict) -> "SeedState":
        """Parse one seed entry from the round API response."""
        grid = np.array(seed_data["grid"], dtype=int)
        settlements = [
            Settlement(
                x=int(item["x"]),
                y=int(item["y"]),
                has_port=bool(item.get("has_port", False)),
            )
            for item in seed_data.get("settlements", [])
        ]
        static_mask = np.isin(grid, [INTERNAL_OCEAN, INTERNAL_MOUNTAIN])
        return cls(
            seed_index=seed_index,
            grid=grid,
            settlements=settlements,
            static_mask=static_mask,
        )

    def class_grid(self) -> np.ndarray:
        """Map internal terrain codes to prediction classes."""
        result = np.zeros_like(self.grid)
        for terrain_code, class_id in TERRAIN_TO_CLASS.items():
            result[self.grid == terrain_code] = class_id
        return result


@dataclass(slots=True)
class ObservationSummary:
    """Aggregated observation evidence across the round."""

    num_viewports: int = 0
    coverage_ratio: float = 0.0
    observed_seed_fraction: float = 0.0
    mean_observations_per_covered_cell: float = 0.0
    settlement_density: float = 0.0
    port_density: float = 0.0
    ruin_density: float = 0.0
    forest_density: float = 0.0
    empty_density: float = 0.0
    alive_ratio: float = 0.5
    port_ratio: float = 0.0
    mean_population: float = 0.0
    mean_food: float = 0.0
    mean_wealth: float = 0.0
    mean_defense: float = 0.0
    owner_diversity: float = 0.0
    reclaim_ratio: float = 0.0
    rebuild_ratio: float = 0.0


@dataclass(slots=True)
class RoundLatentState:
    """Named shared round-level latent factors."""

    expansion_pressure: float = 0.0
    winter_harshness: float = 0.0
    raid_intensity: float = 0.0
    trade_strength: float = 0.0
    rebuild_tendency: float = 0.0
    reclaim_tendency: float = 0.0
    uncertainty: float = 0.0
    confidence: float = 0.0

    def as_array(self) -> np.ndarray:
        """Return the latent vector as a fixed-order array."""
        return np.array(
            [
                self.expansion_pressure,
                self.winter_harshness,
                self.raid_intensity,
                self.trade_strength,
                self.rebuild_tendency,
                self.reclaim_tendency,
                self.uncertainty,
                self.confidence,
            ],
            dtype=float,
        )
