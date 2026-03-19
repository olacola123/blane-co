"""Observation storage and aggregation across a round."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .constants import GRID_SIZE, NUM_CLASSES, TERRAIN_TO_CLASS
from .probability import safe_normalize
from .types import ObservationSummary, SeedState, ViewportObservation


@dataclass(slots=True)
class AggregatedSettlementStats:
    """Running summary for a settlement coordinate."""

    count: int = 0
    alive_count: int = 0
    port_count: int = 0
    population_sum: float = 0.0
    food_sum: float = 0.0
    wealth_sum: float = 0.0
    defense_sum: float = 0.0
    owner_ids: Counter = field(default_factory=Counter)

    def update(self, settlement) -> None:
        """Accumulate one observed settlement record."""
        self.count += 1
        self.alive_count += int(settlement.alive)
        self.port_count += int(settlement.has_port)
        self.population_sum += settlement.population
        self.food_sum += settlement.food
        self.wealth_sum += settlement.wealth
        self.defense_sum += settlement.defense
        self.owner_ids[settlement.owner_id] += 1

    @property
    def alive_ratio(self) -> float:
        return self.alive_count / self.count if self.count else 0.5

    @property
    def port_ratio(self) -> float:
        return self.port_count / self.count if self.count else 0.0

    @property
    def mean_population(self) -> float:
        return self.population_sum / self.count if self.count else 0.0

    @property
    def mean_food(self) -> float:
        return self.food_sum / self.count if self.count else 0.0

    @property
    def mean_wealth(self) -> float:
        return self.wealth_sum / self.count if self.count else 0.0

    @property
    def mean_defense(self) -> float:
        return self.defense_sum / self.count if self.count else 0.0

    @property
    def owner_diversity(self) -> float:
        return len(self.owner_ids) / self.count if self.count else 0.0


class SeedObservationGrid:
    """Per-seed terrain and settlement observation memory."""

    def __init__(self) -> None:
        self.counts = np.zeros((GRID_SIZE, GRID_SIZE, NUM_CLASSES), dtype=float)
        self.observed = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.settlement_stats: dict[tuple[int, int], AggregatedSettlementStats] = {}

    def add_observation(self, observation: ViewportObservation) -> None:
        """Integrate one viewport result."""
        viewport = observation.viewport
        for row_idx, row in enumerate(observation.grid):
            gy = viewport.y + row_idx
            if gy >= GRID_SIZE:
                break
            for col_idx, terrain_code in enumerate(row):
                gx = viewport.x + col_idx
                if gx >= GRID_SIZE:
                    break
                class_id = TERRAIN_TO_CLASS.get(int(terrain_code), 0)
                self.counts[gy, gx, class_id] += 1.0
                self.observed[gy, gx] += 1

        for settlement in observation.settlements:
            key = (settlement.x, settlement.y)
            bucket = self.settlement_stats.setdefault(key, AggregatedSettlementStats())
            bucket.update(settlement)

    def frequencies(self) -> np.ndarray:
        """Observed class frequencies for all cells."""
        mask = self.observed > 0
        freqs = np.zeros_like(self.counts, dtype=float)
        denom = np.maximum(self.observed[..., None], 1)
        freqs[mask] = self.counts[mask] / denom[mask]
        freqs[~mask] = 1.0 / self.counts.shape[-1]
        return safe_normalize(freqs, axis=-1)

    def coverage_ratio(self) -> float:
        """Fraction of cells observed at least once."""
        return float((self.observed > 0).sum()) / float(GRID_SIZE * GRID_SIZE)

    def settlement_at(self, x: int, y: int) -> AggregatedSettlementStats | None:
        """Look up aggregated settlement evidence at a coordinate."""
        return self.settlement_stats.get((x, y))


class RoundObservationStore:
    """Round-wide shared observation memory across all seeds."""

    def __init__(self, seed_states: Iterable[SeedState]):
        self.seed_states = {state.seed_index: state for state in seed_states}
        self.seed_memory = {
            seed_index: SeedObservationGrid()
            for seed_index in self.seed_states
        }
        self.observations: list[ViewportObservation] = []

    def add_observation(self, observation: ViewportObservation) -> None:
        """Store one parsed observation."""
        self.observations.append(observation)
        self.seed_memory[observation.seed_index].add_observation(observation)

    def add_simulation_result(self, round_id: str, seed_index: int, payload: dict) -> None:
        """Parse and store one raw API simulation response."""
        observation = ViewportObservation.from_simulation_result(round_id, seed_index, payload)
        self.add_observation(observation)

    def get_seed_memory(self, seed_index: int) -> SeedObservationGrid:
        """Return per-seed observation memory."""
        return self.seed_memory[seed_index]

    def build_summary(self) -> ObservationSummary:
        """Aggregate all observations into a simple round-level summary."""
        if not self.observations:
            return ObservationSummary(
                observed_seed_fraction=0.0,
                coverage_ratio=0.0,
                mean_observations_per_covered_cell=0.0,
            )

        seed_coverages = [memory.coverage_ratio() for memory in self.seed_memory.values()]
        covered_seed_fraction = sum(coverage > 0.0 for coverage in seed_coverages) / max(
            len(seed_coverages),
            1,
        )
        combined_counts = sum(memory.counts for memory in self.seed_memory.values())
        total_class_observations = combined_counts.sum()
        if total_class_observations <= 0:
            class_density = np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
        else:
            class_density = combined_counts.sum(axis=(0, 1)) / total_class_observations

        settlement_records = []
        reclaim_hits = 0
        reclaim_total = 0
        rebuild_hits = 0
        rebuild_total = 0

        for seed_index, memory in self.seed_memory.items():
            seed_state = self.seed_states[seed_index]
            for (x, y), stats in memory.settlement_stats.items():
                settlement_records.append(stats)
            observed_positions = np.argwhere(memory.observed > 0)
            for y, x in observed_positions:
                initial_code = int(seed_state.grid[y, x])
                if initial_code == 3:
                    reclaim_total += int(memory.observed[y, x])
                    reclaim_hits += int(memory.counts[y, x, 4])
                    rebuild_hits += int(memory.counts[y, x, 1] + memory.counts[y, x, 2])
                    rebuild_total += int(memory.observed[y, x])

        counts_per_covered_cell = []
        for memory in self.seed_memory.values():
            observed_mask = memory.observed > 0
            if observed_mask.any():
                counts_per_covered_cell.append(float(memory.observed[observed_mask].mean()))

        mean_population = np.mean([stats.mean_population for stats in settlement_records]) if settlement_records else 0.0
        mean_food = np.mean([stats.mean_food for stats in settlement_records]) if settlement_records else 0.0
        mean_wealth = np.mean([stats.mean_wealth for stats in settlement_records]) if settlement_records else 0.0
        mean_defense = np.mean([stats.mean_defense for stats in settlement_records]) if settlement_records else 0.0
        alive_ratio = np.mean([stats.alive_ratio for stats in settlement_records]) if settlement_records else 0.5
        port_ratio = np.mean([stats.port_ratio for stats in settlement_records]) if settlement_records else 0.0
        owner_diversity = np.mean([stats.owner_diversity for stats in settlement_records]) if settlement_records else 0.0

        return ObservationSummary(
            num_viewports=len(self.observations),
            coverage_ratio=float(np.mean(seed_coverages)),
            observed_seed_fraction=float(covered_seed_fraction),
            mean_observations_per_covered_cell=float(np.mean(counts_per_covered_cell)) if counts_per_covered_cell else 0.0,
            empty_density=float(class_density[0]),
            settlement_density=float(class_density[1]),
            port_density=float(class_density[2]),
            ruin_density=float(class_density[3]),
            forest_density=float(class_density[4]),
            alive_ratio=float(alive_ratio),
            port_ratio=float(port_ratio),
            mean_population=float(mean_population),
            mean_food=float(mean_food),
            mean_wealth=float(mean_wealth),
            mean_defense=float(mean_defense),
            owner_diversity=float(owner_diversity),
            reclaim_ratio=float(reclaim_hits / reclaim_total) if reclaim_total else 0.0,
            rebuild_ratio=float(rebuild_hits / rebuild_total) if rebuild_total else 0.0,
        )
