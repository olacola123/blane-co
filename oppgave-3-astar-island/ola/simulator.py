"""
Astar Island Forward Simulator v2
====================================
Two-layer prediction system:

LAYER 1: Empirical model (fast, prior-only)
  - Exact lookup tables from 13 rounds of ground truth
  - Conditioned on: initial_terrain × distance × world_type
  - Used as prior / fallback

LAYER 2: Monte Carlo simulator (slow, observation-aware)
  - Cellular automaton running 50 years × 5 phases
  - Hidden parameters inferred from observations
  - Run N times → probability distribution

Ground truth analysis (13 rounds × seed 0):
  - Dead worlds (R3,R8,R10): settlement survival < 8%
  - Stable worlds (R4,R9,R13): survival 20-30%
  - Boom worlds (R1,R2,R5,R6,R7,R11,R12): survival 36-58%
  - Ocean/mountain: 100% static
  - Seeds have DIFFERENT grids within a round (~700 cell diffs)
  - Ports appear inland too (avg 1.5% in alive worlds)
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np

MAP_W, MAP_H = 40, 40
NUM_CLASSES = 6  # [empty, settlement, ruin, port, forest, mountain]

# Terrain codes (from API)
OCEAN = 10
PLAINS = 11
EMPTY_T = 0
SETTLEMENT_T = 1
PORT_T = 2       # terrain code 2 = Port = class 2
RUIN_T = 3       # terrain code 3 = Ruin = class 3
FOREST_T = 4
MOUNTAIN_T = 5

STATIC_TERRAIN = {OCEAN, MOUNTAIN_T}

# ====================================================================
# LAYER 1: EMPIRICAL MODEL — exact data from ground truth analysis
# ====================================================================
# Format: dist → [empty, settlement, port, ruin, forest, mountain]
# Class indices: 0=empty, 1=settlement, 2=port, 3=ruin, 4=forest, 5=mountain
# Values computed directly from GT[row][col][class_idx] — indices match API format

# ALL DYNAMIC CELLS by distance to nearest settlement — ALIVE worlds (R1,2,4,5,6,7,9,11,12,13)
ALIVE_BY_DIST = {
    0: [0.39495, 0.37592, 0.01196, 0.02813, 0.18905, 0.0],
    1: [0.51721, 0.25314, 0.01421, 0.02026, 0.19518, 0.0],
    2: [0.58384, 0.16931, 0.01098, 0.01548, 0.22040, 0.0],
    3: [0.64098, 0.11753, 0.00791, 0.01171, 0.22188, 0.0],
    4: [0.65834, 0.09079, 0.00835, 0.00879, 0.23373, 0.0],
    5: [0.70422, 0.07149, 0.00653, 0.00709, 0.21068, 0.0],
    6: [0.71921, 0.04375, 0.00573, 0.00378, 0.22752, 0.0],
    7: [0.69185, 0.02214, 0.00342, 0.00158, 0.28101, 0.0],
    8: [0.73958, 0.01025, 0.00150, 0.00092, 0.24775, 0.0],
    9: [0.81529, 0.00824, 0.00206, 0.00029, 0.17412, 0.0],
}

# ALL DYNAMIC CELLS by distance — DEAD worlds (R3,R8,R10)
DEAD_BY_DIST = {
    0: [0.63430, 0.05088, 0.00180, 0.00879, 0.30423, 0.0],
    1: [0.71114, 0.02936, 0.00094, 0.00498, 0.25358, 0.0],
    2: [0.72280, 0.01459, 0.00081, 0.00257, 0.25922, 0.0],
    3: [0.73402, 0.00688, 0.00064, 0.00118, 0.25728, 0.0],
    4: [0.73270, 0.00229, 0.00022, 0.00042, 0.26437, 0.0],
    5: [0.66060, 0.00109, 0.00016, 0.00018, 0.33797, 0.0],
    6: [0.80964, 0.00012, 0.00000, 0.00000, 0.19024, 0.0],
    7: [0.80000, 0.00000, 0.00000, 0.00000, 0.20000, 0.0],
    8: [1.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.0],
}

# INITIAL FOREST cells by distance — ALIVE worlds
FOREST_ALIVE_BY_DIST = {
    1: [0.12154, 0.26360, 0.01411, 0.02138, 0.57937, 0.0],
    2: [0.08658, 0.17465, 0.00996, 0.01567, 0.71313, 0.0],
    3: [0.06365, 0.12016, 0.00710, 0.01215, 0.79694, 0.0],
    4: [0.04057, 0.09177, 0.00940, 0.00894, 0.84933, 0.0],
    5: [0.02574, 0.07420, 0.00488, 0.00675, 0.88843, 0.0],
    6: [0.01119, 0.04275, 0.00531, 0.00325, 0.93750, 0.0],
    7: [0.00260, 0.01542, 0.00292, 0.00125, 0.97781, 0.0],
    8: [0.00167, 0.00867, 0.00167, 0.00100, 0.98700, 0.0],
    9: [0.00000, 0.01333, 0.00333, 0.00000, 0.98333, 0.0],
}

# INITIAL FOREST cells by distance — DEAD worlds
FOREST_DEAD_BY_DIST = {
    1: [0.10899, 0.03030, 0.00063, 0.00549, 0.85460, 0.0],
    2: [0.03807, 0.01430, 0.00085, 0.00222, 0.94457, 0.0],
    3: [0.02180, 0.00612, 0.00073, 0.00107, 0.97028, 0.0],
    4: [0.00723, 0.00256, 0.00017, 0.00067, 0.98937, 0.0],
    5: [0.00300, 0.00123, 0.00015, 0.00023, 0.99538, 0.0],
    6: [0.00219, 0.00031, 0.00000, 0.00000, 0.99750, 0.0],
    7: [0.00000, 0.00000, 0.00000, 0.00000, 1.00000, 0.0],
}

# Initial terrain type → GT (aggregated across ALL rounds, ALL distances)
TERRAIN_TO_GT_ALL = {
    SETTLEMENT_T: [0.45179, 0.30208, 0.00598, 0.02363, 0.21652, 0.0],
    RUIN_T:       [0.56763, 0.05079, 0.10132, 0.01158, 0.26868, 0.0],
    FOREST_T:     [0.06953, 0.12682, 0.00761, 0.01155, 0.78448, 0.0],
    PLAINS:       [0.82921, 0.12001, 0.00805, 0.01100, 0.03173, 0.0],
}


def _chebyshev_dist(y, x, settlements):
    if not settlements:
        return 99
    return min(max(abs(x - s["x"]), abs(y - s["y"])) for s in settlements)


def _is_coastal(grid, y, x, H=40, W=40):
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                if grid[ny][nx] == OCEAN:
                    return True
    return False


def _get_from_table(table, dist):
    """Get distribution from distance table with clamping."""
    max_d = max(table.keys())
    min_d = min(table.keys())
    d = max(min_d, min(dist, max_d))
    if d in table:
        return np.array(table[d], dtype=float)
    # Interpolate
    keys = sorted(table.keys())
    for i in range(len(keys) - 1):
        if keys[i] <= d <= keys[i + 1]:
            lo, hi = keys[i], keys[i + 1]
            t = (d - lo) / (hi - lo)
            return (1 - t) * np.array(table[lo]) + t * np.array(table[hi])
    return np.array(table[max_d], dtype=float)


def empirical_predict(grid, settlements, vitality=0.5, floor=0.005):
    """
    Layer 1: Empirical model using exact ground truth statistics.

    Args:
        grid: 40×40 initial grid (terrain codes)
        settlements: list of {x, y, has_port, ...}
        vitality: 0.0=dead, 0.5=stable, 1.0=booming
        floor: minimum probability per class

    Returns: (40, 40, 6) probability tensor
    """
    grid_arr = np.array(grid, dtype=int) if not isinstance(grid, np.ndarray) else grid
    H, W = grid_arr.shape
    pred = np.zeros((H, W, NUM_CLASSES), dtype=float)

    is_dead = vitality < 0.15

    for y in range(H):
        for x in range(W):
            terrain = int(grid_arr[y, x])

            # Static: ocean
            if terrain == OCEAN:
                pred[y, x] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                continue

            # Static: mountain
            if terrain == MOUNTAIN_T:
                pred[y, x] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                continue

            dist = _chebyshev_dist(y, x, settlements)

            if terrain == FOREST_T:
                # Initial forest cell — use terrain-specific tables
                if is_dead:
                    p = _get_from_table(FOREST_DEAD_BY_DIST, dist)
                else:
                    dead_p = _get_from_table(FOREST_DEAD_BY_DIST, dist)
                    alive_p = _get_from_table(FOREST_ALIVE_BY_DIST, dist)
                    # Blend based on vitality (0→dead, 0.5→alive)
                    t = min(1.0, vitality / 0.5)
                    p = (1 - t) * dead_p + t * alive_p

            elif terrain == SETTLEMENT_T:
                # Initial settlement cell — highly dependent on vitality
                # Use the aggregate data: survival at dist=0
                if is_dead:
                    p = _get_from_table(DEAD_BY_DIST, 0)
                else:
                    dead_p = _get_from_table(DEAD_BY_DIST, 0)
                    alive_p = _get_from_table(ALIVE_BY_DIST, 0)
                    t = min(1.0, vitality / 0.5)
                    p = (1 - t) * dead_p + t * alive_p
                # Boost settlement/ruin for settlement cells relative to generic dist=0
                # GT data: settlement cells avg [0.452, 0.302, 0.006, 0.024, 0.217, 0]
                # Generic dist=0 alive: [0.395, 0.376, 0.012, 0.028, 0.189, 0]
                # Settlement cells are slightly less likely to still be settlements
                # but we want terrain-specific behavior
                if not is_dead:
                    terrain_specific = np.array(TERRAIN_TO_GT_ALL[SETTLEMENT_T])
                    # Blend 50% generic distance + 50% terrain-specific
                    p = 0.5 * p + 0.5 * terrain_specific

            elif terrain == RUIN_T:
                # Initial ruin cell
                if is_dead:
                    p = _get_from_table(DEAD_BY_DIST, dist)
                    # Boost ruin probability for initial ruin cells
                    p[2] *= 3.0  # ruin stays ruin more
                else:
                    alive_p = _get_from_table(ALIVE_BY_DIST, dist)
                    # Terrain-specific: ruins have 10% chance to stay ruin
                    terrain_spec = np.array(TERRAIN_TO_GT_ALL[RUIN_T])
                    p = 0.4 * alive_p + 0.6 * terrain_spec

            elif terrain == PORT_T:
                # Initial port cell — similar to settlement but with port boost
                if is_dead:
                    p = _get_from_table(DEAD_BY_DIST, 0)
                else:
                    alive_p = _get_from_table(ALIVE_BY_DIST, 0)
                    p = alive_p.copy()
                    p[3] *= 2.0  # boost port probability
                    p[1] *= 0.8  # slightly less regular settlement

            else:
                # Plains (11), empty (0), or other land
                if is_dead:
                    p = _get_from_table(DEAD_BY_DIST, dist)
                else:
                    dead_p = _get_from_table(DEAD_BY_DIST, dist)
                    alive_p = _get_from_table(ALIVE_BY_DIST, dist)
                    t = min(1.0, vitality / 0.5)
                    p = (1 - t) * dead_p + t * alive_p

            # Mountain class always 0 for dynamic cells
            p[5] = 0.0

            # Normalize
            p = np.maximum(p, 0.0)
            total = p.sum()
            if total > 0:
                p /= total

            # Apply floor via Joakim's method: p*(1-6ε)+ε
            eps = floor
            p = p * (1 - NUM_CLASSES * eps) + eps
            p /= p.sum()

            pred[y, x] = p

    return pred


# ====================================================================
# LAYER 2: MONTE CARLO FORWARD SIMULATOR
# ====================================================================

@dataclass
class SimParams:
    """Hidden simulation parameters shared across seeds in a round."""
    winter_severity: float = 0.3    # 0-1: how deadly winters are
    growth_rate: float = 0.08       # expansion probability per year
    raid_intensity: float = 0.1     # raid frequency
    raid_kill_prob: float = 0.3     # probability raid kills defender
    forest_spread: float = 0.03     # forest expansion rate
    forest_reclaim: float = 0.05    # ruin → forest rate
    port_develop: float = 0.02      # coastal settlement → port
    trade_bonus: float = 0.15       # food bonus from ports/trade
    settle_reclaim: float = 0.04    # ruin → settlement (if adj settlement)
    forest_food: float = 0.1        # food bonus from adjacent forest


@dataclass
class Cell:
    """Mutable cell state for simulation."""
    population: float = 1.0
    food: float = 1.0
    wealth: float = 0.0
    defense: float = 0.5
    has_port: bool = False
    alive: bool = True
    owner_id: int = 0


class ForwardSimulator:
    """Run N independent simulations → probability distributions."""

    def __init__(self, initial_grid, initial_settlements,
                 params: SimParams, n_years=50):
        self.initial_grid = np.array(initial_grid, dtype=int)
        self.initial_settlements = initial_settlements
        self.params = params
        self.n_years = n_years
        self.H, self.W = self.initial_grid.shape

        # Precompute static masks
        self.land_mask = ~np.isin(self.initial_grid, [OCEAN, MOUNTAIN_T])
        self.coastal = np.zeros((self.H, self.W), dtype=bool)
        for r in range(self.H):
            for c in range(self.W):
                if self.land_mask[r, c]:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < self.H and 0 <= nc < self.W:
                                if self.initial_grid[nr, nc] == OCEAN:
                                    self.coastal[r, c] = True

    def _init_state(self, rng):
        """Create fresh simulation state."""
        grid = self.initial_grid.copy()
        cells = {}

        for i, s in enumerate(self.initial_settlements):
            sx, sy = s["x"], s["y"]
            has_port = s.get("has_port", False)
            cells[(sy, sx)] = Cell(
                population=1.0 + rng.random() * 2.0,
                food=0.5 + rng.random() * 1.5,
                wealth=rng.random() * 0.5,
                defense=0.3 + rng.random() * 0.4,
                has_port=has_port,
                owner_id=i,
            )
            grid[sy, sx] = PORT_T if has_port else SETTLEMENT_T

        return grid, cells

    def _neighbors(self, r, c):
        result = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.H and 0 <= nc < self.W and self.land_mask[nr, nc]:
                    result.append((nr, nc))
        return result

    def _count_adj_type(self, grid, r, c, terrain):
        return sum(1 for nr, nc in self._neighbors(r, c) if grid[nr, nc] == terrain)

    def _count_adj_settlements(self, grid, r, c):
        return sum(1 for nr, nc in self._neighbors(r, c)
                   if grid[nr, nc] in (SETTLEMENT_T, PORT_T))

    def _run_one(self, seed):
        """Run one full simulation."""
        rng = random.Random(seed)
        p = self.params
        grid, cells = self._init_state(rng)

        for year in range(self.n_years):
            # === GROWTH ===
            active = [(pos, c) for pos, c in cells.items() if c.alive]
            for (r, c_), cell in active:
                adj_forest = self._count_adj_type(grid, r, c_, FOREST_T)
                food_prod = 0.3 + adj_forest * 0.15
                if cell.has_port:
                    food_prod += p.trade_bonus
                cell.food = min(cell.food + food_prod, 3.0)

                if cell.food > 0.5:
                    growth = 0.1 * cell.food * (1.0 - cell.population / 5.0)
                    cell.population = min(cell.population + max(0, growth), 5.0)

                cell.defense = min(cell.defense + 0.02 * cell.wealth, 1.0)

                # Expansion
                if cell.population > 1.5 and cell.food > 0.8:
                    exp_chance = p.growth_rate * (cell.population / 3.0)
                    if rng.random() < exp_chance:
                        candidates = [
                            (nr, nc) for nr, nc in self._neighbors(r, c_)
                            if grid[nr, nc] in (EMPTY_T, PLAINS)
                        ]
                        if candidates:
                            nr, nc = rng.choice(candidates)
                            new_cell = Cell(
                                population=0.5 + rng.random() * 0.5,
                                food=cell.food * 0.3,
                                wealth=cell.wealth * 0.1,
                                defense=0.2 + rng.random() * 0.2,
                                owner_id=cell.owner_id,
                            )
                            cells[(nr, nc)] = new_cell
                            grid[nr, nc] = SETTLEMENT_T
                            cell.population *= 0.8
                            cell.food *= 0.7

                # Port development
                if not cell.has_port and self.coastal[r, c_]:
                    if rng.random() < p.port_develop * (1 + cell.wealth):
                        cell.has_port = True
                        grid[r, c_] = PORT_T

            # === CONFLICT ===
            active = [(pos, c) for pos, c in cells.items() if c.alive]
            rng.shuffle(active)
            for (r, c_), attacker in active:
                if not attacker.alive:
                    continue
                desperation = max(0, 1.0 - attacker.food)
                if rng.random() < p.raid_intensity * (0.3 + 0.7 * desperation):
                    targets = [
                        ((nr, nc), cells[(nr, nc)])
                        for nr, nc in self._neighbors(r, c_)
                        if (nr, nc) in cells and cells[(nr, nc)].alive
                        and cells[(nr, nc)].owner_id != attacker.owner_id
                    ]
                    if targets:
                        (tr, tc), defender = rng.choice(targets)
                        att_str = attacker.population * (0.5 + 0.5 * attacker.defense)
                        def_str = defender.population * (0.5 + 0.5 * defender.defense)
                        if att_str > def_str * (0.8 + rng.random() * 0.4):
                            attacker.food += defender.food * 0.5
                            attacker.wealth += defender.wealth * 0.5
                            if rng.random() < p.raid_kill_prob:
                                defender.alive = False
                                grid[tr, tc] = RUIN_T
                            else:
                                defender.population *= 0.5
                                defender.food *= 0.3
                        else:
                            attacker.population *= 0.7
                            attacker.food *= 0.5

            # === TRADE ===
            ports = [(pos, c) for pos, c in cells.items() if c.alive and c.has_port]
            for (r, c_), port in ports:
                if port.alive:
                    port.wealth += 0.1 + 0.05 * len(ports)
                    port.food += p.trade_bonus

            # === WINTER ===
            severity = p.winter_severity * (0.5 + rng.random())
            active = [(pos, c) for pos, c in cells.items() if c.alive]
            for (r, c_), cell in active:
                food_needed = 0.3 + 0.1 * cell.population
                cell.food -= food_needed
                adj_forest = self._count_adj_type(grid, r, c_, FOREST_T)
                cell.food += adj_forest * p.forest_food * 0.5

                food_factor = max(0, 1.0 - cell.food) if cell.food < 0.5 else 0
                death_chance = severity * (0.2 + 0.8 * food_factor)
                if cell.has_port:
                    death_chance *= 0.6
                death_chance *= (1.0 - 0.3 * cell.defense)

                if rng.random() < death_chance:
                    cell.alive = False
                    grid[r, c_] = RUIN_T
                else:
                    cell.food = max(cell.food, 0.0)

            # === ENVIRONMENT ===
            # Ruins → settlement or forest
            ruin_cells = [(r, c) for r in range(self.H) for c in range(self.W)
                          if grid[r, c] == RUIN_T]
            for r, c_ in ruin_cells:
                adj_sett = self._count_adj_settlements(grid, r, c_)
                if adj_sett > 0 and rng.random() < p.settle_reclaim * adj_sett:
                    best_owner = 0
                    best_str = 0
                    for nr, nc in self._neighbors(r, c_):
                        if (nr, nc) in cells and cells[(nr, nc)].alive:
                            s = cells[(nr, nc)]
                            str_ = s.population * (0.5 + 0.5 * s.defense)
                            if str_ > best_str:
                                best_str = str_
                                best_owner = s.owner_id
                    new_cell = Cell(
                        population=0.3 + rng.random() * 0.3,
                        food=0.3, wealth=0.0, defense=0.2,
                        owner_id=best_owner,
                    )
                    if self.coastal[r, c_] and rng.random() < 0.3:
                        new_cell.has_port = True
                        grid[r, c_] = PORT_T
                    else:
                        grid[r, c_] = SETTLEMENT_T
                    cells[(r, c_)] = new_cell
                    continue

                adj_forest = self._count_adj_type(grid, r, c_, FOREST_T)
                if rng.random() < p.forest_reclaim * (1 + adj_forest * 0.5):
                    grid[r, c_] = FOREST_T
                    if (r, c_) in cells:
                        del cells[(r, c_)]

            # Forest spread
            forest_cells = [(r, c) for r in range(self.H) for c in range(self.W)
                            if grid[r, c] == FOREST_T]
            new_forests = []
            for r, c_ in forest_cells:
                for nr, nc in self._neighbors(r, c_):
                    if grid[nr, nc] in (EMPTY_T, PLAINS):
                        adj_f = self._count_adj_type(grid, nr, nc, FOREST_T)
                        if rng.random() < p.forest_spread * (0.5 + 0.5 * adj_f / 8):
                            new_forests.append((nr, nc))

            for r, c_ in new_forests:
                if grid[r, c_] in (EMPTY_T, PLAINS):
                    grid[r, c_] = FOREST_T

        return grid

    def run_monte_carlo(self, n_sims=200, floor=0.005):
        """Run N simulations and return probability tensor H×W×6."""
        counts = np.zeros((self.H, self.W, NUM_CLASSES), dtype=float)

        # Mapping: terrain code → class index
        # Class 2 = Port (terrain 2), Class 3 = Ruin (terrain 3)
        code_to_class = {
            OCEAN: 0, PLAINS: 0, EMPTY_T: 0,
            SETTLEMENT_T: 1, PORT_T: 2, RUIN_T: 3,
            FOREST_T: 4, MOUNTAIN_T: 5,
        }

        for i in range(n_sims):
            final = self._run_one(seed=i * 12345 + 42)
            for r in range(self.H):
                for c in range(self.W):
                    cls = code_to_class.get(int(final[r, c]), 0)
                    counts[r, c, cls] += 1

        probs = counts / n_sims

        # Apply floor
        eps = floor
        probs = probs * (1 - NUM_CLASSES * eps) + eps
        probs /= probs.sum(axis=-1, keepdims=True)

        # Force static cells
        for r in range(self.H):
            for c in range(self.W):
                if self.initial_grid[r, c] == OCEAN:
                    probs[r, c] = [1.0 - 5 * eps, eps, eps, eps, eps, eps]
                elif self.initial_grid[r, c] == MOUNTAIN_T:
                    probs[r, c] = [eps, eps, eps, eps, eps, 1.0 - 5 * eps]

        return probs


# ====================================================================
# PARAMETER PRESETS
# ====================================================================

DEAD_PARAMS = SimParams(
    winter_severity=0.65, growth_rate=0.03, raid_intensity=0.15,
    raid_kill_prob=0.5, forest_spread=0.04, forest_reclaim=0.08,
    port_develop=0.005, trade_bonus=0.1, settle_reclaim=0.01,
    forest_food=0.05,
)

STABLE_PARAMS = SimParams(
    winter_severity=0.35, growth_rate=0.06, raid_intensity=0.1,
    raid_kill_prob=0.3, forest_spread=0.03, forest_reclaim=0.05,
    port_develop=0.015, trade_bonus=0.15, settle_reclaim=0.03,
    forest_food=0.1,
)

BOOM_PARAMS = SimParams(
    winter_severity=0.15, growth_rate=0.10, raid_intensity=0.08,
    raid_kill_prob=0.25, forest_spread=0.02, forest_reclaim=0.04,
    port_develop=0.025, trade_bonus=0.2, settle_reclaim=0.05,
    forest_food=0.12,
)


# ====================================================================
# VITALITY INFERENCE
# ====================================================================

def infer_vitality_from_observations(observers, settlements_per_seed):
    """
    Infer vitality from observations.
    Check: of observed initial settlement cells, how many are still settlements?
    """
    total = 0
    survived = 0.0

    for obs, settlements in zip(observers, settlements_per_seed):
        for s in settlements:
            sx, sy = s["x"], s["y"]
            if 0 <= sx < MAP_W and 0 <= sy < MAP_H:
                if hasattr(obs, 'observed') and obs.observed[sy, sx] > 0:
                    total += 1
                    if hasattr(obs, 'counts'):
                        survived += obs.counts[sy, sx, 1] / obs.observed[sy, sx]

    if total == 0:
        return 0.5  # No data

    rate = survived / total
    if rate < 0.05:
        return 0.0
    elif rate < 0.15:
        return 0.1 + rate
    elif rate < 0.30:
        return 0.3 + rate * 0.5
    elif rate < 0.50:
        return 0.5 + (rate - 0.3) * 1.0
    else:
        return min(1.0, 0.7 + (rate - 0.5) * 0.6)


def params_from_vitality(vitality):
    """Get simulation parameters based on vitality."""
    if vitality < 0.15:
        return DEAD_PARAMS
    elif vitality < 0.35:
        # Interpolate dead → stable
        t = (vitality - 0.15) / 0.20
        return _lerp_params(DEAD_PARAMS, STABLE_PARAMS, t)
    elif vitality < 0.55:
        return STABLE_PARAMS
    elif vitality < 0.75:
        t = (vitality - 0.55) / 0.20
        return _lerp_params(STABLE_PARAMS, BOOM_PARAMS, t)
    else:
        return BOOM_PARAMS


def _lerp_params(a: SimParams, b: SimParams, t: float) -> SimParams:
    """Linear interpolation between two parameter sets."""
    return SimParams(**{
        field: getattr(a, field) * (1 - t) + getattr(b, field) * t
        for field in SimParams.__dataclass_fields__
    })


# ====================================================================
# SCORING UTILITIES
# ====================================================================

def weighted_kl(ground_truth, prediction):
    gt = np.array(ground_truth, dtype=float)
    pred = np.array(prediction, dtype=float)
    gt_safe = np.clip(gt, 1e-12, 1.0)
    pred_safe = np.clip(pred, 1e-12, 1.0)
    cell_kl = np.sum(gt_safe * (np.log(gt_safe) - np.log(pred_safe)), axis=-1)
    cell_entropy = -np.sum(gt_safe * np.log(gt_safe), axis=-1)
    tw = cell_entropy.sum()
    if tw <= 0:
        return float(cell_kl.mean())
    return float((cell_kl * cell_entropy).sum() / tw)


def score_from_kl(wkl):
    return max(0, min(100, 100 * np.exp(-3 * wkl)))


if __name__ == "__main__":
    print("Simulator v2 loaded OK")
    print(f"Presets: DEAD, STABLE, BOOM")
    print(f"Empirical tables: {len(ALIVE_BY_DIST)} alive distances, {len(DEAD_BY_DIST)} dead distances")
