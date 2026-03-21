"""
Super-Prior: Olas beste prediksjonsmodul for Astar Island.
===========================================================
Bygget fra 14 runder × 5 seeds = 70 ground truth datasett (95k datapunkter).

Bruk:
    from super_prior import super_predict, vitality_to_vbin

    # Med oracle vitality (fra GT eller observasjoner)
    vbin = vitality_to_vbin(survival_rate)  # "DEAD", "LOW", "MED", "HIGH"
    pred = super_predict(grid, settlements, vbin)  # (40, 40, 6) tensor

    # Vitality fra observasjoner:
    #   survival_rate = (antall settlement+port observasjoner) / (antall observerte init-settlements)

Score (oracle vitality, seed 0): 83.7 avg
Score (observasjonsbasert):       81.6 avg
Gammel solution.py:               71.0 avg

Features:
    - Vitality bin (DEAD/LOW/MED/HIGH) — bestemmer overlevelsesmønster
    - Terrain type (plains/settlement/ruin/forest/port) → ulik base-distribusjon
    - Chebyshev-avstand til nærmeste settlement → settlements ekspanderer nær senter
    - Coastal (nabo til hav) → påvirker port-sannsynlighet
    - Settlement-density r3 (antall settlements innen dist 3)
    - Forest-density r2 (antall skog-celler innen dist 2)

Per-vbin optimal floor: DEAD=0.001, LOW=0.002, MED=0.003, HIGH=0.003
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NUM_CLASSES = 6
MAP_W, MAP_H = 40, 40

# Optimal floor per vitality bin (grid search over 14 runder)
VBIN_FLOOR = {"DEAD": 0.001, "LOW": 0.002, "MED": 0.003, "HIGH": 0.003}

# Vitality bin boundaries
VBIN_BOUNDARIES = [(0.08, "DEAD"), (0.25, "LOW"), (0.45, "MED")]
# Above 0.45 → "HIGH"

_cal_cache = None


def _load_calibration():
    global _cal_cache
    if _cal_cache is not None:
        return _cal_cache
    cal_path = Path(__file__).parent / "super_calibration.json"
    if not cal_path.exists():
        return None
    data = json.loads(cal_path.read_text())
    _cal_cache = {
        "density": data.get("table_density", {}),
        "specific": data.get("table_specific", {}),
        "simple": data.get("table_simple", {}),
    }
    return _cal_cache


def vitality_to_vbin(survival_rate: float) -> str:
    """Map settlement survival rate → vitality bin.

    survival_rate: fraction of initial settlements that are still settlement/port
                   in ground truth or observations (0.0 to 1.0)
    """
    if survival_rate < 0.08:
        return "DEAD"
    elif survival_rate < 0.25:
        return "LOW"
    elif survival_rate < 0.45:
        return "MED"
    else:
        return "HIGH"


def _terrain_group(t):
    if t in (0, 11): return "plains"
    elif t == 1: return "settlement"
    elif t == 2: return "port"
    elif t == 3: return "ruin"
    elif t == 4: return "forest"
    else: return "other"


def _dist_bin(d):
    if d <= 0: return 0
    elif d <= 1: return 1
    elif d <= 2: return 2
    elif d <= 3: return 3
    elif d <= 5: return 4
    elif d <= 8: return 5
    else: return 6


def _settle_density_bin(n):
    return 0 if n == 0 else (1 if n <= 2 else 2)


def _forest_density_bin(n):
    if n == 0: return 0
    elif n <= 4: return 1
    elif n <= 10: return 2
    else: return 3


def super_predict(grid, settlements, vbin: str, floor: float | None = None) -> np.ndarray:
    """
    Build (H, W, 6) prediction tensor using super-calibration tables.

    Args:
        grid: 40×40 initial terrain grid (list of lists or numpy array)
        settlements: list of {"x", "y", "has_port", ...} dicts
        vbin: "DEAD", "LOW", "MED", or "HIGH"
        floor: min probability per class (default: per-vbin optimal)

    Returns:
        (40, 40, 6) numpy array with probabilities per cell
    """
    if floor is None:
        floor = VBIN_FLOOR.get(vbin, 0.003)

    cal = _load_calibration()
    if cal is None:
        raise FileNotFoundError("super_calibration.json not found")

    table_density = cal["density"]
    table_specific = cal["specific"]
    table_simple = cal["simple"]

    grid_arr = np.array(grid, dtype=int) if not isinstance(grid, np.ndarray) else grid
    H, W = grid_arr.shape if hasattr(grid_arr, 'shape') else (len(grid), len(grid[0]))
    pred = np.zeros((H, W, NUM_CLASSES), dtype=float)

    for y in range(H):
        for x in range(W):
            terrain = int(grid_arr[y, x]) if isinstance(grid_arr, np.ndarray) else int(grid[y][x])

            if terrain == 10:  # ocean
                pred[y, x] = [1.0 - 5 * floor, floor, floor, floor, floor, floor]
                continue
            if terrain == 5:  # mountain
                pred[y, x] = [floor, floor, floor, floor, floor, 1.0 - 5 * floor]
                continue

            # Compute features
            min_dist = 99
            n_settle_r3 = 0
            for s in settlements:
                d = max(abs(y - s["y"]), abs(x - s["x"]))
                if d < min_dist:
                    min_dist = d
                if d <= 3:
                    n_settle_r3 += 1

            coastal = False
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        t = int(grid_arr[ny, nx]) if isinstance(grid_arr, np.ndarray) else int(grid[ny][nx])
                        if t == 10:
                            coastal = True
                            break
                if coastal:
                    break

            n_forest_r2 = 0
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        t = int(grid_arr[ny, nx]) if isinstance(grid_arr, np.ndarray) else int(grid[ny][nx])
                        if t == 4:
                            n_forest_r2 += 1

            tg = _terrain_group(terrain)
            db = _dist_bin(min_dist)
            c = int(coastal)
            sdb = _settle_density_bin(n_settle_r3)
            fdb = _forest_density_bin(n_forest_r2)

            # Cascading lookup: most specific → least specific
            p = None

            key_d = f"{vbin}_{tg}_{db}_{c}_{sdb}_{fdb}"
            if key_d in table_density:
                p = np.array(table_density[key_d]["distribution"])

            if p is None:
                key_s = f"{vbin}_{tg}_{db}_{c}"
                if key_s in table_specific:
                    p = np.array(table_specific[key_s]["distribution"])

            if p is None:
                key_simple = f"{tg}_{db}"
                if key_simple in table_simple:
                    p = np.array(table_simple[key_simple]["distribution"])

            if p is None:
                p = np.ones(NUM_CLASSES) / NUM_CLASSES

            # Floor via Joakim's method: p*(1-6ε)+ε
            p = p * (1 - NUM_CLASSES * floor) + floor
            p /= p.sum()
            pred[y, x] = p

    return pred


def infer_vitality_from_observations(observers) -> float:
    """
    Infer settlement survival rate from observation data.

    Args:
        observers: list of SeedObserver objects (from solution.py)
                   Each must have .settlements, .observed, .counts attributes

    Returns:
        survival_rate: float 0-1 (use with vitality_to_vbin())
    """
    total_init = 0
    total_survived = 0.0

    for obs in observers:
        for s in obs.settlements:
            sx = s.get("x", -1)
            sy = s.get("y", -1)
            if not (0 <= sx < MAP_W and 0 <= sy < MAP_H):
                continue
            if obs.observed[sy, sx] > 0:
                total_init += 1
                n = obs.observed[sy, sx]
                # Count settlement (class 1) + port (class 2) as survived
                total_survived += (obs.counts[sy, sx, 1] + obs.counts[sy, sx, 2]) / n

    if total_init == 0:
        return 0.35  # No data → default to MED-ish

    return total_survived / total_init
