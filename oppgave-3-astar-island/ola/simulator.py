"""
Astar Island Forward Simulator v1
==================================
Empirisk simulator basert på reverse-engineered mekanikk fra 13 runder ground truth.

I stedet for å simulere 50 år med 5 faser, bruker vi EMPIRISKE
overgangssannsynligheter kondisjonert på:
  1. Initial terrain (land/forest/settlement/port)
  2. Chebyshev distance til nærmeste settlement
  3. Coastal vs inland
  4. World vitality parameter (0.0=dead, 1.0=booming)

Dette er en "learned simulator" — den reproduserer ground truth-mønstrene
uten å simulere de underliggende mekanikkene.

Bruk:
    from simulator import predict_distribution
    pred = predict_distribution(grid, settlements, vitality=0.5)
    # pred shape: (40, 40, 6) — probability distribution per cell
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict

MAP_W, MAP_H = 40, 40
NUM_CLASSES = 6  # empty, settlement, port, ruin, forest, mountain

# === EMPIRISKE OVERGANGSTABELLER ===
# Fra analyse av 13 runder × 5 seeds ground truth
# Format: [empty, settlement, port, ruin, forest, mountain]

# Land (terrain=11) ved ulike avstander, ALIVE worlds
LAND_ALIVE = {
    0: [0.3949, 0.3759, 0.0281, 0.0120, 0.1890, 0.0],
    1: [0.5072, 0.2483, 0.0199, 0.0139, 0.1914, 0.0],
    2: [0.5701, 0.1653, 0.0151, 0.0107, 0.2152, 0.0],
    3: [0.6218, 0.1140, 0.0114, 0.0077, 0.2153, 0.0],
    4: [0.6422, 0.0886, 0.0086, 0.0081, 0.2280, 0.0],
    5: [0.6850, 0.0695, 0.0069, 0.0063, 0.2049, 0.0],
    6: [0.6945, 0.0422, 0.0037, 0.0055, 0.2197, 0.0],
    7: [0.6878, 0.0220, 0.0016, 0.0034, 0.2793, 0.0],
    8: [0.7396, 0.0103, 0.0009, 0.0015, 0.2478, 0.0],
    9: [0.8153, 0.0082, 0.0003, 0.0021, 0.1741, 0.0],
}

# Land (terrain=11) ved ulike avstander, DEAD worlds
LAND_DEAD = {
    0: [0.6343, 0.0509, 0.0088, 0.0018, 0.3042, 0.0],
    1: [0.6997, 0.0289, 0.0049, 0.0009, 0.2495, 0.0],
    2: [0.7096, 0.0143, 0.0025, 0.0008, 0.2545, 0.0],
    3: [0.7100, 0.0070, 0.0012, 0.0004, 0.2600, 0.0],
    4: [0.7100, 0.0035, 0.0006, 0.0002, 0.2700, 0.0],
    5: [0.7100, 0.0018, 0.0003, 0.0001, 0.2800, 0.0],
}

# Forest (terrain=4) ved ulike avstander fra settlement, ALIVE
FOREST_ALIVE = {
    0: [0.1500, 0.3000, 0.0150, 0.0100, 0.5250, 0.0],
    1: [0.1206, 0.2636, 0.0120, 0.0100, 0.5794, 0.0],
    2: [0.0869, 0.1747, 0.0090, 0.0080, 0.7131, 0.0],
    3: [0.0631, 0.1202, 0.0060, 0.0060, 0.7969, 0.0],
    4: [0.0450, 0.0800, 0.0040, 0.0040, 0.8600, 0.0],
    5: [0.0350, 0.0550, 0.0025, 0.0030, 0.8980, 0.0],
    6: [0.0250, 0.0350, 0.0015, 0.0020, 0.9300, 0.0],
    7: [0.0154, 0.0154, 0.0008, 0.0010, 0.9674, 0.0],
    8: [0.0100, 0.0080, 0.0004, 0.0005, 0.9811, 0.0],
    9: [0.0080, 0.0040, 0.0002, 0.0003, 0.9875, 0.0],
}

# Forest (terrain=4) ved ulike avstander, DEAD
FOREST_DEAD = {
    0: [0.0650, 0.0300, 0.0020, 0.0010, 0.9020, 0.0],
    1: [0.0500, 0.0150, 0.0010, 0.0005, 0.9335, 0.0],
    2: [0.0400, 0.0080, 0.0005, 0.0003, 0.9512, 0.0],
    3: [0.0300, 0.0040, 0.0003, 0.0001, 0.9656, 0.0],
    4: [0.0250, 0.0020, 0.0001, 0.0001, 0.9728, 0.0],
    5: [0.0200, 0.0010, 0.0001, 0.0001, 0.9788, 0.0],
}

# Settlement-celler (terrain=1), betinget på vitality
# Vitality 0.0=dead, 0.5=stable, 1.0=booming
SETTLEMENT_BY_VITALITY = {
    # vitality: [empty, settlement, port, ruin, forest, mountain]
    0.0: [0.5640, 0.0200, 0.0030, 0.0060, 0.4070, 0.0],
    0.2: [0.5000, 0.1000, 0.0040, 0.0150, 0.3810, 0.0],
    0.4: [0.4200, 0.2500, 0.0050, 0.0250, 0.3000, 0.0],
    0.5: [0.3850, 0.3980, 0.0040, 0.0310, 0.1830, 0.0],
    0.6: [0.3500, 0.4200, 0.0060, 0.0350, 0.1890, 0.0],
    0.8: [0.3090, 0.5020, 0.0160, 0.0290, 0.1440, 0.0],
    1.0: [0.2600, 0.5700, 0.0200, 0.0350, 0.1150, 0.0],
}

# Port-settlement-celler (terrain=2 / has_port=True)
PORT_SETTLEMENT_BY_VITALITY = {
    0.0: [0.5680, 0.0100, 0.0050, 0.0050, 0.4120, 0.0],
    0.5: [0.4600, 0.0510, 0.1010, 0.0200, 0.2700, 0.0],
    1.0: [0.3500, 0.0800, 0.2000, 0.0300, 0.1800, 0.0],
}


def _chebyshev_distance(y, x, settlements):
    """Chebyshev distance til nærmeste settlement."""
    if not settlements:
        return 99
    return min(max(abs(x - s["x"]), abs(y - s["y"])) for s in settlements)


def _is_coastal(grid, y, x):
    """Sjekk om cellen grenser til vann (terrain 10 eller 5)."""
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < MAP_H and 0 <= nx < MAP_W:
                if grid[ny][nx] in (10, 5):
                    return True
    return False


def _interpolate_vitality(table, vitality):
    """Interpoler mellom to nærmeste vitality-verdier i tabell."""
    keys = sorted(table.keys())
    if vitality <= keys[0]:
        return np.array(table[keys[0]], dtype=float)
    if vitality >= keys[-1]:
        return np.array(table[keys[-1]], dtype=float)

    # Finn de to nærmeste
    for i in range(len(keys) - 1):
        if keys[i] <= vitality <= keys[i + 1]:
            lo, hi = keys[i], keys[i + 1]
            t = (vitality - lo) / (hi - lo)
            arr = (1 - t) * np.array(table[lo]) + t * np.array(table[hi])
            return arr
    return np.array(table[keys[-1]], dtype=float)


def _get_dist_table(table, dist):
    """Hent distribusjon fra avstandstabell, clamp til maks key."""
    max_key = max(table.keys())
    d = min(dist, max_key)
    if d in table:
        return np.array(table[d], dtype=float)
    # Interpoler
    keys = sorted(table.keys())
    for i in range(len(keys) - 1):
        if keys[i] <= d <= keys[i + 1]:
            lo, hi = keys[i], keys[i + 1]
            t = (d - lo) / (hi - lo)
            return (1 - t) * np.array(table[lo]) + t * np.array(table[hi])
    return np.array(table[max_key], dtype=float)


def predict_distribution(grid, settlements, vitality=0.5,
                         coastal_ruin_boost=True, floor=0.005):
    """
    Prediker 40×40×6 sannsynlighetsfordeling basert på:
    - Initial grid og settlements
    - Vitality parameter (0.0=dead, 1.0=booming)

    Returns: np.array shape (40, 40, 6)
    """
    grid_arr = np.array(grid, dtype=int) if not isinstance(grid, np.ndarray) else grid
    pred = np.zeros((MAP_H, MAP_W, NUM_CLASSES), dtype=float)

    # Velg tabeller basert på vitality
    is_dead = vitality < 0.15

    for y in range(MAP_H):
        for x in range(MAP_W):
            terrain = int(grid_arr[y, x])

            # Statiske celler
            if terrain == 10 or terrain == 5:  # Ocean / shallow water
                pred[y, x] = [0.998, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004]
                continue

            # Mountain — terrain code 5 er egentlig vann per analyse
            # Men sjekk om det finnes ekte mountain-celler
            # TERRAIN_TO_CLASS: 5 → mountain. Behold for sikkerhets skyld.

            dist = _chebyshev_distance(y, x, settlements)
            coastal = _is_coastal(grid_arr.tolist(), y, x)

            if terrain == 1:
                # Settlement-celle
                p = _interpolate_vitality(SETTLEMENT_BY_VITALITY, vitality)
            elif terrain == 2:
                # Port-settlement
                p = _interpolate_vitality(PORT_SETTLEMENT_BY_VITALITY, vitality)
            elif terrain == 4:
                # Forest
                if is_dead:
                    p = _get_dist_table(FOREST_DEAD, dist)
                else:
                    # Interpoler mellom dead og alive basert på vitality
                    dead_p = _get_dist_table(FOREST_DEAD, dist)
                    alive_p = _get_dist_table(FOREST_ALIVE, dist)
                    t = min(1.0, vitality / 0.5)  # 0→dead, 0.5+→alive
                    p = (1 - t) * dead_p + t * alive_p
            else:
                # Land (terrain=11, 0, 3, etc.)
                if is_dead:
                    p = _get_dist_table(LAND_DEAD, dist)
                else:
                    dead_p = _get_dist_table(LAND_DEAD, dist)
                    alive_p = _get_dist_table(LAND_ALIVE, dist)
                    t = min(1.0, vitality / 0.5)
                    p = (1 - t) * dead_p + t * alive_p

            # Coastal adjustments
            if coastal and not is_dead:
                # Boost port probability for coastal cells near settlements
                if dist <= 3:
                    port_boost = max(0, 0.03 - 0.008 * dist)
                    p[2] += port_boost
                # Ruin er KUN kystfenomen (fra data: inland ruin = 0.0000)
                if coastal_ruin_boost and dist <= 5:
                    pass  # Ruin allerede i tabellen for coastal

            if not coastal:
                # Inland: ruin og port er nesten null
                p[2] = 0.0  # Port umulig inland
                p[3] = 0.0  # Ruin umulig inland (per data)
                # Redistribuer til empty og forest
                surplus = 1.0 - p.sum()
                if surplus < 0:
                    p[0] += surplus * 0.7
                    p[4] += surplus * 0.3

            # Mountain alltid umulig på dynamisk celle
            p[5] = 0.0

            # Normaliser
            p = np.maximum(p, 0.0)
            s = p.sum()
            if s > 0:
                p /= s

            # Floor via linear mixing
            eps = floor
            p = p * (1 - NUM_CLASSES * eps) + eps
            p /= p.sum()

            pred[y, x] = p

    return pred


def infer_vitality_from_observations(observers, settlements_per_seed):
    """
    Inferér vitality fra observasjoner.

    Sjekk: av observerte settlement-celler, hvor mange er fortsatt settlements?
    - survival_rate > 0.4 → booming (vitality 0.8-1.0)
    - survival_rate 0.2-0.4 → stable (vitality 0.4-0.6)
    - survival_rate < 0.1 → dead (vitality 0.0-0.1)
    """
    total_initial_settlements = 0
    total_survived = 0

    for obs, settlements in zip(observers, settlements_per_seed):
        for s in settlements:
            sx, sy = s["x"], s["y"]
            if 0 <= sx < MAP_W and 0 <= sy < MAP_H:
                if obs.observed[sy, sx] > 0:
                    total_initial_settlements += 1
                    # Sjekk om cellen fortsatt er settlement (class 1)
                    if obs.counts[sy, sx, 1] > 0:
                        total_survived += obs.counts[sy, sx, 1] / obs.observed[sy, sx]

    if total_initial_settlements == 0:
        return 0.5  # Ingen data, default

    survival_rate = total_survived / total_initial_settlements

    # Map survival rate til vitality
    if survival_rate < 0.05:
        return 0.0
    elif survival_rate < 0.15:
        return 0.1 + survival_rate
    elif survival_rate < 0.30:
        return 0.3 + survival_rate * 0.5
    elif survival_rate < 0.50:
        return 0.5 + (survival_rate - 0.3) * 1.0
    else:
        return min(1.0, 0.7 + (survival_rate - 0.5) * 0.6)


def score_from_kl(wkl):
    """Convert weighted KL divergence to score."""
    return max(0, min(100, 100 * np.exp(-3 * wkl)))


def weighted_kl(ground_truth, prediction):
    """Entropy-weighted KL divergence (competition metric)."""
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
