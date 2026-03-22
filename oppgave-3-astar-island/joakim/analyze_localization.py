"""
Analyze settlement localization patterns from ground truth.

Measures "urbanization" (settlements stay close to initial positions) vs
"ruralization" (settlements spread randomly) for each completed round.

This computes the GROUND TRUTH localization metric per round,
so we can understand what hidden parameter to predict.

Usage:
    export API_KEY='din-jwt-token'
    python analyze_localization.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://api.ainm.no/astar-island"
API_KEY = os.environ.get("API_KEY", "")
MAP_W, MAP_H = 40, 40
NUM_CLASSES = 6


class AstarClient:
    def __init__(self):
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.headers.update({
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        })

    def get(self, path):
        r = self.session.get(f"{BASE_URL}/{path.lstrip('/')}", timeout=30)
        r.raise_for_status()
        return r.json()


def compute_localization_metrics(initial_grid, initial_settlements, gt_probs):
    """
    Compute multiple metrics of settlement localization from ground truth.

    Returns a dict with:
    - expansion_radius: mean distance of final settlement probability from nearest initial settlement
    - concentration_ratio: fraction of total settlement probability within dist<=5 of initial settlements
    - new_settlement_spread: mean distance of NEW settlement probability (cells that were NOT initially settlements)
    - settlement_gini: Gini coefficient of settlement probability spatial distribution (higher = more concentrated)
    - active_cell_clustering: fraction of active cells (>10% settlement+port prob) that have an active neighbor
    """
    grid = np.array(initial_grid, dtype=int)
    gt = np.array(gt_probs, dtype=float)

    # Compute Manhattan distance to nearest initial settlement for each cell
    dist_map = np.full((MAP_H, MAP_W), 99.0)
    for s in initial_settlements:
        sy, sx = s["y"], s["x"]
        for y in range(MAP_H):
            for x in range(MAP_W):
                d = abs(y - sy) + abs(x - sx)
                if d < dist_map[y, x]:
                    dist_map[y, x] = d

    # Static mask
    static_mask = (grid == 10) | (grid == 5)
    dynamic_mask = ~static_mask

    # Settlement probability = class 1 (settlement) + class 2 (port)
    settle_prob = gt[:, :, 1] + gt[:, :, 2]
    settle_prob_dynamic = settle_prob * dynamic_mask

    total_settle_prob = settle_prob_dynamic.sum()
    if total_settle_prob < 0.01:
        # Dead world — no settlements
        return {
            "expansion_radius": 0.0,
            "concentration_ratio": 1.0,
            "new_settlement_spread": 0.0,
            "settlement_gini": 1.0,
            "active_cell_clustering": 0.0,
            "total_settle_prob": float(total_settle_prob),
            "n_active_cells": 0,
            "mean_active_dist": 0.0,
            "median_active_dist": 0.0,
            "p90_active_dist": 0.0,
            "frac_at_initial": 0.0,
            "frac_within_3": 0.0,
            "frac_within_5": 0.0,
            "frac_within_8": 0.0,
            "frac_beyond_8": 0.0,
        }

    # 1. Expansion radius: probability-weighted mean distance from initial settlements
    weighted_dist = (settle_prob_dynamic * dist_map).sum() / total_settle_prob

    # 2. Concentration ratio: fraction of settle prob within dist<=5
    close_mask = (dist_map <= 5) & dynamic_mask
    concentration = settle_prob[close_mask].sum() / total_settle_prob

    # 3. New settlement spread (exclude initial settlement positions)
    initial_mask = np.zeros((MAP_H, MAP_W), dtype=bool)
    for s in initial_settlements:
        sy, sx = s["y"], s["x"]
        if 0 <= sy < MAP_H and 0 <= sx < MAP_W:
            initial_mask[sy, sx] = True

    new_settle = settle_prob_dynamic * (~initial_mask)
    total_new = new_settle.sum()
    if total_new > 0.01:
        new_spread = (new_settle * dist_map).sum() / total_new
    else:
        new_spread = 0.0

    # 4. Gini coefficient of settlement probability
    probs = settle_prob_dynamic[dynamic_mask].flatten()
    probs_sorted = np.sort(probs)
    n = len(probs_sorted)
    if n > 0 and probs_sorted.sum() > 0:
        cumulative = np.cumsum(probs_sorted)
        gini = 1.0 - 2.0 * cumulative.sum() / (n * probs_sorted.sum())
    else:
        gini = 1.0

    # 5. Active cell clustering
    active_threshold = 0.10
    active_cells = set()
    for y in range(MAP_H):
        for x in range(MAP_W):
            if not static_mask[y, x] and settle_prob[y, x] > active_threshold:
                active_cells.add((y, x))

    n_active = len(active_cells)
    if n_active > 1:
        clustered = 0
        for (y, x) in active_cells:
            has_neighbor = False
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (y + dy, x + dx) in active_cells:
                    has_neighbor = True
                    break
            if has_neighbor:
                clustered += 1
        clustering = clustered / n_active
    else:
        clustering = 0.0

    # 6. Distance distribution stats for active cells
    active_dists = [dist_map[y, x] for (y, x) in active_cells]
    if active_dists:
        mean_active_dist = np.mean(active_dists)
        median_active_dist = np.median(active_dists)
        p90_active_dist = np.percentile(active_dists, 90)
    else:
        mean_active_dist = median_active_dist = p90_active_dist = 0.0

    # 7. Fraction at different distance bands
    frac_at_initial = settle_prob[initial_mask].sum() / total_settle_prob if initial_mask.any() else 0.0
    frac_within_3 = settle_prob[(dist_map <= 3) & dynamic_mask].sum() / total_settle_prob
    frac_within_5 = settle_prob[(dist_map <= 5) & dynamic_mask].sum() / total_settle_prob
    frac_within_8 = settle_prob[(dist_map <= 8) & dynamic_mask].sum() / total_settle_prob
    frac_beyond_8 = 1.0 - frac_within_8

    return {
        "expansion_radius": float(weighted_dist),
        "concentration_ratio": float(concentration),
        "new_settlement_spread": float(new_spread),
        "settlement_gini": float(gini),
        "active_cell_clustering": float(clustering),
        "total_settle_prob": float(total_settle_prob),
        "n_active_cells": n_active,
        "mean_active_dist": float(mean_active_dist),
        "median_active_dist": float(median_active_dist),
        "p90_active_dist": float(p90_active_dist),
        "frac_at_initial": float(frac_at_initial),
        "frac_within_3": float(frac_within_3),
        "frac_within_5": float(frac_within_5),
        "frac_within_8": float(frac_within_8),
        "frac_beyond_8": float(frac_beyond_8),
    }


def compute_initial_state_features(initial_grid, initial_settlements):
    """
    Compute features from the initial state that might predict localization.

    Returns dict with:
    - n_settlements: number of initial settlements
    - mean_inter_settlement_dist: mean pairwise distance between settlements
    - settlement_spread: std of settlement positions
    - land_fraction: fraction of cells that are land
    - coastal_settlements: fraction of settlements that are coastal
    - forest_near_settlements: mean number of forest cells within dist<=3 of settlements
    - settlement_cluster_count: number of settlement clusters (connected components within dist<=3)
    """
    grid = np.array(initial_grid, dtype=int)
    n = len(initial_settlements)

    # Basic
    land_mask = ~np.isin(grid, [5, 10])
    land_fraction = land_mask.sum() / (MAP_H * MAP_W)

    if n == 0:
        return {
            "n_settlements": 0,
            "mean_inter_settlement_dist": 0.0,
            "settlement_spread": 0.0,
            "land_fraction": float(land_fraction),
            "coastal_settlements_frac": 0.0,
            "forest_near_settlements": 0.0,
            "settlement_cluster_count": 0,
        }

    positions = [(s["y"], s["x"]) for s in initial_settlements]

    # Mean inter-settlement Manhattan distance
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(positions[i][0] - positions[j][0]) + abs(positions[i][1] - positions[j][1])
            dists.append(d)
    mean_inter = np.mean(dists) if dists else 0.0

    # Spread (std of positions)
    ys = [p[0] for p in positions]
    xs = [p[1] for p in positions]
    spread = np.sqrt(np.std(ys)**2 + np.std(xs)**2)

    # Coastal settlements
    coastal_count = 0
    for s in initial_settlements:
        sy, sx = s["y"], s["x"]
        is_coastal = False
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = sy + dy, sx + dx
                if 0 <= ny < MAP_H and 0 <= nx < MAP_W and grid[ny, nx] == 10:
                    is_coastal = True
                    break
            if is_coastal:
                break
        if is_coastal:
            coastal_count += 1

    # Forest near settlements
    forest_counts = []
    for s in initial_settlements:
        sy, sx = s["y"], s["x"]
        fc = 0
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                ny, nx = sy + dy, sx + dx
                if 0 <= ny < MAP_H and 0 <= nx < MAP_W and grid[ny, nx] == 4:
                    fc += 1
        forest_counts.append(fc)

    # Settlement clusters (union-find with dist<=3)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            d = abs(positions[i][0] - positions[j][0]) + abs(positions[i][1] - positions[j][1])
            if d <= 5:
                union(i, j)

    cluster_count = len(set(find(i) for i in range(n)))

    return {
        "n_settlements": n,
        "mean_inter_settlement_dist": float(mean_inter),
        "settlement_spread": float(spread),
        "land_fraction": float(land_fraction),
        "coastal_settlements_frac": float(coastal_count / n),
        "forest_near_settlements": float(np.mean(forest_counts)),
        "settlement_cluster_count": cluster_count,
    }


def main():
    if not API_KEY:
        print("FEIL: Sett API_KEY")
        sys.exit(1)

    client = AstarClient()

    # Fetch all completed rounds
    rounds = client.get("/rounds")
    completed = sorted(
        [r for r in rounds if isinstance(r, dict) and r.get("status") == "completed"],
        key=lambda r: r.get("round_number", 0)
    )
    print(f"Fant {len(completed)} fullførte runder\n")

    results = []

    for round_info in completed:
        round_id = round_info["id"]
        round_num = round_info.get("round_number", "?")
        round_weight = round_info.get("round_weight", 1.0)

        try:
            round_data = client.get(f"/rounds/{round_id}")
        except Exception as e:
            print(f"Runde {round_num}: feil henting — {e}")
            continue

        seeds = round_data.get("seeds", round_data.get("initial_states", []))
        if not seeds:
            continue

        round_result = {
            "round_number": round_num,
            "round_id": round_id,
            "round_weight": round_weight,
            "n_seeds": len(seeds),
            "seeds": [],
        }

        # Compute initial state features (same for all seeds of same map)
        initial_features = compute_initial_state_features(
            seeds[0].get("grid", []), seeds[0].get("settlements", []))
        round_result["initial_features"] = initial_features

        print(f"Runde {round_num} ({initial_features['n_settlements']} settlements, "
              f"{initial_features['settlement_cluster_count']} clusters):")

        seed_metrics_list = []
        for seed_idx in range(len(seeds)):
            try:
                analysis = client.get(f"/analysis/{round_id}/{seed_idx}")
                gt = analysis.get("ground_truth")
                if gt is None:
                    print(f"  Seed {seed_idx}: ingen GT")
                    continue

                initial_grid = seeds[seed_idx].get("grid", [])
                initial_settlements = seeds[seed_idx].get("settlements", [])

                metrics = compute_localization_metrics(initial_grid, initial_settlements, gt)
                seed_metrics_list.append(metrics)

                round_result["seeds"].append({
                    "seed_index": seed_idx,
                    **metrics
                })

                print(f"  Seed {seed_idx}: expansion={metrics['expansion_radius']:.1f}, "
                      f"conc={metrics['concentration_ratio']:.3f}, "
                      f"gini={metrics['settlement_gini']:.3f}, "
                      f"cluster={metrics['active_cell_clustering']:.3f}, "
                      f"n_active={metrics['n_active_cells']}, "
                      f"frac_beyond_8={metrics['frac_beyond_8']:.3f}")

                time.sleep(0.25)
            except Exception as e:
                print(f"  Seed {seed_idx}: feil — {e}")
                time.sleep(0.5)

        # Compute round averages
        if seed_metrics_list:
            avg_metrics = {}
            for key in seed_metrics_list[0]:
                vals = [m[key] for m in seed_metrics_list]
                avg_metrics[key] = float(np.mean(vals))
            round_result["round_average"] = avg_metrics

            print(f"  AVG: expansion={avg_metrics['expansion_radius']:.1f}, "
                  f"conc={avg_metrics['concentration_ratio']:.3f}, "
                  f"gini={avg_metrics['settlement_gini']:.3f}, "
                  f"cluster={avg_metrics['active_cell_clustering']:.3f}, "
                  f"frac_beyond_8={avg_metrics['frac_beyond_8']:.3f}")

        results.append(round_result)
        print()

    # Print summary table
    print("\n" + "=" * 120)
    print("ROUND LOCALIZATION SUMMARY")
    print("=" * 120)
    print(f"{'Rnd':>4} {'N_set':>5} {'Clust':>5} {'Expan':>6} {'Conc':>6} {'Gini':>6} "
          f"{'ACClust':>7} {'N_Act':>6} {'F<=3':>6} {'F<=5':>6} {'F<=8':>6} {'F>8':>6} {'Type':>12}")
    print("-" * 120)

    for r in results:
        if "round_average" not in r:
            continue
        a = r["round_average"]
        f = r["initial_features"]

        # Classify localization type
        if a["total_settle_prob"] < 1.0:
            loc_type = "DEAD"
        elif a["concentration_ratio"] > 0.80 and a["frac_beyond_8"] < 0.05:
            loc_type = "URBAN"
        elif a["concentration_ratio"] < 0.50 or a["frac_beyond_8"] > 0.20:
            loc_type = "RURAL/SPREAD"
        else:
            loc_type = "MIXED"

        print(f"{r['round_number']:>4} {f['n_settlements']:>5} {f['settlement_cluster_count']:>5} "
              f"{a['expansion_radius']:>6.1f} {a['concentration_ratio']:>6.3f} {a['settlement_gini']:>6.3f} "
              f"{a['active_cell_clustering']:>7.3f} {a['n_active_cells']:>6.0f} "
              f"{a['frac_within_3']:>6.3f} {a['frac_within_5']:>6.3f} {a['frac_within_8']:>6.3f} "
              f"{a['frac_beyond_8']:>6.3f} {loc_type:>12}")

    # Save results
    output_path = Path(__file__).parent / "localization_analysis.json"
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResultater lagret til {output_path}")


if __name__ == "__main__":
    main()
