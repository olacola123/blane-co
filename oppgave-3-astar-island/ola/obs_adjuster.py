"""
Observation-Based Prior Adjustment for Astar Island
====================================================
Uses observations to adjust super-prior predictions per round.

TESTED APPROACHES (14 rounds x 5 seeds = 70 seeds, oracle vitality):

| Approach                          |  2 vp  |  5 vp  | 10 vp  |
|-----------------------------------|--------|--------|--------|
| Group ratio (obs/prior)           | -44.6  |   --   |   --   |
| Group delta (obs - prior)         |  -1.8  |  -0.4  |  -0.4  |
| Cross-seed evidence               |  -0.2  |   --   |  -0.2  |
| Direct obs (ps=50, ds=3)          | +0.09  | +0.10  |  -0.07 |
| Direct obs (ps=50, ds=1)          | +0.05  | +0.11  | +0.18  |
| Direct obs (ps=100, ds=3)         | +0.07  | +0.13  | +0.18  |

Key findings:
- Group-level adjustments ALWAYS hurt (prior already well-calibrated per group)
- Direct cell-level Bayesian update is the only thing that helps
- More observations need MORE conservative blending (higher ps or lower ds)
- Adaptive: ps = 50 + 5*n_viewports, ds = 3/sqrt(n_viewports) works well

Usage:
    from obs_adjuster import ObsAdjuster
    adjuster = ObsAdjuster(grid, settlements, vbin)
    adjuster.add_observations(counts, observed)  # from queries
    adjusted_pred = adjuster.get_adjusted_prediction()
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).parent))
from super_prior import super_predict, vitality_to_vbin

MAP_W, MAP_H = 40, 40
NUM_CLASSES = 6


class ObsAdjuster:
    """
    Adjusts super-prior predictions using direct observations.

    For observed cells: Bayesian posterior with configurable prior strength.
    For unobserved cells: returns super-prior unchanged (group adjustments hurt).
    """

    def __init__(self, grid, settlements, vbin: str, floor: float = None):
        """
        Args:
            grid: 40x40 initial terrain grid
            settlements: list of {"x", "y", ...} dicts
            vbin: vitality bin ("DEAD", "LOW", "MED", "HIGH")
            floor: min probability per class
        """
        self.grid = np.array(grid, dtype=int) if not isinstance(grid, np.ndarray) else grid
        self.settlements = settlements
        self.vbin = vbin
        self.floor = floor or {"DEAD": 0.001, "LOW": 0.002, "MED": 0.003, "HIGH": 0.003}.get(vbin, 0.003)

        H, W = self.grid.shape
        self.H, self.W = H, W

        # Identify static cells
        self.is_static = np.zeros((H, W), dtype=bool)
        for y in range(H):
            for x in range(W):
                t = int(self.grid[y, x])
                if t == 10 or t == 5:  # ocean, mountain
                    self.is_static[y, x] = True

        # Super-prior prediction (baseline)
        self.prior = super_predict(grid, settlements, vbin, floor=self.floor)

        # Observation accumulators
        self.obs_counts = np.zeros((H, W, NUM_CLASSES), dtype=float)
        self.obs_n = np.zeros((H, W), dtype=int)

    def add_observations(self, counts: np.ndarray, observed: np.ndarray):
        """
        Add observation data from queries.

        Args:
            counts: (H, W, 6) array — how many times each class was observed per cell
            observed: (H, W) array — how many times each cell was observed
        """
        self.obs_counts += counts
        self.obs_n += observed.astype(int)

    def add_viewport(self, viewport_data: list, top_y: int, left_x: int):
        """
        Add a single viewport observation (15x15 grid of class indices).

        Args:
            viewport_data: 2D list of class indices (0-5)
            top_y, left_x: top-left corner of viewport on the 40x40 map
        """
        for vy, row in enumerate(viewport_data):
            for vx, cls in enumerate(row):
                y = top_y + vy
                x = left_x + vx
                if 0 <= y < self.H and 0 <= x < self.W:
                    self.obs_counts[y, x, int(cls)] += 1
                    self.obs_n[y, x] += 1

    def get_adjusted_prediction(self, prior_strength: float = None,
                                 direct_obs_weight: float = None) -> np.ndarray:
        """
        Build adjusted prediction tensor.

        For each observed cell:
            posterior ∝ prior_strength * prior_dist + direct_obs_weight * obs_counts

        For unobserved cells: returns prior unchanged.

        If prior_strength/direct_obs_weight are None, uses adaptive values
        based on observation density (more observations → more conservative).

        Returns:
            (H, W, 6) numpy array with adjusted probabilities
        """
        # Adaptive parameter selection based on observation density
        if prior_strength is None or direct_obs_weight is None:
            n_observed = int((self.obs_n > 0).sum())
            n_dynamic = int((~self.is_static).sum())
            coverage = n_observed / max(n_dynamic, 1)
            # Estimate effective viewports from coverage
            # Each 15x15 viewport covers ~225 cells, map has ~1000 dynamic cells
            est_viewports = max(1, coverage * n_dynamic / 200)

            if prior_strength is None:
                # More viewports → stronger prior (more conservative)
                prior_strength = 50.0 + 5.0 * est_viewports
            if direct_obs_weight is None:
                # More viewports → lower obs weight per observation
                direct_obs_weight = 3.0 / max(1.0, np.sqrt(est_viewports))

        pred = self.prior.copy()

        for y in range(self.H):
            for x in range(self.W):
                n = self.obs_n[y, x]
                if n == 0 or self.is_static[y, x]:
                    continue

                # Bayesian posterior: prior pseudocounts + observation counts
                p = self.prior[y, x] * prior_strength + self.obs_counts[y, x] * direct_obs_weight
                total = p.sum()
                if total > 0:
                    p /= total
                else:
                    p = self.prior[y, x]

                # Apply floor
                p = np.maximum(p, self.floor)
                p /= p.sum()
                pred[y, x] = p

        return pred


# === SCORING ===

def weighted_kl(ground_truth, prediction):
    """Entropy-weighted KL divergence (competition metric)."""
    gt = np.array(ground_truth, dtype=float)
    pred = np.array(prediction, dtype=float)
    gt_safe = np.clip(gt, 1e-12, 1.0)
    pred_safe = np.clip(pred, 1e-12, 1.0)
    cell_kl = np.sum(gt_safe * (np.log(gt_safe) - np.log(pred_safe)), axis=-1)
    cell_entropy = -np.sum(gt_safe * np.log(gt_safe), axis=-1)
    total_weight = cell_entropy.sum()
    if total_weight <= 0:
        return float(cell_kl.mean())
    return float((cell_kl * cell_entropy).sum() / total_weight)


def score_from_kl(wkl):
    return max(0, min(100, 100 * np.exp(-3 * wkl)))


def simulate_viewports(gt, n_viewports=2, rng=None):
    """Simulate viewport observations from one-hot ground truth."""
    if rng is None:
        rng = np.random.default_rng(42)
    H, W = gt.shape[:2]
    counts = np.zeros((H, W, NUM_CLASSES), dtype=float)
    observed = np.zeros((H, W), dtype=int)
    for _ in range(n_viewports):
        top = rng.integers(0, H - 15 + 1)
        left = rng.integers(0, W - 15 + 1)
        for y in range(top, top + 15):
            for x in range(left, left + 15):
                cls = np.argmax(gt[y, x])
                counts[y, x, cls] += 1
                observed[y, x] += 1
    return counts, observed


# === BACKTEST ===

def backtest():
    """Full backtest across all completed rounds with parameter grid search."""
    BASE_URL = "https://api.ainm.no/astar-island"
    API_KEY = os.environ.get("API_KEY",
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZDY4OWRmZC01NGM0LTQwZmYtYTM2My01MzMyYjc0ZDY4M2EiLCJlbWFpbCI6Im9sYWd1ZGJyYW5kQGdtYWlsLmNvbSIsImlzX2FkbWluIjpmYWxzZSwiZXhwIjoxNzc0NTYyNTMzfQ.zEUXW0mk5hfMuTTtXu5EwF9m1Ex6vh6tOUYRMnNvs7c")

    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"})

    def api_get(path):
        r = session.get(f"{BASE_URL}/{path.lstrip('/')}")
        r.raise_for_status()
        return r.json()

    print("=" * 70)
    print("BACKTEST: super_prior vs obs_adjuster")
    print("Oracle vitality, 14 rounds x 5 seeds = 70 seeds")
    print("=" * 70)

    # Load all data
    print("\nLoading round data from API...")
    rounds = api_get("/my-rounds")
    completed = [r for r in rounds if isinstance(r, dict) and r.get("status") == "completed"]
    completed.sort(key=lambda r: r.get("round_number", 0))

    round_cache = []
    for rd in completed:
        round_id = rd["id"]
        rnum = rd.get("round_number", "?")
        round_data = api_get(f"/rounds/{round_id}")
        seeds_data = round_data.get("seeds", round_data.get("initial_states", []))
        seed_entries = []
        for seed_idx in range(len(seeds_data)):
            seed_data = seeds_data[seed_idx]
            grid = seed_data["grid"]
            settlements = seed_data["settlements"]
            try:
                analysis = api_get(f"/analysis/{round_id}/{seed_idx}")
                gt = analysis.get("ground_truth")
                if gt is None: continue
                gt_arr = np.array(gt, dtype=float)
            except Exception: continue
            total_init, total_survived = 0, 0.0
            for s in settlements:
                sx, sy = s["x"], s["y"]
                if 0 <= sx < MAP_W and 0 <= sy < MAP_H:
                    total_init += 1
                    total_survived += gt_arr[sy, sx, 1] + gt_arr[sy, sx, 2]
            vrate = total_survived / total_init if total_init > 0 else 0.35
            vbin = vitality_to_vbin(vrate)
            prior_pred = super_predict(grid, settlements, vbin)
            prior_score = score_from_kl(weighted_kl(gt_arr, prior_pred))
            seed_entries.append({
                "grid": grid, "settlements": settlements, "gt": gt_arr,
                "vbin": vbin, "prior_score": prior_score, "seed_idx": seed_idx,
            })
            time.sleep(0.1)
        round_cache.append({"rnum": rnum, "seeds": seed_entries,
                           "vbin": seed_entries[0]["vbin"] if seed_entries else "?"})
    total_seeds = sum(len(r["seeds"]) for r in round_cache)
    print(f"Loaded {total_seeds} seeds from {len(round_cache)} rounds\n")

    # Grid search over parameters x viewport counts
    print("=" * 70)
    print("PARAMETER GRID SEARCH")
    print("=" * 70)

    param_configs = [
        (None, None, "adaptive"),
        (50, 1.0, "ps50_ds1"),
        (50, 3.0, "ps50_ds3"),
        (100, 3.0, "ps100_ds3"),
        (100, 5.0, "ps100_ds5"),
    ]

    for n_vp in [2, 5, 10]:
        print(f"\n  {n_vp} viewports per seed:")
        best_imp, best_label = -999, ""

        for ps, ds, label in param_configs:
            all_imps = []
            for rc in round_cache:
                for se in rc["seeds"]:
                    rng = np.random.default_rng(se["seed_idx"] * 1000 + (rc["rnum"] if isinstance(rc["rnum"], int) else 0))
                    counts, observed = simulate_viewports(se["gt"], n_viewports=n_vp, rng=rng)
                    adjuster = ObsAdjuster(se["grid"], se["settlements"], se["vbin"])
                    adjuster.add_observations(counts, observed)
                    adj_pred = adjuster.get_adjusted_prediction(prior_strength=ps, direct_obs_weight=ds)
                    adj_score = score_from_kl(weighted_kl(se["gt"], adj_pred))
                    all_imps.append(adj_score - se["prior_score"])

            mean_imp = np.mean(all_imps)
            n_better = sum(1 for x in all_imps if x > 0)
            marker = " <-- BEST" if mean_imp > best_imp else ""
            if mean_imp > best_imp:
                best_imp = mean_imp
                best_label = label
            print(f"    {label:12s}: imp={mean_imp:+.3f}  better={n_better}/{len(all_imps)}{marker}")

        print(f"    Best: {best_label} ({best_imp:+.3f})")

    # Detailed per-round for adaptive config
    for n_vp in [2, 10]:
        print(f"\n{'='*70}")
        print(f"DETAILED: adaptive params, {n_vp} viewports")
        print(f"{'='*70}")
        total_prior, total_adj = [], []
        for rc in round_cache:
            rp, ra = [], []
            for se in rc["seeds"]:
                rng = np.random.default_rng(se["seed_idx"] * 1000 + (rc["rnum"] if isinstance(rc["rnum"], int) else 0))
                counts, observed = simulate_viewports(se["gt"], n_viewports=n_vp, rng=rng)
                adj = ObsAdjuster(se["grid"], se["settlements"], se["vbin"])
                adj.add_observations(counts, observed)
                pred = adj.get_adjusted_prediction()  # adaptive params
                score = score_from_kl(weighted_kl(se["gt"], pred))
                rp.append(se["prior_score"]); ra.append(score)
            avg_p, avg_a = np.mean(rp), np.mean(ra)
            total_prior.extend(rp); total_adj.extend(ra)
            imp = avg_a - avg_p
            sym = "+" if imp > 0 else ""
            print(f"  R{rc['rnum']:>2} [{rc['vbin']:>4}]: prior={avg_p:5.1f}  adj={avg_a:5.1f}  {sym}{imp:.2f}")

        avg_p, avg_a = np.mean(total_prior), np.mean(total_adj)
        n_better = sum(1 for p,a in zip(total_prior,total_adj) if a>p)
        print(f"\n  TOTAL: prior={avg_p:.2f}  adj={avg_a:.2f}  imp={avg_a-avg_p:+.3f}")
        print(f"  Seeds improved: {n_better}/{len(total_prior)} ({100*n_better/len(total_prior):.0f}%)")
        print(f"  Best seed imp:  {max(a-p for p,a in zip(total_prior,total_adj)):+.2f}")
        print(f"  Worst seed imp: {min(a-p for p,a in zip(total_prior,total_adj)):+.2f}")
    print("=" * 70)


if __name__ == "__main__":
    backtest()
