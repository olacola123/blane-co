"""
Joakim's Astar Island Solver v2 — Pooled Round-Specific Empirical Model
========================================================================
Key innovation: Pool ALL 50 observations across all seeds by feature key
into round-specific empirical distributions. Use these as the PRIMARY
prediction source (up to 75% weight) with calibration tables as fallback.

All 5 seeds share the same hidden simulation parameters, so cells with the
same features (terrain_group, distance_band, coastal) have the same underlying
distribution regardless of seed. Pooling gives 5x more data per feature key.

Bruk:
    export API_KEY='din-jwt-token'
    python solution_v2.py                    # Løs aktiv runde
    python solution_v2.py --dry-run          # Vis plan uten API-kall
    python solution_v2.py --no-submit        # Observer uten submit
    python solution_v2.py --leaderboard      # Vis leaderboard
    python solution_v2.py --backtest         # Backtest mot historiske runder
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# === CONFIG ===
BASE_URL = "https://api.ainm.no/astar-island"
API_KEY = os.environ.get("API_KEY", "")
MAP_W, MAP_H = 40, 40
NUM_CLASSES = 6
MAX_VP = 15
FLOOR = 0.001

TERRAIN_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}
TERRAIN_GROUP = {
    0: "plains", 1: "settlement", 2: "port", 3: "ruin",
    4: "forest", 5: "mountain", 10: "ocean", 11: "plains",
}
OPT_DIST_BANDS = [(0, 0), (1, 2), (3, 3), (4, 5), (6, 8), (9, 12), (13, 99)]

FALLBACK_PRIOR = {
    "0":  [0.725, 0.082, 0.031, 0.062, 0.100, 0.0001],
    "10": [0.998, 0.0005, 0.0005, 0.0005, 0.0005, 0.0001],
    "11": [0.822, 0.121, 0.009, 0.012, 0.035, 0.0001],
    "1":  [0.462, 0.293, 0.004, 0.026, 0.214, 0.0001],
    "2":  [0.484, 0.089, 0.173, 0.022, 0.231, 0.0001],
    "3":  [0.224, 0.158, 0.084, 0.337, 0.197, 0.0001],
    "4":  [0.096, 0.152, 0.009, 0.018, 0.724, 0.0001],
    "5":  [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.9996],
}

PROJECT_ROOT = Path(__file__).resolve().parent
OLA_DIR = Path(__file__).parent.parent / "ola"
HISTORY_ROOT = PROJECT_ROOT / "history"

CALIBRATION_OPT_FILE = PROJECT_ROOT / "calibration_manhattan_opt.json"
MODEL_TABLES_FILE = PROJECT_ROOT / "model_tables_17r.json"
MODEL_TABLES_FILE_OLA = OLA_DIR / "model_tables.json"


# === HELPERS ===

def get_opt_band(dist):
    for i, (lo, hi) in enumerate(OPT_DIST_BANDS):
        if lo <= dist <= hi:
            return i
    return len(OPT_DIST_BANDS) - 1


def tg(terrain_code):
    return TERRAIN_GROUP.get(terrain_code, "plains")


def manhattan_nearest(y, x, settlements):
    if not settlements:
        return 99
    return min(abs(x - s["x"]) + abs(y - s["y"]) for s in settlements)


def is_coastal(grid, y, x):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ny, nx = y + dy, x + dx
            if 0 <= ny < MAP_H and 0 <= nx < MAP_W:
                if grid[ny][nx] == 10:
                    return True
    return False


# === CALIBRATION LOADING ===

_opt_cache = None

def load_opt_tables():
    global _opt_cache
    if _opt_cache is not None:
        return _opt_cache
    for f, label in [
        (CALIBRATION_OPT_FILE, "Joakim Manhattan"),
        (OLA_DIR / "calibration_optimized.json", "Ola Chebyshev"),
    ]:
        if f.exists():
            data = json.loads(f.read_text())
            _opt_cache = data.get("tables", {})
            print(f"Loaded opt_tables ({label}): {len(_opt_cache)} entries")
            return _opt_cache
    _opt_cache = {}
    return _opt_cache


_model_cache = None

def load_model_tables():
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    for f, label in [(MODEL_TABLES_FILE, "Joakim 17r"), (MODEL_TABLES_FILE_OLA, "Ola 14r")]:
        if f.exists():
            data = json.loads(f.read_text())
            _model_cache = {
                "specific": data.get("table_specific", {}),
                "medium": data.get("table_medium", {}),
                "simple": data.get("table_simple", {}),
            }
            print(f"Loaded model_tables ({label}): "
                  f"{len(_model_cache['specific'])}s + {len(_model_cache['medium'])}m + {len(_model_cache['simple'])}b")
            return _model_cache
    _model_cache = {}
    return _model_cache


def _dist_bin(d):
    if d <= 0: return 0
    elif d <= 2: return 1
    elif d <= 3: return 2
    elif d <= 5: return 3
    elif d <= 8: return 4
    elif d <= 12: return 5
    else: return 6


def _settle_density_bin(n):
    return 0 if n == 0 else (1 if n <= 2 else 2)


def _forest_density_bin(n):
    if n == 0: return 0
    elif n <= 4: return 1
    elif n <= 10: return 2
    else: return 3


def get_cal_dist(terrain_code, dist, coastal, settlements, grid_arr,
                 y, x, vbin, opt_tables, model_tables):
    """Get calibration distribution with cascading fallback.

    Priority: model_tables.specific → model_tables.medium → opt_tables → model_tables.simple → fallback
    """
    terrain_g = tg(terrain_code)
    band = get_opt_band(dist)
    c = int(coastal)

    # model_tables: specific (uses settle_density, forest_density)
    if model_tables:
        n_settle_r5 = sum(1 for s in settlements
                          if abs(y - s["y"]) + abs(x - s["x"]) <= 5)
        n_forest_r2 = 0
        for dy2 in range(-2, 3):
            for dx2 in range(-2, 3):
                ny2, nx2 = y + dy2, x + dx2
                if 0 <= ny2 < MAP_H and 0 <= nx2 < MAP_W:
                    if int(grid_arr[ny2, nx2]) == 4:
                        n_forest_r2 += 1
        db = _dist_bin(dist)
        sdb = _settle_density_bin(n_settle_r5)
        fdb = _forest_density_bin(n_forest_r2)

        spec_key = f"{vbin}_{terrain_g}_{db}_{c}_{sdb}_{fdb}"
        if spec_key in model_tables.get("specific", {}):
            return np.array(model_tables["specific"][spec_key]["distribution"], dtype=float)

        med_key = f"{vbin}_{terrain_g}_{db}_{c}"
        if med_key in model_tables.get("medium", {}):
            return np.array(model_tables["medium"][med_key]["distribution"], dtype=float)

    # opt_tables: world_type specific
    if opt_tables:
        # Map vbin → world_type for opt_tables
        wtype_map = {"DEAD": "DEAD", "LOW": "STABLE", "MED": "STABLE", "HIGH": "BOOM"}
        wtype = wtype_map.get(vbin, "STABLE")
        for wt in [wtype, "ALL"]:
            for key in [f"{wt}_{terrain_g}_{band}_{c}", f"{wt}_{terrain_g}_{band}_any"]:
                if key in opt_tables:
                    return np.array(opt_tables[key]["distribution"], dtype=float)

    # model_tables simple fallback
    if model_tables:
        simp_key = f"{terrain_g}_{_dist_bin(dist)}"
        if simp_key in model_tables.get("simple", {}):
            return np.array(model_tables["simple"][simp_key]["distribution"], dtype=float)

    # Final fallback
    fb = FALLBACK_PRIOR.get(str(terrain_code))
    if fb:
        return np.array(fb, dtype=float)
    return np.ones(NUM_CLASSES) / NUM_CLASSES


def get_blended_cal(terrain_code, dist, coastal, settlements, grid_arr,
                    y, x, type_weights, opt_tables, model_tables):
    """Get calibration distribution blended across world types."""
    vbin_map = {"DEAD": "DEAD", "STABLE": "MED", "BOOM": "HIGH"}
    blended = np.zeros(NUM_CLASSES, dtype=float)
    for wtype, w in type_weights.items():
        if w < 0.01:
            continue
        vbin = vbin_map.get(wtype, "MED")
        d = get_cal_dist(terrain_code, dist, coastal, settlements, grid_arr,
                         y, x, vbin, opt_tables, model_tables)
        blended += w * d
    s = blended.sum()
    if s > 0:
        blended /= s
    else:
        blended = np.ones(NUM_CLASSES) / NUM_CLASSES
    return blended


# === API CLIENT ===

class AstarClient:
    def __init__(self):
        if not API_KEY:
            print("FEIL: Sett API_KEY=din-jwt-token")
            sys.exit(1)
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        })

    def get(self, path, params=None):
        r = self.session.get(f"{BASE_URL}/{path.lstrip('/')}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def post(self, path, data):
        r = self.session.post(f"{BASE_URL}/{path.lstrip('/')}", json=data, timeout=60)
        r.raise_for_status()
        return r.json()

    def get_rounds(self): return self.get("/rounds")
    def get_round(self, rid): return self.get(f"/rounds/{rid}")
    def get_budget(self): return self.get("/budget")
    def get_my_rounds(self): return self.get("/my-rounds")
    def get_leaderboard(self): return self.get("/leaderboard")
    def get_analysis(self, rid, si): return self.get(f"/analysis/{rid}/{si}")

    def simulate(self, round_id, seed_index, x, y, w=MAX_VP, h=MAX_VP):
        x = max(0, min(x, MAP_W - w))
        y = max(0, min(y, MAP_H - h))
        return self.post("/simulate", {
            "round_id": round_id, "seed_index": seed_index,
            "viewport_x": x, "viewport_y": y, "viewport_w": w, "viewport_h": h,
        })

    def submit(self, round_id, seed_index, prediction):
        return self.post("/submit", {
            "round_id": round_id, "seed_index": seed_index, "prediction": prediction,
        })


# === ROUND EMPIRICAL MODEL ===

class RoundModel:
    """
    Pools observations from all seeds by feature key into round-specific
    empirical distributions.

    Key insight: All seeds share the same hidden params, so cells with the
    same (terrain_group, dist_band, coastal) have the same distribution
    regardless of seed. Pooling gives ~5x data per feature key.
    """

    def __init__(self, seeds_data):
        self.seeds_data = seeds_data
        self.n_seeds = len(seeds_data)

        # Per-seed precomputed data
        self.grids = []
        self.settlements = []
        self.static_masks = []
        self.coastal_masks = []
        self.dist_maps = []

        for sd in seeds_data:
            grid = np.array(sd.get("grid", []), dtype=int)
            sett = sd.get("settlements", [])
            self.grids.append(grid)
            self.settlements.append(sett)

            static = (grid == 10) | (grid == 5)
            self.static_masks.append(static)

            coast = np.zeros((MAP_H, MAP_W), dtype=bool)
            for y2 in range(MAP_H):
                for x2 in range(MAP_W):
                    if static[y2, x2]:
                        continue
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y2 + dy, x2 + dx
                            if 0 <= ny < MAP_H and 0 <= nx < MAP_W and grid[ny, nx] == 10:
                                coast[y2, x2] = True
                                break
                        if coast[y2, x2]:
                            break
            self.coastal_masks.append(coast)

            # Precompute distance map
            dm = np.full((MAP_H, MAP_W), 99, dtype=int)
            for s in sett:
                sy, sx = s["y"], s["x"]
                for y2 in range(MAP_H):
                    for x2 in range(MAP_W):
                        d = abs(y2 - sy) + abs(x2 - sx)
                        if d < dm[y2, x2]:
                            dm[y2, x2] = d
            self.dist_maps.append(dm)

        # Pooled empirical data by feature key (across all seeds)
        self.key_counts = defaultdict(lambda: np.zeros(NUM_CLASSES, dtype=float))
        self.key_total = defaultdict(int)

        # Per-seed per-cell observation data
        self.cell_counts = defaultdict(lambda: np.zeros(NUM_CLASSES, dtype=float))
        self.cell_total = defaultdict(int)

        # Fingerprint tracking
        self._sett_obs = 0
        self._sett_survived = 0.0

    def _feature_key(self, y, x, si):
        terrain = int(self.grids[si][y, x])
        terrain_g = tg(terrain)
        dist = int(self.dist_maps[si][y, x])
        band = get_opt_band(dist)
        coastal = int(self.coastal_masks[si][y, x])
        return (terrain_g, band, coastal), dist

    def add_observation(self, seed_index, grid_data, vx, vy):
        """Add one viewport observation to the pooled model."""
        si = seed_index
        for dy, row in enumerate(grid_data):
            for dx, val in enumerate(row):
                y, x = vy + dy, vx + dx
                if not (0 <= y < MAP_H and 0 <= x < MAP_W):
                    continue
                if self.static_masks[si][y, x]:
                    continue

                cls = TERRAIN_TO_CLASS.get(val, 0)

                # Per-cell (seed-specific)
                ck = (si, y, x)
                self.cell_counts[ck][cls] += 1
                self.cell_total[ck] += 1

                # Pooled by feature key (across all seeds)
                fkey, _ = self._feature_key(y, x, si)
                self.key_counts[fkey][cls] += 1
                self.key_total[fkey] += 1

                # Fingerprint: track initial settlement survival
                initial_terrain = int(self.grids[si][y, x])
                if initial_terrain in (1, 2):
                    self._sett_obs += 1
                    if cls in (1, 2):
                        self._sett_survived += 1

    def get_vitality(self):
        if self._sett_obs == 0:
            return 0.33
        return max(0.0, min(1.0, self._sett_survived / self._sett_obs))

    def compute_type_weights(self):
        s = self.get_vitality()
        w_dead = max(0.0, min(1.0, (0.20 - s) / 0.12)) if s < 0.20 else 0.0
        w_boom = max(0.0, min(1.0, (s - 0.25) / 0.05)) if s > 0.25 else 0.0
        w_stable = max(0.0, 1.0 - w_dead - w_boom)
        total = w_dead + w_stable + w_boom
        if total < 1e-10:
            return {"DEAD": 0.0, "STABLE": 1.0, "BOOM": 0.0}
        return {"DEAD": w_dead / total, "STABLE": w_stable / total, "BOOM": w_boom / total}

    def build_prediction(self, seed_index, opt_tables, model_tables, type_weights=None):
        """
        Build prediction for one seed.

        For each dynamic cell:
        1. Get calibration dist (blended across world types, uses model_tables cascade)
        2. Get round-specific empirical dist (pooled from ALL seeds by feature key)
        3. Blend: empirical_weight * empirical + (1-empirical_weight) * calibration
           empirical_weight scales with sample count: 5→5%, 50→50% (cap 50%)
        4. Bayesian update with per-cell observations: (cell_counts + alpha*base) / (n+alpha)
           alpha=50 (very conservative — pooled empirical dominates, cell obs are noise)
        5. Floor + normalize
        """
        if type_weights is None:
            type_weights = self.compute_type_weights()

        si = seed_index
        grid = self.grids[si]
        sett = self.settlements[si]
        static = self.static_masks[si]
        coastal = self.coastal_masks[si]
        dist_map = self.dist_maps[si]
        pred = np.zeros((MAP_H, MAP_W, NUM_CLASSES), dtype=float)

        for y in range(MAP_H):
            for x in range(MAP_W):
                if grid[y, x] == 10:
                    pred[y, x] = [0.998, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004]
                    continue
                if grid[y, x] == 5:
                    pred[y, x] = [0.0004, 0.0004, 0.0004, 0.0004, 0.0004, 0.998]
                    continue

                terrain_code = int(grid[y, x])
                dist = int(dist_map[y, x])
                is_coast = bool(coastal[y, x])

                # 1. Calibration prior (blended across world types)
                cal = get_blended_cal(
                    terrain_code, dist, is_coast, sett, grid,
                    y, x, type_weights, opt_tables, model_tables,
                )

                # 2. Round-specific empirical (pooled across all seeds)
                fkey = (tg(terrain_code), get_opt_band(dist), int(is_coast))
                emp_n = self.key_total.get(fkey, 0)

                if emp_n >= 5:
                    emp = self.key_counts[fkey] / emp_n
                    # Scale weight with sample count: 5→5%, 50→50%
                    emp_weight = min(0.50, emp_n / 100.0)
                    base = emp_weight * emp + (1.0 - emp_weight) * cal
                else:
                    base = cal

                # 3. Bayesian update with per-cell observations
                # High alpha → per-cell obs have minimal influence (each is
                # a single stochastic sample, so pooled empirical is better)
                ck = (si, y, x)
                cell_n = self.cell_total.get(ck, 0)
                if cell_n > 0:
                    alpha = 50.0  # 1 obs → 2% cell, 98% base
                    posterior = (self.cell_counts[ck] + alpha * base) / (cell_n + alpha)
                    pred[y, x] = posterior
                else:
                    pred[y, x] = base

                # 4. Floor + normalize
                pred[y, x] = np.maximum(pred[y, x], FLOOR)
                pred[y, x, 5] = FLOOR * 0.1  # Mountain impossible on dynamic
                if not is_coast:
                    pred[y, x, 2] = FLOOR * 0.1  # Port impossible inland
                pred[y, x] /= pred[y, x].sum()

        return pred

    def build_prior_prediction(self, seed_index, opt_tables, model_tables, type_weights=None):
        """Build prediction using ONLY calibration tables (no observations)."""
        if type_weights is None:
            type_weights = {"DEAD": 0.0, "STABLE": 0.50, "BOOM": 0.50}

        si = seed_index
        grid = self.grids[si]
        sett = self.settlements[si]
        coastal = self.coastal_masks[si]
        dist_map = self.dist_maps[si]
        pred = np.zeros((MAP_H, MAP_W, NUM_CLASSES), dtype=float)

        for y in range(MAP_H):
            for x in range(MAP_W):
                if grid[y, x] == 10:
                    pred[y, x] = [0.998, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004]
                    continue
                if grid[y, x] == 5:
                    pred[y, x] = [0.0004, 0.0004, 0.0004, 0.0004, 0.0004, 0.998]
                    continue

                terrain_code = int(grid[y, x])
                dist = int(dist_map[y, x])
                is_coast = bool(coastal[y, x])

                cal = get_blended_cal(
                    terrain_code, dist, is_coast, sett, grid,
                    y, x, type_weights, opt_tables, model_tables,
                )
                pred[y, x] = cal
                pred[y, x] = np.maximum(pred[y, x], FLOOR)
                pred[y, x, 5] = FLOOR * 0.1
                if not is_coast:
                    pred[y, x, 2] = FLOOR * 0.1
                pred[y, x] /= pred[y, x].sum()

        return pred


# === QUERY PLANNING ===

def plan_viewports(grid, settlements, n_queries=10):
    """Plan viewport positions maximizing dynamic cell coverage."""
    grid_arr = np.array(grid, dtype=int)
    heatmap = np.zeros((MAP_H, MAP_W), dtype=float)
    for s in settlements:
        sx, sy = s["x"], s["y"]
        for dy in range(-18, 19):
            for dx in range(-18, 19):
                y, x = sy + dy, sx + dx
                if 0 <= y < MAP_H and 0 <= x < MAP_W:
                    d = abs(dx) + abs(dy)
                    if d == 0: heatmap[y, x] += 5.0
                    elif d <= 3: heatmap[y, x] += 3.0
                    elif d <= 8: heatmap[y, x] += 1.5
                    elif d <= 12: heatmap[y, x] += 0.5
                    elif d <= 18: heatmap[y, x] += 0.1
    heatmap[grid_arr == 10] = 0.0
    heatmap[grid_arr == 5] = 0.0

    viewports = []
    obs_count = np.zeros((MAP_H, MAP_W), dtype=int)
    for q in range(n_queries):
        overlap_penalty = 0.20 if q < 2 else 0.50
        best_score, best_vp = -1, (0, 0)
        for vy in range(0, MAP_H - MAX_VP + 1, 2):
            for vx in range(0, MAP_W - MAX_VP + 1, 2):
                rh = heatmap[vy:vy + MAX_VP, vx:vx + MAX_VP]
                ro = obs_count[vy:vy + MAX_VP, vx:vx + MAX_VP]
                score = (rh / (1.0 + overlap_penalty * ro)).sum()
                if score > best_score:
                    best_score, best_vp = score, (vx, vy)
        vx, vy = best_vp
        viewports.append((vx, vy, MAX_VP, MAX_VP))
        obs_count[vy:vy + MAX_VP, vx:vx + MAX_VP] += 1
    return viewports


# === SOLVER ===

def solve_round(client, round_id, round_data, submit=True):
    """Solve a round with the pooled empirical model."""
    opt_tables = load_opt_tables()
    model_tables = load_model_tables()

    seeds_data = round_data.get("seeds", round_data.get("initial_states", []))
    if not seeds_data:
        print("FEIL: Ingen seeds.")
        return []

    n_seeds = len(seeds_data)
    model = RoundModel(seeds_data)
    print(f"\n{n_seeds} seeds, settlements per seed: "
          f"{[len(sd.get('settlements', [])) for sd in seeds_data]}")

    # === FASE 1: Safety submit (blended prior, default weights) ===
    default_weights = {"DEAD": 0.0, "STABLE": 0.50, "BOOM": 0.50}
    prior_scores = []
    if submit:
        print("\n  FASE 1: Safety submit (prior only)...")
        for si in range(n_seeds):
            pred = model.build_prior_prediction(si, opt_tables, model_tables, default_weights)
            try:
                resp = client.submit(round_id, si, pred.tolist())
                score = resp.get("score", resp.get("seed_score", "?"))
                prior_scores.append(score)
                print(f"    Seed {si} prior → {score}")
                time.sleep(0.4)
            except Exception as e:
                print(f"    Seed {si} prior FEIL: {e}")
                prior_scores.append(None)

    # === FASE 2: Probe + observe (all 50 queries) ===
    # Allocate: 4 probe queries (1 per seed on 4 seeds), rest distributed evenly
    n_probe_seeds = min(4, n_seeds)
    total_budget = 50
    probe_total = n_probe_seeds  # 1 query per probe seed

    # Pre-plan per-seed query counts
    per_seed_queries = []
    budget_left = total_budget
    for si in range(n_seeds):
        seeds_remaining = n_seeds - si
        q_for_this = budget_left // seeds_remaining
        per_seed_queries.append(q_for_this)
        budget_left -= q_for_this

    print(f"\n  FASE 2: Observe (queries per seed: {per_seed_queries})...")

    # Execute queries: first do 1 probe per seed on 4 seeds
    probe_done = 0
    for si in range(n_probe_seeds):
        sd = seeds_data[si]
        grid = sd.get("grid", [])
        sett = sd.get("settlements", [])
        vps = plan_viewports(grid, sett, n_queries=1)
        if vps:
            vx, vy, vw, vh = vps[0]
            try:
                result = client.simulate(round_id, si, vx, vy, vw, vh)
                gd = result.get("grid", [])
                if gd:
                    model.add_observation(si, gd, vx, vy)
                used = result.get("queries_used", "?")
                print(f"    Probe seed {si}: ({vx},{vy}) → {used}/50")
                probe_done += 1
                time.sleep(0.25)
            except Exception as e:
                print(f"    Probe seed {si} FEIL: {e}")

    # Infer world type from probes
    vitality = model.get_vitality()
    type_weights = model.compute_type_weights()
    tw_str = ", ".join(f"{k}:{v:.2f}" for k, v in type_weights.items() if v > 0.01)
    print(f"\n  After probes: vitality={vitality:.3f}, weights={{{tw_str}}}")
    print(f"  Empirical keys with data: {sum(1 for v in model.key_total.values() if v >= 5)}")

    # Remaining queries per seed
    queries_left_per_seed = []
    for si in range(n_seeds):
        probe_used = 1 if si < n_probe_seeds else 0
        queries_left_per_seed.append(per_seed_queries[si] - probe_used)

    # Execute remaining queries
    for si in range(n_seeds):
        n_remaining = queries_left_per_seed[si]
        if n_remaining <= 0:
            continue
        sd = seeds_data[si]
        grid = sd.get("grid", [])
        sett = sd.get("settlements", [])
        vps = plan_viewports(grid, sett, n_queries=n_remaining + (1 if si < n_probe_seeds else 0))
        # Skip the first viewport if already probed
        start_idx = 1 if si < n_probe_seeds else 0
        for i in range(start_idx, start_idx + n_remaining):
            if i >= len(vps):
                break
            vx, vy, vw, vh = vps[i]
            try:
                result = client.simulate(round_id, si, vx, vy, vw, vh)
                gd = result.get("grid", [])
                if gd:
                    model.add_observation(si, gd, vx, vy)
                used = result.get("queries_used", "?")
                print(f"    Seed {si} Q{i}: ({vx},{vy}) → {used}/50")
                time.sleep(0.25)
            except Exception as e:
                print(f"    Seed {si} Q{i} FEIL: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    if e.response.status_code == 429:
                        print(f"    Budget exhausted, stopping queries.")
                        break

    # Recompute type_weights with all data
    vitality = model.get_vitality()
    type_weights = model.compute_type_weights()
    tw_str = ", ".join(f"{k}:{v:.2f}" for k, v in type_weights.items() if v > 0.01)
    n_keys = sum(1 for v in model.key_total.values() if v >= 5)
    total_obs = sum(model.key_total.values())
    print(f"\n  Final: vitality={vitality:.3f}, weights={{{tw_str}}}")
    print(f"  Empirical: {n_keys} keys, {total_obs} total observations")

    # === FASE 3: Build predictions and resubmit ===
    print(f"\n  FASE 3: Final predictions...")
    results = []
    for si in range(n_seeds):
        pred = model.build_prediction(si, opt_tables, model_tables, type_weights)
        dm = ~model.static_masks[si]
        obs_cells = sum(1 for y2 in range(MAP_H) for x2 in range(MAP_W)
                        if not model.static_masks[si][y2, x2] and model.cell_total.get((si, y2, x2), 0) > 0)
        total_dyn = dm.sum()
        print(f"  Seed {si}: {obs_cells}/{total_dyn} dynamic cells observed")

        if not submit:
            results.append({"seed_index": si, "status": "no-submit"})
            continue

        try:
            resp = client.submit(round_id, si, pred.tolist())
            score = resp.get("score", resp.get("seed_score", "?"))
            prior_s = prior_scores[si] if si < len(prior_scores) else None
            diff_str = ""
            if isinstance(score, (int, float)) and isinstance(prior_s, (int, float)):
                diff = score - prior_s
                diff_str = f" ({diff:+.1f} vs safety)"
            print(f"    → {score}{diff_str}")
            results.append({"seed_index": si, "score": score, "prior_score": prior_s})
            time.sleep(0.4)
        except Exception as e:
            print(f"    submit FEIL: {e}")
            results.append({"seed_index": si, "error": str(e)})

    return results


# === BACKTEST ===

def weighted_kl(ground_truth, prediction, initial_grid):
    eps = 1e-12
    gt = np.clip(np.asarray(ground_truth, dtype=float), eps, 1.0)
    pred_arr = np.clip(np.asarray(prediction, dtype=float), eps, 1.0)
    grid = np.asarray(initial_grid, dtype=int)

    cell_kl = np.sum(gt * np.log(gt / pred_arr), axis=-1)
    cell_entropy = -np.sum(gt * np.log(gt), axis=-1)
    dynamic_mask = ~np.isin(grid, [5, 10])

    masked_entropy = cell_entropy * dynamic_mask
    total_entropy = float(masked_entropy.sum())
    if total_entropy > 0:
        return float((masked_entropy * cell_kl).sum() / total_entropy)
    dynamic_kls = cell_kl[dynamic_mask]
    return float(dynamic_kls.mean()) if dynamic_kls.size else float(cell_kl.mean())


def score_from_wkl(value):
    return max(0.0, min(100.0, 100.0 * math.exp(-3.0 * value)))


def run_backtest():
    """Fully offline backtest using locally cached history data + ground truth."""
    opt_tables = load_opt_tables()
    model_tables = load_model_tables()

    if not HISTORY_ROOT.exists():
        print("FEIL: Ingen history-mappe funnet")
        return

    history_dirs = sorted(HISTORY_ROOT.iterdir(), key=lambda d: d.name)
    history_dirs = [d for d in history_dirs if d.is_dir()]
    print(f"\n{len(history_dirs)} history rounds found")

    all_results = []
    hr_scores = []
    po_scores = []

    for hdir in history_dirs:
        manifest_path = hdir / "manifest.json"
        if not manifest_path.exists():
            continue

        manifest = json.loads(manifest_path.read_text())
        rid = manifest.get("round_id", hdir.name)
        rmeta = manifest.get("round_metadata", {})
        rnum = rmeta.get("round_number", "?")

        # Load initial states from manifest
        init_states = manifest.get("initial_states", [])
        if not init_states:
            print(f"  R{rnum}: skip (no initial_states in manifest)")
            continue

        # Build seeds_data from local files
        seeds_data = []
        for ist in init_states:
            si = ist["seed_index"]
            grid_path = hdir / ist["grid_path"]
            if not grid_path.exists():
                break
            grid_arr = np.load(grid_path)
            seeds_data.append({
                "grid": grid_arr.tolist(),
                "settlements": ist.get("settlements", []),
                "seed_index": si,
            })

        if len(seeds_data) != len(init_states):
            print(f"  R{rnum}: skip (incomplete grid files)")
            continue

        n_seeds = len(seeds_data)

        # Load ground truth from local files
        gt_data = {}
        for si in range(n_seeds):
            gt_path = hdir / "arrays" / f"seed_{si}_ground_truth.npy"
            if gt_path.exists():
                gt_data[si] = np.load(gt_path).astype(float)

        if len(gt_data) < n_seeds:
            print(f"  R{rnum}: skip (incomplete GT: {len(gt_data)}/{n_seeds})")
            continue

        model = RoundModel(seeds_data)

        # Load stored observations
        queries = manifest.get("queries", [])
        if queries:
            try:
                for q in queries:
                    si = q["seed_index"]
                    vp = q["viewport"]
                    grid_path = hdir / q["grid_path"]
                    if grid_path.exists():
                        obs_grid = np.load(grid_path).tolist()
                        model.add_observation(si, obs_grid, vp["x"], vp["y"])
                mode = "history_replay"
            except Exception as e:
                print(f"  R{rnum}: history load error ({e}), using prior_only")
                mode = "prior_only"
                model = RoundModel(seeds_data)
        else:
            mode = "prior_only"

        # Build predictions
        if mode == "history_replay":
            type_weights = model.compute_type_weights()
        else:
            type_weights = {"DEAD": 0.0, "STABLE": 0.50, "BOOM": 0.50}

        seed_scores = []
        for si in range(n_seeds):
            if mode == "history_replay":
                pred = model.build_prediction(si, opt_tables, model_tables, type_weights)
            else:
                pred = model.build_prior_prediction(si, opt_tables, model_tables, type_weights)

            grid = seeds_data[si].get("grid", [])
            wkl = weighted_kl(gt_data[si], pred, grid)
            sc = score_from_wkl(wkl)
            seed_scores.append(sc)

        round_score = sum(seed_scores) / len(seed_scores)
        if mode == "history_replay":
            hr_scores.append(round_score)
        else:
            po_scores.append(round_score)

        vit_str = f"v={model.get_vitality():.2f}" if mode == "history_replay" else ""
        print(f"  R{rnum} ({mode}): {round_score:.2f}  "
              f"[{', '.join(f'{s:.1f}' for s in seed_scores)}]  {vit_str}")

        all_results.append({
            "round_number": rnum, "round_id": rid, "mode": mode,
            "round_score": round_score, "seed_scores": seed_scores,
        })

    print(f"\n=== BACKTEST SUMMARY ===")
    if hr_scores:
        print(f"  History replay: {sum(hr_scores)/len(hr_scores):.2f} avg ({len(hr_scores)} rounds)")
    if po_scores:
        print(f"  Prior only:     {sum(po_scores)/len(po_scores):.2f} avg ({len(po_scores)} rounds)")
    if all_results:
        all_avg = sum(r["round_score"] for r in all_results) / len(all_results)
        print(f"  Total:          {all_avg:.2f} avg ({len(all_results)} rounds)")

    out = PROJECT_ROOT / "backtest_v2_results.json"
    out.write_text(json.dumps({
        "completed_rounds": all_results,
        "summary": {
            "hr_avg": sum(hr_scores) / len(hr_scores) if hr_scores else 0,
            "po_avg": sum(po_scores) / len(po_scores) if po_scores else 0,
            "total_avg": sum(r["round_score"] for r in all_results) / len(all_results) if all_results else 0,
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    print(f"  Results saved: {out}")


# === MAIN ===

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=None)
    parser.add_argument("--no-submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--leaderboard", action="store_true")
    parser.add_argument("--backtest", action="store_true")
    args = parser.parse_args()

    if args.backtest:
        run_backtest()
        return

    if not API_KEY:
        print("FEIL: export API_KEY='din-jwt-token'")
        sys.exit(1)

    client = AstarClient()

    if args.leaderboard:
        try:
            lb = client.get_leaderboard()
            print("=== LEADERBOARD ===")
            for i, e in enumerate(lb[:15]):
                if isinstance(e, dict):
                    name = e.get("team_name", e.get("username", "?"))
                    sc = e.get("weighted_score", e.get("score", "?"))
                    print(f"  #{i+1} {name}: {sc}")
        except Exception as e:
            print(f"Leaderboard feil: {e}")
        return

    try:
        rounds = client.get_rounds()
    except Exception as e:
        print(f"FEIL: {e}")
        sys.exit(1)

    if args.round:
        round_id = args.round
    else:
        active = [r for r in rounds if isinstance(r, dict) and r.get("status") == "active"]
        if not active:
            print("Ingen aktive runder.")
            sys.exit(1)
        round_id = active[-1]["id"]

    try:
        round_data = client.get_round(round_id)
    except Exception as e:
        print(f"FEIL: {e}")
        sys.exit(1)

    rnum = round_data.get("round_number", "?")
    weight = round_data.get("round_weight", "?")
    print(f"Runde {rnum} (vekt {weight}): {round_id[:12]}...")
    print(f"Stenger: {round_data.get('closes_at', '?')}")

    if args.dry_run:
        print("\n[DRY RUN]")
        seeds = round_data.get("seeds", round_data.get("initial_states", []))
        for i, s in enumerate(seeds):
            print(f"  Seed {i}: {len(s.get('settlements', []))} settlements")
        return

    results = solve_round(client, round_id, round_data, submit=not args.no_submit)

    print("\n=== RESULTATER ===")
    scores = []
    for r in results:
        s = r.get("score", r.get("error", r.get("status", "?")))
        print(f"  Seed {r['seed_index']}: {s}")
        if isinstance(r.get("score"), (int, float)):
            scores.append(r["score"])
    if scores:
        avg = sum(scores) / len(scores)
        print(f"\n  Snitt: {avg:.1f}")


if __name__ == "__main__":
    main()
