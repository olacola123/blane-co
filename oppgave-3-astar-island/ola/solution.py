"""
Ola's Astar Island Solver v8 — Soft regime inference
=====================================================================
Forbedringer over v6/v7:
1. Soft regime blending: smooth transitions between DEAD/STABLE/BOOM priors
2. Multi-signal fingerprint: survival + ruin + empty + forest rates
3. Diagnostic probes across 4 seeds (not 2) for better regime inference
4. Global recalibration instead of aggressive cell-level observation updates
5. Reduced cross-seed aggressiveness (50% max, was 80%)
6. Prior-only safety submit med soft-blended default

Bevart fra v6:
- Safety submit → probe → observe → cross-seed → resubmit pipeline
- Optimerte calibration tabeller (calibration_optimized.json)
- Floor/renormalisering/robuste defaults

Bruk:
    export API_KEY='din-jwt-token'
    python solution.py                    # Løs aktiv runde
    python solution.py --dry-run          # Vis plan uten API-kall
    python solution.py --no-submit        # Observer uten submit
    python solution.py --leaderboard      # Vis leaderboard
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

# === CONFIG ===
BASE_URL = "https://api.ainm.no/astar-island"
API_KEY = os.environ.get("API_KEY", "")

MAP_W, MAP_H = 40, 40
NUM_CLASSES = 6
MAX_VIEWPORT = 15
PROB_FLOOR = 0.001      # optimal floor (fra grid search over 14 runder × 5 seeds)
# Per-vbin optimal floor:
VBIN_FLOOR = {"DEAD": 0.001, "LOW": 0.001, "MED": 0.001, "HIGH": 0.001}
SHARP_FLOOR = 0.001     # floor for observerte/statiske celler
WIDE_FLOOR = 0.003      # floor for uobserverte usikre celler
NEAR_ZERO = 0.001       # floor for umulige klasser (port innland, mountain på slette)
DEFAULT_ALPHA = 3.5     # Joakim-inspirert: lavere prior-styrke → obs teller mer

TERRAIN_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}
DISTANCE_BANDS = [0, 1, 2, 3, 5, 8, 12, 99]

# BUG 2 FIX: Mountain=0.0001 for alle dynamiske terreng (aldri fjell på slette)
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

CALIBRATION_FILE = Path(__file__).parent / "calibration_data.json"
CALIBRATION_BY_TYPE_FILE = Path(__file__).parent / "calibration_by_type.json"
CALIBRATION_4TYPE_FILE = Path(__file__).parent / "calibration_4type.json"
CALIBRATION_OPT_FILE = Path(__file__).parent / "calibration_optimized.json"
SUPER_CALIBRATION_FILE = Path(__file__).parent / "super_calibration.json"
MODEL_TABLES_FILE = Path(__file__).parent / "model_tables.json"
LEARNING_FILE = Path(__file__).parent / "learning_state.json"

# Terrain code → terrain group for optimized tables
TERRAIN_GROUP = {0: "plains", 1: "settlement", 2: "port", 3: "ruin", 4: "forest", 5: "mountain", 10: "ocean", 11: "plains"}
# Optimized distance bands: [(0,0), (1,1), (2,2), (3,3), (4,5), (6,8), (9,99)]
OPT_DIST_BANDS = [(0,0), (1,1), (2,2), (3,3), (4,5), (6,8), (9,99)]

def get_opt_band(dist):
    for i, (lo, hi) in enumerate(OPT_DIST_BANDS):
        if lo <= dist <= hi:
            return i
    return len(OPT_DIST_BANDS) - 1


# === CALIBRATION + LEARNING STATE ===

def load_optimized_calibration():
    """Last optimerte prior-tabeller (wtype × terrain_group × dist_band × coastal)."""
    if not CALIBRATION_OPT_FILE.exists():
        return None
    data = json.loads(CALIBRATION_OPT_FILE.read_text())
    tables = data.get("tables", {})
    if tables:
        print(f"Lastet optimerte tabeller: {len(tables)} entries fra {data.get('num_rounds', '?')} runder × {data.get('num_seeds', '?')} seeds")
    return tables


# === SUPER-CALIBRATION ===
# Built from 14 rounds × 5 seeds = 70 GT datasets, 95k data points
# Features: vitality_bin × terrain_group × dist_bin × coastal × settle_density × forest_density

_model_cache = None

def load_model_tables():
    """Load new model tables (built from 14 rounds × 5 seeds = 70 GT datasets)."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    if not MODEL_TABLES_FILE.exists():
        print("ADVARSEL: Ingen model_tables.json")
        return None
    data = json.loads(MODEL_TABLES_FILE.read_text())
    _model_cache = {
        "specific": data.get("table_specific", {}),
        "medium": data.get("table_medium", {}),
        "simple": data.get("table_simple", {}),
    }
    n_spec = len(_model_cache["specific"])
    n_med = len(_model_cache["medium"])
    n_simp = len(_model_cache["simple"])
    print(f"Lastet model v7: {n_spec} specific + {n_med} medium + {n_simp} simple entries")
    return _model_cache


# Also keep old super_calibration as fallback
_super_cal_cache = None

def load_super_calibration():
    """Load super-calibration tables (legacy fallback)."""
    global _super_cal_cache
    if _super_cal_cache is not None:
        return _super_cal_cache
    if not SUPER_CALIBRATION_FILE.exists():
        return None
    data = json.loads(SUPER_CALIBRATION_FILE.read_text())
    _super_cal_cache = {
        "specific": data.get("table_specific", {}),
        "density": data.get("table_density", {}),
        "simple": data.get("table_simple", {}),
    }
    return _super_cal_cache


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


def super_predict(grid, settlements, vbin, floor=None):
    """
    Build prediction using model v7 tables (14 rounds × 5 seeds = 95k data points).
    vbin: "DEAD", "LOW", "MED", "HIGH"
    Returns: (H, W, 6) numpy array

    Cascading lookup: specific → medium → simple → uniform
    - specific: vbin × terrain × dist × coastal × settle_density × forest_density
    - medium:   vbin × terrain × dist × coastal
    - simple:   terrain × dist
    """
    if floor is None:
        floor = VBIN_FLOOR.get(vbin, 0.001)

    model = load_model_tables()
    if model is None:
        return None

    table_specific = model["specific"]
    table_medium = model["medium"]
    table_simple = model["simple"]

    grid_arr = np.array(grid, dtype=int) if not isinstance(grid, np.ndarray) else grid
    H, W = grid_arr.shape if hasattr(grid_arr, 'shape') else (len(grid), len(grid[0]))
    pred = np.zeros((H, W, NUM_CLASSES), dtype=float)

    for y in range(H):
        for x in range(W):
            terrain = int(grid_arr[y][x]) if isinstance(grid_arr, np.ndarray) else grid[y][x]

            if terrain == 10:  # ocean
                pred[y, x] = [1.0-5*floor, floor, floor, floor, floor, floor]
                continue
            if terrain == 5:  # mountain
                pred[y, x] = [floor, floor, floor, floor, floor, 1.0-5*floor]
                continue

            # Features
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
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < H and 0 <= nx < W:
                        t = int(grid_arr[ny][nx]) if isinstance(grid_arr, np.ndarray) else grid[ny][nx]
                        if t == 10:
                            coastal = True
                            break
                if coastal:
                    break

            n_forest_r2 = 0
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < H and 0 <= nx < W:
                        t = int(grid_arr[ny][nx]) if isinstance(grid_arr, np.ndarray) else grid[ny][nx]
                        if t == 4:
                            n_forest_r2 += 1

            tg = _terrain_group(terrain)
            db = _dist_bin(min_dist)
            c = int(coastal)
            sdb = _settle_density_bin(n_settle_r3)
            fdb = _forest_density_bin(n_forest_r2)

            # Cascading lookup: specific → medium → simple → uniform
            p = None

            key_spec = f"{vbin}_{tg}_{db}_{c}_{sdb}_{fdb}"
            if key_spec in table_specific:
                p = np.array(table_specific[key_spec]["distribution"])

            if p is None:
                key_med = f"{vbin}_{tg}_{db}_{c}"
                if key_med in table_medium:
                    p = np.array(table_medium[key_med]["distribution"])

            if p is None:
                key_simp = f"{tg}_{db}"
                if key_simp in table_simple:
                    p = np.array(table_simple[key_simp]["distribution"])

            if p is None:
                p = np.ones(NUM_CLASSES) / NUM_CLASSES

            # Floor via Joakim's linear mixing method
            p = p * (1 - NUM_CLASSES * floor) + floor
            p /= p.sum()
            pred[y, x] = p

    return pred


def vitality_to_vbin(vitality):
    """Map vitality float to model bin.
    Bins calibrated from 14 rounds × 5 seeds ground truth:
    DEAD: vitality < 0.08 (rounds 3, 8, 10)
    LOW:  vitality 0.08-0.20 (rare, only 1 seed in data)
    MED:  vitality 0.20-0.35 (rounds 4, 9, 13)
    HIGH: vitality >= 0.35 (rounds 1, 2, 5, 6, 7, 11, 12, 14)
    """
    if vitality < 0.08:
        return "DEAD"
    elif vitality < 0.20:
        return "LOW"
    elif vitality < 0.35:
        return "MED"
    else:
        return "HIGH"


def load_calibration():
    if not CALIBRATION_FILE.exists():
        print("ADVARSEL: Ingen calibration_data.json — bruker fallback priors")
        return None, None
    data = json.loads(CALIBRATION_FILE.read_text())
    transition_table = data.get("transition_table", {})
    simple_prior = data.get("simple_prior", {})
    print(f"Lastet kalibrering fra {data.get('num_rounds', '?')} runder, {data.get('num_seeds', '?')} seeds")
    return transition_table, simple_prior


def load_calibration_by_type():
    """Last type-spesifikke calibration-tabeller. Foretrekker 4-type (DEAD/STABLE/BOOM_SPREAD/BOOM_CONC)."""
    # Foretrekk 4-type calibration
    if CALIBRATION_4TYPE_FILE.exists():
        data = json.loads(CALIBRATION_4TYPE_FILE.read_text())
        tables = data.get("tables", {})
        if tables:
            # Map BOOM_SPREAD/BOOM_CONC til BOOMING for blending.py-kompatibilitet
            # men behold originale nøkler for direkte oppslag
            print(f"Lastet 4-type tabeller: {', '.join(f'{k}({len(v)} keys)' for k, v in tables.items())}")
            return tables

    if not CALIBRATION_BY_TYPE_FILE.exists():
        return None
    data = json.loads(CALIBRATION_BY_TYPE_FILE.read_text())
    tables = data.get("tables", {})
    if tables:
        print(f"Lastet 3-type tabeller: {', '.join(f'{k}({len(v)} keys)' for k, v in tables.items())}")
    return tables


def load_learning_state():
    """Last adaptiv tilstand (optimal alpha, runde-historikk)."""
    if LEARNING_FILE.exists():
        try:
            return json.loads(LEARNING_FILE.read_text())
        except Exception:
            pass
    return {"alpha": DEFAULT_ALPHA, "round_history": [], "obs_helped_count": 0, "obs_hurt_count": 0}


def save_learning_state(state):
    LEARNING_FILE.write_text(json.dumps(state, indent=2, default=str))


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

    def simulate(self, round_id, seed_index, x, y, w=MAX_VIEWPORT, h=MAX_VIEWPORT):
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


# === HELPERS ===

def distance_to_nearest_settlement(y, x, settlements):
    if not settlements:
        return 99
    return min(max(abs(x - s["x"]), abs(y - s["y"])) for s in settlements)

def get_distance_band(dist):
    for i in range(len(DISTANCE_BANDS) - 1):
        if dist <= DISTANCE_BANDS[i]:
            return i
    return len(DISTANCE_BANDS) - 2

def is_coastal(grid, y, x):
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < MAP_H and 0 <= nx < MAP_W:
                if grid[ny][nx] == 10:
                    return True
    return False

def get_prior(terrain, band, coastal, transition_table, simple_prior,
              typed_table=None, opt_tables=None, world_type=None, dist_raw=None):
    """Hent prior med cascading fallback.

    Prioritet:
    1. Optimerte tabeller (wtype × terrain_group × dist_band × coastal) — best
    2. Type-spesifikk tabell (4-type calibration)
    3. Generell calibration
    4. Simple prior / fallback
    """
    # === PRIORITET 1: Optimerte tabeller ===
    if opt_tables and world_type:
        tg = TERRAIN_GROUP.get(terrain, "plains")
        d = dist_raw if dist_raw is not None else band
        opt_band = get_opt_band(d) if dist_raw is not None else band

        # Try most specific → least specific
        for key in [
            f"{world_type}_{tg}_{opt_band}_{int(coastal)}",
            f"{world_type}_{tg}_{opt_band}_any",
            f"ALL_{tg}_{opt_band}_{int(coastal)}",
            f"ALL_{tg}_{opt_band}_any",
        ]:
            if key in opt_tables:
                return np.array(opt_tables[key]["distribution"], dtype=float)

    # === PRIORITET 2: Type-spesifikk tabell ===
    key = f"{terrain}_{band}_{int(coastal)}"
    key_nc = f"{terrain}_{band}_0"
    if typed_table:
        if key in typed_table:
            return np.array(typed_table[key]["distribution"], dtype=float)
        if key_nc in typed_table:
            return np.array(typed_table[key_nc]["distribution"], dtype=float)

    # === PRIORITET 3: Generell calibration ===
    if transition_table:
        if key in transition_table:
            return np.array(transition_table[key]["distribution"], dtype=float)
        if key_nc in transition_table:
            return np.array(transition_table[key_nc]["distribution"], dtype=float)

    # === PRIORITET 4: Fallback ===
    if simple_prior and str(terrain) in simple_prior:
        return np.array(simple_prior[str(terrain)], dtype=float)
    if str(terrain) in FALLBACK_PRIOR:
        return np.array(FALLBACK_PRIOR[str(terrain)], dtype=float)
    return np.ones(NUM_CLASSES, dtype=float) / NUM_CLASSES

def cell_key(grid, y, x, settlements):
    terrain = int(grid[y][x]) if isinstance(grid, np.ndarray) else grid[y][x]
    dist = distance_to_nearest_settlement(y, x, settlements)
    band = get_distance_band(dist)
    coastal = is_coastal(grid if isinstance(grid, list) else grid.tolist(), y, x)
    return terrain, band, coastal


# === [FORBEDRING 1] SETTLEMENT-ATTRIBUTT-PREDIKTOR ===

def settlement_survival_signal(settlement_data):
    """
    Bruk settlement-attributter til å predikere overlevelse.
    Returnerer (class_idx, confidence) basert på attributter.

    API gir: population, food, wealth, defense, has_port, alive, owner_id
    """
    alive = settlement_data.get("alive", True)
    has_port = settlement_data.get("has_port", False)
    pop = settlement_data.get("population", 1.0)
    food = settlement_data.get("food", 0.5)
    wealth = settlement_data.get("wealth", 0.5)
    defense = settlement_data.get("defense", 0.5)

    if not alive:
        # Død settlement → ruin eller empty
        return np.array([0.45, 0.05, 0.02, 0.35, 0.10, 0.03]), 0.5

    # Overlevelsesscore basert på attributter
    survival = 0.3 * min(food, 1.0) + 0.3 * min(pop, 2.0)/2.0 + 0.2 * min(defense, 1.0) + 0.2 * min(wealth, 1.0)
    survival = max(0.0, min(1.0, survival))

    if has_port:
        # Port med god overlevelse
        if survival > 0.5:
            return np.array([0.15, 0.15, 0.50, 0.05, 0.12, 0.03]), 0.4
        else:
            return np.array([0.35, 0.10, 0.25, 0.15, 0.12, 0.03]), 0.3
    else:
        if survival > 0.6:
            return np.array([0.20, 0.50, 0.05, 0.08, 0.14, 0.03]), 0.4
        elif survival > 0.3:
            return np.array([0.35, 0.30, 0.05, 0.12, 0.15, 0.03]), 0.3
        else:
            return np.array([0.45, 0.15, 0.03, 0.20, 0.14, 0.03]), 0.3


# === [FORBEDRING 3] SPATIAL SMOOTHING ===

def spatial_smooth(pred, static_mask, sigma=0.7):
    """
    Lett spatial smoothing — naboceller er korrelert.
    Bruker manuell 3×3 average (unngår scipy-avhengighet).
    """
    smoothed = pred.copy()
    kernel = np.array([[0.05, 0.10, 0.05],
                       [0.10, 0.40, 0.10],
                       [0.05, 0.10, 0.05]])
    kernel /= kernel.sum()

    for k in range(NUM_CLASSES):
        layer = pred[:, :, k].copy()
        out = np.zeros_like(layer)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                w = kernel[dy+1, dx+1]
                shifted = np.roll(np.roll(layer, dy, axis=0), dx, axis=1)
                out += w * shifted
        smoothed[:, :, k] = out

    # Ikke smooth statiske celler
    smoothed[static_mask] = pred[static_mask]

    # Renormaliser
    sums = smoothed.sum(axis=2, keepdims=True)
    sums = np.maximum(sums, 1e-10)
    smoothed /= sums

    # Floor via linear mixing
    eps = PROB_FLOOR
    smoothed = smoothed * (1 - NUM_CLASSES * eps) + eps
    smoothed /= smoothed.sum(axis=2, keepdims=True)

    return smoothed


# === OBSERVATION STORE ===

class SeedObserver:
    def __init__(self, initial_grid, settlements, transition_table, simple_prior,
                 alpha=DEFAULT_ALPHA, typed_table=None, type_tables=None, vitality=0.5,
                 opt_tables=None, world_type=None):
        self.grid = np.array(initial_grid, dtype=int)
        self.settlements = settlements
        self.transition_table = transition_table
        self.simple_prior = simple_prior
        self.alpha = alpha
        self.opt_tables = opt_tables
        self.world_type = world_type

        self.counts = np.zeros((MAP_H, MAP_W, NUM_CLASSES), dtype=float)
        self.observed = np.zeros((MAP_H, MAP_W), dtype=int)

        self.ocean_mask = (self.grid == 10)
        self.mountain_mask = (self.grid == 5)
        self.static_mask = self.ocean_mask | self.mountain_mask

        self.coastal_mask = np.zeros((MAP_H, MAP_W), dtype=bool)
        for y in range(MAP_H):
            for x in range(MAP_W):
                if self.static_mask[y, x]:
                    continue
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < MAP_H and 0 <= nx < MAP_W:
                            if self.grid[ny, nx] == 10:
                                self.coastal_mask[y, x] = True
                                break
                    if self.coastal_mask[y, x]:
                        break

        # Forhåndsberegn prior + floor per celle
        self._prior_cache = np.zeros((MAP_H, MAP_W, NUM_CLASSES), dtype=float)
        self._floor_cache = np.full((MAP_H, MAP_W, NUM_CLASSES), PROB_FLOOR, dtype=float)
        self._rebuild_priors(typed_table, type_tables, vitality)

    def _rebuild_priors(self, typed_table=None, type_tables=None, vitality=0.5):
        """Bygg prior-cache. Kalles på nytt hvis world_type endres."""
        grid_list = self.grid.tolist()
        for y in range(MAP_H):
            for x in range(MAP_W):
                if self.static_mask[y, x]:
                    continue
                terrain = int(self.grid[y, x])
                dist = distance_to_nearest_settlement(y, x, self.settlements)
                band = get_distance_band(dist)
                coastal = self.coastal_mask[y, x]

                # Primær: optimerte tabeller (best)
                if self.opt_tables and self.world_type:
                    self._prior_cache[y, x] = get_prior(
                        terrain, band, coastal,
                        self.transition_table, self.simple_prior,
                        opt_tables=self.opt_tables,
                        world_type=self.world_type,
                        dist_raw=dist,
                    )
                elif type_tables:
                    from blending import get_blended_prior
                    self._prior_cache[y, x] = get_blended_prior(
                        terrain, band, coastal, type_tables, vitality,
                        transition_table=self.transition_table,
                        simple_prior=self.simple_prior,
                        fallback_prior=FALLBACK_PRIOR,
                    )
                else:
                    self._prior_cache[y, x] = get_prior(
                        terrain, band, coastal,
                        self.transition_table, self.simple_prior,
                        typed_table=typed_table,
                    )
                # Mountain umulig på dynamiske celler
                self._floor_cache[y, x, 5] = NEAR_ZERO
                if not coastal:
                    self._floor_cache[y, x, 2] = NEAR_ZERO

    def update_world_type(self, new_type):
        """Oppdater world type og rebuild priors."""
        if new_type != self.world_type:
            self.world_type = new_type
            self._rebuild_priors()

    def add_observation(self, grid_data, viewport_x, viewport_y):
        for dy, row in enumerate(grid_data):
            for dx, val in enumerate(row):
                y, x = viewport_y + dy, viewport_x + dx
                if 0 <= y < MAP_H and 0 <= x < MAP_W:
                    cls = TERRAIN_TO_CLASS.get(val, 0)
                    self.counts[y, x, cls] += 1
                    self.observed[y, x] += 1

    def add_settlement_obs(self, settlements_data):
        """Settlement-attributter (DEAKTIVERT — marginal effekt, legger til risiko)."""
        pass

    def build_prediction(self, apply_smoothing=True, world_type=None):
        """Bayesiansk posterior med Joakim-inspirert alpha-decay + type-spesifikk floor."""
        pred = np.zeros((MAP_H, MAP_W, NUM_CLASSES), dtype=float)
        alpha = self.alpha

        # Optimal floor = 0.001 for alle typer (grid search over 14 runder × 5 seeds)
        base_floor = 0.001

        for y in range(MAP_H):
            for x in range(MAP_W):
                if self.ocean_mask[y, x]:
                    pred[y, x] = [0.998, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004]
                    continue
                if self.mountain_mask[y, x]:
                    pred[y, x] = [0.0004, 0.0004, 0.0004, 0.0004, 0.0004, 0.998]
                    continue

                prior = self._prior_cache[y, x]
                n_obs = self.observed[y, x]

                if n_obs == 0:
                    pred[y, x] = prior.copy()
                else:
                    # VIKTIG: Med <50 obs per celle er prioren mer nøyaktig enn obs!
                    # Analysen viser: 1 obs → score 0.3, prior alene → score 82.
                    # Bruk svært høy prior-vekt for å unngå at støy ødelegger.
                    # a = prior pseudo-count (høy = mer prior-tillit)
                    if n_obs >= 10:
                        a = 0.5   # 10+ obs: 67% obs, 33% prior
                    elif n_obs >= 5:
                        a = 2.0   # 5 obs: 71% prior, 29% obs
                    elif n_obs >= 3:
                        a = 4.0   # 3 obs: 57% prior, 43% obs
                    elif n_obs >= 2:
                        a = 6.0   # 2 obs: 75% prior, 25% obs
                    else:
                        a = 10.0  # 1 obs: 91% prior, 9% obs — minimal obs-effekt

                    pred[y, x] = self.counts[y, x] + a * prior

                # Normaliser
                s = pred[y, x].sum()
                if s > 0:
                    pred[y, x] /= s

                # Floor: observerte celler får skarpere floor
                if n_obs >= 2:
                    eps = base_floor * 0.5  # halvparten av base
                elif n_obs >= 1:
                    eps = base_floor
                else:
                    eps = base_floor

                pred[y, x] = pred[y, x] * (1 - NUM_CLASSES * eps) + eps
                pred[y, x] /= pred[y, x].sum()

        return pred


# === CROSS-SEED LEARNING ===

def build_cross_seed_prior(all_observers):
    key_counts = defaultdict(lambda: np.zeros(NUM_CLASSES, dtype=float))
    for obs in all_observers:
        grid_list = obs.grid.tolist()
        for y in range(MAP_H):
            for x in range(MAP_W):
                if obs.static_mask[y, x] or obs.observed[y, x] == 0:
                    continue
                terrain, band, coastal = cell_key(grid_list, y, x, obs.settlements)
                key = f"{terrain}_{band}_{int(coastal)}"
                key_counts[key] += obs.counts[y, x]
    cross_table = {}
    for key, counts in key_counts.items():
        total = counts.sum()
        if total >= 3:
            cross_table[key] = {"distribution": (counts / total).tolist(), "sample_count": int(total)}
    return cross_table


def apply_cross_seed(observers, cross_table, calibration_table):
    if not cross_table:
        return
    for obs in observers:
        grid_list = obs.grid.tolist()
        for y in range(MAP_H):
            for x in range(MAP_W):
                if obs.static_mask[y, x]:
                    continue
                terrain, band, coastal = cell_key(grid_list, y, x, obs.settlements)
                key = f"{terrain}_{band}_{int(coastal)}"
                if key in cross_table:
                    cross_dist = np.array(cross_table[key]["distribution"])
                    cross_n = cross_table[key]["sample_count"]
                    hist_prior = obs._prior_cache[y, x]
                    # Moderat cross-seed: round-spesifikk data, men begrenset
                    # for å unngå overfitting til støy fra få observasjoner
                    cross_weight = min(0.50, cross_n / 80.0)
                    blended = (1 - cross_weight) * hist_prior + cross_weight * cross_dist
                    # BUG 3 FIX: Klasse-spesifikk floor (ikke uniform)
                    np.maximum(blended, obs._floor_cache[y, x], out=blended)
                    blended /= blended.sum()
                    obs._prior_cache[y, x] = blended


# === [FORBEDRING 4] RUNDETYPE-MATCHING ===

def infer_vitality(observers):
    """
    Inferér world vitality fra observasjoner (continuous).
    Uses piecewise linear mapping fra blending.py.
    Returns: float 0.0 (dead) til 1.0 (booming)
    """
    from blending import infer_vitality_continuous
    return infer_vitality_continuous(observers)


def classify_world_type(seeds_data, vitality):
    """
    4-type klassifisering: DEAD / STABLE / BOOM_SPREAD / BOOM_CONC.
    Bruker vitality (fra observasjoner) + n_settlements (gratis fra initial data).

    Nøkkel-innsikt: n_settlements >= 40 → BOOM_CONC (konsentrert, kort rekkevidde)
    """
    n_settlements = len(seeds_data[0].get("settlements", []))

    if vitality < 0.20:
        return "DEAD", n_settlements
    elif vitality < 0.55:
        return "STABLE", n_settlements
    else:
        # Booming — skill spread vs concentrated
        if n_settlements >= 40:
            return "BOOM_CONC", n_settlements
        else:
            return "BOOM_SPREAD", n_settlements


def adjust_priors_for_vitality(observers, vitality):
    """DEPRECATED — blending.py handles this via get_blended_prior()."""
    pass  # Replaced by continuous blending in v6


# === SOFT REGIME INFERENCE (v8) ===

def compute_type_weights(survival_rate):
    """Map survival rate to soft weights over DEAD/STABLE/BOOM world types.

    Replaces hard classification with smooth transitions at bin boundaries.
    Calibrated from 14 rounds ground truth:
    - DEAD:    survival < 0.10 (R3, R8, R10)
    - STABLE:  survival 0.10-0.33 (R4, R9, R13)
    - BOOM:    survival > 0.33 (R1, R2, R5, R6, R7, R11, R12, R14)

    Transition zones: ±0.05 around each boundary.
    """
    s = max(0.0, min(1.0, survival_rate))

    # DEAD: full weight below 0.05, fades to 0 at 0.15
    w_dead = max(0.0, min(1.0, (0.15 - s) / 0.10)) if s < 0.15 else 0.0
    # BOOM: 0 below 0.28, full weight above 0.38
    w_boom = max(0.0, min(1.0, (s - 0.28) / 0.10)) if s > 0.28 else 0.0
    # STABLE: fills the remainder
    w_stable = max(0.0, 1.0 - w_dead - w_boom)

    total = w_dead + w_stable + w_boom
    if total < 1e-10:
        return {"DEAD": 0.0, "STABLE": 1.0, "BOOM": 0.0}

    return {
        "DEAD": w_dead / total,
        "STABLE": w_stable / total,
        "BOOM": w_boom / total,
    }


def compute_round_fingerprint(probe_observers):
    """Extract multi-signal round fingerprint from probe observations.

    Aggregates statistics across all probed seeds for robust regime inference.
    More signals = less chance of misclassification from noisy single metric.

    Returns dict with:
    - survival_rate: fraction of initial settlements still alive (settlement+port)
    - port_rate: ports among survived settlements
    - ruin_rate: fraction of settlement positions now ruins
    - forest_rate: forest encroachment around settlement positions
    - empty_rate: fraction of settlement positions now empty
    - n_observed: total settlements observed (confidence metric)
    """
    total_settlements = 0
    survived = 0.0
    ports = 0.0
    ruins = 0.0
    empties = 0.0
    forest_neighbors = 0
    total_neighbors = 0

    for obs in probe_observers:
        for s in obs.settlements:
            sx, sy = s.get("x", -1), s.get("y", -1)
            if not (0 <= sx < 40 and 0 <= sy < 40):
                continue
            if obs.observed[sy, sx] == 0:
                continue

            total_settlements += 1
            n = obs.observed[sy, sx]
            cls_frac = obs.counts[sy, sx] / n

            survived += cls_frac[1] + cls_frac[2]  # settlement + port = alive
            ports += cls_frac[2]
            ruins += cls_frac[3]
            empties += cls_frac[0]

            # Check neighbors for forest encroachment signal
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = sy + dy, sx + dx
                    if 0 <= ny < 40 and 0 <= nx < 40 and obs.observed[ny, nx] > 0:
                        total_neighbors += 1
                        forest_neighbors += obs.counts[ny, nx, 4] / obs.observed[ny, nx]

    if total_settlements == 0:
        return {
            "survival_rate": 0.30,  # default conservative estimate
            "port_rate": 0.1,
            "ruin_rate": 0.1,
            "forest_rate": 0.2,
            "empty_rate": 0.4,
            "n_observed": 0,
        }

    return {
        "survival_rate": survived / total_settlements,
        "port_rate": ports / max(survived, 0.01),
        "ruin_rate": ruins / total_settlements,
        "forest_rate": forest_neighbors / max(total_neighbors, 1),
        "empty_rate": empties / total_settlements,
        "n_observed": total_settlements,
    }


def fingerprint_to_vitality(fp):
    """Convert multi-signal fingerprint to vitality estimate.

    Primary signal: survival_rate (most predictive, maps ~linearly to vitality).
    Secondary signals: ruin_rate and empty_rate refine estimate when enough data.
    """
    vitality = fp["survival_rate"]

    # Small secondary adjustments when we have enough observed settlements
    if fp["n_observed"] >= 3:
        # More ruins than average → world is slightly more destructive
        vitality -= (fp["ruin_rate"] - 0.12) * 0.08
        # More empty than average → also more destructive
        vitality -= (fp["empty_rate"] - 0.40) * 0.05

    return max(0.0, min(1.0, vitality))


def build_blended_prediction(grid, settlements, transition_table, simple_prior,
                              opt_tables, type_weights):
    """Build prior-only prediction with soft world-type blending.

    Creates a SeedObserver for each world type with significant weight,
    builds their prior-only predictions, and blends them.

    This avoids the cliff effect of hard type classification — a survival
    rate near the STABLE/BOOM boundary gets contributions from both tables.
    """
    preds = []
    weights = []
    for wtype, w in type_weights.items():
        if w < 0.01:
            continue
        obs = SeedObserver(grid, settlements, transition_table, simple_prior,
                          alpha=DEFAULT_ALPHA, opt_tables=opt_tables, world_type=wtype)
        pred = obs.build_prediction(apply_smoothing=False)
        preds.append(pred)
        weights.append(w)

    if not preds:
        return None

    result = np.zeros_like(preds[0])
    total_w = sum(weights)
    for pred, w in zip(preds, weights):
        result += (w / total_w) * pred

    # Renormalize (float precision from weighted sum)
    sums = result.sum(axis=2, keepdims=True)
    result /= np.maximum(sums, 1e-10)

    return result


def recalibrate_pred(pred, fingerprint, static_mask):
    """Global prior recalibration based on round fingerprint.

    Applies multiplicative adjustments to class probabilities based on how
    this round's fingerprint deviates from the "average" round. This provides
    continuous fine-tuning within each world type, especially helping BOOM
    rounds where survival can range from 0.35 to 0.60+.

    Adjustments are clamped to ±15% to prevent overcorrection.
    """
    if fingerprint["n_observed"] < 3:
        return pred  # not enough data for reliable adjustment

    survival = fingerprint["survival_rate"]
    ruin_rate = fingerprint["ruin_rate"]

    # Expected values for a "typical" round (weighted avg across calibration data)
    expected_survival = 0.33
    expected_ruin = 0.12

    survival_dev = survival - expected_survival
    ruin_dev = ruin_rate - expected_ruin

    # Adjustment strength — modest to avoid overcorrection
    strength = 0.12

    # Class order: [empty, settlement, port, ruin, forest, mountain]
    adjustments = np.ones(NUM_CLASSES)
    adjustments[0] -= survival_dev * strength        # less empty if more survived
    adjustments[1] += survival_dev * strength        # more settlement if more survived
    adjustments[2] += survival_dev * strength * 0.5  # ports correlate with survival
    adjustments[3] -= survival_dev * strength        # less ruin if more survived
    adjustments[4] -= survival_dev * strength * 0.3  # forest slightly less if thriving
    # Extra ruin adjustment
    adjustments[3] += ruin_dev * strength * 0.5
    adjustments[1] -= ruin_dev * strength * 0.3

    # Clamp: never more than ±15% relative change
    adjustments = np.clip(adjustments, 0.85, 1.15)

    result = pred.copy()
    dynamic = ~static_mask
    # Apply adjustments to all dynamic cells at once
    result[dynamic] *= adjustments
    # Floor
    result[dynamic] = np.maximum(result[dynamic], PROB_FLOOR)
    # Renormalize
    sums = result.sum(axis=2, keepdims=True)
    result /= np.maximum(sums, 1e-10)

    return result


def apply_cross_seed_to_pred(pred, grid, settlements, cross_table, static_mask,
                              max_weight=0.15):
    """Apply light cross-seed signal to a prediction.

    Uses round-specific cross-seed observations (aggregated across seeds) to
    make small adjustments. Much lighter than the old approach of replacing
    up to 80% of the prior.
    """
    if not cross_table:
        return pred

    result = pred.copy()
    grid_list = grid if isinstance(grid, list) else np.array(grid).tolist()
    for y in range(MAP_H):
        for x in range(MAP_W):
            if static_mask[y, x]:
                continue
            terrain, band, coastal = cell_key(grid_list, y, x, settlements)
            key = f"{terrain}_{band}_{int(coastal)}"
            if key in cross_table:
                cross_dist = np.array(cross_table[key]["distribution"])
                cross_n = cross_table[key]["sample_count"]
                # Scale by sample count: 30+ samples → full max_weight
                w = max_weight * min(1.0, cross_n / 30.0)
                result[y, x] = (1 - w) * result[y, x] + w * cross_dist
                np.maximum(result[y, x], PROB_FLOOR, out=result[y, x])
                result[y, x] /= result[y, x].sum()

    return result


# === QUERY PLANNER ===

def build_dynamism_heatmap(initial_grid, settlements):
    grid = np.array(initial_grid, dtype=int)
    heatmap = np.zeros((MAP_H, MAP_W), dtype=float)
    for s in settlements:
        sx, sy = s["x"], s["y"]
        for dy in range(-12, 13):
            for dx in range(-12, 13):
                y, x = sy + dy, sx + dx
                if 0 <= y < MAP_H and 0 <= x < MAP_W:
                    d = max(abs(dx), abs(dy))
                    if d == 0: heatmap[y, x] += 5.0
                    elif d <= 2: heatmap[y, x] += 3.0
                    elif d <= 5: heatmap[y, x] += 1.5
                    elif d <= 8: heatmap[y, x] += 0.5
                    else: heatmap[y, x] += 0.1
    heatmap[grid == 10] = 0.0
    heatmap[grid == 5] = 0.0
    return heatmap


def plan_queries(initial_grid, settlements, n_queries=10, entropy_map=None):
    """
    Plan viewport placements med hybrid settlement + entropy heuristikk.
    Første 2 queries: maks settlement-dekning (for type detection).
    Resten: balanse mellom settlement-nærhet og entropi (usikkerhet).
    """
    heatmap = build_dynamism_heatmap(initial_grid, settlements)

    # Legg til entropy-komponent hvis tilgjengelig
    if entropy_map is not None:
        # Normaliser entropy til 0-1
        e_max = entropy_map.max()
        if e_max > 0:
            norm_entropy = entropy_map / e_max
            heatmap = 0.6 * heatmap + 0.4 * norm_entropy * heatmap.max()

    viewports = []
    obs_count = np.zeros((MAP_H, MAP_W), dtype=int)
    for q in range(n_queries):
        # Sterkere overlap-penalty etter de første 2 queries
        overlap_penalty = 0.20 if q < 2 else 0.50
        best_score, best_vp = -1, (0, 0)
        for vy in range(0, MAP_H - MAX_VIEWPORT + 1, 2):
            for vx in range(0, MAP_W - MAX_VIEWPORT + 1, 2):
                rh = heatmap[vy:vy+MAX_VIEWPORT, vx:vx+MAX_VIEWPORT]
                ro = obs_count[vy:vy+MAX_VIEWPORT, vx:vx+MAX_VIEWPORT]
                score = (rh / (1.0 + overlap_penalty * ro)).sum()
                if score > best_score:
                    best_score, best_vp = score, (vx, vy)
        vx, vy = best_vp
        viewports.append((vx, vy, MAX_VIEWPORT, MAX_VIEWPORT))
        obs_count[vy:vy+MAX_VIEWPORT, vx:vx+MAX_VIEWPORT] += 1
    return viewports


# === SOLVER ===

def solve_seed(client, round_id, seed_index, seed_data, transition_table,
               simple_prior, total_queries=10, alpha=DEFAULT_ALPHA):
    grid = seed_data.get("grid", [])
    settlements = seed_data.get("settlements", [])
    print(f"\n  Seed {seed_index}: {len(settlements)} settlements")
    observer = SeedObserver(grid, settlements, transition_table, simple_prior, alpha=alpha)
    viewports = plan_queries(grid, settlements, n_queries=total_queries)

    for i, (vx, vy, vw, vh) in enumerate(viewports):
        try:
            result = client.simulate(round_id, seed_index, vx, vy, vw, vh)
            gd = result.get("grid", [])
            if gd:
                observer.add_observation(gd, vx, vy)
                so = result.get("settlements", [])
                if so:
                    observer.add_settlement_obs(so)
            used = result.get("queries_used", "?")
            mx = result.get("queries_max", "?")
            print(f"    Q{i+1}: ({vx},{vy}) → {used}/{mx}")
            time.sleep(0.25)
        except Exception as e:
            print(f"    Q{i+1} FEIL: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"    {e.response.text[:150]}")

    dm = ~observer.static_mask
    do = (observer.observed[dm] > 0).sum()
    dt = dm.sum()
    mo = observer.observed[dm].mean() if dt > 0 else 0
    print(f"  Dynamiske: {do}/{dt} observert, snitt {mo:.1f} obs/celle")
    return observer


def solve_round(client, round_id, round_data, transition_table, simple_prior,
                queries_per_seed=10, submit=True, alpha=DEFAULT_ALPHA,
                type_tables=None, safety_submit=True, opt_tables=None):
    """
    Round-solving strategy v8 — soft regime inference.

    Faser:
    1. SAFETY: Soft-blended prior submit (balanced STABLE/BOOM default)
    2. PROBE: Diagnostic queries across up to 4 seeds → round fingerprint
    3. TYPE-AWARE: Rebuild observers with best-guess world type
    4. OBSERVE: Distribute remaining queries across all seeds
    5. CROSS-SEED: Continuous cross-seed learning
    6. RESUBMIT: Soft-blended prediction + global recalibration + light cross-seed

    Key improvements over v7:
    - Soft blending between DEAD/STABLE/BOOM (no hard classification cliff)
    - Multi-signal fingerprint (survival + ruin + empty + forest rates)
    - Diagnostic probes spread across 4 seeds instead of 2
    - Global recalibration instead of cell-level observation updates
    """
    seeds_data = round_data.get("seeds", round_data.get("initial_states", []))
    if not seeds_data:
        print(f"FEIL: Ingen seeds.")
        return []

    n_seeds = len(seeds_data)
    n_settlements_s0 = len(seeds_data[0].get("settlements", []))
    print(f"\n{n_seeds} seeds, {n_settlements_s0} initial settlements (alpha={alpha:.1f})")

    # === FASE 1: SAFETY SUBMIT med soft-blended prior (balanced default) ===
    prior_scores = []
    if safety_submit and submit:
        # Default: survival 0.33 = right at STABLE/BOOM boundary → balanced blend
        default_weights = compute_type_weights(0.33)
        dw_str = ", ".join(f"{k}:{v:.2f}" for k, v in default_weights.items() if v > 0.01)
        print(f"\n  FASE 1: Soft-blended safety submit ({{{dw_str}}})...")
        for si in range(n_seeds):
            sd = seeds_data[si]
            grid = sd.get("grid", [])
            settlements = sd.get("settlements", [])

            # Soft-blended prior across world types (no hard classification)
            pred = build_blended_prediction(grid, settlements, transition_table, simple_prior,
                                            opt_tables, default_weights)
            if pred is None:
                # Fallback: single-type prediction
                obs = SeedObserver(grid, settlements, transition_table, simple_prior,
                                 alpha=alpha, opt_tables=opt_tables, world_type="STABLE")
                pred = obs.build_prediction(apply_smoothing=False)

            try:
                resp = client.submit(round_id, si, pred.tolist())
                score = resp.get("score", resp.get("seed_score", "?"))
                prior_scores.append(score)
                print(f"    Seed {si} super-prior → {score}")
                time.sleep(0.4)
            except Exception as e:
                print(f"    Seed {si} prior submit FEIL: {e}")
                prior_scores.append(None)

    # === FASE 2: PROBE — Diagnostic queries across multiple seeds ===
    # Probing more seeds gives more settlement observations (hidden params are shared)
    n_probe_seeds = min(4, n_seeds)
    probe_queries_each = max(1, 4 // n_probe_seeds)  # ~4 total probe queries
    total_probe_queries = n_probe_seeds * probe_queries_each
    print(f"\n  FASE 2: Diagnostic probe ({n_probe_seeds} seeds × {probe_queries_each} query)...")
    probe_observers = []
    for si in range(n_probe_seeds):
        sd = seeds_data[si]
        grid = sd.get("grid", [])
        settlements = sd.get("settlements", [])
        obs = SeedObserver(grid, settlements, transition_table, simple_prior,
                         alpha=alpha, opt_tables=opt_tables, world_type="STABLE")

        # Target settlement-rich areas for maximum diagnostic value
        viewports = plan_queries(grid, settlements, n_queries=probe_queries_each)
        for i, (vx, vy, vw, vh) in enumerate(viewports):
            try:
                result = client.simulate(round_id, si, vx, vy, vw, vh)
                gd = result.get("grid", [])
                if gd:
                    obs.add_observation(gd, vx, vy)
                used = result.get("queries_used", "?")
                mx = result.get("queries_max", "?")
                print(f"    Seed {si} probe Q{i+1}: ({vx},{vy}) → {used}/{mx}")
                time.sleep(0.25)
            except Exception as e:
                print(f"    Seed {si} probe FEIL: {e}")
        probe_observers.append(obs)

    # Multi-signal fingerprint from all probed seeds
    fingerprint = compute_round_fingerprint(probe_observers)
    vitality = fingerprint_to_vitality(fingerprint)
    type_weights = compute_type_weights(vitality)

    # Hard classification for backward compatibility (SeedObserver world_type)
    world_type, n_sett = classify_world_type(seeds_data, vitality)
    opt_wtype = world_type
    if opt_wtype in ("BOOM_CONC", "BOOM_SPREAD"):
        opt_wtype = "BOOM"

    tw_str = ", ".join(f"{k}:{v:.2f}" for k, v in type_weights.items() if v > 0.01)
    print(f"\n  Fingerprint: survival={fingerprint['survival_rate']:.3f}, "
          f"ruin={fingerprint['ruin_rate']:.2f}, empty={fingerprint['empty_rate']:.2f} "
          f"({fingerprint['n_observed']} settlements observed)")
    print(f"  → vitality={vitality:.3f}, weights={{{tw_str}}}, hard_type={world_type}")

    # === FASE 3: TYPE-AWARE OBSERVERS ===
    # Velg type-spesifikk tabell for blending
    typed_table = None
    if type_tables:
        # Direkte match (4-type)
        if world_type in type_tables:
            typed_table = type_tables[world_type]
            print(f"  Bruker {world_type}-tabell ({len(typed_table)} keys)")
        # Fallback: map BOOM_SPREAD/BOOM_CONC → BOOMING (3-type)
        elif world_type.startswith("BOOM") and "BOOMING" in type_tables:
            typed_table = type_tables["BOOMING"]
            print(f"  Fallback: BOOMING-tabell ({len(typed_table)} keys)")

    # Rebuild alle observers med blended/typed priors
    # Probe-observers (seed 0-1) beholder sine observasjoner men oppdaterer priors
    observers = []

    # Bestem queries per seed — bruk ALLE queries
    total_budget = queries_per_seed * n_seeds
    queries_used = total_probe_queries  # probe queries brukt
    queries_left = total_budget - queries_used

    print(f"  Query-plan: {queries_left} queries fordelt på {n_seeds} seeds")

    # === FASE 4: OBSERVE ALLE SEEDS ===
    for si in range(n_seeds):
        sd = seeds_data[si]
        grid = sd.get("grid", [])
        settlements = sd.get("settlements", [])

        if si < len(probe_observers):
            # Probed seed: gjenbruk observer, rebuild priors med riktig world type
            obs = probe_observers[si]
            obs.opt_tables = opt_tables
            obs.world_type = opt_wtype
            obs._rebuild_priors()  # Rebuild med riktig type
        else:
            # Unprobed seed: ny observer med opt_tables + korrekt world type
            obs = SeedObserver(grid, settlements, transition_table, simple_prior,
                             alpha=alpha, opt_tables=opt_tables, world_type=opt_wtype)

        # Gi denne seeden sin andel av gjenstående queries
        seeds_remaining = n_seeds - si
        for_this_seed = queries_left // seeds_remaining
        if si < len(probe_observers):
            for_this_seed -= probe_queries_each  # allerede brukt probe queries
        queries_left -= (for_this_seed + (probe_queries_each if si < len(probe_observers) else 0))

        print(f"\n  Seed {si}: {len(settlements)} settlements, {for_this_seed} queries")

        # Planlegg og observer
        viewports = plan_queries(grid, settlements, n_queries=for_this_seed)
        for i, (vx, vy, vw, vh) in enumerate(viewports):
            try:
                result = client.simulate(round_id, si, vx, vy, vw, vh)
                gd = result.get("grid", [])
                if gd:
                    obs.add_observation(gd, vx, vy)
                used = result.get("queries_used", "?")
                mx = result.get("queries_max", "?")
                print(f"    Q{i+1}: ({vx},{vy}) → {used}/{mx}")
                time.sleep(0.25)
            except Exception as e:
                print(f"    Q{i+1} FEIL: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    status = e.response.status_code
                    if status == 429:
                        print(f"    Budget oppbrukt! Stopper queries.")
                        break

        observers.append(obs)

        # === FASE 5: KONTINUERLIG CROSS-SEED ===
        # Oppdater cross-seed etter HVER seed (ikke bare på slutten)
        if len(observers) >= 2:
            cross_table = build_cross_seed_prior(observers)
            if cross_table:
                n_keys = len(cross_table)
                total_obs_count = sum(v["sample_count"] for v in cross_table.values())
                apply_cross_seed(observers, cross_table, transition_table)
                if si == n_seeds - 1:  # Bare print på siste
                    print(f"\n  Cross-seed: {n_keys} kategorier, {total_obs_count} obs")

    # === FASE 6: RESUBMIT med soft-blended priors + global recalibration ===
    tw_str = ", ".join(f"{k}:{v:.2f}" for k, v in type_weights.items() if v > 0.01)
    print(f"\n  FASE 6: Soft-blended resubmit (vitality={vitality:.3f}, {{{tw_str}}})...")

    # Build cross-seed table for light final adjustment
    final_cross_table = build_cross_seed_prior(observers) if len(observers) >= 2 else {}

    results = []
    for si, obs in enumerate(observers):
        dm = ~obs.static_mask
        do = (obs.observed[dm] > 0).sum()
        dt = dm.sum()
        mo = obs.observed[dm].mean() if dt > 0 else 0

        sd = seeds_data[si]
        grid = sd.get("grid", [])
        settlements = sd.get("settlements", [])

        # Step 1: Soft-blended prior prediction (no hard type cliff)
        blended_pred = build_blended_prediction(
            grid, settlements, transition_table, simple_prior,
            opt_tables, type_weights)

        if blended_pred is not None:
            # Step 2: Global recalibration based on round fingerprint
            final_pred = recalibrate_pred(blended_pred, fingerprint, obs.static_mask)
            # Step 3: Light cross-seed adjustment (15% max, not 80%)
            if final_cross_table and do > 0:
                final_pred = apply_cross_seed_to_pred(
                    final_pred, grid, settlements, final_cross_table,
                    obs.static_mask, max_weight=0.15)
            reason = f"soft-blend+recal (v={vitality:.2f})"
        else:
            # Fallback: hard-typed prediction from observer
            final_pred = obs.build_prediction(apply_smoothing=False, world_type=opt_wtype)
            reason = "hard-typed fallback"

        print(f"  Seed {si}: {do}/{dt} obs, snitt {mo:.1f}/celle → {reason}")

        if not submit:
            results.append({"seed_index": si, "status": "no-submit"})
            continue

        try:
            resp = client.submit(round_id, si, final_pred.tolist())
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


def main():
    if not API_KEY:
        print("FEIL: export API_KEY='din-jwt-token'")
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=None)
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--no-submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--leaderboard", action="store_true")
    args = parser.parse_args()

    client = AstarClient()
    transition_table, simple_prior = load_calibration()
    type_tables = load_calibration_by_type()
    opt_tables = load_optimized_calibration()
    learning = load_learning_state()
    alpha = learning.get("alpha", DEFAULT_ALPHA)

    if args.leaderboard:
        try:
            lb = client.get_leaderboard()
            print("=== LEADERBOARD (topp 15) ===")
            for i, e in enumerate(lb[:15]):
                if isinstance(e, dict):
                    name = e.get("team_name", e.get("username", "?"))
                    sc = e.get("weighted_score", e.get("score", "?"))
                    st = e.get("hot_streak", "?")
                    rn = e.get("rounds_played", "?")
                    print(f"  #{i+1} {name}: {sc} (streak={st}, runder={rn})")
        except Exception as e:
            print(f"Leaderboard feil: {e}")
        return

    try:
        budget = client.get_budget()
        print(f"Budget: {budget.get('queries_used',0)}/{budget.get('queries_max',50)} brukt")
    except Exception:
        pass

    try:
        rounds = client.get_rounds()
    except Exception as e:
        print(f"FEIL: Kunne ikke hente runder: {e}")
        sys.exit(1)
    if not rounds:
        print("Ingen runder"); sys.exit(1)

    if args.round:
        round_id = args.round
    else:
        active = [r for r in rounds if isinstance(r, dict) and r.get("status") == "active"]
        if not active:
            print("Ingen aktive runder."); sys.exit(1)
        round_id = active[-1]["id"]

    try:
        round_data = client.get_round(round_id)
    except Exception as e:
        print(f"FEIL: Kunne ikke hente runde {round_id}: {e}")
        sys.exit(1)
    rnum = round_data.get("round_number", "?")
    weight = round_data.get("round_weight", "?")
    print(f"\nRunde {rnum} (vekt {weight}, alpha={alpha:.1f}): {round_id[:12]}...")
    print(f"Stenger: {round_data.get('closes_at', '?')}")

    if args.dry_run:
        print("\n[DRY RUN]")
        seeds = round_data.get("seeds", round_data.get("initial_states", []))
        for i, s in enumerate(seeds):
            print(f"  Seed {i}: {len(s.get('settlements',[]))} settlements")
        return

    results = solve_round(client, round_id, round_data, transition_table, simple_prior,
                         queries_per_seed=args.queries, submit=not args.no_submit, alpha=alpha,
                         type_tables=type_tables, safety_submit=True, opt_tables=opt_tables)

    print("\n=== RESULTATER ===")
    scores = []
    for r in results:
        s = r.get("score", r.get("error", r.get("status", "?")))
        print(f"  Seed {r['seed_index']}: {s}")
        if isinstance(r.get("score"), (int, float)):
            scores.append(r["score"])
    if scores:
        avg = sum(scores)/len(scores)
        print(f"\n  Snitt: {avg:.1f}")
        if isinstance(weight, (int, float)):
            print(f"  Weighted: {avg * weight:.1f}")

    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps({"round_id": round_id, "round_number": rnum,
        "results": results, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2, default=str))


if __name__ == "__main__":
    main()
