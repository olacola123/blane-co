"""
Joakim's Astar Island Solver — Diamond distance variant (based on Ola's v8)
=====================================================================
Uses Manhattan distance (|dx|+|dy|) throughout, with calibration tables
rebuilt from ground truth using Manhattan distance (calibrate_manhattan.py).

Key changes from v8:
- Manhattan distance for all settlement influence calculations
- Own calibration files built with Manhattan distance (no Chebyshev mismatch)
- Fixed query allocation bug: all 50 queries are now used (was 46)

Bruk:
    export API_KEY='din-jwt-token'
    python solution_diamond.py                    # Løs aktiv runde
    python solution_diamond.py --dry-run          # Vis plan uten API-kall
    python solution_diamond.py --no-submit        # Observer uten submit
    python solution_diamond.py --leaderboard      # Vis leaderboard
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astar_solver.history import RoundDatasetStore
from astar_solver.observations import RoundObservationStore
from astar_solver.tuning import extract_target_tensor
from astar_solver.types import SeedState

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
ANALYSIS_FETCH_TIMEOUT_SECONDS = 20.0
ANALYSIS_FETCH_POLL_SECONDS = 2.0
TRANSIENT_ANALYSIS_STATUS_CODES = {400, 404, 409, 425, 429}

TERRAIN_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}
# Manhattan distance bands — matching calibrate_manhattan.py
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

# Calibration files — prefer Manhattan-distance tables (joakim/), fallback to Ola's Chebyshev
JOAKIM_DIR = PROJECT_ROOT
OLA_DIR = Path(__file__).parent.parent / "ola"
HISTORY_ROOT = JOAKIM_DIR / "history"

# Manhattan-distance calibration (built by calibrate_manhattan.py)
CALIBRATION_FILE_MANHATTAN = JOAKIM_DIR / "calibration_manhattan.json"
CALIBRATION_4TYPE_FILE_MANHATTAN = JOAKIM_DIR / "calibration_manhattan_4type.json"
CALIBRATION_OPT_FILE_MANHATTAN = JOAKIM_DIR / "calibration_manhattan_opt.json"

# Fallback: Ola's Chebyshev-distance calibration
CALIBRATION_FILE = OLA_DIR / "calibration_data.json"
CALIBRATION_BY_TYPE_FILE = OLA_DIR / "calibration_by_type.json"
CALIBRATION_4TYPE_FILE = OLA_DIR / "calibration_4type.json"
CALIBRATION_OPT_FILE = OLA_DIR / "calibration_optimized.json"
SUPER_CALIBRATION_FILE = OLA_DIR / "super_calibration.json"
# Prefer Joakim's 17-round tables, fallback to Ola's 14-round tables
MODEL_TABLES_FILE_17R = JOAKIM_DIR / "model_tables_17r.json"
MODEL_TABLES_FILE_OLA = OLA_DIR / "model_tables.json"
LEARNING_FILE = OLA_DIR / "learning_state.json"

# Terrain code → terrain group for optimized tables
TERRAIN_GROUP = {0: "plains", 1: "settlement", 2: "port", 3: "ruin", 4: "forest", 5: "mountain", 10: "ocean", 11: "plains"}
# Manhattan optimized distance bands — matching calibrate_manhattan.py
OPT_DIST_BANDS = [(0,0), (1,2), (3,3), (4,5), (6,8), (9,12), (13,99)]

def get_opt_band(dist):
    for i, (lo, hi) in enumerate(OPT_DIST_BANDS):
        if lo <= dist <= hi:
            return i
    return len(OPT_DIST_BANDS) - 1


# === CALIBRATION + LEARNING STATE ===

def load_optimized_calibration():
    """Last optimerte prior-tabeller (wtype × terrain_group × dist_band × coastal).
    Prefer Manhattan-distance tables if available."""
    # Prefer Manhattan-distance calibration
    if CALIBRATION_OPT_FILE_MANHATTAN.exists():
        data = json.loads(CALIBRATION_OPT_FILE_MANHATTAN.read_text())
        tables = data.get("tables", {})
        if tables:
            print(f"Lastet Manhattan-optimerte tabeller: {len(tables)} entries "
                  f"fra {data.get('num_rounds', '?')} runder × {data.get('num_seeds', '?')} seeds")
            return tables

    # Fallback: Ola's Chebyshev tables (NOTE: distance mismatch with Manhattan solver)
    if not CALIBRATION_OPT_FILE.exists():
        return None
    data = json.loads(CALIBRATION_OPT_FILE.read_text())
    tables = data.get("tables", {})
    if tables:
        print(f"ADVARSEL: Bruker Chebyshev-tabeller (Manhattan-tabeller ikke bygget ennå)")
        print(f"  Kjør: python calibrate_manhattan.py")
        print(f"  Lastet {len(tables)} entries fra {data.get('num_rounds', '?')} runder × {data.get('num_seeds', '?')} seeds")
    return tables


# === SUPER-CALIBRATION ===
# Built from 14 rounds × 5 seeds = 70 GT datasets, 95k data points
# Features: vitality_bin × terrain_group × dist_bin × coastal × settle_density × forest_density

_model_cache = None

def load_model_tables():
    """Load model tables. Prefer 17-round Joakim tables, fallback to Ola's 14-round."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    # Prefer Joakim's 17-round tables
    for tables_file, label in [
        (MODEL_TABLES_FILE_17R, "Joakim 17r"),
        (MODEL_TABLES_FILE_OLA, "Ola 14r"),
    ]:
        if tables_file.exists():
            data = json.loads(tables_file.read_text())
            _model_cache = {
                "specific": data.get("table_specific", {}),
                "medium": data.get("table_medium", {}),
                "simple": data.get("table_simple", {}),
            }
            n_spec = len(_model_cache["specific"])
            n_med = len(_model_cache["medium"])
            n_simp = len(_model_cache["simple"])
            n_rounds = data.get("num_rounds", "?")
            print(f"Lastet model ({label}): {n_spec} specific + {n_med} medium + {n_simp} simple"
                  f" fra {n_rounds} runder")
            return _model_cache
    print("ADVARSEL: Ingen model_tables funnet")
    return None


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
    """Manhattan distance bins — matching calibrate_manhattan.py OPT_DIST_BANDS."""
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


def super_predict(grid, settlements, vbin, floor=None):
    """
    Build prediction using model v7 tables (14 rounds × 5 seeds = 95k data points).
    vbin: "DEAD", "LOW", "MED", "HIGH"
    Returns: (H, W, 6) numpy array

    Uses Manhattan distance to settlements with native Manhattan-calibrated tables.
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

            # Features — Manhattan distance
            min_dist = 99
            n_settle_r5 = 0
            for s in settlements:
                d = abs(y - s["y"]) + abs(x - s["x"])
                if d < min_dist:
                    min_dist = d
                if d <= 5:
                    n_settle_r5 += 1

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
            sdb = _settle_density_bin(n_settle_r5)
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
    Bins calibrated from 17 rounds × 5 seeds ground truth:
    DEAD: vitality < 0.08 (rounds 3, 8, 10)
    LOW:  vitality 0.08-0.20 (no rounds — fallback for inference noise)
    MED:  vitality 0.20-0.35 (rounds 4, 5, 9, 13, 15, 16)
    HIGH: vitality >= 0.35 (rounds 1, 2, 6, 7, 11, 12, 14, 17)
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
    # Prefer Manhattan-distance calibration
    if CALIBRATION_FILE_MANHATTAN.exists():
        data = json.loads(CALIBRATION_FILE_MANHATTAN.read_text())
        transition_table = data.get("transition_table", {})
        simple_prior = data.get("simple_prior", {})
        print(f"Lastet Manhattan-kalibrering fra {data.get('num_rounds', '?')} runder, {data.get('num_seeds', '?')} seeds")
        return transition_table, simple_prior

    if not CALIBRATION_FILE.exists():
        print("ADVARSEL: Ingen calibration — bruker fallback priors")
        print("  Kjør: python calibrate_manhattan.py")
        return None, None
    data = json.loads(CALIBRATION_FILE.read_text())
    transition_table = data.get("transition_table", {})
    simple_prior = data.get("simple_prior", {})
    print(f"ADVARSEL: Bruker Chebyshev-kalibrering (Manhattan ikke bygget)")
    return transition_table, simple_prior


def load_calibration_by_type():
    """Last type-spesifikke calibration-tabeller. Foretrekker Manhattan 4-type."""
    # Prefer Manhattan 4-type calibration
    if CALIBRATION_4TYPE_FILE_MANHATTAN.exists():
        data = json.loads(CALIBRATION_4TYPE_FILE_MANHATTAN.read_text())
        tables = data.get("tables", {})
        if tables:
            print(f"Lastet Manhattan 4-type: {', '.join(f'{k}({len(v)} keys)' for k, v in tables.items())}")
            return tables

    # Fallback: Ola's Chebyshev 4-type
    if CALIBRATION_4TYPE_FILE.exists():
        data = json.loads(CALIBRATION_4TYPE_FILE.read_text())
        tables = data.get("tables", {})
        if tables:
            print(f"Lastet Chebyshev 4-type (fallback): {', '.join(f'{k}({len(v)} keys)' for k, v in tables.items())}")
            return tables

    if not CALIBRATION_BY_TYPE_FILE.exists():
        return None
    data = json.loads(CALIBRATION_BY_TYPE_FILE.read_text())
    tables = data.get("tables", {})
    if tables:
        print(f"Lastet 3-type tabeller (fallback): {', '.join(f'{k}({len(v)} keys)' for k, v in tables.items())}")
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


def build_history_round_metadata(round_id, round_data, seed_count):
    """Trim round metadata before persisting it to the history bundle."""
    metadata = {
        "id": round_data.get("id", round_id),
        "round_number": round_data.get("round_number"),
        "round_weight": round_data.get("round_weight"),
        "status": round_data.get("status"),
        "event_date": round_data.get("event_date"),
        "started_at": round_data.get("started_at"),
        "closes_at": round_data.get("closes_at"),
        "seed_count": seed_count,
    }
    return {key: value for key, value in metadata.items() if value is not None}


def record_history_observation(observation_store, round_id, seed_index, result, vx, vy, vw, vh):
    """Persist one simulation response in the shared history observation store."""
    if "grid" not in result:
        return
    try:
        payload = dict(result)
        payload["viewport"] = {"x": int(vx), "y": int(vy), "w": int(vw), "h": int(vh)}
        observation_store.add_simulation_result(round_id, seed_index, payload)
    except Exception as exc:
        print(f"    History-logg FEIL for seed {seed_index}: {exc}")


def fetch_analyses_best_effort(
    client,
    round_id,
    seed_indices,
    timeout_seconds=ANALYSIS_FETCH_TIMEOUT_SECONDS,
    poll_interval_seconds=ANALYSIS_FETCH_POLL_SECONDS,
):
    """Poll the analysis endpoint briefly so live runs can store ground truth when ready."""
    pending = set(int(seed_index) for seed_index in seed_indices)
    analyses = {}
    if not pending:
        return analyses

    deadline = time.time() + timeout_seconds
    while pending and time.time() < deadline:
        for seed_index in list(pending):
            try:
                analyses[seed_index] = client.get_analysis(round_id, seed_index)
                pending.remove(seed_index)
                print(f"    Analysis seed {seed_index} hentet")
            except Exception as exc:
                response = getattr(exc, "response", None)
                status_code = getattr(response, "status_code", None)
                if status_code not in TRANSIENT_ANALYSIS_STATUS_CODES:
                    response_text = ""
                    if response is not None:
                        try:
                            response_text = response.text[:200]
                        except Exception:
                            response_text = "<unavailable>"
                    print(
                        f"    Analysis seed {seed_index} FEIL (status={status_code}): "
                        f"{exc} {response_text}"
                    )
                    pending.remove(seed_index)
        if pending and time.time() < deadline:
            time.sleep(poll_interval_seconds)

    if pending:
        waiters = ", ".join(str(seed_index) for seed_index in sorted(pending))
        print(f"    Analysis ikke klar ennå for seeds: {waiters}")
    return analyses


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
    def get_analysis(self, round_id, seed_index): return self.get(f"/analysis/{round_id}/{seed_index}")

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
    """Manhattan distance to nearest settlement."""
    if not settlements:
        return 99
    return min(abs(x - s["x"]) + abs(y - s["y"]) for s in settlements)

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

        # Build fallback chain: expansion-specific → merged BOOM → ALL
        fallback_types = [world_type]
        if world_type in ("BOOM_CONC", "BOOM_SPREAD"):
            fallback_types.append("BOOM")
        fallback_types.append("ALL")

        for wt in fallback_types:
            for key in [
                f"{wt}_{tg}_{opt_band}_{int(coastal)}",
                f"{wt}_{tg}_{opt_band}_any",
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
                    # Add ola dir to path for blending import
                    if str(OLA_DIR) not in sys.path:
                        sys.path.insert(0, str(OLA_DIR))
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
                    cross_weight = min(0.50, cross_n / 80.0)
                    blended = (1 - cross_weight) * hist_prior + cross_weight * cross_dist
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
    if str(OLA_DIR) not in sys.path:
        sys.path.insert(0, str(OLA_DIR))
    from blending import infer_vitality_continuous
    return infer_vitality_continuous(observers)


# === EXPANSION RANGE (replaces spatial coherence) ===
#
# Measures how far from initial settlement positions new activity has spread.
# "Urban" rounds: settlements stay close to start → low expansion_range
# "Rural" rounds: settlements scatter far from start → high expansion_range
#
# This captures a hidden simulation parameter that controls settlement expansion
# behavior, observed as localization vs dispersal in ground truth.

def infer_expansion_range(observers):
    """
    Measure settlement expansion range from probe observations.

    Computes the probability-weighted mean Manhattan distance of observed
    settlement/port activity from the nearest INITIAL settlement position.

    Returns: (expansion_range, evidence_strength)
    - expansion_range: float [0, 1] where 0=highly urban, 1=highly rural
    - evidence_strength: float [0, 1] based on how many active cells we observed

    Key insight: instead of measuring clustering (which is unreliable with few
    observations), we measure how FAR from start positions settlements have
    spread. This directly measures the hidden parameter we want to predict.
    """
    # Collect active cells with their distances from initial settlements
    active_data = []  # list of (distance_to_nearest_initial, activity_strength)
    total_observed_dynamic = 0
    n_at_initial = 0  # settlements still at starting positions

    for obs in observers:
        # Build distance map from initial settlements
        initial_positions = set()
        for s in obs.settlements:
            sy, sx = s.get("y", -1), s.get("x", -1)
            if 0 <= sy < MAP_H and 0 <= sx < MAP_W:
                initial_positions.add((sy, sx))

        for y in range(MAP_H):
            for x in range(MAP_W):
                if obs.static_mask[y, x] or obs.observed[y, x] == 0:
                    continue
                total_observed_dynamic += 1

                n = obs.observed[y, x]
                # Settlement + port fraction = "active settlement" signal
                active_frac = (obs.counts[y, x, 1] + obs.counts[y, x, 2]) / n

                if active_frac < 0.15:
                    continue  # not enough settlement signal

                # Manhattan distance to nearest initial settlement
                min_dist = 99
                for (iy, ix) in initial_positions:
                    d = abs(y - iy) + abs(x - ix)
                    if d < min_dist:
                        min_dist = d
                        if min_dist == 0:
                            break

                active_data.append((min_dist, active_frac))
                if min_dist == 0:
                    n_at_initial += 1

    n_active = len(active_data)

    # Evidence: need enough active cells to measure expansion
    if n_active < 3:
        return 0.5, 0.0  # Neutral — not enough data

    evidence = min(1.0, n_active / 12.0)

    # Compute probability-weighted mean distance from initial positions
    total_weight = sum(frac for _, frac in active_data)
    if total_weight < 0.01:
        return 0.5, 0.0

    weighted_mean_dist = sum(d * frac for d, frac in active_data) / total_weight

    # Also compute the fraction of activity that's far from initial (beyond dist 5)
    far_weight = sum(frac for d, frac in active_data if d > 5)
    far_fraction = far_weight / total_weight

    # And fraction right at initial positions (dist 0)
    initial_weight = sum(frac for d, frac in active_data if d == 0)
    initial_fraction = initial_weight / total_weight

    # Combine into a single expansion score [0, 1]
    # weighted_mean_dist: typical range 0-15 for urban, 5-25 for rural
    # Normalize: 0 at dist=0, 1 at dist≈15
    dist_score = min(1.0, weighted_mean_dist / 15.0)

    # far_fraction: 0 for urban, 0.3-0.7 for rural
    far_score = min(1.0, far_fraction / 0.5)

    # initial_fraction: high for urban, low for rural (inverted)
    initial_score = 1.0 - min(1.0, initial_fraction / 0.6)

    # Weighted combination
    raw_expansion = (
        0.45 * dist_score +
        0.30 * far_score +
        0.25 * initial_score
    )
    raw_expansion = max(0.0, min(1.0, raw_expansion))

    # Blend with neutral based on evidence
    expansion_range = evidence * raw_expansion + (1.0 - evidence) * 0.5

    return expansion_range, evidence


def classify_world_type(seeds_data, vitality, expansion_range=0.5):
    """
    4-type klassifisering: DEAD / STABLE / BOOM_SPREAD / BOOM_CONC.
    Bruker vitality + expansion_range (fra observasjoner) + n_settlements (gratis).

    expansion_range: 0=urban/concentrated, 1=rural/spread
    Low expansion → BOOM_CONC (settlements stay near start, concentrated growth)
    High expansion → BOOM_SPREAD (settlements scatter, diffuse growth)
    """
    n_settlements = len(seeds_data[0].get("settlements", []))

    if vitality < 0.20:
        return "DEAD", n_settlements
    elif vitality < 0.55:
        return "STABLE", n_settlements
    else:
        # expansion_range: low = concentrated (urban), high = spread (rural)
        if expansion_range < 0.35:
            return "BOOM_CONC", n_settlements
        elif expansion_range > 0.60:
            return "BOOM_SPREAD", n_settlements
        else:
            # Ambiguous → fall back to n_settlements heuristic
            if n_settlements >= 40:
                return "BOOM_CONC", n_settlements
            else:
                return "BOOM_SPREAD", n_settlements


def adjust_priors_for_vitality(observers, vitality):
    """DEPRECATED — blending.py handles this via get_blended_prior()."""
    pass


# === SOFT REGIME INFERENCE (v8) ===

def compute_type_weights(survival_rate):
    """Map survival rate to soft weights over DEAD/STABLE/BOOM world types.

    Optimized from grid search over 8 history_replay rounds × 17-round opt_tables:
    - dead ramp: 0.08-0.20 — matches DEAD table boundary (rounds 3,8,10: vit<0.08)
    - boom ramp: 0.25-0.30 — early boom detection, validated against live replay data
    - STABLE zone: 0.20-0.25 — narrow pure-STABLE zone
    """
    s = max(0.0, min(1.0, survival_rate))

    # Dead ramp: full dead below 0.08, linear to 0 at 0.20
    w_dead = max(0.0, min(1.0, (0.20 - s) / 0.12)) if s < 0.20 else 0.0
    # Boom ramp: starts at 0.25, full boom at 0.30
    w_boom = max(0.0, min(1.0, (s - 0.25) / 0.05)) if s > 0.25 else 0.0
    w_stable = max(0.0, 1.0 - w_dead - w_boom)

    total = w_dead + w_stable + w_boom
    if total < 1e-10:
        return {"DEAD": 0.0, "STABLE": 1.0, "BOOM": 0.0}

    return {
        "DEAD": w_dead / total,
        "STABLE": w_stable / total,
        "BOOM": w_boom / total,
    }


def split_boom_by_expansion(type_weights, expansion_range):
    """Split the BOOM weight into BOOM_CONC and BOOM_SPREAD based on expansion_range.

    expansion_range: 0=concentrated (near start), 1=spread (far from start)
    Returns new weight dict with BOOM_CONC and BOOM_SPREAD instead of BOOM.

    The splitting uses a sigmoid-like function centered at 0.45 (the midpoint
    between observed concentrated and spread rounds).
    """
    boom_w = type_weights.get("BOOM", 0.0)
    if boom_w < 0.01:
        return {k: v for k, v in type_weights.items() if k != "BOOM"}

    # Sigmoid-like split: smooth transition around center
    # expansion < 0.30 → mostly CONC, > 0.60 → mostly SPREAD
    center = 0.45
    width = 0.15  # controls transition sharpness
    t = max(0.0, min(1.0, (expansion_range - center + width) / (2 * width)))

    spread_frac = t
    conc_frac = 1.0 - t

    result = {}
    for k, v in type_weights.items():
        if k == "BOOM":
            result["BOOM_CONC"] = boom_w * conc_frac
            result["BOOM_SPREAD"] = boom_w * spread_frac
        else:
            result[k] = v
    return result


def compute_round_fingerprint(probe_observers):
    """Extract multi-signal round fingerprint from probe observations."""
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

            survived += cls_frac[1] + cls_frac[2]
            ports += cls_frac[2]
            ruins += cls_frac[3]
            empties += cls_frac[0]

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
            "survival_rate": 0.30,
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
    """Convert multi-signal fingerprint to vitality estimate."""
    vitality = fp["survival_rate"]

    if fp["n_observed"] >= 3:
        vitality -= (fp["ruin_rate"] - 0.12) * 0.08
        vitality -= (fp["empty_rate"] - 0.40) * 0.05

    return max(0.0, min(1.0, vitality))


def build_blended_prediction(grid, settlements, transition_table, simple_prior,
                              opt_tables, type_weights, expansion_range=0.5):
    """Build prior-only prediction with soft world-type blending.

    Splits BOOM weight into BOOM_CONC/BOOM_SPREAD based on expansion_range,
    using separate calibration tables for concentrated vs spread growth patterns.
    This replaces the old post-hoc expansion modulation approach.
    """
    # Split BOOM into expansion-specific types
    expanded_weights = split_boom_by_expansion(type_weights, expansion_range)

    preds = []
    weights = []
    for wtype, w in expanded_weights.items():
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

    sums = result.sum(axis=2, keepdims=True)
    result /= np.maximum(sums, 1e-10)

    return result


def _apply_expansion_modulation(pred, grid, settlements, expansion_range):
    """
    Modulate prediction based on expansion range (urbanization vs ruralization).

    Instead of boosting/reducing settlement probabilities directly (which the
    old coherence approach did, and which hurt scores), this shifts probability
    mass between distance bands by blending the prediction with a
    "distance-shifted" version.

    Low expansion (urban, <0.5): Nearby cells get MORE settlement probability
    (taken from mid-range cells). Settlements cluster tighter.

    High expansion (rural, >0.5): Mid/far cells get MORE settlement probability
    (taken from nearby cells). Settlements scatter wider.

    At 0.5: no modification.

    The key difference from the old approach: we redistribute probability
    between distance zones rather than blindly boosting/reducing, which
    preserves total settlement probability mass and avoids systematic bias.
    """
    deviation = expansion_range - 0.5  # [-0.5, 0.5]
    if abs(deviation) < 0.08:
        return pred  # Near neutral, skip

    grid_arr = np.array(grid, dtype=int) if not isinstance(grid, np.ndarray) else grid
    static_mask = (grid_arr == 10) | (grid_arr == 5)

    # Compute Manhattan distance to nearest initial settlement for each cell
    dist_map = np.full((MAP_H, MAP_W), 99.0)
    for s in settlements:
        sy, sx = s["y"], s["x"]
        for y in range(MAP_H):
            for x in range(MAP_W):
                d = abs(y - sy) + abs(x - sx)
                if d < dist_map[y, x]:
                    dist_map[y, x] = d

    # Modulation strength: gentle, capped at 0.08 max adjustment per cell
    strength = min(abs(deviation) * 0.20, 0.08)

    result = pred.copy()
    # Active classes affected by expansion
    active_classes = [1, 2, 3]  # settlement, port, ruin

    # Compute total active probability in near (0-3), mid (4-8), far (9+) zones
    # This lets us redistribute rather than create/destroy probability
    near_mask = (dist_map <= 3) & (~static_mask)
    mid_mask = (dist_map > 3) & (dist_map <= 8) & (~static_mask)
    far_mask = (dist_map > 8) & (dist_map <= 20) & (~static_mask)

    for c in active_classes:
        near_total = result[:, :, c][near_mask].sum()
        mid_total = result[:, :, c][mid_mask].sum()
        far_total = result[:, :, c][far_mask].sum()
        zone_total = near_total + mid_total + far_total
        if zone_total < 0.01:
            continue

        if deviation < 0:
            # URBAN: transfer probability from mid/far → near
            # Near cells get boosted, mid cells get reduced
            transfer = strength * mid_total  # amount to move

            if near_mask.sum() > 0 and mid_mask.sum() > 0:
                near_boost = transfer / near_mask.sum()
                mid_reduction = transfer / mid_mask.sum()

                result[:, :, c][near_mask] += near_boost
                result[:, :, c][mid_mask] -= mid_reduction
                # Also reduce far slightly
                if far_mask.sum() > 0:
                    far_reduction = strength * 0.5 * far_total / far_mask.sum()
                    result[:, :, c][far_mask] -= far_reduction
                    result[:, :, c][near_mask] += (strength * 0.5 * far_total) / near_mask.sum()
        else:
            # RURAL: transfer probability from near → mid/far
            transfer = strength * near_total

            if near_mask.sum() > 0 and mid_mask.sum() > 0:
                near_reduction = transfer / near_mask.sum()
                mid_boost = transfer * 0.7 / mid_mask.sum()

                result[:, :, c][near_mask] -= near_reduction
                result[:, :, c][mid_mask] += mid_boost
                if far_mask.sum() > 0:
                    far_boost = transfer * 0.3 / far_mask.sum()
                    result[:, :, c][far_mask] += far_boost

    # Re-floor and renormalize all dynamic cells
    dynamic_mask = ~static_mask
    result[dynamic_mask] = np.maximum(result[dynamic_mask], PROB_FLOOR)
    sums = result.sum(axis=2, keepdims=True)
    result /= np.maximum(sums, 1e-10)

    return result


def recalibrate_pred(pred, fingerprint, static_mask):
    """Global prior recalibration based on round fingerprint."""
    if fingerprint["n_observed"] < 3:
        return pred

    survival = fingerprint["survival_rate"]
    ruin_rate = fingerprint["ruin_rate"]

    expected_survival = 0.33
    expected_ruin = 0.12

    survival_dev = survival - expected_survival
    ruin_dev = ruin_rate - expected_ruin

    strength = 0.25  # Optimized (was 0.12)

    adjustments = np.ones(NUM_CLASSES)
    adjustments[0] -= survival_dev * strength
    adjustments[1] += survival_dev * strength
    adjustments[2] += survival_dev * strength * 0.5
    adjustments[3] -= survival_dev * strength
    adjustments[4] -= survival_dev * strength * 0.3
    adjustments[3] += ruin_dev * strength * 0.5
    adjustments[1] -= ruin_dev * strength * 0.3

    adjustments = np.clip(adjustments, 0.85, 1.15)

    result = pred.copy()
    dynamic = ~static_mask
    result[dynamic] *= adjustments
    result[dynamic] = np.maximum(result[dynamic], PROB_FLOOR)
    sums = result.sum(axis=2, keepdims=True)
    result /= np.maximum(sums, 1e-10)

    return result


def scale_for_vitality(pred, vitality, static_mask):
    """Scale settlement/port probabilities based on continuous vitality.

    The BOOM calibration tables encode average patterns (~0.50 vitality center).
    This function applies proportional scaling for rounds with extreme vitality,
    so strong boom rounds (vit>0.50) get more settlement probability.

    Only activates above thresh=0.50 to avoid disturbing DEAD/STABLE predictions.
    Optimized via grid search over 8 history_replay rounds:
      center=0.50, strength=2.0, thresh=0.50 → +0.6 avg, +7.0 on R18.
    """
    if vitality is None or vitality <= 0.50:
        return pred

    center = 0.50
    strength = 2.0
    deviation = vitality - center
    scale = 1.0 + deviation * strength
    scale = max(0.5, min(2.5, scale))

    result = pred.copy()
    dyn = ~static_mask
    result[dyn, 1] *= scale
    result[dyn, 2] *= scale
    result[dyn] = np.maximum(result[dyn], PROB_FLOOR)
    sums = result.sum(axis=2, keepdims=True)
    result /= np.maximum(sums, 1e-10)

    return result


def apply_cross_seed_to_pred(pred, grid, settlements, cross_table, static_mask,
                              max_weight=0.15):
    """Apply light cross-seed signal to a prediction."""
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
                w = max_weight * min(1.0, cross_n / 30.0)
                result[y, x] = (1 - w) * result[y, x] + w * cross_dist
                np.maximum(result[y, x], PROB_FLOOR, out=result[y, x])
                result[y, x] /= result[y, x].sum()

    return result


# === QUERY PLANNER ===

def build_dynamism_heatmap(initial_grid, settlements):
    """Manhattan-based dynamism heatmap for query planning."""
    grid = np.array(initial_grid, dtype=int)
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
    heatmap[grid == 10] = 0.0
    heatmap[grid == 5] = 0.0
    return heatmap


def plan_queries(initial_grid, settlements, n_queries=10, entropy_map=None,
                 diagnostic=False):
    """
    Plan viewport placements med hybrid settlement + entropy heuristikk.

    If diagnostic=True (for early probe queries):
    - First 2-4 queries use higher overlap penalty to spread geographically
    - This helps infer both vitality AND spatial coherence
    - Spreads across different settlement regions

    Otherwise (default): standard settlement + entropy heatmap.
    """
    heatmap = build_dynamism_heatmap(initial_grid, settlements)

    if entropy_map is not None:
        e_max = entropy_map.max()
        if e_max > 0:
            norm_entropy = entropy_map / e_max
            heatmap = 0.6 * heatmap + 0.4 * norm_entropy * heatmap.max()

    viewports = []
    obs_count = np.zeros((MAP_H, MAP_W), dtype=int)

    # For diagnostic queries, determine how many to spread aggressively
    n_diagnostic = min(n_queries, 3) if diagnostic else 0

    for q in range(n_queries):
        if q < n_diagnostic:
            # Diagnostic: high overlap penalty to maximize geographic spread
            # This ensures we sample from multiple regions for coherence inference
            overlap_penalty = 1.5
        elif q < 2:
            overlap_penalty = 0.20
        else:
            overlap_penalty = 0.50

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
    Round-solving strategy — diamond distance variant of v8.
    """
    seeds_data = round_data.get("seeds", round_data.get("initial_states", []))
    if not seeds_data:
        print(f"FEIL: Ingen seeds.")
        return []

    seed_states = [
        SeedState.from_round_data(seed_index, seed_data)
        for seed_index, seed_data in enumerate(seeds_data)
    ]
    history_store = RoundDatasetStore(HISTORY_ROOT)
    observation_store = RoundObservationStore(seed_states)
    safety_responses: dict[int, dict[str, Any]] = {}
    submission_responses: dict[int, dict[str, Any]] = {}
    predictions: dict[int, np.ndarray] = {}
    analyses: dict[int, dict[str, Any]] = {}
    prediction_diagnostics: dict[str, dict[str, Any]] = {}

    n_seeds = len(seeds_data)
    n_settlements_s0 = len(seeds_data[0].get("settlements", []))
    print(f"\n{n_seeds} seeds, {n_settlements_s0} initial settlements (alpha={alpha:.1f})")
    print(f"  [Manhattan distance + native Manhattan calibration tables]")

    # === FASE 1: SAFETY SUBMIT med soft-blended prior (balanced default) ===
    prior_scores = []
    if safety_submit and submit:
        # Conservative safety weights using old ramp (before optimization)
        # Old formula at vitality=0.33: 50% STABLE, 50% BOOM — good default
        _s = 0.33
        _wd = max(0.0, min(1.0, (0.15 - _s) / 0.10)) if _s < 0.15 else 0.0
        _wb = max(0.0, min(1.0, (_s - 0.28) / 0.10)) if _s > 0.28 else 0.0
        _ws = max(0.0, 1.0 - _wd - _wb)
        _t = _wd + _ws + _wb
        default_weights = {"DEAD": _wd/_t, "STABLE": _ws/_t, "BOOM": _wb/_t}
        dw_str = ", ".join(f"{k}:{v:.2f}" for k, v in default_weights.items() if v > 0.01)
        print(f"\n  FASE 1: Soft-blended safety submit ({{{dw_str}}})...")
        for si in range(n_seeds):
            sd = seeds_data[si]
            grid = sd.get("grid", [])
            settlements = sd.get("settlements", [])

            pred = build_blended_prediction(grid, settlements, transition_table, simple_prior,
                                            opt_tables, default_weights)
            if pred is None:
                obs = SeedObserver(grid, settlements, transition_table, simple_prior,
                                 alpha=alpha, opt_tables=opt_tables, world_type="STABLE")
                pred = obs.build_prediction(apply_smoothing=False)

            try:
                resp = client.submit(round_id, si, pred.tolist())
                score = resp.get("score", resp.get("seed_score", "?"))
                prior_scores.append(score)
                safety_responses[si] = dict(resp)
                print(f"    Seed {si} super-prior → {score}")
                time.sleep(0.4)
            except Exception as e:
                print(f"    Seed {si} prior submit FEIL: {e}")
                prior_scores.append(None)

    # === FASE 2: PROBE — Diagnostic queries across multiple seeds ===
    n_probe_seeds = min(4, n_seeds)
    probe_queries_each = max(1, 4 // n_probe_seeds)
    total_probe_queries = n_probe_seeds * probe_queries_each
    print(f"\n  FASE 2: Diagnostic probe ({n_probe_seeds} seeds × {probe_queries_each} query)...")
    probe_observers = []
    for si in range(n_probe_seeds):
        sd = seeds_data[si]
        grid = sd.get("grid", [])
        settlements = sd.get("settlements", [])
        obs = SeedObserver(grid, settlements, transition_table, simple_prior,
                         alpha=alpha, opt_tables=opt_tables, world_type="STABLE")

        # Use diagnostic=True for probe queries to spread geographically
        viewports = plan_queries(grid, settlements, n_queries=probe_queries_each,
                                 diagnostic=True)
        for i, (vx, vy, vw, vh) in enumerate(viewports):
            try:
                result = client.simulate(round_id, si, vx, vy, vw, vh)
                gd = result.get("grid", [])
                if gd:
                    obs.add_observation(gd, vx, vy)
                    record_history_observation(observation_store, round_id, si, result, vx, vy, vw, vh)
                used = result.get("queries_used", "?")
                mx = result.get("queries_max", "?")
                print(f"    Seed {si} probe Q{i+1}: ({vx},{vy}) → {used}/{mx}")
                time.sleep(0.25)
            except Exception as e:
                print(f"    Seed {si} probe FEIL: {e}")
        probe_observers.append(obs)

    fingerprint = compute_round_fingerprint(probe_observers)
    vitality = fingerprint_to_vitality(fingerprint)
    type_weights = compute_type_weights(vitality)

    # Infer expansion range from probe observations
    expansion_range, expansion_evidence = infer_expansion_range(probe_observers)

    world_type, n_sett = classify_world_type(seeds_data, vitality, expansion_range)
    # Use expansion-specific type directly (BOOM_CONC/BOOM_SPREAD stay separate)
    opt_wtype = world_type

    # For expansion-split blending weights
    expanded_weights = split_boom_by_expansion(type_weights, expansion_range)
    tw_str = ", ".join(f"{k}:{v:.2f}" for k, v in type_weights.items() if v > 0.01)
    ew_str = ", ".join(f"{k}:{v:.2f}" for k, v in expanded_weights.items() if v > 0.01)
    print(f"\n  Fingerprint: survival={fingerprint['survival_rate']:.3f}, "
          f"ruin={fingerprint['ruin_rate']:.2f}, empty={fingerprint['empty_rate']:.2f} "
          f"({fingerprint['n_observed']} settlements observed)")
    print(f"  → vitality={vitality:.3f}, expansion={expansion_range:.3f} "
          f"(evidence={expansion_evidence:.2f})")
    print(f"  → base_weights={{{tw_str}}}, expanded={{{ew_str}}}")
    print(f"  → hard_type={world_type}")

    # === FASE 3: TYPE-AWARE OBSERVERS ===
    typed_table = None
    if type_tables:
        if world_type in type_tables:
            typed_table = type_tables[world_type]
            print(f"  Bruker {world_type}-tabell ({len(typed_table)} keys)")
        elif world_type.startswith("BOOM") and "BOOMING" in type_tables:
            typed_table = type_tables["BOOMING"]
            print(f"  Fallback: BOOMING-tabell ({len(typed_table)} keys)")

    observers = []

    total_budget = queries_per_seed * n_seeds
    queries_used = total_probe_queries
    queries_left = total_budget - queries_used

    # Pre-compute per-seed query allocation (fixes integer division remainder bug)
    seed_new_queries = []
    _ql = queries_left
    for si in range(n_seeds):
        seeds_remaining = n_seeds - si
        for_this_seed = _ql // seeds_remaining
        probe_already = probe_queries_each if si < n_probe_seeds else 0
        new_q = max(0, for_this_seed - probe_already)
        seed_new_queries.append(new_q)
        _ql -= (new_q + probe_already)
    # Distribute any remainder queries (from integer division) to seeds in order
    used_total = sum(seed_new_queries) + total_probe_queries
    remainder = total_budget - used_total
    for i in range(remainder):
        seed_new_queries[i % n_seeds] += 1

    total_planned = sum(seed_new_queries) + total_probe_queries
    print(f"  Query-plan: {total_planned}/{total_budget} queries fordelt på {n_seeds} seeds")

    # === FASE 4: OBSERVE ALLE SEEDS ===
    for si in range(n_seeds):
        sd = seeds_data[si]
        grid = sd.get("grid", [])
        settlements = sd.get("settlements", [])

        if si < len(probe_observers):
            obs = probe_observers[si]
            obs.opt_tables = opt_tables
            obs.world_type = opt_wtype
            obs._rebuild_priors()
        else:
            obs = SeedObserver(grid, settlements, transition_table, simple_prior,
                             alpha=alpha, opt_tables=opt_tables, world_type=opt_wtype)

        for_this_seed = seed_new_queries[si]

        print(f"\n  Seed {si}: {len(settlements)} settlements, {for_this_seed} queries")

        viewports = plan_queries(grid, settlements, n_queries=for_this_seed)
        for i, (vx, vy, vw, vh) in enumerate(viewports):
            try:
                result = client.simulate(round_id, si, vx, vy, vw, vh)
                gd = result.get("grid", [])
                if gd:
                    obs.add_observation(gd, vx, vy)
                    record_history_observation(observation_store, round_id, si, result, vx, vy, vw, vh)
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
        if len(observers) >= 2:
            cross_table = build_cross_seed_prior(observers)
            if cross_table:
                n_keys = len(cross_table)
                total_obs_count = sum(v["sample_count"] for v in cross_table.values())
                apply_cross_seed(observers, cross_table, transition_table)
                if si == n_seeds - 1:
                    print(f"\n  Cross-seed: {n_keys} kategorier, {total_obs_count} obs")

    # === FASE 6: RESUBMIT med soft-blended priors + expansion modulation ===
    tw_str = ", ".join(f"{k}:{v:.2f}" for k, v in type_weights.items() if v > 0.01)
    print(f"\n  FASE 6: Soft-blended resubmit (v={vitality:.3f}, exp={expansion_range:.3f}, {{{tw_str}}})...")

    # Re-infer expansion with all observations (more data now)
    final_expansion, final_evidence = infer_expansion_range(observers)
    if final_evidence > expansion_evidence:
        expansion_range = final_expansion
        expansion_evidence = final_evidence
        print(f"  Updated expansion={expansion_range:.3f} (evidence={expansion_evidence:.2f})")

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

        blended_pred = build_blended_prediction(
            grid, settlements, transition_table, simple_prior,
            opt_tables, type_weights, expansion_range=expansion_range)

        if blended_pred is not None:
            final_pred = recalibrate_pred(blended_pred, fingerprint, obs.static_mask)
            final_pred = scale_for_vitality(final_pred, vitality, obs.static_mask)
            if final_cross_table and do > 0:
                final_pred = apply_cross_seed_to_pred(
                    final_pred, grid, settlements, final_cross_table,
                    obs.static_mask, max_weight=0.30)  # Optimized (was 0.15)
            reason = f"diamond+blend+exp={expansion_range:.2f} (v={vitality:.2f})"
        else:
            final_pred = obs.build_prediction(apply_smoothing=False, world_type=opt_wtype)
            reason = "hard-typed fallback"

        predictions[si] = final_pred
        prediction_diagnostics[str(si)] = {
            "reason": reason,
            "observed_dynamic_cells": int(do),
            "dynamic_cells": int(dt),
            "mean_observations_per_dynamic_cell": float(mo),
            "query_count": int(sum(1 for query in observation_store.observations if query.seed_index == si)),
        }
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
            submission_record = dict(resp)
            submission_record["submission_phase"] = "final"
            if si in safety_responses:
                submission_record["safety_response"] = safety_responses[si]
                if isinstance(prior_s, (int, float)):
                    submission_record["safety_score"] = prior_s
            submission_responses[si] = submission_record
            results.append({"seed_index": si, "score": score, "prior_score": prior_s})
            time.sleep(0.4)
        except Exception as e:
            print(f"    submit FEIL: {e}")
            results.append({"seed_index": si, "error": str(e)})

    submitted_seed_indices = sorted(submission_responses)
    if submit and submitted_seed_indices:
        print("\n  FASE 7: Best-effort analyse-fetch...")
        analyses = fetch_analyses_best_effort(client, round_id, submitted_seed_indices)

    ground_truth = {
        seed_index: target
        for seed_index, payload in analyses.items()
        if (target := extract_target_tensor(payload)) is not None
    }

    try:
        history_dir = history_store.save_round(
            round_id=round_id,
            round_metadata=build_history_round_metadata(round_id, round_data, n_seeds),
            seed_states=seed_states,
            observation_store=observation_store,
            predictions=predictions,
            submission_responses=submission_responses,
            analyses=analyses,
            ground_truth=ground_truth,
            config={
                "solver": Path(__file__).name,
                "variant": "diamond",
                "alpha": alpha,
                "queries_per_seed": queries_per_seed,
                "planned_total_queries": queries_per_seed * n_seeds,
                "submit": bool(submit),
                "safety_submit": bool(safety_submit and submit),
                "analysis_fetch_timeout_seconds": ANALYSIS_FETCH_TIMEOUT_SECONDS if submit else 0.0,
            },
            diagnostics={
                "query": {
                    "planned_total_queries": queries_per_seed * n_seeds,
                    "probe_seed_count": n_probe_seeds,
                    "probe_queries_each": probe_queries_each,
                    "probe_queries_total": total_probe_queries,
                    "seed_new_queries": seed_new_queries,
                    "recorded_queries": len(observation_store.observations),
                    "observation_summary": asdict(observation_store.build_summary()),
                },
                "inference": {
                    "fingerprint": fingerprint,
                    "vitality": vitality,
                    "type_weights": type_weights,
                    "hard_world_type": world_type,
                    "opt_world_type": opt_wtype,
                    "expansion_range": expansion_range,
                    "expansion_evidence": expansion_evidence,
                    "final_cross_table_size": len(final_cross_table),
                },
                "prediction": prediction_diagnostics,
                "submission": {
                    "safety_scores": {
                        str(seed_index): score
                        for seed_index, score in enumerate(prior_scores)
                        if isinstance(score, (int, float))
                    },
                    "safety_responses": {str(seed_index): payload for seed_index, payload in safety_responses.items()},
                    "final_results": results,
                    "submitted_seed_indices": submitted_seed_indices,
                    "analyses_fetched": sorted(analyses),
                    "ground_truth_saved": sorted(ground_truth),
                },
            },
        )
        print(f"\n  History lagret: {history_dir}")
    except Exception as e:
        print(f"\n  ADVARSEL: Kunne ikke lagre history: {e}")

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

    out = Path(__file__).parent / "results_diamond.json"
    out.write_text(json.dumps({"round_id": round_id, "round_number": rnum,
        "results": results, "history_dir": str(HISTORY_ROOT / round_id),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2, default=str))


if __name__ == "__main__":
    main()
