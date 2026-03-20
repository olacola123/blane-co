"""
Ola's Astar Island Solver v5 — Selvforbedrende
=================================================
Forbedringer over v4:
1. Settlement-attributter (food/pop/wealth) som prediktor
2. Adaptiv alpha (lærer optimal verdi fra forrige runder)
3. Spatial smoothing (naboceller informerer hverandre)
4. Rundetype-matching mot historikk

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
PROB_FLOOR = 0.003      # standard floor for mulige klasser
NEAR_ZERO = 0.0001     # floor for umulige klasser (mountain på slette, port innland)
DEFAULT_ALPHA = 15.0

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
    "4":  [0.079, 0.127, 0.009, 0.013, 0.772, 0.0001],
    "5":  [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.9996],
}

CALIBRATION_FILE = Path(__file__).parent / "calibration_data.json"
LEARNING_FILE = Path(__file__).parent / "learning_state.json"


# === CALIBRATION + LEARNING STATE ===

def load_calibration():
    if not CALIBRATION_FILE.exists():
        print("ADVARSEL: Ingen calibration_data.json — bruker fallback priors")
        return None, None
    data = json.loads(CALIBRATION_FILE.read_text())
    transition_table = data.get("transition_table", {})
    simple_prior = data.get("simple_prior", {})
    print(f"Lastet kalibrering fra {data.get('num_rounds', '?')} runder, {data.get('num_seeds', '?')} seeds")
    return transition_table, simple_prior


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
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        })

    def get(self, path, params=None):
        r = self.session.get(f"{BASE_URL}/{path.lstrip('/')}", params=params)
        r.raise_for_status()
        return r.json()

    def post(self, path, data):
        r = self.session.post(f"{BASE_URL}/{path.lstrip('/')}", json=data)
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

def get_prior(terrain, band, coastal, transition_table, simple_prior):
    if transition_table:
        key = f"{terrain}_{band}_{int(coastal)}"
        if key in transition_table:
            return np.array(transition_table[key]["distribution"], dtype=float)
        key_nc = f"{terrain}_{band}_0"
        if key_nc in transition_table:
            return np.array(transition_table[key_nc]["distribution"], dtype=float)
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

    # Floor
    smoothed = np.maximum(smoothed, PROB_FLOOR)
    smoothed /= smoothed.sum(axis=2, keepdims=True)

    return smoothed


# === OBSERVATION STORE ===

class SeedObserver:
    def __init__(self, initial_grid, settlements, transition_table, simple_prior, alpha=DEFAULT_ALPHA):
        self.grid = np.array(initial_grid, dtype=int)
        self.settlements = settlements
        self.transition_table = transition_table
        self.simple_prior = simple_prior
        self.alpha = alpha

        self.counts = np.zeros((MAP_H, MAP_W, NUM_CLASSES), dtype=float)
        self.observed = np.zeros((MAP_H, MAP_W), dtype=int)

        self.ocean_mask = (self.grid == 10)
        self.mountain_mask = (self.grid == 5)
        self.static_mask = self.ocean_mask | self.mountain_mask

        # BUG 1 FIX A: Preberegn coastal_mask (i stedet for per-celle is_coastal())
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

        # Forhåndsberegn prior + klasse-spesifikk floor per celle
        self._prior_cache = np.zeros((MAP_H, MAP_W, NUM_CLASSES), dtype=float)
        self._floor_cache = np.full((MAP_H, MAP_W, NUM_CLASSES), PROB_FLOOR, dtype=float)
        grid_list = self.grid.tolist()
        for y in range(MAP_H):
            for x in range(MAP_W):
                if self.static_mask[y, x]:
                    continue
                terrain, band, coastal = cell_key(grid_list, y, x, settlements)
                self._prior_cache[y, x] = get_prior(
                    terrain, band, coastal, transition_table, simple_prior
                )
                # BUG 1 FIX B: Klasse-spesifikke floors
                # Mountain umulig på ikke-fjell-celler
                self._floor_cache[y, x, 5] = NEAR_ZERO
                # Port umulig på innlandsceller
                if not self.coastal_mask[y, x]:
                    self._floor_cache[y, x, 2] = NEAR_ZERO

    def add_observation(self, grid_data, viewport_x, viewport_y):
        for dy, row in enumerate(grid_data):
            for dx, val in enumerate(row):
                y, x = viewport_y + dy, viewport_x + dx
                if 0 <= y < MAP_H and 0 <= x < MAP_W:
                    cls = TERRAIN_TO_CLASS.get(val, 0)
                    self.counts[y, x, cls] += 1
                    self.observed[y, x] += 1

    def add_settlement_obs(self, settlements_data):
        """[FORBEDRING 1] Bruk settlement-attributter som signal."""
        for s in settlements_data:
            x, y = s.get("x", -1), s.get("y", -1)
            if not (0 <= x < MAP_W and 0 <= y < MAP_H):
                continue

            signal, confidence = settlement_survival_signal(s)

            # Blend settlement-signal inn i prior (ikke obs counts)
            current_prior = self._prior_cache[y, x]
            blended = (1 - confidence * 0.3) * current_prior + confidence * 0.3 * signal
            np.maximum(blended, self._floor_cache[y, x], out=blended)
            blended /= blended.sum()
            self._prior_cache[y, x] = blended

    def build_prediction(self, apply_smoothing=True):
        """Bayesiansk posterior med adaptiv alpha + spatial smoothing."""
        pred = np.zeros((MAP_H, MAP_W, NUM_CLASSES), dtype=float)
        alpha = self.alpha

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
                    # [FORBEDRING 2] Adaptiv alpha fra learning state
                    if n_obs >= 10:
                        a = alpha * 0.33
                    elif n_obs >= 5:
                        a = alpha * 0.53
                    elif n_obs >= 3:
                        a = alpha * 0.80
                    else:
                        a = alpha

                    pred[y, x] = self.counts[y, x] + a * prior

                # BUG 1+4 FIX: Normaliser FØRST, klasse-spesifikk floor ETTER
                s = pred[y, x].sum()
                if s > 0:
                    pred[y, x] /= s
                np.maximum(pred[y, x], self._floor_cache[y, x], out=pred[y, x])
                pred[y, x] /= pred[y, x].sum()

        # [FORBEDRING 3] Spatial smoothing
        if apply_smoothing:
            pred = spatial_smooth(pred, self.static_mask)

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
                    cross_weight = min(0.5, cross_n / 100.0)
                    blended = (1 - cross_weight) * hist_prior + cross_weight * cross_dist
                    # BUG 3 FIX: Klasse-spesifikk floor (ikke uniform)
                    np.maximum(blended, obs._floor_cache[y, x], out=blended)
                    blended /= blended.sum()
                    obs._prior_cache[y, x] = blended


# === [FORBEDRING 4] RUNDETYPE-MATCHING ===

def infer_round_type(observers):
    """Analyser observasjoner for å gjette rundetype."""
    total_counts = np.zeros(NUM_CLASSES, dtype=float)
    total_cells = 0
    for obs in observers:
        for y in range(MAP_H):
            for x in range(MAP_W):
                if obs.static_mask[y, x] or obs.observed[y, x] == 0:
                    continue
                total_counts += obs.counts[y, x]
                total_cells += obs.observed[y, x]

    if total_cells < 50:
        return {"settlement": 1.0, "forest": 1.0, "ruin": 1.0, "port": 1.0}

    fracs = total_counts / total_cells
    hist = {"settlement": 0.12, "forest": 0.13, "ruin": 0.015, "port": 0.01}

    def clamp(v): return max(0.5, min(2.0, v))

    factors = {
        "settlement": clamp(fracs[1] / hist["settlement"]) if hist["settlement"] > 0 else 1.0,
        "forest": clamp(fracs[4] / hist["forest"]) if hist["forest"] > 0 else 1.0,
        "ruin": clamp(fracs[3] / hist["ruin"]) if hist["ruin"] > 0 else 1.0,
        "port": clamp(fracs[2] / hist["port"]) if hist["port"] > 0 else 1.0,
    }
    return factors


def adjust_priors_for_round(observers, factors):
    """Juster priors basert på inferert rundetype."""
    class_factors = np.array([1.0, factors["settlement"], factors["port"],
                              factors["ruin"], factors["forest"], 1.0])
    for obs in observers:
        for y in range(MAP_H):
            for x in range(MAP_W):
                if obs.static_mask[y, x]:
                    continue
                prior = obs._prior_cache[y, x].copy()
                prior *= class_factors
                prior = np.maximum(prior, PROB_FLOOR)
                prior /= prior.sum()
                obs._prior_cache[y, x] = prior


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


def plan_queries(initial_grid, settlements, n_queries=10):
    heatmap = build_dynamism_heatmap(initial_grid, settlements)
    viewports = []
    obs_count = np.zeros((MAP_H, MAP_W), dtype=int)
    for _ in range(n_queries):
        best_score, best_vp = -1, (0, 0)
        for vy in range(0, MAP_H - MAX_VIEWPORT + 1, 2):
            for vx in range(0, MAP_W - MAX_VIEWPORT + 1, 2):
                rh = heatmap[vy:vy+MAX_VIEWPORT, vx:vx+MAX_VIEWPORT]
                ro = obs_count[vy:vy+MAX_VIEWPORT, vx:vx+MAX_VIEWPORT]
                score = (rh / (1.0 + 0.35 * ro)).sum()
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
                queries_per_seed=10, submit=True, alpha=DEFAULT_ALPHA):
    seeds_data = round_data.get("seeds", round_data.get("initial_states", []))
    if not seeds_data:
        print(f"FEIL: Ingen seeds.")
        return []

    n_seeds = len(seeds_data)
    print(f"\n{n_seeds} seeds (alpha={alpha:.1f})")

    # Fase 1: Observer alle seeds
    observers = []
    for si, sd in enumerate(seeds_data):
        obs = solve_seed(client, round_id, si, sd, transition_table, simple_prior,
                        total_queries=queries_per_seed, alpha=alpha)
        observers.append(obs)
        time.sleep(0.3)

    # Fase 2: Rundetype-inferens
    factors = infer_round_type(observers)
    print(f"\n  Rundetype: sett={factors['settlement']:.2f}× forest={factors['forest']:.2f}× "
          f"ruin={factors['ruin']:.2f}× port={factors['port']:.2f}×")
    adjust_priors_for_round(observers, factors)

    # Fase 3: Cross-seed learning
    cross_table = build_cross_seed_prior(observers)
    if cross_table:
        n_keys = len(cross_table)
        total_obs = sum(v["sample_count"] for v in cross_table.values())
        print(f"  Cross-seed: {n_keys} kategorier, {total_obs} obs")
        apply_cross_seed(observers, cross_table, transition_table)

    # Fase 4: Submit
    results = []
    for si, obs in enumerate(observers):
        pred = obs.build_prediction(apply_smoothing=True)
        if not submit:
            results.append({"seed_index": si, "status": "no-submit"})
            continue
        try:
            resp = client.submit(round_id, si, pred.tolist())
            score = resp.get("score", resp.get("seed_score", "?"))
            print(f"  Seed {si} → {score}")
            results.append({"seed_index": si, "score": score})
            time.sleep(0.4)
        except Exception as e:
            print(f"  Seed {si} submit FEIL: {e}")
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

    rounds = client.get_rounds()
    if not rounds:
        print("Ingen runder"); sys.exit(1)

    if args.round:
        round_id = args.round
    else:
        active = [r for r in rounds if isinstance(r, dict) and r.get("status") == "active"]
        if not active:
            print("Ingen aktive runder."); sys.exit(1)
        round_id = active[-1]["id"]

    round_data = client.get_round(round_id)
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
                         queries_per_seed=args.queries, submit=not args.no_submit, alpha=alpha)

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
