"""
Ola's Astar Island Solver v4 — Kalibrert
==========================================
Backtest-bevist: prior-only scorer 73 snitt, 90 beste.
Observasjoner skal NUDGE prioren forsiktig, ikke overskrive den.

Nøkkelprinsipp:
- Prioren (fra 9 runder fasit) er pålitelig — bygd fra tusenvis av celler
- Én observasjon er bare ÉN tilfeldig utfall av hundrevis mulige
- Alpha MÅ være høy: prior dominerer til vi har mange observasjoner
- Cross-seed: alle 5 seeds deler skjulte parametere → poolt observasjoner

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
PROB_FLOOR = 0.003  # minimum sannsynlighet — aldri 0.0

TERRAIN_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}

# Avstandsbånd (matcher calibrate.py)
DISTANCE_BANDS = [0, 1, 2, 3, 5, 8, 12, 99]

# Fallback priors
FALLBACK_PRIOR = {
    "0":  [0.70, 0.08, 0.03, 0.06, 0.10, 0.03],
    "10": [0.999, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002],
    "11": [0.822, 0.121, 0.009, 0.012, 0.035, 0.001],
    "1":  [0.462, 0.293, 0.004, 0.026, 0.216, 0.001],
    "2":  [0.484, 0.089, 0.173, 0.022, 0.232, 0.001],
    "3":  [0.20, 0.15, 0.08, 0.32, 0.20, 0.05],
    "4":  [0.079, 0.127, 0.009, 0.013, 0.772, 0.001],
    "5":  [0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.999],
}

CALIBRATION_FILE = Path(__file__).parent / "calibration_data.json"


# === CALIBRATION LOADER ===

def load_calibration():
    if not CALIBRATION_FILE.exists():
        print("ADVARSEL: Ingen calibration_data.json — bruker fallback priors")
        return None, None
    data = json.loads(CALIBRATION_FILE.read_text())
    transition_table = data.get("transition_table", {})
    simple_prior = data.get("simple_prior", {})
    print(f"Lastet kalibrering fra {data.get('num_rounds', '?')} runder, {data.get('num_seeds', '?')} seeds")
    return transition_table, simple_prior


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

    def get_rounds(self):
        return self.get("/rounds")

    def get_round(self, round_id):
        return self.get(f"/rounds/{round_id}")

    def get_budget(self):
        return self.get("/budget")

    def simulate(self, round_id, seed_index, x, y, w=MAX_VIEWPORT, h=MAX_VIEWPORT):
        x = max(0, min(x, MAP_W - w))
        y = max(0, min(y, MAP_H - h))
        return self.post("/simulate", {
            "round_id": round_id,
            "seed_index": seed_index,
            "viewport_x": x,
            "viewport_y": y,
            "viewport_w": w,
            "viewport_h": h,
        })

    def submit(self, round_id, seed_index, prediction):
        return self.post("/submit", {
            "round_id": round_id,
            "seed_index": seed_index,
            "prediction": prediction,
        })

    def get_my_rounds(self):
        return self.get("/my-rounds")

    def get_leaderboard(self):
        return self.get("/leaderboard")


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
        key_no_coast = f"{terrain}_{band}_0"
        if key_no_coast in transition_table:
            return np.array(transition_table[key_no_coast]["distribution"], dtype=float)
    if simple_prior and str(terrain) in simple_prior:
        return np.array(simple_prior[str(terrain)], dtype=float)
    if str(terrain) in FALLBACK_PRIOR:
        return np.array(FALLBACK_PRIOR[str(terrain)], dtype=float)
    return np.ones(NUM_CLASSES, dtype=float) / NUM_CLASSES


def cell_key(grid, y, x, settlements):
    """Nøkkel for en celle: terreng_avstandsbånd_kyst."""
    terrain = int(grid[y][x]) if isinstance(grid, np.ndarray) else grid[y][x]
    dist = distance_to_nearest_settlement(y, x, settlements)
    band = get_distance_band(dist)
    coastal = is_coastal(grid if isinstance(grid, list) else grid.tolist(), y, x)
    return terrain, band, coastal


# === OBSERVATION STORE ===

class SeedObserver:
    def __init__(self, initial_grid, settlements, transition_table, simple_prior):
        self.grid = np.array(initial_grid, dtype=int)
        self.settlements = settlements
        self.transition_table = transition_table
        self.simple_prior = simple_prior

        self.counts = np.zeros((MAP_H, MAP_W, NUM_CLASSES), dtype=float)
        self.observed = np.zeros((MAP_H, MAP_W), dtype=int)

        self.ocean_mask = (self.grid == 10)
        self.mountain_mask = (self.grid == 5)
        self.static_mask = self.ocean_mask | self.mountain_mask

        # Forhåndsberegn prior for hver celle
        self._prior_cache = np.zeros((MAP_H, MAP_W, NUM_CLASSES), dtype=float)
        grid_list = self.grid.tolist()
        for y in range(MAP_H):
            for x in range(MAP_W):
                if self.static_mask[y, x]:
                    continue
                terrain, band, coastal = cell_key(grid_list, y, x, settlements)
                self._prior_cache[y, x] = get_prior(
                    terrain, band, coastal, transition_table, simple_prior
                )

    def add_observation(self, grid_data, viewport_x, viewport_y):
        for dy, row in enumerate(grid_data):
            for dx, val in enumerate(row):
                y, x = viewport_y + dy, viewport_x + dx
                if 0 <= y < MAP_H and 0 <= x < MAP_W:
                    cls = TERRAIN_TO_CLASS.get(val, 0)
                    self.counts[y, x, cls] += 1
                    self.observed[y, x] += 1

    def add_settlement_obs(self, settlements_data):
        """Settlement-attributter som svakt signal."""
        for s in settlements_data:
            x, y = s.get("x", -1), s.get("y", -1)
            if not (0 <= x < MAP_W and 0 <= y < MAP_H):
                continue
            alive = s.get("alive", True)
            has_port = s.get("has_port", False)
            # Svakt signal: 0.3 av en observasjon (prioren skal dominere)
            if not alive:
                self.counts[y, x, 3] += 0.3
            elif has_port:
                self.counts[y, x, 2] += 0.3
            else:
                self.counts[y, x, 1] += 0.3

    def build_prediction(self):
        """
        Bayesiansk posterior med HØY prior-styrke.

        Backtest viser: prior-only = 73 snitt.
        Observasjoner skal nudge, ikke overskrive.

        posterior = (obs_counts + alpha × prior) / (n_obs + alpha)
        alpha er HØY → prioren dominerer med få observasjoner.
        """
        pred = np.zeros((MAP_H, MAP_W, NUM_CLASSES), dtype=float)

        for y in range(MAP_H):
            for x in range(MAP_W):
                # Statiske celler
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
                    # Backtest-optimalisert: alpha=15 er optimal
                    # Prior dominerer med få obs, obs nudger forsiktig
                    # Med 1 obs og alpha=15: prior har 94% vekt
                    # Med 3 obs og alpha=12: prior har 80% vekt
                    # Med 10 obs og alpha=5: prior har 33% vekt
                    if n_obs >= 10:
                        alpha = 5.0
                    elif n_obs >= 5:
                        alpha = 8.0
                    elif n_obs >= 3:
                        alpha = 12.0
                    else:
                        alpha = 15.0

                    pred[y, x] = self.counts[y, x] + alpha * prior

                # Floor og normaliser
                np.maximum(pred[y, x], PROB_FLOOR, out=pred[y, x])
                pred[y, x] /= pred[y, x].sum()

        return pred


# === CROSS-SEED LEARNING ===

def build_cross_seed_prior(all_observers):
    """
    Bygg rundespesifikk prior fra observasjoner på tvers av alle seeds.
    Bruker terreng+avstand+kyst (ikke bare terreng) for bedre presisjon.

    Idé: alle seeds deler skjulte parametere. Så observasjoner fra seed 0
    informerer seed 4, spesielt for samme type celle.
    """
    # Akkumuler per (terreng, avstandsbånd, kyst)
    key_counts = defaultdict(lambda: np.zeros(NUM_CLASSES, dtype=float))
    key_n = defaultdict(int)

    for obs in all_observers:
        grid_list = obs.grid.tolist()
        for y in range(MAP_H):
            for x in range(MAP_W):
                if obs.static_mask[y, x] or obs.observed[y, x] == 0:
                    continue
                terrain, band, coastal = cell_key(grid_list, y, x, obs.settlements)
                key = f"{terrain}_{band}_{int(coastal)}"
                key_counts[key] += obs.counts[y, x]
                key_n[key] += obs.observed[y, x]

    # Normaliser til distribusjoner
    cross_table = {}
    for key, counts in key_counts.items():
        total = counts.sum()
        if total >= 3:  # Minst 3 observasjoner for å være pålitelig
            cross_table[key] = {
                "distribution": (counts / total).tolist(),
                "sample_count": int(total),
            }

    return cross_table


def apply_cross_seed(observers, cross_table, calibration_table):
    """
    Bland cross-seed prior med historisk prior.
    Cross-seed er rundespesifikk (fanger skjulte parametere).
    Historisk er gjennomsnitt over runder (mer stabil).
    """
    if not cross_table:
        return

    # Oppdater prior-cachen i hver observer
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

                    # Blend: vekt cross-seed mer når vi har mange observasjoner
                    # cross_weight øker med antall observasjoner
                    cross_weight = min(0.5, cross_n / 100.0)
                    blended = (1 - cross_weight) * hist_prior + cross_weight * cross_dist

                    # Normaliser
                    blended = np.maximum(blended, PROB_FLOOR)
                    blended /= blended.sum()

                    obs._prior_cache[y, x] = blended


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
                    dist = max(abs(dx), abs(dy))
                    if dist == 0:
                        heatmap[y, x] += 5.0
                    elif dist <= 2:
                        heatmap[y, x] += 3.0
                    elif dist <= 5:
                        heatmap[y, x] += 1.5
                    elif dist <= 8:
                        heatmap[y, x] += 0.5
                    else:
                        heatmap[y, x] += 0.1

    heatmap[grid == 10] = 0.0
    heatmap[grid == 5] = 0.0
    return heatmap


def plan_queries(initial_grid, settlements, n_queries=10):
    """Greedy viewport-plassering. Overlap gir nye stokastiske samples."""
    heatmap = build_dynamism_heatmap(initial_grid, settlements)
    viewports = []
    obs_count = np.zeros((MAP_H, MAP_W), dtype=int)

    for _ in range(n_queries):
        best_score = -1
        best_vp = (0, 0)

        for vy in range(0, MAP_H - MAX_VIEWPORT + 1, 2):
            for vx in range(0, MAP_W - MAX_VIEWPORT + 1, 2):
                region_heat = heatmap[vy:vy+MAX_VIEWPORT, vx:vx+MAX_VIEWPORT]
                region_obs = obs_count[vy:vy+MAX_VIEWPORT, vx:vx+MAX_VIEWPORT]
                obs_value = 1.0 / (1.0 + 0.35 * region_obs)
                score = (region_heat * obs_value).sum()

                if score > best_score:
                    best_score = score
                    best_vp = (vx, vy)

        vx, vy = best_vp
        viewports.append((vx, vy, MAX_VIEWPORT, MAX_VIEWPORT))
        obs_count[vy:vy+MAX_VIEWPORT, vx:vx+MAX_VIEWPORT] += 1

    return viewports


# === SOLVER ===

def solve_seed(client, round_id, seed_index, seed_data, transition_table,
               simple_prior, total_queries=10):
    grid = seed_data.get("grid", [])
    settlements = seed_data.get("settlements", [])

    print(f"\n  Seed {seed_index}: {len(settlements)} settlements")
    observer = SeedObserver(grid, settlements, transition_table, simple_prior)

    viewports = plan_queries(grid, settlements, n_queries=total_queries)

    for i, (vx, vy, vw, vh) in enumerate(viewports):
        try:
            result = client.simulate(round_id, seed_index, vx, vy, vw, vh)
            grid_data = result.get("grid", [])
            if grid_data:
                observer.add_observation(grid_data, vx, vy)
                sett_obs = result.get("settlements", [])
                if sett_obs:
                    observer.add_settlement_obs(sett_obs)
            budget_used = result.get("queries_used", "?")
            budget_max = result.get("queries_max", "?")
            print(f"    Q{i+1}: ({vx},{vy}) → {budget_used}/{budget_max}")
            time.sleep(0.25)
        except Exception as e:
            print(f"    Q{i+1} FEIL: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"    Body: {e.response.text[:200]}")

    dynamic_mask = ~observer.static_mask
    dynamic_obs = (observer.observed[dynamic_mask] > 0).sum()
    dynamic_total = dynamic_mask.sum()
    mean_obs = observer.observed[dynamic_mask].mean() if dynamic_total > 0 else 0
    print(f"  Dynamiske: {dynamic_obs}/{dynamic_total} observert, snitt {mean_obs:.1f} obs/celle")

    return observer


def solve_round(client, round_id, round_data, transition_table, simple_prior,
                queries_per_seed=10, submit=True):
    seeds_data = round_data.get("seeds", round_data.get("initial_states", []))
    if not seeds_data:
        print(f"FEIL: Ingen seeds. Keys: {list(round_data.keys())}")
        return []

    print(f"\n{len(seeds_data)} seeds")

    # Fase 1: Observer alle seeds
    observers = []
    for seed_idx, seed_data in enumerate(seeds_data):
        obs = solve_seed(client, round_id, seed_idx, seed_data,
                        transition_table, simple_prior,
                        total_queries=queries_per_seed)
        observers.append(obs)
        time.sleep(0.3)

    # Fase 2: Cross-seed learning (rundespesifikk kalibrering)
    cross_table = build_cross_seed_prior(observers)
    if cross_table:
        n_keys = len(cross_table)
        total_obs = sum(v["sample_count"] for v in cross_table.values())
        print(f"\n  Cross-seed: {n_keys} kategorier, {total_obs} observasjoner")
        apply_cross_seed(observers, cross_table, transition_table)
        print(f"  Priors oppdatert med rundespesifikk data")

    # Fase 3: Bygg prediksjoner og submit
    results = []
    for seed_idx, obs in enumerate(observers):
        pred = obs.build_prediction()

        if not submit:
            results.append({"seed_index": seed_idx, "status": "no-submit"})
            continue

        try:
            resp = client.submit(round_id, seed_idx, pred.tolist())
            score = resp.get("score", resp.get("seed_score", "?"))
            print(f"  Seed {seed_idx} → score: {score}")
            results.append({"seed_index": seed_idx, "score": score})
            time.sleep(0.3)
        except Exception as e:
            print(f"  Seed {seed_idx} submit FEIL: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"  Body: {e.response.text[:300]}")
            results.append({"seed_index": seed_idx, "error": str(e)})

    return results


def main():
    if not API_KEY:
        print("FEIL: export API_KEY='din-jwt-token'")
        print("Hent: app.ainm.no → F12 → Application → Cookies → access_token")
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=None, help="Round ID")
    parser.add_argument("--queries", type=int, default=10, help="Queries per seed")
    parser.add_argument("--no-submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--leaderboard", action="store_true")
    args = parser.parse_args()

    client = AstarClient()
    transition_table, simple_prior = load_calibration()

    if args.leaderboard:
        try:
            lb = client.get_leaderboard()
            print("=== LEADERBOARD (topp 15) ===")
            for i, entry in enumerate(lb[:15]):
                if isinstance(entry, dict):
                    name = entry.get("team_name", entry.get("username", "?"))
                    score = entry.get("weighted_score", entry.get("score", "?"))
                    streak = entry.get("hot_streak", "?")
                    rounds = entry.get("rounds_played", "?")
                    print(f"  #{i+1} {name}: {score} (streak={streak}, runder={rounds})")
        except Exception as e:
            print(f"Leaderboard feil: {e}")
        return

    try:
        budget = client.get_budget()
        used = budget.get("queries_used", 0)
        total = budget.get("queries_max", 50)
        print(f"Budget: {used}/{total} brukt")
        if used >= total and not args.dry_run:
            print("ADVARSEL: Budget oppbrukt!")
    except Exception as e:
        print(f"Budget feil: {e}")

    rounds = client.get_rounds()
    if not rounds:
        print("Ingen runder")
        sys.exit(1)

    if args.round:
        round_id = args.round
    else:
        active = [r for r in rounds if isinstance(r, dict) and r.get("status") == "active"]
        if not active:
            print("Ingen aktive runder.")
            for r in rounds[-3:]:
                if isinstance(r, dict):
                    print(f"  {r.get('round_number', '?')}: {r.get('status')} ({r.get('id', '?')[:12]}...)")
            sys.exit(1)
        round_id = active[-1]["id"]

    round_data = client.get_round(round_id)
    rnum = round_data.get("round_number", "?")
    closes = round_data.get("closes_at", "?")
    weight = round_data.get("round_weight", "?")
    print(f"\nRunde {rnum} (vekt {weight}): {round_id[:12]}...")
    print(f"Stenger: {closes}")

    if args.dry_run:
        print("\n[DRY RUN]")
        seeds = round_data.get("seeds", round_data.get("initial_states", []))
        for i, s in enumerate(seeds):
            setts = s.get("settlements", [])
            grid = s.get("grid", [])
            vps = plan_queries(grid, setts, n_queries=args.queries)
            print(f"\n  Seed {i}: {len(setts)} settlements")
            print(f"  Viewports: {[(vx,vy) for vx,vy,_,_ in vps]}")
        return

    results = solve_round(
        client, round_id, round_data,
        transition_table, simple_prior,
        queries_per_seed=args.queries,
        submit=not args.no_submit,
    )

    print("\n=== RESULTATER ===")
    for r in results:
        s = r.get("score", r.get("error", r.get("status", "?")))
        print(f"  Seed {r['seed_index']}: {s}")

    scores = [r["score"] for r in results
              if "score" in r and isinstance(r.get("score"), (int, float))]
    if scores:
        avg = sum(scores) / len(scores)
        print(f"\n  Snitt: {avg:.1f}")
        if weight and isinstance(weight, (int, float)):
            print(f"  Weighted: {avg * weight:.1f}")

    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps({
        "round_id": round_id, "round_number": rnum,
        "results": results, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2, default=str))
    print(f"Lagret: {out}")


if __name__ == "__main__":
    main()
