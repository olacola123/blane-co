"""
Ola's Astar Island Solver
==========================
Nøkkelinnsikter:
1. simulate() bruker tilfeldig sim_seed — hvert kall er én stokastisk sample av slutttilstanden
2. Fokuser queries på dynamiske soner (nær initielle settlements) — observer dem FLERE ganger
3. Ocean/fjell er statiske → prediker med 99.5% sikkerhet
4. Observerte celler → empirisk distribusjon fra counts
5. Aldri 0.0 probabilitet

Strategi (10 queries per seed):
- Query 1: Stor viewport som dekker mest mulig av kartet
- Query 2-7: Fokuserte viewports rundt settlement-klynger
- Query 8-10: Revisit av mest usikre dynamiske soner

Bruk:
    export API_KEY='din-jwt-token'
    cd oppgave-3-astar-island/ola
    python solution.py
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
MIN_PROB = 0.001   # sharp floor for observerte/statiske celler
BASE_PROB = 0.01   # standard floor

# Terrain codes → klasser
# 10=Ocean→0, 11=Plains→0, 0=Empty→0, 1=Settlement→1, 2=Port→2, 3=Ruin→3, 4=Forest→4, 5=Mountain→5
TERRAIN_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}

# Terrain-prior: hva forventer vi at en celle ender som gitt startterrenget?
# Format: [empty, settlement, port, ruin, forest, mountain]
# Basert på simuleringsreglene: settlements vokser, ports utvikles fra settlements ved kyst,
# ruins oppstår fra kollaps, skog gjenreiser ruiner, fjell/hav er statiske
TERRAIN_PRIOR = {
    0:  [0.70, 0.08, 0.03, 0.06, 0.10, 0.03],  # Empty
    10: [0.998, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005],  # Ocean — nesten alltid empty
    11: [0.55, 0.18, 0.06, 0.08, 0.10, 0.03],  # Plains — mer dynamisk enn generic empty
    1:  [0.15, 0.45, 0.18, 0.12, 0.07, 0.03],  # Settlement — kan forbli, vokse til port, kollapse
    2:  [0.10, 0.25, 0.45, 0.10, 0.07, 0.03],  # Port — stabilt, kan kollapse
    3:  [0.25, 0.15, 0.08, 0.30, 0.18, 0.04],  # Ruin — gjenreises eller skog
    4:  [0.15, 0.12, 0.04, 0.08, 0.55, 0.06],  # Forest — stabilt, men kan ryddes
    5:  [0.003, 0.001, 0.001, 0.001, 0.001, 0.993],  # Mountain — statisk
}


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
        """Korrekt API-format: viewport_x/y/w/h (ikke nestet dict)."""
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

    def get_analysis(self, round_id, seed_index):
        return self.get(f"/analysis/{round_id}/{seed_index}")


# === OBSERVATION STORE ===

class SeedObserver:
    """Holder styr på observerte celler og bygger prediksjoner."""

    def __init__(self, initial_grid, settlements):
        self.grid = np.array(initial_grid, dtype=int)   # H×W terrengkoder
        self.settlements = settlements                   # [{x, y, has_port, alive}]

        # counts[y,x,c] = antall ganger celle (y,x) ble observert som klasse c
        self.counts = np.zeros((MAP_H, MAP_W, NUM_CLASSES), dtype=float)
        # observed[y,x] = antall ganger vi har observert denne cellen
        self.observed = np.zeros((MAP_H, MAP_W), dtype=int)

        # Forhåndsberegn statiske masker
        self.ocean_mask = (self.grid == 10)
        self.mountain_mask = (self.grid == 5)
        self.static_mask = self.ocean_mask | self.mountain_mask

        # Forhåndsberegn settlement-posisjoner
        self.settlement_positions = [(s["x"], s["y"]) for s in settlements]

    def add_observation(self, grid_data, viewport_x, viewport_y):
        """Legg til observasjoner fra én simulate-respons."""
        for dy, row in enumerate(grid_data):
            for dx, val in enumerate(row):
                y, x = viewport_y + dy, viewport_x + dx
                if 0 <= y < MAP_H and 0 <= x < MAP_W:
                    cls = TERRAIN_TO_CLASS.get(val, 0)
                    self.counts[y, x, cls] += 1
                    self.observed[y, x] += 1

    def add_settlement_obs(self, settlements_data, viewport_x, viewport_y, viewport_w, viewport_h):
        """Bruk settlement-attributter direkte: alive=False → ruin."""
        for s in settlements_data:
            x, y = s.get("x", -1), s.get("y", -1)
            if not (0 <= x < MAP_W and 0 <= y < MAP_H):
                continue
            alive = s.get("alive", True)
            has_port = s.get("has_port", False)
            if not alive:
                # Kollapsed settlement → ruin
                self.counts[y, x, 3] += 2   # sterkt signal
            elif has_port:
                # Port
                self.counts[y, x, 2] += 2
            else:
                # Aktiv settlement
                self.counts[y, x, 1] += 2
            self.observed[y, x] = max(self.observed[y, x], 1)

    def build_prediction(self):
        """
        Bygg 40×40×6 prediksjon.

        Prioritetsrekkefølge:
        1. Ocean/fjell → hardkoder med 99.5% sikkerhet
        2. Observerte celler (2+ ganger) → empirisk distribusjon med lite smoothing
        3. Observerte celler (1 gang) → empirisk med mer smoothing
        4. Uobserverte celler → terrain-prior
        """
        pred = np.zeros((MAP_H, MAP_W, NUM_CLASSES), dtype=float)

        for y in range(MAP_H):
            for x in range(MAP_W):
                terrain_code = self.grid[y, x]
                n_obs = self.observed[y, x]

                if self.ocean_mask[y, x]:
                    # Ocean: nesten alltid empty
                    pred[y, x] = [0.994, 0.001, 0.001, 0.001, 0.001, 0.002]

                elif self.mountain_mask[y, x]:
                    # Fjell: statisk
                    pred[y, x] = [0.001, 0.001, 0.001, 0.001, 0.001, 0.995]

                elif n_obs >= 3:
                    # Observert mange ganger → lav smoothing, sterk empirisk signal
                    alpha = 0.15
                    prior = np.array(TERRAIN_PRIOR.get(terrain_code, TERRAIN_PRIOR[11]))
                    pred[y, x] = self.counts[y, x] + alpha * prior
                    apply_floor(pred[y, x], MIN_PROB)

                elif n_obs >= 1:
                    # Observert én eller to ganger → medium smoothing
                    alpha = 0.5 if n_obs == 1 else 0.25
                    prior = np.array(TERRAIN_PRIOR.get(terrain_code, TERRAIN_PRIOR[11]))
                    pred[y, x] = self.counts[y, x] + alpha * prior
                    apply_floor(pred[y, x], MIN_PROB)

                else:
                    # Uobservert → terrain-prior
                    prior = np.array(TERRAIN_PRIOR.get(terrain_code, TERRAIN_PRIOR[11]))
                    pred[y, x] = prior
                    apply_floor(pred[y, x], BASE_PROB)

                # Normaliser
                s = pred[y, x].sum()
                if s > 0:
                    pred[y, x] /= s
                else:
                    pred[y, x] = np.ones(NUM_CLASSES) / NUM_CLASSES

        return pred


def apply_floor(arr, floor):
    np.maximum(arr, floor, out=arr)


# === QUERY PLANNER ===

def plan_queries(initial_grid, settlements, n_queries=10):
    """
    Plan viewport-posisjoner for å maksimere score.

    Matematisk optimal strategi (fra scoring-analyse):
    - 3×3 grid med x=[0,13,25], y=[0,13,25] → 100% coverage på 9 queries
    - Viewport 15×15: 0-14, 13-27, 25-39 dekker hele 40×40 uten huller
    - Resterende queries → revisit av mest dynamiske soner
    """
    # 3×3 grid = 100% coverage, 9 queries
    coverage_viewports = []
    for vy in [0, 13, 25]:
        for vx in [0, 13, 25]:
            coverage_viewports.append((vx, vy, 15, 15))

    return coverage_viewports[:n_queries]


def adaptive_revisit_queries(observer: SeedObserver, n=3):
    """
    Velg de beste revisit-viewportene basert på usikkerhet.
    Prioriter dynamiske soner med høy entropi.
    """
    # Beregn usikkerhet per celle
    # Høy usikkerhet = celle er observert men fremdeles ikke sikker
    uncertainty = np.zeros((MAP_H, MAP_W), dtype=float)
    for y in range(MAP_H):
        for x in range(MAP_W):
            if observer.static_mask[y, x]:
                continue
            if observer.observed[y, x] > 0:
                # Entropi av empirisk distribusjon
                p = observer.counts[y, x] + 0.1
                p /= p.sum()
                h = -np.sum(p * np.log(p + 1e-9))
                uncertainty[y, x] = h * (1 + observer.observed[y, x])  # Bonus for celler vi vet er dynamiske

    # Finn topp-N viewport-sentre basert på usikkerhet
    viewports = []
    temp_mask = np.zeros((MAP_H, MAP_W), dtype=bool)

    for _ in range(n):
        # Beregn sliding-window sum av usikkerhet
        best_score = -1
        best_viewport = None
        stride = 5
        for vy in range(0, MAP_H - 14, stride):
            for vx in range(0, MAP_W - 14, stride):
                region = uncertainty[vy:vy+15, vx:vx+15]
                already_covered = temp_mask[vy:vy+15, vx:vx+15].mean()
                score = region.mean() * (1 - 0.5 * already_covered)
                if score > best_score:
                    best_score = score
                    best_viewport = (vx, vy, 15, 15)

        if best_viewport:
            viewports.append(best_viewport)
            vx, vy, w, h = best_viewport
            temp_mask[vy:vy+h, vx:vx+w] = True

    return viewports


# === SOLVER ===

def solve_seed(client: AstarClient, round_id: str, seed_index: int,
               seed_data: dict, total_queries: int = 10, submit: bool = True) -> dict:
    """Løs én seed med optimal query-strategi."""
    grid = seed_data.get("grid", [])
    settlements = seed_data.get("settlements", [])

    print(f"\n  Seed {seed_index}: {len(settlements)} initielle settlements")
    observer = SeedObserver(grid, settlements)

    # Del budget: 70% coverage, 30% revisit
    coverage_queries = max(7, int(total_queries * 0.70))
    revisit_queries = total_queries - coverage_queries

    # Plan coverage queries
    planned = plan_queries(grid, settlements, n_queries=coverage_queries)

    queries_used = 0

    # === Coverage fase ===
    print(f"  Coverage fase: {len(planned)} queries")
    for i, (vx, vy, vw, vh) in enumerate(planned):
        try:
            result = client.simulate(round_id, seed_index, vx, vy, vw, vh)
            grid_data = result.get("grid", [])
            if grid_data:
                observer.add_observation(grid_data, vx, vy)
                settlements_obs = result.get("settlements", [])
                if settlements_obs:
                    observer.add_settlement_obs(settlements_obs, vx, vy, vw, vh)
            queries_used += 1
            used = result.get("queries_used", "?")
            print(f"    Q{i+1}: ({vx},{vy}) {vw}×{vh} → budget={used}")
            time.sleep(0.3)
        except Exception as e:
            print(f"    Q{i+1} FEIL: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"    Body: {e.response.text[:200]}")

    # === Revisit fase ===
    if revisit_queries > 0:
        revisits = adaptive_revisit_queries(observer, n=revisit_queries)
        print(f"  Revisit fase: {len(revisits)} queries")
        for i, (vx, vy, vw, vh) in enumerate(revisits):
            try:
                result = client.simulate(round_id, seed_index, vx, vy, vw, vh)
                grid_data = result.get("grid", [])
                if grid_data:
                    observer.add_observation(grid_data, vx, vy)
                    settlements_obs = result.get("settlements", [])
                    if settlements_obs:
                        observer.add_settlement_obs(settlements_obs, vx, vy, vw, vh)
                queries_used += 1
                used = result.get("queries_used", "?")
                print(f"    R{i+1}: ({vx},{vy}) revisit → budget={used}")
                time.sleep(0.3)
            except Exception as e:
                print(f"    R{i+1} FEIL: {e}")

    # Bygg prediksjon
    pred = observer.build_prediction()

    coverage = (observer.observed > 0).sum() / (MAP_H * MAP_W)
    mean_obs = observer.observed.mean()
    print(f"  Coverage: {coverage:.1%}, mean observations/celle: {mean_obs:.2f}")

    if not submit:
        return {"seed_index": seed_index, "coverage": coverage}

    # Submit
    try:
        resp = client.submit(round_id, seed_index, pred.tolist())
        score = resp.get("score", "?")
        print(f"  Seed {seed_index} score: {score}")
        return {"seed_index": seed_index, "score": score, "coverage": coverage}
    except Exception as e:
        print(f"  Submit FEIL: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Body: {e.response.text[:300]}")
        return {"seed_index": seed_index, "error": str(e)}


def main():
    if not API_KEY:
        print("FEIL: export API_KEY='din-jwt-token'")
        print("Hent token: app.ainm.no → F12 → Application → Cookies → access_token")
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=None, help="Round ID (default: auto)")
    parser.add_argument("--seeds", type=int, default=5, help="Antall seeds (default: 5)")
    parser.add_argument("--queries", type=int, default=10, help="Queries per seed (default: 10)")
    parser.add_argument("--no-submit", action="store_true", help="Ikke submit, bare observer")
    parser.add_argument("--dry-run", action="store_true", help="Ingen API-kall, bare test")
    args = parser.parse_args()

    client = AstarClient()

    # Budget-sjekk
    try:
        budget = client.get_budget()
        print(f"Budget: {json.dumps(budget)}")
    except Exception as e:
        print(f"Budget-sjekk feilet: {e}")

    # Finn aktiv runde
    rounds = client.get_rounds()
    if not rounds:
        print("Ingen runder tilgjengelig")
        sys.exit(1)

    if args.round:
        round_id = args.round
        round_data = client.get_round(round_id)
    else:
        # Bruk nyeste aktive runde
        active = [r for r in rounds if isinstance(r, dict) and r.get("status") == "active"]
        if not active:
            active = rounds  # Ta siste uansett
        latest = active[-1]
        round_id = latest["id"] if isinstance(latest, dict) else latest
        round_data = client.get_round(round_id)

    print(f"\nRunde: {round_id}")
    print(f"Kart: {round_data.get('width', '?')}×{round_data.get('height', '?')}")
    print(f"Stenger: {round_data.get('closes_at', '?')}")

    if args.dry_run:
        print("\n[DRY RUN] Ingen API-kall")
        return

    # Hent seeds
    seeds_data = round_data.get("seeds", round_data.get("initial_states", []))
    if not seeds_data:
        print("FEIL: Ingen seeds i round data")
        print(f"Round data keys: {list(round_data.keys())}")
        sys.exit(1)

    print(f"\n{len(seeds_data)} seeds funnet")

    results = []
    for seed_index, seed_data in enumerate(seeds_data[:args.seeds]):
        result = solve_seed(
            client=client,
            round_id=round_id,
            seed_index=seed_index,
            seed_data=seed_data,
            total_queries=args.queries,
            submit=not args.no_submit,
        )
        results.append(result)
        time.sleep(0.5)

    # Oppsummering
    print("\n=== OPPSUMMERING ===")
    scores = [r.get("score") for r in results if "score" in r]
    for r in results:
        score = r.get("score", r.get("error", "ingen score"))
        print(f"  Seed {r['seed_index']}: {score}")

    # Lagre
    out_path = Path(__file__).parent / "results.json"
    out_path.write_text(json.dumps({
        "round_id": round_id,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2, default=str))
    print(f"\nResultater lagret: {out_path}")


if __name__ == "__main__":
    main()
