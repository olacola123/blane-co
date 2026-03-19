"""
Astar Island: Norse World Prediction
=====================================
Strategi:
1. Hent initial state for alle 5 seeds
2. Kjør 10 simulate-queries per seed med viewport-grid som dekker hele 40x40
3. Aggreger observasjoner til sannsynlighetsfordelinger
4. Floor 0.01, renormaliser, submit
"""

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# === CONFIG ===
BASE_URL = "https://api.ainm.no/astar-island"
API_KEY = os.environ.get("API_KEY", "")
ROUND_ID = None  # Auto-detect from /rounds

MAP_W, MAP_H = 40, 40
NUM_CLASSES = 6
VIEWPORT_SIZE = 15
MIN_PROB = 0.01

# Terrain type → prediction class mapping (from OPPGAVE.md)
# 8 terrain types → 6 classes
# Class 0: Empty/Ocean/Plains
# Class 1: Settlements
# Class 2: Ports
# Class 3: Ruins
# Class 4: Forests
# Class 5: Mountains

# We'll discover the mapping from simulation data

# === API CLIENT ===

class AstarClient:
    def __init__(self):
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        })

    def get(self, endpoint, params=None):
        url = f"{BASE_URL}/{endpoint.lstrip('/')}"
        r = self.session.get(url, params=params)
        r.raise_for_status()
        return r.json()

    def post(self, endpoint, data):
        url = f"{BASE_URL}/{endpoint.lstrip('/')}"
        r = self.session.post(url, json=data)
        r.raise_for_status()
        return r.json()


def get_active_round(client):
    """Hent aktiv runde."""
    rounds = client.get("/rounds")
    active = [r for r in rounds if r["status"] == "active"]
    if not active:
        print("INGEN aktive runder!")
        sys.exit(1)
    r = active[0]
    print(f"Runde {r['round_number']}: {r['id']}")
    print(f"  Kart: {r['map_width']}x{r['map_height']}")
    print(f"  Stenger: {r['closes_at']}")
    print(f"  Vekt: {r['round_weight']}")
    return r


def get_round_data(client, round_id):
    """Hent initial state med alle seeds."""
    data = client.get(f"/rounds/{round_id}")
    return data


def check_budget(client):
    """Sjekk gjenværende queries."""
    try:
        budget = client.get("/budget")
        print(f"Budget: {budget}")
        return budget
    except Exception as e:
        print(f"Budget-sjekk feilet: {e}")
        return None


def generate_viewport_positions():
    """
    Generer 10 viewport-posisjoner som dekker hele 40x40.
    3x3 grid = 9 viewports, pluss 1 ekstra i sentrum.
    """
    positions = []

    # 3x3 grid med 15x15 viewports
    # Steg: (40 - 15) / 2 = 12.5 → bruk 0, 12, 25 som start-x/y
    starts = [0, 13, 25]  # 25 + 15 = 40, perfekt

    for y in starts:
        for x in starts:
            positions.append((x, y))

    # 10. query: sentrum for ekstra data på dynamisk område
    positions.append((12, 12))  # Sentrum-ish

    return positions


def simulate_query(client, round_id, seed_index, x, y, width=VIEWPORT_SIZE, height=VIEWPORT_SIZE):
    """Kjør én simuleringsquery."""
    # Clamp viewport til kartkanter
    x = min(x, MAP_W - width)
    y = min(y, MAP_H - height)

    payload = {
        "round_id": round_id,
        "seed_index": seed_index,
        "viewport": {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        }
    }

    result = client.post("/simulate", payload)
    return result


def parse_terrain_value(value):
    """
    Map terrain values til prediction classes.
    Vi oppdager mappingen fra data, men starter med rimelige antakelser.
    """
    # Basert på initial state observasjoner:
    # Verdier sett: 1, 2, 4, 5, 10, 11
    # Mulige terrain types i simulator: ocean, plains, forest, mountain, settlement, port, ruins, empty

    # Mapping basert på OPPGAVE.md klasser:
    terrain_map = {
        0: 0,   # Empty → class 0
        1: 1,   # Settlement → class 1
        2: 2,   # Port → class 2
        3: 3,   # Ruins → class 3
        4: 4,   # Forest → class 4
        5: 5,   # Mountain → class 5
        # Verdier fra initial grid som vi har sett
        10: 0,  # Border/Ocean → class 0
        11: 0,  # Land/Plains → class 0
    }

    return terrain_map.get(value, 0)  # Default til class 0


def run_queries_for_seed(client, round_id, seed_index, positions):
    """Kjør alle queries for én seed og returner observasjoner."""
    # observations[y][x] = list of observed classes
    observations = [[[] for _ in range(MAP_W)] for _ in range(MAP_H)]

    for i, (vx, vy) in enumerate(positions):
        print(f"  Seed {seed_index}, query {i+1}/10: viewport ({vx},{vy})")

        try:
            result = simulate_query(client, round_id, seed_index, vx, vy)

            # Parse viewport-data
            viewport_data = result.get("viewport", result.get("grid", result.get("terrain", result.get("data", []))))

            if not viewport_data:
                print(f"    WARN: Tom respons, keys: {list(result.keys())}")
                # Dump respons for debugging
                print(f"    Respons: {json.dumps(result)[:500]}")
                continue

            # Parse grid data
            for dy, row in enumerate(viewport_data):
                for dx, val in enumerate(row):
                    map_x = vx + dx
                    map_y = vy + dy
                    if 0 <= map_x < MAP_W and 0 <= map_y < MAP_H:
                        cls = parse_terrain_value(val)
                        observations[map_y][map_x].append(cls)

            print(f"    OK: {len(viewport_data)}x{len(viewport_data[0]) if viewport_data else 0} celler")

        except requests.exceptions.HTTPError as e:
            print(f"    FEIL: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"    Body: {e.response.text[:300]}")
        except Exception as e:
            print(f"    FEIL: {e}")

        time.sleep(0.5)  # Rate limiting

    return observations


def build_predictions(observations_per_seed, initial_states=None):
    """
    Bygg prediction tensor fra observasjoner.
    For celler med observasjoner: frekvensbasert distribusjon
    For celler uten: uniform prior
    """
    predictions = {}

    for seed_idx, observations in observations_per_seed.items():
        pred = np.full((MAP_H, MAP_W, NUM_CLASSES), 1.0 / NUM_CLASSES)

        for y in range(MAP_H):
            for x in range(MAP_W):
                obs = observations[y][x]
                if obs:
                    # Frekvensbasert med smoothing
                    counts = np.zeros(NUM_CLASSES)
                    for cls in obs:
                        if 0 <= cls < NUM_CLASSES:
                            counts[cls] += 1

                    # Laplace smoothing
                    counts += 0.5
                    pred[y][x] = counts / counts.sum()

        # Apply initial state info if available
        if initial_states and seed_idx in initial_states:
            pred = incorporate_initial_state(pred, initial_states[seed_idx])

        # Floor og renormalisering
        pred = apply_floor(pred)

        predictions[seed_idx] = pred

    return predictions


def incorporate_initial_state(pred, initial_state):
    """
    Bruk initial state til å informere predictions.
    Fjell og hav endres sjelden. Settlements er dynamiske.
    """
    grid = initial_state.get("grid", initial_state.get("terrain", []))
    settlements = initial_state.get("settlements", [])

    if grid:
        for y, row in enumerate(grid):
            for x, val in enumerate(row):
                if y < MAP_H and x < MAP_W:
                    # Fjell (5) og hav/border (10) endres sjelden
                    if val == 5:  # Mountain
                        pred[y][x] = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.90])
                    elif val == 10:  # Ocean/border
                        pred[y][x] = np.array([0.90, 0.02, 0.02, 0.02, 0.02, 0.02])

    return pred


def apply_floor(pred, floor=MIN_PROB):
    """Sett minimum probability og renormaliser."""
    pred = np.maximum(pred, floor)
    # Renormaliser langs klasse-aksen
    sums = pred.sum(axis=2, keepdims=True)
    pred = pred / sums
    return pred


def submit_predictions(client, round_id, predictions):
    """Submit predictions for alle seeds."""
    results = {}

    for seed_idx, pred in sorted(predictions.items()):
        print(f"\nSubmitter seed {seed_idx}...")

        # Konverter til nested list
        pred_list = pred.tolist()

        # Verifiser format
        assert len(pred_list) == MAP_H
        assert len(pred_list[0]) == MAP_W
        assert len(pred_list[0][0]) == NUM_CLASSES

        # Sjekk at alle rader summerer til ~1.0
        for y in range(MAP_H):
            for x in range(MAP_W):
                s = sum(pred_list[y][x])
                assert abs(s - 1.0) < 0.02, f"Cell ({x},{y}) sums to {s}"

        payload = {
            "round_id": round_id,
            "seed_index": seed_idx,
            "prediction": pred_list,
        }

        try:
            result = client.post("/submit", payload)
            score = result.get("score", "?")
            print(f"  Seed {seed_idx}: score = {score}")
            results[seed_idx] = result
        except requests.exceptions.HTTPError as e:
            print(f"  FEIL: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"  Body: {e.response.text[:500]}")
            results[seed_idx] = {"error": str(e)}

    return results


def submit_uniform_baseline(client, round_id, num_seeds=5):
    """Submit uniform prior som baseline — får score på tavla ASAP."""
    print("\n=== BASELINE: Uniform prior (1/6 per klasse) ===")

    pred = np.full((MAP_H, MAP_W, NUM_CLASSES), 1.0 / NUM_CLASSES)
    pred = apply_floor(pred)

    predictions = {i: pred.copy() for i in range(num_seeds)}
    return submit_predictions(client, round_id, predictions)


def main():
    if not API_KEY:
        print("FEIL: Sett API_KEY!")
        print("  export API_KEY='din-nøkkel-her'")
        sys.exit(1)

    client = AstarClient()

    # 1. Finn aktiv runde
    round_info = get_active_round(client)
    round_id = round_info["id"]

    # 2. Sjekk budget
    budget = check_budget(client)

    # 3. Hent initial state
    print("\n=== Henter initial state ===")
    round_data = get_round_data(client, round_id)

    # Debug: vis struktur
    if isinstance(round_data, dict):
        print(f"Round data keys: {list(round_data.keys())}")
        # Finn seeds/initial states
        for key in round_data:
            val = round_data[key]
            if isinstance(val, list) and len(val) > 0:
                print(f"  {key}: list med {len(val)} elementer")
                if isinstance(val[0], dict):
                    print(f"    Element keys: {list(val[0].keys())}")
            elif isinstance(val, dict):
                print(f"  {key}: dict med keys {list(val.keys())[:10]}")

    # Parse initial states
    initial_states = {}
    seeds_data = round_data.get("seeds", round_data.get("initial_states", round_data.get("states", [])))
    if isinstance(seeds_data, list):
        for i, state in enumerate(seeds_data):
            initial_states[i] = state
            print(f"  Seed {i}: {list(state.keys()) if isinstance(state, dict) else type(state)}")

    # 4. Submit baseline FØRST
    print("\n=== Steg 1: Baseline submission ===")
    if "--skip-baseline" not in sys.argv:
        baseline_results = submit_uniform_baseline(client, round_id)

    # 5. Kjør simulate queries
    print("\n=== Steg 2: Simulate queries ===")
    positions = generate_viewport_positions()
    print(f"Viewport-posisjoner: {positions}")

    observations_per_seed = {}
    for seed_idx in range(5):
        print(f"\n--- Seed {seed_idx} ---")
        obs = run_queries_for_seed(client, round_id, seed_idx, positions)
        observations_per_seed[seed_idx] = obs

    # 6. Bygg prediksjoner
    print("\n=== Steg 3: Bygg prediksjoner ===")
    predictions = build_predictions(observations_per_seed, initial_states)

    # 7. Submit forbedrede prediksjoner
    print("\n=== Steg 4: Submit forbedrede prediksjoner ===")
    results = submit_predictions(client, round_id, predictions)

    # 8. Oppsummering
    print("\n=== OPPSUMMERING ===")
    for seed_idx, result in sorted(results.items()):
        score = result.get("score", result.get("error", "?"))
        print(f"  Seed {seed_idx}: {score}")

    # Lagre resultater
    with open("/Users/olatandberg/vault/Prosjekter/NM i AI/hovedkonkurranse/oppgave-3-astar-island/ola/results.json", "w") as f:
        json.dump({
            "round_id": round_id,
            "results": {str(k): v for k, v in results.items()},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
