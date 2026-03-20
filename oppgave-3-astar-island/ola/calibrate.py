"""
Astar Island Calibration
=========================
Henter ground truth fra alle fullførte runder og bygger empiriske overgangstabeller.

Hva dette gjør:
1. Henter fasit (ground truth) fra /analysis for alle fullførte runder
2. For hver celle: ser på initial terreng + avstand til nærmeste settlement → hva ble det?
3. Lagrer en overgangstabell som solution.py bruker som kalibrert prior

Bruk:
    export API_KEY='din-jwt-token'
    python calibrate.py           # Hent data og bygg tabell
    python calibrate.py --show    # Vis eksisterende tabell
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
NUM_CLASSES = 6
MAP_W, MAP_H = 40, 40

# Terrain codes
TERRAIN_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}

# Avstandsbånd for conditioning
DISTANCE_BANDS = [0, 1, 2, 3, 5, 8, 12, 99]  # → bands: 0, 1, 2, 3-5, 6-8, 9-12, 13+
BAND_LABELS = ["d=0", "d=1", "d=2", "d=3-5", "d=6-8", "d=9-12", "d=13+"]

OUTPUT_FILE = Path(__file__).parent / "calibration_data.json"


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


def distance_to_nearest_settlement(y, x, settlements):
    """Chebyshev-avstand til nærmeste settlement."""
    if not settlements:
        return 99
    return min(max(abs(x - s["x"]), abs(y - s["y"])) for s in settlements)


def get_distance_band(dist):
    """Konverter avstand til båndindeks."""
    for i in range(len(DISTANCE_BANDS) - 1):
        if dist <= DISTANCE_BANDS[i]:
            return i
    return len(DISTANCE_BANDS) - 2


def is_coastal(grid, y, x):
    """Sjekk om cellen grenser til hav (terreng 10)."""
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < MAP_H and 0 <= nx < MAP_W:
                if grid[ny][nx] == 10:
                    return True
    return False


def fetch_all_ground_truth(client):
    """Hent ground truth fra alle fullførte runder."""
    rounds = client.get("/rounds")
    completed = [r for r in rounds if isinstance(r, dict) and r.get("status") == "completed"]

    print(f"Fant {len(completed)} fullførte runder")

    all_data = []

    for round_info in completed:
        round_id = round_info["id"]
        round_num = round_info.get("round_number", "?")
        print(f"\nRunde {round_num} ({round_id[:12]}...):")

        # Hent initial state
        try:
            round_data = client.get(f"/rounds/{round_id}")
        except Exception as e:
            print(f"  Kunne ikke hente round data: {e}")
            continue

        seeds = round_data.get("seeds", round_data.get("initial_states", []))
        if not seeds:
            print("  Ingen seeds")
            continue

        for seed_idx in range(len(seeds)):
            try:
                analysis = client.get(f"/analysis/{round_id}/{seed_idx}")
                gt = analysis.get("ground_truth")
                score = analysis.get("score", "?")

                if gt is None:
                    print(f"  Seed {seed_idx}: ingen ground truth")
                    continue

                initial_grid = seeds[seed_idx].get("grid", [])
                settlements = seeds[seed_idx].get("settlements", [])

                all_data.append({
                    "round_id": round_id,
                    "round_number": round_num,
                    "seed_index": seed_idx,
                    "initial_grid": initial_grid,
                    "settlements": settlements,
                    "ground_truth": gt,
                    "score": score,
                })
                print(f"  Seed {seed_idx}: score={score}, gt shape={len(gt)}×{len(gt[0]) if gt else 0}")
                time.sleep(0.3)  # rate limit

            except Exception as e:
                print(f"  Seed {seed_idx}: feil — {e}")
                time.sleep(0.5)

    return all_data


def build_transition_table(all_data):
    """
    Bygg overgangstabell: terrain_type × distance_band × coastal → class_distribution

    Returnerer dict med nøkkel "(terrain, band, coastal)" → [count_class_0, ..., count_class_5]
    """
    # Akkumuler counts
    # Key: (initial_terrain_code, distance_band, is_coastal) → counts per class
    counts = defaultdict(lambda: np.zeros(NUM_CLASSES, dtype=float))
    total_cells = 0
    total_dynamic = 0

    for entry in all_data:
        grid = entry["initial_grid"]
        gt = entry["ground_truth"]
        settlements = entry["settlements"]

        if not grid or not gt:
            continue

        h, w = len(grid), len(grid[0])

        for y in range(h):
            for x in range(w):
                terrain = grid[y][x]
                total_cells += 1

                # Skip statiske celler (ocean og mountain teller ikke i scoring)
                if terrain in (10, 5):
                    continue

                total_dynamic += 1

                dist = distance_to_nearest_settlement(y, x, settlements)
                band = get_distance_band(dist)
                coastal = is_coastal(grid, y, x)

                # Ground truth er en probability vector [p0, p1, p2, p3, p4, p5]
                gt_cell = gt[y][x]

                if isinstance(gt_cell, list) and len(gt_cell) == NUM_CLASSES:
                    gt_probs = np.array(gt_cell, dtype=float)
                elif isinstance(gt_cell, (int, float)):
                    # Argmax format — konverter til one-hot
                    cls = TERRAIN_TO_CLASS.get(int(gt_cell), 0)
                    gt_probs = np.zeros(NUM_CLASSES)
                    gt_probs[cls] = 1.0
                else:
                    continue

                key = f"{terrain}_{band}_{int(coastal)}"
                counts[key] += gt_probs

    print(f"\nTotalt: {total_cells} celler, {total_dynamic} dynamiske")
    print(f"Unike nøkler: {len(counts)}")

    # Konverter til normalerte distribusjoner
    table = {}
    for key, c in counts.items():
        total = c.sum()
        if total > 0:
            dist = (c / total).tolist()
            table[key] = {
                "distribution": dist,
                "sample_count": int(total),
            }

    return table


def build_simple_terrain_prior(all_data):
    """
    Enklere prior: bare terrain_type → distribusjon (uten avstand/kyst).
    Nyttig som fallback.
    """
    counts = defaultdict(lambda: np.zeros(NUM_CLASSES, dtype=float))

    for entry in all_data:
        grid = entry["initial_grid"]
        gt = entry["ground_truth"]
        if not grid or not gt:
            continue

        for y in range(len(grid)):
            for x in range(len(grid[0])):
                terrain = grid[y][x]
                gt_cell = gt[y][x]

                if isinstance(gt_cell, list) and len(gt_cell) == NUM_CLASSES:
                    gt_probs = np.array(gt_cell, dtype=float)
                elif isinstance(gt_cell, (int, float)):
                    cls = TERRAIN_TO_CLASS.get(int(gt_cell), 0)
                    gt_probs = np.zeros(NUM_CLASSES)
                    gt_probs[cls] = 1.0
                else:
                    continue

                counts[str(terrain)] += gt_probs

    table = {}
    for key, c in counts.items():
        total = c.sum()
        if total > 0:
            table[key] = (c / total).tolist()

    return table


def print_table(table, simple_prior):
    """Vis overgangstabellen lesbart."""
    terrain_names = {
        "0": "Empty", "1": "Settlement", "2": "Port", "3": "Ruin",
        "4": "Forest", "5": "Mountain", "10": "Ocean", "11": "Plains"
    }
    class_names = ["Empty", "Settl", "Port", "Ruin", "Forest", "Mount"]

    print("\n" + "=" * 80)
    print("KALIBRERTE TERRAIN-PRIORS (enkel)")
    print("=" * 80)
    print(f"{'Terrain':<12} | {'Empty':>7} {'Settl':>7} {'Port':>7} {'Ruin':>7} {'Forest':>7} {'Mount':>7}")
    print("-" * 70)

    for terrain_code in ["0", "10", "11", "1", "2", "3", "4", "5"]:
        if terrain_code in simple_prior:
            dist = simple_prior[terrain_code]
            name = terrain_names.get(terrain_code, terrain_code)
            vals = " ".join(f"{v:>7.3f}" for v in dist)
            print(f"{name:<12} | {vals}")

    print("\n" + "=" * 80)
    print("AVSTANDSBASERT OVERGANGSTABELL (terrain × avstand × kyst)")
    print("=" * 80)

    # Grupper per terrain
    by_terrain = defaultdict(list)
    for key, data in sorted(table.items()):
        parts = key.split("_")
        terrain = parts[0]
        by_terrain[terrain].append((key, data))

    for terrain, entries in sorted(by_terrain.items()):
        name = terrain_names.get(terrain, terrain)
        if terrain in ("10", "5"):
            continue  # Skip statiske

        print(f"\n--- {name} (kode {terrain}) ---")
        print(f"  {'Key':<15} {'N':>5} | {'Empty':>7} {'Settl':>7} {'Port':>7} {'Ruin':>7} {'Forest':>7} {'Mount':>7}")

        for key, data in entries:
            parts = key.split("_")
            band_idx = int(parts[1])
            coastal = bool(int(parts[2]))
            label = f"b{band_idx}{'C' if coastal else ' '}"
            n = data["sample_count"]
            dist = data["distribution"]
            vals = " ".join(f"{v:>7.3f}" for v in dist)
            print(f"  {label:<15} {n:>5} | {vals}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="Vis eksisterende kalibrering")
    args = parser.parse_args()

    if args.show:
        if OUTPUT_FILE.exists():
            data = json.loads(OUTPUT_FILE.read_text())
            print_table(data.get("transition_table", {}), data.get("simple_prior", {}))
            print(f"\nData fra {data.get('num_rounds', '?')} runder, {data.get('num_seeds', '?')} seeds")
        else:
            print("Ingen kalibreringsdata funnet. Kjør uten --show først.")
        return

    if not API_KEY:
        print("FEIL: export API_KEY='din-jwt-token'")
        sys.exit(1)

    client = AstarClient()

    print("=== Henter ground truth fra alle fullførte runder ===\n")
    all_data = fetch_all_ground_truth(client)

    if not all_data:
        print("Ingen ground truth data tilgjengelig!")
        sys.exit(1)

    print(f"\n=== Bygger overgangstabeller fra {len(all_data)} seeds ===")

    transition_table = build_transition_table(all_data)
    simple_prior = build_simple_terrain_prior(all_data)

    # Lagre
    output = {
        "transition_table": transition_table,
        "simple_prior": simple_prior,
        "num_rounds": len(set(d["round_id"] for d in all_data)),
        "num_seeds": len(all_data),
        "rounds": list(set(d.get("round_number", "?") for d in all_data)),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    print(f"\nLagret til {OUTPUT_FILE}")

    # Vis
    print_table(transition_table, simple_prior)


if __name__ == "__main__":
    main()
