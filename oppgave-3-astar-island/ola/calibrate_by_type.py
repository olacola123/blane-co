"""
Bygg type-spesifikke calibration-tabeller fra ground truth.

I stedet for én tabell for alle runder, lager vi 3:
- DEAD: runder der settlements kollapser (R3, R8, R10)
- STABLE: moderate runder (R4, R5, R9, R13)
- BOOMING: settlements ekspanderer aggressivt (R1, R2, R6, R7, R11, R12)

Bruk:
    export API_KEY='...'
    python calibrate_by_type.py
"""
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
PROJECT_DIR = Path(__file__).parent

MAP_W, MAP_H = 40, 40
NUM_CLASSES = 6
TERRAIN_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}
DISTANCE_BANDS = [0, 1, 2, 3, 5, 8, 12, 99]


class Client:
    def __init__(self):
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.headers.update({
            "Authorization": f"Bearer {API_KEY}", "Accept": "application/json",
        })

    def get(self, path):
        r = self.session.get(f"{BASE_URL}/{path.lstrip('/')}")
        r.raise_for_status()
        return r.json()


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
                if grid[ny][nx] in (10, 5):
                    return True
    return False


def classify_round(client, round_id, seeds_data):
    """Klassifiser runde som DEAD, STABLE eller BOOMING basert på ground truth."""
    total_settle_prob = 0
    total_cells = 0

    for seed_idx in range(min(2, len(seeds_data))):  # Sjekk 2 seeds
        try:
            analysis = client.get(f"/analysis/{round_id}/{seed_idx}")
            gt = analysis.get("ground_truth")
            if not gt:
                continue
            gt_arr = np.array(gt, dtype=float)

            # Sjekk settlement-prob på settlement-celler
            grid = seeds_data[seed_idx].get("grid", [])
            settlements = seeds_data[seed_idx].get("settlements", [])
            for s in settlements:
                sx, sy = s["x"], s["y"]
                if 0 <= sx < MAP_W and 0 <= sy < MAP_H:
                    total_settle_prob += gt_arr[sy, sx, 1]
                    total_cells += 1

            time.sleep(0.2)
        except Exception:
            continue

    if total_cells == 0:
        return "STABLE"

    avg_settle = total_settle_prob / total_cells

    if avg_settle < 0.08:
        return "DEAD"
    elif avg_settle < 0.35:
        return "STABLE"
    else:
        return "BOOMING"


def build_transition_table(client, rounds_by_type, all_seeds_data):
    """Bygg en transition table per world type."""
    tables = {}

    for world_type, round_ids in rounds_by_type.items():
        key_counts = defaultdict(lambda: np.zeros(NUM_CLASSES, dtype=float))
        key_totals = defaultdict(float)

        for round_id in round_ids:
            seeds_data = all_seeds_data[round_id]

            for seed_idx in range(len(seeds_data)):
                try:
                    analysis = client.get(f"/analysis/{round_id}/{seed_idx}")
                    gt = analysis.get("ground_truth")
                    if not gt:
                        continue
                    gt_arr = np.array(gt, dtype=float)
                except Exception:
                    continue

                grid = seeds_data[seed_idx].get("grid", [])
                settlements = seeds_data[seed_idx].get("settlements", [])

                for y in range(MAP_H):
                    for x in range(MAP_W):
                        terrain = grid[y][x]
                        if terrain in (10, 5):  # Skip static
                            continue

                        dist = distance_to_nearest_settlement(y, x, settlements)
                        band = get_distance_band(dist)
                        coastal = is_coastal(grid, y, x)

                        key = f"{terrain}_{band}_{int(coastal)}"
                        gt_dist = gt_arr[y, x]

                        if gt_dist.sum() > 0:
                            key_counts[key] += gt_dist
                            key_totals[key] += 1.0

                time.sleep(0.2)
                print(f"  {world_type}: runde {round_id[:8]} seed {seed_idx} OK")

        # Normaliser
        table = {}
        for key, counts in key_counts.items():
            total = key_totals[key]
            if total >= 1:
                dist = counts / total
                dist = dist / dist.sum()  # Renormaliser
                table[key] = {
                    "distribution": dist.tolist(),
                    "sample_count": int(total),
                }

        tables[world_type] = table
        print(f"\n{world_type}: {len(table)} keys fra {len(round_ids)} runder\n")

    return tables


def main():
    if not API_KEY:
        print("FEIL: export API_KEY='...'")
        sys.exit(1)

    client = Client()
    rounds = client.get("/rounds")
    completed = [r for r in rounds if isinstance(r, dict) and r.get("status") == "completed"]
    completed.sort(key=lambda r: r.get("round_number", 0))

    print(f"=== CALIBRATE BY TYPE: {len(completed)} runder ===\n")

    # Hent all seed data og klassifiser
    all_seeds_data = {}
    rounds_by_type = defaultdict(list)

    for r in completed:
        round_id = r["id"]
        rnum = r.get("round_number", "?")

        round_data = client.get(f"/rounds/{round_id}")
        seeds_data = round_data.get("seeds", round_data.get("initial_states", []))
        all_seeds_data[round_id] = seeds_data

        world_type = classify_round(client, round_id, seeds_data)
        rounds_by_type[world_type].append(round_id)
        print(f"Runde {rnum}: {world_type}")

        time.sleep(0.3)

    print(f"\nDEAD: {len(rounds_by_type['DEAD'])} runder")
    print(f"STABLE: {len(rounds_by_type['STABLE'])} runder")
    print(f"BOOMING: {len(rounds_by_type['BOOMING'])} runder")

    # Bygg tabeller
    print("\n=== Bygger type-spesifikke tabeller ===\n")
    tables = build_transition_table(client, rounds_by_type, all_seeds_data)

    # Lagre
    output = {
        "world_types": {wt: [rid[:12] for rid in rids] for wt, rids in rounds_by_type.items()},
        "tables": tables,
        "num_rounds": len(completed),
    }

    out_file = PROJECT_DIR / "calibration_by_type.json"
    out_file.write_text(json.dumps(output, indent=2))
    print(f"\nLagret til {out_file}")

    # Vis statistikk
    for wt, table in tables.items():
        print(f"\n{wt} ({len(table)} keys):")
        for key in sorted(table.keys())[:5]:
            d = table[key]["distribution"]
            n = table[key]["sample_count"]
            print(f"  {key}: n={n:>5}  empty={d[0]:.3f} settle={d[1]:.3f} port={d[2]:.3f} "
                  f"ruin={d[3]:.3f} forest={d[4]:.3f}")


if __name__ == "__main__":
    main()
