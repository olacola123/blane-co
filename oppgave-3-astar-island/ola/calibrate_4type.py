"""
Bygg 4-type calibration: DEAD, STABLE, BOOM_SPREAD, BOOM_CONCENTRATED.

BOOM_SPREAD: R1, R2, R6, R11 (29-38 settlements, ekspanderer langt)
BOOM_CONCENTRATED: R7, R12 (43-44 settlements, ekspanderer kort)
"""
import json, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://api.ainm.no/astar-island"
API_KEY = os.environ.get("API_KEY", "")
PROJECT_DIR = Path(__file__).parent
MAP_W, MAP_H, NUM_CLASSES = 40, 40, 6
DISTANCE_BANDS = [0, 1, 2, 3, 5, 8, 12, 99]

class Client:
    def __init__(self):
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.headers.update({"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"})
    def get(self, path):
        r = self.session.get(f"{BASE_URL}/{path.lstrip('/')}"); r.raise_for_status(); return r.json()

def dist_to_settle(y, x, settlements):
    if not settlements: return 99
    return min(max(abs(x-s["x"]), abs(y-s["y"])) for s in settlements)

def get_band(dist):
    for i in range(len(DISTANCE_BANDS)-1):
        if dist <= DISTANCE_BANDS[i]: return i
    return len(DISTANCE_BANDS)-2

def is_coastal(grid, y, x):
    for dy in [-1,0,1]:
        for dx in [-1,0,1]:
            ny, nx = y+dy, x+dx
            if 0<=ny<MAP_H and 0<=nx<MAP_W and grid[ny][nx] in (10,5):
                return True
    return False

def classify_round(client, round_id, seeds_data):
    """Klassifiser med settlement survival + antall settlements."""
    total_prob, total_cells = 0, 0
    n_settlements = len(seeds_data[0].get("settlements", []))

    for si in range(min(2, len(seeds_data))):
        try:
            analysis = client.get(f"/analysis/{round_id}/{si}")
            gt = analysis.get("ground_truth")
            if not gt: continue
            gt_arr = np.array(gt, dtype=float)
            for s in seeds_data[si].get("settlements", []):
                sx, sy = s["x"], s["y"]
                if 0<=sx<MAP_W and 0<=sy<MAP_H:
                    total_prob += gt_arr[sy, sx, 1]
                    total_cells += 1
            time.sleep(0.2)
        except: continue

    if total_cells == 0: return "STABLE"
    avg = total_prob / total_cells

    if avg < 0.08:
        return "DEAD"
    elif avg < 0.35:
        return "STABLE"
    else:
        # BOOMING sub-type: concentrated if many initial settlements
        if n_settlements >= 40:
            return "BOOM_CONC"
        else:
            return "BOOM_SPREAD"

def build_tables(client, rounds_by_type, all_seeds):
    tables = {}
    for wt, rids in rounds_by_type.items():
        key_counts = defaultdict(lambda: np.zeros(NUM_CLASSES))
        key_totals = defaultdict(float)
        for rid in rids:
            seeds = all_seeds[rid]
            for si in range(len(seeds)):
                try:
                    gt = client.get(f"/analysis/{rid}/{si}").get("ground_truth")
                    if not gt: continue
                    gt_arr = np.array(gt, dtype=float)
                except: continue
                grid = seeds[si].get("grid", [])
                settlements = seeds[si].get("settlements", [])
                for y in range(MAP_H):
                    for x in range(MAP_W):
                        t = grid[y][x]
                        if t in (10, 5): continue
                        d = dist_to_settle(y, x, settlements)
                        b = get_band(d)
                        c = is_coastal(grid, y, x)
                        key = f"{t}_{b}_{int(c)}"
                        if gt_arr[y,x].sum() > 0:
                            key_counts[key] += gt_arr[y,x]
                            key_totals[key] += 1.0
                time.sleep(0.15)
                print(f"  {wt}: {rid[:8]} seed {si}")

        table = {}
        for key, counts in key_counts.items():
            tot = key_totals[key]
            if tot >= 1:
                d = counts / tot; d /= d.sum()
                table[key] = {"distribution": d.tolist(), "sample_count": int(tot)}
        tables[wt] = table
        print(f"\n{wt}: {len(table)} keys fra {len(rids)} runder\n")
    return tables

def main():
    if not API_KEY: print("FEIL: API_KEY"); sys.exit(1)
    client = Client()
    rounds = client.get("/rounds")
    completed = sorted([r for r in rounds if isinstance(r,dict) and r.get("status")=="completed"],
                       key=lambda r: r.get("round_number",0))
    print(f"=== 4-TYPE CALIBRATION: {len(completed)} runder ===\n")

    all_seeds = {}
    rounds_by_type = defaultdict(list)
    for r in completed:
        rid, rnum = r["id"], r.get("round_number","?")
        rd = client.get(f"/rounds/{rid}")
        seeds = rd.get("seeds", rd.get("initial_states", []))
        all_seeds[rid] = seeds
        wt = classify_round(client, rid, seeds)
        rounds_by_type[wt].append(rid)
        n_s = len(seeds[0].get("settlements",[])) if seeds else 0
        print(f"Runde {rnum}: {wt} ({n_s} settlements)")
        time.sleep(0.2)

    for wt, rids in rounds_by_type.items():
        print(f"\n{wt}: {len(rids)} runder")

    print("\n=== Bygger tabeller ===\n")
    tables = build_tables(client, rounds_by_type, all_seeds)

    output = {
        "world_types": {wt: [rid[:12] for rid in rids] for wt, rids in rounds_by_type.items()},
        "tables": tables,
        "num_types": len(tables),
    }
    out = PROJECT_DIR / "calibration_4type.json"
    out.write_text(json.dumps(output, indent=2))
    print(f"\nLagret til {out}")

    for wt, table in tables.items():
        print(f"\n{wt} ({len(table)} keys):")
        for key in sorted(table.keys())[:3]:
            d = table[key]["distribution"]
            print(f"  {key}: settle={d[1]:.3f} forest={d[4]:.3f} empty={d[0]:.3f}")

if __name__ == "__main__":
    main()
