"""
Manhattan-Distance Calibration Builder
========================================
Rebuilds ALL calibration tables using Manhattan distance (|dx|+|dy|) instead of
Chebyshev (max(|dx|,|dy|)).

Produces:
  - calibration_manhattan.json       (basic: terrain × dist_band × coastal)
  - calibration_manhattan_4type.json (4-type: DEAD/STABLE/BOOM_SPREAD/BOOM_CONC)
  - calibration_manhattan_opt.json   (optimized: wtype × terrain_group × dist_band × coastal)

These are consumed by solution_diamond.py.

Bruk:
    export API_KEY='din-jwt-token'
    python calibrate_manhattan.py           # Bygg alle tabeller
    python calibrate_manhattan.py --show    # Vis eksisterende tabeller
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

PROJECT_DIR = Path(__file__).parent

# Manhattan distance bands — matching the bins in solution_diamond.py
# These are chosen so each band contains a comparable "ring area" as Chebyshev bands
# Manhattan d=0 → 1 cell, d=1 → 4 cells, d=2 → 8 cells, d=3 → 12 cells, etc.
DISTANCE_BANDS = [0, 1, 2, 3, 5, 8, 12, 99]
BAND_LABELS = ["d=0", "d=1", "d=2", "d=3", "d=4-5", "d=6-8", "d=9-12", "d=13+"]

# Optimized distance bands (finer resolution)
OPT_DIST_BANDS = [(0, 0), (1, 2), (3, 3), (4, 5), (6, 8), (9, 12), (13, 99)]

TERRAIN_GROUP = {
    0: "plains", 1: "settlement", 2: "port", 3: "ruin",
    4: "forest", 5: "mountain", 10: "ocean", 11: "plains"
}


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

    def get(self, path):
        r = self.session.get(f"{BASE_URL}/{path.lstrip('/')}", timeout=30)
        r.raise_for_status()
        return r.json()


# === MANHATTAN DISTANCE ===

def manhattan_dist_to_nearest(y, x, settlements):
    """Manhattan distance to nearest settlement."""
    if not settlements:
        return 99
    return min(abs(x - s["x"]) + abs(y - s["y"]) for s in settlements)


def get_band(dist):
    """Convert Manhattan distance to band index."""
    for i in range(len(DISTANCE_BANDS) - 1):
        if dist <= DISTANCE_BANDS[i]:
            return i
    return len(DISTANCE_BANDS) - 2


def get_opt_band(dist):
    """Convert Manhattan distance to optimized band index."""
    for i, (lo, hi) in enumerate(OPT_DIST_BANDS):
        if lo <= dist <= hi:
            return i
    return len(OPT_DIST_BANDS) - 1


def is_coastal(grid, y, x):
    """Check if cell borders ocean (terrain 10)."""
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < MAP_H and 0 <= nx < MAP_W:
                if grid[ny][nx] == 10:
                    return True
    return False


# === DATA FETCHING ===

def fetch_all_ground_truth(client):
    """Fetch ground truth from all completed rounds."""
    rounds = client.get("/rounds")
    completed = sorted(
        [r for r in rounds if isinstance(r, dict) and r.get("status") == "completed"],
        key=lambda r: r.get("round_number", 0)
    )
    print(f"Fant {len(completed)} fullførte runder")

    all_data = []
    for round_info in completed:
        round_id = round_info["id"]
        round_num = round_info.get("round_number", "?")
        print(f"\nRunde {round_num} ({round_id[:12]}...):")

        try:
            round_data = client.get(f"/rounds/{round_id}")
        except Exception as e:
            print(f"  Feil: {e}")
            continue

        seeds = round_data.get("seeds", round_data.get("initial_states", []))
        if not seeds:
            print("  Ingen seeds")
            continue

        for seed_idx in range(len(seeds)):
            try:
                analysis = client.get(f"/analysis/{round_id}/{seed_idx}")
                gt = analysis.get("ground_truth")
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
                })
                print(f"  Seed {seed_idx}: OK (gt {len(gt)}×{len(gt[0]) if gt else 0})")
                time.sleep(0.25)
            except Exception as e:
                print(f"  Seed {seed_idx}: feil — {e}")
                time.sleep(0.5)

    return all_data


# === ROUND CLASSIFICATION ===

def compute_gt_expansion(gt_data_for_round):
    """Compute ground truth expansion metrics from GT probability tensors.

    Returns: (avg_vitality, avg_concentration_ratio, avg_frac_beyond_8)
    - concentration_ratio: fraction of settlement probability within dist<=5 of initial settlements
    - frac_beyond_8: fraction of settlement probability beyond dist 8 from initial settlements
    """
    vitalities = []
    concentration_ratios = []
    frac_beyond_8s = []

    for entry in gt_data_for_round:
        gt = np.array(entry["ground_truth"], dtype=float)
        grid = np.array(entry["initial_grid"], dtype=int)
        settlements = entry["settlements"]

        # Vitality: average settlement survival probability at initial positions
        total_prob, total_cells = 0, 0
        for s in settlements:
            sx, sy = s["x"], s["y"]
            if 0 <= sx < MAP_W and 0 <= sy < MAP_H:
                total_prob += gt[sy, sx, 1]
                total_cells += 1
        vitalities.append(total_prob / max(total_cells, 1))

        # Expansion metrics: how far settlement probability spreads from initial positions
        dist_map = np.full((MAP_H, MAP_W), 99.0)
        for s in settlements:
            sy, sx = s["y"], s["x"]
            for y in range(MAP_H):
                for x in range(MAP_W):
                    d = abs(y - sy) + abs(x - sx)
                    if d < dist_map[y, x]:
                        dist_map[y, x] = d

        static_mask = (grid == 10) | (grid == 5)
        dynamic_mask = ~static_mask
        settle_prob = (gt[:, :, 1] + gt[:, :, 2]) * dynamic_mask
        total_settle = settle_prob.sum()

        if total_settle > 0.01:
            close_mask = (dist_map <= 5) & dynamic_mask
            concentration = settle_prob[close_mask].sum() / total_settle
            far_mask = (dist_map > 8) & dynamic_mask
            frac_far = settle_prob[far_mask].sum() / total_settle
            concentration_ratios.append(float(concentration))
            frac_beyond_8s.append(float(frac_far))
        else:
            concentration_ratios.append(1.0)  # dead = fully concentrated (doesn't matter)
            frac_beyond_8s.append(0.0)

    return (
        float(np.mean(vitalities)),
        float(np.mean(concentration_ratios)),
        float(np.mean(frac_beyond_8s)),
    )


def classify_round(gt_data_for_round, seeds_data):
    """Classify round type from ground truth using actual expansion metrics."""
    avg_vitality, avg_concentration, avg_frac_beyond = compute_gt_expansion(gt_data_for_round)

    if avg_vitality < 0.08:
        return "DEAD"
    elif avg_vitality < 0.35:
        return "STABLE"
    else:
        # Use actual expansion metrics instead of n_settlements heuristic
        # High concentration_ratio + low frac_beyond = CONCENTRATED growth
        # Low concentration_ratio + high frac_beyond = SPREAD growth
        if avg_concentration > 0.55 and avg_frac_beyond < 0.15:
            return "BOOM_CONC"
        elif avg_concentration < 0.45 or avg_frac_beyond > 0.20:
            return "BOOM_SPREAD"
        else:
            return "BOOM_CONC"  # default to concentrated


# === TABLE BUILDERS ===

def build_basic_table(all_data):
    """Build basic transition table: terrain × dist_band × coastal → distribution."""
    counts = defaultdict(lambda: np.zeros(NUM_CLASSES, dtype=float))
    totals = defaultdict(float)

    for entry in all_data:
        grid = entry["initial_grid"]
        gt = entry["ground_truth"]
        settlements = entry["settlements"]
        if not grid or not gt:
            continue

        for y in range(MAP_H):
            for x in range(MAP_W):
                terrain = grid[y][x]
                if terrain in (10, 5):
                    continue

                dist = manhattan_dist_to_nearest(y, x, settlements)
                band = get_band(dist)
                coastal = is_coastal(grid, y, x)

                gt_cell = gt[y][x]
                if isinstance(gt_cell, list) and len(gt_cell) == NUM_CLASSES:
                    gt_probs = np.array(gt_cell, dtype=float)
                elif isinstance(gt_cell, (int, float)):
                    gt_probs = np.zeros(NUM_CLASSES)
                    gt_probs[int(gt_cell)] = 1.0
                else:
                    continue

                key = f"{terrain}_{band}_{int(coastal)}"
                counts[key] += gt_probs
                totals[key] += 1.0

    table = {}
    for key, c in counts.items():
        tot = totals[key]
        if tot >= 1:
            d = c / tot
            d /= d.sum()
            table[key] = {"distribution": d.tolist(), "sample_count": int(tot)}

    # Simple prior (terrain only)
    terrain_counts = defaultdict(lambda: np.zeros(NUM_CLASSES, dtype=float))
    terrain_totals = defaultdict(float)
    for key, c in counts.items():
        terrain = key.split("_")[0]
        terrain_counts[terrain] += c
        terrain_totals[terrain] += totals[key]

    simple_prior = {}
    for t, c in terrain_counts.items():
        tot = terrain_totals[t]
        if tot >= 1:
            d = c / tot
            d /= d.sum()
            simple_prior[t] = d.tolist()

    return table, simple_prior


def build_4type_tables(all_data):
    """Build 4-type calibration tables (DEAD/STABLE/BOOM_SPREAD/BOOM_CONC)."""
    # Group data by round
    by_round = defaultdict(list)
    for entry in all_data:
        by_round[entry["round_id"]].append(entry)

    # Classify each round
    rounds_by_type = defaultdict(list)
    round_seeds = {}  # round_id → seeds_data (for logging)
    for round_id, entries in by_round.items():
        seeds_data = [{"settlements": e["settlements"], "grid": e["initial_grid"]} for e in entries]
        wtype = classify_round(entries, seeds_data)
        rounds_by_type[wtype].append(round_id)
        round_seeds[round_id] = seeds_data
        n_s = len(entries[0]["settlements"]) if entries else 0
        print(f"  Runde {entries[0]['round_number']}: {wtype} ({n_s} settlements)")

    # Build per-type tables
    tables = {}
    for wtype, rids in rounds_by_type.items():
        key_counts = defaultdict(lambda: np.zeros(NUM_CLASSES, dtype=float))
        key_totals = defaultdict(float)

        for rid in rids:
            for entry in by_round[rid]:
                grid = entry["initial_grid"]
                gt = entry["ground_truth"]
                settlements = entry["settlements"]
                if not grid or not gt:
                    continue
                gt_arr = np.array(gt, dtype=float)

                for y in range(MAP_H):
                    for x in range(MAP_W):
                        t = grid[y][x]
                        if t in (10, 5):
                            continue
                        d = manhattan_dist_to_nearest(y, x, settlements)
                        b = get_band(d)
                        c = is_coastal(grid, y, x)
                        key = f"{t}_{b}_{int(c)}"
                        if gt_arr[y, x].sum() > 0:
                            key_counts[key] += gt_arr[y, x]
                            key_totals[key] += 1.0

        table = {}
        for key, c in key_counts.items():
            tot = key_totals[key]
            if tot >= 1:
                d = c / tot
                d /= d.sum()
                table[key] = {"distribution": d.tolist(), "sample_count": int(tot)}
        tables[wtype] = table
        print(f"  {wtype}: {len(table)} keys fra {len(rids)} runder")

    return tables, rounds_by_type


def build_optimized_tables(all_data):
    """Build optimized tables: wtype × terrain_group × opt_dist_band × coastal → distribution.

    Keeps BOOM_CONC and BOOM_SPREAD separate to capture different distance decay patterns.
    Also builds a merged BOOM table as fallback for sparse expansion-specific entries.
    """
    # Group by round, classify
    by_round = defaultdict(list)
    for entry in all_data:
        by_round[entry["round_id"]].append(entry)

    rounds_by_type = defaultdict(list)
    expansion_metrics = {}  # round_id → (vitality, concentration, frac_beyond)
    for round_id, entries in by_round.items():
        seeds_data = [{"settlements": e["settlements"], "grid": e["initial_grid"]} for e in entries]
        metrics = compute_gt_expansion(entries)
        expansion_metrics[round_id] = metrics
        wtype = classify_round(entries, seeds_data)
        rounds_by_type[wtype].append(round_id)
        n_s = len(entries[0]["settlements"]) if entries else 0
        print(f"  Runde {entries[0]['round_number']}: {wtype} "
              f"(vitality={metrics[0]:.3f}, conc={metrics[1]:.3f}, beyond8={metrics[2]:.3f}, {n_s} sett)")

    # Also build merged BOOM for fallback
    boom_rids = rounds_by_type.get("BOOM_CONC", []) + rounds_by_type.get("BOOM_SPREAD", [])
    if boom_rids:
        rounds_by_type["BOOM"] = boom_rids

    # Build tables for each world type + ALL
    all_types = list(rounds_by_type.keys()) + ["ALL"]
    tables = {}

    for wtype in all_types:
        if wtype == "ALL":
            rids = [rid for rids in rounds_by_type.values() for rid in rids]
        else:
            rids = rounds_by_type[wtype]

        key_counts = defaultdict(lambda: np.zeros(NUM_CLASSES, dtype=float))
        key_totals = defaultdict(float)

        for rid in rids:
            for entry in by_round[rid]:
                grid = entry["initial_grid"]
                gt = entry["ground_truth"]
                settlements = entry["settlements"]
                if not grid or not gt:
                    continue
                gt_arr = np.array(gt, dtype=float)

                for y in range(MAP_H):
                    for x in range(MAP_W):
                        terrain = grid[y][x]
                        if terrain in (10, 5):
                            continue

                        tg = TERRAIN_GROUP.get(terrain, "plains")
                        d = manhattan_dist_to_nearest(y, x, settlements)
                        ob = get_opt_band(d)
                        coastal = is_coastal(grid, y, x)

                        # Most specific: wtype_terrain_band_coastal
                        key_spec = f"{wtype}_{tg}_{ob}_{int(coastal)}"
                        key_any = f"{wtype}_{tg}_{ob}_any"

                        if gt_arr[y, x].sum() > 0:
                            key_counts[key_spec] += gt_arr[y, x]
                            key_totals[key_spec] += 1.0
                            key_counts[key_any] += gt_arr[y, x]
                            key_totals[key_any] += 1.0

        for key, c in key_counts.items():
            tot = key_totals[key]
            if tot >= 5:  # minimum samples
                d = c / tot
                d /= d.sum()
                tables[key] = {"distribution": d.tolist(), "count": int(tot)}

    print(f"  Optimized: {len(tables)} entries")
    return tables, rounds_by_type


# === MAIN ===

def main():
    if not API_KEY:
        print("FEIL: export API_KEY='din-jwt-token'")
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="Vis eksisterende tabeller")
    args = parser.parse_args()

    if args.show:
        for fname in ["calibration_manhattan.json", "calibration_manhattan_4type.json", "calibration_manhattan_opt.json"]:
            p = PROJECT_DIR / fname
            if p.exists():
                d = json.loads(p.read_text())
                if "transition_table" in d:
                    print(f"\n{fname}: {len(d['transition_table'])} keys")
                elif "tables" in d:
                    if isinstance(d["tables"], dict) and all(isinstance(v, dict) and "distribution" in v for v in list(d["tables"].values())[:1]):
                        print(f"\n{fname}: {len(d['tables'])} entries")
                    else:
                        for wt, tbl in d["tables"].items():
                            print(f"\n{fname} [{wt}]: {len(tbl)} keys")
            else:
                print(f"\n{fname}: finnes ikke")
        return

    client = AstarClient()

    print("=== MANHATTAN CALIBRATION BUILDER ===\n")
    print("Henter ground truth fra alle fullførte runder...\n")
    all_data = fetch_all_ground_truth(client)
    print(f"\nTotalt: {len(all_data)} seed-datasett\n")

    if not all_data:
        print("Ingen data! Sjekk API_KEY.")
        sys.exit(1)

    # 1. Basic table
    print("=== Bygger basic Manhattan-tabell ===")
    basic_table, simple_prior = build_basic_table(all_data)
    basic_out = {
        "description": "Manhattan-distance calibration tables",
        "distance_metric": "manhattan",
        "distance_bands": DISTANCE_BANDS,
        "band_labels": BAND_LABELS,
        "num_rounds": len(set(e["round_id"] for e in all_data)),
        "num_seeds": len(all_data),
        "transition_table": basic_table,
        "simple_prior": simple_prior,
    }
    p = PROJECT_DIR / "calibration_manhattan.json"
    p.write_text(json.dumps(basic_out, indent=2))
    print(f"  → {p} ({len(basic_table)} keys)\n")

    # 2. 4-type table
    print("=== Bygger 4-type Manhattan-tabell ===")
    type_tables, rounds_by_type = build_4type_tables(all_data)
    type_out = {
        "description": "4-type Manhattan-distance calibration",
        "distance_metric": "manhattan",
        "world_types": {wt: [rid[:12] for rid in rids] for wt, rids in rounds_by_type.items()},
        "tables": type_tables,
    }
    p = PROJECT_DIR / "calibration_manhattan_4type.json"
    p.write_text(json.dumps(type_out, indent=2))
    print(f"  → {p}\n")

    # 3. Optimized table
    print("=== Bygger optimized Manhattan-tabell ===")
    opt_tables, opt_rounds = build_optimized_tables(all_data)
    opt_out = {
        "description": "Optimized Manhattan-distance prior tables",
        "distance_metric": "manhattan",
        "dist_bands": OPT_DIST_BANDS,
        "terrain_groups": list(set(TERRAIN_GROUP.values()) - {"mountain", "ocean"}),
        "world_types": sorted(set(opt_rounds.keys()) | {"ALL"}),
        "class_order": ["empty", "settlement", "port", "ruin", "forest", "mountain"],
        "num_rounds": len(set(e["round_id"] for e in all_data)),
        "num_seeds": len(all_data),
        "tables": opt_tables,
    }
    p = PROJECT_DIR / "calibration_manhattan_opt.json"
    p.write_text(json.dumps(opt_out, indent=2))
    print(f"  → {p}\n")

    # Summary
    print("\n=== FERDIG ===")
    print(f"Basic:     {len(basic_table)} keys")
    for wt, tbl in type_tables.items():
        print(f"4-type {wt}: {len(tbl)} keys")
    print(f"Optimized: {len(opt_tables)} entries")

    # Show some sample distributions for sanity check
    print("\n=== Stikkprøve (plains nær settlement) ===")
    for key in sorted(basic_table.keys()):
        if key.startswith("11_0_") or key.startswith("11_1_"):
            d = basic_table[key]["distribution"]
            n = basic_table[key]["sample_count"]
            print(f"  {key}: empty={d[0]:.3f} settle={d[1]:.3f} port={d[2]:.3f} "
                  f"ruin={d[3]:.3f} forest={d[4]:.3f} (n={n})")


if __name__ == "__main__":
    main()
