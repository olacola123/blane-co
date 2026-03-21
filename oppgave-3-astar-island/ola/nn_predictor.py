"""
Nearest-Neighbor Round Matching for Astar Island Predictions.
=============================================================
Instead of using calibration tables averaged over ALL rounds,
find the 2-3 most SIMILAR historical rounds and use only their data.

Leave-one-out CV to evaluate vs. baseline (82.2 avg).

Usage:
    export API_KEY='...'
    python nn_predictor.py
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
API_KEY = os.environ.get("API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZDY4OWRmZC01NGM0LTQwZmYtYTM2My01MzMyYjc0ZDY4M2EiLCJlbWFpbCI6Im9sYWd1ZGJyYW5kQGdtYWlsLmNvbSIsImlzX2FkbWluIjpmYWxzZSwiZXhwIjoxNzc0NTYyNTMzfQ.zEUXW0mk5hfMuTTtXu5EwF9m1Ex6vh6tOUYRMnNvs7c")
PROJECT_DIR = Path(__file__).parent
MAP_W, MAP_H, NUM_CLASSES = 40, 40, 6

# Floor per vitality bin
VBIN_FLOOR = {"DEAD": 0.001, "LOW": 0.002, "MED": 0.003, "HIGH": 0.003}
VBIN_BOUNDARIES = [(0.08, "DEAD"), (0.25, "LOW"), (0.45, "MED")]


# ── API Client ──────────────────────────────────────────────────────────────

class Client:
    def __init__(self):
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.headers.update({
            "Authorization": f"Bearer {API_KEY}",
            "Accept": "application/json",
        })

    def get(self, path):
        r = self.session.get(f"{BASE_URL}/{path.lstrip('/')}")
        r.raise_for_status()
        return r.json()


# ── Feature helpers ─────────────────────────────────────────────────────────

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


def chebyshev_dist(y1, x1, y2, x2):
    return max(abs(y1 - y2), abs(x1 - x2))


# ── Round fingerprint ──────────────────────────────────────────────────────

def compute_round_fingerprint(grid, settlements, gt=None):
    """
    Compute a feature vector describing a round's characteristics.
    Uses only initial state (grid + settlements), plus vitality from GT if available.
    """
    grid_arr = np.array(grid, dtype=int)
    H, W = grid_arr.shape

    # Count terrain types
    n_land = np.sum((grid_arr != 10) & (grid_arr != 5))
    n_forest = np.sum(grid_arr == 4)
    n_settlements = len(settlements)
    n_ports = sum(1 for s in settlements if s.get("has_port", False) or grid_arr[s["y"], s["x"]] == 2)
    n_ruins = np.sum(grid_arr == 3)

    # Settlement density
    settlement_density = n_settlements / max(n_land, 1)

    # Forest ratio
    forest_ratio = n_forest / max(n_land, 1)

    # Port fraction
    port_fraction = n_ports / max(n_settlements, 1)

    # Avg pairwise distance between settlements (Chebyshev)
    coords = [(s["y"], s["x"]) for s in settlements]
    if len(coords) >= 2:
        dists = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dists.append(chebyshev_dist(*coords[i], *coords[j]))
        avg_settle_dist = np.mean(dists)
    else:
        avg_settle_dist = 0.0

    # Settlement clustering: std of pairwise distances (low = clustered, high = spread)
    if len(coords) >= 3:
        settle_clustering = np.std(dists)
    else:
        settle_clustering = 0.0

    # Coastal settlements (fraction next to ocean)
    coastal_count = 0
    for s in settlements:
        sy, sx = s["y"], s["x"]
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = sy + dy, sx + dx
                if 0 <= ny < H and 0 <= nx < W and grid_arr[ny, nx] == 10:
                    coastal_count += 1
                    break
            else:
                continue
            break
    coastal_fraction = coastal_count / max(n_settlements, 1)

    # Vitality from ground truth (if available)
    vitality = 0.35  # default
    if gt is not None:
        gt_arr = np.array(gt, dtype=float)
        total_s, survived = 0, 0.0
        for s in settlements:
            sx, sy = s["x"], s["y"]
            if 0 <= sx < W and 0 <= sy < H:
                total_s += 1
                survived += gt_arr[sy, sx, 1] + gt_arr[sy, sx, 2]
        if total_s > 0:
            vitality = survived / total_s

    fp = {
        "vitality": vitality,
        "n_settlements": n_settlements,
        "n_ports": n_ports,
        "settlement_density": settlement_density,
        "forest_ratio": forest_ratio,
        "n_ruins": n_ruins,
        "avg_settle_dist": avg_settle_dist,
        "settle_clustering": settle_clustering,
        "port_fraction": port_fraction,
        "coastal_fraction": coastal_fraction,
    }
    return fp


def fp_to_vector(fp):
    """Convert fingerprint dict to numpy vector for distance computation."""
    return np.array([
        fp["vitality"],
        fp["n_settlements"],
        fp["n_ports"],
        fp["settlement_density"],
        fp["forest_ratio"],
        fp["n_ruins"],
        fp["avg_settle_dist"],
        fp["settle_clustering"],
        fp["port_fraction"],
        fp["coastal_fraction"],
    ], dtype=float)


# ── Data fetching ──────────────────────────────────────────────────────────

def fetch_all_rounds(client):
    """Fetch all completed rounds with initial states and ground truth."""
    rounds = client.get("/rounds")
    completed = sorted(
        [r for r in rounds if isinstance(r, dict) and r.get("status") == "completed"],
        key=lambda r: r.get("round_number", 0)
    )
    print(f"Fant {len(completed)} fullførte runder\n")

    round_data = []
    for r in completed:
        rid = r["id"]
        rnum = r.get("round_number", "?")
        rd = client.get(f"/rounds/{rid}")
        seeds = rd.get("seeds", rd.get("initial_states", []))
        if not seeds:
            print(f"  R{rnum}: ingen seeds, skipper")
            continue

        seed_entries = []
        for si in range(len(seeds)):
            try:
                analysis = client.get(f"/analysis/{rid}/{si}")
                gt = analysis.get("ground_truth")
                if gt is None:
                    continue
                seed_entries.append({
                    "seed_index": si,
                    "grid": seeds[si].get("grid", []),
                    "settlements": seeds[si].get("settlements", []),
                    "ground_truth": gt,
                })
                time.sleep(0.15)
            except Exception as e:
                print(f"  R{rnum} seed {si}: {e}")
                time.sleep(0.3)

        if seed_entries:
            round_data.append({
                "round_id": rid,
                "round_number": rnum,
                "seeds": seed_entries,
            })
            print(f"  R{rnum}: {len(seed_entries)} seeds hentet")

    return round_data


# ── Nearest neighbor lookup ────────────────────────────────────────────────

def compute_round_fingerprints(round_data):
    """Compute fingerprint per round (averaged over seeds for vitality)."""
    fingerprints = {}
    for rd in round_data:
        rnum = rd["round_number"]
        # Use seed 0 for grid features, average vitality over all seeds
        seed0 = rd["seeds"][0]
        fp = compute_round_fingerprint(
            seed0["grid"], seed0["settlements"], seed0["ground_truth"]
        )
        # Average vitality over all seeds
        vits = []
        for s in rd["seeds"]:
            fp_s = compute_round_fingerprint(s["grid"], s["settlements"], s["ground_truth"])
            vits.append(fp_s["vitality"])
        fp["vitality"] = np.mean(vits)
        fingerprints[rnum] = fp
    return fingerprints


def find_nearest_rounds(target_fp, all_fps, exclude_rnum=None, k=3):
    """
    Find k nearest rounds by euclidean distance on z-score normalized fingerprints.
    Returns list of (round_number, distance, weight).
    """
    # Collect all vectors for normalization
    all_rnums = [rn for rn in all_fps if rn != exclude_rnum]
    if not all_rnums:
        return []

    vectors = np.array([fp_to_vector(all_fps[rn]) for rn in all_rnums])
    target_vec = fp_to_vector(target_fp)

    # Z-score normalize
    mean = vectors.mean(axis=0)
    std = vectors.std(axis=0)
    std[std < 1e-10] = 1.0  # avoid div by zero

    norm_vectors = (vectors - mean) / std
    norm_target = (target_vec - mean) / std

    # Euclidean distances
    dists = np.sqrt(np.sum((norm_vectors - norm_target) ** 2, axis=1))

    # Sort by distance
    order = np.argsort(dists)
    results = []
    for i in order[:k]:
        rn = all_rnums[i]
        d = dists[i]
        # Weight: inverse distance (with smoothing)
        w = 1.0 / (d + 0.1)
        results.append((rn, float(d), float(w)))

    # Normalize weights
    total_w = sum(w for _, _, w in results)
    results = [(rn, d, w / total_w) for rn, d, w in results]
    return results


# ── Build calibration tables from specific rounds ──────────────────────────

def vitality_to_vbin(rate):
    if rate < 0.08: return "DEAD"
    elif rate < 0.25: return "LOW"
    elif rate < 0.45: return "MED"
    else: return "HIGH"


def build_nn_calibration(round_data, neighbor_rnums, neighbor_weights):
    """
    Build calibration tables using only data from the given rounds,
    weighted by similarity.

    Returns: (table_density, table_specific, table_simple)
    Same format as super_calibration.json tables.
    """
    # Map round_number → round_data entry
    rnum_to_data = {rd["round_number"]: rd for rd in round_data}

    # Accumulate weighted counts at 3 specificity levels
    counts_density = defaultdict(lambda: np.zeros(NUM_CLASSES))
    counts_specific = defaultdict(lambda: np.zeros(NUM_CLASSES))
    counts_simple = defaultdict(lambda: np.zeros(NUM_CLASSES))
    n_density = defaultdict(float)
    n_specific = defaultdict(float)
    n_simple = defaultdict(float)

    for rnum, weight in zip(neighbor_rnums, neighbor_weights):
        rd = rnum_to_data.get(rnum)
        if rd is None:
            continue

        for seed_entry in rd["seeds"]:
            grid = seed_entry["grid"]
            gt = seed_entry["ground_truth"]
            settlements = seed_entry["settlements"]
            grid_arr = np.array(grid, dtype=int)
            gt_arr = np.array(gt, dtype=float)

            # Compute vitality for this seed → determines vbin
            total_s, survived = 0, 0.0
            for s in settlements:
                sx, sy = s["x"], s["y"]
                if 0 <= sx < MAP_W and 0 <= sy < MAP_H:
                    total_s += 1
                    survived += gt_arr[sy, sx, 1] + gt_arr[sy, sx, 2]
            vbin = vitality_to_vbin(survived / total_s if total_s > 0 else 0.35)

            for y in range(MAP_H):
                for x in range(MAP_W):
                    terrain = int(grid_arr[y, x])
                    if terrain in (10, 5):
                        continue

                    gt_cell = gt_arr[y, x]
                    if gt_cell.sum() < 0.5:
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
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < MAP_H and 0 <= nx < MAP_W:
                                if grid_arr[ny, nx] == 10:
                                    coastal = True
                                    break
                        if coastal:
                            break

                    n_forest_r2 = 0
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < MAP_H and 0 <= nx < MAP_W:
                                if grid_arr[ny, nx] == 4:
                                    n_forest_r2 += 1

                    tg = _terrain_group(terrain)
                    db = _dist_bin(min_dist)
                    c = int(coastal)
                    sdb = _settle_density_bin(n_settle_r3)
                    fdb = _forest_density_bin(n_forest_r2)

                    # Three specificity levels
                    key_d = f"{vbin}_{tg}_{db}_{c}_{sdb}_{fdb}"
                    key_s = f"{vbin}_{tg}_{db}_{c}"
                    key_simple = f"{tg}_{db}"

                    counts_density[key_d] += gt_cell * weight
                    n_density[key_d] += weight

                    counts_specific[key_s] += gt_cell * weight
                    n_specific[key_s] += weight

                    counts_simple[key_simple] += gt_cell * weight
                    n_simple[key_simple] += weight

    # Normalize to distributions
    def make_table(counts, ns):
        table = {}
        for key in counts:
            c = counts[key]
            total = c.sum()
            if total > 0:
                dist = c / total
                table[key] = {
                    "distribution": dist.tolist(),
                    "sample_count": int(ns[key]),
                }
        return table

    return (
        make_table(counts_density, n_density),
        make_table(counts_specific, n_specific),
        make_table(counts_simple, n_simple),
    )


# ── Prediction ─────────────────────────────────────────────────────────────

def nn_predict(grid, settlements, vbin, table_density, table_specific, table_simple, floor=None):
    """
    Predict using NN-calibration tables. Same logic as super_predict but with
    the NN-specific tables.
    """
    if floor is None:
        floor = VBIN_FLOOR.get(vbin, 0.003)

    grid_arr = np.array(grid, dtype=int) if not isinstance(grid, np.ndarray) else grid
    H, W = grid_arr.shape
    pred = np.zeros((H, W, NUM_CLASSES), dtype=float)

    for y in range(H):
        for x in range(W):
            terrain = int(grid_arr[y, x])

            if terrain == 10:
                pred[y, x] = [1.0 - 5 * floor, floor, floor, floor, floor, floor]
                continue
            if terrain == 5:
                pred[y, x] = [floor, floor, floor, floor, floor, 1.0 - 5 * floor]
                continue

            # Compute features
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
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        if grid_arr[ny, nx] == 10:
                            coastal = True
                            break
                if coastal:
                    break

            n_forest_r2 = 0
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        if grid_arr[ny, nx] == 4:
                            n_forest_r2 += 1

            tg = _terrain_group(terrain)
            db = _dist_bin(min_dist)
            c = int(coastal)
            sdb = _settle_density_bin(n_settle_r3)
            fdb = _forest_density_bin(n_forest_r2)

            # Cascading lookup
            p = None

            key_d = f"{vbin}_{tg}_{db}_{c}_{sdb}_{fdb}"
            if key_d in table_density:
                p = np.array(table_density[key_d]["distribution"])

            if p is None:
                key_s = f"{vbin}_{tg}_{db}_{c}"
                if key_s in table_specific:
                    p = np.array(table_specific[key_s]["distribution"])

            if p is None:
                key_simple = f"{tg}_{db}"
                if key_simple in table_simple:
                    p = np.array(table_simple[key_simple]["distribution"])

            if p is None:
                p = np.ones(NUM_CLASSES) / NUM_CLASSES

            # Floor: p*(1-6ε)+ε
            p = p * (1 - NUM_CLASSES * floor) + floor
            p /= p.sum()
            pred[y, x] = p

    return pred


# ── Scoring ────────────────────────────────────────────────────────────────

def weighted_kl(ground_truth, prediction):
    gt = np.array(ground_truth, dtype=float)
    pred = np.array(prediction, dtype=float)
    gt_safe = np.clip(gt, 1e-12, 1.0)
    pred_safe = np.clip(pred, 1e-12, 1.0)
    cell_kl = np.sum(gt_safe * (np.log(gt_safe) - np.log(pred_safe)), axis=-1)
    cell_entropy = -np.sum(gt_safe * np.log(gt_safe), axis=-1)
    total_weight = cell_entropy.sum()
    if total_weight <= 0:
        return float(cell_kl.mean())
    return float((cell_kl * cell_entropy).sum() / total_weight)


def score_from_kl(wkl):
    return max(0, min(100, 100 * np.exp(-3 * wkl)))


# ── Leave-one-out CV ──────────────────────────────────────────────────────

def run_leave_one_out(round_data, fingerprints, k=3, use_oracle_vbin=True):
    """
    For each round: exclude it, find k nearest neighbors, build prediction, score.

    use_oracle_vbin: if True, use GT-derived vbin for the target round too
                     (like current system's oracle vitality)
    """
    rnum_to_data = {rd["round_number"]: rd for rd in round_data}
    results = {}

    for target_rd in round_data:
        target_rnum = target_rd["round_number"]
        target_fp = fingerprints[target_rnum]

        # Find nearest neighbors (excluding target)
        neighbors = find_nearest_rounds(target_fp, fingerprints, exclude_rnum=target_rnum, k=k)
        neighbor_rnums = [rn for rn, _, _ in neighbors]
        neighbor_weights = [w for _, _, w in neighbors]
        neighbor_dists = [d for _, d, _ in neighbors]

        # Build calibration from neighbors
        table_d, table_s, table_simple = build_nn_calibration(
            round_data, neighbor_rnums, neighbor_weights
        )

        # Score each seed
        seed_scores = []
        for seed_entry in target_rd["seeds"]:
            grid = seed_entry["grid"]
            gt = seed_entry["ground_truth"]
            settlements = seed_entry["settlements"]

            # Determine vbin
            if use_oracle_vbin:
                gt_arr = np.array(gt, dtype=float)
                total_s, survived = 0, 0.0
                for s in settlements:
                    sx, sy = s["x"], s["y"]
                    if 0 <= sx < MAP_W and 0 <= sy < MAP_H:
                        total_s += 1
                        survived += gt_arr[sy, sx, 1] + gt_arr[sy, sx, 2]
                vbin = vitality_to_vbin(survived / total_s if total_s > 0 else 0.35)
            else:
                # Use fingerprint vitality (initial-state based, no GT)
                vbin = vitality_to_vbin(target_fp["vitality"])

            pred = nn_predict(grid, settlements, vbin, table_d, table_s, table_simple)
            wkl = weighted_kl(gt, pred)
            score = score_from_kl(wkl)
            seed_scores.append(score)

        avg_score = np.mean(seed_scores)
        results[target_rnum] = {
            "avg_score": avg_score,
            "seed_scores": seed_scores,
            "neighbors": [(rn, round(d, 2)) for rn, d, _ in neighbors],
            "vbin": vbin if not use_oracle_vbin else "oracle",
            "vitality": target_fp["vitality"],
        }

    return results


def run_baseline_loocv(round_data, fingerprints):
    """
    Baseline: build calibration from ALL rounds except target (simulating
    current system's approach of using all data).
    Same scoring as NN but k=13 (all remaining).
    """
    results = {}
    all_rnums = [rd["round_number"] for rd in round_data]

    for target_rd in round_data:
        target_rnum = target_rd["round_number"]
        # Use ALL other rounds (equally weighted)
        other_rnums = [rn for rn in all_rnums if rn != target_rnum]
        weights = [1.0 / len(other_rnums)] * len(other_rnums)

        table_d, table_s, table_simple = build_nn_calibration(
            round_data, other_rnums, weights
        )

        seed_scores = []
        for seed_entry in target_rd["seeds"]:
            grid = seed_entry["grid"]
            gt = seed_entry["ground_truth"]
            settlements = seed_entry["settlements"]

            gt_arr = np.array(gt, dtype=float)
            total_s, survived = 0, 0.0
            for s in settlements:
                sx, sy = s["x"], s["y"]
                if 0 <= sx < MAP_W and 0 <= sy < MAP_H:
                    total_s += 1
                    survived += gt_arr[sy, sx, 1] + gt_arr[sy, sx, 2]
            vbin = vitality_to_vbin(survived / total_s if total_s > 0 else 0.35)

            pred = nn_predict(grid, settlements, vbin, table_d, table_s, table_simple)
            wkl = weighted_kl(gt, pred)
            score = score_from_kl(wkl)
            seed_scores.append(score)

        results[target_rnum] = {
            "avg_score": np.mean(seed_scores),
            "seed_scores": seed_scores,
        }

    return results


# ── Hybrid: blend NN tables with global tables ────────────────────────────

def blend_tables(table_a, table_b, alpha=0.5):
    """Blend two calibration tables: alpha * A + (1-alpha) * B."""
    merged = {}
    all_keys = set(table_a.keys()) | set(table_b.keys())
    for key in all_keys:
        if key in table_a and key in table_b:
            da = np.array(table_a[key]["distribution"])
            db = np.array(table_b[key]["distribution"])
            blended = alpha * da + (1 - alpha) * db
            blended /= blended.sum()
            merged[key] = {"distribution": blended.tolist(),
                           "sample_count": table_a[key]["sample_count"] + table_b[key]["sample_count"]}
        elif key in table_a:
            merged[key] = table_a[key]
        else:
            merged[key] = table_b[key]
    return merged


def run_hybrid_loocv(round_data, fingerprints, k=3, alpha=0.5):
    """
    Hybrid: blend NN-weighted tables (alpha) with all-data tables (1-alpha).
    This keeps the stability of global tables while biasing toward similar rounds.
    """
    rnum_to_data = {rd["round_number"]: rd for rd in round_data}
    all_rnums = [rd["round_number"] for rd in round_data]
    results = {}

    for target_rd in round_data:
        target_rnum = target_rd["round_number"]
        target_fp = fingerprints[target_rnum]
        other_rnums = [rn for rn in all_rnums if rn != target_rnum]

        # Global tables (all other rounds, equal weight)
        global_weights = [1.0 / len(other_rnums)] * len(other_rnums)
        global_d, global_s, global_simple = build_nn_calibration(
            round_data, other_rnums, global_weights
        )

        # NN tables (k nearest, similarity-weighted)
        neighbors = find_nearest_rounds(target_fp, fingerprints, exclude_rnum=target_rnum, k=k)
        nn_rnums = [rn for rn, _, _ in neighbors]
        nn_weights = [w for _, _, w in neighbors]
        nn_d, nn_s, nn_simple = build_nn_calibration(
            round_data, nn_rnums, nn_weights
        )

        # Blend
        blended_d = blend_tables(nn_d, global_d, alpha)
        blended_s = blend_tables(nn_s, global_s, alpha)
        blended_simple = blend_tables(nn_simple, global_simple, alpha)

        seed_scores = []
        for seed_entry in target_rd["seeds"]:
            grid = seed_entry["grid"]
            gt = seed_entry["ground_truth"]
            settlements = seed_entry["settlements"]

            gt_arr = np.array(gt, dtype=float)
            total_s, survived = 0, 0.0
            for s in settlements:
                sx, sy = s["x"], s["y"]
                if 0 <= sx < MAP_W and 0 <= sy < MAP_H:
                    total_s += 1
                    survived += gt_arr[sy, sx, 1] + gt_arr[sy, sx, 2]
            vbin = vitality_to_vbin(survived / total_s if total_s > 0 else 0.35)

            pred = nn_predict(grid, settlements, vbin, blended_d, blended_s, blended_simple)
            wkl = weighted_kl(gt, pred)
            score = score_from_kl(wkl)
            seed_scores.append(score)

        results[target_rnum] = {
            "avg_score": np.mean(seed_scores),
            "seed_scores": seed_scores,
        }

    return results


def run_vbin_aware_nn_loocv(round_data, fingerprints, k=3):
    """
    NN matching but only within the same vbin group.
    Falls back to all rounds in the vbin if fewer than k available.
    """
    results = {}
    all_rnums = [rd["round_number"] for rd in round_data]

    # Group rounds by vbin
    vbin_map = {rn: vitality_to_vbin(fingerprints[rn]["vitality"]) for rn in fingerprints}

    for target_rd in round_data:
        target_rnum = target_rd["round_number"]
        target_vbin = vbin_map[target_rnum]
        target_fp = fingerprints[target_rnum]

        # Rounds in same vbin (excluding target)
        same_vbin = [rn for rn in all_rnums if rn != target_rnum and vbin_map[rn] == target_vbin]

        if len(same_vbin) == 0:
            # No rounds in same vbin — use all other rounds
            same_vbin = [rn for rn in all_rnums if rn != target_rnum]

        # Find nearest within same vbin
        same_vbin_fps = {rn: fingerprints[rn] for rn in same_vbin}
        neighbors = find_nearest_rounds(target_fp, same_vbin_fps, k=min(k, len(same_vbin)))
        nn_rnums = [rn for rn, _, _ in neighbors]
        nn_weights = [w for _, _, w in neighbors]

        table_d, table_s, table_simple = build_nn_calibration(
            round_data, nn_rnums, nn_weights
        )

        seed_scores = []
        for seed_entry in target_rd["seeds"]:
            grid = seed_entry["grid"]
            gt = seed_entry["ground_truth"]
            settlements = seed_entry["settlements"]

            gt_arr = np.array(gt, dtype=float)
            total_s, survived = 0, 0.0
            for s in settlements:
                sx, sy = s["x"], s["y"]
                if 0 <= sx < MAP_W and 0 <= sy < MAP_H:
                    total_s += 1
                    survived += gt_arr[sy, sx, 1] + gt_arr[sy, sx, 2]
            vbin = vitality_to_vbin(survived / total_s if total_s > 0 else 0.35)

            pred = nn_predict(grid, settlements, vbin, table_d, table_s, table_simple)
            wkl = weighted_kl(gt, pred)
            score = score_from_kl(wkl)
            seed_scores.append(score)

        results[target_rnum] = {
            "avg_score": np.mean(seed_scores),
            "seed_scores": seed_scores,
            "neighbors": [(rn, round(d, 2)) for rn, d, _ in neighbors],
        }

    return results


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("NEAREST-NEIGHBOR ROUND MATCHING — Astar Island")
    print("=" * 70)

    client = Client()

    # Step 1: Fetch all ground truth data
    print("\n[1/4] Henter ground truth fra alle fullførte runder...\n")
    round_data = fetch_all_rounds(client)
    print(f"\nHentet {len(round_data)} runder, {sum(len(r['seeds']) for r in round_data)} seeds totalt")

    # Step 2: Compute fingerprints
    print("\n[2/4] Beregner round fingerprints...\n")
    fingerprints = compute_round_fingerprints(round_data)

    print(f"{'Runde':<8} {'Vitality':>10} {'#Settle':>8} {'#Port':>6} {'Density':>8} {'Forest%':>8} {'#Ruin':>6} {'AvgDist':>8} {'Cluster':>8}")
    print("-" * 80)
    for rnum in sorted(fingerprints.keys()):
        fp = fingerprints[rnum]
        vbin = vitality_to_vbin(fp["vitality"])
        print(f"R{rnum:<7} {fp['vitality']:>9.3f} {fp['n_settlements']:>8} {fp['n_ports']:>6} "
              f"{fp['settlement_density']:>8.4f} {fp['forest_ratio']:>8.3f} {fp['n_ruins']:>6} "
              f"{fp['avg_settle_dist']:>8.1f} {fp['settle_clustering']:>8.1f}  [{vbin}]")

    # Step 3: Leave-one-out CV with NN (oracle vitality)
    print("\n[3/4] Leave-one-out CV: NN (k=3, oracle vbin)...\n")
    nn_results_k3 = run_leave_one_out(round_data, fingerprints, k=3, use_oracle_vbin=True)

    # Also test k=2 and k=5
    print("  Testing k=2...")
    nn_results_k2 = run_leave_one_out(round_data, fingerprints, k=2, use_oracle_vbin=True)
    print("  Testing k=5...")
    nn_results_k5 = run_leave_one_out(round_data, fingerprints, k=5, use_oracle_vbin=True)

    # Baseline: all rounds
    print("\n[4/6] Leave-one-out CV: Baseline (all rounds)...\n")
    baseline_results = run_baseline_loocv(round_data, fingerprints)

    # Hybrid: blend NN with global
    print("\n[5/6] Hybrid: NN blended with global (alpha sweep)...\n")
    hybrid_results = {}
    for alpha in [0.3, 0.5, 0.7]:
        print(f"  alpha={alpha}...")
        hybrid_results[alpha] = run_hybrid_loocv(round_data, fingerprints, k=3, alpha=alpha)

    # VBin-aware NN
    print("\n[6/6] VBin-aware NN (match within same vbin)...\n")
    vbin_nn_results = run_vbin_aware_nn_loocv(round_data, fingerprints, k=3)

    # ── Results ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("RESULTATER — Leave-One-Out Cross-Validation")
    print("=" * 110)

    # Find best hybrid alpha
    hybrid_avgs = {a: np.mean([r["avg_score"] for r in hybrid_results[a].values()]) for a in hybrid_results}
    best_alpha = max(hybrid_avgs, key=hybrid_avgs.get)

    print(f"\n{'Runde':<8} {'Baseline':>10} {'NN k=3':>10} {'NN k=5':>10} {'Hybrid':>10} {'VBin-NN':>10} {'Best':>10} {'Δ':>8} {'Neighbors (k=3)':>30}")
    print("-" * 116)

    all_scores = {"baseline": [], "nn3": [], "nn5": [], "hybrid": [], "vbin_nn": []}

    for rnum in sorted(nn_results_k3.keys()):
        b = baseline_results[rnum]["avg_score"]
        n3 = nn_results_k3[rnum]["avg_score"]
        n5 = nn_results_k5[rnum]["avg_score"]
        hyb = hybrid_results[best_alpha][rnum]["avg_score"]
        vnn = vbin_nn_results[rnum]["avg_score"]
        best = max(b, n3, n5, hyb, vnn)
        delta = best - b
        nb_str = ", ".join(f"R{rn}({d:.1f})" for rn, d in nn_results_k3[rnum]["neighbors"])
        which = "base" if best == b else ("nn3" if best == n3 else ("nn5" if best == n5 else ("hyb" if best == hyb else "vnn")))
        marker = f" [{which}]"

        print(f"R{rnum:<7} {b:>10.1f} {n3:>10.1f} {n5:>10.1f} {hyb:>10.1f} {vnn:>10.1f} {best:>10.1f} {delta:>+7.1f}{marker}")

        all_scores["baseline"].append(b)
        all_scores["nn3"].append(n3)
        all_scores["nn5"].append(n5)
        all_scores["hybrid"].append(hyb)
        all_scores["vbin_nn"].append(vnn)

    print("-" * 116)
    for key in all_scores:
        all_scores[key] = np.mean(all_scores[key])

    print(f"{'AVG':<8} {all_scores['baseline']:>10.1f} {all_scores['nn3']:>10.1f} {all_scores['nn5']:>10.1f} "
          f"{all_scores['hybrid']:>10.1f} {all_scores['vbin_nn']:>10.1f}")

    print(f"\nBaseline LOOCV avg:        {all_scores['baseline']:.1f}")
    print(f"NN k=3 avg:                {all_scores['nn3']:.1f}")
    print(f"NN k=5 avg:                {all_scores['nn5']:.1f}")
    print(f"Hybrid (alpha={best_alpha}) avg:    {all_scores['hybrid']:.1f}")
    print(f"VBin-aware NN avg:         {all_scores['vbin_nn']:.1f}")
    print(f"Reference baseline:        82.2 avg")

    # Hybrid alpha sweep
    print(f"\nHybrid alpha sweep (k=3):")
    for alpha in sorted(hybrid_results.keys()):
        avg = np.mean([r["avg_score"] for r in hybrid_results[alpha].values()])
        print(f"  alpha={alpha}: {avg:.1f}")

    # Per-vbin analysis
    print("\n\nPer vitality bin breakdown:")
    print(f"{'VBin':<8} {'N':>4} {'Baseline':>10} {'Hybrid':>10} {'VBin-NN':>10} {'Best delta':>10}")
    print("-" * 56)
    vbin_groups = defaultdict(list)
    for rnum in sorted(nn_results_k3.keys()):
        vbin = vitality_to_vbin(fingerprints[rnum]["vitality"])
        b = baseline_results[rnum]["avg_score"]
        hyb = hybrid_results[best_alpha][rnum]["avg_score"]
        vnn = vbin_nn_results[rnum]["avg_score"]
        vbin_groups[vbin].append((b, hyb, vnn))

    for vbin in ["DEAD", "LOW", "MED", "HIGH"]:
        if vbin in vbin_groups:
            bs = [x[0] for x in vbin_groups[vbin]]
            hs = [x[1] for x in vbin_groups[vbin]]
            vs = [x[2] for x in vbin_groups[vbin]]
            best_d = max(np.mean(hs) - np.mean(bs), np.mean(vs) - np.mean(bs))
            print(f"{vbin:<8} {len(bs):>4} {np.mean(bs):>10.1f} {np.mean(hs):>10.1f} {np.mean(vs):>10.1f} {best_d:>+9.1f}")

    # Summary
    best_method = max(all_scores, key=all_scores.__getitem__) if isinstance(all_scores, dict) else "baseline"
    print(f"\n{'='*60}")
    print(f"CONCLUSION:")
    print(f"  Baseline (all rounds, equal weight): {all_scores['baseline'] if isinstance(all_scores, dict) else all_scores:.1f}")
    print(f"  Best NN variant:                     {max(all_scores.values()) if isinstance(all_scores, dict) else 0:.1f}")
    print(f"  Reference (super_prior oracle):      82.2")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
