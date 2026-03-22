"""
Full recalibration using all 17 rounds from joakim_data/.

Produces:
  - calibration_manhattan_opt.json  (optimized wtype×terrain×dist×coastal tables)
  - model_tables.json symlink → joakim_data/calibration/model_tables.json
  - Prints optimal vitality thresholds found via grid search + LOO cross-validation

Usage:
    python calibrate_all_rounds.py                # Build + backtest
    python calibrate_all_rounds.py --optimize     # Also optimize vitality thresholds
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# === CONSTANTS ===
MAP_W, MAP_H = 40, 40
NUM_CLASSES = 6
STATIC_TERRAIN = {10, 5}
DATA_DIR = Path(__file__).parent.parent / "joakim_data"
OUTPUT_DIR = Path(__file__).parent

# Manhattan distance bins matching solution_diamond.py
OPT_DIST_BANDS = [(0, 0), (1, 2), (3, 3), (4, 5), (6, 8), (9, 12), (13, 99)]

TERRAIN_GROUP = {
    0: "plains", 1: "settlement", 2: "port", 3: "ruin",
    4: "forest", 5: "mountain", 10: "ocean", 11: "plains"
}

PROB_FLOOR = 0.001

# === DATA LOADING ===


def load_round_data(round_num: int) -> list[dict]:
    """Load all seeds for a round from joakim_data/runde_N/."""
    round_dir = DATA_DIR / f"runde_{round_num}"
    if not round_dir.exists():
        return []

    seeds = []
    for seed_idx in range(5):
        try:
            initial = json.loads((round_dir / f"seed_{seed_idx}_initial.json").read_text())
            gt = np.array(json.loads((round_dir / f"seed_{seed_idx}_ground_truth.json").read_text()))
            score_data = json.loads((round_dir / f"seed_{seed_idx}_score.json").read_text())
            seeds.append({
                "seed_index": seed_idx,
                "grid": initial["grid"],
                "settlements": initial["settlements"],
                "ground_truth": gt,
                "score": score_data.get("score"),
            })
        except Exception as e:
            print(f"  Skip runde {round_num} seed {seed_idx}: {e}")
    return seeds


def compute_vitality(seeds: list[dict]) -> float:
    """Compute average vitality (settlement survival prob) for a round."""
    vitalities = []
    for seed in seeds:
        gt = seed["ground_truth"]
        total_prob, count = 0.0, 0
        for s in seed["settlements"]:
            sx, sy = s["x"], s["y"]
            if 0 <= sx < MAP_W and 0 <= sy < MAP_H:
                total_prob += gt[sy, sx, 1] + gt[sy, sx, 2]  # settlement + port
                count += 1
        vitalities.append(total_prob / max(count, 1))
    return float(np.mean(vitalities)) if vitalities else 0.0


# === FEATURE EXTRACTION ===


def _terrain_group(t: int) -> str:
    if t in (0, 11):
        return "plains"
    elif t == 1:
        return "settlement"
    elif t == 2:
        return "port"
    elif t == 3:
        return "ruin"
    elif t == 4:
        return "forest"
    else:
        return "other"


def _dist_bin(d: int) -> int:
    for i, (lo, hi) in enumerate(OPT_DIST_BANDS):
        if lo <= d <= hi:
            return i
    return len(OPT_DIST_BANDS) - 1


def _settle_density_bin(n: int) -> int:
    return 0 if n == 0 else (1 if n <= 2 else 2)


def _forest_density_bin(n: int) -> int:
    if n == 0:
        return 0
    elif n <= 4:
        return 1
    elif n <= 10:
        return 2
    else:
        return 3


def extract_features(grid, settlements, y, x):
    """Extract features for a single cell."""
    terrain = grid[y][x]
    tg = _terrain_group(terrain)

    # Manhattan distance to nearest settlement
    min_dist = 99
    n_settle_r5 = 0
    for s in settlements:
        d = abs(y - s["y"]) + abs(x - s["x"])
        if d < min_dist:
            min_dist = d
        if d <= 5:
            n_settle_r5 += 1

    # Coastal check
    coastal = False
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < MAP_H and 0 <= nx < MAP_W:
                if grid[ny][nx] == 10:
                    coastal = True
                    break
        if coastal:
            break

    # Forest density
    n_forest_r2 = 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            ny, nx = y + dy, x + dx
            if 0 <= ny < MAP_H and 0 <= nx < MAP_W:
                if grid[ny][nx] == 4:
                    n_forest_r2 += 1

    db = _dist_bin(min_dist)
    c = int(coastal)
    sdb = _settle_density_bin(n_settle_r5)
    fdb = _forest_density_bin(n_forest_r2)

    return tg, db, c, sdb, fdb


# === TABLE BUILDING ===


def build_model_tables(
    all_rounds: dict[int, list[dict]],
    vitalities: dict[int, float],
    vbin_thresholds: tuple[float, float, float] = (0.08, 0.20, 0.35),
) -> dict:
    """Build 3-tier model tables from all round data.

    Args:
        all_rounds: {round_num: [seed_data, ...]}
        vitalities: {round_num: vitality_float}
        vbin_thresholds: (dead_hi, low_hi, med_hi) boundaries for vitality bins
    """
    dead_hi, low_hi, med_hi = vbin_thresholds

    def vitality_to_vbin(v):
        if v < dead_hi:
            return "DEAD"
        elif v < low_hi:
            return "LOW"
        elif v < med_hi:
            return "MED"
        else:
            return "HIGH"

    # Accumulators: key → [sum_of_distributions, count]
    specific_acc = defaultdict(lambda: [np.zeros(NUM_CLASSES), 0])
    medium_acc = defaultdict(lambda: [np.zeros(NUM_CLASSES), 0])
    simple_acc = defaultdict(lambda: [np.zeros(NUM_CLASSES), 0])

    total_points = 0

    for round_num, seeds in all_rounds.items():
        vit = vitalities[round_num]
        vbin = vitality_to_vbin(vit)

        for seed in seeds:
            grid = seed["grid"]
            settlements = seed["settlements"]
            gt = seed["ground_truth"]

            for y in range(MAP_H):
                for x in range(MAP_W):
                    terrain = grid[y][x]
                    if terrain in STATIC_TERRAIN:
                        continue

                    gt_dist = gt[y, x]
                    tg, db, c, sdb, fdb = extract_features(grid, settlements, y, x)

                    key_spec = f"{vbin}_{tg}_{db}_{c}_{sdb}_{fdb}"
                    specific_acc[key_spec][0] += gt_dist
                    specific_acc[key_spec][1] += 1

                    key_med = f"{vbin}_{tg}_{db}_{c}"
                    medium_acc[key_med][0] += gt_dist
                    medium_acc[key_med][1] += 1

                    key_simp = f"{tg}_{db}"
                    simple_acc[key_simp][0] += gt_dist
                    simple_acc[key_simp][1] += 1

                    total_points += 1

    # Convert to distribution tables
    def to_table(acc):
        table = {}
        for key, (dist_sum, count) in acc.items():
            if count >= 5:  # min samples
                dist = dist_sum / count
                dist = np.clip(dist, 0, None)
                dist /= dist.sum()
                table[key] = {
                    "distribution": dist.tolist(),
                    "sample_count": count,
                }
        return table

    round_vit_info = {}
    for rn, vit in vitalities.items():
        round_vit_info[str(rn)] = {
            "avg": vit,
            "bin": vitality_to_vbin(vit),
        }

    return {
        "table_specific": to_table(specific_acc),
        "table_medium": to_table(medium_acc),
        "table_simple": to_table(simple_acc),
        "round_vitalities": round_vit_info,
        "num_rounds": len(all_rounds),
        "num_seeds": sum(len(s) for s in all_rounds.values()),
        "total_data_points": total_points,
        "vbin_thresholds": list(vbin_thresholds),
    }


# === SCORING (matching backtest_diamond_full.py) ===


def weighted_kl(ground_truth, prediction, initial_grid):
    eps = 1e-12
    gt = np.clip(np.asarray(ground_truth, dtype=float), eps, 1.0)
    pred = np.clip(np.asarray(prediction, dtype=float), eps, 1.0)
    grid = np.asarray(initial_grid, dtype=int)

    cell_kl = np.sum(gt * np.log(gt / pred), axis=-1)
    cell_entropy = -np.sum(gt * np.log(gt), axis=-1)
    dynamic_mask = ~np.isin(grid, list(STATIC_TERRAIN))

    masked_entropy = cell_entropy * dynamic_mask
    total_entropy = float(masked_entropy.sum())
    if total_entropy > 0:
        return float((masked_entropy * cell_kl).sum() / total_entropy)

    dynamic_kls = cell_kl[dynamic_mask]
    if dynamic_kls.size == 0:
        return float(cell_kl.mean())
    return float(dynamic_kls.mean())


def score_from_wkl(value):
    return max(0.0, min(100.0, 100.0 * math.exp(-3.0 * value)))


# === PREDICTION (prior-only, using model tables) ===


def predict_with_tables(grid, settlements, tables, vbin, floor=PROB_FLOOR):
    """Build prediction using model tables — same logic as solution_diamond.super_predict."""
    table_specific = tables["table_specific"]
    table_medium = tables["table_medium"]
    table_simple = tables["table_simple"]

    pred = np.zeros((MAP_H, MAP_W, NUM_CLASSES), dtype=float)

    for y in range(MAP_H):
        for x in range(MAP_W):
            terrain = grid[y][x]

            if terrain == 10:  # ocean
                pred[y, x] = [1.0 - 5 * floor, floor, floor, floor, floor, floor]
                continue
            if terrain == 5:  # mountain
                pred[y, x] = [floor, floor, floor, floor, floor, 1.0 - 5 * floor]
                continue

            tg, db, c, sdb, fdb = extract_features(grid, settlements, y, x)

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

            # Floor mixing
            p = p * (1 - NUM_CLASSES * floor) + floor
            p /= p.sum()
            pred[y, x] = p

    return pred


def predict_blended(grid, settlements, tables, vitality, vbin_thresholds, floor=PROB_FLOOR):
    """Blended prediction using soft type weights — matches live solver logic."""
    dead_hi, low_hi, med_hi = vbin_thresholds

    def vitality_to_vbin(v):
        if v < dead_hi:
            return "DEAD"
        elif v < low_hi:
            return "LOW"
        elif v < med_hi:
            return "MED"
        else:
            return "HIGH"

    vbin = vitality_to_vbin(vitality)
    return predict_with_tables(grid, settlements, tables, vbin, floor)


# === BACKTEST ===


def backtest_round(
    round_num: int,
    seeds: list[dict],
    tables: dict,
    vitality: float,
    vbin_thresholds: tuple[float, float, float],
) -> dict:
    """Score predictions for one round using given tables and vitality."""
    dead_hi, low_hi, med_hi = vbin_thresholds

    def vitality_to_vbin(v):
        if v < dead_hi:
            return "DEAD"
        elif v < low_hi:
            return "LOW"
        elif v < med_hi:
            return "MED"
        else:
            return "HIGH"

    vbin = vitality_to_vbin(vitality)

    scores = []
    for seed in seeds:
        pred = predict_with_tables(seed["grid"], seed["settlements"], tables, vbin)
        wkl = weighted_kl(seed["ground_truth"], pred, seed["grid"])
        sc = score_from_wkl(wkl)
        scores.append(sc)

    return {
        "round_number": round_num,
        "vbin": vbin,
        "vitality": vitality,
        "mean_score": float(np.mean(scores)),
        "seed_scores": [round(s, 2) for s in scores],
    }


def backtest_all(
    all_rounds: dict[int, list[dict]],
    vitalities: dict[int, float],
    tables: dict,
    vbin_thresholds: tuple[float, float, float],
    weights: dict[int, float] | None = None,
) -> tuple[float, list[dict]]:
    """Backtest all rounds and return weighted average score."""
    results = []
    for rn in sorted(all_rounds.keys()):
        result = backtest_round(rn, all_rounds[rn], tables, vitalities[rn], vbin_thresholds)
        result["weight"] = weights.get(rn, 1.0) if weights else 1.0
        results.append(result)

    if weights:
        total_w = sum(r["weight"] for r in results)
        avg = sum(r["mean_score"] * r["weight"] for r in results) / total_w
    else:
        avg = float(np.mean([r["mean_score"] for r in results]))

    return avg, results


# === LOO CROSS-VALIDATION for vitality thresholds ===


def loo_cv_score(
    all_rounds: dict[int, list[dict]],
    vitalities: dict[int, float],
    vbin_thresholds: tuple[float, float, float],
    weights: dict[int, float] | None = None,
) -> float:
    """Leave-one-out cross-validation: for each round, build tables from the
    other rounds and score the held-out round."""
    round_nums = sorted(all_rounds.keys())
    cv_scores = []
    cv_weights = []

    for held_out in round_nums:
        # Build tables from all rounds except held_out
        train_rounds = {rn: seeds for rn, seeds in all_rounds.items() if rn != held_out}
        train_vit = {rn: v for rn, v in vitalities.items() if rn != held_out}
        tables = build_model_tables(train_rounds, train_vit, vbin_thresholds)

        # Score held-out round
        result = backtest_round(
            held_out, all_rounds[held_out], tables, vitalities[held_out], vbin_thresholds
        )
        w = weights.get(held_out, 1.0) if weights else 1.0
        cv_scores.append(result["mean_score"])
        cv_weights.append(w)

    if weights:
        total_w = sum(cv_weights)
        return sum(s * w for s, w in zip(cv_scores, cv_weights)) / total_w
    return float(np.mean(cv_scores))


def optimize_vbin_thresholds(
    all_rounds: dict[int, list[dict]],
    vitalities: dict[int, float],
    weights: dict[int, float] | None = None,
    use_loo: bool = False,
) -> tuple[tuple[float, float, float], float]:
    """Grid search for optimal vitality bin thresholds."""
    best_score = -1.0
    best_thresholds = (0.08, 0.20, 0.35)

    # Grid: dead_hi from 0.04-0.12, low_hi from 0.12-0.30, med_hi from 0.25-0.50
    candidates = []
    for dead_hi in np.arange(0.04, 0.13, 0.02):
        for low_hi in np.arange(max(dead_hi + 0.04, 0.12), 0.31, 0.02):
            for med_hi in np.arange(max(low_hi + 0.04, 0.25), 0.51, 0.02):
                candidates.append((round(dead_hi, 3), round(low_hi, 3), round(med_hi, 3)))

    print(f"  Searching {len(candidates)} threshold combinations...")

    if use_loo:
        # LOO is expensive, so first do a quick non-LOO pass to narrow candidates
        print("  Phase 1: quick screening (no LOO)...")
        quick_results = []
        for thresh in candidates:
            tables = build_model_tables(all_rounds, vitalities, thresh)
            score, _ = backtest_all(all_rounds, vitalities, tables, thresh, weights)
            quick_results.append((score, thresh))

        # Take top 20 for LOO
        quick_results.sort(reverse=True)
        top_candidates = [thresh for _, thresh in quick_results[:20]]

        print(f"  Phase 2: LOO CV on top {len(top_candidates)} candidates...")
        for thresh in top_candidates:
            score = loo_cv_score(all_rounds, vitalities, thresh, weights)
            if score > best_score:
                best_score = score
                best_thresholds = thresh
                print(f"    New best: {thresh} → {score:.4f}")
    else:
        for i, thresh in enumerate(candidates):
            tables = build_model_tables(all_rounds, vitalities, thresh)
            score, _ = backtest_all(all_rounds, vitalities, tables, thresh, weights)
            if score > best_score:
                best_score = score
                best_thresholds = thresh
                if i % 50 == 0 or score > best_score - 0.1:
                    print(f"    [{i+1}/{len(candidates)}] {thresh} → {score:.4f}")

    return best_thresholds, best_score


# === OPT_TABLES FOR SOLVER ===
# The solver uses world_type names: DEAD, STABLE, BOOM, BOOM_CONC, BOOM_SPREAD, ALL
# These are different from vitality bins (DEAD, LOW, MED, HIGH).
# Mapping: vitality < 0.20 → DEAD, 0.20-0.45 → STABLE, >= 0.45 → BOOM

WORLD_TYPE_THRESHOLDS = (0.20, 0.45)  # (dead_hi, stable_hi)


def vitality_to_world_type(v: float) -> str:
    if v < WORLD_TYPE_THRESHOLDS[0]:
        return "DEAD"
    elif v < WORLD_TYPE_THRESHOLDS[1]:
        return "STABLE"
    else:
        return "BOOM"


def build_opt_tables_for_solver(
    all_rounds: dict[int, list[dict]],
    vitalities: dict[int, float],
    vbin_thresholds: tuple[float, float, float],
) -> dict:
    """Build opt_tables keyed by world_type (DEAD/STABLE/BOOM/ALL).

    These match what get_prior() in solution_diamond.py looks up.
    Key format: "{world_type}_{terrain_group}_{dist_band}_{coastal}"
    """
    # Accumulators per world_type and ALL
    acc = defaultdict(lambda: [np.zeros(NUM_CLASSES), 0])

    for round_num, seeds in all_rounds.items():
        vit = vitalities[round_num]
        wtype = vitality_to_world_type(vit)

        for seed in seeds:
            grid = seed["grid"]
            settlements = seed["settlements"]
            gt = seed["ground_truth"]

            for y in range(MAP_H):
                for x in range(MAP_W):
                    terrain = grid[y][x]
                    if terrain in STATIC_TERRAIN:
                        continue

                    gt_dist = gt[y, x]
                    tg, db, c, _, _ = extract_features(grid, settlements, y, x)

                    # Per world_type
                    key = f"{wtype}_{tg}_{db}_{c}"
                    acc[key][0] += gt_dist
                    acc[key][1] += 1

                    # ALL aggregate
                    all_key = f"ALL_{tg}_{db}_{c}"
                    acc[all_key][0] += gt_dist
                    acc[all_key][1] += 1

    # Convert to tables
    opt_tables = {}
    for key, (dist_sum, count) in acc.items():
        if count >= 5:
            dist = dist_sum / count
            dist = np.clip(dist, 0, None)
            dist /= dist.sum()
            opt_tables[key] = {
                "distribution": dist.tolist(),
                "count": int(count),
            }

    # Add "_any" coastal variants (merge coastal=0 and coastal=1)
    any_acc = defaultdict(lambda: [np.zeros(NUM_CLASSES), 0])
    for key, entry in opt_tables.items():
        parts = key.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in ("0", "1"):
            any_key = f"{parts[0]}_any"
            any_acc[any_key][0] += np.array(entry["distribution"]) * entry["count"]
            any_acc[any_key][1] += entry["count"]

    for key, (dist_sum, count) in any_acc.items():
        if count > 0:
            dist = dist_sum / count
            dist /= dist.sum()
            opt_tables[key] = {
                "distribution": dist.tolist(),
                "count": int(count),
            }

    # Show world_type distribution
    wtype_rounds = defaultdict(list)
    for rn, vit in vitalities.items():
        wtype_rounds[vitality_to_world_type(vit)].append(rn)
    for wt in ["DEAD", "STABLE", "BOOM"]:
        rounds = wtype_rounds.get(wt, [])
        wt_keys = sum(1 for k in opt_tables if k.startswith(f"{wt}_"))
        print(f"  {wt}: {len(rounds)} rounds ({rounds}), {wt_keys} table entries")

    return opt_tables


# === MAIN ===


def main():
    parser = argparse.ArgumentParser(description="Full recalibration from joakim_data")
    parser.add_argument("--optimize", action="store_true", help="Optimize vitality thresholds")
    parser.add_argument("--loo", action="store_true", help="Use LOO cross-validation (slower)")
    parser.add_argument("--weighted", action="store_true", help="Use round weights in scoring")
    args = parser.parse_args()

    # Load all rounds
    print("Loading all rounds from joakim_data/...")
    all_rounds = {}
    vitalities = {}
    weights = {}

    for round_num in range(1, 30):  # scan for all available
        seeds = load_round_data(round_num)
        if seeds:
            all_rounds[round_num] = seeds
            vit = compute_vitality(seeds)
            vitalities[round_num] = vit
            meta_path = DATA_DIR / f"runde_{round_num}" / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                weights[round_num] = meta.get("round_weight", 1.0)
            else:
                weights[round_num] = 1.0

    print(f"Loaded {len(all_rounds)} rounds, {sum(len(s) for s in all_rounds.values())} seeds")
    for rn in sorted(all_rounds.keys()):
        print(f"  Runde {rn:2d}: vitality={vitalities[rn]:.4f}, weight={weights[rn]:.2f}, "
              f"seeds={len(all_rounds[rn])}")

    # Current thresholds
    current_thresholds = (0.08, 0.20, 0.35)

    # Build tables with current thresholds
    print(f"\n=== Building tables with current thresholds {current_thresholds} ===")
    tables_current = build_model_tables(all_rounds, vitalities, current_thresholds)
    score_current, results_current = backtest_all(
        all_rounds, vitalities, tables_current, current_thresholds,
        weights if args.weighted else None,
    )

    print(f"\nBacktest results (current thresholds, {'weighted' if args.weighted else 'unweighted'}):")
    print(f"{'Round':>5} {'VBin':>6} {'Vit':>7} {'Score':>8} {'Weight':>7} {'Seeds':>20}")
    print("-" * 70)
    for r in results_current:
        seeds_str = ", ".join(f"{s:.1f}" for s in r["seed_scores"])
        print(f"{r['round_number']:5d} {r['vbin']:>6} {r['vitality']:7.4f} "
              f"{r['mean_score']:8.2f} {r['weight']:7.2f} [{seeds_str}]")
    print(f"\n  Average score: {score_current:.4f}")

    if args.optimize:
        print(f"\n=== Optimizing vitality thresholds ===")
        best_thresh, best_score = optimize_vbin_thresholds(
            all_rounds, vitalities,
            weights if args.weighted else None,
            use_loo=args.loo,
        )
        print(f"\nBest thresholds: {best_thresh} → score {best_score:.4f}")
        print(f"Current thresholds: {current_thresholds} → score {score_current:.4f}")
        print(f"Improvement: {best_score - score_current:+.4f}")

        # Rebuild and show results with optimal thresholds
        tables_opt = build_model_tables(all_rounds, vitalities, best_thresh)
        _, results_opt = backtest_all(
            all_rounds, vitalities, tables_opt, best_thresh,
            weights if args.weighted else None,
        )

        print(f"\nBacktest results (optimized thresholds):")
        print(f"{'Round':>5} {'VBin':>6} {'Vit':>7} {'Score':>8} {'Δ':>7}")
        print("-" * 50)
        for ro, rc in zip(results_opt, results_current):
            delta = ro["mean_score"] - rc["mean_score"]
            print(f"{ro['round_number']:5d} {ro['vbin']:>6} {ro['vitality']:7.4f} "
                  f"{ro['mean_score']:8.2f} {delta:+7.2f}")

        # Save optimized tables
        save_tables = tables_opt
        save_thresh = best_thresh
    else:
        save_tables = tables_current
        save_thresh = current_thresholds

    # Save model_tables.json
    out_path = OUTPUT_DIR / "model_tables_17r.json"
    out_path.write_text(json.dumps(save_tables, indent=2, default=str))
    print(f"\nSaved {out_path} ({len(save_tables['table_specific'])} specific + "
          f"{len(save_tables['table_medium'])} medium + {len(save_tables['table_simple'])} simple)")

    # Build opt_tables for solution_diamond.py's get_prior() function
    # KEY FORMAT: "{world_type}_{terrain_group}_{dist_band}_{coastal}"
    # world_type = DEAD/STABLE/BOOM/BOOM_CONC/BOOM_SPREAD/ALL (NOT vbin names)
    # VALUE FORMAT: {"distribution": [...], "count": N}
    opt_tables = build_opt_tables_for_solver(all_rounds, vitalities, save_thresh)
    opt_out = {
        "tables": opt_tables,
        "num_rounds": save_tables["num_rounds"],
        "num_seeds": save_tables["num_seeds"],
        "vbin_thresholds": list(save_thresh),
    }
    opt_path = OUTPUT_DIR / "calibration_manhattan_opt.json"
    opt_path.write_text(json.dumps(opt_out, indent=2))
    print(f"Saved {opt_path} ({len(opt_tables)} entries)")

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Rounds: {len(all_rounds)}")
    print(f"Seeds: {sum(len(s) for s in all_rounds.values())}")
    print(f"Data points: {save_tables['total_data_points']}")
    print(f"Thresholds: {save_thresh}")
    print(f"Score: {score_current:.4f}" if not args.optimize else f"Score: {best_score:.4f}")

    # Show vitality distribution
    dead_hi, low_hi, med_hi = save_thresh
    bins = {"DEAD": [], "LOW": [], "MED": [], "HIGH": []}
    for rn, v in vitalities.items():
        if v < dead_hi:
            bins["DEAD"].append(rn)
        elif v < low_hi:
            bins["LOW"].append(rn)
        elif v < med_hi:
            bins["MED"].append(rn)
        else:
            bins["HIGH"].append(rn)

    for bname, rounds in bins.items():
        if rounds:
            print(f"  {bname}: runder {rounds}")


if __name__ == "__main__":
    main()
