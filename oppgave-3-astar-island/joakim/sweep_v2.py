"""Quick parameter sweep for solution_v2.py to find optimal settings."""
import json
import math
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

# Import everything from solution_v2
import solution_v2 as v2

HISTORY_ROOT = v2.HISTORY_ROOT
MAP_H, MAP_W = v2.MAP_H, v2.MAP_W
NUM_CLASSES = v2.NUM_CLASSES


def weighted_kl(gt, pred, grid):
    eps = 1e-12
    gt = np.clip(np.asarray(gt, dtype=float), eps, 1.0)
    pred = np.clip(np.asarray(pred, dtype=float), eps, 1.0)
    grid = np.asarray(grid, dtype=int)
    cell_kl = np.sum(gt * np.log(gt / pred), axis=-1)
    cell_entropy = -np.sum(gt * np.log(gt), axis=-1)
    dynamic_mask = ~np.isin(grid, [10, 5])
    masked_entropy = cell_entropy * dynamic_mask
    total_entropy = float(masked_entropy.sum())
    if total_entropy > 0:
        return float((masked_entropy * cell_kl).sum() / total_entropy)
    dynamic_kls = cell_kl[dynamic_mask]
    if dynamic_kls.size == 0:
        return float(cell_kl.mean())
    return float(dynamic_kls.mean())


def score_from_wkl(val):
    return max(0.0, min(100.0, 100.0 * math.exp(-3.0 * val)))


def load_rounds():
    """Load all history rounds with GT and queries."""
    opt_tables = v2.load_opt_tables()
    model_tables = v2.load_model_tables()

    rounds = []
    for hdir in sorted(HISTORY_ROOT.iterdir()):
        if not hdir.is_dir():
            continue
        manifest_path = hdir / "manifest.json"
        if not manifest_path.exists():
            continue

        manifest = json.loads(manifest_path.read_text())
        rmeta = manifest.get("round_metadata", {})
        rnum = rmeta.get("round_number", "?")

        init_states = manifest.get("initial_states", [])
        queries = manifest.get("queries", [])
        if not init_states or not queries:
            continue  # Only test history_replay rounds

        seeds_data = []
        for ist in init_states:
            grid_path = hdir / ist["grid_path"]
            if not grid_path.exists():
                break
            seeds_data.append({
                "grid": np.load(grid_path).tolist(),
                "settlements": ist.get("settlements", []),
                "seed_index": ist["seed_index"],
            })
        if len(seeds_data) != len(init_states):
            continue

        n_seeds = len(seeds_data)
        gt_data = {}
        for si in range(n_seeds):
            gt_path = hdir / "arrays" / f"seed_{si}_ground_truth.npy"
            if gt_path.exists():
                gt_data[si] = np.load(gt_path).astype(float)
        if len(gt_data) < n_seeds:
            continue

        # Load query observations
        obs_list = []
        for q in queries:
            grid_path = hdir / q["grid_path"]
            if grid_path.exists():
                obs_list.append({
                    "seed_index": q["seed_index"],
                    "grid_data": np.load(grid_path).tolist(),
                    "vp": q["viewport"],
                })

        rounds.append({
            "rnum": rnum, "seeds_data": seeds_data, "gt_data": gt_data,
            "observations": obs_list, "n_seeds": n_seeds,
        })

    return rounds, opt_tables, model_tables


def test_params(rounds, opt_tables, model_tables, emp_max, emp_scale, alpha, min_emp_n):
    """Test a parameter combination on all rounds."""
    hr_scores = []

    for rnd in rounds:
        seeds_data = rnd["seeds_data"]
        n_seeds = rnd["n_seeds"]
        gt_data = rnd["gt_data"]

        model = v2.RoundModel(seeds_data)
        for obs in rnd["observations"]:
            model.add_observation(obs["seed_index"], obs["grid_data"],
                                  obs["vp"]["x"], obs["vp"]["y"])

        type_weights = model.compute_type_weights()
        seed_scores = []

        for si in range(n_seeds):
            # Custom build_prediction with tunable params
            grid = model.grids[si]
            sett = model.settlements[si]
            static = model.static_masks[si]
            coastal = model.coastal_masks[si]
            dist_map = model.dist_maps[si]
            pred = np.zeros((MAP_H, MAP_W, NUM_CLASSES), dtype=float)

            for y in range(MAP_H):
                for x in range(MAP_W):
                    if grid[y, x] == 10:
                        pred[y, x] = [0.998, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004]
                        continue
                    if grid[y, x] == 5:
                        pred[y, x] = [0.0004, 0.0004, 0.0004, 0.0004, 0.0004, 0.998]
                        continue

                    terrain_code = int(grid[y, x])
                    dist = int(dist_map[y, x])
                    is_coast = bool(coastal[y, x])

                    cal = v2.get_blended_cal(
                        terrain_code, dist, is_coast, sett, grid,
                        y, x, type_weights, opt_tables, model_tables,
                    )

                    fkey, _ = model._feature_key(y, x, si)
                    emp_n = model.key_total.get(fkey, 0)

                    if emp_n >= min_emp_n:
                        emp = model.key_counts[fkey] / emp_n
                        emp_weight = min(emp_max, emp_n / emp_scale)
                        base = emp_weight * emp + (1.0 - emp_weight) * cal
                    else:
                        base = cal

                    ck = (si, y, x)
                    cell_n = model.cell_total.get(ck, 0)
                    if cell_n > 0:
                        posterior = (model.cell_counts[ck] + alpha * base) / (cell_n + alpha)
                        pred[y, x] = posterior
                    else:
                        pred[y, x] = base

                    pred[y, x] = np.maximum(pred[y, x], v2.FLOOR)
                    pred[y, x, 5] = v2.FLOOR * 0.1
                    if not is_coast:
                        pred[y, x, 2] = v2.FLOOR * 0.1
                    pred[y, x] /= pred[y, x].sum()

            grid_list = seeds_data[si].get("grid", [])
            wkl = weighted_kl(gt_data[si], pred, grid_list)
            seed_scores.append(score_from_wkl(wkl))

        hr_scores.append(sum(seed_scores) / len(seed_scores))

    return sum(hr_scores) / len(hr_scores) if hr_scores else 0, hr_scores


def main():
    print("Loading rounds...")
    rounds, opt_tables, model_tables = load_rounds()
    print(f"Loaded {len(rounds)} history_replay rounds")
    for r in rounds:
        print(f"  R{r['rnum']}: {r['n_seeds']} seeds, {len(r['observations'])} obs")

    # Test vitality scaling on top of best params
    configs = [
        (0.50, 100.0, 50.0, 5, "best_baseline"),
    ]

    print(f"\nSweeping {len(configs)} configurations...")
    results = []
    for emp_max, emp_scale, alpha, min_n, label in configs:
        avg, per_round = test_params(rounds, opt_tables, model_tables,
                                     emp_max, emp_scale, alpha, min_n)
        per_round_str = " ".join(f"{s:.1f}" for s in per_round)
        results.append((avg, label, per_round))
        print(f"  {label:30s}  avg={avg:.2f}  [{per_round_str}]")

    results.sort(key=lambda x: -x[0])
    print(f"\n=== TOP 5 ===")
    for avg, label, per_round in results[:5]:
        per_round_str = " ".join(f"{s:.1f}" for s in per_round)
        print(f"  {label:30s}  avg={avg:.2f}  [{per_round_str}]")

    print(f"\n=== BOTTOM 3 ===")
    for avg, label, per_round in results[-3:]:
        per_round_str = " ".join(f"{s:.1f}" for s in per_round)
        print(f"  {label:30s}  avg={avg:.2f}  [{per_round_str}]")


if __name__ == "__main__":
    main()
