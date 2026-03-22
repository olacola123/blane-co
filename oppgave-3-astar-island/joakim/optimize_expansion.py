"""
Expansion + Vitality Parameter Optimizer
==========================================
Grid search over ALL prediction parameters using history_replay backtest.

Optimizes:
1. Vitality thresholds (dead_threshold, boom_threshold for compute_type_weights)
2. Expansion split parameters (split_center, split_width)
3. Expansion detection normalization (dist_norm, far_norm)
4. Classification thresholds (classify_dead, classify_stable)
5. Recalibration strength
6. Cross-seed max weight

Uses history_replay rounds for scoring (most representative of live performance).

Usage:
    export API_KEY='din-jwt-token'
    python optimize_expansion.py                    # Full grid search
    python optimize_expansion.py --quick            # Quick search
    python optimize_expansion.py --analyze          # Show expansion metrics per round
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import solution_diamond as diamond

STATIC_TERRAIN = {10, 5}
HISTORY_ROOT = Path(__file__).with_name("history")


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
    return float(dynamic_kls.mean()) if dynamic_kls.size > 0 else float(cell_kl.mean())


def score_from_wkl(value):
    return max(0.0, min(100.0, 100.0 * math.exp(-3.0 * value)))


def load_query_grid(round_id, query_entry):
    return np.load(HISTORY_ROOT / round_id / query_entry["grid_path"]).tolist()


def load_manifest(round_id):
    p = HISTORY_ROOT / round_id / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else None


# === PARAMETERIZED FUNCTIONS ===

def compute_type_weights_param(survival_rate, dead_lo=0.05, dead_hi=0.15, boom_lo=0.28, boom_hi=0.38):
    """Parameterized type weights with configurable ramp boundaries."""
    s = max(0.0, min(1.0, survival_rate))
    w_dead = max(0.0, min(1.0, (dead_hi - s) / max(dead_hi - dead_lo, 0.01))) if s < dead_hi else 0.0
    w_boom = max(0.0, min(1.0, (s - boom_lo) / max(boom_hi - boom_lo, 0.01))) if s > boom_lo else 0.0
    w_stable = max(0.0, 1.0 - w_dead - w_boom)
    total = w_dead + w_stable + w_boom
    if total < 1e-10:
        return {"DEAD": 0.0, "STABLE": 1.0, "BOOM": 0.0}
    return {"DEAD": w_dead / total, "STABLE": w_stable / total, "BOOM": w_boom / total}


def classify_world_type_param(seeds_data, vitality, expansion_range,
                               dead_thresh=0.20, stable_thresh=0.55,
                               exp_low=0.35, exp_high=0.60):
    """Parameterized world type classification."""
    n_settlements = len(seeds_data[0].get("settlements", []))
    if vitality < dead_thresh:
        return "DEAD"
    elif vitality < stable_thresh:
        return "STABLE"
    else:
        if expansion_range < exp_low:
            return "BOOM_CONC"
        elif expansion_range > exp_high:
            return "BOOM_SPREAD"
        else:
            return "BOOM_CONC" if n_settlements >= 40 else "BOOM_SPREAD"


def split_boom_param(type_weights, expansion_range, center=0.45, width=0.15):
    """Split BOOM weight into CONC/SPREAD."""
    boom_w = type_weights.get("BOOM", 0.0)
    if boom_w < 0.01:
        return {k: v for k, v in type_weights.items() if k != "BOOM"}
    t = max(0.0, min(1.0, (expansion_range - center + width) / (2 * width)))
    result = {}
    for k, v in type_weights.items():
        if k == "BOOM":
            result["BOOM_CONC"] = boom_w * (1.0 - t)
            result["BOOM_SPREAD"] = boom_w * t
        else:
            result[k] = v
    return result


def recalibrate_pred_param(pred, fingerprint, static_mask, strength=0.12):
    """Parameterized recalibration."""
    if fingerprint["n_observed"] < 3:
        return pred
    survival = fingerprint["survival_rate"]
    ruin_rate = fingerprint["ruin_rate"]
    survival_dev = survival - 0.33
    ruin_dev = ruin_rate - 0.12
    adjustments = np.ones(6)
    adjustments[0] -= survival_dev * strength
    adjustments[1] += survival_dev * strength
    adjustments[2] += survival_dev * strength * 0.5
    adjustments[3] -= survival_dev * strength
    adjustments[4] -= survival_dev * strength * 0.3
    adjustments[3] += ruin_dev * strength * 0.5
    adjustments[1] -= ruin_dev * strength * 0.3
    adjustments = np.clip(adjustments, 0.85, 1.15)
    result = pred.copy()
    dynamic = ~static_mask
    result[dynamic] *= adjustments
    result[dynamic] = np.maximum(result[dynamic], 0.001)
    sums = result.sum(axis=2, keepdims=True)
    result /= np.maximum(sums, 1e-10)
    return result


def predict_with_params(
    round_id, round_data, manifest, transition_table, simple_prior,
    opt_tables, alpha, ground_truth_by_seed,
    # Vitality thresholds
    dead_lo=0.05, dead_hi=0.15, boom_lo=0.28, boom_hi=0.38,
    dead_thresh=0.20, stable_thresh=0.55,
    # Expansion params
    split_center=0.45, split_width=0.15,
    exp_low=0.35, exp_high=0.60,
    # Recalibration
    recal_strength=0.12,
    # Cross-seed
    cross_max_weight=0.15,
):
    """Run history_replay prediction with fully parameterized settings."""
    seeds_data = round_data.get("seeds", round_data.get("initial_states", []))
    n_seeds = len(seeds_data)

    queries = sorted(manifest.get("queries", []), key=lambda q: q.get("queries_used", 10**9))
    queries_by_seed = {si: [] for si in range(n_seeds)}
    for q in queries:
        queries_by_seed[int(q["seed_index"])].append(q)

    n_probe = min(4, n_seeds)
    pq_each = max(1, 4 // n_probe)

    probe_observers = []
    for si in range(n_probe):
        sd = seeds_data[si]
        obs = diamond.SeedObserver(
            sd.get("grid", []), sd.get("settlements", []),
            transition_table, simple_prior,
            alpha=alpha, opt_tables=opt_tables, world_type="STABLE")
        for q in queries_by_seed[si][:pq_each]:
            vp = q["viewport"]
            obs.add_observation(load_query_grid(round_id, q), vp["x"], vp["y"])
        probe_observers.append(obs)

    fingerprint = diamond.compute_round_fingerprint(probe_observers)
    vitality = diamond.fingerprint_to_vitality(fingerprint)
    type_weights = compute_type_weights_param(vitality, dead_lo, dead_hi, boom_lo, boom_hi)

    expansion_range, expansion_evidence = diamond.infer_expansion_range(probe_observers)

    world_type = classify_world_type_param(
        seeds_data, vitality, expansion_range,
        dead_thresh, stable_thresh, exp_low, exp_high)
    opt_wtype = world_type

    observers = []
    for si in range(n_seeds):
        sd = seeds_data[si]
        if si < len(probe_observers):
            obs = probe_observers[si]
            obs.opt_tables = opt_tables
            obs.world_type = opt_wtype
            obs._rebuild_priors()
            remaining = queries_by_seed[si][pq_each:]
        else:
            obs = diamond.SeedObserver(
                sd.get("grid", []), sd.get("settlements", []),
                transition_table, simple_prior,
                alpha=alpha, opt_tables=opt_tables, world_type=opt_wtype)
            remaining = queries_by_seed[si]

        for q in remaining:
            vp = q["viewport"]
            obs.add_observation(load_query_grid(round_id, q), vp["x"], vp["y"])
        observers.append(obs)

        if len(observers) >= 2:
            cross_table = diamond.build_cross_seed_prior(observers)
            if cross_table:
                diamond.apply_cross_seed(observers, cross_table, transition_table)

    final_exp, final_ev = diamond.infer_expansion_range(observers)
    if final_ev > expansion_evidence:
        expansion_range = final_exp

    expanded_weights = split_boom_param(type_weights, expansion_range, split_center, split_width)
    final_cross_table = diamond.build_cross_seed_prior(observers) if len(observers) >= 2 else {}
    predictions = {}

    for si, obs in enumerate(observers):
        sd = seeds_data[si]
        grid = sd.get("grid", [])
        settlements = sd.get("settlements", [])

        # Build blended prediction with expanded weights
        preds_list = []
        weights_list = []
        for wtype, w in expanded_weights.items():
            if w < 0.01:
                continue
            tmp_obs = diamond.SeedObserver(
                grid, settlements, transition_table, simple_prior,
                alpha=diamond.DEFAULT_ALPHA, opt_tables=opt_tables, world_type=wtype)
            pred = tmp_obs.build_prediction(apply_smoothing=False)
            preds_list.append(pred)
            weights_list.append(w)

        if preds_list:
            blended = np.zeros_like(preds_list[0])
            tw = sum(weights_list)
            for p, w in zip(preds_list, weights_list):
                blended += (w / tw) * p
            blended /= np.maximum(blended.sum(axis=2, keepdims=True), 1e-10)

            prediction = recalibrate_pred_param(blended, fingerprint, obs.static_mask, recal_strength)
            dynamic_observed = int((obs.observed[~obs.static_mask] > 0).sum())
            if final_cross_table and dynamic_observed > 0:
                prediction = diamond.apply_cross_seed_to_pred(
                    prediction, grid, settlements, final_cross_table,
                    obs.static_mask, max_weight=cross_max_weight)
        else:
            prediction = obs.build_prediction(apply_smoothing=False, world_type=opt_wtype)

        predictions[si] = prediction

    seed_scores = []
    for si in range(n_seeds):
        grid = seeds_data[si].get("grid", [])
        gt = ground_truth_by_seed[si]["ground_truth"]
        wkl = weighted_kl(gt, predictions[si], grid)
        seed_scores.append(score_from_wkl(wkl))

    return float(np.mean(seed_scores)), expansion_range, world_type, vitality


def fetch_ground_truth(client, round_id, n_seeds):
    data = {}
    for si in range(n_seeds):
        analysis = client.get(f"/analysis/{round_id}/{si}")
        gt = analysis.get("ground_truth")
        if not gt:
            raise RuntimeError(f"No GT for round={round_id} seed={si}")
        data[si] = {"ground_truth": np.asarray(gt, dtype=float), "api_score": analysis.get("score")}
        time.sleep(0.05)
    return data


def analyze_expansion(client, rounds_with_history):
    """Show expansion metrics per round from both probes and GT."""
    from calibrate_manhattan import compute_gt_expansion

    print(f"\n{'Rnd':>4} {'PVital':>6} {'GTVital':>7} {'PExp':>6} {'GTConc':>7} {'GTFar8':>7} "
          f"{'PType':>12} {'GTType':>12}")
    print("-" * 85)

    transition_table, simple_prior = diamond.load_calibration()
    opt_tables = diamond.load_optimized_calibration()

    for round_info, round_data, manifest, gt_data in rounds_with_history:
        rn = round_info.get("round_number", "?")
        seeds_data = round_data.get("seeds", round_data.get("initial_states", []))
        n_seeds = len(seeds_data)

        gt_entries = [{
            "ground_truth": gt_data[si]["ground_truth"].tolist(),
            "initial_grid": seeds_data[si].get("grid", []),
            "settlements": seeds_data[si].get("settlements", []),
        } for si in range(n_seeds)]

        avg_vital, avg_conc, avg_beyond = compute_gt_expansion(gt_entries)

        if avg_vital < 0.08: gt_class = "DEAD"
        elif avg_vital < 0.35: gt_class = "STABLE"
        elif avg_conc > 0.55: gt_class = "BOOM_CONC"
        else: gt_class = "BOOM_SPREAD"

        queries = sorted(manifest.get("queries", []), key=lambda q: q.get("queries_used", 10**9))
        qbs = {si: [] for si in range(n_seeds)}
        for q in queries:
            qbs[int(q["seed_index"])].append(q)

        probe_obs = []
        for si in range(min(4, n_seeds)):
            sd = seeds_data[si]
            obs = diamond.SeedObserver(
                sd.get("grid", []), sd.get("settlements", []),
                transition_table, simple_prior,
                alpha=diamond.DEFAULT_ALPHA, opt_tables=opt_tables, world_type="STABLE")
            for q in qbs[si][:1]:
                vp = q["viewport"]
                obs.add_observation(load_query_grid(round_info["id"], q), vp["x"], vp["y"])
            probe_obs.append(obs)

        fp = diamond.compute_round_fingerprint(probe_obs)
        vital = diamond.fingerprint_to_vitality(fp)
        exp, ev = diamond.infer_expansion_range(probe_obs)

        if vital < 0.20: p_class = "DEAD"
        elif vital < 0.55: p_class = "STABLE"
        else: p_class = "BOOM_CONC" if exp < 0.35 else "BOOM_SPREAD"

        match = "✓" if p_class == gt_class else "✗"
        print(f"{rn:>4} {vital:>6.3f} {avg_vital:>7.3f} {exp:>6.3f} {avg_conc:>7.3f} {avg_beyond:>7.3f} "
              f"{p_class:>12} {gt_class:>12} {match}")


def main():
    parser = argparse.ArgumentParser(description="Optimize prediction parameters")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("optimize_expansion_results.json"))
    args = parser.parse_args()

    client = diamond.AstarClient()
    transition_table, simple_prior = diamond.load_calibration()
    diamond.load_calibration_by_type()
    opt_tables = diamond.load_optimized_calibration()
    learning = diamond.load_learning_state()
    alpha = float(learning.get("alpha", diamond.DEFAULT_ALPHA))

    rounds = client.get_rounds()
    completed = sorted(
        [r for r in rounds if isinstance(r, dict) and r.get("status") == "completed"],
        key=lambda r: r.get("round_number", 0))

    rounds_with_history = []
    for ri in completed:
        manifest = load_manifest(ri["id"])
        if not manifest or not manifest.get("queries"):
            continue
        rd = client.get_round(ri["id"])
        seeds_data = rd.get("seeds", rd.get("initial_states", []))
        try:
            gt = fetch_ground_truth(client, ri["id"], len(seeds_data))
        except Exception as e:
            print(f"  Runde {ri.get('round_number')}: GT feil — {e}")
            continue
        rounds_with_history.append((ri, rd, manifest, gt))
        print(f"  Runde {ri.get('round_number')}: OK ({len(manifest.get('queries', []))} queries)")

    print(f"\n{len(rounds_with_history)} history_replay runder")

    if args.analyze:
        analyze_expansion(client, rounds_with_history)
        return

    if not rounds_with_history:
        print("Ingen runder med historikk!")
        return

    # === BASELINE ===
    print("\n=== BASELINE ===")
    baseline_params = dict(
        dead_lo=0.05, dead_hi=0.15, boom_lo=0.28, boom_hi=0.38,
        dead_thresh=0.20, stable_thresh=0.55,
        split_center=0.45, split_width=0.15,
        exp_low=0.35, exp_high=0.60,
        recal_strength=0.12, cross_max_weight=0.15)

    baseline_scores = []
    for ri, rd, manifest, gt in rounds_with_history:
        score, exp, wtype, vital = predict_with_params(
            ri["id"], rd, manifest, transition_table, simple_prior,
            opt_tables, alpha, gt, **baseline_params)
        baseline_scores.append(score)
        print(f"  Runde {ri.get('round_number')}: {score:.2f} (v={vital:.3f}, exp={exp:.3f}, type={wtype})")

    baseline_avg = float(np.mean(baseline_scores))
    print(f"\n  BASELINE snitt: {baseline_avg:.4f}")

    # === GRID SEARCH ===
    print("\n=== GRID SEARCH ===")

    if args.quick:
        param_grid = {
            "dead_lo": [0.03, 0.05, 0.08],
            "dead_hi": [0.12, 0.15, 0.18, 0.22],
            "boom_lo": [0.22, 0.28, 0.33],
            "boom_hi": [0.33, 0.38, 0.45],
            "dead_thresh": [0.15, 0.20, 0.25],
            "stable_thresh": [0.40, 0.50, 0.55, 0.60],
            "recal_strength": [0.06, 0.12, 0.18],
            "cross_max_weight": [0.10, 0.15, 0.20],
        }
        # Fixed expansion params (not enough data to optimize)
        fixed = dict(split_center=0.45, split_width=0.15, exp_low=0.35, exp_high=0.60)
    else:
        param_grid = {
            "dead_lo": [0.02, 0.05, 0.08, 0.10],
            "dead_hi": [0.10, 0.13, 0.15, 0.18, 0.22],
            "boom_lo": [0.20, 0.24, 0.28, 0.32, 0.36],
            "boom_hi": [0.30, 0.35, 0.38, 0.42, 0.48],
            "dead_thresh": [0.12, 0.15, 0.18, 0.20, 0.25],
            "stable_thresh": [0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
            "recal_strength": [0.0, 0.06, 0.10, 0.12, 0.15, 0.20],
            "cross_max_weight": [0.08, 0.12, 0.15, 0.20, 0.25],
        }
        fixed = dict(split_center=0.45, split_width=0.15, exp_low=0.35, exp_high=0.60)

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))
    print(f"  {len(combos)} parameter-kombinasjoner")

    # Filter invalid combinations (dead_hi must be < boom_lo, etc)
    valid_combos = []
    for combo in combos:
        p = dict(zip(keys, combo))
        if p["dead_lo"] >= p["dead_hi"]:
            continue
        if p["dead_hi"] >= p["boom_lo"]:
            continue
        if p["boom_lo"] >= p["boom_hi"]:
            continue
        if p["dead_thresh"] >= p["stable_thresh"]:
            continue
        valid_combos.append(combo)

    print(f"  {len(valid_combos)} gyldige kombinasjoner (etter filtrering)")

    best_score = baseline_avg
    best_params = baseline_params.copy()
    results = []
    n_improved = 0
    t0 = time.time()

    for i, combo in enumerate(valid_combos):
        params = dict(zip(keys, combo))
        params.update(fixed)

        scores = []
        for ri, rd, manifest, gt in rounds_with_history:
            try:
                score, _, _, _ = predict_with_params(
                    ri["id"], rd, manifest, transition_table, simple_prior,
                    opt_tables, alpha, gt, **params)
                scores.append(score)
            except Exception:
                scores.append(0.0)

        avg = float(np.mean(scores))
        if avg > best_score:
            best_score = avg
            best_params = {**params}
            n_improved += 1

        results.append({"params": params, "avg_score": avg, "scores": scores})

        if (i + 1) % 500 == 0 or i == len(valid_combos) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(valid_combos) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(valid_combos)}] Best: {best_score:.4f} "
                  f"(+{best_score - baseline_avg:.4f}) "
                  f"({n_improved} improvements, {rate:.1f}/s, ETA {eta:.0f}s)")

    results.sort(key=lambda r: r["avg_score"], reverse=True)

    print(f"\n=== TOPP 20 ===")
    print(f"{'Rank':>4} {'Score':>8} {'Diff':>6} {'DLo':>4} {'DHi':>4} {'BLo':>4} {'BHi':>4} "
          f"{'DThr':>5} {'SThr':>5} {'Recal':>5} {'XSW':>4}")
    print("-" * 75)
    for i, r in enumerate(results[:20]):
        p = r["params"]
        diff = r["avg_score"] - baseline_avg
        print(f"{i+1:>4} {r['avg_score']:>8.4f} {diff:>+6.2f} "
              f"{p['dead_lo']:>4.2f} {p['dead_hi']:>4.2f} "
              f"{p['boom_lo']:>4.2f} {p['boom_hi']:>4.2f} "
              f"{p['dead_thresh']:>5.2f} {p['stable_thresh']:>5.2f} "
              f"{p['recal_strength']:>5.2f} {p['cross_max_weight']:>4.2f}")

    if best_params:
        print(f"\n=== BESTE PARAMETERE ===")
        print(f"Score: {best_score:.4f} (baseline: {baseline_avg:.4f}, diff: {best_score - baseline_avg:+.4f})")
        for k, v in best_params.items():
            bl_v = baseline_params.get(k, v)
            changed = " ←" if v != bl_v else ""
            print(f"  {k}: {v}{changed}")

        print(f"\nPer-runde med beste parametere:")
        for idx, (ri, rd, manifest, gt) in enumerate(rounds_with_history):
            score, exp, wtype, vital = predict_with_params(
                ri["id"], rd, manifest, transition_table, simple_prior,
                opt_tables, alpha, gt, **best_params)
            diff = score - baseline_scores[idx]
            print(f"  Runde {ri.get('round_number')}: {score:.2f} ({diff:+.2f}) "
                  f"v={vital:.3f} type={wtype}")

    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_history_rounds": len(rounds_with_history),
        "n_combos_tested": len(valid_combos),
        "baseline_avg": baseline_avg,
        "baseline_scores": baseline_scores,
        "baseline_params": baseline_params,
        "best_avg": best_score,
        "best_params": best_params,
        "improvement": best_score - baseline_avg,
        "top_20": results[:20],
    }
    args.output.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResultater lagret til {args.output}")


if __name__ == "__main__":
    main()
