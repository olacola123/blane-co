"""
Fast greedy optimizer — tests one parameter at a time while keeping others fixed.
Much faster than full grid search. Converges to local optimum.

Usage:
    export API_KEY='...'
    python optimize_fast.py
"""
from __future__ import annotations
import json, math, time, sys
from pathlib import Path
import numpy as np
import solution_diamond as diamond

STATIC_TERRAIN = {10, 5}
HISTORY_ROOT = Path(__file__).with_name("history")

def weighted_kl(gt, pred, grid):
    eps = 1e-12
    gt = np.clip(np.asarray(gt, float), eps, 1.0)
    pred = np.clip(np.asarray(pred, float), eps, 1.0)
    grid = np.asarray(grid, int)
    kl = np.sum(gt * np.log(gt / pred), axis=-1)
    ent = -np.sum(gt * np.log(gt), axis=-1)
    dyn = ~np.isin(grid, list(STATIC_TERRAIN))
    me = ent * dyn
    te = me.sum()
    if te > 0: return float((me * kl).sum() / te)
    dk = kl[dyn]
    return float(dk.mean()) if dk.size > 0 else float(kl.mean())

def score_from_wkl(v): return max(0.0, min(100.0, 100.0 * math.exp(-3.0 * v)))

def load_query_grid(rid, q):
    return np.load(HISTORY_ROOT / rid / q["grid_path"]).tolist()

def predict_round(ri, rd, manifest, tt, sp, ot, alpha, gt,
                  dead_lo=0.05, dead_hi=0.15, boom_lo=0.28, boom_hi=0.38,
                  dead_thresh=0.20, stable_thresh=0.55,
                  recal_strength=0.12, cross_max=0.15):
    seeds = rd.get("seeds", rd.get("initial_states", []))
    n = len(seeds)
    queries = sorted(manifest.get("queries", []), key=lambda q: q.get("queries_used", 1e9))
    qbs = {si: [] for si in range(n)}
    for q in queries: qbs[int(q["seed_index"])].append(q)
    np_ = min(4, n); pqe = max(1, 4 // np_)

    probes = []
    for si in range(np_):
        sd = seeds[si]
        o = diamond.SeedObserver(sd.get("grid",[]), sd.get("settlements",[]),
                                 tt, sp, alpha=alpha, opt_tables=ot, world_type="STABLE")
        for q in qbs[si][:pqe]:
            vp = q["viewport"]; o.add_observation(load_query_grid(ri["id"], q), vp["x"], vp["y"])
        probes.append(o)

    fp = diamond.compute_round_fingerprint(probes)
    vital = diamond.fingerprint_to_vitality(fp)

    # Parameterized type weights
    s = max(0.0, min(1.0, vital))
    w_dead = max(0.0, min(1.0, (dead_hi - s) / max(dead_hi - dead_lo, 0.01))) if s < dead_hi else 0.0
    w_boom = max(0.0, min(1.0, (s - boom_lo) / max(boom_hi - boom_lo, 0.01))) if s > boom_lo else 0.0
    w_stable = max(0.0, 1.0 - w_dead - w_boom)
    tot = w_dead + w_stable + w_boom
    if tot < 1e-10: tw = {"DEAD": 0.0, "STABLE": 1.0, "BOOM": 0.0}
    else: tw = {"DEAD": w_dead/tot, "STABLE": w_stable/tot, "BOOM": w_boom/tot}

    exp, ev = diamond.infer_expansion_range(probes)

    # Classify
    if vital < dead_thresh: wtype = "DEAD"
    elif vital < stable_thresh: wtype = "STABLE"
    else: wtype = "BOOM_CONC"

    # Split boom
    expanded = diamond.split_boom_by_expansion(tw, exp)

    observers = []
    for si in range(n):
        sd = seeds[si]
        if si < len(probes):
            o = probes[si]; o.opt_tables = ot; o.world_type = wtype; o._rebuild_priors()
            rem = qbs[si][pqe:]
        else:
            o = diamond.SeedObserver(sd.get("grid",[]), sd.get("settlements",[]),
                                     tt, sp, alpha=alpha, opt_tables=ot, world_type=wtype)
            rem = qbs[si]
        for q in rem:
            vp = q["viewport"]; o.add_observation(load_query_grid(ri["id"], q), vp["x"], vp["y"])
        observers.append(o)
        if len(observers) >= 2:
            ct = diamond.build_cross_seed_prior(observers)
            if ct: diamond.apply_cross_seed(observers, ct, tt)

    fe, fev = diamond.infer_expansion_range(observers)
    if fev > ev: exp = fe
    expanded = diamond.split_boom_by_expansion(tw, exp)
    fct = diamond.build_cross_seed_prior(observers) if len(observers) >= 2 else {}

    scores = []
    for si, obs in enumerate(observers):
        sd = seeds[si]; grid = sd.get("grid",[]); sett = sd.get("settlements",[])
        plist, wlist = [], []
        for wt, w in expanded.items():
            if w < 0.01: continue
            to = diamond.SeedObserver(grid, sett, tt, sp, alpha=diamond.DEFAULT_ALPHA,
                                       opt_tables=ot, world_type=wt)
            plist.append(to.build_prediction(apply_smoothing=False)); wlist.append(w)
        if plist:
            bl = np.zeros_like(plist[0]); tw2 = sum(wlist)
            for p, w in zip(plist, wlist): bl += (w/tw2) * p
            bl /= np.maximum(bl.sum(axis=2, keepdims=True), 1e-10)
            # Recalibrate
            if fp["n_observed"] >= 3:
                sv_dev = fp["survival_rate"] - 0.33; ru_dev = fp["ruin_rate"] - 0.12
                adj = np.ones(6)
                adj[0] -= sv_dev * recal_strength; adj[1] += sv_dev * recal_strength
                adj[2] += sv_dev * recal_strength * 0.5; adj[3] -= sv_dev * recal_strength
                adj[4] -= sv_dev * recal_strength * 0.3
                adj[3] += ru_dev * recal_strength * 0.5; adj[1] -= ru_dev * recal_strength * 0.3
                adj = np.clip(adj, 0.85, 1.15)
                dyn = ~obs.static_mask; bl[dyn] *= adj
                bl[dyn] = np.maximum(bl[dyn], 0.001)
                bl /= np.maximum(bl.sum(axis=2, keepdims=True), 1e-10)
            pred = bl
            do = int((obs.observed[~obs.static_mask] > 0).sum())
            if fct and do > 0:
                pred = diamond.apply_cross_seed_to_pred(pred, grid, sett, fct, obs.static_mask, max_weight=cross_max)
        else:
            pred = obs.build_prediction(apply_smoothing=False)
        wkl = weighted_kl(gt[si]["ground_truth"], pred, grid)
        scores.append(score_from_wkl(wkl))
    return float(np.mean(scores))

def main():
    client = diamond.AstarClient()
    tt, sp = diamond.load_calibration()
    diamond.load_calibration_by_type()
    ot = diamond.load_optimized_calibration()
    lr = diamond.load_learning_state()
    alpha = float(lr.get("alpha", diamond.DEFAULT_ALPHA))

    rounds = client.get_rounds()
    completed = sorted([r for r in rounds if isinstance(r, dict) and r.get("status") == "completed"],
                       key=lambda r: r.get("round_number", 0))

    history = []
    for ri in completed:
        mp = HISTORY_ROOT / ri["id"] / "manifest.json"
        if not mp.exists(): continue
        m = json.loads(mp.read_text())
        if not m.get("queries"): continue
        rd = client.get_round(ri["id"])
        seeds = rd.get("seeds", rd.get("initial_states", []))
        gt = {}
        for si in range(len(seeds)):
            a = client.get(f"/analysis/{ri['id']}/{si}")
            gtd = a.get("ground_truth")
            if not gtd: continue
            gt[si] = {"ground_truth": np.asarray(gtd, float)}
            time.sleep(0.05)
        if len(gt) == len(seeds):
            history.append((ri, rd, m, gt))
            print(f"  Runde {ri.get('round_number')}: OK")

    print(f"\n{len(history)} history_replay runder\n")

    # Current best params
    params = dict(dead_lo=0.05, dead_hi=0.15, boom_lo=0.28, boom_hi=0.38,
                  dead_thresh=0.20, stable_thresh=0.55, recal_strength=0.12, cross_max=0.15)

    def eval_params(p):
        scores = []
        for ri, rd, m, gt in history:
            scores.append(predict_round(ri, rd, m, tt, sp, ot, alpha, gt, **p))
        return float(np.mean(scores)), scores

    base_avg, base_scores = eval_params(params)
    print(f"BASELINE: {base_avg:.4f}")
    for i, (ri, _, _, _) in enumerate(history):
        print(f"  Runde {ri.get('round_number')}: {base_scores[i]:.2f}")

    # Greedy coordinate descent
    search_space = {
        "dead_lo":       [0.02, 0.03, 0.05, 0.08, 0.10],
        "dead_hi":       [0.10, 0.12, 0.15, 0.18, 0.20, 0.25],
        "boom_lo":       [0.18, 0.22, 0.25, 0.28, 0.32, 0.36],
        "boom_hi":       [0.28, 0.33, 0.38, 0.42, 0.48, 0.55],
        "dead_thresh":   [0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30],
        "stable_thresh": [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65],
        "recal_strength":[0.0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.25],
        "cross_max":     [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30],
    }

    best_avg = base_avg
    best_params = params.copy()

    for iteration in range(3):  # 3 rounds of coordinate descent
        print(f"\n=== ITERATION {iteration+1} ===")
        improved = False
        for key in search_space:
            best_val = best_params[key]
            for val in search_space[key]:
                test = best_params.copy()
                test[key] = val
                # Validate constraints
                if test["dead_lo"] >= test["dead_hi"]: continue
                if test["dead_hi"] >= test["boom_lo"]: continue
                if test["boom_lo"] >= test["boom_hi"]: continue
                if test["dead_thresh"] >= test["stable_thresh"]: continue

                avg, _ = eval_params(test)
                if avg > best_avg:
                    best_avg = avg
                    best_val = val
                    improved = True

            if best_val != best_params[key]:
                print(f"  {key}: {best_params[key]} → {best_val} (score: {best_avg:.4f}, +{best_avg - base_avg:.4f})")
                best_params[key] = best_val
            else:
                print(f"  {key}: {best_params[key]} (unchanged)")

        if not improved:
            print("  Ingen forbedring — stopper")
            break

    print(f"\n=== RESULTAT ===")
    print(f"Baseline: {base_avg:.4f}")
    print(f"Optimert: {best_avg:.4f} ({best_avg - base_avg:+.4f})")
    print(f"\nOptimale parametere:")
    for k, v in best_params.items():
        changed = " ←" if v != params[k] else ""
        print(f"  {k}: {v}{changed}")

    final_avg, final_scores = eval_params(best_params)
    print(f"\nPer-runde:")
    for i, (ri, _, _, _) in enumerate(history):
        diff = final_scores[i] - base_scores[i]
        print(f"  Runde {ri.get('round_number')}: {final_scores[i]:.2f} ({diff:+.2f})")

    Path(__file__).with_name("optimize_expansion_results.json").write_text(
        json.dumps({"baseline": base_avg, "optimized": best_avg, "params": best_params,
                     "baseline_scores": base_scores, "optimized_scores": final_scores}, indent=2, default=str))
    print(f"\nLagret til optimize_expansion_results.json")

if __name__ == "__main__":
    main()
