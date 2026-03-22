"""Backtest: Ola v8 (Chebyshev/square) vs Diamond (Manhattan) on all historical rounds.

Loads ground truth from history/ and builds prior-only predictions from both
solvers. No API calls needed — all local computation.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HISTORY_ROOT = Path(__file__).parent / "history"
OLA_DIR = Path(__file__).parent.parent / "ola"


def load_module(name, path):
    """Import a .py file as a module."""
    # Mock requests module to avoid ImportError (not needed for backtest)
    import types
    if "requests" not in sys.modules:
        mock_requests = types.ModuleType("requests")
        mock_requests.Session = type("Session", (), {"__init__": lambda s: None})
        sys.modules["requests"] = mock_requests
    if "requests.adapters" not in sys.modules:
        mock_adapters = types.ModuleType("requests.adapters")
        mock_adapters.HTTPAdapter = type("HTTPAdapter", (), {"__init__": lambda s, **kw: None})
        sys.modules["requests.adapters"] = mock_adapters
    if "urllib3.util.retry" not in sys.modules:
        mock_urllib3 = types.ModuleType("urllib3")
        mock_util = types.ModuleType("urllib3.util")
        mock_retry = types.ModuleType("urllib3.util.retry")
        mock_retry.Retry = type("Retry", (), {"__init__": lambda s, **kw: None})
        sys.modules["urllib3"] = mock_urllib3
        sys.modules["urllib3.util"] = mock_util
        sys.modules["urllib3.util.retry"] = mock_retry

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Ensure ola dir is in path for blending.py imports
    if str(OLA_DIR) not in sys.path:
        sys.path.insert(0, str(OLA_DIR))
    spec.loader.exec_module(mod)
    return mod


def kl_divergence(gt, pred):
    """Mean per-cell KL divergence."""
    eps = 1e-12
    kl = np.sum(gt * np.log(np.clip(gt, eps, 1.0) / np.clip(pred, eps, 1.0)), axis=-1)
    return float(kl.mean())


def kl_to_score(kl):
    return max(0.0, 100.0 * (1.0 - kl / 2.5))


def build_prediction(mod, grid, settlements):
    """Build prior-only prediction using a solution module's blended prediction."""
    # Load calibration data
    transition_table, simple_prior = mod.load_calibration()
    opt_tables = mod.load_optimized_calibration()

    # Default balanced weights (same as safety submit)
    type_weights = mod.compute_type_weights(0.33)

    # Build blended prediction
    pred = mod.build_blended_prediction(
        grid, settlements, transition_table, simple_prior,
        opt_tables, type_weights)

    if pred is None:
        # Fallback: single STABLE type
        obs = mod.SeedObserver(grid, settlements, transition_table, simple_prior,
                               alpha=mod.DEFAULT_ALPHA, opt_tables=opt_tables,
                               world_type="STABLE")
        pred = obs.build_prediction(apply_smoothing=False)

    return pred


def main():
    # Suppress print output from module loading
    import io
    import contextlib

    # Load both modules (suppress their print output)
    print("Loading modules...")
    with contextlib.redirect_stdout(io.StringIO()):
        v8 = load_module("ola_v8", OLA_DIR / "solution.py")

    # Reset model cache so diamond module loads fresh
    with contextlib.redirect_stdout(io.StringIO()):
        diamond = load_module("diamond", Path(__file__).parent / "solution_diamond.py")

    # Find all rounds with ground truth
    rounds = []
    for round_dir in sorted(HISTORY_ROOT.iterdir()):
        manifest_path = round_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.load(open(manifest_path))
        gt = manifest.get("ground_truth", {})
        ist = manifest.get("initial_states", [])
        if gt and ist:
            rounds.append(manifest)

    rounds.sort(key=lambda m: m.get("round_metadata", {}).get("round_number", 0))

    if not rounds:
        print("No rounds with ground truth found.")
        return

    print(f"Found {len(rounds)} rounds with ground truth\n")
    print(f"{'Round':>6} {'Setts':>5} | {'V8 KL':>8} {'V8 Score':>9} | {'Diamond KL':>10} {'Dia Score':>9} | {'Delta':>7}")
    print("-" * 85)

    v8_scores = []
    dia_scores = []

    for manifest in rounds:
        round_id = manifest["round_id"]
        round_num = manifest.get("round_metadata", {}).get("round_number", "?")
        initial_states = manifest.get("initial_states", [])
        ground_truth_paths = manifest.get("ground_truth", {})

        v8_kls = []
        dia_kls = []

        for ist in initial_states:
            seed_idx = ist["seed_index"]
            gt_key = str(seed_idx)
            if gt_key not in ground_truth_paths:
                continue

            # Load grid
            grid_path = HISTORY_ROOT / round_id / ist["grid_path"]
            if not grid_path.exists():
                continue
            grid = np.load(grid_path).tolist()

            # Load settlements
            settlements = ist["settlements"]
            n_sett = len(settlements)

            # Load ground truth
            gt_path = HISTORY_ROOT / round_id / ground_truth_paths[gt_key]
            if not gt_path.exists():
                continue
            gt = np.load(gt_path)

            # Build predictions (suppress prints)
            with contextlib.redirect_stdout(io.StringIO()):
                # Reset caches between runs
                v8._model_cache = None
                v8._super_cal_cache = None
                pred_v8 = build_prediction(v8, grid, settlements)

                diamond._model_cache = None
                diamond._super_cal_cache = None
                pred_dia = build_prediction(diamond, grid, settlements)

            v8_kls.append(kl_divergence(gt, pred_v8))
            dia_kls.append(kl_divergence(gt, pred_dia))

        if not v8_kls:
            continue

        n_sett = len(initial_states[0]["settlements"]) if initial_states else 0
        v8_mean_kl = np.mean(v8_kls)
        dia_mean_kl = np.mean(dia_kls)
        v8_score = kl_to_score(v8_mean_kl)
        dia_score = kl_to_score(dia_mean_kl)
        delta = dia_score - v8_score
        v8_scores.append(v8_score)
        dia_scores.append(dia_score)

        sign = "+" if delta >= 0 else ""
        print(f"R{round_num:>4} {n_sett:>5} | {v8_mean_kl:>8.4f} {v8_score:>8.1f}% | {dia_mean_kl:>10.4f} {dia_score:>8.1f}% | {sign}{delta:>6.1f}%")

    if v8_scores:
        print("-" * 85)
        v8_avg = np.mean(v8_scores)
        dia_avg = np.mean(dia_scores)
        delta_avg = dia_avg - v8_avg
        sign = "+" if delta_avg >= 0 else ""
        print(f"{'AVG':>6} {'':>5} | {'':>8} {v8_avg:>8.1f}% | {'':>10} {dia_avg:>8.1f}% | {sign}{delta_avg:>6.1f}%")

        # Per-round breakdown
        print(f"\nV8 wins: {sum(1 for v, d in zip(v8_scores, dia_scores) if v > d)}")
        print(f"Diamond wins: {sum(1 for v, d in zip(v8_scores, dia_scores) if d > v)}")
        print(f"Ties: {sum(1 for v, d in zip(v8_scores, dia_scores) if v == d)}")


if __name__ == "__main__":
    main()
