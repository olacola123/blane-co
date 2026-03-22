"""
Hybrid backtest for solution_diamond.py on historical rounds.

Constraints discovered during implementation:
- The API exposes ground truth for completed rounds via /analysis.
- The API does NOT allow /simulate on completed rounds ("Round is not active").

So this script uses the best available hybrid strategy:
- `history_replay`: For rounds with locally stored query observations, rebuild the
  post-observation prediction pipeline from those observations.
- `prior_only`: For rounds without stored observations, score the prior-only
  prediction against API ground truth.

Usage:
    export API_KEY='...'
    python3 backtest_diamond_full.py
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import solution_diamond as diamond


STATIC_TERRAIN = {10, 5}
HISTORY_ROOT = Path(__file__).with_name("history")


def weighted_kl(ground_truth: np.ndarray, prediction: np.ndarray, initial_grid: list[list[int]]) -> float:
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


def score_from_wkl(value: float) -> float:
    return max(0.0, min(100.0, 100.0 * math.exp(-3.0 * value)))


def fetch_ground_truth(client: diamond.AstarClient, round_id: str, n_seeds: int) -> dict[int, dict]:
    data: dict[int, dict] = {}
    for seed_index in range(n_seeds):
        analysis = client.get(f"/analysis/{round_id}/{seed_index}")
        ground_truth = analysis.get("ground_truth")
        if not ground_truth:
            raise RuntimeError(f"Mangler ground truth for round={round_id} seed={seed_index}")
        data[seed_index] = {
            "ground_truth": np.asarray(ground_truth, dtype=float),
            "api_score": analysis.get("score"),
        }
        time.sleep(0.05)
    return data


def build_prior_prediction(
    grid: list[list[int]],
    settlements: list[dict],
    transition_table: dict | None,
    simple_prior: dict | None,
    opt_tables: dict | None,
) -> np.ndarray:
    # Conservative safety weights using old ramp (50% STABLE, 50% BOOM at v=0.33)
    _s = 0.33
    _wd = max(0.0, min(1.0, (0.15 - _s) / 0.10)) if _s < 0.15 else 0.0
    _wb = max(0.0, min(1.0, (_s - 0.28) / 0.10)) if _s > 0.28 else 0.0
    _ws = max(0.0, 1.0 - _wd - _wb)
    _t = _wd + _ws + _wb
    type_weights = {"DEAD": _wd/_t, "STABLE": _ws/_t, "BOOM": _wb/_t}
    prediction = diamond.build_blended_prediction(
        grid,
        settlements,
        transition_table,
        simple_prior,
        opt_tables,
        type_weights,
        expansion_range=0.5,
    )
    if prediction is None:
        observer = diamond.SeedObserver(
            grid,
            settlements,
            transition_table,
            simple_prior,
            alpha=diamond.DEFAULT_ALPHA,
            opt_tables=opt_tables,
            world_type="STABLE",
        )
        prediction = observer.build_prediction(apply_smoothing=False)
    return prediction


def load_manifest(round_id: str) -> dict | None:
    manifest_path = HISTORY_ROOT / round_id / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text())


def load_query_grid(round_id: str, query_entry: dict) -> list[list[int]]:
    grid_path = HISTORY_ROOT / round_id / query_entry["grid_path"]
    return np.load(grid_path).tolist()


def score_predictions(
    round_id: str,
    round_data: dict,
    predictions: dict[int, np.ndarray],
    ground_truth_by_seed: dict[int, dict],
    mode: str,
    extra: dict,
) -> dict:
    seeds_data = round_data.get("seeds", round_data.get("initial_states", []))
    n_seeds = len(seeds_data)
    seed_results = []
    seed_scores = []
    seed_wkls = []

    for seed_index in range(n_seeds):
        grid = seeds_data[seed_index].get("grid", [])
        prediction = predictions[seed_index]
        gt_entry = ground_truth_by_seed[seed_index]
        wkl = weighted_kl(gt_entry["ground_truth"], prediction, grid)
        score = score_from_wkl(wkl)
        seed_scores.append(score)
        seed_wkls.append(wkl)

        seed_result = {
            "seed_index": seed_index,
            "score": round(score, 4),
            "weighted_kl": round(wkl, 6),
            "historical_api_score": gt_entry["api_score"],
        }
        seed_result.update(extra.get("seed_results", {}).get(seed_index, {}))
        seed_results.append(seed_result)

    round_weight = float(round_data.get("round_weight", 1.0))
    round_score = float(sum(seed_scores) / n_seeds) if n_seeds else 0.0
    return {
        "round_id": round_id,
        "round_number": round_data.get("round_number"),
        "round_weight": round_weight,
        "mode": mode,
        "n_seeds": n_seeds,
        "n_settlements_seed0": len(seeds_data[0].get("settlements", [])) if seeds_data else 0,
        "round_score": round(round_score, 4),
        "mean_weighted_kl": round(float(np.mean(seed_wkls)), 6),
        "weighted_round_score": round(round_score * round_weight, 4),
        "seed_results": seed_results,
        **{k: v for k, v in extra.items() if k != "seed_results"},
    }


def run_prior_only_round(
    round_id: str,
    round_data: dict,
    transition_table: dict | None,
    simple_prior: dict | None,
    opt_tables: dict | None,
    ground_truth_by_seed: dict[int, dict],
) -> dict:
    predictions, extra = predict_prior_only_round(
        round_data=round_data,
        transition_table=transition_table,
        simple_prior=simple_prior,
        opt_tables=opt_tables,
    )

    return score_predictions(
        round_id=round_id,
        round_data=round_data,
        predictions=predictions,
        ground_truth_by_seed=ground_truth_by_seed,
        mode="prior_only",
        extra=extra,
    )


def predict_prior_only_round(
    round_data: dict,
    transition_table: dict | None,
    simple_prior: dict | None,
    opt_tables: dict | None,
) -> tuple[dict[int, np.ndarray], dict]:
    predictions = {}
    seeds_data = round_data.get("seeds", round_data.get("initial_states", []))
    for seed_index, seed_data in enumerate(seeds_data):
        predictions[seed_index] = build_prior_prediction(
            seed_data.get("grid", []),
            seed_data.get("settlements", []),
            transition_table,
            simple_prior,
            opt_tables,
        )
    return predictions, {"query_count": 0}


def run_history_replay_round(
    round_id: str,
    round_data: dict,
    manifest: dict,
    transition_table: dict | None,
    simple_prior: dict | None,
    opt_tables: dict | None,
    alpha: float,
    ground_truth_by_seed: dict[int, dict],
) -> dict:
    predictions, extra = predict_history_replay_round(
        round_id=round_id,
        round_data=round_data,
        manifest=manifest,
        transition_table=transition_table,
        simple_prior=simple_prior,
        opt_tables=opt_tables,
        alpha=alpha,
    )

    return score_predictions(
        round_id=round_id,
        round_data=round_data,
        predictions=predictions,
        ground_truth_by_seed=ground_truth_by_seed,
        mode="history_replay",
        extra=extra,
    )


def predict_history_replay_round(
    round_id: str,
    round_data: dict,
    manifest: dict,
    transition_table: dict | None,
    simple_prior: dict | None,
    opt_tables: dict | None,
    alpha: float,
) -> tuple[dict[int, np.ndarray], dict]:
    seeds_data = round_data.get("seeds", round_data.get("initial_states", []))
    n_seeds = len(seeds_data)

    queries = sorted(manifest.get("queries", []), key=lambda item: item.get("queries_used", 10**9))
    queries_by_seed: dict[int, list[dict]] = {seed_index: [] for seed_index in range(n_seeds)}
    for query in queries:
        queries_by_seed[int(query["seed_index"])].append(query)

    n_probe_seeds = min(4, n_seeds)
    probe_queries_each = max(1, 4 // n_probe_seeds)

    probe_observers: list[diamond.SeedObserver] = []
    for seed_index in range(n_probe_seeds):
        seed_data = seeds_data[seed_index]
        observer = diamond.SeedObserver(
            seed_data.get("grid", []),
            seed_data.get("settlements", []),
            transition_table,
            simple_prior,
            alpha=alpha,
            opt_tables=opt_tables,
            world_type="STABLE",
        )
        for query in queries_by_seed[seed_index][:probe_queries_each]:
            viewport = query["viewport"]
            observer.add_observation(
                load_query_grid(round_id, query),
                viewport["x"],
                viewport["y"],
            )
        probe_observers.append(observer)

    fingerprint = diamond.compute_round_fingerprint(probe_observers)
    vitality = diamond.fingerprint_to_vitality(fingerprint)
    type_weights = diamond.compute_type_weights(vitality)
    expansion_range, expansion_evidence = diamond.infer_expansion_range(probe_observers)
    world_type, _ = diamond.classify_world_type(seeds_data, vitality, expansion_range)
    # Use expansion-specific type directly (BOOM_CONC/BOOM_SPREAD stay separate)
    opt_world_type = world_type

    observers: list[diamond.SeedObserver] = []
    seed_meta: dict[int, dict] = {}

    for seed_index in range(n_seeds):
        seed_data = seeds_data[seed_index]
        if seed_index < len(probe_observers):
            observer = probe_observers[seed_index]
            observer.opt_tables = opt_tables
            observer.world_type = opt_world_type
            observer._rebuild_priors()
            remaining_queries = queries_by_seed[seed_index][probe_queries_each:]
        else:
            observer = diamond.SeedObserver(
                seed_data.get("grid", []),
                seed_data.get("settlements", []),
                transition_table,
                simple_prior,
                alpha=alpha,
                opt_tables=opt_tables,
                world_type=opt_world_type,
            )
            remaining_queries = queries_by_seed[seed_index]

        for query in remaining_queries:
            viewport = query["viewport"]
            observer.add_observation(
                load_query_grid(round_id, query),
                viewport["x"],
                viewport["y"],
            )

        observers.append(observer)

        if len(observers) >= 2:
            cross_table = diamond.build_cross_seed_prior(observers)
            if cross_table:
                diamond.apply_cross_seed(observers, cross_table, transition_table)

    final_expansion, final_evidence = diamond.infer_expansion_range(observers)
    if final_evidence > expansion_evidence:
        expansion_range = final_expansion
        expansion_evidence = final_evidence

    final_cross_table = diamond.build_cross_seed_prior(observers) if len(observers) >= 2 else {}
    predictions: dict[int, np.ndarray] = {}

    for seed_index, observer in enumerate(observers):
        seed_data = seeds_data[seed_index]
        grid = seed_data.get("grid", [])
        settlements = seed_data.get("settlements", [])

        blended_pred = diamond.build_blended_prediction(
            grid,
            settlements,
            transition_table,
            simple_prior,
            opt_tables,
            type_weights,
            expansion_range=expansion_range,
        )
        if blended_pred is not None:
            prediction = diamond.recalibrate_pred(blended_pred, fingerprint, observer.static_mask)
            prediction = diamond.scale_for_vitality(prediction, vitality, observer.static_mask)
            dynamic_observed = int((observer.observed[~observer.static_mask] > 0).sum())
            if final_cross_table and dynamic_observed > 0:
                prediction = diamond.apply_cross_seed_to_pred(
                    prediction,
                    grid,
                    settlements,
                    final_cross_table,
                    observer.static_mask,
                    max_weight=0.15,
                )
            reason = f"diamond+blend+exp={expansion_range:.3f}"
        else:
            prediction = observer.build_prediction(apply_smoothing=False, world_type=opt_world_type)
            reason = "fallback-observer"

        predictions[seed_index] = prediction

        dynamic_total = int((~observer.static_mask).sum())
        dynamic_observed = int((observer.observed[~observer.static_mask] > 0).sum())
        mean_obs = float(observer.observed[~observer.static_mask].mean()) if dynamic_total else 0.0
        seed_meta[seed_index] = {
            "dynamic_observed_cells": dynamic_observed,
            "dynamic_total_cells": dynamic_total,
            "mean_observations_per_dynamic_cell": round(mean_obs, 4),
            "query_count": len(queries_by_seed[seed_index]),
            "reason": reason,
        }

    return predictions, {
        "query_count": len(queries),
        "probe_queries_each": probe_queries_each,
        "fingerprint": {
            "survival_rate": round(float(fingerprint["survival_rate"]), 6),
            "ruin_rate": round(float(fingerprint["ruin_rate"]), 6),
            "empty_rate": round(float(fingerprint["empty_rate"]), 6),
            "forest_rate": round(float(fingerprint["forest_rate"]), 6),
            "n_observed": int(fingerprint["n_observed"]),
        },
        "vitality": round(float(vitality), 6),
        "expansion_range": round(float(expansion_range), 6),
        "expansion_evidence": round(float(expansion_evidence), 6),
        "world_type": world_type,
        "seed_results": seed_meta,
    }


def write_output(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid backtest for solution_diamond.py")
    parser.add_argument("--exclude-round-number", type=int, default=17)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("backtest_diamond_full_results.json"),
    )
    args = parser.parse_args()

    client = diamond.AstarClient()
    transition_table, simple_prior = diamond.load_calibration()
    _ = diamond.load_calibration_by_type()
    opt_tables = diamond.load_optimized_calibration()
    learning = diamond.load_learning_state()
    alpha = float(learning.get("alpha", diamond.DEFAULT_ALPHA))

    rounds = client.get_rounds()
    completed_rounds = [
        round_meta
        for round_meta in rounds
        if isinstance(round_meta, dict)
        and round_meta.get("status") == "completed"
        and round_meta.get("round_number") != args.exclude_round_number
    ]
    completed_rounds.sort(key=lambda round_meta: round_meta.get("round_number", 0))

    if not completed_rounds:
        raise SystemExit("Ingen completed rounds funnet for backtest.")

    print(
        f"Backtesting {len(completed_rounds)} completed rounds "
        f"(exclude round {args.exclude_round_number}, alpha={alpha:.3f})"
    )
    print(f"{'Round':>5} {'Mode':>14} {'Weight':>6} {'Score':>8} {'W.Score':>9} {'WKL':>8} {'Q':>4}")
    print("-" * 72)

    started_at = time.time()
    results = []
    total_weighted_score = 0.0

    for round_meta in completed_rounds:
        round_id = round_meta["id"]
        round_data = client.get_round(round_id)
        seeds_data = round_data.get("seeds", round_data.get("initial_states", []))
        ground_truth_by_seed = fetch_ground_truth(client, round_id, len(seeds_data))
        manifest = load_manifest(round_id)

        if manifest and len(manifest.get("queries", [])) > 0:
            result = run_history_replay_round(
                round_id=round_id,
                round_data=round_data,
                manifest=manifest,
                transition_table=transition_table,
                simple_prior=simple_prior,
                opt_tables=opt_tables,
                alpha=alpha,
                ground_truth_by_seed=ground_truth_by_seed,
            )
        else:
            result = run_prior_only_round(
                round_id=round_id,
                round_data=round_data,
                transition_table=transition_table,
                simple_prior=simple_prior,
                opt_tables=opt_tables,
                ground_truth_by_seed=ground_truth_by_seed,
            )

        results.append(result)
        total_weighted_score += result["weighted_round_score"]

        print(
            f"{result['round_number']:>5} {result['mode']:>14} {result['round_weight']:>6.2f} "
            f"{result['round_score']:>8.2f} {result['weighted_round_score']:>9.2f} "
            f"{result['mean_weighted_kl']:>8.4f} {result['query_count']:>4}"
        )

        payload = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "solver": "solution_diamond hybrid historical backtest",
            "alpha": alpha,
            "excluded_round_number": args.exclude_round_number,
            "completed_rounds": results,
            "summary": {
                "num_rounds": len(results),
                "history_replay_rounds": sum(1 for item in results if item["mode"] == "history_replay"),
                "prior_only_rounds": sum(1 for item in results if item["mode"] == "prior_only"),
                "mean_round_score": round(float(np.mean([item["round_score"] for item in results])), 4),
                "weighted_total_score": round(float(total_weighted_score), 4),
                "weighted_average_score": round(
                    float(
                        np.sum([item["round_score"] * item["round_weight"] for item in results])
                        / np.sum([item["round_weight"] for item in results])
                    ),
                    4,
                ),
                "elapsed_seconds": round(time.time() - started_at, 2),
            },
        }
        write_output(args.output, payload)

    print("-" * 72)
    print(
        f"AVG score={np.mean([item['round_score'] for item in results]):.2f}  "
        f"weighted_total={total_weighted_score:.2f}  "
        f"weighted_avg={np.sum([item['round_score'] * item['round_weight'] for item in results]) / np.sum([item['round_weight'] for item in results]):.2f}"
    )
    print(f"Output written to {args.output}")


if __name__ == "__main__":
    main()
