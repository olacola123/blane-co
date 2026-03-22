"""
Post-hoc coherence optimization for one stored history round.

Purpose:
- replay a locally stored round with the normal Diamond history-replay pipeline
- override only the round-level coherence scalar and coherence modulation strength
- measure whether that alone can materially improve the final score

Usage:
    python3 optimize_round_coherence.py --round-id <round-id>
    python3 optimize_round_coherence.py --round-number 17
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import backtest_diamond_full as bt
import solution_diamond as diamond


HISTORY_ROOT = Path(__file__).with_name("history")
OUTPUT_PATH = Path(__file__).with_name("optimize_round_coherence_results.json")


def replay_manifest(round_id: str) -> dict:
    manifest_path = HISTORY_ROOT / round_id / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Fant ikke manifest for round_id={round_id}")
    return json.loads(manifest_path.read_text())


def resolve_round_id(round_id: str | None, round_number: int | None) -> str:
    if round_id:
        return round_id
    if round_number is None:
        raise ValueError("Mangler --round-id eller --round-number")

    matches = []
    for manifest_path in sorted(HISTORY_ROOT.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("round_metadata") or {}).get("round_number") == round_number:
            matches.append(manifest["round_id"])
    if not matches:
        raise FileNotFoundError(f"Fant ingen lagret round_number={round_number}")
    if len(matches) > 1:
        raise ValueError(f"Flere lagrede runder med round_number={round_number}: {matches}")
    return matches[0]


def local_ground_truth_by_seed(manifest: dict) -> dict[int, dict]:
    analyses = manifest.get("analyses", {})
    result = {}
    for seed_index, path in manifest.get("ground_truth", {}).items():
        result[int(seed_index)] = {
            "ground_truth": np.load(HISTORY_ROOT / manifest["round_id"] / path),
            "api_score": analyses.get(str(seed_index), {}).get("score"),
        }
    return result


def local_round_data(manifest: dict) -> dict:
    round_metadata = manifest.get("round_metadata", {})
    initial_states = []
    for item in manifest.get("initial_states", []):
        grid = np.load(HISTORY_ROOT / manifest["round_id"] / item["grid_path"]).tolist()
        initial_states.append(
            {
                "seed_index": item["seed_index"],
                "grid": grid,
                "settlements": item.get("settlements", []),
            }
        )
    return {
        "round_number": round_metadata.get("round_number"),
        "round_weight": round_metadata.get("round_weight", 1.0),
        "initial_states": initial_states,
    }


def evaluate_round(
    manifest: dict,
    round_data: dict,
    transition_table: dict | None,
    simple_prior: dict | None,
    opt_tables: dict | None,
    alpha: float,
    coherence: float | None,
    multiplier: float,
) -> dict:
    original_multiplier = diamond.COHERENCE_STRENGTH_MULTIPLIER
    original_bt_multiplier = bt.diamond.COHERENCE_STRENGTH_MULTIPLIER
    original_infer = diamond.infer_spatial_coherence
    original_bt_infer = bt.diamond.infer_spatial_coherence

    try:
        diamond.COHERENCE_STRENGTH_MULTIPLIER = multiplier
        bt.diamond.COHERENCE_STRENGTH_MULTIPLIER = multiplier

        if coherence is not None:
            def fixed_infer(_observers):
                return float(coherence), 1.0

            diamond.infer_spatial_coherence = fixed_infer
            bt.diamond.infer_spatial_coherence = fixed_infer

        result = bt.run_history_replay_round(
            round_id=manifest["round_id"],
            round_data=round_data,
            manifest=manifest,
            transition_table=transition_table,
            simple_prior=simple_prior,
            opt_tables=opt_tables,
            alpha=alpha,
            ground_truth_by_seed=local_ground_truth_by_seed(manifest),
        )
        return result
    finally:
        diamond.COHERENCE_STRENGTH_MULTIPLIER = original_multiplier
        bt.diamond.COHERENCE_STRENGTH_MULTIPLIER = original_bt_multiplier
        diamond.infer_spatial_coherence = original_infer
        bt.diamond.infer_spatial_coherence = original_bt_infer


def summarize_result(tag: str, result: dict) -> dict:
    return {
        "tag": tag,
        "round_score": result["round_score"],
        "weighted_round_score": result["weighted_round_score"],
        "world_type": result.get("world_type"),
        "spatial_coherence": result.get("spatial_coherence"),
        "coherence_evidence": result.get("coherence_evidence"),
        "mean_weighted_kl": result.get("mean_weighted_kl"),
        "seed_scores": [
            {
                "seed_index": seed["seed_index"],
                "score": seed["score"],
                "weighted_kl": seed["weighted_kl"],
            }
            for seed in result.get("seed_results", [])
        ],
    }


def search_grid(
    manifest: dict,
    round_data: dict,
    transition_table: dict | None,
    simple_prior: dict | None,
    opt_tables: dict | None,
    alpha: float,
    coherences: list[float],
    multipliers: list[float],
) -> list[dict]:
    rows = []
    for multiplier in multipliers:
        for coherence in coherences:
            result = evaluate_round(
                manifest=manifest,
                round_data=round_data,
                transition_table=transition_table,
                simple_prior=simple_prior,
                opt_tables=opt_tables,
                alpha=alpha,
                coherence=coherence,
                multiplier=multiplier,
            )
            rows.append(
                {
                    "coherence": round(float(coherence), 6),
                    "multiplier": round(float(multiplier), 6),
                    "round_score": result["round_score"],
                    "weighted_round_score": result["weighted_round_score"],
                    "world_type": result.get("world_type"),
                    "mean_weighted_kl": result.get("mean_weighted_kl"),
                }
            )
    rows.sort(key=lambda item: item["round_score"], reverse=True)
    return rows


def float_grid(start: float, stop: float, step: float) -> list[float]:
    values = []
    x = float(start)
    while x <= stop + 1e-9:
        values.append(round(x, 6))
        x += step
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc optimize coherence for one round")
    parser.add_argument("--round-id", default=None)
    parser.add_argument("--round-number", type=int, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    round_id = resolve_round_id(args.round_id, args.round_number)
    manifest = replay_manifest(round_id)

    if not manifest.get("ground_truth"):
        raise SystemExit(f"Runden {round_id} mangler ground truth i lokal history")
    if not manifest.get("queries"):
        raise SystemExit(f"Runden {round_id} mangler lagrede queries")

    transition_table, simple_prior = diamond.load_calibration()
    _ = diamond.load_calibration_by_type()
    opt_tables = diamond.load_optimized_calibration()
    learning = diamond.load_learning_state()
    alpha = float(learning.get("alpha", diamond.DEFAULT_ALPHA))
    round_data = local_round_data(manifest)

    baseline = evaluate_round(
        manifest=manifest,
        round_data=round_data,
        transition_table=transition_table,
        simple_prior=simple_prior,
        opt_tables=opt_tables,
        alpha=alpha,
        coherence=None,
        multiplier=diamond.COHERENCE_STRENGTH_MULTIPLIER,
    )

    inferred_coherence = float(baseline.get("spatial_coherence") or 0.5)
    coarse_rows = search_grid(
        manifest=manifest,
        round_data=round_data,
        transition_table=transition_table,
        simple_prior=simple_prior,
        opt_tables=opt_tables,
        alpha=alpha,
        coherences=float_grid(0.0, 1.0, 0.10),
        multipliers=float_grid(0.0, 10.0, 1.00),
    )

    coarse_best = coarse_rows[0]
    fine_rows = search_grid(
        manifest=manifest,
        round_data=round_data,
        transition_table=transition_table,
        simple_prior=simple_prior,
        opt_tables=opt_tables,
        alpha=alpha,
        coherences=float_grid(
            max(0.0, float(coarse_best["coherence"]) - 0.10),
            min(1.0, float(coarse_best["coherence"]) + 0.10),
            0.02,
        ),
        multipliers=float_grid(
            max(0.0, float(coarse_best["multiplier"]) - 1.0),
            min(10.0, float(coarse_best["multiplier"]) + 1.0),
            0.25,
        ),
    )

    best_row = fine_rows[0]
    best = evaluate_round(
        manifest=manifest,
        round_data=round_data,
        transition_table=transition_table,
        simple_prior=simple_prior,
        opt_tables=opt_tables,
        alpha=alpha,
        coherence=float(best_row["coherence"]),
        multiplier=float(best_row["multiplier"]),
    )
    neutral = evaluate_round(
        manifest=manifest,
        round_data=round_data,
        transition_table=transition_table,
        simple_prior=simple_prior,
        opt_tables=opt_tables,
        alpha=alpha,
        coherence=0.5,
        multiplier=diamond.COHERENCE_STRENGTH_MULTIPLIER,
    )
    no_mod = evaluate_round(
        manifest=manifest,
        round_data=round_data,
        transition_table=transition_table,
        simple_prior=simple_prior,
        opt_tables=opt_tables,
        alpha=alpha,
        coherence=inferred_coherence,
        multiplier=0.0,
    )

    payload = {
        "round_id": round_id,
        "round_number": round_data.get("round_number"),
        "round_weight": round_data.get("round_weight"),
        "alpha": alpha,
        "current_multiplier": diamond.COHERENCE_STRENGTH_MULTIPLIER,
        "baseline": summarize_result("baseline", baseline),
        "neutral_coherence": summarize_result("neutral_coherence", neutral),
        "no_modulation": summarize_result("no_modulation", no_mod),
        "best": summarize_result("best", best),
        "best_delta_vs_baseline": round(float(best["round_score"] - baseline["round_score"]), 6),
        "coarse_top_10": coarse_rows[:10],
        "fine_top_20": fine_rows[:20],
    }
    args.output.write_text(json.dumps(payload, indent=2))

    print(
        f"Round {payload['round_number']} baseline={baseline['round_score']:.4f} "
        f"(coh={baseline.get('spatial_coherence'):.4f}, mult={diamond.COHERENCE_STRENGTH_MULTIPLIER:.2f})"
    )
    print(
        f"Neutral coherence={neutral['round_score']:.4f}  "
        f"No modulation={no_mod['round_score']:.4f}"
    )
    print(
        f"Best={best['round_score']:.4f} "
        f"(coh={best.get('spatial_coherence'):.4f}, mult={best_row['multiplier']:.2f})  "
        f"delta={payload['best_delta_vs_baseline']:+.4f}"
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
