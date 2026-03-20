"""Backtest v1 vs v2 solver on historical rounds with ground truth.

Saves v2 predictions as .npy files alongside existing history data so that
render_history_viewer.py can display them as a third column.
"""

import json
import logging
from pathlib import Path

import numpy as np

from astar_solver.config import SolverConfig as V1Config
from astar_solver.observations import RoundObservationStore as V1ObservationStore
from astar_solver.predictor import ProbabilisticMapPredictor as V1Predictor
from astar_solver.types import (
    ObservedSettlement,
    SeedState,
    Settlement,
    Viewport,
    ViewportObservation,
)

from astar_solver_v2.config import SolverConfig as V2Config
from astar_solver_v2.observations import RoundObservationStore as V2ObservationStore
from astar_solver_v2.predictor import ProbabilisticMapPredictor as V2Predictor

HISTORY_ROOT = Path("history")


def kl_divergence(gt: np.ndarray, pred: np.ndarray) -> float:
    eps = 1e-12
    kl = np.sum(gt * np.log(np.clip(gt, eps, 1.0) / np.clip(pred, eps, 1.0)), axis=-1)
    return float(kl.mean())


def kl_divergence_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    eps = 1e-12
    return np.sum(gt * np.log(np.clip(gt, eps, 1.0) / np.clip(pred, eps, 1.0)), axis=-1)


def kl_to_score(kl: float) -> float:
    return max(0.0, 100.0 * (1.0 - kl / 2.5))


def reconstruct_seed_states(manifest: dict) -> list[SeedState]:
    states = []
    for item in manifest["initial_states"]:
        grid = np.load(HISTORY_ROOT / manifest["round_id"] / item["grid_path"])
        settlements = [
            Settlement(x=s["x"], y=s["y"], has_port=s.get("has_port", False))
            for s in item["settlements"]
        ]
        states.append(SeedState(
            seed_index=item["seed_index"],
            grid=grid,
            settlements=settlements,
            static_mask=np.isin(grid, [10, 11]),
        ))
    return states


def reconstruct_observations(manifest: dict, round_id: str) -> list[ViewportObservation]:
    observations = []
    for query in manifest["queries"]:
        grid = np.load(HISTORY_ROOT / round_id / query["grid_path"])
        vp = query["viewport"]
        settlements = [
            ObservedSettlement(
                x=s["x"], y=s["y"],
                population=s.get("population", 0.0),
                food=s.get("food", 0.0),
                wealth=s.get("wealth", 0.0),
                defense=s.get("defense", 0.0),
                has_port=s.get("has_port", False),
                alive=s.get("alive", False),
                owner_id=s.get("owner_id", -1),
            )
            for s in query.get("settlements", [])
        ]
        observations.append(ViewportObservation(
            round_id=round_id,
            seed_index=query["seed_index"],
            viewport=Viewport(x=vp["x"], y=vp["y"], w=vp["w"], h=vp["h"]),
            grid=grid,
            settlements=settlements,
            queries_used=query.get("queries_used"),
            queries_max=query.get("queries_max"),
        ))
    return observations


def load_ground_truth(manifest: dict, round_id: str) -> dict[int, np.ndarray]:
    gt = {}
    for seed_str, path in manifest.get("ground_truth", {}).items():
        gt[int(seed_str)] = np.load(HISTORY_ROOT / round_id / path)
    return gt


def run_predictor(predictor, observation_store, seed_states) -> dict[int, np.ndarray]:
    predictions = {}
    for seed_state in seed_states:
        artifacts = predictor.predict_seed(seed_state, observation_store)
        predictions[seed_state.seed_index] = artifacts.probabilities
    return predictions


def main():
    logging.basicConfig(level=logging.WARNING)

    all_rounds = []
    for round_dir in sorted(HISTORY_ROOT.iterdir()):
        manifest_path = round_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest.get("queries"):
            all_rounds.append(manifest)

    if not all_rounds:
        print("No rounds with queries found.")
        return

    print(f"Found {len(all_rounds)} rounds with queries\n")
    print(f"{'Round':>6} {'ID':>10} | {'V1 KL':>8} {'V1 Score':>9} | {'V2 KL':>8} {'V2 Score':>9} | {'Delta':>7} | Saved")
    print("-" * 95)

    v1_scores = []
    v2_scores = []
    class_names = ["empty", "settl", "port", "ruin", "forest", "mount"]

    for manifest in sorted(all_rounds, key=lambda m: m["round_metadata"].get("round_number", 0)):
        round_id = manifest["round_id"]
        round_num = manifest["round_metadata"].get("round_number", "?")
        round_dir = HISTORY_ROOT / round_id / "arrays"

        seed_states = reconstruct_seed_states(manifest)
        observations = reconstruct_observations(manifest, round_id)
        ground_truth = load_ground_truth(manifest, round_id)

        # --- V1 (recompute fresh for fair comparison) ---
        v1_config = V1Config()
        v1_predictor = V1Predictor(v1_config)
        v1_store = V1ObservationStore(seed_states)
        for obs in observations:
            v1_store.add_observation(obs)
        v1_preds = run_predictor(v1_predictor, v1_store, seed_states)

        # --- V2 ---
        v2_config = V2Config()
        v2_predictor = V2Predictor(v2_config)
        v2_store = V2ObservationStore(seed_states)
        for obs in observations:
            v2_store.add_observation(obs)
        v2_preds = run_predictor(v2_predictor, v2_store, seed_states)

        # Save v2 predictions as .npy files
        saved_count = 0
        for seed_idx, pred in v2_preds.items():
            out_path = round_dir / f"seed_{seed_idx}_prediction_v2.npy"
            np.save(out_path, pred)
            saved_count += 1

        # Compare if ground truth available
        if ground_truth:
            v1_kls = []
            v2_kls = []
            v1_class_kls = [[] for _ in range(6)]
            v2_class_kls = [[] for _ in range(6)]
            for seed_idx in sorted(ground_truth.keys()):
                gt = ground_truth[seed_idx]
                if seed_idx in v1_preds:
                    v1_kls.append(kl_divergence(gt, v1_preds[seed_idx]))
                if seed_idx in v2_preds:
                    v2_kls.append(kl_divergence(gt, v2_preds[seed_idx]))
                eps = 1e-12
                for c in range(6):
                    gt_c = np.clip(gt[..., c], eps, 1.0)
                    if seed_idx in v1_preds:
                        p1_c = np.clip(v1_preds[seed_idx][..., c], eps, 1.0)
                        v1_class_kls[c].append(float((gt_c * np.log(gt_c / p1_c)).mean()))
                    if seed_idx in v2_preds:
                        p2_c = np.clip(v2_preds[seed_idx][..., c], eps, 1.0)
                        v2_class_kls[c].append(float((gt_c * np.log(gt_c / p2_c)).mean()))

            v1_mean_kl = np.mean(v1_kls)
            v2_mean_kl = np.mean(v2_kls)
            v1_score = kl_to_score(v1_mean_kl)
            v2_score = kl_to_score(v2_mean_kl)
            delta = v2_score - v1_score
            v1_scores.append(v1_score)
            v2_scores.append(v2_score)
            sign = "+" if delta >= 0 else ""
            print(f"{round_num:>6} {round_id[:8]:>10} | {v1_mean_kl:>8.4f} {v1_score:>8.1f}% | {v2_mean_kl:>8.4f} {v2_score:>8.1f}% | {sign}{delta:>6.1f}% | {saved_count} seeds")
            parts = []
            for c in range(6):
                v1c = np.mean(v1_class_kls[c]) if v1_class_kls[c] else 0
                v2c = np.mean(v2_class_kls[c]) if v2_class_kls[c] else 0
                d = v2c - v1c
                parts.append(f"{class_names[c]}:{d:+.4f}")
            print(f"       {'':>10}   class deltas: {', '.join(parts)}")
        else:
            print(f"{round_num:>6} {round_id[:8]:>10} | {'no GT':>8} {'':>9} | {'no GT':>8} {'':>9} | {'':>7} | {saved_count} seeds")

    if v1_scores:
        print("-" * 95)
        print(f"{'AVG':>6} {'':>10} | {'':>8} {np.mean(v1_scores):>8.1f}% | {'':>8} {np.mean(v2_scores):>8.1f}% | {'+' if np.mean(v2_scores) >= np.mean(v1_scores) else ''}{np.mean(v2_scores) - np.mean(v1_scores):>6.1f}%")

    print(f"\nV2 predictions saved to history/*/arrays/seed_*_prediction_v2.npy")
    print("Run: python render_history_viewer.py  to regenerate the viewer")


if __name__ == "__main__":
    main()
