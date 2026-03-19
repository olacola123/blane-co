#!/usr/bin/env python3
"""
Astar Island Night Bot - Automated round solver with analysis feedback loop.

Runs continuously, polling for new rounds, solving them, fetching analyses,
and logging diagnostics so we learn from every round overnight.

Usage:
    python nightbot.py                    # Run with defaults
    python nightbot.py --poll-interval 180  # Poll every 3 min
    python nightbot.py --dry-run          # Poll + log without submitting
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

# Ensure the solver package is importable
sys.path.insert(0, str(PROJECT_ROOT))

from astar_solver import AstarClient, RoundSolver, SolverConfig
from astar_solver.constants import DEFAULT_TOTAL_QUERIES
from astar_solver.evaluation import calibration_diagnostics, weighted_kl
from astar_solver.history import RoundDatasetStore

HISTORY_ROOT = str(PROJECT_ROOT / "history")
STATE_FILE = PROJECT_ROOT / "nightbot_state.json"
DIAGNOSTICS_LOG = PROJECT_ROOT / "nightbot_diagnostics.jsonl"

logger = logging.getLogger("nightbot")


def load_state() -> dict:
    """Load persisted bot state (solved rounds, pending analysis, etc.)."""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"solved_rounds": [], "pending_analysis": [], "round_scores": {}}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def log_diagnostics(entry: dict) -> None:
    """Append a diagnostics entry to the JSONL log."""
    with open(DIAGNOSTICS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def fetch_and_score_analyses(
    client: AstarClient,
    history_store: RoundDatasetStore,
    round_id: str,
    seed_count: int,
    timeout: float = 300.0,
    poll_interval: float = 30.0,
) -> dict | None:
    """Poll for analysis data for a round. Returns diagnostics dict or None."""
    logger.info("Waiting for analysis for round %s (timeout %.0fs)...", round_id[:12], timeout)
    deadline = time.time() + timeout
    analyses = {}

    while time.time() < deadline:
        for seed_index in range(seed_count):
            if seed_index in analyses:
                continue
            try:
                analysis = client.get_analysis(round_id, seed_index)
                analyses[seed_index] = analysis
                logger.info("Got analysis for seed %d", seed_index)
            except Exception:
                pass

        if len(analyses) == seed_count:
            break
        time.sleep(poll_interval)

    if not analyses:
        logger.warning("No analyses available for round %s", round_id[:12])
        return None

    # Update stored history with analyses
    history_store.update_round_analyses(round_id, analyses)

    # Compute diagnostics from stored predictions vs ground truth
    diagnostics = compute_round_diagnostics(history_store, round_id, analyses)
    return diagnostics


def compute_round_diagnostics(
    history_store: RoundDatasetStore,
    round_id: str,
    analyses: dict,
) -> dict:
    """Compare predictions to ground truth and compute scoring metrics."""
    round_dir = history_store.root / round_id
    diagnostics = {
        "round_id": round_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seeds": {},
        "summary": {},
    }

    kl_values = []

    for seed_index, analysis in analyses.items():
        seed_key = str(seed_index)

        # Load our prediction
        prediction_path = round_dir / "arrays" / f"seed_{seed_index}_prediction.npy"
        if not prediction_path.exists():
            logger.warning("No prediction file for seed %d", seed_index)
            continue

        prediction = np.load(prediction_path)

        # Extract ground truth from analysis
        ground_truth = None
        if "ground_truth" in analysis:
            gt_raw = analysis["ground_truth"]
            if isinstance(gt_raw, list):
                ground_truth = np.array(gt_raw, dtype=np.float64)
        elif "actual_distribution" in analysis:
            gt_raw = analysis["actual_distribution"]
            if isinstance(gt_raw, list):
                ground_truth = np.array(gt_raw, dtype=np.float64)

        seed_diag = {"seed_index": seed_index}

        # Use score from analysis if available
        if "score" in analysis:
            seed_diag["api_score"] = analysis["score"]
        if "kl_divergence" in analysis:
            seed_diag["api_kl"] = analysis["kl_divergence"]

        # Compute our own metrics if we have ground truth
        if ground_truth is not None and ground_truth.shape == prediction.shape:
            # Save ground truth for future training
            gt_path = round_dir / "arrays" / f"seed_{seed_index}_ground_truth.npy"
            np.save(gt_path, ground_truth)

            diag = calibration_diagnostics(ground_truth, prediction)
            seed_diag["weighted_kl"] = diag.weighted_kl_value
            seed_diag["nll"] = diag.nll
            seed_diag["brier"] = diag.brier
            seed_diag["ece"] = diag.ece
            kl_values.append(diag.weighted_kl_value)

        diagnostics["seeds"][seed_key] = seed_diag

    if kl_values:
        diagnostics["summary"] = {
            "mean_weighted_kl": float(np.mean(kl_values)),
            "std_weighted_kl": float(np.std(kl_values)),
            "min_weighted_kl": float(np.min(kl_values)),
            "max_weighted_kl": float(np.max(kl_values)),
            "num_seeds_scored": len(kl_values),
        }

    return diagnostics


def get_new_round(client: AstarClient, solved_rounds: list[str]) -> tuple[str, dict] | None:
    """Check for a round we haven't solved yet. Returns (round_id, round_data) or None."""
    try:
        rounds = client.get_rounds()
    except Exception as exc:
        logger.error("Failed to fetch rounds: %s", exc)
        return None

    if not rounds:
        return None

    for round_info in reversed(rounds):
        round_id = round_info["id"] if isinstance(round_info, dict) else round_info
        if round_id not in solved_rounds:
            try:
                round_data = client.get_round(round_id)
                return round_id, round_data
            except Exception as exc:
                logger.error("Failed to fetch round %s: %s", round_id[:12], exc)
                return None

    return None


def solve_round(
    client: AstarClient,
    round_id: str,
    round_data: dict,
    dry_run: bool = False,
) -> dict[int, object] | None:
    """Run the full solver pipeline for one round."""
    config = SolverConfig(history_root=HISTORY_ROOT)
    solver = RoundSolver(client=client, config=config)

    try:
        budget = client.get_budget()
        logger.info("Budget: %s", json.dumps(budget))
    except Exception:
        pass

    artifacts = solver.solve_round(
        round_id=round_id,
        round_data=round_data,
        queries_per_seed=10,
        total_queries=DEFAULT_TOTAL_QUERIES,
        submit=not dry_run,
        dry_run=dry_run,
    )

    for seed_index, artifact in artifacts.items():
        logger.info(
            "Seed %d: mean_entropy=%.4f mean_dynamic_mass=%.4f",
            seed_index,
            float(artifact.entropy_map.mean()),
            float(artifact.dynamic_mass.mean()),
        )

    return artifacts


def try_fetch_pending_analyses(
    client: AstarClient,
    state: dict,
    history_store: RoundDatasetStore,
) -> None:
    """Try to fetch analyses for rounds that were solved but not yet analyzed."""
    still_pending = []

    for pending in state.get("pending_analysis", []):
        round_id = pending["round_id"]
        seed_count = pending["seed_count"]
        submitted_at = pending.get("submitted_at", "")

        logger.info("Checking analysis for round %s (submitted %s)...", round_id[:12], submitted_at)

        diagnostics = fetch_and_score_analyses(
            client=client,
            history_store=history_store,
            round_id=round_id,
            seed_count=seed_count,
            timeout=60.0,  # Short timeout for pending checks
            poll_interval=10.0,
        )

        if diagnostics and diagnostics.get("seeds"):
            log_diagnostics(diagnostics)
            state["round_scores"][round_id] = diagnostics.get("summary", {})
            logger.info(
                "Round %s diagnostics: %s",
                round_id[:12],
                json.dumps(diagnostics.get("summary", {}), indent=2),
            )
        else:
            still_pending.append(pending)
            logger.info("Analysis not yet available for round %s", round_id[:12])

    state["pending_analysis"] = still_pending


def run_nightbot(
    poll_interval: int = 300,
    analysis_wait: int = 600,
    dry_run: bool = False,
    max_rounds: int = 0,
) -> None:
    """Main bot loop."""
    client = AstarClient()
    history_store = RoundDatasetStore(HISTORY_ROOT)
    state = load_state()
    rounds_solved = 0

    logger.info("=== Nightbot started ===")
    logger.info("Poll interval: %ds | Analysis wait: %ds | Dry run: %s", poll_interval, analysis_wait, dry_run)
    logger.info("Previously solved rounds: %d", len(state["solved_rounds"]))
    logger.info("Pending analysis: %d", len(state.get("pending_analysis", [])))

    while True:
        try:
            # 1. Check for pending analyses from previous rounds
            if state.get("pending_analysis"):
                try_fetch_pending_analyses(client, state, history_store)
                save_state(state)

            # 2. Check for a new round to solve
            new = get_new_round(client, state["solved_rounds"])

            if new is not None:
                round_id, round_data = new
                logger.info("=== New round found: %s ===", round_id[:12])

                try:
                    artifacts = solve_round(client, round_id, round_data, dry_run=dry_run)

                    state["solved_rounds"].append(round_id)

                    # Count seeds for analysis polling
                    seed_payloads = round_data.get("seeds", round_data.get("initial_states", []))
                    seed_count = len(seed_payloads)

                    state["pending_analysis"].append({
                        "round_id": round_id,
                        "seed_count": seed_count,
                        "submitted_at": datetime.now(timezone.utc).isoformat(),
                    })
                    save_state(state)

                    rounds_solved += 1
                    logger.info("Round %s solved and submitted. Total solved this session: %d", round_id[:12], rounds_solved)

                    if max_rounds > 0 and rounds_solved >= max_rounds:
                        logger.info("Reached max rounds (%d). Stopping.", max_rounds)
                        break

                except Exception as exc:
                    logger.error("Failed to solve round %s: %s", round_id[:12], exc)
                    logger.error(traceback.format_exc())
                    # Don't add to solved_rounds so we retry next poll
            else:
                logger.info("No new rounds. Waiting %ds...", poll_interval)

            # 3. Wait before next poll
            time.sleep(poll_interval)

        except KeyboardInterrupt:
            logger.info("=== Nightbot stopped by user ===")
            save_state(state)
            break
        except Exception as exc:
            logger.error("Unexpected error in main loop: %s", exc)
            logger.error(traceback.format_exc())
            save_state(state)
            time.sleep(60)  # Back off on unexpected errors


def main():
    parser = argparse.ArgumentParser(description="Astar Island Night Bot")
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=300,
        help="Seconds between round checks (default: 300 = 5 min)",
    )
    parser.add_argument(
        "--analysis-wait",
        type=int,
        default=600,
        help="Max seconds to wait for analysis after submission (default: 600)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Poll and log without submitting",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=0,
        help="Stop after N rounds (0 = unlimited)",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Clear solved rounds state and start fresh",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                PROJECT_ROOT / "nightbot.log",
                encoding="utf-8",
            ),
        ],
    )

    if args.reset_state and STATE_FILE.exists():
        STATE_FILE.unlink()
        logger.info("State file cleared.")

    run_nightbot(
        poll_interval=args.poll_interval,
        analysis_wait=args.analysis_wait,
        dry_run=args.dry_run,
        max_rounds=args.max_rounds,
    )


if __name__ == "__main__":
    main()
