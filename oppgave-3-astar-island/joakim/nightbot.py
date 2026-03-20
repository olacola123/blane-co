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
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

# Ensure the solver package is importable
sys.path.insert(0, str(PROJECT_ROOT))

from astar_solver import RoundSolver, SolverConfig
from astar_solver.constants import DEFAULT_TOTAL_QUERIES
from astar_solver.evaluation import (
    bucketed_error_diagnostics,
    calibration_diagnostics,
    classwise_expected_calibration_error,
)
from astar_solver.features import MapFeatureExtractor
from astar_solver.history import RoundDatasetStore
from astar_solver.tuning import extract_target_tensor
from astar_solver.types import SeedState

if TYPE_CHECKING:
    from astar_solver import AstarClient as AstarClientType

HISTORY_ROOT = str(PROJECT_ROOT / "history")
STATE_FILE = PROJECT_ROOT / "nightbot_state.json"
DIAGNOSTICS_LOG = PROJECT_ROOT / "nightbot_diagnostics.jsonl"

logger = logging.getLogger("nightbot")
TRANSIENT_ANALYSIS_STATUS_CODES = {400, 404, 409, 425, 429}
AstarClient = None


def _build_client() -> "AstarClientType":
    """Create the API client lazily so tests can import this module without requests installed."""
    global AstarClient
    if AstarClient is None:
        from astar_solver import AstarClient as client_cls

        AstarClient = client_cls
    return AstarClient()


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


def bootstrap_state_from_history(history_store: RoundDatasetStore, state: dict) -> dict:
    """Reconcile state with locally stored round history."""
    bootstrapped = {
        "solved_rounds": list(state.get("solved_rounds", [])),
        "pending_analysis": list(state.get("pending_analysis", [])),
        "round_scores": dict(state.get("round_scores", {})),
    }
    solved_rounds = set(bootstrapped["solved_rounds"])
    pending_by_round = {
        str(item.get("round_id")): dict(item)
        for item in bootstrapped["pending_analysis"]
        if item.get("round_id")
    }

    if not history_store.root.exists():
        bootstrapped["solved_rounds"] = sorted(solved_rounds)
        bootstrapped["pending_analysis"] = list(pending_by_round.values())
        return bootstrapped

    for round_dir in sorted(path for path in history_store.root.iterdir() if path.is_dir()):
        manifest_path = round_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as exc:
            logger.warning("Skipping unreadable history manifest %s: %s", manifest_path, exc)
            continue

        round_id = str(manifest.get("round_id") or round_dir.name)
        seed_count = len(manifest.get("initial_states", []))
        submissions = manifest.get("submission_responses", {})
        analyses = manifest.get("analyses", {})
        ground_truth = manifest.get("ground_truth", {})
        diagnostics = manifest.get("diagnostics", {})

        fully_submitted = seed_count > 0 and len(submissions) >= seed_count
        fully_analyzed = seed_count > 0 and max(len(analyses), len(ground_truth)) >= seed_count

        if fully_submitted:
            solved_rounds.add(round_id)
            if not fully_analyzed:
                pending_by_round[round_id] = {
                    "round_id": round_id,
                    "seed_count": seed_count,
                    "submitted_at": manifest.get("saved_at_utc", ""),
                }
            elif round_id in pending_by_round:
                del pending_by_round[round_id]

        analysis_summary = diagnostics.get("analysis_summary")
        if (
            isinstance(analysis_summary, dict)
            and analysis_summary
            and (analysis_summary.get("analysis_complete") is True or fully_analyzed)
        ):
            bootstrapped["round_scores"][round_id] = analysis_summary

    bootstrapped["solved_rounds"] = sorted(solved_rounds)
    bootstrapped["pending_analysis"] = sorted(
        pending_by_round.values(),
        key=lambda item: str(item.get("submitted_at", "")),
    )
    return bootstrapped


def fetch_and_score_analyses(
    client: "AstarClientType",
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
            except Exception as exc:
                response = getattr(exc, "response", None)
                status_code = getattr(response, "status_code", None)
                if status_code not in TRANSIENT_ANALYSIS_STATUS_CODES:
                    logger.warning(
                        "Analysis fetch failed for round %s seed %d status=%s error=%s",
                        round_id[:12],
                        seed_index,
                        status_code,
                        exc,
                    )

        if len(analyses) == seed_count:
            break
        time.sleep(poll_interval)

    stored_analyses: dict[int, dict] = {}
    if analyses:
        history_store.update_round_analyses(round_id, analyses)
    round_bundle = history_store.load_round(round_id)
    stored_analyses = {
        int(seed_key): payload
        for seed_key, payload in round_bundle["manifest"].get("analyses", {}).items()
    }

    if not stored_analyses:
        logger.warning("No analyses available for round %s", round_id[:12])
        return None

    diagnostics = compute_round_diagnostics(history_store, round_id)
    history_store.update_round_diagnostics(
        round_id,
        {
            "analysis": diagnostics.get("seeds", {}),
            "analysis_summary": diagnostics.get("summary", {}),
        },
    )
    return diagnostics


def _seed_states_from_bundle(bundle: dict) -> dict[int, SeedState]:
    """Rebuild typed seed states from a stored round bundle."""
    seed_states: dict[int, SeedState] = {}
    manifest = bundle["manifest"]
    for item in manifest.get("initial_states", []):
        seed_index = int(item["seed_index"])
        grid = bundle["initial_grids"].get(seed_index)
        if grid is None:
            continue
        seed_states[seed_index] = SeedState.from_round_data(
            seed_index=seed_index,
            seed_data={
                "grid": grid.tolist(),
                "settlements": item.get("settlements", []),
            },
        )
    return seed_states


def compute_round_diagnostics(
    history_store: RoundDatasetStore,
    round_id: str,
    analyses: dict | None = None,
) -> dict:
    """Compare predictions to ground truth and compute scoring metrics."""
    bundle = history_store.load_round(round_id)
    if analyses is None:
        analyses = {
            int(seed_key): payload
            for seed_key, payload in bundle["manifest"].get("analyses", {}).items()
        }
    seed_states = _seed_states_from_bundle(bundle)
    feature_extractor = MapFeatureExtractor()
    feature_cache = {
        seed_index: feature_extractor.extract(seed_state)
        for seed_index, seed_state in seed_states.items()
    }
    diagnostics = {
        "round_id": round_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seeds": {},
        "summary": {},
    }

    kl_values = []
    expected_seed_count = len(bundle["manifest"].get("initial_states", []))

    for seed_index, analysis in analyses.items():
        seed_key = str(seed_index)

        # Load our prediction
        prediction = bundle["predictions"].get(seed_index)
        if prediction is None:
            logger.warning("No prediction file for seed %d", seed_index)
            continue

        # Extract ground truth from analysis
        ground_truth = extract_target_tensor(analysis)
        if ground_truth is None:
            ground_truth = bundle["ground_truth"].get(seed_index)

        seed_diag = {"seed_index": seed_index}

        # Use score from analysis if available
        if "score" in analysis:
            seed_diag["api_score"] = analysis["score"]
        if "kl_divergence" in analysis:
            seed_diag["api_kl"] = analysis["kl_divergence"]

        # Compute our own metrics if we have ground truth
        if ground_truth is not None and ground_truth.shape == prediction.shape:
            diag = calibration_diagnostics(ground_truth, prediction)
            seed_diag["weighted_kl"] = diag.weighted_kl_value
            seed_diag["nll"] = diag.nll
            seed_diag["brier"] = diag.brier
            seed_diag["ece"] = diag.ece
            seed_diag["classwise_ece"] = classwise_expected_calibration_error(ground_truth, prediction)
            if seed_index in feature_cache:
                seed_diag["bucketed_kl"] = bucketed_error_diagnostics(
                    ground_truth,
                    prediction,
                    feature_cache[seed_index],
                )
            kl_values.append(diag.weighted_kl_value)

        diagnostics["seeds"][seed_key] = seed_diag

    diagnostics["summary"] = {
        "expected_seed_count": expected_seed_count,
        "num_analysis_payloads": len(analyses),
        "num_ground_truth_tensors": len(bundle["ground_truth"]),
        "analysis_complete": expected_seed_count > 0 and len(analyses) >= expected_seed_count,
    }

    if kl_values:
        diagnostics["summary"].update(
            {
            "mean_weighted_kl": float(np.mean(kl_values)),
            "std_weighted_kl": float(np.std(kl_values)),
            "min_weighted_kl": float(np.min(kl_values)),
            "max_weighted_kl": float(np.max(kl_values)),
            "num_seeds_scored": len(kl_values),
            }
        )

    return diagnostics


def get_new_round(client: "AstarClientType", solved_rounds: list[str]) -> tuple[str, dict] | None:
    """Check for a round we haven't solved yet. Returns (round_id, round_data) or None."""
    try:
        rounds = client.get_rounds()
    except Exception as exc:
        logger.error("Failed to fetch rounds: %s", exc)
        return None

    if not rounds:
        return None

    solved_set = set(solved_rounds)
    highest_known_round_number = -1
    for round_info in rounds:
        if not isinstance(round_info, dict):
            continue
        round_id = str(round_info.get("id", ""))
        if round_id not in solved_set:
            continue
        round_number = round_info.get("round_number")
        if isinstance(round_number, int):
            highest_known_round_number = max(highest_known_round_number, round_number)

    def round_sort_key(round_info) -> tuple[int, str]:
        if not isinstance(round_info, dict):
            return (-1, str(round_info))
        round_number = round_info.get("round_number")
        if not isinstance(round_number, int):
            round_number = -1
        tie_breaker = str(round_info.get("started_at") or round_info.get("event_date") or "")
        return (round_number, tie_breaker)

    for round_info in sorted(rounds, key=round_sort_key, reverse=True):
        round_id = round_info["id"] if isinstance(round_info, dict) else round_info
        if round_id in solved_set:
            continue
        round_number = round_info.get("round_number") if isinstance(round_info, dict) else None
        if isinstance(round_number, int) and round_number <= highest_known_round_number:
            continue
        try:
            round_data = client.get_round(round_id)
            return round_id, round_data
        except Exception as exc:
            logger.error("Failed to fetch round %s: %s", round_id[:12], exc)
            return None

    return None


def solve_round(
    client: "AstarClientType",
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
    client: "AstarClientType",
    state: dict,
    history_store: RoundDatasetStore,
    timeout: float = 60.0,
    poll_interval: float = 10.0,
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
            timeout=timeout,
            poll_interval=poll_interval,
        )

        analysis_complete = bool(diagnostics and diagnostics.get("summary", {}).get("analysis_complete"))
        if diagnostics and diagnostics.get("seeds") and analysis_complete:
            log_diagnostics(diagnostics)
            state["round_scores"][round_id] = diagnostics.get("summary", {})
            logger.info(
                "Round %s diagnostics: %s",
                round_id[:12],
                json.dumps(diagnostics.get("summary", {}), indent=2),
            )
        else:
            still_pending.append(pending)
            if diagnostics and diagnostics.get("seeds"):
                logger.info(
                    "Partial analysis for round %s: %s/%s payloads available",
                    round_id[:12],
                    diagnostics["summary"].get("num_analysis_payloads", 0),
                    diagnostics["summary"].get("expected_seed_count", seed_count),
                )
            else:
                logger.info("Analysis not yet available for round %s", round_id[:12])

    state["pending_analysis"] = still_pending


def run_nightbot(
    poll_interval: int = 300,
    analysis_wait: int = 600,
    dry_run: bool = False,
    max_rounds: int = 0,
) -> None:
    """Main bot loop."""
    client = _build_client()
    history_store = RoundDatasetStore(HISTORY_ROOT)
    loaded_state = load_state()
    state = bootstrap_state_from_history(history_store, loaded_state)
    if state != loaded_state:
        save_state(state)
    rounds_solved = 0
    known_rounds = set(state["solved_rounds"])
    pending_timeout = min(60.0, max(float(poll_interval) / 2.0, 15.0))
    pending_poll_interval = min(10.0, max(pending_timeout / 6.0, 1.0))

    logger.info("=== Nightbot started ===")
    logger.info("Poll interval: %ds | Analysis wait: %ds | Dry run: %s", poll_interval, analysis_wait, dry_run)
    logger.info("Previously solved rounds: %d", len(state["solved_rounds"]))
    logger.info("Pending analysis: %d", len(state.get("pending_analysis", [])))

    while True:
        try:
            # 1. Check for pending analyses from previous rounds
            if state.get("pending_analysis"):
                try_fetch_pending_analyses(
                    client,
                    state,
                    history_store,
                    timeout=pending_timeout,
                    poll_interval=pending_poll_interval,
                )
                save_state(state)

            # 2. Check for a new round to solve
            new = get_new_round(client, list(known_rounds))

            if new is not None:
                round_id, round_data = new
                logger.info("=== New round found: %s ===", round_id[:12])

                try:
                    artifacts = solve_round(client, round_id, round_data, dry_run=dry_run)
                    known_rounds.add(round_id)

                    if not dry_run:
                        if round_id not in state["solved_rounds"]:
                            state["solved_rounds"].append(round_id)

                        # Count seeds for analysis polling
                        seed_payloads = round_data.get("seeds", round_data.get("initial_states", []))
                        seed_count = len(seed_payloads)

                        pending_entry = {
                            "round_id": round_id,
                            "seed_count": seed_count,
                            "submitted_at": datetime.now(timezone.utc).isoformat(),
                        }
                        state["pending_analysis"] = [
                            pending
                            for pending in state.get("pending_analysis", [])
                            if pending.get("round_id") != round_id
                        ]
                        state["pending_analysis"].append(pending_entry)
                        save_state(state)

                        if analysis_wait > 0:
                            diagnostics = fetch_and_score_analyses(
                                client=client,
                                history_store=history_store,
                                round_id=round_id,
                                seed_count=seed_count,
                                timeout=float(analysis_wait),
                                poll_interval=min(10.0, max(float(analysis_wait) / 12.0, 1.0)),
                            )
                            analysis_complete = bool(
                                diagnostics and diagnostics.get("summary", {}).get("analysis_complete")
                            )
                            if diagnostics and diagnostics.get("seeds") and analysis_complete:
                                log_diagnostics(diagnostics)
                                state["round_scores"][round_id] = diagnostics.get("summary", {})
                                state["pending_analysis"] = [
                                    pending
                                    for pending in state.get("pending_analysis", [])
                                    if pending.get("round_id") != round_id
                                ]
                                save_state(state)
                                logger.info(
                                    "Round %s diagnostics: %s",
                                    round_id[:12],
                                    json.dumps(diagnostics.get("summary", {}), indent=2),
                                )
                            elif diagnostics and diagnostics.get("seeds"):
                                save_state(state)
                                logger.info(
                                    "Partial analysis for round %s: %s/%s payloads available",
                                    round_id[:12],
                                    diagnostics["summary"].get("num_analysis_payloads", 0),
                                    diagnostics["summary"].get("expected_seed_count", seed_count),
                                )

                    rounds_solved += 1
                    if dry_run:
                        logger.info(
                            "Round %s dry-run completed. Total processed this session: %d",
                            round_id[:12],
                            rounds_solved,
                        )
                    else:
                        logger.info(
                            "Round %s solved and submitted. Total solved this session: %d",
                            round_id[:12],
                            rounds_solved,
                        )

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
