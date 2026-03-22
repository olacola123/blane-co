#!/usr/bin/env python3
"""
Astar Island Night Bot (Diamond solver version)
================================================
Polls for active rounds and solves them using solution_diamond.py.

Usage:
    export API_KEY='din-jwt-token'
    python3 nightbot_diamond.py                     # Run with defaults
    python3 nightbot_diamond.py --poll-interval 180 # Poll every 3 min
    python3 nightbot_diamond.py --dry-run           # Poll + log without submitting
    nohup python3 nightbot_diamond.py &             # Run in background
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import solution_diamond as diamond

STATE_FILE = PROJECT_ROOT / "nightbot_diamond_state.json"
LOG_FILE = PROJECT_ROOT / "nightbot_diamond.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),
    ],
)
logger = logging.getLogger("nightbot_diamond")


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"solved_rounds": [], "round_scores": {}}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def fetch_analysis_scores(client, round_id: str, n_seeds: int) -> dict | None:
    """Try to fetch analysis scores for a completed round."""
    scores = []
    for seed_index in range(n_seeds):
        try:
            analysis = client.get(f"/analysis/{round_id}/{seed_index}")
            score = analysis.get("score")
            if score is not None:
                scores.append(float(score))
            else:
                # Compute from ground truth
                gt = analysis.get("ground_truth")
                if gt is not None:
                    scores.append(None)  # Can't compute without prediction
            time.sleep(0.1)
        except Exception:
            return None  # Not ready yet
    if scores and all(s is not None for s in scores):
        return {
            "mean_score": sum(scores) / len(scores),
            "scores": scores,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
    return None


def solve_active_round(client, round_data, state, dry_run=False):
    """Solve one active round using solution_diamond."""
    round_id = round_data["id"]
    rnum = round_data.get("round_number", "?")
    weight = round_data.get("round_weight", 1.0)
    closes_at = round_data.get("closes_at", "?")

    logger.info(f"=== Round {rnum} (weight={weight:.2f}, closes={closes_at}) ===")

    if round_id in state["solved_rounds"]:
        logger.info(f"  Already solved, skipping")
        return

    # Check budget
    try:
        budget = client.get_budget()
        used = budget.get("queries_used", 0)
        total = budget.get("queries_max", 50)
        if used >= total:
            logger.warning(f"  Budget exhausted ({used}/{total}), skipping")
            state["solved_rounds"].append(round_id)
            save_state(state)
            return
        logger.info(f"  Budget: {used}/{total}")
    except Exception as e:
        logger.warning(f"  Budget check failed: {e}")

    if dry_run:
        logger.info(f"  [DRY RUN] Would solve round {rnum}")
        return

    # Load calibration
    transition_table, simple_prior = diamond.load_calibration()
    type_tables = diamond.load_calibration_by_type()
    opt_tables = diamond.load_optimized_calibration()
    learning = diamond.load_learning_state()
    alpha = learning.get("alpha", diamond.DEFAULT_ALPHA)

    try:
        results = diamond.solve_round(
            client, round_id, round_data,
            transition_table, simple_prior,
            queries_per_seed=10,
            submit=True,
            alpha=alpha,
            type_tables=type_tables,
            safety_submit=True,
            opt_tables=opt_tables,
        )

        # Log results
        scores = []
        for r in results:
            s = r.get("score", "?")
            logger.info(f"  Seed {r['seed_index']}: {s}")
            if isinstance(r.get("score"), (int, float)):
                scores.append(r["score"])

        if scores:
            avg = sum(scores) / len(scores)
            logger.info(f"  Average: {avg:.1f}, Weighted: {avg * weight:.1f}")

        state["solved_rounds"].append(round_id)
        save_state(state)

        # Save results
        out = PROJECT_ROOT / "results_diamond.json"
        out.write_text(json.dumps({
            "round_id": round_id,
            "round_number": rnum,
            "results": results,
            "history_dir": str(diamond.HISTORY_ROOT / round_id),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, indent=2, default=str))

        logger.info(f"  Round {rnum} solved successfully")

    except Exception as e:
        logger.error(f"  FAILED to solve round {rnum}: {e}")
        logger.error(traceback.format_exc())


def check_pending_analyses(client, state):
    """Check for completed rounds and fetch their analysis scores."""
    try:
        rounds = client.get_rounds()
    except Exception:
        return

    for round_id in list(state["solved_rounds"]):
        if round_id in state.get("round_scores", {}):
            continue

        # Find round info
        round_info = next((r for r in rounds if r.get("id") == round_id), None)
        if not round_info or round_info.get("status") != "completed":
            continue

        rnum = round_info.get("round_number", "?")
        seeds_count = round_info.get("seeds_count", 5)

        scores = fetch_analysis_scores(client, round_id, seeds_count)
        if scores:
            state["round_scores"][round_id] = scores
            save_state(state)
            logger.info(f"  Analysis R{rnum}: avg={scores['mean_score']:.1f} {scores['scores']}")


def main():
    parser = argparse.ArgumentParser(description="Nightbot using solution_diamond.py")
    parser.add_argument("--poll-interval", type=int, default=180, help="Seconds between polls")
    parser.add_argument("--dry-run", action="store_true", help="Poll without submitting")
    args = parser.parse_args()

    if not diamond.API_KEY:
        logger.error("FEIL: export API_KEY='din-jwt-token'")
        sys.exit(1)

    logger.info(f"Nightbot Diamond started (poll every {args.poll_interval}s)")
    state = load_state()
    logger.info(f"  Previously solved: {len(state['solved_rounds'])} rounds")

    client = diamond.AstarClient()

    while True:
        try:
            rounds = client.get_rounds()
            active = [r for r in rounds if isinstance(r, dict) and r.get("status") == "active"]

            if active:
                for round_data_summary in active:
                    round_id = round_data_summary["id"]
                    if round_id not in state["solved_rounds"]:
                        # Fetch full round data
                        round_data = client.get_round(round_id)
                        solve_active_round(client, round_data, state, dry_run=args.dry_run)
            else:
                logger.debug("No active rounds")

            # Check for analysis results on previously solved rounds
            check_pending_analyses(client, state)

        except KeyboardInterrupt:
            logger.info("Shutting down")
            break
        except Exception as e:
            logger.error(f"Poll error: {e}")
            logger.error(traceback.format_exc())

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
