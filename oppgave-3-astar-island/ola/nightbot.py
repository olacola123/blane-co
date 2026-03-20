#!/usr/bin/env python3
"""
Ola's Astar Island Nightbot — Selvforbedrende
================================================
Kjører automatisk, løser nye runder, og LÆRER av hver runde.

Syklus per runde:
1. Ny runde oppdaget → kjør solver → submit
2. Runden stenger → hent fasit (ground truth)
3. Oppdater overgangstabellen med nye data
4. Neste runde bruker bedre priors

Jo flere runder, jo bedre priors → jo høyere score.

Bruk:
    export API_KEY='din-jwt-token'
    python nightbot.py                     # Start bot
    python nightbot.py --poll-interval 120 # Poll hvert 2 min
    python nightbot.py --dry-run           # Ingen submit
    python nightbot.py --max-rounds 3      # Stopp etter 3 runder
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from solution import (
    AstarClient, load_calibration, solve_round,
    MAP_W, MAP_H, NUM_CLASSES, TERRAIN_TO_CLASS,
    DISTANCE_BANDS, PROB_FLOOR,
    distance_to_nearest_settlement, get_distance_band, is_coastal,
)

PROJECT_DIR = Path(__file__).parent
STATE_FILE = PROJECT_DIR / "nightbot_state.json"
LOG_FILE = PROJECT_DIR / "nightbot.log"
CALIBRATION_FILE = PROJECT_DIR / "calibration_data.json"

API_KEY = os.environ.get("API_KEY", "")

logger = logging.getLogger("ola-nightbot")


# === STATE ===

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"solved_rounds": [], "scores": {}, "calibrated_rounds": []}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


# === ROUND FINDING ===

def find_new_round(client, solved_rounds):
    """Finn en aktiv runde vi ikke har løst."""
    try:
        rounds = client.get_rounds()
    except Exception as e:
        logger.error(f"Kunne ikke hente runder: {e}")
        return None, None

    solved_set = set(solved_rounds)

    for r in reversed(rounds):
        if not isinstance(r, dict):
            continue
        rid = r.get("id", "")
        status = r.get("status", "")

        if status == "active" and rid not in solved_set:
            try:
                round_data = client.get_round(rid)
                return rid, round_data
            except Exception as e:
                logger.error(f"Kunne ikke hente runde {rid[:12]}: {e}")
                return None, None

    return None, None


# === SELVFORBEDRING: REKALIBRERING ===

def recalibrate(client, state):
    """
    Hent ground truth fra nye fullførte runder og OPPDATER calibration_data.json.

    Dette er kjernen i selvforbedringen:
    - Etter hver fullført runde henter vi fasiten
    - Vi beregner nye overgangstabeller fra ALL tilgjengelig data
    - Neste runde bruker oppdaterte priors
    """
    try:
        rounds = client.get_rounds()
    except Exception:
        return False

    completed = [r for r in rounds if isinstance(r, dict) and r.get("status") == "completed"]
    already_calibrated = set(state.get("calibrated_rounds", []))

    # Finn nye fullførte runder
    new_round_ids = [r["id"] for r in completed if r["id"] not in already_calibrated]

    if not new_round_ids:
        return False

    logger.info(f"Fant {len(new_round_ids)} nye fullførte runder å kalibrere fra")

    # Hent ground truth for nye runder
    new_entries = []
    for rid in new_round_ids:
        try:
            round_data = client.get_round(rid)
            seeds = round_data.get("seeds", round_data.get("initial_states", []))
            rnum = round_data.get("round_number", "?")

            for seed_idx in range(len(seeds)):
                try:
                    analysis = client.get(f"/analysis/{rid}/{seed_idx}")
                    gt = analysis.get("ground_truth")
                    score = analysis.get("score", "?")

                    if gt:
                        new_entries.append({
                            "initial_grid": seeds[seed_idx].get("grid", []),
                            "settlements": seeds[seed_idx].get("settlements", []),
                            "ground_truth": gt,
                        })
                        logger.info(f"  Runde {rnum} seed {seed_idx}: score={score}")
                except Exception:
                    pass
                time.sleep(0.3)

            already_calibrated.add(rid)
        except Exception as e:
            logger.warning(f"  Kunne ikke hente runde {rid[:12]}: {e}")

    if not new_entries:
        state["calibrated_rounds"] = list(already_calibrated)
        return False

    # Last eksisterende kalibrering og legg til nye data
    existing = {}
    if CALIBRATION_FILE.exists():
        existing = json.loads(CALIBRATION_FILE.read_text())

    # Beregn nye tabeller fra ALLE data (gammel + ny)
    # Vi rebuilder fra scratch ved å akkumulere counts
    transition_counts = defaultdict(lambda: np.zeros(NUM_CLASSES, dtype=float))
    simple_counts = defaultdict(lambda: np.zeros(NUM_CLASSES, dtype=float))

    # Hent counts fra eksisterende tabell (approximate fra distribusjon × sample_count)
    old_table = existing.get("transition_table", {})
    for key, data in old_table.items():
        dist = np.array(data["distribution"], dtype=float)
        n = data["sample_count"]
        transition_counts[key] += dist * n

    old_simple = existing.get("simple_prior", {})
    old_num_seeds = existing.get("num_seeds", 0)
    for key, dist in old_simple.items():
        # Estimér counts fra antall seeds × 1600 celler / antall terrengtyper
        simple_counts[key] += np.array(dist, dtype=float) * old_num_seeds * 200

    # Legg til nye data
    for entry in new_entries:
        grid = entry["initial_grid"]
        gt = entry["ground_truth"]
        settlements = entry["settlements"]

        if not grid or not gt:
            continue

        for y in range(len(grid)):
            for x in range(len(grid[0])):
                terrain = grid[y][x]
                gt_cell = gt[y][x]

                if isinstance(gt_cell, list) and len(gt_cell) == NUM_CLASSES:
                    gt_probs = np.array(gt_cell, dtype=float)
                elif isinstance(gt_cell, (int, float)):
                    cls = TERRAIN_TO_CLASS.get(int(gt_cell), 0)
                    gt_probs = np.zeros(NUM_CLASSES)
                    gt_probs[cls] = 1.0
                else:
                    continue

                # Simple prior
                simple_counts[str(terrain)] += gt_probs

                # Avstandsbasert (skip statiske)
                if terrain in (10, 5):
                    continue

                dist = distance_to_nearest_settlement(y, x, settlements)
                band = get_distance_band(dist)
                coastal = is_coastal(grid, y, x)
                key = f"{terrain}_{band}_{int(coastal)}"
                transition_counts[key] += gt_probs

    # Normaliser
    new_transition_table = {}
    for key, counts in transition_counts.items():
        total = counts.sum()
        if total > 0:
            new_transition_table[key] = {
                "distribution": (counts / total).tolist(),
                "sample_count": int(total),
            }

    new_simple_prior = {}
    for key, counts in simple_counts.items():
        total = counts.sum()
        if total > 0:
            new_simple_prior[key] = (counts / total).tolist()

    # Lagre oppdatert kalibrering
    total_rounds = len(already_calibrated)
    total_seeds = existing.get("num_seeds", 0) + len(new_entries)

    output = {
        "transition_table": new_transition_table,
        "simple_prior": new_simple_prior,
        "num_rounds": total_rounds,
        "num_seeds": total_seeds,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    CALIBRATION_FILE.write_text(json.dumps(output, indent=2))
    state["calibrated_rounds"] = list(already_calibrated)

    logger.info(f"Kalibrering oppdatert: {total_rounds} runder, {total_seeds} seeds, "
                f"{len(new_transition_table)} nøkler")

    return True


# === HOVEDLOOP ===

def run_bot(poll_interval=120, dry_run=False, max_rounds=0):
    client = AstarClient()
    transition_table, simple_prior = load_calibration()
    state = load_state()
    rounds_solved = 0

    logger.info("=== Ola's Nightbot startet ===")
    logger.info(f"Poll: {poll_interval}s | Dry-run: {dry_run}")
    logger.info(f"Tidligere løst: {len(state['solved_rounds'])} runder")
    logger.info(f"Kalibrering: {len(transition_table or {})} nøkler fra "
                f"{len(state.get('calibrated_rounds', []))} runder")

    while True:
        try:
            # === 1. Rekalibrering fra nye fullførte runder ===
            try:
                recalibrated = recalibrate(client, state)
                if recalibrated:
                    # Last inn oppdatert kalibrering
                    transition_table, simple_prior = load_calibration()
                    logger.info("Solver bruker nå oppdaterte priors!")
                save_state(state)
            except Exception as e:
                logger.warning(f"Rekalibrering feilet: {e}")

            # === 2. Sjekk budget ===
            try:
                budget = client.get_budget()
                budget_used = budget.get("queries_used", 0)
                budget_max = budget.get("queries_max", 50)
                logger.info(f"Budget: {budget_used}/{budget_max}")
            except Exception:
                budget_used = 0

            # === 3. Finn og løs ny runde ===
            round_id, round_data = find_new_round(client, state["solved_rounds"])

            if round_id:
                rnum = round_data.get("round_number", "?")
                closes = round_data.get("closes_at", "?")
                weight = round_data.get("round_weight", "?")
                logger.info(f"=== Runde {rnum} (vekt {weight}) ===")
                logger.info(f"Stenger: {closes}")

                # Advarsel hvis budget allerede brukt
                try:
                    budget = client.get_budget()
                    budget_used = budget.get("queries_used", 0)
                    if budget_used > 0:
                        logger.warning(f"Budget {budget_used}/50 allerede brukt!")
                        if budget_used >= 50:
                            logger.error("Budget oppbrukt! Noen andre kjører. Skipper.")
                            state["solved_rounds"].append(round_id)
                            save_state(state)
                            time.sleep(poll_interval)
                            continue
                except Exception:
                    pass

                try:
                    results = solve_round(
                        client, round_id, round_data,
                        transition_table, simple_prior,
                        queries_per_seed=10,
                        submit=not dry_run,
                    )

                    state["solved_rounds"].append(round_id)

                    # Logg scores
                    scores = []
                    for r in results:
                        score = r.get("score")
                        if isinstance(score, (int, float)):
                            scores.append(score)

                    if scores:
                        avg = sum(scores) / len(scores)
                        state["scores"][round_id] = {
                            "round_number": rnum,
                            "avg_score": avg,
                            "per_seed": [r.get("score", "?") for r in results],
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        logger.info(f"*** Runde {rnum} ferdig! Snitt: {avg:.1f} ***")

                        # Vis trend
                        all_scores = [v["avg_score"] for v in state["scores"].values()
                                      if isinstance(v.get("avg_score"), (int, float))]
                        if len(all_scores) >= 2:
                            last3 = all_scores[-3:]
                            trend = "↑" if last3[-1] > last3[0] else "↓"
                            logger.info(f"Trend: {' → '.join(f'{s:.1f}' for s in last3)} {trend}")
                    else:
                        logger.info(f"Runde {rnum} løst (score kommer etter runden)")

                    save_state(state)
                    rounds_solved += 1

                    if max_rounds > 0 and rounds_solved >= max_rounds:
                        logger.info(f"Nådd maks ({max_rounds} runder). Stopper.")
                        break

                except Exception as e:
                    logger.error(f"Feil i runde {rnum}: {e}")
                    logger.error(traceback.format_exc())

            else:
                logger.info(f"Ingen nye runder. Venter {poll_interval}s...")

            time.sleep(poll_interval)

        except KeyboardInterrupt:
            logger.info("=== Stoppet av bruker ===")
            save_state(state)
            break
        except Exception as e:
            logger.error(f"Uventet feil: {e}")
            logger.error(traceback.format_exc())
            save_state(state)
            time.sleep(60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ola's Astar Island Nightbot")
    parser.add_argument("--poll-interval", type=int, default=120,
                        help="Sekunder mellom sjekker (default: 120)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=0,
                        help="Stopp etter N runder (0=uendelig)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
        ],
    )

    if not API_KEY:
        print("FEIL: export API_KEY='din-jwt-token'")
        sys.exit(1)

    run_bot(
        poll_interval=args.poll_interval,
        dry_run=args.dry_run,
        max_rounds=args.max_rounds,
    )


if __name__ == "__main__":
    main()
