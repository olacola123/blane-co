#!/usr/bin/env python3
"""
Ola's Astar Island Nightbot v2 — Selvforbedrende nattdrift
============================================================

Arkitektur per runde:
1. UMIDDELBAR SUBMIT: Prior-only for alle 5 seeds (sikkerhetsnett ~74)
2. UTFORSKNING: Observer seed 0-1, inferér rundetype (mild/hard/reclaimation)
3. INFORMERT: Observer seed 2-4 med justerte priors
4. RESUBMIT: Alle seeds med observasjoner + justert prior
5. ETTER RUNDEN: Hent fasit, rekalibrér, logg forbedring

Sikkerhet:
- Prior-only submit FØRST → aldri under ~74 uansett hva som skjer
- Observasjoner nudger forsiktig (alpha=15)
- Krasj-sikker: state lagres etter hvert steg
- Budget-sjekk: stopper hvis noen andre har brukt queries

Bruk:
    export API_KEY='din-jwt-token'
    nohup python nightbot.py > nightbot_output.log 2>&1 &

    # Sjekk status:
    tail -f nightbot_output.log
    cat nightbot_state.json | python -m json.tool
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
    AstarClient, load_calibration, SeedObserver,
    build_cross_seed_prior, apply_cross_seed,
    plan_queries, infer_round_type, adjust_priors_for_round,
    load_learning_state, save_learning_state,
    MAP_W, MAP_H, NUM_CLASSES, TERRAIN_TO_CLASS, PROB_FLOOR, DEFAULT_ALPHA,
    distance_to_nearest_settlement, get_distance_band, is_coastal, get_prior,
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
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"solved_rounds": [], "scores": {}, "calibrated_rounds": [],
            "total_rounds_solved": 0}


def save_state(state):
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2, default=str))
    except Exception as e:
        logger.error(f"Kunne ikke lagre state: {e}")


# === ROUND FINDING ===

def find_new_round(client, solved_rounds):
    try:
        rounds = client.get_rounds()
    except Exception as e:
        logger.error(f"Kunne ikke hente runder: {e}")
        return None, None

    solved_set = set(solved_rounds)
    for r in reversed(rounds):
        if not isinstance(r, dict):
            continue
        if r.get("status") == "active" and r.get("id") not in solved_set:
            try:
                return r["id"], client.get_round(r["id"])
            except Exception as e:
                logger.error(f"Kunne ikke hente runde: {e}")
                return None, None
    return None, None


# === ROUND-TYPE INFERENCE ===

def infer_round_type(observers):
    """
    Analyser observasjoner fra seed 0-1 for å gjette rundetype.

    Returnerer justeringsfaktorer for prioren:
    - settlement_factor: >1 = flere settlements overlever enn normalt
    - forest_factor: >1 = mer skog enn normalt
    - ruin_factor: >1 = flere ruiner enn normalt
    """
    # Tell observerte klasser på tvers av seeds
    total_counts = np.zeros(NUM_CLASSES, dtype=float)
    total_cells = 0

    for obs in observers:
        for y in range(MAP_H):
            for x in range(MAP_W):
                if obs.static_mask[y, x] or obs.observed[y, x] == 0:
                    continue
                total_counts += obs.counts[y, x]
                total_cells += obs.observed[y, x]

    if total_cells < 100:
        return {"settlement": 1.0, "forest": 1.0, "ruin": 1.0, "port": 1.0}

    # Normaliser
    fracs = total_counts / total_cells

    # Sammenlign med historisk gjennomsnitt (fra kalibrering)
    # Historisk: empty~72%, settlement~12%, port~1%, ruin~1.5%, forest~13%
    hist_settlement = 0.12
    hist_forest = 0.13
    hist_ruin = 0.015
    hist_port = 0.01

    settlement_factor = (fracs[1] / hist_settlement) if hist_settlement > 0 else 1.0
    forest_factor = (fracs[4] / hist_forest) if hist_forest > 0 else 1.0
    ruin_factor = (fracs[3] / hist_ruin) if hist_ruin > 0 else 1.0
    port_factor = (fracs[2] / hist_port) if hist_port > 0 else 1.0

    # Clamp til rimelig range (0.5 - 2.0)
    def clamp(v):
        return max(0.5, min(2.0, v))

    factors = {
        "settlement": clamp(settlement_factor),
        "forest": clamp(forest_factor),
        "ruin": clamp(ruin_factor),
        "port": clamp(port_factor),
    }

    logger.info(f"  Rundetype: sett={factors['settlement']:.2f}× "
                f"forest={factors['forest']:.2f}× ruin={factors['ruin']:.2f}× "
                f"port={factors['port']:.2f}×")

    return factors


def adjust_priors_for_round(observers, factors):
    """Juster prior-cachen basert på infererte rundeparametere."""
    class_factors = np.array([
        1.0,                    # empty — justeres automatisk av normalisering
        factors["settlement"],
        factors["port"],
        factors["ruin"],
        factors["forest"],
        1.0,                    # mountain — statisk
    ])

    for obs in observers:
        for y in range(MAP_H):
            for x in range(MAP_W):
                if obs.static_mask[y, x]:
                    continue
                prior = obs._prior_cache[y, x].copy()
                prior *= class_factors
                prior = np.maximum(prior, PROB_FLOOR)
                prior /= prior.sum()
                obs._prior_cache[y, x] = prior


# === TWO-PASS SOLVER ===

def update_alpha(state, prior_avg, obs_avg):
    """
    [ADAPTIV ALPHA] Juster alpha basert på om observasjoner hjalp eller skadet.
    Hvis obs hjalp → senk alpha litt (stol mer på obs).
    Hvis obs skadet → øk alpha (stol mer på prior).
    """
    learning = load_learning_state()
    current_alpha = learning.get("alpha", DEFAULT_ALPHA)

    if isinstance(prior_avg, (int, float)) and isinstance(obs_avg, (int, float)):
        diff = obs_avg - prior_avg
        if diff > 0:
            learning["obs_helped_count"] = learning.get("obs_helped_count", 0) + 1
            # Obs hjalp → senk alpha litt (min 8)
            new_alpha = max(8.0, current_alpha - 1.0)
            logger.info(f"  Alpha: obs hjalp ({diff:+.1f}) → {current_alpha:.0f} → {new_alpha:.0f}")
        else:
            learning["obs_hurt_count"] = learning.get("obs_hurt_count", 0) + 1
            # Obs skadet → øk alpha (max 30)
            new_alpha = min(30.0, current_alpha + 1.5)
            logger.info(f"  Alpha: obs skadet ({diff:+.1f}) → {current_alpha:.0f} → {new_alpha:.0f}")

        learning["alpha"] = new_alpha
        learning.setdefault("round_history", []).append({
            "prior_avg": prior_avg, "obs_avg": obs_avg,
            "diff": diff, "alpha_used": current_alpha, "new_alpha": new_alpha,
        })
        save_learning_state(learning)

    return learning.get("alpha", DEFAULT_ALPHA)


def solve_round_two_pass(client, round_id, round_data, transition_table, simple_prior):
    """
    To-pass strategi:
    1. Submit prior-only umiddelbart (sikkerhetsnett)
    2. Observer, blend, resubmit
    """
    seeds_data = round_data.get("seeds", round_data.get("initial_states", []))
    if not seeds_data:
        logger.error(f"Ingen seeds i round data")
        return []

    n_seeds = len(seeds_data)
    learning = load_learning_state()
    alpha = learning.get("alpha", DEFAULT_ALPHA)
    logger.info(f"{n_seeds} seeds (alpha={alpha:.0f})")

    # === PASS 1: Submit prior-only ===
    logger.info("PASS 1: Submitter prior-only (sikkerhetsnett)...")
    prior_scores = []
    observers = []

    for seed_idx in range(n_seeds):
        seed = seeds_data[seed_idx]
        grid = seed.get("grid", [])
        settlements = seed.get("settlements", [])

        obs = SeedObserver(grid, settlements, transition_table, simple_prior, alpha=alpha)
        observers.append(obs)

        pred = obs.build_prediction(apply_smoothing=False)
        try:
            resp = client.submit(round_id, seed_idx, pred.tolist())
            score = resp.get("score", resp.get("seed_score", "?"))
            prior_scores.append(score)
            logger.info(f"  Seed {seed_idx} prior-only → {score}")
            time.sleep(0.4)
        except Exception as e:
            logger.error(f"  Seed {seed_idx} submit feil: {e}")
            prior_scores.append(None)

    # === Sjekk budget ===
    try:
        budget = client.get_budget()
        budget_used = budget.get("queries_used", 0)
        budget_max = budget.get("queries_max", 50)
        logger.info(f"Budget: {budget_used}/{budget_max}")
        if budget_used >= budget_max:
            logger.warning("Budget oppbrukt! Beholder prior-only.")
            return [{"seed_index": i, "score": s, "method": "prior-only"}
                    for i, s in enumerate(prior_scores)]
        queries_available = budget_max - budget_used
    except Exception:
        queries_available = 50

    # === PASS 2: Observer og resubmit ===
    logger.info(f"\nPASS 2: Observer ({queries_available} queries tilgjengelig)...")

    queries_per_seed = queries_available // n_seeds

    if queries_per_seed < 3:
        logger.warning(f"For få queries ({queries_per_seed}/seed). Beholder prior-only.")
        return [{"seed_index": i, "score": s, "method": "prior-only"}
                for i, s in enumerate(prior_scores)]

    # Observer seed 0-1 først (utforskningsfase)
    explore_seeds = min(2, n_seeds)
    for seed_idx in range(explore_seeds):
        seed = seeds_data[seed_idx]
        grid = seed.get("grid", [])
        settlements = seed.get("settlements", [])
        viewports = plan_queries(grid, settlements, n_queries=queries_per_seed)

        for i, (vx, vy, vw, vh) in enumerate(viewports):
            try:
                result = client.simulate(round_id, seed_idx, vx, vy, vw, vh)
                grid_data = result.get("grid", [])
                if grid_data:
                    observers[seed_idx].add_observation(grid_data, vx, vy)
                    sett_obs = result.get("settlements", [])
                    if sett_obs:
                        observers[seed_idx].add_settlement_obs(sett_obs)
                used = result.get("queries_used", "?")
                logger.info(f"  Seed {seed_idx} Q{i+1}: ({vx},{vy}) → {used}/{budget_max}")
                time.sleep(0.25)
            except Exception as e:
                logger.error(f"  Seed {seed_idx} Q{i+1} feil: {e}")

    # Inferér rundetype fra seed 0-1
    factors = infer_round_type(observers[:explore_seeds])

    # Juster priors for ALLE seeds basert på rundetype
    adjust_priors_for_round(observers, factors)

    # Observer seed 2-4 med justerte priors
    for seed_idx in range(explore_seeds, n_seeds):
        seed = seeds_data[seed_idx]
        grid = seed.get("grid", [])
        settlements = seed.get("settlements", [])
        viewports = plan_queries(grid, settlements, n_queries=queries_per_seed)

        for i, (vx, vy, vw, vh) in enumerate(viewports):
            try:
                result = client.simulate(round_id, seed_idx, vx, vy, vw, vh)
                grid_data = result.get("grid", [])
                if grid_data:
                    observers[seed_idx].add_observation(grid_data, vx, vy)
                    sett_obs = result.get("settlements", [])
                    if sett_obs:
                        observers[seed_idx].add_settlement_obs(sett_obs)
                used = result.get("queries_used", "?")
                logger.info(f"  Seed {seed_idx} Q{i+1}: ({vx},{vy}) → {used}/{budget_max}")
                time.sleep(0.25)
            except Exception as e:
                logger.error(f"  Seed {seed_idx} Q{i+1} feil: {e}")

    # Cross-seed learning
    cross_table = build_cross_seed_prior(observers)
    if cross_table:
        logger.info(f"  Cross-seed: {len(cross_table)} kategorier")
        apply_cross_seed(observers, cross_table, transition_table)

    # Resubmit alle seeds med smoothing
    logger.info("\nResubmit med observasjoner + smoothing...")
    results = []
    for seed_idx, obs in enumerate(observers):
        pred = obs.build_prediction(apply_smoothing=True)
        try:
            resp = client.submit(round_id, seed_idx, pred.tolist())
            score = resp.get("score", resp.get("seed_score", "?"))
            prior_s = prior_scores[seed_idx]
            improved = ""
            if isinstance(score, (int, float)) and isinstance(prior_s, (int, float)):
                diff = score - prior_s
                improved = f" ({diff:+.1f} vs prior)"
            logger.info(f"  Seed {seed_idx} → {score}{improved}")
            results.append({
                "seed_index": seed_idx,
                "score": score,
                "prior_score": prior_s,
                "method": "observed+cross-seed+smoothing",
            })
            time.sleep(0.4)
        except Exception as e:
            logger.error(f"  Seed {seed_idx} resubmit feil: {e}")
            results.append({"seed_index": seed_idx, "score": prior_s, "method": "prior-only"})

    # [ADAPTIV ALPHA] Sammenlign prior vs obs og juster
    obs_scores = [r["score"] for r in results if isinstance(r.get("score"), (int, float))]
    pri_scores = [r["prior_score"] for r in results if isinstance(r.get("prior_score"), (int, float))]
    if obs_scores and pri_scores:
        obs_avg = sum(obs_scores) / len(obs_scores)
        pri_avg = sum(pri_scores) / len(pri_scores)
        update_alpha(state={}, prior_avg=pri_avg, obs_avg=obs_avg)

    return results


# === REKALIBRERING ===

def recalibrate(client, state):
    """Hent fasit fra nye fullførte runder og oppdater kalibrering."""
    try:
        rounds = client.get_rounds()
    except Exception:
        return False

    completed = [r for r in rounds if isinstance(r, dict) and r.get("status") == "completed"]
    already = set(state.get("calibrated_rounds", []))

    new_ids = [r["id"] for r in completed if r["id"] not in already]
    if not new_ids:
        return False

    logger.info(f"Rekalibrerer fra {len(new_ids)} nye runder...")

    new_entries = []
    for rid in new_ids:
        try:
            rd = client.get_round(rid)
            seeds = rd.get("seeds", rd.get("initial_states", []))
            rnum = rd.get("round_number", "?")

            for si in range(len(seeds)):
                try:
                    analysis = client.get(f"/analysis/{rid}/{si}")
                    gt = analysis.get("ground_truth")
                    if gt:
                        new_entries.append({
                            "initial_grid": seeds[si].get("grid", []),
                            "settlements": seeds[si].get("settlements", []),
                            "ground_truth": gt,
                        })
                except Exception:
                    pass
                time.sleep(0.3)
            already.add(rid)
        except Exception:
            pass

    if not new_entries:
        state["calibrated_rounds"] = list(already)
        return False

    # Oppdater calibration_data.json
    existing = {}
    if CALIBRATION_FILE.exists():
        try:
            existing = json.loads(CALIBRATION_FILE.read_text())
        except Exception:
            pass

    transition_counts = defaultdict(lambda: np.zeros(NUM_CLASSES, dtype=float))
    simple_counts = defaultdict(lambda: np.zeros(NUM_CLASSES, dtype=float))

    old_table = existing.get("transition_table", {})
    for key, data in old_table.items():
        dist = np.array(data["distribution"], dtype=float)
        n = data["sample_count"]
        transition_counts[key] += dist * n

    old_simple = existing.get("simple_prior", {})
    old_seeds = existing.get("num_seeds", 0)
    for key, dist in old_simple.items():
        simple_counts[key] += np.array(dist, dtype=float) * old_seeds * 200

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

                simple_counts[str(terrain)] += gt_probs
                if terrain in (10, 5):
                    continue

                dist = distance_to_nearest_settlement(y, x, settlements)
                band = get_distance_band(dist)
                coastal = is_coastal(grid, y, x)
                key = f"{terrain}_{band}_{int(coastal)}"
                transition_counts[key] += gt_probs

    new_table = {}
    for key, counts in transition_counts.items():
        total = counts.sum()
        if total > 0:
            new_table[key] = {"distribution": (counts / total).tolist(), "sample_count": int(total)}

    new_simple = {}
    for key, counts in simple_counts.items():
        total = counts.sum()
        if total > 0:
            new_simple[key] = (counts / total).tolist()

    output = {
        "transition_table": new_table,
        "simple_prior": new_simple,
        "num_rounds": len(already),
        "num_seeds": existing.get("num_seeds", 0) + len(new_entries),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    CALIBRATION_FILE.write_text(json.dumps(output, indent=2))
    state["calibrated_rounds"] = list(already)

    logger.info(f"Kalibrering oppdatert: {len(already)} runder, {len(new_table)} nøkler")
    return True


# === HOVEDLOOP ===

def run_bot(poll_interval=120, dry_run=False, max_rounds=0):
    client = AstarClient()
    transition_table, simple_prior = load_calibration()
    state = load_state()
    rounds_solved = 0

    logger.info("=" * 60)
    logger.info("  OLA'S NIGHTBOT v2 STARTET")
    logger.info("=" * 60)
    logger.info(f"Poll: {poll_interval}s | Dry-run: {dry_run}")
    logger.info(f"Tidligere løst: {len(state['solved_rounds'])} runder")
    logger.info(f"Kalibrering: {len(transition_table or {})} nøkler")

    while True:
        try:
            # Rekalibrér fra nye fullførte runder
            try:
                if recalibrate(client, state):
                    transition_table, simple_prior = load_calibration()
                    logger.info("Solver bruker oppdaterte priors!")
                save_state(state)
            except Exception as e:
                logger.warning(f"Rekalibrering feilet: {e}")

            # Finn ny runde
            round_id, round_data = find_new_round(client, state["solved_rounds"])

            if round_id:
                rnum = round_data.get("round_number", "?")
                weight = round_data.get("round_weight", "?")
                closes = round_data.get("closes_at", "?")
                logger.info(f"\n{'='*60}")
                logger.info(f"  RUNDE {rnum} (vekt {weight})")
                logger.info(f"  Stenger: {closes}")
                logger.info(f"{'='*60}")

                # Budget-sjekk
                try:
                    budget = client.get_budget()
                    used = budget.get("queries_used", 0)
                    if used > 0:
                        logger.warning(f"Budget {used}/50 allerede brukt!")
                        if used >= 50:
                            logger.error("Budget oppbrukt! Skipper runden.")
                            state["solved_rounds"].append(round_id)
                            save_state(state)
                            time.sleep(poll_interval)
                            continue
                except Exception:
                    pass

                try:
                    if dry_run:
                        logger.info("[DRY RUN] Skipper solve")
                        results = []
                    else:
                        results = solve_round_two_pass(
                            client, round_id, round_data,
                            transition_table, simple_prior,
                        )

                    state["solved_rounds"].append(round_id)

                    # Logg scores
                    scores = [r["score"] for r in results
                              if isinstance(r.get("score"), (int, float))]
                    prior_scores = [r["prior_score"] for r in results
                                    if isinstance(r.get("prior_score"), (int, float))]

                    if scores:
                        avg = sum(scores) / len(scores)
                        weighted = avg * weight if isinstance(weight, (int, float)) else "?"
                        state["scores"][round_id] = {
                            "round_number": rnum,
                            "avg_score": avg,
                            "weighted": weighted,
                            "per_seed": [r.get("score", "?") for r in results],
                            "prior_avg": sum(prior_scores) / len(prior_scores) if prior_scores else "?",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        logger.info(f"\n  ★ Runde {rnum}: snitt={avg:.1f}, weighted={weighted}")

                        if prior_scores:
                            prior_avg = sum(prior_scores) / len(prior_scores)
                            logger.info(f"  Prior-only: {prior_avg:.1f} → Obs: {avg:.1f} "
                                        f"({avg - prior_avg:+.1f})")

                        # Trend
                        all_avgs = [v["avg_score"] for v in state["scores"].values()
                                    if isinstance(v.get("avg_score"), (int, float))]
                        if len(all_avgs) >= 2:
                            last = all_avgs[-3:]
                            trend = " → ".join(f"{s:.0f}" for s in last)
                            arrow = "↑" if last[-1] > last[0] else "↓"
                            logger.info(f"  Trend: {trend} {arrow}")

                    save_state(state)
                    rounds_solved += 1
                    state["total_rounds_solved"] = state.get("total_rounds_solved", 0) + 1

                    if max_rounds > 0 and rounds_solved >= max_rounds:
                        logger.info(f"Nådd maks ({max_rounds} runder). Stopper.")
                        break

                except Exception as e:
                    logger.error(f"Feil i runde {rnum}: {e}")
                    logger.error(traceback.format_exc())
                    # Ikke legg til i solved_rounds → prøver igjen neste poll

            else:
                logger.info(f"Ingen nye runder. Venter {poll_interval}s...")

            time.sleep(poll_interval)

        except KeyboardInterrupt:
            logger.info("Stoppet av bruker.")
            save_state(state)
            break
        except Exception as e:
            logger.error(f"Uventet feil: {e}")
            logger.error(traceback.format_exc())
            save_state(state)
            time.sleep(60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ola's Astar Island Nightbot v2")
    parser.add_argument("--poll-interval", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=0)
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
