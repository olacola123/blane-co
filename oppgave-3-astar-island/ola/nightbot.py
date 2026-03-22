#!/usr/bin/env python3
"""
Ola's Astar Island Nightbot v3 — Kernel-prediksjon med urban-deteksjon
=======================================================================

Bruker solve_round() fra solution.py (v9) som nå inkluderer:
- Kernel-weighted prediction (vitality + urban)
- Safety submit → probe → resubmit med kernel
- Automatisk urban-estimering fra observasjoner

Nightbot-loop:
1. Poll for nye aktive runder
2. Kjør solve_round() (safety submit → probe → kernel resubmit)
3. Vent på runde-slutt, hent fasit + scores
4. Lagre all data (observasjoner, prediksjoner, GT, scores)
5. Oppdater kernel_training_data.json med ny GT
6. Gjenta

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
    AstarClient, load_calibration, load_optimized_calibration,
    solve_round, super_predict, kernel_predict, estimate_urban,
    vitality_to_vbin, _terrain_group, _dist_bin,
    MAP_W, MAP_H, NUM_CLASSES,
)

PROJECT_DIR = Path(__file__).parent
STATE_FILE = PROJECT_DIR / "nightbot_state.json"
LOG_FILE = PROJECT_DIR / "nightbot.log"
HISTORY_DIR = PROJECT_DIR / "round_history"
KERNEL_DATA_FILE = PROJECT_DIR / "kernel_training_data.json"

API_KEY = os.environ.get("API_KEY", "")

logger = logging.getLogger("ola-nightbot-v3")


# === STATE ===

def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {
        "solved_rounds": [],
        "scores": {},
        "calibrated_rounds": [],
        "total_rounds_solved": 0,
        "version": "v3-kernel",
    }


def save_state(state):
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2, default=str))
    except Exception as e:
        logger.error(f"Kunne ikke lagre state: {e}")


def save_round_data(round_number, data_type, data):
    """Lagre data per runde i round_history/runde_N/."""
    round_dir = HISTORY_DIR / f"runde_{round_number}"
    round_dir.mkdir(parents=True, exist_ok=True)
    filepath = round_dir / f"{data_type}.json"

    if isinstance(data, np.ndarray):
        data = data.tolist()

    try:
        filepath.write_text(json.dumps(data, indent=2, default=str))
    except Exception as e:
        logger.warning(f"Kunne ikke lagre {data_type}: {e}")


# === ROUND FINDING ===

def find_new_round(client, solved_rounds):
    """Finn aktiv runde som ikke er løst ennå."""
    try:
        rounds = client.get_rounds()
    except Exception as e:
        logger.error(f"Kunne ikke hente runder: {e}")
        return None, None, None

    solved_set = set(solved_rounds)
    for r in reversed(rounds):
        if not isinstance(r, dict):
            continue
        if r.get("status") == "active" and r.get("id") not in solved_set:
            try:
                round_data = client.get_round(r["id"])
                return r["id"], round_data, r
            except Exception as e:
                logger.error(f"Kunne ikke hente runde: {e}")
                return None, None, None
    return None, None, None


# === SOLVE ===

def solve_and_log(client, round_id, round_data, round_meta, state):
    """Kjør solve_round() og lagre all data."""
    rnum = round_data.get("round_number", round_meta.get("round_number", "?"))
    weight = round_meta.get("round_weight", round_data.get("round_weight", 1.0))

    logger.info(f"\n{'='*60}")
    logger.info(f"  RUNDE {rnum} (vekt {weight:.3f})")
    logger.info(f"{'='*60}")

    # Budget-sjekk
    try:
        budget = client.get_budget()
        used = budget.get("queries_used", 0)
        max_q = budget.get("queries_max", 50)
        logger.info(f"Budget: {used}/{max_q}")
        if used >= max_q:
            logger.error("Budget oppbrukt! Skipper runden.")
            return None
    except Exception:
        pass

    # Last kalibrering
    transition_table, simple_prior = load_calibration()
    opt_tables = load_optimized_calibration()

    # Last 4-type tabeller
    cal4_file = PROJECT_DIR / "calibration_4type.json"
    type_tables = None
    if cal4_file.exists():
        try:
            data4 = json.loads(cal4_file.read_text())
            type_tables = data4.get("tables", {})
        except Exception:
            pass

    # Kjør solve_round (bruker v9 kernel i FASE 6)
    results = solve_round(
        client, round_id, round_data,
        transition_table, simple_prior,
        queries_per_seed=10,
        submit=True,
        type_tables=type_tables,
        safety_submit=True,
        opt_tables=opt_tables,
    )

    # Lagre resultater
    save_round_data(rnum, "results", {
        "round_id": round_id,
        "round_number": rnum,
        "weight": weight,
        "results": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    # Lagre initial states
    seeds_data = round_data.get("seeds", round_data.get("initial_states", []))
    for si, seed in enumerate(seeds_data):
        save_round_data(rnum, f"seed_{si}_initial", {
            "grid": seed.get("grid"),
            "settlements": seed.get("settlements"),
        })

    # Log scores
    scores = [r.get("score", r.get("seed_score"))
              for r in results if isinstance(r.get("score", r.get("seed_score")), (int, float))]

    if scores:
        avg = sum(scores) / len(scores)
        logger.info(f"\n  ★ Runde {rnum}: snitt={avg:.1f}, vektet={avg * weight:.1f}")

        state["scores"][round_id] = {
            "round_number": rnum,
            "avg_score": round(avg, 1),
            "weight": weight,
            "per_seed": scores,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    return results


# === POST-ROUND: HENT FASIT OG OPPDATER KERNEL DATA ===

def fetch_and_store_results(client, state):
    """Hent fasit fra fullførte runder og oppdater kernel-treningsdata."""
    try:
        rounds = client.get_rounds()
    except Exception:
        return False

    completed = [r for r in rounds if isinstance(r, dict) and r.get("status") == "completed"]
    already = set(state.get("calibrated_rounds", []))

    new_rounds = [r for r in completed if r["id"] not in already]
    if not new_rounds:
        return False

    logger.info(f"Henter fasit fra {len(new_rounds)} nye fullførte runder...")

    new_kernel_seeds = []

    for r in new_rounds:
        rid = r["id"]
        try:
            rd = client.get_round(rid)
            rnum = rd.get("round_number", r.get("round_number", "?"))
            seeds = rd.get("seeds", rd.get("initial_states", []))

            for si in range(len(seeds)):
                try:
                    analysis = client.get(f"/analysis/{rid}/{si}")
                    gt = analysis.get("ground_truth")
                    score = analysis.get("score", "?")

                    if gt:
                        # Lagre GT
                        save_round_data(rnum, f"seed_{si}_ground_truth", gt)
                        save_round_data(rnum, f"seed_{si}_score", score)

                        # Bygg kernel-treningsdata for denne seeden
                        grid = seeds[si].get("grid", [])
                        settlements = seeds[si].get("settlements", [])
                        kernel_entry = build_kernel_entry(
                            grid, settlements, gt, rnum, si
                        )
                        if kernel_entry:
                            new_kernel_seeds.append(kernel_entry)

                        logger.info(f"  R{rnum} seed {si}: score={score}")

                except Exception as e:
                    logger.warning(f"  R{rnum} seed {si} feilet: {e}")
                time.sleep(0.3)

            already.add(rid)
        except Exception as e:
            logger.warning(f"  Runde {rid} feilet: {e}")

    state["calibrated_rounds"] = list(already)

    # Oppdater kernel_training_data.json
    if new_kernel_seeds:
        update_kernel_data(new_kernel_seeds)

    return len(new_kernel_seeds) > 0


def build_kernel_entry(grid, settlements, gt, rnum, si):
    """Bygg én kernel-treningsdata-entry fra GT."""
    try:
        grid_arr = np.array(grid, dtype=int)
        gt_arr = np.array(gt, dtype=float)

        # Vitality
        total_s, survived_s = 0, 0.0
        for s in settlements:
            sx, sy = s["x"], s["y"]
            if 0 <= sx < 40 and 0 <= sy < 40:
                total_s += 1
                survived_s += gt_arr[sy, sx, 1] + gt_arr[sy, sx, 2]
        vitality = survived_s / total_s if total_s > 0 else 0

        # Urban
        spos = [(s["x"], s["y"]) for s in settlements]
        mn, cn, mf, cf = 0, 0, 0, 0
        for y in range(40):
            for x in range(40):
                t = int(grid_arr[y, x])
                if t in (10, 5):
                    continue
                md = min(max(abs(y - sy), abs(x - sx)) for sx, sy in spos)
                sp = gt_arr[y, x, 1] + gt_arr[y, x, 2]
                if md <= 2:
                    mn += sp; cn += 1
                elif md >= 5:
                    mf += sp; cf += 1
        avg_near = mn / cn if cn > 0 else 0
        avg_far = mf / cf if cf > 0 else 0
        urban = np.log10(max(avg_near, 1e-6)) - np.log10(max(avg_far, 1e-6))

        # Per-group distributions
        groups = defaultdict(lambda: {"sum": np.zeros(6), "count": 0})
        for y in range(40):
            for x in range(40):
                t = int(grid_arr[y, x])
                if t in (10, 5):
                    continue
                md = 99
                for s in settlements:
                    d = max(abs(y - s["y"]), abs(x - s["x"]))
                    if d < md:
                        md = d
                c = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < 40 and 0 <= nx < 40 and int(grid_arr[ny, nx]) == 10:
                            c = 1; break
                    if c:
                        break
                key = f"{_terrain_group(t)}_{_dist_bin(md)}_{c}"
                groups[key]["sum"] += gt_arr[y, x]
                groups[key]["count"] += 1

        group_data = {}
        for key, data in groups.items():
            avg = (data["sum"] / data["count"]).tolist()
            group_data[key] = [round(v, 5) for v in avg]

        return {
            "round": rnum,
            "seed": si,
            "vitality": round(vitality, 4),
            "urban": round(urban, 2),
            "groups": group_data,
        }
    except Exception as e:
        logger.warning(f"build_kernel_entry feilet: {e}")
        return None


def update_kernel_data(new_seeds):
    """Legg til nye seeds i kernel_training_data.json."""
    try:
        if KERNEL_DATA_FILE.exists():
            data = json.loads(KERNEL_DATA_FILE.read_text())
        else:
            data = {"n_rounds": 0, "n_seeds": 0, "bw_vitality": 0.15,
                    "bw_urban": 2.0, "seeds": []}

        # Sjekk for duplikater
        existing = set()
        for s in data["seeds"]:
            existing.add((s.get("round"), s.get("seed")))

        added = 0
        for ns in new_seeds:
            key = (ns["round"], ns["seed"])
            if key not in existing:
                data["seeds"].append(ns)
                existing.add(key)
                added += 1

        if added > 0:
            data["n_seeds"] = len(data["seeds"])
            # Teller unike runder
            data["n_rounds"] = len(set(s["round"] for s in data["seeds"]))

            KERNEL_DATA_FILE.write_text(json.dumps(data))
            logger.info(f"  Kernel data oppdatert: +{added} seeds → {data['n_seeds']} totalt")

            # Invalidér cache så neste prediksjon bruker ny data
            import solution
            solution._kernel_data_cache = None
    except Exception as e:
        logger.warning(f"  Kernel data oppdatering feilet: {e}")


# === HOVEDLOOP ===

def run_bot(poll_interval=120, dry_run=False, max_rounds=0):
    client = AstarClient()
    state = load_state()
    rounds_solved = 0

    logger.info("=" * 60)
    logger.info("  OLA'S NIGHTBOT v3 — KERNEL EDITION")
    logger.info("=" * 60)
    logger.info(f"Poll: {poll_interval}s | Dry-run: {dry_run}")
    logger.info(f"Tidligere løst: {len(state['solved_rounds'])} runder")

    # Sjekk at kernel-data finnes
    if KERNEL_DATA_FILE.exists():
        kd = json.loads(KERNEL_DATA_FILE.read_text())
        logger.info(f"Kernel data: {kd['n_seeds']} seeds fra {kd['n_rounds']} runder")
    else:
        logger.warning("ADVARSEL: Ingen kernel_training_data.json!")

    while True:
        try:
            # 1. Hent fasit fra fullførte runder og oppdater kernel data
            try:
                if fetch_and_store_results(client, state):
                    logger.info("Kernel-treningsdata oppdatert med ny GT!")
                save_state(state)
            except Exception as e:
                logger.warning(f"Fasit-henting feilet: {e}")

            # 2. Finn ny aktiv runde
            round_id, round_data, round_meta = find_new_round(client, state["solved_rounds"])

            if round_id:
                try:
                    if dry_run:
                        logger.info(f"[DRY RUN] Runde {round_meta.get('round_number', '?')}")
                        results = []
                    else:
                        results = solve_and_log(client, round_id, round_data, round_meta, state)

                    state["solved_rounds"].append(round_id)
                    state["total_rounds_solved"] = state.get("total_rounds_solved", 0) + 1
                    save_state(state)

                    rounds_solved += 1
                    if max_rounds > 0 and rounds_solved >= max_rounds:
                        logger.info(f"Nådd maks ({max_rounds} runder). Stopper.")
                        break

                except Exception as e:
                    logger.error(f"Feil i runde: {e}")
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
    parser = argparse.ArgumentParser(description="Ola's Astar Island Nightbot v3 — Kernel Edition")
    parser.add_argument("--poll-interval", type=int, default=120,
                        help="Sekunder mellom polling (default 120)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Ikke submit, bare vis plan")
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
