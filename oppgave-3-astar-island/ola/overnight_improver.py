#!/usr/bin/env python3
"""
Overnight Improver — Analyserer resultater og forbedrer solveren automatisk.

Kjører i bakgrunnen og sjekker hvert 10. minutt:
1. Har vi nye scores? → Analyser feil
2. Har vi ground truth? → Backtest med ulike parametere
3. Fant vi forbedring? → Oppdater calibration_data.json
4. Nightboten plukker opp endringer automatisk

Bruk:
    export API_KEY='...'
    python overnight_improver.py
"""

import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).parent))
from solution import (
    load_calibration, SeedObserver, load_learning_state, save_learning_state,
    MAP_W, MAP_H, NUM_CLASSES, TERRAIN_TO_CLASS, PROB_FLOOR, NEAR_ZERO, DEFAULT_ALPHA,
    distance_to_nearest_settlement, get_distance_band, is_coastal, get_prior, cell_key,
)

BASE_URL = "https://api.ainm.no/astar-island"
API_KEY = os.environ.get("API_KEY", "")
PROJECT_DIR = Path(__file__).parent
CALIBRATION_FILE = PROJECT_DIR / "calibration_data.json"
IMPROVE_LOG = PROJECT_DIR / "improvement_log.jsonl"


class Client:
    def __init__(self):
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.headers.update({
            "Authorization": f"Bearer {API_KEY}", "Accept": "application/json",
        })

    def get(self, path):
        r = self.session.get(f"{BASE_URL}/{path.lstrip('/')}")
        r.raise_for_status()
        return r.json()


def weighted_kl(gt, pred):
    gt_safe = np.clip(gt, 1e-12, 1.0)
    pred_safe = np.clip(pred, 1e-12, 1.0)
    cell_kl = np.sum(gt_safe * (np.log(gt_safe) - np.log(pred_safe)), axis=-1)
    cell_entropy = -np.sum(gt_safe * np.log(gt_safe), axis=-1)
    tw = cell_entropy.sum()
    if tw <= 0:
        return float(cell_kl.mean())
    return float((cell_kl * cell_entropy).sum() / tw)


def score_from_kl(wkl):
    return max(0, min(100, 100 * np.exp(-3 * wkl)))


def log_improvement(entry):
    with open(IMPROVE_LOG, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def try_alpha_sweep(client, transition_table, simple_prior, rounds_to_test):
    """Test ulike alpha-verdier mot ferske runder."""
    alphas = [12, 14, 15, 16, 18, 20]
    results = {a: [] for a in alphas}

    for round_info in rounds_to_test[-2:]:  # Siste 2 runder
        rid = round_info["id"]
        rnum = round_info.get("round_number", "?")

        try:
            rd = client.get(f"/rounds/{rid}")
            seeds = rd.get("seeds", rd.get("initial_states", []))
        except Exception:
            continue

        for seed_idx in range(min(3, len(seeds))):  # 3 seeds for hastighet
            try:
                analysis = client.get(f"/analysis/{rid}/{seed_idx}")
                gt = analysis.get("ground_truth")
                if not gt:
                    continue
                gt_array = np.array(gt, dtype=float)
            except Exception:
                continue

            seed_data = seeds[seed_idx]
            grid = seed_data.get("grid", [])
            settlements = seed_data.get("settlements", [])

            for alpha in alphas:
                obs = SeedObserver(grid, settlements, transition_table, simple_prior, alpha=alpha)

                # Simuler 2 observasjoner fra ground truth
                np.random.seed(42 + seed_idx)
                for y in range(MAP_H):
                    for x in range(MAP_W):
                        if obs.static_mask[y, x]:
                            continue
                        if np.random.random() < 0.6:
                            probs = gt_array[y, x]
                            if probs.sum() > 0:
                                probs = probs / probs.sum()
                                for _ in range(2):
                                    cls = np.random.choice(NUM_CLASSES, p=probs)
                                    obs.counts[y, x, cls] += 1
                                    obs.observed[y, x] += 1

                pred = obs.build_prediction(apply_smoothing=True)
                s = score_from_kl(weighted_kl(gt_array, pred))
                results[alpha].append(s)

            time.sleep(0.2)

    # Finn beste alpha
    best_alpha = DEFAULT_ALPHA
    best_score = -1
    for alpha, scores in results.items():
        if scores:
            avg = np.mean(scores)
            if avg > best_score:
                best_score = avg
                best_alpha = alpha

    return best_alpha, best_score, results


def run_improver():
    client = Client()
    analyzed_rounds = set()
    iteration = 0

    print("=== OVERNIGHT IMPROVER STARTET ===")
    print("Sjekker hvert 10. minutt for nye resultater...\n")

    while True:
        try:
            iteration += 1
            transition_table, simple_prior = load_calibration()
            learning = load_learning_state()
            current_alpha = learning.get("alpha", DEFAULT_ALPHA)

            rounds = client.get("/rounds")
            completed = [r for r in rounds if isinstance(r, dict) and r.get("status") == "completed"]

            new_completed = [r for r in completed if r["id"] not in analyzed_rounds]

            if new_completed:
                print(f"\n[{time.strftime('%H:%M')}] {len(new_completed)} nye fullførte runder")

                # Alpha-sweep på siste runder
                best_alpha, best_score, alpha_results = try_alpha_sweep(
                    client, transition_table, simple_prior, completed
                )

                print(f"  Alpha-sweep resultater:")
                for alpha, scores in sorted(alpha_results.items()):
                    if scores:
                        print(f"    alpha={alpha:>3}: {np.mean(scores):.1f}")

                if best_alpha != current_alpha and not learning.get("alpha_locked", False):
                    print(f"  → Optimal alpha endret: {current_alpha} → {best_alpha} (+{best_score - np.mean(alpha_results.get(int(current_alpha), [best_score])):.1f})")
                    learning["alpha"] = best_alpha
                    save_learning_state(learning)
                elif learning.get("alpha_locked", False):
                    print(f"  → Alpha={current_alpha} er LÅST (alpha_locked=True)")
                else:
                    print(f"  → Alpha={current_alpha} er fremdeles optimal")

                # Logg
                log_improvement({
                    "iteration": iteration,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "new_rounds": len(new_completed),
                    "best_alpha": best_alpha,
                    "best_score": best_score,
                    "current_alpha": current_alpha,
                })

                for r in new_completed:
                    analyzed_rounds.add(r["id"])

                # Sjekk våre scores
                try:
                    my_rounds = client.get("/my-rounds")
                    for mr in my_rounds[:3]:
                        rnum = mr.get("round_number", "?")
                        score = mr.get("round_score", mr.get("score"))
                        if score and isinstance(score, (int, float)):
                            print(f"  Runde {rnum}: score={score:.1f}")
                except Exception:
                    pass

            else:
                print(f"[{time.strftime('%H:%M')}] Ingen nye runder. Venter...")

            time.sleep(600)  # 10 minutter

        except KeyboardInterrupt:
            print("\nStoppet.")
            break
        except Exception as e:
            print(f"Feil: {e}")
            traceback.print_exc()
            time.sleep(120)


if __name__ == "__main__":
    if not API_KEY:
        print("FEIL: export API_KEY='...'")
        sys.exit(1)
    run_improver()
