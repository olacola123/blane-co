"""
Backtest med simulerte observasjoner — finn optimal alpha.

Simulerer hva som skjer når vi observer celler og blander med prior:
1. Hent fasit (ground truth distribusjon) for en runde
2. "Observer" celler ved å sample fra ground truth (simulerer /simulate)
3. Bland observasjoner med prior ved ulike alpha-verdier
4. Beregn score for hver alpha
5. Finn optimal alpha

Bruk:
    export API_KEY='din-jwt-token'
    python backtest_alpha.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).parent))
from solution import (
    load_calibration, SeedObserver, build_cross_seed_prior, apply_cross_seed,
    MAP_W, MAP_H, NUM_CLASSES, TERRAIN_TO_CLASS, PROB_FLOOR,
    distance_to_nearest_settlement, get_distance_band, is_coastal, get_prior,
    cell_key,
)

BASE_URL = "https://api.ainm.no/astar-island"
API_KEY = os.environ.get("API_KEY", "")


class SimpleClient:
    def __init__(self):
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.headers.update({
            "Authorization": f"Bearer {API_KEY}",
            "Accept": "application/json",
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
    total_weight = cell_entropy.sum()
    if total_weight <= 0:
        return float(cell_kl.mean())
    return float((cell_kl * cell_entropy).sum() / total_weight)


def score_from_kl(wkl):
    return max(0, min(100, 100 * np.exp(-3 * wkl)))


def simulate_observations(gt_array, n_obs_per_cell=1):
    """
    Simuler observasjoner ved å sample fra ground truth.
    Hvert sample er ett tilfeldig utfall (som /simulate gir).
    """
    h, w, k = gt_array.shape
    counts = np.zeros_like(gt_array)

    for y in range(h):
        for x in range(w):
            probs = gt_array[y, x]
            if probs.sum() < 0.001:
                continue
            probs = probs / probs.sum()
            for _ in range(n_obs_per_cell):
                cls = np.random.choice(k, p=probs)
                counts[y, x, cls] += 1

    return counts


def test_alpha_on_round(client, round_id, round_num, seeds_data,
                        transition_table, simple_prior, alphas_to_test,
                        n_obs=2, n_trials=3):
    """Test ulike alpha-verdier på én runde med simulerte observasjoner."""
    results_by_alpha = {a: [] for a in alphas_to_test}
    results_by_alpha["prior_only"] = []

    for seed_idx in range(min(3, len(seeds_data))):  # Test 3 seeds for speed
        seed_data = seeds_data[seed_idx]
        grid = seed_data.get("grid", [])
        settlements = seed_data.get("settlements", [])

        # Hent fasit
        try:
            analysis = client.get(f"/analysis/{round_id}/{seed_idx}")
            gt = analysis.get("ground_truth")
            if gt is None:
                continue
            gt_array = np.array(gt, dtype=float)
        except Exception:
            continue

        grid_np = np.array(grid, dtype=int)
        ocean_mask = (grid_np == 10)
        mountain_mask = (grid_np == 5)
        static_mask = ocean_mask | mountain_mask

        # Prior-only score
        observer = SeedObserver(grid, settlements, transition_table, simple_prior)
        pred_prior = observer.build_prediction()
        prior_score = score_from_kl(weighted_kl(gt_array, pred_prior))
        results_by_alpha["prior_only"].append(prior_score)

        # Test hver alpha med simulerte observasjoner
        for alpha in alphas_to_test:
            trial_scores = []
            for trial in range(n_trials):
                obs_counts = simulate_observations(gt_array, n_obs_per_cell=n_obs)

                # Bygg prediksjon med denne alpha
                pred = np.zeros((MAP_H, MAP_W, NUM_CLASSES), dtype=float)
                grid_list = grid_np.tolist()

                for y in range(MAP_H):
                    for x in range(MAP_W):
                        if ocean_mask[y, x]:
                            pred[y, x] = [0.998, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004]
                            continue
                        if mountain_mask[y, x]:
                            pred[y, x] = [0.0004, 0.0004, 0.0004, 0.0004, 0.0004, 0.998]
                            continue

                        terrain, band, coastal = cell_key(grid_list, y, x, settlements)
                        prior = get_prior(terrain, band, coastal,
                                         transition_table, simple_prior)

                        cell_obs = obs_counts[y, x]
                        cell_n = cell_obs.sum()

                        if cell_n == 0:
                            pred[y, x] = prior.copy()
                        else:
                            pred[y, x] = cell_obs + alpha * prior

                        np.maximum(pred[y, x], PROB_FLOOR, out=pred[y, x])
                        pred[y, x] /= pred[y, x].sum()

                s = score_from_kl(weighted_kl(gt_array, pred))
                trial_scores.append(s)

            results_by_alpha[alpha].append(np.mean(trial_scores))

        time.sleep(0.2)

    return results_by_alpha


def main():
    if not API_KEY:
        print("FEIL: export API_KEY='din-jwt-token'")
        sys.exit(1)

    client = SimpleClient()
    transition_table, simple_prior = load_calibration()

    # Test disse alpha-verdiene
    alphas = [1, 2, 5, 8, 10, 15, 20, 30, 50]

    print("=== ALPHA-TUNING BACKTEST ===")
    print(f"Tester alpha: {alphas}")
    print(f"Simulerer 2 observasjoner per celle, 3 trials per alpha\n")

    rounds = client.get("/rounds")
    completed = [r for r in rounds if isinstance(r, dict) and r.get("status") == "completed"]
    completed.sort(key=lambda r: r.get("round_number", 0))

    # Test på siste 4 runder for hastighet
    test_rounds = completed[-4:]

    all_results = {a: [] for a in alphas}
    all_results["prior_only"] = []

    for r in test_rounds:
        round_id = r["id"]
        rnum = r.get("round_number", "?")
        round_data = client.get(f"/rounds/{round_id}")
        seeds_data = round_data.get("seeds", round_data.get("initial_states", []))

        print(f"--- Runde {rnum} ---")
        results = test_alpha_on_round(
            client, round_id, rnum, seeds_data,
            transition_table, simple_prior,
            alphas_to_test=alphas,
            n_obs=2, n_trials=3,
        )

        for key, scores in results.items():
            all_results[key].extend(scores)
            if scores:
                avg = np.mean(scores)
                label = f"alpha={key}" if key != "prior_only" else "prior_only"
                print(f"  {label:>15}: {avg:.1f}")
        print()

    # Oppsummering
    print("=== TOTAL OPPSUMMERING ===")
    print(f"{'Alpha':>15} {'Snitt':>8} {'Min':>8} {'Max':>8}")
    print("-" * 45)

    best_alpha = None
    best_score = -1

    for key in ["prior_only"] + alphas:
        scores = all_results[key]
        if not scores:
            continue
        avg = np.mean(scores)
        lo = np.min(scores)
        hi = np.max(scores)
        label = f"alpha={key}" if key != "prior_only" else "prior_only"
        marker = ""
        if key != "prior_only" and avg > best_score:
            best_score = avg
            best_alpha = key
        print(f"{label:>15} {avg:>8.1f} {lo:>8.1f} {hi:>8.1f} {marker}")

    print(f"\n→ Beste alpha: {best_alpha} (snitt {best_score:.1f})")
    print(f"  Prior-only: {np.mean(all_results['prior_only']):.1f}")
    print(f"  Forbedring fra observasjoner: {best_score - np.mean(all_results['prior_only']):+.1f}")


if __name__ == "__main__":
    main()
