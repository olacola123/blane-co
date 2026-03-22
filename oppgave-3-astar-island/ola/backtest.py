"""
Backtest — Test solveren mot historisk fasit UTEN å bruke API.

Hva dette gjør:
1. For hver av de 9 fullførte rundene:
   - Hent startkart + fasit fra API (GET, ingen simulate/submit)
   - Kjør solveren med BARE priors (ingen observasjoner)
   - Beregn score: 100 × exp(-3 × weighted_KL)
   - Sammenlign med hva vi faktisk scoret

Dette viser oss om priorsne alene er gode nok,
og hjelper oss tune alpha-verdier.

Bruk:
    export API_KEY='din-jwt-token'
    python backtest.py
    python backtest.py --with-alpha 15   # Test med spesifikk alpha
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
    load_calibration, load_optimized_calibration, load_model_tables, SeedObserver,
    MAP_W, MAP_H, NUM_CLASSES, TERRAIN_TO_CLASS,
    PROB_FLOOR, super_predict, vitality_to_vbin,
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


def weighted_kl(ground_truth, prediction):
    """Beregn entropy-weighted KL divergence (konkurranse-metrikken)."""
    gt = np.array(ground_truth, dtype=float)
    pred = np.array(prediction, dtype=float)

    # Clip for numerisk stabilitet
    gt_safe = np.clip(gt, 1e-12, 1.0)
    pred_safe = np.clip(pred, 1e-12, 1.0)

    # Per-celle KL divergence
    cell_kl = np.sum(gt_safe * (np.log(gt_safe) - np.log(pred_safe)), axis=-1)

    # Per-celle entropi (vekt)
    cell_entropy = -np.sum(gt_safe * np.log(gt_safe), axis=-1)

    # Vektet gjennomsnitt
    total_weight = cell_entropy.sum()
    if total_weight <= 0:
        return float(cell_kl.mean())

    return float((cell_kl * cell_entropy).sum() / total_weight)


def score_from_kl(wkl):
    return max(0, min(100, 100 * np.exp(-3 * wkl)))


def backtest_round(client, round_id, round_num, seeds_data, transition_table,
                   simple_prior, alpha_override=None, typed_table=None,
                   opt_tables=None, world_type=None, vbin=None):
    """Test solveren mot fasit for én runde."""
    results = []

    for seed_idx in range(len(seeds_data)):
        seed_data = seeds_data[seed_idx]
        grid = seed_data.get("grid", [])
        settlements = seed_data.get("settlements", [])

        # Bruk super_predict (samme som live FASE 6) hvis vbin er satt
        pred = None
        if vbin is not None:
            pred = super_predict(grid, settlements, vbin)

        if pred is None:
            # Fallback til gammel metode
            observer = SeedObserver(grid, settlements, transition_table, simple_prior,
                                   typed_table=typed_table,
                                   opt_tables=opt_tables, world_type=world_type)
            pred = observer.build_prediction(world_type=world_type)

        # Hent fasit
        try:
            analysis = client.get(f"/analysis/{round_id}/{seed_idx}")
            gt = analysis.get("ground_truth")
            actual_score = analysis.get("score", "?")

            if gt is None:
                results.append({"seed": seed_idx, "status": "ingen fasit"})
                continue

            gt_array = np.array(gt, dtype=float)
            pred_array = np.array(pred, dtype=float)

            wkl = weighted_kl(gt_array, pred_array)
            prior_score = score_from_kl(wkl)

            results.append({
                "seed": seed_idx,
                "prior_only_score": round(prior_score, 1),
                "actual_score": actual_score,
                "weighted_kl": round(wkl, 4),
                "improvement": round(prior_score - (actual_score if isinstance(actual_score, (int, float)) else 0), 1),
            })

            time.sleep(0.2)
        except Exception as e:
            results.append({"seed": seed_idx, "error": str(e)})

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-alpha", type=float, default=None,
                        help="Test med spesifikk alpha-verdi")
    parser.add_argument("--rounds", type=int, default=0,
                        help="Antall runder å teste (0=alle)")
    args = parser.parse_args()

    if not API_KEY:
        print("FEIL: export API_KEY='din-jwt-token'")
        sys.exit(1)

    client = SimpleClient()
    transition_table, simple_prior = load_calibration()
    opt_tables = load_optimized_calibration()

    # Last 4-type calibration
    cal4_file = Path(__file__).parent / "calibration_4type.json"
    type_tables = None
    if cal4_file.exists():
        data4 = json.loads(cal4_file.read_text())
        type_tables = data4.get("tables", {})
        print(f"Lastet 4-type tabeller: {', '.join(f'{k}({len(v)})' for k,v in type_tables.items())}\n")

    print("=== BACKTEST: Prior-only vs. faktisk score ===\n")

    rounds = client.get("/rounds")
    completed = [r for r in rounds if isinstance(r, dict) and r.get("status") == "completed"]
    completed.sort(key=lambda r: r.get("round_number", 0))

    if args.rounds > 0:
        completed = completed[-args.rounds:]

    all_prior_scores = []
    all_actual_scores = []

    for r in completed:
        round_id = r["id"]
        rnum = r.get("round_number", "?")
        weight = r.get("round_weight", 1.0)

        round_data = client.get(f"/rounds/{round_id}")
        seeds_data = round_data.get("seeds", round_data.get("initial_states", []))

        # Oracle world type: beregn fra GT
        wt = "STABLE"  # default
        try:
            analysis = client.get(f"/analysis/{round_id}/0")
            gt = analysis.get("ground_truth")
            if gt:
                gt_arr = np.array(gt, dtype=float)
                settlements = seeds_data[0].get("settlements", [])
                total_s, survived_s = 0, 0.0
                for s in settlements:
                    sx, sy = s["x"], s["y"]
                    if 0 <= sx < 40 and 0 <= sy < 40:
                        total_s += 1
                        survived_s += gt_arr[sy, sx, 1] + gt_arr[sy, sx, 2]  # settle + port
                if total_s > 0:
                    rate = survived_s / total_s
                    n_settlements = len(settlements)
                    if rate < 0.10:
                        wt = "DEAD"
                    elif rate < 0.35:
                        wt = "STABLE"
                    else:
                        wt = "BOOM_CONC" if n_settlements >= 40 else "BOOM_SPREAD"
                    print(f"--- Runde {rnum} (vekt {weight:.3f}) [{wt} surv={rate:.2f} n={n_settlements}] ---")
                else:
                    print(f"--- Runde {rnum} (vekt {weight:.3f}) ---")
            else:
                print(f"--- Runde {rnum} (vekt {weight:.3f}) ---")
        except Exception:
            print(f"--- Runde {rnum} (vekt {weight:.3f}) ---")

        # Velg riktig tabell for denne runden
        typed_table = type_tables.get(wt) if type_tables else None
        # Map 4-type til 3-type for optimerte tabeller
        opt_wtype = wt
        if wt in ("BOOM_CONC", "BOOM_SPREAD"):
            opt_wtype = "BOOM"
        # Beregn vbin fra oracle survival rate (samme som live ville gjøre med perfekt detection)
        oracle_vbin = vitality_to_vbin(rate) if total_s > 0 else "MED"
        results = backtest_round(client, round_id, rnum, seeds_data,
                                transition_table, simple_prior, args.with_alpha,
                                typed_table=typed_table,
                                opt_tables=opt_tables, world_type=opt_wtype,
                                vbin=oracle_vbin)

        for res in results:
            if "prior_only_score" in res:
                ps = res["prior_only_score"]
                actual = res["actual_score"]
                imp = res["improvement"]
                wkl = res["weighted_kl"]
                symbol = "✓" if imp > 0 else "✗"
                print(f"  Seed {res['seed']}: prior={ps:>5.1f}  faktisk={actual:>5.1f}  "
                      f"diff={imp:>+6.1f} {symbol}  wkl={wkl:.4f}")
                all_prior_scores.append(ps)
                if isinstance(actual, (int, float)):
                    all_actual_scores.append(actual)
            else:
                print(f"  Seed {res['seed']}: {res.get('error', res.get('status', '?'))}")

        # Snitt for runden
        round_priors = [r["prior_only_score"] for r in results if "prior_only_score" in r]
        round_actuals = [r["actual_score"] for r in results
                        if isinstance(r.get("actual_score"), (int, float))]
        if round_priors:
            avg_p = sum(round_priors) / len(round_priors)
            avg_a = sum(round_actuals) / len(round_actuals) if round_actuals else 0
            print(f"  SNITT: prior={avg_p:.1f}  faktisk={avg_a:.1f}  diff={avg_p - avg_a:+.1f}")
        print()

    # Total oppsummering
    if all_prior_scores:
        avg_prior = sum(all_prior_scores) / len(all_prior_scores)
        avg_actual = sum(all_actual_scores) / len(all_actual_scores) if all_actual_scores else 0
        print(f"=== TOTALT ===")
        print(f"  Snitt prior-only:  {avg_prior:.1f}")
        print(f"  Snitt faktisk:     {avg_actual:.1f}")
        print(f"  Prior er {avg_prior - avg_actual:+.1f} poeng bedre")
        print(f"\n  Beste prior-score: {max(all_prior_scores):.1f}")
        print(f"  Dårligste:         {min(all_prior_scores):.1f}")


if __name__ == "__main__":
    main()
