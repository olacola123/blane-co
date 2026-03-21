"""
Backtest for simulator.py — test empirisk simulator mot ground truth.

For hver runde: prøv ulike vitality-verdier, finn optimal, og rapporter score.
Viser også hva scoren blir med "oracle" vitality (best mulige).

Bruk:
    export API_KEY='...'
    python backtest_simulator.py
"""
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
from simulator import predict_distribution, weighted_kl, score_from_kl

BASE_URL = "https://api.ainm.no/astar-island"
API_KEY = os.environ.get("API_KEY", "")


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


def main():
    if not API_KEY:
        print("FEIL: export API_KEY='...'")
        sys.exit(1)

    client = Client()
    rounds = client.get("/rounds")
    completed = [r for r in rounds if isinstance(r, dict) and r.get("status") == "completed"]
    completed.sort(key=lambda r: r.get("round_number", 0))

    print("=== SIMULATOR BACKTEST ===\n")

    vitality_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    all_best_scores = []
    all_default_scores = []

    for r in completed:
        round_id = r["id"]
        rnum = r.get("round_number", "?")
        weight = r.get("round_weight", 1.0)

        round_data = client.get(f"/rounds/{round_id}")
        seeds_data = round_data.get("seeds", round_data.get("initial_states", []))

        if not seeds_data:
            continue

        # Test seed 0
        seed_data = seeds_data[0]
        grid = seed_data.get("grid", [])
        settlements = seed_data.get("settlements", [])

        try:
            analysis = client.get(f"/analysis/{round_id}/0")
            gt = analysis.get("ground_truth")
            if not gt:
                print(f"Runde {rnum}: ingen ground truth")
                continue
            gt_array = np.array(gt, dtype=float)
        except Exception as e:
            print(f"Runde {rnum}: feil {e}")
            continue

        # Test alle vitality-verdier
        best_score = -1
        best_v = 0.5
        scores_by_v = {}

        for v in vitality_values:
            pred = predict_distribution(grid, settlements, vitality=v)
            wkl = weighted_kl(gt_array, pred)
            s = score_from_kl(wkl)
            scores_by_v[v] = s
            if s > best_score:
                best_score = s
                best_v = v

        default_score = scores_by_v[0.5]
        all_best_scores.append(best_score)
        all_default_scores.append(default_score)

        # Vis resultater
        v_str = " ".join(f"{v:.1f}:{scores_by_v[v]:5.1f}" for v in [0.0, 0.3, 0.5, 0.7, 1.0])
        print(f"Runde {rnum:>2} (w={weight:.2f}): best={best_score:5.1f} @v={best_v:.1f}  "
              f"default={default_score:5.1f}  [{v_str}]")

        time.sleep(0.3)

    print(f"\n=== TOTALT (seed 0) ===")
    if all_best_scores:
        print(f"  Oracle (best vitality):  {np.mean(all_best_scores):.1f}  "
              f"(min={min(all_best_scores):.1f}, max={max(all_best_scores):.1f})")
        print(f"  Default (vitality=0.5):  {np.mean(all_default_scores):.1f}  "
              f"(min={min(all_default_scores):.1f}, max={max(all_default_scores):.1f})")
        print(f"\n  Oracle forbedring: +{np.mean(all_best_scores) - np.mean(all_default_scores):.1f}")


if __name__ == "__main__":
    main()
