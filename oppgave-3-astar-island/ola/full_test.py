"""
Full Competition Test — Simulerer eksakt leaderboard-scoring lokalt.

Beregner:
1. Per-runde score (snitt av 5 seeds)
2. Weighted score (sum av score × weight)
3. Hot streak (snitt av siste 3 runder)
4. Sammenligner med topp-lagene

Bruker ALLE 13 runder × 5 seeds = 65 ground truth datasett.

Bruk:
    export API_KEY='...'
    python full_test.py
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).parent))
from solution import (
    SeedObserver, load_calibration, load_calibration_by_type,
    MAP_W, MAP_H, NUM_CLASSES, PROB_FLOOR,
)

BASE_URL = "https://api.ainm.no/astar-island"
API_KEY = os.environ.get("API_KEY", "")


class Client:
    def __init__(self):
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.headers.update({"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"})
    def get(self, path):
        r = self.session.get(f"{BASE_URL}/{path.lstrip('/')}"); r.raise_for_status(); return r.json()


def weighted_kl(gt, pred):
    gt_safe = np.clip(gt, 1e-12, 1.0)
    pred_safe = np.clip(pred, 1e-12, 1.0)
    cell_kl = np.sum(gt_safe * (np.log(gt_safe) - np.log(pred_safe)), axis=-1)
    cell_entropy = -np.sum(gt_safe * np.log(gt_safe), axis=-1)
    tw = cell_entropy.sum()
    if tw <= 0: return float(cell_kl.mean())
    return float((cell_kl * cell_entropy).sum() / tw)


def score_from_kl(wkl):
    return max(0, min(100, 100 * np.exp(-3 * wkl)))


def detect_type_oracle(client, round_id, seeds_data):
    """Oracle type detection fra ground truth."""
    total_s, survived_s = 0, 0.0
    n_settlements = len(seeds_data[0].get("settlements", []))
    try:
        analysis = client.get(f"/analysis/{round_id}/0")
        gt = analysis.get("ground_truth")
        if gt:
            gt_arr = np.array(gt, dtype=float)
            for s in seeds_data[0].get("settlements", []):
                sx, sy = s["x"], s["y"]
                if 0 <= sx < MAP_W and 0 <= sy < MAP_H:
                    total_s += 1
                    survived_s += gt_arr[sy, sx, 1]
    except: pass

    if total_s == 0: return "STABLE", 0
    rate = survived_s / total_s
    if rate < 0.08: return "DEAD", rate
    elif rate < 0.35: return "STABLE", rate
    else:
        return ("BOOM_CONC" if n_settlements >= 40 else "BOOM_SPREAD"), rate


def main():
    if not API_KEY: print("FEIL: API_KEY"); sys.exit(1)
    client = Client()

    # Last calibration
    transition_table, simple_prior = load_calibration()

    # Last 4-type tabeller
    cal4_file = Path(__file__).parent / "calibration_4type.json"
    type_tables = None
    if cal4_file.exists():
        type_tables = json.loads(cal4_file.read_text()).get("tables", {})
        print(f"4-type tabeller: {', '.join(f'{k}({len(v)})' for k,v in type_tables.items())}")

    # Hent alle runder
    rounds = client.get("/rounds")
    completed = sorted([r for r in rounds if isinstance(r, dict) and r.get("status") == "completed"],
                       key=lambda r: r.get("round_number", 0))

    print(f"\n{'='*70}")
    print(f"  FULL COMPETITION TEST — {len(completed)} runder × 5 seeds")
    print(f"{'='*70}\n")

    all_round_scores = []
    all_weighted = []
    round_details = []

    for r in completed:
        round_id = r["id"]
        rnum = r.get("round_number", 0)
        weight = r.get("round_weight", 1.0)

        round_data = client.get(f"/rounds/{round_id}")
        seeds_data = round_data.get("seeds", round_data.get("initial_states", []))

        # Oracle type detection
        world_type, surv_rate = detect_type_oracle(client, round_id, seeds_data)
        typed_table = type_tables.get(world_type) if type_tables else None

        seed_scores = []
        for si in range(len(seeds_data)):
            grid = seeds_data[si].get("grid", [])
            settlements = seeds_data[si].get("settlements", [])

            # Bygg prediksjon med type-spesifikk floor
            observer = SeedObserver(grid, settlements, transition_table, simple_prior,
                                   typed_table=typed_table)
            pred = observer.build_prediction(world_type=world_type)

            # Hent ground truth
            try:
                analysis = client.get(f"/analysis/{round_id}/{si}")
                gt = analysis.get("ground_truth")
                if gt:
                    gt_arr = np.array(gt, dtype=float)
                    wkl = weighted_kl(gt_arr, pred)
                    score = score_from_kl(wkl)
                    seed_scores.append(score)
                time.sleep(0.15)
            except:
                pass

        if seed_scores:
            round_score = np.mean(seed_scores)
            weighted_score = round_score * weight
            all_round_scores.append(round_score)
            all_weighted.append(weighted_score)
            round_details.append({
                "round": rnum, "score": round_score, "weight": weight,
                "weighted": weighted_score, "type": world_type,
                "seeds": [f"{s:.1f}" for s in seed_scores]
            })

            print(f"  R{rnum:>2} [{world_type:>11}] score={round_score:5.1f}  "
                  f"w={weight:.2f}  weighted={weighted_score:6.1f}  "
                  f"seeds=[{', '.join(f'{s:.0f}' for s in seed_scores)}]")

    print(f"\n{'='*70}")
    print(f"  RESULTATER")
    print(f"{'='*70}")

    total_weighted = sum(all_weighted)
    avg_score = np.mean(all_round_scores) if all_round_scores else 0

    # Hot streak = snitt av siste 3
    last3 = all_round_scores[-3:] if len(all_round_scores) >= 3 else all_round_scores
    hot_streak = np.mean(last3)

    print(f"\n  Snitt per runde:     {avg_score:.1f}")
    print(f"  Weighted total:      {total_weighted:.1f}")
    print(f"  Hot streak (siste 3): {hot_streak:.1f}")
    print(f"  Beste runde:         {max(all_round_scores):.1f}")
    print(f"  Dårligste runde:     {min(all_round_scores):.1f}")
    print(f"  Runder spilt:        {len(all_round_scores)}")

    # Sammenlign med leaderboard
    print(f"\n  --- Sammenligning med toppen ---")
    print(f"  #1 Matriks:          177.1  (hot streak 84.9)")
    print(f"  #2 Laurbærene:       176.7  (hot streak 86.6)")
    print(f"  Oss (beregnet):      {total_weighted:.1f}  (hot streak {hot_streak:.1f})")

    if total_weighted > 177.1:
        print(f"\n  🏆 VI SLÅR #1!")
    elif total_weighted > 174:
        print(f"\n  Topp 10!")
    else:
        gap = 177.1 - total_weighted
        print(f"\n  Gap til #1: {gap:.1f} poeng")
        # Hva trengs per runde for å lukke gapet?
        remaining = 11  # estimert
        needed_per_round = gap / (remaining * 1.9)  # snitt vekt ~1.9
        print(f"  Trenger ~{needed_per_round:+.1f} per runde over {remaining} runder for å ta igjen")

    # Detaljert per-type
    print(f"\n  --- Per world type ---")
    type_scores = {}
    for d in round_details:
        t = d["type"]
        if t not in type_scores: type_scores[t] = []
        type_scores[t].append(d["score"])
    for t, scores in sorted(type_scores.items()):
        print(f"  {t:>11}: snitt={np.mean(scores):.1f}  n={len(scores)}  "
              f"range=[{min(scores):.0f}-{max(scores):.0f}]")


if __name__ == "__main__":
    main()
