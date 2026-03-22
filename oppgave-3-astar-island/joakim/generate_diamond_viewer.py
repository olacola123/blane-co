"""
Generate a visual HTML viewer for Diamond (Manhattan) solver predictions vs ground truth.
Fetches all completed rounds from the API, generates prior-only predictions with
solution_diamond.py, scores using the official entropy-weighted KL divergence formula.

Usage:
    export API_KEY='jwt-token'
    cd oppgave-3-astar-island/joakim
    python generate_diamond_viewer.py
"""
import importlib.util
import io
import contextlib
import json
import math
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import backtest_diamond_full as bt
import solution_diamond as diamond_solver

BASE_URL = "https://api.ainm.no/astar-island"
API_KEY = os.environ.get("API_KEY", "")

OLA_DIR = Path(__file__).parent.parent / "ola"
JOAKIM_DIR = Path(__file__).parent
HISTORY_ROOT = JOAKIM_DIR / "history"

MAP_W, MAP_H = 40, 40
NUM_CLASSES = 6

# Static terrain codes (excluded from scoring)
STATIC_TERRAIN = {10, 5}  # ocean, mountain


class Client:
    def __init__(self):
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.headers.update({
            "Authorization": f"Bearer {API_KEY}",
            "Accept": "application/json",
        })

    def get(self, path):
        r = self.session.get(f"{BASE_URL}/{path.lstrip('/')}", timeout=30)
        r.raise_for_status()
        return r.json()


def load_diamond_module():
    """Import solution_diamond.py with mocked requests (we use our own Client)."""
    # Save real modules
    saved = {}
    for mod_name in ["requests", "requests.adapters", "urllib3", "urllib3.util", "urllib3.util.retry"]:
        if mod_name in sys.modules:
            saved[mod_name] = sys.modules[mod_name]

    # Install mocks so solution_diamond.py can import without errors
    mock_requests = types.ModuleType("requests")
    mock_requests.Session = type("Session", (), {"__init__": lambda s: None})
    sys.modules["requests"] = mock_requests

    mock_adapters = types.ModuleType("requests.adapters")
    mock_adapters.HTTPAdapter = type("HTTPAdapter", (), {"__init__": lambda s, **kw: None})
    sys.modules["requests.adapters"] = mock_adapters

    mock_urllib3 = types.ModuleType("urllib3")
    mock_util = types.ModuleType("urllib3.util")
    mock_retry = types.ModuleType("urllib3.util.retry")
    mock_retry.Retry = type("Retry", (), {"__init__": lambda s, **kw: None})
    sys.modules["urllib3"] = mock_urllib3
    sys.modules["urllib3.util"] = mock_util
    sys.modules["urllib3.util.retry"] = mock_retry

    if str(OLA_DIR) not in sys.path:
        sys.path.insert(0, str(OLA_DIR))

    spec = importlib.util.spec_from_file_location("diamond", JOAKIM_DIR / "solution_diamond.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Restore real modules
    for mod_name, real_mod in saved.items():
        sys.modules[mod_name] = real_mod

    return mod


def build_prediction(mod, grid, settlements):
    """Build prior-only prediction using diamond module's blended prediction."""
    transition_table, simple_prior = mod.load_calibration()
    opt_tables = mod.load_optimized_calibration()
    type_weights = mod.compute_type_weights(0.33)

    pred = mod.build_blended_prediction(
        grid, settlements, transition_table, simple_prior,
        opt_tables, type_weights)

    if pred is None:
        obs = mod.SeedObserver(grid, settlements, transition_table, simple_prior,
                               alpha=mod.DEFAULT_ALPHA, opt_tables=opt_tables,
                               world_type="STABLE")
        pred = obs.build_prediction(apply_smoothing=False)

    return pred


def official_score(gt_arr, pred_arr, initial_grid):
    """
    Compute the official competition score using entropy-weighted KL divergence.
    Only dynamic cells count (not ocean=10, mountain=5).

    Formula:
        KL(cell) = sum(gt_i * log(gt_i / pred_i))
        H(cell) = -sum(gt_i * log(gt_i))
        weighted_kl = sum(H * KL) / sum(H)   [dynamic cells only]
        score = max(0, min(100, 100 * exp(-3 * weighted_kl)))
    """
    eps = 1e-12
    gt_safe = np.clip(gt_arr, eps, 1.0)
    pred_safe = np.clip(pred_arr, eps, 1.0)

    # Per-cell KL divergence
    cell_kl = np.sum(gt_safe * np.log(gt_safe / pred_safe), axis=-1)

    # Per-cell entropy
    cell_entropy = -np.sum(gt_safe * np.log(gt_safe), axis=-1)

    # Dynamic cell mask
    grid = np.array(initial_grid)
    dynamic_mask = np.ones((MAP_H, MAP_W), dtype=bool)
    for t in STATIC_TERRAIN:
        dynamic_mask &= (grid != t)

    # Entropy-weighted KL over dynamic cells
    masked_entropy = cell_entropy * dynamic_mask
    masked_kl = cell_kl * dynamic_mask

    total_entropy = masked_entropy.sum()
    if total_entropy > 0:
        weighted_kl = (masked_entropy * masked_kl).sum() / total_entropy
    else:
        weighted_kl = masked_kl.mean()

    score = max(0.0, min(100.0, 100.0 * math.exp(-3.0 * weighted_kl)))
    return score, cell_kl


def detect_world_type(gt_arr, settlements):
    """Detect world type from ground truth (oracle for backtest)."""
    total_s, survived_s = 0, 0.0
    n_settlements = len(settlements)
    for s in settlements:
        sx, sy = s["x"], s["y"]
        if 0 <= sx < MAP_W and 0 <= sy < MAP_H:
            total_s += 1
            survived_s += gt_arr[sy, sx, 1]
    if total_s == 0:
        return "STABLE", 0
    rate = survived_s / total_s
    if rate < 0.08:
        return "DEAD", rate
    elif rate < 0.35:
        return "STABLE", rate
    else:
        return ("BOOM_CONC" if n_settlements >= 40 else "BOOM_SPREAD"), rate


def compact_tensor(tensor, precision=4):
    return [
        [
            [round(float(tensor[y, x, c]), precision) for c in range(NUM_CLASSES)]
            for x in range(MAP_W)
        ]
        for y in range(MAP_H)
    ]


def compact_grid(grid):
    if isinstance(grid, np.ndarray):
        return grid.tolist()
    if isinstance(grid, list) and len(grid) > 0 and isinstance(grid[0], list):
        return grid
    return grid


def load_array(round_dir, relative_path):
    if not relative_path:
        return None
    path = round_dir / relative_path
    if not path.exists():
        return None
    return np.load(path)


def parse_numeric_score(value):
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_live_history_rounds(existing_round_ids):
    """Load locally stored prediction-only rounds so live submissions are visible in the viewer."""
    viewer_rounds = []
    if not HISTORY_ROOT.exists():
        return viewer_rounds

    for round_dir in sorted(HISTORY_ROOT.iterdir()):
        if not round_dir.is_dir():
            continue
        manifest_path = round_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        manifest = json.loads(manifest_path.read_text())
        round_id = str(manifest.get("round_id") or round_dir.name)
        if round_id in existing_round_ids:
            continue

        predictions_manifest = manifest.get("predictions", {})
        if not predictions_manifest:
            continue

        round_metadata = manifest.get("round_metadata") or {}
        diagnostics = manifest.get("diagnostics") or {}
        inference = diagnostics.get("inference") or {}
        prediction_diag = diagnostics.get("prediction") or {}
        final_results = diagnostics.get("submission", {}).get("final_results") or []
        seed_result_by_index = {
            int(item.get("seed_index")): item
            for item in final_results
            if isinstance(item, dict) and item.get("seed_index") is not None
        }
        query_counts = {}
        for query in manifest.get("queries", []):
            seed_index = int(query.get("seed_index", -1))
            query_counts[seed_index] = query_counts.get(seed_index, 0) + 1

        viewer_seeds = []
        seed_scores = []
        for initial_state in sorted(
            manifest.get("initial_states", []),
            key=lambda item: int(item["seed_index"]),
        ):
            seed_index = int(initial_state["seed_index"])
            initial_grid = load_array(round_dir, initial_state.get("grid_path"))
            prediction = load_array(round_dir, predictions_manifest.get(str(seed_index)))
            if initial_grid is None or prediction is None:
                continue

            seed_diag = prediction_diag.get(str(seed_index), {})
            result_score = parse_numeric_score(
                seed_result_by_index.get(seed_index, {}).get("score")
            )
            if result_score is not None:
                seed_scores.append(result_score)

            viewer_seeds.append(
                {
                    "seed_index": seed_index,
                    "label": f"Seed {seed_index}",
                    "initial_grid": compact_grid(initial_grid),
                    "prediction": compact_tensor(prediction),
                    "ground_truth": None,
                    "kl_map": None,
                    "score": result_score,
                    "query_count": int(query_counts.get(seed_index, 0)),
                    "reason": seed_diag.get("reason"),
                    "dynamic_observed_cells": seed_diag.get("observed_dynamic_cells"),
                    "dynamic_total_cells": seed_diag.get("dynamic_cells"),
                    "mean_observations_per_dynamic_cell": seed_diag.get("mean_observations_per_dynamic_cell"),
                }
            )

        if not viewer_seeds:
            continue

        round_score = float(np.mean(seed_scores)) if seed_scores else None
        round_number = round_metadata.get("round_number")
        label = f"Round {round_number}" if round_number is not None else round_id
        world_type = inference.get("hard_world_type", "UNKNOWN")
        mode = "live_history"
        print(f"  {label} [local/{world_type}] queries={len(manifest.get('queries', []))}")
        viewer_rounds.append(
            {
                "label": label,
                "round_number": round_number,
                "round_id": round_id,
                "weight": round_metadata.get("round_weight", 1.0),
                "mode": mode,
                "world_type": world_type,
                "query_count": len(manifest.get("queries", [])),
                "probe_queries_each": diagnostics.get("query", {}).get("probe_queries_each"),
                "vitality": inference.get("vitality"),
                "spatial_coherence": inference.get("spatial_coherence"),
                "coherence_evidence": inference.get("coherence_evidence"),
                "score": round(round_score, 1) if round_score is not None else None,
                "seeds": viewer_seeds,
            }
        )

    return viewer_rounds


def main():
    if not API_KEY:
        print("FEIL: API_KEY mangler. export API_KEY='din-jwt-token'")
        sys.exit(1)

    client = Client()
    transition_table, simple_prior = diamond_solver.load_calibration()
    _ = diamond_solver.load_calibration_by_type()
    opt_tables = diamond_solver.load_optimized_calibration()
    learning = diamond_solver.load_learning_state()
    alpha = float(learning.get("alpha", diamond_solver.DEFAULT_ALPHA))
    coherence_multiplier = float(getattr(diamond_solver, 'COHERENCE_STRENGTH_MULTIPLIER', 1.0))

    # Fetch all completed rounds
    all_rounds = client.get("/rounds")
    completed = sorted(
        [
            r for r in all_rounds
            if isinstance(r, dict)
            and r.get("status") == "completed"
        ],
        key=lambda r: r.get("round_number", 0),
    )

    print(f"Found {len(completed)} completed rounds\n")

    viewer_rounds = []

    for ri, r in enumerate(completed):
        round_id = r["id"]
        rnum = r.get("round_number", 0)
        weight = r.get("round_weight", 1.0)

        print(f"  R{rnum} ({ri+1}/{len(completed)})...", end=" ", flush=True)

        # Fetch round details (grids + settlements)
        round_data = client.get(f"/rounds/{round_id}")
        seeds_data = round_data.get("seeds", round_data.get("initial_states", []))
        ground_truth_by_seed = bt.fetch_ground_truth(client, round_id, len(seeds_data))
        manifest = bt.load_manifest(round_id)

        if manifest and len(manifest.get("queries", [])) > 0:
            predictions, replay_meta = bt.predict_history_replay_round(
                round_id=round_id,
                round_data=round_data,
                manifest=manifest,
                transition_table=transition_table,
                simple_prior=simple_prior,
                opt_tables=opt_tables,
                alpha=alpha,
            )
            mode = "history_replay"
            world_type = replay_meta.get("world_type", "STABLE")
        else:
            predictions, replay_meta = bt.predict_prior_only_round(
                round_data=round_data,
                transition_table=transition_table,
                simple_prior=simple_prior,
                opt_tables=opt_tables,
            )
            mode = "prior_only"
            world_type = "STABLE"

        viewer_seeds = []
        seed_scores = []

        for si in range(len(seeds_data)):
            grid = seeds_data[si].get("grid", [])
            settlements = seeds_data[si].get("settlements", [])
            pred = predictions[si]
            gt_arr = ground_truth_by_seed[si]["ground_truth"]
            seed_score, cell_kl = official_score(gt_arr, pred, grid)
            kl_map = cell_kl
            seed_scores.append(seed_score)

            if si == 0 and mode == "prior_only":
                world_type, _ = detect_world_type(gt_arr, settlements)

            seed_meta = replay_meta.get("seed_results", {}).get(si, {})

            seed_data = {
                "seed_index": si,
                "label": f"Seed {si}",
                "initial_grid": compact_grid(grid),
                "prediction": compact_tensor(pred),
                "score": round(seed_score, 1) if seed_score is not None else None,
                "query_count": int(seed_meta.get("query_count", 0)),
                "reason": seed_meta.get("reason"),
                "dynamic_observed_cells": seed_meta.get("dynamic_observed_cells"),
                "dynamic_total_cells": seed_meta.get("dynamic_total_cells"),
                "mean_observations_per_dynamic_cell": seed_meta.get("mean_observations_per_dynamic_cell"),
            }
            seed_data["ground_truth"] = compact_tensor(gt_arr)
            seed_data["kl_map"] = [[round(float(kl_map[y, x]), 4) for x in range(MAP_W)] for y in range(MAP_H)]

            viewer_seeds.append(seed_data)

        round_score = float(np.mean(seed_scores)) if seed_scores else 0
        print(f"[{mode}/{world_type}] score={round_score:.1f}")

        viewer_rounds.append({
            "label": f"Round {rnum}",
            "round_number": rnum,
            "round_id": round_id,
            "weight": weight,
            "mode": mode,
            "world_type": world_type,
            "query_count": int(replay_meta.get("query_count", 0)),
            "probe_queries_each": replay_meta.get("probe_queries_each"),
            "vitality": replay_meta.get("vitality"),
            "spatial_coherence": replay_meta.get("spatial_coherence"),
            "coherence_evidence": replay_meta.get("coherence_evidence"),
            "score": round(round_score, 1),
            "seeds": viewer_seeds,
        })

    existing_round_ids = {round_entry["round_id"] for round_entry in viewer_rounds}
    viewer_rounds.extend(build_live_history_rounds(existing_round_ids))
    viewer_rounds.sort(
        key=lambda item: (
            item.get("round_number") is None,
            item.get("round_number") if item.get("round_number") is not None else 10**9,
            item.get("round_id", ""),
        )
    )

    data = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "solver": "Diamond — hybrid historical backtest with local live submissions",
        "alpha": alpha,
        "coherence_multiplier": coherence_multiplier,
        "grid_size": 40,
        "class_names": ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"],
        "class_keys": ["empty", "settlement", "port", "ruin", "forest", "mountain"],
        "class_colors": {
            "empty": "#e7d7a7",
            "settlement": "#f59e0b",
            "port": "#22d3ee",
            "ruin": "#ef4444",
            "forest": "#5bbf59",
            "mountain": "#6366f1",
        },
        "terrain_labels": {
            "0": "Empty", "10": "Ocean", "11": "Plains",
            "1": "Settlement", "2": "Port", "3": "Ruin",
            "4": "Forest", "5": "Mountain",
        },
        "terrain_base_colors": {
            "0": "#16181d", "10": "#0c2542", "11": "#16181d",
            "1": "#332711", "2": "#113345", "3": "#341617",
            "4": "#142317", "5": "#211d3f",
        },
        "rounds": viewer_rounds,
    }

    html = build_html(json.dumps(data, separators=(",", ":")))

    out_path = JOAKIM_DIR / "diamond_viewer.html"
    out_path.write_text(html)
    print(f"\nSkrevet til {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")


def build_html(data_json):
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Diamond — Prediction vs Ground Truth</title>
  <style>
    :root {{
      --bg: #0d1830;
      --panel: #132240;
      --border: rgba(140, 170, 255, 0.18);
      --text: #ecf2ff;
      --muted: #9ca9c9;
      --accent: #8bb3ff;
      --shadow: rgba(0, 0, 0, 0.32);
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: "SFMono-Regular", "Menlo", "Monaco", "Consolas", monospace;
      background:
        radial-gradient(circle at top left, rgba(88, 122, 220, 0.18), transparent 34%),
        linear-gradient(180deg, #102041 0%, #0a1428 100%);
      color: var(--text);
      min-height: 100vh;
    }}
    .app {{
      width: min(1500px, calc(100vw - 32px));
      margin: 16px auto;
      padding: 20px;
      background: rgba(11, 22, 45, 0.82);
      border: 1px solid var(--border);
      border-radius: 24px;
      box-shadow: 0 18px 60px var(--shadow);
      backdrop-filter: blur(10px);
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 16px;
      flex-wrap: wrap;
    }}
    .title h1 {{ margin: 0 0 6px; font-size: 28px; letter-spacing: -0.03em; }}
    .title p {{ color: var(--muted); max-width: 760px; line-height: 1.5; font-size: 13px; }}
    .controls {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }}
    label {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      color: var(--muted);
      font-size: 13px;
    }}
    select {{
      min-width: 200px;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: #12213e;
      color: var(--text);
      font: inherit;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin: 18px 0 22px;
    }}
    .summary-card {{
      background: rgba(20, 34, 64, 0.88);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 14px 16px;
    }}
    .summary-card .label {{
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }}
    .summary-card .value {{
      font-size: 22px;
      font-weight: 700;
      letter-spacing: -0.03em;
    }}
    .summary-card .subvalue {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
    }}

    .overview-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 16px;
      margin-bottom: 24px;
    }}
    .overview-panel {{
      background: rgba(19, 33, 62, 0.92);
      border: 1px solid var(--border);
      border-radius: 22px;
      padding: 14px;
      text-align: center;
    }}
    .overview-panel h3 {{
      font-size: 14px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 10px;
    }}
    canvas {{
      width: 100%;
      aspect-ratio: 1 / 1;
      display: block;
      border-radius: 14px;
      background: #0b1018;
      image-rendering: pixelated;
      cursor: crosshair;
    }}

    .layer-header {{
      display: grid;
      grid-template-columns: 120px 1fr 1fr 1fr;
      gap: 14px;
      align-items: end;
      margin-bottom: 10px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .layer-grid {{
      display: grid;
      gap: 16px;
    }}
    .layer-row {{
      display: grid;
      grid-template-columns: 120px 1fr 1fr 1fr;
      gap: 14px;
      align-items: center;
    }}
    .layer-label {{
      font-size: 16px;
      letter-spacing: -0.02em;
      font-weight: 600;
    }}
    .layer-cell {{
      background: rgba(11, 20, 38, 0.72);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 8px;
    }}
    .dimmed {{ opacity: 0.35; filter: grayscale(0.4); }}

    .cell-panel {{
      background: rgba(19, 33, 62, 0.92);
      border: 1px solid var(--border);
      border-radius: 22px;
      padding: 18px;
      margin-top: 24px;
    }}
    .cell-title {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: baseline;
      margin-bottom: 14px;
    }}
    .cell-title h2 {{ font-size: 18px; letter-spacing: -0.03em; }}
    .cell-title .meta {{ color: var(--muted); font-size: 13px; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      background: rgba(20, 34, 64, 0.88);
      border: 1px solid var(--border);
      border-radius: 999px;
      font-size: 12px;
      color: var(--muted);
    }}
    .badge strong {{ color: var(--text); }}
    .cell-grid {{
      display: grid;
      grid-template-columns: repeat(6, minmax(100px, 1fr));
      gap: 10px;
    }}
    .cell-card {{
      background: rgba(14, 23, 42, 0.84);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px 10px;
    }}
    .cell-card .class-name {{
      color: var(--muted);
      margin-bottom: 8px;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .cell-card .prediction {{
      font-size: 24px;
      font-weight: 700;
      letter-spacing: -0.04em;
      line-height: 1;
      margin-bottom: 6px;
    }}
    .cell-card .ground-truth,
    .cell-card .delta {{
      font-size: 12px;
      color: var(--muted);
      line-height: 1.4;
    }}

    .type-STABLE {{ color: #34d399; }}
    .type-DEAD {{ color: #f87171; }}
    .type-BOOM_SPREAD {{ color: #fbbf24; }}
    .type-BOOM_CONC {{ color: #fb923c; }}
    .mode-history_replay {{ color: #34d399; }}
    .mode-prior_only {{ color: #fbbf24; }}
    .mode-live_history {{ color: #60a5fa; }}

    @media (max-width: 1100px) {{
      .overview-grid {{ grid-template-columns: 1fr; }}
      .layer-header, .layer-row {{ grid-template-columns: 1fr; }}
      .cell-grid {{ grid-template-columns: repeat(3, 1fr); }}
    }}
    @media (max-width: 700px) {{
      .app {{ width: calc(100vw - 16px); margin: 8px auto; padding: 14px; }}
      .header {{ flex-direction: column; }}
      .controls, label, select {{ width: 100%; }}
      .cell-grid {{ grid-template-columns: 1fr 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <div class="header">
      <div class="title">
        <h1>Diamond — Prediction vs Ground Truth</h1>
        <p>
          Hybrid historisk backtest med kalibrert coherence-modulasjon.
          Viewer viser `history_replay` der lokale query-logger finnes, ellers `prior_only`.
          Nye live-runder uten ground truth vises som `live_history` med prediction-only visning.
        </p>
      </div>
      <div class="controls">
        <label>
          Round
          <select id="round-select"></select>
        </label>
        <label>
          Seed
          <select id="seed-select"></select>
        </label>
      </div>
    </div>

    <div id="summary-grid" class="summary-grid"></div>

    <div class="overview-grid">
      <div class="overview-panel">
        <h3>Prediction (Diamond)</h3>
        <canvas id="pred-overview" width="40" height="40"></canvas>
      </div>
      <div class="overview-panel">
        <h3>Ground Truth</h3>
        <canvas id="gt-overview" width="40" height="40"></canvas>
      </div>
      <div class="overview-panel">
        <h3>KL Divergence</h3>
        <canvas id="kl-overview" width="40" height="40"></canvas>
      </div>
    </div>

    <div class="layer-header">
      <div></div>
      <div>Prediction</div>
      <div>Ground Truth</div>
      <div>KL Error</div>
    </div>
    <div id="layer-grid" class="layer-grid"></div>

    <div class="cell-panel">
      <div class="cell-title">
        <h2 id="cell-title">Cell (20, 20)</h2>
        <div class="meta" id="cell-meta"></div>
      </div>
      <div id="cell-grid" class="cell-grid"></div>
    </div>
  </div>

  <script id="history-data" type="application/json">{data_json}</script>
  <script>
    const DATA = JSON.parse(document.getElementById("history-data").textContent);
    const roundSelect = document.getElementById("round-select");
    const seedSelect = document.getElementById("seed-select");
    const summaryGrid = document.getElementById("summary-grid");
    const layerGrid = document.getElementById("layer-grid");
    const cellGridEl = document.getElementById("cell-grid");
    const cellTitleEl = document.getElementById("cell-title");
    const cellMetaEl = document.getElementById("cell-meta");
    const predOverview = document.getElementById("pred-overview");
    const gtOverview = document.getElementById("gt-overview");
    const klOverview = document.getElementById("kl-overview");

    const S = {{ roundIndex: Math.max(DATA.rounds.length - 1, 0), seedIndex: 0, cell: {{ x: 20, y: 20 }} }};
    const G = DATA.grid_size;

    function clamp(v, lo, hi) {{ return Math.max(lo, Math.min(hi, v)); }}
    function parseHex(h) {{
      const c = h.replace("#", "");
      return [parseInt(c.slice(0,2),16), parseInt(c.slice(2,4),16), parseInt(c.slice(4,6),16)];
    }}
    function mix(a, b, t) {{
      t = clamp(t, 0, 1);
      return [Math.round(a[0]+(b[0]-a[0])*t), Math.round(a[1]+(b[1]-a[1])*t), Math.round(a[2]+(b[2]-a[2])*t)];
    }}
    function pct(v) {{ return (v*100).toFixed(1)+"%"; }}
    function dpct(p, t) {{
      const d = (p - t) * 100;
      return (d >= 0 ? "+" : "") + d.toFixed(1) + " pp";
    }}

    function getRound() {{ return DATA.rounds[S.roundIndex]; }}
    function getSeed() {{ return getRound().seeds[S.seedIndex]; }}

    function drawHighlight(ctx) {{
      ctx.save();
      ctx.fillStyle = "rgba(255,255,255,0.3)";
      ctx.fillRect(S.cell.x, S.cell.y, 1, 1);
      ctx.restore();
    }}

    function drawOverviewCanvas(canvas, grid, tensor) {{
      const ctx = canvas.getContext("2d");
      const img = ctx.createImageData(G, G);
      const tc = DATA.terrain_base_colors;
      for (let y = 0; y < G; y++) {{
        for (let x = 0; x < G; x++) {{
          const base = parseHex(tc[String(grid[y][x])] || "#111827");
          let px = base;
          if (tensor) {{
            let best = 0, bestV = tensor[y][x][0];
            for (let c = 1; c < 6; c++) {{
              if (tensor[y][x][c] > bestV) {{ bestV = tensor[y][x][c]; best = c; }}
            }}
            const accent = parseHex(DATA.class_colors[DATA.class_keys[best]]);
            px = mix(base, accent, Math.pow(clamp(bestV, 0, 1), 0.92));
          }}
          const i = (y * G + x) * 4;
          img.data[i] = px[0]; img.data[i+1] = px[1]; img.data[i+2] = px[2]; img.data[i+3] = 255;
        }}
      }}
      ctx.putImageData(img, 0, 0);
      drawHighlight(ctx);
    }}

    function drawKLOverview(canvas, klMap) {{
      const ctx = canvas.getContext("2d");
      const img = ctx.createImageData(G, G);
      const bg = [11, 16, 24];
      const maxKL = 1.5;
      for (let y = 0; y < G; y++) {{
        for (let x = 0; x < G; x++) {{
          const kl = klMap ? clamp(klMap[y][x] / maxKL, 0, 1) : 0;
          const t = Math.pow(kl, 0.6);
          const r = Math.round(bg[0] + (248-bg[0])*t);
          const g = Math.round(bg[1] + (113-bg[1])*t*0.4);
          const b = Math.round(bg[2] + (113-bg[2])*t*0.3);
          const i = (y*G+x)*4;
          img.data[i] = r; img.data[i+1] = g; img.data[i+2] = b; img.data[i+3] = 255;
        }}
      }}
      ctx.putImageData(img, 0, 0);
      drawHighlight(ctx);
    }}

    function drawLayer(canvas, matrix, ci) {{
      const ctx = canvas.getContext("2d");
      const img = ctx.createImageData(G, G);
      const accent = parseHex(DATA.class_colors[DATA.class_keys[ci]]);
      const bg = parseHex("#0b1018");
      for (let y = 0; y < G; y++) {{
        for (let x = 0; x < G; x++) {{
          const t = matrix ? Math.pow(clamp(matrix[y][x], 0, 1), 0.88) : 0;
          const px = mix(bg, accent, t);
          const i = (y*G+x)*4;
          img.data[i] = px[0]; img.data[i+1] = px[1]; img.data[i+2] = px[2]; img.data[i+3] = matrix ? 255 : 96;
        }}
      }}
      ctx.putImageData(img, 0, 0);
      drawHighlight(ctx);
    }}

    function drawKLLayer(canvas, klMap) {{
      const ctx = canvas.getContext("2d");
      const img = ctx.createImageData(G, G);
      const maxKL = 1.5;
      for (let y = 0; y < G; y++) {{
        for (let x = 0; x < G; x++) {{
          const kl = klMap ? clamp(klMap[y][x] / maxKL, 0, 1) : 0;
          const t = Math.pow(kl, 0.6);
          const r = Math.round(11 + (248-11)*t);
          const g = Math.round(16 + (113-16)*t*0.4);
          const b = Math.round(24 + (113-24)*t*0.3);
          const i = (y*G+x)*4;
          img.data[i] = r; img.data[i+1] = g; img.data[i+2] = b; img.data[i+3] = 255;
        }}
      }}
      ctx.putImageData(img, 0, 0);
      drawHighlight(ctx);
    }}

    function bindCanvas(c) {{
      if (c.dataset.bound) return;
      c.dataset.bound = "1";
      function handler(e) {{
        const r = c.getBoundingClientRect();
        S.cell.x = clamp(Math.floor((e.clientX-r.left)/r.width*G), 0, G-1);
        S.cell.y = clamp(Math.floor((e.clientY-r.top)/r.height*G), 0, G-1);
        renderCanvases();
        renderCell();
      }}
      c.addEventListener("mousemove", handler);
      c.addEventListener("click", handler);
    }}

    function renderSummary() {{
      const rd = getRound();
      const sd = getSeed();
      const hasGT = !!sd.ground_truth;
      const scoreLabel = sd.score != null ? sd.score.toFixed(1) : "\u2014";
      const scoreColor = sd.score == null ? "#9ca9c9" : sd.score >= 85 ? "#34d399" : sd.score >= 70 ? "#fbbf24" : "#f87171";
      const modeLabel = rd.mode || "prior_only";
      const modeClass = `mode-${{modeLabel}}`;
      const coherenceLabel = rd.spatial_coherence != null ? rd.spatial_coherence.toFixed(3) : "\u2014";
      const queryLabel = rd.query_count != null ? String(rd.query_count) : "0";
      const querySub = sd.query_count != null ? `${{sd.label}}: ${{sd.query_count}} queries` : sd.label;
      const roundScoreLabel = rd.score != null ? rd.score.toFixed(1) : "\u2014";
      const roundScoreSub = rd.score != null ? `Round score: ${{roundScoreLabel}}` : "Prediction only";

      // Compute overall average
      const allScores = DATA.rounds.map(r => r.score).filter(s => s > 0);
      const avgScore = allScores.length > 0 ? (allScores.reduce((a,b) => a+b, 0) / allScores.length) : 0;

      summaryGrid.innerHTML = [
        `<div class="summary-card"><div class="label">Round</div><div class="value">${{rd.label}}</div><div class="subvalue">Weight: ${{rd.weight.toFixed(4)}}</div></div>`,
        `<div class="summary-card"><div class="label">Mode</div><div class="value ${{modeClass}}">${{modeLabel}}</div><div class="subvalue">Stored queries: ${{queryLabel}}</div></div>`,
        `<div class="summary-card"><div class="label">World Type</div><div class="value type-${{rd.world_type}}">${{rd.world_type}}</div><div class="subvalue">${{roundScoreSub}}</div></div>`,
        `<div class="summary-card"><div class="label">Coherence</div><div class="value">${{coherenceLabel}}</div><div class="subvalue">Multiplier: ${{DATA.coherence_multiplier.toFixed(2)}}</div></div>`,
        `<div class="summary-card"><div class="label">Seed Score</div><div class="value" style="color:${{scoreColor}}">${{scoreLabel}}</div><div class="subvalue">${{sd.label}}</div></div>`,
        `<div class="summary-card"><div class="label">Seed Queries</div><div class="value">${{sd.query_count ?? 0}}</div><div class="subvalue">${{querySub}}</div></div>`,
        `<div class="summary-card"><div class="label">Overall Avg</div><div class="value" style="color:${{avgScore >= 85 ? '#34d399' : avgScore >= 70 ? '#fbbf24' : '#f87171'}}">${{avgScore.toFixed(1)}}</div><div class="subvalue">${{allScores.length}} scored rounds</div></div>`,
        `<div class="summary-card"><div class="label">Scoring</div><div class="value" style="font-size:14px">${{hasGT ? "Official" : "Pending"}}</div><div class="subvalue">${{hasGT ? "entropy-weighted KL" : "ground truth not available"}}</div></div>`,
      ].join("");
    }}

    function renderLayers() {{
      const sd = getSeed();
      const hasGT = !!sd.ground_truth;
      layerGrid.innerHTML = "";
      for (let ci = 0; ci < 6; ci++) {{
        const row = document.createElement("div");
        row.className = "layer-row";
        row.innerHTML = `
          <div class="layer-label" style="color:${{DATA.class_colors[DATA.class_keys[ci]]}}">${{DATA.class_names[ci]}}</div>
          <div class="layer-cell"><canvas id="pred-${{ci}}" width="40" height="40"></canvas></div>
          <div class="layer-cell ${{hasGT ? "" : "dimmed"}}"><canvas id="gt-${{ci}}" width="40" height="40"></canvas></div>
          <div class="layer-cell ${{sd.kl_map ? "" : "dimmed"}}"><canvas id="kl-${{ci}}" width="40" height="40"></canvas></div>
        `;
        layerGrid.appendChild(row);
      }}
      for (const c of document.querySelectorAll("canvas")) bindCanvas(c);
    }}

    function renderCanvases() {{
      const sd = getSeed();
      const hasGT = !!sd.ground_truth;
      drawOverviewCanvas(predOverview, sd.initial_grid, sd.prediction);
      drawOverviewCanvas(gtOverview, sd.initial_grid, hasGT ? sd.ground_truth : null);
      drawKLOverview(klOverview, sd.kl_map || null);

      for (let ci = 0; ci < 6; ci++) {{
        const predC = document.getElementById(`pred-${{ci}}`);
        const gtC = document.getElementById(`gt-${{ci}}`);
        const klC = document.getElementById(`kl-${{ci}}`);
        if (predC) drawLayer(predC, sd.prediction.map(r => r.map(c => c[ci])), ci);
        if (gtC) drawLayer(gtC, hasGT ? sd.ground_truth.map(r => r.map(c => c[ci])) : null, ci);
        if (klC) {{
          drawKLLayer(klC, sd.kl_map || null);
        }}
      }}
    }}

    function renderCell() {{
      const sd = getSeed();
      const {{x, y}} = S.cell;
      const terrain = sd.initial_grid[y][x];
      const tLabel = DATA.terrain_labels[String(terrain)] || `Code ${{terrain}}`;
      const hasGT = !!sd.ground_truth;
      const cellKL = sd.kl_map ? sd.kl_map[y][x] : null;
      const isStatic = terrain === 10 || terrain === 5;

      cellTitleEl.textContent = `Cell (${{x}}, ${{y}})`;
      let badges = `<span class="badge"><strong>Terrain</strong> ${{tLabel}}</span>`;
      if (isStatic) {{
        badges += ` <span class="badge" style="color:#666"><strong>Static</strong> (not scored)</span>`;
      }}
      if (sd.reason) {{
        badges += ` <span class="badge"><strong>Reason</strong> ${{sd.reason}}</span>`;
      }}
      if (sd.dynamic_observed_cells != null && sd.dynamic_total_cells != null) {{
        badges += ` <span class="badge"><strong>Observed</strong> ${{sd.dynamic_observed_cells}}/${{sd.dynamic_total_cells}}</span>`;
      }}
      if (cellKL != null) {{
        const klColor = cellKL < 0.1 ? "#34d399" : cellKL < 0.5 ? "#fbbf24" : "#f87171";
        badges += ` <span class="badge"><strong>KL</strong> <span style="color:${{klColor}}">${{cellKL.toFixed(4)}}</span></span>`;
      }}
      cellMetaEl.innerHTML = badges;

      cellGridEl.innerHTML = "";
      for (let ci = 0; ci < 6; ci++) {{
        const p = sd.prediction[y][x][ci];
        const gt = hasGT ? sd.ground_truth[y][x][ci] : null;
        const card = document.createElement("div");
        card.className = "cell-card";
        const pColor = DATA.class_colors[DATA.class_keys[ci]];
        card.innerHTML = `
          <div class="class-name" style="color:${{pColor}}">${{DATA.class_names[ci]}}</div>
          <div class="prediction">${{pct(p)}}</div>
          <div class="ground-truth">${{gt != null ? "GT: " + pct(gt) : "GT: \u2014"}}</div>
          <div class="delta">${{gt != null ? "Delta: " + dpct(p, gt) : ""}}</div>
        `;
        cellGridEl.appendChild(card);
      }}
    }}

    function renderAll() {{
      renderSummary();
      renderLayers();
      renderCanvases();
      renderCell();
    }}

    // Init
    DATA.rounds.forEach((r, i) => {{
      const o = document.createElement("option");
      o.value = String(i);
      const sc = r.score == null ? "" : r.score >= 85 ? "+" : r.score < 70 ? "!" : "";
      const scoreLabel = r.score != null ? r.score.toFixed(1) : "\u2014";
      o.textContent = `${{r.label}} [${{r.mode}} / ${{r.world_type}}] ${{scoreLabel}} ${{sc}}`;
      roundSelect.appendChild(o);
    }});
    roundSelect.value = String(S.roundIndex);

    function populateSeeds() {{
      seedSelect.innerHTML = "";
      getRound().seeds.forEach((s, i) => {{
        const o = document.createElement("option");
        o.value = String(i);
        o.textContent = `${{s.label}}${{s.score != null ? " ("+s.score.toFixed(1)+")" : ""}} q=${{s.query_count ?? 0}}`;
        seedSelect.appendChild(o);
      }});
      if (S.seedIndex >= getRound().seeds.length) S.seedIndex = 0;
      seedSelect.value = String(S.seedIndex);
    }}

    roundSelect.addEventListener("change", () => {{
      S.roundIndex = parseInt(roundSelect.value);
      S.seedIndex = 0;
      S.cell = {{ x: 20, y: 20 }};
      populateSeeds();
      renderAll();
    }});
    seedSelect.addEventListener("change", () => {{
      S.seedIndex = parseInt(seedSelect.value);
      S.cell = {{ x: 20, y: 20 }};
      renderAll();
    }});

    populateSeeds();
    renderAll();
  </script>
</body>
</html>'''


if __name__ == "__main__":
    main()
