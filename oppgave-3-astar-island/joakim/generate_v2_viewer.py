"""
Generate an interactive HTML viewer showing solution_v2 predictions vs ground truth
for all rounds with history data. Uses the same visualization format as ola_v8_viewer.html.
"""
import json, numpy as np, math, sys, time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
import solution_v2 as sv2

MAP_W, MAP_H = 40, 40
NUM_CLASSES = 6
FLOOR = 0.001

history_root = Path(__file__).parent / "history"
project_root = Path(__file__).parent

# Load tables (solution_v2 will use its configured paths)
opt_tables = sv2.load_opt_tables()
model_tables = sv2.load_model_tables()


def classify_oracle_world_type(initial_grids, gt_arrays):
    """Classify world type from GT (oracle)."""
    total_initial = 0
    total_survived = 0.0
    for si in range(len(initial_grids)):
        grid = initial_grids[si]
        gt = gt_arrays[si]
        for y in range(MAP_H):
            for x in range(MAP_W):
                if grid[y, x] in (1, 2):
                    total_initial += 1
                    total_survived += gt[y, x, 1] + gt[y, x, 2]
    if total_initial == 0:
        return "STABLE", 0.5
    sr = total_survived / total_initial
    if sr < 0.15:
        return "DEAD", sr
    elif sr > 0.40:
        return "BOOM", sr
    else:
        return "STABLE", sr


def compute_kl_map(gt, pred, grid):
    """Compute per-cell KL divergence."""
    eps = 1e-12
    gt_c = np.clip(gt, eps, 1.0)
    pred_c = np.clip(pred, eps, 1.0)
    kl = np.sum(gt_c * np.log(gt_c / pred_c), axis=-1)
    # Zero out static cells
    for y in range(MAP_H):
        for x in range(MAP_W):
            if grid[y, x] in (5, 10):
                kl[y, x] = 0.0
    return kl


def score_from_wkl(wkl):
    return max(0.0, min(100.0, 100.0 * math.exp(-3.0 * wkl)))


print("=== Generating V2 Viewer Data ===\n")

rounds_data = []

for hdir in sorted(history_root.iterdir(), key=lambda d: d.name):
    if not hdir.is_dir():
        continue
    manifest_path = hdir / "manifest.json"
    if not manifest_path.exists():
        continue

    manifest = json.loads(manifest_path.read_text())
    rmeta = manifest.get("round_metadata", {})
    rnum = rmeta.get("round_number", "?")
    rid = manifest.get("round_id", hdir.name)
    weight = rmeta.get("round_weight", 1.0) or 1.0

    # Load initial states
    init_states = manifest.get("initial_states", [])
    seeds_data = []
    initial_grids = []
    gt_arrays = {}

    for ist in init_states:
        si = ist["seed_index"]
        grid_path = hdir / ist["grid_path"]
        gt_path = hdir / "arrays" / f"seed_{si}_ground_truth.npy"
        if not grid_path.exists():
            break
        grid = np.load(grid_path).astype(int)
        initial_grids.append(grid)
        seeds_data.append({
            "grid": grid.tolist(),
            "settlements": ist.get("settlements", []),
            "seed_index": si,
        })
        if gt_path.exists():
            gt_arrays[si] = np.load(gt_path).astype(float)

    if len(seeds_data) != len(init_states) or not seeds_data:
        continue

    n_seeds = len(seeds_data)
    has_gt = len(gt_arrays) == n_seeds

    # Build model
    model = sv2.RoundModel(seeds_data)

    # Load stored observations
    queries = manifest.get("queries", [])
    has_queries = False
    query_counts_per_seed = defaultdict(int)
    if queries:
        try:
            for q in queries:
                si_q = q["seed_index"]
                vp = q["viewport"]
                gpath = hdir / q["grid_path"]
                if gpath.exists():
                    obs_grid = np.load(gpath).tolist()
                    model.add_observation(si_q, obs_grid, vp["x"], vp["y"])
                    query_counts_per_seed[si_q] += 1
            has_queries = True
        except Exception as e:
            print(f"  R{rnum}: query load error ({e})")
            model = sv2.RoundModel(seeds_data)

    mode = "history_replay" if has_queries else "prior_only"
    if has_queries:
        type_weights = model.compute_type_weights()
    else:
        type_weights = {"DEAD": 0.0, "STABLE": 0.50, "BOOM": 0.50}

    vitality = model.get_vitality() if has_queries else None

    # Determine world type
    dominant = max(type_weights, key=type_weights.get)
    world_type = dominant

    # Oracle world type from GT
    oracle_wt = "?"
    oracle_sr = 0
    if has_gt:
        oracle_wt, oracle_sr = classify_oracle_world_type(initial_grids, list(gt_arrays.values()))

    # Build predictions for each seed
    round_seeds = []
    seed_scores = []

    for si in range(n_seeds):
        if has_queries:
            pred = model.build_prediction(si, opt_tables, model_tables, type_weights)
        else:
            pred = model.build_prior_prediction(si, opt_tables, model_tables, type_weights)

        # Dynamic cell counts
        static_mask = model.static_masks[si]
        n_dynamic = int((~static_mask).sum())
        n_observed = sum(1 for y in range(MAP_H) for x in range(MAP_W)
                         if not static_mask[y, x] and model.cell_total.get((si, y, x), 0) > 0)

        # Compute score and KL map
        kl_map = None
        score = None
        if has_gt:
            gt = gt_arrays[si]
            kl_map = compute_kl_map(gt, pred, initial_grids[si]).tolist()
            wkl = sv2.weighted_kl(gt, pred, seeds_data[si]["grid"])
            score = score_from_wkl(wkl)
            seed_scores.append(score)

        seed_entry = {
            "seed_index": si,
            "label": f"Seed {si}",
            "initial_grid": seeds_data[si]["grid"],
            "prediction": pred.tolist(),
            "ground_truth": gt_arrays[si].tolist() if has_gt else None,
            "kl_map": kl_map,
            "score": round(score, 2) if score is not None else None,
            "query_count": query_counts_per_seed.get(si, 0),
            "dynamic_observed_cells": n_observed,
            "dynamic_total_cells": n_dynamic,
        }
        round_seeds.append(seed_entry)

    round_score = round(sum(seed_scores) / len(seed_scores), 2) if seed_scores else None

    round_entry = {
        "label": f"Round {rnum}",
        "round_number": rnum if isinstance(rnum, int) else 0,
        "round_id": rid,
        "weight": weight,
        "mode": mode,
        "world_type": world_type,
        "oracle_world_type": oracle_wt,
        "query_count": len(queries),
        "vitality": round(vitality, 4) if vitality is not None else None,
        "score": round_score,
        "seeds": round_seeds,
    }
    rounds_data.append(round_entry)

    sc_str = f"{round_score:.1f}" if round_score else "—"
    print(f"  R{rnum} ({mode}, {world_type}): score={sc_str}, seeds={n_seeds}")

# Sort by round number
rounds_data.sort(key=lambda r: r["round_number"])

# Build full data object
viewer_data = {
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "solver": "Joakim v2 — Pooled Empirical Model (22r tables)",
    "alpha": 100.0,
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
    "rounds": rounds_data,
}

data_json = json.dumps(viewer_data, separators=(',', ':'))

# HTML template (same structure as ola_v8_viewer.html)
html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Joakim v2 — Prediction vs Ground Truth</title>
  <style>
    :root {
      --bg: #0d1830;
      --panel: #132240;
      --border: rgba(140, 170, 255, 0.18);
      --text: #ecf2ff;
      --muted: #9ca9c9;
      --accent: #8bb3ff;
      --shadow: rgba(0, 0, 0, 0.32);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: "SFMono-Regular", "Menlo", "Monaco", "Consolas", monospace;
      background:
        radial-gradient(circle at top left, rgba(88, 122, 220, 0.18), transparent 34%),
        linear-gradient(180deg, #102041 0%, #0a1428 100%);
      color: var(--text);
      min-height: 100vh;
    }
    .app {
      width: min(1500px, calc(100vw - 32px));
      margin: 16px auto;
      padding: 20px;
      background: rgba(11, 22, 45, 0.82);
      border: 1px solid var(--border);
      border-radius: 24px;
      box-shadow: 0 18px 60px var(--shadow);
      backdrop-filter: blur(10px);
    }
    .header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 16px;
      flex-wrap: wrap;
    }
    .title h1 { margin: 0 0 6px; font-size: 28px; letter-spacing: -0.03em; }
    .title p { color: var(--muted); max-width: 760px; line-height: 1.5; font-size: 13px; }
    .controls {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }
    label {
      display: flex;
      flex-direction: column;
      gap: 6px;
      color: var(--muted);
      font-size: 13px;
    }
    select {
      min-width: 200px;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: #12213e;
      color: var(--text);
      font: inherit;
    }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin: 18px 0 22px;
    }
    .summary-card {
      background: rgba(20, 34, 64, 0.88);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 14px 16px;
    }
    .summary-card .label {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }
    .summary-card .value {
      font-size: 22px;
      font-weight: 700;
      letter-spacing: -0.03em;
    }
    .summary-card .subvalue {
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
    }

    .overview-grid {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 16px;
      margin-bottom: 24px;
    }
    .overview-panel {
      background: rgba(19, 33, 62, 0.92);
      border: 1px solid var(--border);
      border-radius: 22px;
      padding: 14px;
      text-align: center;
    }
    .overview-panel h3 {
      font-size: 14px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 10px;
    }
    canvas {
      width: 100%;
      aspect-ratio: 1 / 1;
      display: block;
      border-radius: 14px;
      background: #0b1018;
      image-rendering: pixelated;
      cursor: crosshair;
    }

    .layer-header {
      display: grid;
      grid-template-columns: 120px 1fr 1fr 1fr;
      gap: 14px;
      align-items: end;
      margin-bottom: 10px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .layer-grid {
      display: grid;
      gap: 16px;
    }
    .layer-row {
      display: grid;
      grid-template-columns: 120px 1fr 1fr 1fr;
      gap: 14px;
      align-items: center;
    }
    .layer-label {
      font-size: 16px;
      font-weight: 700;
    }
    .layer-cell {
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid var(--border);
    }
    .layer-cell.dimmed { opacity: 0.3; }

    .cell-panel {
      margin-top: 24px;
      background: rgba(19, 33, 62, 0.72);
      border: 1px solid var(--border);
      border-radius: 22px;
      padding: 18px;
    }
    .cell-title {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      margin-bottom: 12px;
    }
    .cell-title h2 { font-size: 20px; }
    .badge {
      display: inline-block;
      padding: 4px 12px;
      font-size: 12px;
      border-radius: 10px;
      background: rgba(20, 34, 64, 0.8);
      border: 1px solid var(--border);
    }
    .cell-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 10px;
    }
    .cell-card {
      background: rgba(16, 26, 50, 0.9);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px;
    }
    .cell-card .class-name {
      font-weight: 700;
      font-size: 14px;
      margin-bottom: 6px;
    }
    .cell-card .prediction { font-size: 18px; font-weight: 700; }
    .cell-card .ground-truth { color: var(--muted); font-size: 13px; margin-top: 4px; }
    .cell-card .delta { font-size: 12px; margin-top: 2px; }

    .mode-history_replay { color: #34d399; }
    .mode-prior_only { color: #fbbf24; }
    .type-DEAD { color: #f87171; }
    .type-STABLE { color: #fbbf24; }
    .type-BOOM { color: #34d399; }
    .meta { display: flex; flex-wrap: wrap; gap: 6px; }
  </style>
</head>
<body>
  <div class="app">
    <div class="header">
      <div class="title">
        <h1>Joakim v2 — Prediction vs Ground Truth</h1>
        <p>Pooled Empirical Model with 22r calibration tables. Shows prediction, ground truth, and KL divergence for each round and seed.</p>
      </div>
      <div class="controls">
        <label>Round
          <select id="round-select"></select>
        </label>
        <label>Seed
          <select id="seed-select"></select>
        </label>
      </div>
    </div>

    <div id="summary-grid" class="summary-grid"></div>

    <div class="overview-grid">
      <div class="overview-panel">
        <h3>Prediction (argmax)</h3>
        <canvas id="pred-overview" width="40" height="40"></canvas>
      </div>
      <div class="overview-panel">
        <h3>Ground Truth (argmax)</h3>
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

  <script id="history-data" type="application/json">DATA_PLACEHOLDER</script>

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

    const S = { roundIndex: Math.max(DATA.rounds.length - 1, 0), seedIndex: 0, cell: { x: 20, y: 20 } };
    const G = DATA.grid_size;

    function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
    function parseHex(h) {
      const c = h.replace("#", "");
      return [parseInt(c.slice(0,2),16), parseInt(c.slice(2,4),16), parseInt(c.slice(4,6),16)];
    }
    function mix(a, b, t) {
      t = clamp(t, 0, 1);
      return [Math.round(a[0]+(b[0]-a[0])*t), Math.round(a[1]+(b[1]-a[1])*t), Math.round(a[2]+(b[2]-a[2])*t)];
    }
    function pct(v) { return (v*100).toFixed(1)+"%"; }
    function dpct(p, t) {
      const d = (p - t) * 100;
      return (d >= 0 ? "+" : "") + d.toFixed(1) + " pp";
    }

    function getRound() { return DATA.rounds[S.roundIndex]; }
    function getSeed() { return getRound().seeds[S.seedIndex]; }

    function drawHighlight(ctx) {
      ctx.save();
      ctx.fillStyle = "rgba(255,255,255,0.3)";
      ctx.fillRect(S.cell.x, S.cell.y, 1, 1);
      ctx.restore();
    }

    function drawOverviewCanvas(canvas, grid, tensor) {
      const ctx = canvas.getContext("2d");
      const img = ctx.createImageData(G, G);
      const tc = DATA.terrain_base_colors;
      for (let y = 0; y < G; y++) {
        for (let x = 0; x < G; x++) {
          const base = parseHex(tc[String(grid[y][x])] || "#111827");
          let px = base;
          if (tensor) {
            let best = 0, bestV = tensor[y][x][0];
            for (let c = 1; c < 6; c++) {
              if (tensor[y][x][c] > bestV) { bestV = tensor[y][x][c]; best = c; }
            }
            const accent = parseHex(DATA.class_colors[DATA.class_keys[best]]);
            px = mix(base, accent, Math.pow(clamp(bestV, 0, 1), 0.92));
          }
          const i = (y * G + x) * 4;
          img.data[i] = px[0]; img.data[i+1] = px[1]; img.data[i+2] = px[2]; img.data[i+3] = 255;
        }
      }
      ctx.putImageData(img, 0, 0);
      drawHighlight(ctx);
    }

    function drawKLOverview(canvas, klMap) {
      const ctx = canvas.getContext("2d");
      const img = ctx.createImageData(G, G);
      const bg = [11, 16, 24];
      const maxKL = 1.5;
      for (let y = 0; y < G; y++) {
        for (let x = 0; x < G; x++) {
          const kl = klMap ? clamp(klMap[y][x] / maxKL, 0, 1) : 0;
          const t = Math.pow(kl, 0.6);
          const r = Math.round(bg[0] + (248-bg[0])*t);
          const g = Math.round(bg[1] + (113-bg[1])*t*0.4);
          const b = Math.round(bg[2] + (113-bg[2])*t*0.3);
          const i = (y*G+x)*4;
          img.data[i] = r; img.data[i+1] = g; img.data[i+2] = b; img.data[i+3] = 255;
        }
      }
      ctx.putImageData(img, 0, 0);
      drawHighlight(ctx);
    }

    function drawLayer(canvas, matrix, ci) {
      const ctx = canvas.getContext("2d");
      const img = ctx.createImageData(G, G);
      const accent = parseHex(DATA.class_colors[DATA.class_keys[ci]]);
      const bg = parseHex("#0b1018");
      for (let y = 0; y < G; y++) {
        for (let x = 0; x < G; x++) {
          const t = matrix ? Math.pow(clamp(matrix[y][x], 0, 1), 0.88) : 0;
          const px = mix(bg, accent, t);
          const i = (y*G+x)*4;
          img.data[i] = px[0]; img.data[i+1] = px[1]; img.data[i+2] = px[2]; img.data[i+3] = matrix ? 255 : 96;
        }
      }
      ctx.putImageData(img, 0, 0);
      drawHighlight(ctx);
    }

    function drawKLLayer(canvas, klMap) {
      const ctx = canvas.getContext("2d");
      const img = ctx.createImageData(G, G);
      const maxKL = 1.5;
      for (let y = 0; y < G; y++) {
        for (let x = 0; x < G; x++) {
          const kl = klMap ? clamp(klMap[y][x] / maxKL, 0, 1) : 0;
          const t = Math.pow(kl, 0.6);
          const r = Math.round(11 + (248-11)*t);
          const g = Math.round(16 + (113-16)*t*0.4);
          const b = Math.round(24 + (113-24)*t*0.3);
          const i = (y*G+x)*4;
          img.data[i] = r; img.data[i+1] = g; img.data[i+2] = b; img.data[i+3] = 255;
        }
      }
      ctx.putImageData(img, 0, 0);
      drawHighlight(ctx);
    }

    function bindCanvas(c) {
      if (c.dataset.bound) return;
      c.dataset.bound = "1";
      function handler(e) {
        const r = c.getBoundingClientRect();
        S.cell.x = clamp(Math.floor((e.clientX-r.left)/r.width*G), 0, G-1);
        S.cell.y = clamp(Math.floor((e.clientY-r.top)/r.height*G), 0, G-1);
        renderCanvases();
        renderCell();
      }
      c.addEventListener("mousemove", handler);
      c.addEventListener("click", handler);
    }

    function renderSummary() {
      const rd = getRound();
      const sd = getSeed();
      const hasGT = !!sd.ground_truth;
      const scoreLabel = sd.score != null ? sd.score.toFixed(1) : "N/A";
      const scoreColor = sd.score == null ? "#9ca9c9" : sd.score >= 85 ? "#34d399" : sd.score >= 70 ? "#fbbf24" : "#f87171";
      const modeLabel = rd.mode || "prior_only";
      const roundScoreLabel = rd.score != null ? rd.score.toFixed(1) : "N/A";
      const vitalityLabel = rd.vitality != null ? rd.vitality.toFixed(3) : "N/A";
      const oracleLabel = rd.oracle_world_type || "N/A";
      const allScores = DATA.rounds.map(r => r.score).filter(s => s != null);
      const avgScore = allScores.length ? (allScores.reduce((a, b) => a + b, 0) / allScores.length) : 0;
      summaryGrid.innerHTML = [
        `<div class="summary-card"><div class="label">Round</div><div class="value">${rd.label}</div><div class="subvalue">Weight: ${rd.weight.toFixed(2)}</div></div>`,
        `<div class="summary-card"><div class="label">Mode</div><div class="value mode-${modeLabel}">${modeLabel}</div><div class="subvalue">Queries: ${rd.query_count ?? 0}</div></div>`,
        `<div class="summary-card"><div class="label">World Type</div><div class="value type-${rd.world_type}">${rd.world_type}</div><div class="subvalue">Round: ${roundScoreLabel} | Oracle: ${oracleLabel}</div></div>`,
        `<div class="summary-card"><div class="label">Vitality</div><div class="value">${vitalityLabel}</div><div class="subvalue">Alpha: ${DATA.alpha.toFixed(0)}</div></div>`,
        `<div class="summary-card"><div class="label">Seed Score</div><div class="value" style="color:${scoreColor}">${scoreLabel}</div><div class="subvalue">${sd.label} | ${sd.dynamic_observed_cells}/${sd.dynamic_total_cells} obs</div></div>`,
        `<div class="summary-card"><div class="label">Seed Queries</div><div class="value">${sd.query_count ?? 0}</div><div class="subvalue">${sd.label}</div></div>`,
        `<div class="summary-card"><div class="label">Overall Avg</div><div class="value" style="color:${avgScore >= 85 ? "#34d399" : avgScore >= 70 ? "#fbbf24" : "#f87171"}">${avgScore.toFixed(1)}</div><div class="subvalue">${allScores.length} scored rounds</div></div>`,
        `<div class="summary-card"><div class="label">Ground Truth</div><div class="value">${hasGT ? "Yes" : "No"}</div><div class="subvalue">${hasGT ? "Available" : "N/A"}</div></div>`,
      ].join("");
    }

    function renderLayers() {
      const sd = getSeed();
      const hasGT = !!sd.ground_truth;
      layerGrid.innerHTML = "";
      for (let ci = 0; ci < 6; ci++) {
        const row = document.createElement("div");
        row.className = "layer-row";
        row.innerHTML = `
          <div class="layer-label" style="color:${DATA.class_colors[DATA.class_keys[ci]]}">${DATA.class_names[ci]}</div>
          <div class="layer-cell"><canvas id="pred-${ci}" width="40" height="40"></canvas></div>
          <div class="layer-cell ${hasGT ? "" : "dimmed"}"><canvas id="gt-${ci}" width="40" height="40"></canvas></div>
          <div class="layer-cell ${sd.kl_map ? "" : "dimmed"}"><canvas id="kl-${ci}" width="40" height="40"></canvas></div>
        `;
        layerGrid.appendChild(row);
      }
      for (const c of document.querySelectorAll("canvas")) bindCanvas(c);
    }

    function renderCanvases() {
      const sd = getSeed();
      const hasGT = !!sd.ground_truth;
      drawOverviewCanvas(predOverview, sd.initial_grid, sd.prediction);
      drawOverviewCanvas(gtOverview, sd.initial_grid, hasGT ? sd.ground_truth : null);
      drawKLOverview(klOverview, sd.kl_map || null);

      for (let ci = 0; ci < 6; ci++) {
        const predC = document.getElementById(`pred-${ci}`);
        const gtC = document.getElementById(`gt-${ci}`);
        const klC = document.getElementById(`kl-${ci}`);
        if (predC) drawLayer(predC, sd.prediction.map(r => r.map(c => c[ci])), ci);
        if (gtC) drawLayer(gtC, hasGT ? sd.ground_truth.map(r => r.map(c => c[ci])) : null, ci);
        if (klC) drawKLLayer(klC, sd.kl_map || null);
      }
    }

    function renderCell() {
      const sd = getSeed();
      const {x, y} = S.cell;
      const terrain = sd.initial_grid[y][x];
      const tLabel = DATA.terrain_labels[String(terrain)] || `Code ${terrain}`;
      const hasGT = !!sd.ground_truth;
      const cellKL = sd.kl_map ? sd.kl_map[y][x] : null;
      const isStatic = terrain === 10 || terrain === 5;

      cellTitleEl.textContent = `Cell (${x}, ${y})`;
      let badges = `<span class="badge"><strong>Terrain</strong> ${tLabel}</span>`;
      if (isStatic) badges += ` <span class="badge" style="color:#666"><strong>Static</strong> (not scored)</span>`;
      if (sd.dynamic_observed_cells != null && sd.dynamic_total_cells != null)
        badges += ` <span class="badge"><strong>Observed</strong> ${sd.dynamic_observed_cells}/${sd.dynamic_total_cells}</span>`;
      if (cellKL != null) {
        const klColor = cellKL < 0.1 ? "#34d399" : cellKL < 0.5 ? "#fbbf24" : "#f87171";
        badges += ` <span class="badge"><strong>KL</strong> <span style="color:${klColor}">${cellKL.toFixed(4)}</span></span>`;
      }
      cellMetaEl.innerHTML = badges;

      cellGridEl.innerHTML = "";
      for (let ci = 0; ci < 6; ci++) {
        const p = sd.prediction[y][x][ci];
        const gt = hasGT ? sd.ground_truth[y][x][ci] : null;
        const card = document.createElement("div");
        card.className = "cell-card";
        const pColor = DATA.class_colors[DATA.class_keys[ci]];
        card.innerHTML = `
          <div class="class-name" style="color:${pColor}">${DATA.class_names[ci]}</div>
          <div class="prediction">${pct(p)}</div>
          <div class="ground-truth">${gt != null ? "GT: " + pct(gt) : "GT: N/A"}</div>
          <div class="delta">${gt != null ? "Delta: " + dpct(p, gt) : ""}</div>
        `;
        cellGridEl.appendChild(card);
      }
    }

    function renderAll() {
      renderSummary();
      renderLayers();
      renderCanvases();
      renderCell();
    }

    // Init
    DATA.rounds.forEach((r, i) => {
      const o = document.createElement("option");
      o.value = String(i);
      const sc = r.score == null ? "" : r.score >= 85 ? "+" : r.score < 70 ? "!" : "";
      const scoreLabel = r.score != null ? r.score.toFixed(1) : "N/A";
      o.textContent = `${r.label} [${r.mode} / ${r.world_type}] ${scoreLabel} ${sc}`;
      roundSelect.appendChild(o);
    });
    roundSelect.value = String(S.roundIndex);

    function populateSeeds() {
      seedSelect.innerHTML = "";
      getRound().seeds.forEach((s, i) => {
        const o = document.createElement("option");
        o.value = String(i);
        o.textContent = `${s.label}${s.score != null ? " ("+s.score.toFixed(1)+")" : ""} q=${s.query_count ?? 0}`;
        seedSelect.appendChild(o);
      });
      if (S.seedIndex >= getRound().seeds.length) S.seedIndex = 0;
      seedSelect.value = String(S.seedIndex);
    }

    roundSelect.addEventListener("change", () => {
      S.roundIndex = parseInt(roundSelect.value);
      S.seedIndex = 0;
      S.cell = { x: 20, y: 20 };
      populateSeeds();
      renderAll();
    });
    seedSelect.addEventListener("change", () => {
      S.seedIndex = parseInt(seedSelect.value);
      S.cell = { x: 20, y: 20 };
      renderAll();
    });

    populateSeeds();
    renderAll();
  </script>
</body>
</html>'''

# Replace placeholder with data
html_output = html_template.replace('DATA_PLACEHOLDER', data_json)

out_path = project_root / "v2_viewer.html"
out_path.write_text(html_output)
print(f"\nViewer saved: {out_path}")
print(f"File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")
print(f"Rounds: {len(rounds_data)}")
