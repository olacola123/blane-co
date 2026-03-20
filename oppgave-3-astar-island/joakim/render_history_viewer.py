#!/usr/bin/env python3
"""Build a local HTML viewer for stored prediction rounds."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from astar_solver.constants import (
    GRID_SIZE,
    INTERNAL_EMPTY,
    INTERNAL_FOREST,
    INTERNAL_MOUNTAIN,
    INTERNAL_OCEAN,
    INTERNAL_PLAINS,
    INTERNAL_PORT,
    INTERNAL_RUIN,
    INTERNAL_SETTLEMENT,
)

PROJECT_ROOT = Path(__file__).resolve().parent

CLASS_NAMES = ("Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain")
CLASS_KEYS = ("empty", "settlement", "port", "ruin", "forest", "mountain")
CLASS_COLORS = {
    "empty": "#e7d7a7",
    "settlement": "#f59e0b",
    "port": "#22d3ee",
    "ruin": "#ef4444",
    "forest": "#5bbf59",
    "mountain": "#6366f1",
}
TERRAIN_LABELS = {
    INTERNAL_EMPTY: "Empty",
    INTERNAL_OCEAN: "Ocean",
    INTERNAL_PLAINS: "Plains",
    INTERNAL_SETTLEMENT: "Settlement",
    INTERNAL_PORT: "Port",
    INTERNAL_RUIN: "Ruin",
    INTERNAL_FOREST: "Forest",
    INTERNAL_MOUNTAIN: "Mountain",
}
TERRAIN_BASE_COLORS = {
    INTERNAL_EMPTY: "#16181d",
    INTERNAL_OCEAN: "#0c2542",
    INTERNAL_PLAINS: "#16181d",
    INTERNAL_SETTLEMENT: "#332711",
    INTERNAL_PORT: "#113345",
    INTERNAL_RUIN: "#341617",
    INTERNAL_FOREST: "#142317",
    INTERNAL_MOUNTAIN: "#211d3f",
}
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Astar Island History Viewer</title>
  <style>
    :root {
      --bg: #0d1830;
      --panel: #132240;
      --panel-soft: #17284a;
      --border: rgba(140, 170, 255, 0.18);
      --text: #ecf2ff;
      --muted: #9ca9c9;
      --accent: #8bb3ff;
      --shadow: rgba(0, 0, 0, 0.32);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
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
    }
    .title h1 {
      margin: 0 0 8px;
      font-size: 30px;
      letter-spacing: -0.03em;
    }
    .title p {
      margin: 0;
      color: var(--muted);
      max-width: 760px;
      line-height: 1.5;
    }
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
      min-width: 220px;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: #12213e;
      color: var(--text);
      font: inherit;
    }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
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
      font-size: 12px;
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
      font-size: 13px;
    }
    .overview-grid {
      display: grid;
      grid-template-columns: minmax(280px, 320px) 1fr;
      gap: 18px;
      margin-bottom: 24px;
    }
    .sidebar,
    .main-panel,
    .cell-panel {
      background: rgba(19, 33, 62, 0.92);
      border: 1px solid var(--border);
      border-radius: 22px;
      padding: 18px;
    }
    .section-title {
      font-size: 15px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin: 0 0 12px;
    }
    .meta-list {
      display: grid;
      gap: 10px;
    }
    .meta-row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding-bottom: 10px;
      border-bottom: 1px solid rgba(140, 170, 255, 0.08);
    }
    .meta-row:last-child { border-bottom: 0; padding-bottom: 0; }
    .meta-row .key {
      color: var(--muted);
      font-size: 13px;
    }
    .meta-row .value {
      text-align: right;
      font-size: 14px;
    }
    .overview-panels {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
    }
    .mini-panel {
      background: rgba(14, 24, 46, 0.82);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 14px;
    }
    .mini-panel h3 {
      margin: 0 0 10px;
      font-size: 18px;
      font-weight: 600;
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
      grid-template-columns: 140px 1fr 1fr 1fr;
      gap: 14px;
      align-items: end;
      margin-bottom: 10px;
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .layer-grid {
      display: grid;
      gap: 18px;
    }
    .layer-row {
      display: grid;
      grid-template-columns: 140px 1fr 1fr 1fr;
      gap: 14px;
      align-items: center;
    }
    .layer-label {
      font-size: 20px;
      letter-spacing: -0.03em;
    }
    .layer-cell {
      background: rgba(11, 20, 38, 0.72);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 12px;
    }
    .cell-panel {
      margin-top: 24px;
    }
    .cell-title {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: baseline;
      margin-bottom: 18px;
      padding-top: 8px;
      border-top: 2px solid rgba(230, 235, 255, 0.8);
    }
    .cell-title h2 {
      margin: 0;
      font-size: 20px;
      letter-spacing: -0.03em;
    }
    .cell-title .meta {
      color: var(--muted);
      font-size: 13px;
    }
    .cell-grid {
      display: grid;
      grid-template-columns: repeat(6, minmax(110px, 1fr));
      gap: 12px;
    }
    .cell-card {
      background: rgba(14, 23, 42, 0.84);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 14px 12px;
      min-height: 116px;
    }
    .cell-card .class-name {
      color: var(--muted);
      margin-bottom: 10px;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .cell-card .prediction {
      font-size: 30px;
      font-weight: 700;
      letter-spacing: -0.04em;
      line-height: 1;
      margin-bottom: 8px;
    }
    .cell-card .ground-truth,
    .cell-card .delta {
      font-size: 13px;
      color: var(--muted);
      line-height: 1.45;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      background: rgba(20, 34, 64, 0.88);
      border: 1px solid var(--border);
      border-radius: 999px;
      font-size: 12px;
      color: var(--muted);
    }
    .badge strong { color: var(--text); }
    .dimmed {
      opacity: 0.42;
      filter: grayscale(0.4);
    }
    @media (max-width: 1100px) {
      .overview-grid { grid-template-columns: 1fr; }
      .layer-header,
      .layer-row { grid-template-columns: 1fr; }
      .cell-grid { grid-template-columns: repeat(2, minmax(110px, 1fr)); }
    }
    @media (max-width: 700px) {
      .app { width: calc(100vw - 16px); margin: 8px auto; padding: 14px; }
      .header { flex-direction: column; }
      .controls, label, select { width: 100%; }
      .cell-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="app">
    <div class="header">
      <div class="title">
        <h1>Astar Island History Viewer</h1>
        <p>
          Local viewer for stored predictions, ground truth, and per-cell comparisons. Open this file in VS Code or a browser,
          switch between rounds and seeds, and inspect the exact probabilities your solver produced.
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

    <div class="summary-grid" id="summary-grid"></div>

    <div class="overview-grid">
      <aside class="sidebar">
        <h2 class="section-title">Round Details</h2>
        <div class="meta-list" id="round-meta"></div>
      </aside>

      <section class="main-panel">
        <h2 class="section-title">Map Overview</h2>
        <div class="overview-panels">
          <div class="mini-panel">
            <h3>V1 Prediction</h3>
            <canvas id="prediction-overview" width="40" height="40"></canvas>
          </div>
          <div class="mini-panel">
            <h3>V2 Prediction</h3>
            <canvas id="prediction-v2-overview" width="40" height="40"></canvas>
          </div>
          <div class="mini-panel">
            <h3>Ground Truth</h3>
            <canvas id="ground-truth-overview" width="40" height="40"></canvas>
          </div>
        </div>
      </section>
    </div>

    <section class="main-panel">
      <div class="layer-header">
        <div>Layer</div>
        <div>V1 Prediction</div>
        <div>V2 Prediction</div>
        <div>Ground Truth</div>
      </div>
      <div class="layer-grid" id="layer-grid"></div>
    </section>

    <section class="cell-panel">
      <div class="cell-title">
        <h2 id="cell-title">Cell (0, 0)</h2>
        <div class="meta" id="cell-meta"></div>
      </div>
      <div class="cell-grid" id="cell-grid"></div>
    </section>
  </div>

  <script id="history-data" type="application/json">__DATA_JSON__</script>
  <script>
    const DATA = JSON.parse(document.getElementById("history-data").textContent);
    const roundSelect = document.getElementById("round-select");
    const seedSelect = document.getElementById("seed-select");
    const summaryGrid = document.getElementById("summary-grid");
    const roundMeta = document.getElementById("round-meta");
    const layerGrid = document.getElementById("layer-grid");
    const cellGrid = document.getElementById("cell-grid");
    const cellTitle = document.getElementById("cell-title");
    const cellMeta = document.getElementById("cell-meta");
    const predictionOverview = document.getElementById("prediction-overview");
    const predictionV2Overview = document.getElementById("prediction-v2-overview");
    const groundTruthOverview = document.getElementById("ground-truth-overview");

    const state = {
      roundIndex: 0,
      seedIndex: 0,
      selectedCell: { x: 20, y: 20 },
    };

    function clamp(value, low, high) {
      return Math.max(low, Math.min(high, value));
    }

    function parseHexColor(hex) {
      const clean = hex.replace("#", "");
      return [
        parseInt(clean.slice(0, 2), 16),
        parseInt(clean.slice(2, 4), 16),
        parseInt(clean.slice(4, 6), 16),
      ];
    }

    function mixColor(base, top, amount) {
      const alpha = clamp(amount, 0, 1);
      return [
        Math.round(base[0] + (top[0] - base[0]) * alpha),
        Math.round(base[1] + (top[1] - base[1]) * alpha),
        Math.round(base[2] + (top[2] - base[2]) * alpha),
      ];
    }

    function percent(value) {
      return `${(value * 100).toFixed(1)}%`;
    }

    function deltaPercent(prediction, truth) {
      const delta = (prediction - truth) * 100;
      const prefix = delta >= 0 ? "+" : "";
      return `${prefix}${delta.toFixed(1)} pp`;
    }

    function getCurrentRound() {
      return DATA.rounds[state.roundIndex];
    }

    function getCurrentSeed() {
      return getCurrentRound().seeds[state.seedIndex];
    }

    function createSummaryCard(label, value, subvalue = "") {
      const card = document.createElement("div");
      card.className = "summary-card";
      card.innerHTML = `
        <div class="label">${label}</div>
        <div class="value">${value}</div>
        <div class="subvalue">${subvalue}</div>
      `;
      return card;
    }

    function renderSummary() {
      const round = getCurrentRound();
      const seed = getCurrentSeed();
      const v1kl = seed.v1_kl_mean;
      const v2kl = seed.v2_kl_mean;
      const hasV2 = seed.prediction_v2 != null;

      let klLabel = "—";
      let klSub = "No ground truth";
      if (v1kl != null) {
        klLabel = v1kl.toFixed(4);
        klSub = `Score: ${Math.max(0, 100 * (1 - v1kl / 2.5)).toFixed(1)}%`;
      }

      let v2Label = "—";
      let v2Sub = hasV2 ? "No ground truth" : "Run backtest.py first";
      if (v2kl != null) {
        v2Label = v2kl.toFixed(4);
        const delta = v2kl - v1kl;
        const sign = delta <= 0 ? "" : "+";
        const color = delta <= 0 ? "#4ade80" : "#f87171";
        v2Sub = `Score: ${Math.max(0, 100 * (1 - v2kl / 2.5)).toFixed(1)}% &nbsp; <span style="color:${color}">${sign}${delta.toFixed(4)}</span>`;
      }

      summaryGrid.replaceChildren(
        createSummaryCard("Round", round.label, round.round_id),
        createSummaryCard("Seed", seed.label, `${seed.query_count} queries, ${seed.coverage_ratio_label} coverage`),
        createSummaryCard("V1 KL", klLabel, klSub),
        createSummaryCard("V2 KL", v2Label, v2Sub),
      );
    }

    function renderRoundMeta() {
      const round = getCurrentRound();
      const seed = getCurrentSeed();
      const rows = [
        ["Round ID", round.round_id],
        ["Status", round.status || "Unknown"],
        ["Event Date", round.event_date || "—"],
        ["Started", round.started_at || "—"],
        ["Closes", round.closes_at || "—"],
        ["Saved Locally", round.saved_at_utc || "—"],
        ["Seeds", String(round.seeds.length)],
        ["Queries", `${round.total_queries} total`],
        ["Seed Coverage", seed.coverage_ratio_label],
        ["Observed Cells", `${seed.covered_cells}/${DATA.grid_size * DATA.grid_size}`],
      ];
      roundMeta.innerHTML = "";
      for (const [key, value] of rows) {
        const row = document.createElement("div");
        row.className = "meta-row";
        row.innerHTML = `<div class="key">${key}</div><div class="value">${value}</div>`;
        roundMeta.appendChild(row);
      }
    }

    function drawLayer(canvas, initialGrid, matrix, classIndex, hasGroundTruth) {
      const context = canvas.getContext("2d");
      const image = context.createImageData(DATA.grid_size, DATA.grid_size);
      const classKey = DATA.class_keys[classIndex];
      const accent = parseHexColor(DATA.class_colors[classKey]);
      const background = parseHexColor("#0b1018");
      for (let y = 0; y < DATA.grid_size; y += 1) {
        for (let x = 0; x < DATA.grid_size; x += 1) {
          const intensity = matrix ? Math.pow(clamp(matrix[y][x], 0, 1), 0.88) : 0;
          const pixel = mixColor(background, accent, intensity);
          const idx = (y * DATA.grid_size + x) * 4;
          image.data[idx] = pixel[0];
          image.data[idx + 1] = pixel[1];
          image.data[idx + 2] = pixel[2];
          image.data[idx + 3] = hasGroundTruth || matrix ? 255 : 96;
        }
      }
      context.putImageData(image, 0, 0);
      drawSelectionHighlight(context);
    }

    function drawOverview(canvas, initialGrid, tensor, hasGroundTruth) {
      const context = canvas.getContext("2d");
      const image = context.createImageData(DATA.grid_size, DATA.grid_size);
      const terrainColors = DATA.terrain_base_colors;
      for (let y = 0; y < DATA.grid_size; y += 1) {
        for (let x = 0; x < DATA.grid_size; x += 1) {
          const terrainCode = initialGrid[y][x];
          const base = parseHexColor(terrainColors[String(terrainCode)] || "#111827");
          let pixel = base;
          if (tensor) {
            let bestClass = 0;
            let bestValue = tensor[y][x][0];
            for (let classIndex = 1; classIndex < DATA.class_names.length; classIndex += 1) {
              if (tensor[y][x][classIndex] > bestValue) {
                bestValue = tensor[y][x][classIndex];
                bestClass = classIndex;
              }
            }
            const accent = parseHexColor(DATA.class_colors[DATA.class_keys[bestClass]]);
            pixel = mixColor(base, accent, Math.pow(clamp(bestValue, 0, 1), 0.92));
          }
          const idx = (y * DATA.grid_size + x) * 4;
          image.data[idx] = pixel[0];
          image.data[idx + 1] = pixel[1];
          image.data[idx + 2] = pixel[2];
          image.data[idx + 3] = tensor || hasGroundTruth ? 255 : 96;
        }
      }
      context.putImageData(image, 0, 0);
      drawSelectionHighlight(context);
    }

    function drawSelectionHighlight(context) {
      const { x, y } = state.selectedCell;
      context.save();
      context.fillStyle = "rgba(255, 255, 255, 0.28)";
      context.fillRect(x, y, 1, 1);
      context.restore();
    }

    function bindCanvasSelection(canvas) {
      if (canvas.dataset.bound === "true") {
        return;
      }
      canvas.dataset.bound = "true";
      canvas.addEventListener("mousemove", (event) => {
        const rect = canvas.getBoundingClientRect();
        const x = clamp(Math.floor(((event.clientX - rect.left) / rect.width) * DATA.grid_size), 0, DATA.grid_size - 1);
        const y = clamp(Math.floor(((event.clientY - rect.top) / rect.height) * DATA.grid_size), 0, DATA.grid_size - 1);
        state.selectedCell = { x, y };
        renderCanvases();
        renderCellInspector();
      });
      canvas.addEventListener("click", (event) => {
        const rect = canvas.getBoundingClientRect();
        const x = clamp(Math.floor(((event.clientX - rect.left) / rect.width) * DATA.grid_size), 0, DATA.grid_size - 1);
        const y = clamp(Math.floor(((event.clientY - rect.top) / rect.height) * DATA.grid_size), 0, DATA.grid_size - 1);
        state.selectedCell = { x, y };
        renderCanvases();
        renderCellInspector();
      });
    }

    function renderLayers() {
      const round = getCurrentRound();
      const seed = getCurrentSeed();
      const hasV2 = seed.prediction_v2 != null;
      layerGrid.innerHTML = "";
      for (let classIndex = 0; classIndex < DATA.class_names.length; classIndex += 1) {
        const row = document.createElement("div");
        row.className = "layer-row";
        const predictionCanvasId = `prediction-layer-${classIndex}`;
        const predictionV2CanvasId = `prediction-v2-layer-${classIndex}`;
        const groundTruthCanvasId = `ground-truth-layer-${classIndex}`;
        row.innerHTML = `
          <div class="layer-label">${DATA.class_names[classIndex]}</div>
          <div class="layer-cell">
            <canvas id="${predictionCanvasId}" width="40" height="40"></canvas>
          </div>
          <div class="layer-cell ${hasV2 ? "" : "dimmed"}">
            <canvas id="${predictionV2CanvasId}" width="40" height="40"></canvas>
          </div>
          <div class="layer-cell ${seed.ground_truth ? "" : "dimmed"}">
            <canvas id="${groundTruthCanvasId}" width="40" height="40"></canvas>
          </div>
        `;
        layerGrid.appendChild(row);
      }

      // KL divergence heatmap row
      if (seed.v1_kl_map || seed.v2_kl_map) {
        const klRow = document.createElement("div");
        klRow.className = "layer-row";
        klRow.innerHTML = `
          <div class="layer-label" style="color:#f87171">KL Divergence</div>
          <div class="layer-cell">
            <canvas id="kl-v1-layer" width="40" height="40"></canvas>
          </div>
          <div class="layer-cell ${seed.v2_kl_map ? "" : "dimmed"}">
            <canvas id="kl-v2-layer" width="40" height="40"></canvas>
          </div>
          <div class="layer-cell">
            <canvas id="kl-diff-layer" width="40" height="40"></canvas>
          </div>
        `;
        layerGrid.appendChild(klRow);
      }

      bindCanvasSelection(predictionOverview);
      bindCanvasSelection(predictionV2Overview);
      bindCanvasSelection(groundTruthOverview);
      for (const canvas of layerGrid.querySelectorAll("canvas")) {
        bindCanvasSelection(canvas);
      }
    }

    function drawKLHeatmap(canvas, klMap, maxKL) {
      if (!klMap) return;
      const context = canvas.getContext("2d");
      const image = context.createImageData(DATA.grid_size, DATA.grid_size);
      const bg = [11, 16, 24];
      for (let y = 0; y < DATA.grid_size; y += 1) {
        for (let x = 0; x < DATA.grid_size; x += 1) {
          const kl = clamp(klMap[y][x] / maxKL, 0, 1);
          const intensity = Math.pow(kl, 0.6);
          const r = Math.round(bg[0] + (248 - bg[0]) * intensity);
          const g = Math.round(bg[1] + (113 - bg[1]) * intensity * 0.4);
          const b = Math.round(bg[2] + (113 - bg[2]) * intensity * 0.3);
          const idx = (y * DATA.grid_size + x) * 4;
          image.data[idx] = r;
          image.data[idx + 1] = g;
          image.data[idx + 2] = b;
          image.data[idx + 3] = 255;
        }
      }
      context.putImageData(image, 0, 0);
      drawSelectionHighlight(context);
    }

    function drawKLDiffHeatmap(canvas, v1Map, v2Map, maxKL) {
      if (!v1Map || !v2Map) return;
      const context = canvas.getContext("2d");
      const image = context.createImageData(DATA.grid_size, DATA.grid_size);
      for (let y = 0; y < DATA.grid_size; y += 1) {
        for (let x = 0; x < DATA.grid_size; x += 1) {
          const diff = v2Map[y][x] - v1Map[y][x];
          const normalized = clamp(diff / maxKL, -1, 1);
          const intensity = Math.pow(Math.abs(normalized), 0.6);
          let r, g, b;
          if (normalized <= 0) {
            r = Math.round(11 + (74 - 11) * intensity);
            g = Math.round(16 + (222 - 16) * intensity);
            b = Math.round(24 + (128 - 24) * intensity);
          } else {
            r = Math.round(11 + (248 - 11) * intensity);
            g = Math.round(16 + (113 - 16) * intensity * 0.4);
            b = Math.round(24 + (113 - 24) * intensity * 0.3);
          }
          const idx = (y * DATA.grid_size + x) * 4;
          image.data[idx] = r;
          image.data[idx + 1] = g;
          image.data[idx + 2] = b;
          image.data[idx + 3] = 255;
        }
      }
      context.putImageData(image, 0, 0);
      drawSelectionHighlight(context);
    }

    function renderCanvases() {
      const seed = getCurrentSeed();
      const hasV2 = seed.prediction_v2 != null;
      drawOverview(predictionOverview, seed.initial_grid, seed.prediction, true);
      drawOverview(predictionV2Overview, seed.initial_grid, hasV2 ? seed.prediction_v2 : null, hasV2);
      drawOverview(groundTruthOverview, seed.initial_grid, seed.ground_truth, Boolean(seed.ground_truth));
      for (let classIndex = 0; classIndex < DATA.class_names.length; classIndex += 1) {
        const predictionCanvas = document.getElementById(`prediction-layer-${classIndex}`);
        const v2Canvas = document.getElementById(`prediction-v2-layer-${classIndex}`);
        const groundTruthCanvas = document.getElementById(`ground-truth-layer-${classIndex}`);
        drawLayer(
          predictionCanvas,
          seed.initial_grid,
          seed.prediction.map((row) => row.map((cell) => cell[classIndex])),
          classIndex,
          true,
        );
        drawLayer(
          v2Canvas,
          seed.initial_grid,
          hasV2 ? seed.prediction_v2.map((row) => row.map((cell) => cell[classIndex])) : null,
          classIndex,
          hasV2,
        );
        drawLayer(
          groundTruthCanvas,
          seed.initial_grid,
          seed.ground_truth ? seed.ground_truth.map((row) => row.map((cell) => cell[classIndex])) : null,
          classIndex,
          Boolean(seed.ground_truth),
        );
      }

      // KL heatmaps
      const maxKL = 1.5;
      const klV1Canvas = document.getElementById("kl-v1-layer");
      const klV2Canvas = document.getElementById("kl-v2-layer");
      const klDiffCanvas = document.getElementById("kl-diff-layer");
      if (klV1Canvas && seed.v1_kl_map) {
        drawKLHeatmap(klV1Canvas, seed.v1_kl_map, maxKL);
        bindCanvasSelection(klV1Canvas);
      }
      if (klV2Canvas && seed.v2_kl_map) {
        drawKLHeatmap(klV2Canvas, seed.v2_kl_map, maxKL);
        bindCanvasSelection(klV2Canvas);
      }
      if (klDiffCanvas && seed.v1_kl_map && seed.v2_kl_map) {
        drawKLDiffHeatmap(klDiffCanvas, seed.v1_kl_map, seed.v2_kl_map, maxKL);
        bindCanvasSelection(klDiffCanvas);
      }
    }

    function renderCellInspector() {
      const seed = getCurrentSeed();
      const { x, y } = state.selectedCell;
      const terrainCode = seed.initial_grid[y][x];
      const terrainLabel = DATA.terrain_labels[String(terrainCode)] || `Code ${terrainCode}`;
      const observedCount = seed.observed_counts[y][x];
      const hasV2 = seed.prediction_v2 != null;
      const cellKLv1 = seed.v1_kl_map ? seed.v1_kl_map[y][x] : null;
      const cellKLv2 = seed.v2_kl_map ? seed.v2_kl_map[y][x] : null;
      cellTitle.textContent = `Cell (${x}, ${y})`;
      let badges = `
        <span class="badge"><strong>Initial Terrain</strong> ${terrainLabel}</span>
        <span class="badge"><strong>Observed</strong> ${observedCount}x</span>
      `;
      if (cellKLv1 != null) {
        badges += `<span class="badge"><strong>V1 KL</strong> ${cellKLv1.toFixed(4)}</span>`;
      }
      if (cellKLv2 != null) {
        const diff = cellKLv2 - cellKLv1;
        const color = diff <= 0 ? "#4ade80" : "#f87171";
        badges += `<span class="badge"><strong>V2 KL</strong> ${cellKLv2.toFixed(4)} <span style="color:${color}">(${diff <= 0 ? "" : "+"}${diff.toFixed(4)})</span></span>`;
      }
      cellMeta.innerHTML = badges;
      cellGrid.innerHTML = "";
      for (let classIndex = 0; classIndex < DATA.class_names.length; classIndex += 1) {
        const prediction = seed.prediction[y][x][classIndex];
        const v2pred = hasV2 ? seed.prediction_v2[y][x][classIndex] : null;
        const truth = seed.ground_truth ? seed.ground_truth[y][x][classIndex] : null;
        const card = document.createElement("div");
        card.className = "cell-card";
        let v2Line = "";
        if (v2pred != null) {
          v2Line = `V2: ${percent(v2pred)}`;
          if (truth != null) {
            v2Line += ` (${deltaPercent(v2pred, truth)})`;
          }
        }
        card.innerHTML = `
          <div class="class-name">${DATA.class_names[classIndex]}</div>
          <div class="prediction">${percent(prediction)}</div>
          <div class="ground-truth">${v2pred != null ? `V2: <strong>${percent(v2pred)}</strong>` : ""}</div>
          <div class="ground-truth">${truth == null ? "GT: not available" : `GT: ${percent(truth)}`}</div>
          <div class="delta">${truth == null ? "" : `V1 Δ: ${deltaPercent(prediction, truth)}`}</div>
          <div class="delta">${truth == null || v2pred == null ? "" : `V2 Δ: ${deltaPercent(v2pred, truth)}`}</div>
        `;
        cellGrid.appendChild(card);
      }
    }

    function repopulateSeedSelect() {
      const round = getCurrentRound();
      seedSelect.innerHTML = "";
      round.seeds.forEach((seed, index) => {
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = seed.label;
        seedSelect.appendChild(option);
      });
      if (state.seedIndex >= round.seeds.length) {
        state.seedIndex = 0;
      }
      seedSelect.value = String(state.seedIndex);
    }

    function renderAll() {
      renderSummary();
      renderRoundMeta();
      renderLayers();
      renderCanvases();
      renderCellInspector();
    }

    function initialize() {
      DATA.rounds.forEach((round, index) => {
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = round.label;
        roundSelect.appendChild(option);
      });
      roundSelect.value = String(state.roundIndex);
      repopulateSeedSelect();
      renderAll();

      roundSelect.addEventListener("change", () => {
        state.roundIndex = Number.parseInt(roundSelect.value, 10);
        state.seedIndex = 0;
        state.selectedCell = { x: 20, y: 20 };
        repopulateSeedSelect();
        renderAll();
      });

      seedSelect.addEventListener("change", () => {
        state.seedIndex = Number.parseInt(seedSelect.value, 10);
        state.selectedCell = { x: 20, y: 20 };
        renderAll();
      });
    }

    initialize();
  </script>
</body>
</html>
"""


def _round_sort_key(manifest: dict) -> tuple[int, str]:
    round_metadata = manifest.get("round_metadata") or {}
    round_number = round_metadata.get("round_number")
    if not isinstance(round_number, int):
        round_number = -1
    tie_breaker = str(round_metadata.get("started_at") or round_metadata.get("event_date") or manifest.get("saved_at_utc", ""))
    return (round_number, tie_breaker)


def build_observation_counts(manifest: dict) -> dict[int, np.ndarray]:
    """Count how many times each cell was observed per seed."""
    counts: dict[int, np.ndarray] = {}
    for initial_state in manifest.get("initial_states", []):
        seed_index = int(initial_state["seed_index"])
        counts[seed_index] = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for query in manifest.get("queries", []):
        seed_index = int(query["seed_index"])
        viewport = query["viewport"]
        x = int(viewport["x"])
        y = int(viewport["y"])
        w = int(viewport["w"])
        h = int(viewport["h"])
        counts.setdefault(seed_index, np.zeros((GRID_SIZE, GRID_SIZE), dtype=int))
        counts[seed_index][y : y + h, x : x + w] += 1
    return counts


def _load_array(round_dir: Path, relative_path: str | None) -> np.ndarray | None:
    if not relative_path:
        return None
    path = round_dir / relative_path
    if not path.exists():
        return None
    return np.load(path)


def _seed_label(seed_index: int, submission: dict | None) -> str:
    if isinstance(submission, dict) and submission.get("score") is not None:
        return f"Seed {seed_index} | score {submission['score']}"
    return f"Seed {seed_index}"


def _serialize_tensor(tensor: np.ndarray | None, decimals: int = 6) -> list | None:
    if tensor is None:
        return None
    return np.round(np.asarray(tensor, dtype=float), decimals=decimals).tolist()


def load_history_dataset(history_root: Path) -> dict:
    """Load stored rounds into a viewer-friendly payload."""
    rounds: list[dict] = []
    for round_dir in history_root.iterdir():
        if not round_dir.is_dir():
            continue
        manifest_path = round_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        round_metadata = manifest.get("round_metadata") or {}
        observation_counts = build_observation_counts(manifest)
        submissions = manifest.get("submission_responses", {})
        diagnostics = manifest.get("diagnostics", {})
        seeds: list[dict] = []
        for initial_state in sorted(manifest.get("initial_states", []), key=lambda item: int(item["seed_index"])):
            seed_index = int(initial_state["seed_index"])
            prediction = _load_array(round_dir, manifest.get("predictions", {}).get(str(seed_index)))
            if prediction is None:
                continue
            ground_truth = _load_array(round_dir, manifest.get("ground_truth", {}).get(str(seed_index)))
            initial_grid = _load_array(round_dir, initial_state.get("grid_path"))
            if initial_grid is None:
                continue
            # Load v2 prediction if backtest has been run
            v2_path = round_dir / "arrays" / f"seed_{seed_index}_prediction_v2.npy"
            prediction_v2 = np.load(v2_path) if v2_path.exists() else None

            # Compute per-cell KL divergence maps when ground truth is available
            v1_kl_map = None
            v2_kl_map = None
            v1_kl_mean = None
            v2_kl_mean = None
            if ground_truth is not None:
                eps = 1e-12
                gt_safe = np.clip(ground_truth, eps, 1.0)
                v1_kl_map = np.sum(gt_safe * np.log(gt_safe / np.clip(prediction, eps, 1.0)), axis=-1)
                v1_kl_mean = float(v1_kl_map.mean())
                if prediction_v2 is not None:
                    v2_kl_map = np.sum(gt_safe * np.log(gt_safe / np.clip(prediction_v2, eps, 1.0)), axis=-1)
                    v2_kl_mean = float(v2_kl_map.mean())

            counts = observation_counts.get(seed_index, np.zeros((GRID_SIZE, GRID_SIZE), dtype=int))
            covered_cells = int((counts > 0).sum())
            seed_data = {
                "seed_index": seed_index,
                "label": _seed_label(seed_index, submissions.get(str(seed_index))),
                "initial_grid": initial_grid.astype(int).tolist(),
                "prediction": _serialize_tensor(prediction),
                "prediction_v2": _serialize_tensor(prediction_v2),
                "ground_truth": _serialize_tensor(ground_truth),
                "v1_kl_map": _serialize_tensor(v1_kl_map, decimals=5) if v1_kl_map is not None else None,
                "v2_kl_map": _serialize_tensor(v2_kl_map, decimals=5) if v2_kl_map is not None else None,
                "v1_kl_mean": round(v1_kl_mean, 4) if v1_kl_mean is not None else None,
                "v2_kl_mean": round(v2_kl_mean, 4) if v2_kl_mean is not None else None,
                "observed_counts": counts.astype(int).tolist(),
                "query_count": int(sum(1 for query in manifest.get("queries", []) if int(query["seed_index"]) == seed_index)),
                "covered_cells": covered_cells,
                "coverage_ratio_label": f"{covered_cells / float(GRID_SIZE * GRID_SIZE):.1%}",
                "submission": submissions.get(str(seed_index)),
                "analysis": manifest.get("analyses", {}).get(str(seed_index)),
            }
            seeds.append(seed_data)
        if not seeds:
            continue
        round_number = round_metadata.get("round_number")
        label = f"Round {round_number}" if isinstance(round_number, int) else manifest.get("round_id", round_dir.name)
        rounds.append(
            {
                "label": label,
                "round_id": str(manifest.get("round_id") or round_dir.name),
                "round_number": round_number,
                "status": round_metadata.get("status"),
                "event_date": round_metadata.get("event_date"),
                "started_at": round_metadata.get("started_at"),
                "closes_at": round_metadata.get("closes_at"),
                "saved_at_utc": manifest.get("saved_at_utc"),
                "total_queries": len(manifest.get("queries", [])),
                "analysis_summary": diagnostics.get("analysis_summary"),
                "analysis_status": (
                    "Ground truth stored"
                    if manifest.get("ground_truth")
                    else "Prediction only"
                ),
                "has_ground_truth": bool(manifest.get("ground_truth")),
                "seeds": seeds,
            }
        )

    rounds.sort(key=lambda round_entry: _round_sort_key({"round_metadata": {"round_number": round_entry.get("round_number"), "started_at": round_entry.get("started_at"), "event_date": round_entry.get("event_date")}, "saved_at_utc": round_entry.get("saved_at_utc")}), reverse=True)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "grid_size": GRID_SIZE,
        "class_names": list(CLASS_NAMES),
        "class_keys": list(CLASS_KEYS),
        "class_colors": CLASS_COLORS,
        "terrain_labels": {str(key): value for key, value in TERRAIN_LABELS.items()},
        "terrain_base_colors": {str(key): value for key, value in TERRAIN_BASE_COLORS.items()},
        "rounds": rounds,
    }


def build_html(dataset: dict) -> str:
    data_json = json.dumps(dataset, separators=(",", ":")).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__DATA_JSON__", data_json)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a local HTML viewer from stored Astar history")
    parser.add_argument(
        "--history-root",
        type=Path,
        default=PROJECT_ROOT / "history",
        help="Directory containing stored round histories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "history_viewer.html",
        help="Where to write the generated HTML file",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset = load_history_dataset(args.history_root)
    if not dataset["rounds"]:
        raise SystemExit(f"No rounds with predictions found in {args.history_root}")
    html = build_html(dataset)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output} with {len(dataset['rounds'])} rounds")


if __name__ == "__main__":
    main()
