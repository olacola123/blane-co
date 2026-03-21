"""
Gradient Boosted Regression Model for Astar Island cell distributions.
=====================================================================
Strategy: Train XGBoost to predict probability distributions directly,
using the lookup table predictions as additional features.
Then blend ML predictions with lookup table using geometric mean.

Usage:
    # Train and evaluate:
    source env/bin/activate
    export API_KEY='...'
    python regression_model.py

    # Use in solution:
    from regression_model import regression_predict
    pred = regression_predict(grid, settlements, vitality)  # (40,40,6)
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# === CONFIG ===
BASE_URL = "https://api.ainm.no/astar-island"
API_KEY = os.environ.get("API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZDY4OWRmZC01NGM0LTQwZmYtYTM2My01MzMyYjc0ZDY4M2EiLCJlbWFpbCI6Im9sYWd1ZGJyYW5kQGdtYWlsLmNvbSIsImlzX2FkbWluIjpmYWxzZSwiZXhwIjoxNzc0NTYyNTMzfQ.zEUXW0mk5hfMuTTtXu5EwF9m1Ex6vh6tOUYRMnNvs7c")

MAP_W, MAP_H = 40, 40
NUM_CLASSES = 6
EPSILON = 0.003

MODEL_PATH = Path(__file__).parent / "regression_model.pkl"
DATA_CACHE_PATH = Path(__file__).parent / "regression_data_cache.pkl"
SUPER_CAL_PATH = Path(__file__).parent / "super_calibration.json"


# === LOOKUP TABLE (from super_calibration.json) ===

_super_cal = None

def _load_super_cal():
    global _super_cal
    if _super_cal is not None:
        return _super_cal
    if not SUPER_CAL_PATH.exists():
        return None
    data = json.loads(SUPER_CAL_PATH.read_text())
    _super_cal = {
        "specific": data.get("table_specific", {}),
        "density": data.get("table_density", {}),
        "simple": data.get("table_simple", {}),
    }
    return _super_cal


def _terrain_group(t):
    if t in (0, 11): return "plains"
    elif t == 1: return "settlement"
    elif t == 2: return "port"
    elif t == 3: return "ruin"
    elif t == 4: return "forest"
    else: return "other"


def _dist_bin(d):
    if d <= 0: return 0
    elif d <= 1: return 1
    elif d <= 2: return 2
    elif d <= 3: return 3
    elif d <= 5: return 4
    elif d <= 8: return 5
    else: return 6


def _settle_density_bin(n):
    return 0 if n == 0 else (1 if n <= 2 else 2)


def _forest_density_bin(n):
    if n == 0: return 0
    elif n <= 4: return 1
    elif n <= 10: return 2
    else: return 3


def vitality_to_vbin(v):
    if v < 0.08: return "DEAD"
    elif v < 0.25: return "LOW"
    elif v < 0.45: return "MED"
    else: return "HIGH"


def lookup_predict_cell(vbin, terrain, dist_settle, coastal, n_settle_r3, n_forest_r2):
    """Get lookup table prediction for a single cell."""
    cal = _load_super_cal()
    if cal is None:
        return np.array([1/6]*6)

    tg = _terrain_group(terrain)
    db = _dist_bin(dist_settle)
    c = int(coastal)
    sd = _settle_density_bin(n_settle_r3)
    fd = _forest_density_bin(n_forest_r2)

    # Try density table first
    key = f"{vbin}_{tg}_{db}_{c}_{sd}_{fd}"
    entry = cal["density"].get(key)
    if entry and entry.get("count", 0) >= 5:
        return np.array(entry["distribution"])

    # Fallback to specific
    key = f"{vbin}_{tg}_{db}_{c}"
    entry = cal["specific"].get(key)
    if entry and entry.get("count", 0) >= 5:
        return np.array(entry["distribution"])

    # Fallback to simple
    key = f"{tg}_{db}"
    entry = cal["simple"].get(key)
    if entry:
        return np.array(entry["distribution"])

    return np.array([0.7, 0.1, 0.03, 0.05, 0.12, 0.0])


# === API CLIENT ===
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
        r = self.session.get(f"{BASE_URL}/{path.lstrip('/')}")
        r.raise_for_status()
        return r.json()


# === FEATURE EXTRACTION ===

def compute_distance_map(grid_arr, target_codes):
    """BFS Chebyshev distance map to nearest cell with terrain in target_codes."""
    H, W = grid_arr.shape
    dist = np.full((H, W), 999, dtype=np.int32)
    queue = deque()
    for y in range(H):
        for x in range(W):
            if grid_arr[y, x] in target_codes:
                dist[y, x] = 0
                queue.append((y, x))
    while queue:
        cy, cx = queue.popleft()
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < H and 0 <= nx < W:
                    nd = dist[cy, cx] + 1
                    if nd < dist[ny, nx]:
                        dist[ny, nx] = nd
                        queue.append((ny, nx))
    return dist


def extract_features_batch(grid, settlements, vitality):
    """
    Extract feature matrix for all cells. Includes lookup table predictions as features.
    Returns: features (H*W, n_features), is_dynamic (H*W,) bool
    """
    from scipy.ndimage import uniform_filter

    grid_arr = np.array(grid, dtype=int)
    H, W = grid_arr.shape
    vbin = vitality_to_vbin(vitality)

    settle_positions = [(s["y"], s["x"]) for s in settlements]
    port_positions = [(s["y"], s["x"]) for s in settlements if s.get("has_port", False)]
    n_settlements_total = len(settlements)

    # Distance maps
    dist_to_settlement = compute_distance_map(grid_arr, {1, 2})
    dist_to_ocean = compute_distance_map(grid_arr, {10})
    dist_to_mountain = compute_distance_map(grid_arr, {5})
    dist_to_forest = compute_distance_map(grid_arr, {4})
    dist_to_ruin = compute_distance_map(grid_arr, {3})

    # Port settlement distance
    if port_positions:
        port_settle_map = np.zeros((H, W), dtype=int)
        for py, px in port_positions:
            port_settle_map[py, px] = 1
        dist_to_port_settlement = compute_distance_map(grid_arr * 0 + np.where(
            np.array([[port_settle_map[y, x] for x in range(W)] for y in range(H)]) > 0, 1, 0
        ).astype(int), {1})
        # Actually simpler:
        dist_to_port_settlement = np.full((H, W), 999, dtype=np.int32)
        queue = deque()
        for py, px in port_positions:
            dist_to_port_settlement[py, px] = 0
            queue.append((py, px))
        while queue:
            cy, cx = queue.popleft()
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0: continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        nd = dist_to_port_settlement[cy, cx] + 1
                        if nd < dist_to_port_settlement[ny, nx]:
                            dist_to_port_settlement[ny, nx] = nd
                            queue.append((ny, nx))
    else:
        dist_to_port_settlement = np.full((H, W), 40, dtype=np.int32)

    # Coastal
    coastal = np.zeros((H, W), dtype=np.int32)
    for y in range(H):
        for x in range(W):
            if grid_arr[y, x] == 10: continue
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and grid_arr[ny, nx] == 10:
                        coastal[y, x] = 1
                        break
                if coastal[y, x]: break

    # Neighborhood counts using prefix sums
    settle_grid = np.zeros((H, W), dtype=np.int32)
    for sy, sx in settle_positions:
        settle_grid[sy, sx] = 1
    forest_grid = (grid_arr == 4).astype(np.int32)
    ruin_grid = (grid_arr == 3).astype(np.int32)

    def chebyshev_sum(arr, radius):
        size = 2 * radius + 1
        filtered = uniform_filter(arr.astype(float), size=size, mode='constant', cval=0)
        return np.round(filtered * size * size).astype(np.int32)

    n_settle_r1 = chebyshev_sum(settle_grid, 1)
    n_settle_r2 = chebyshev_sum(settle_grid, 2)
    n_settle_r3 = chebyshev_sum(settle_grid, 3)
    n_settle_r5 = chebyshev_sum(settle_grid, 5)
    n_settle_r8 = chebyshev_sum(settle_grid, 8)
    n_forest_r1 = chebyshev_sum(forest_grid, 1)
    n_forest_r2 = chebyshev_sum(forest_grid, 2)
    n_forest_r3 = chebyshev_sum(forest_grid, 3)
    n_ruin_r2 = chebyshev_sum(ruin_grid, 2)

    # Landmass sizes
    landmass_label = np.full((H, W), -1, dtype=np.int32)
    landmass_sizes = {}
    label_id = 0
    for y in range(H):
        for x in range(W):
            if grid_arr[y, x] not in (10, 5) and landmass_label[y, x] == -1:
                queue = deque([(y, x)])
                landmass_label[y, x] = label_id
                cells = [(y, x)]
                while queue:
                    cy, cx = queue.popleft()
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0: continue
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < H and 0 <= nx < W and landmass_label[ny, nx] == -1:
                                if grid_arr[ny, nx] not in (10, 5):
                                    landmass_label[ny, nx] = label_id
                                    queue.append((ny, nx))
                                    cells.append((ny, nx))
                landmass_sizes[label_id] = len(cells)
                label_id += 1

    landmass_size_map = np.zeros((H, W), dtype=np.int32)
    for y in range(H):
        for x in range(W):
            lid = landmass_label[y, x]
            if lid >= 0:
                landmass_size_map[y, x] = landmass_sizes[lid]

    # Terrain one-hot
    is_plains = ((grid_arr == 0) | (grid_arr == 11)).astype(float)
    is_settlement = (grid_arr == 1).astype(float)
    is_ruin = (grid_arr == 3).astype(float)
    is_forest = (grid_arr == 4).astype(float)
    is_port = (grid_arr == 2).astype(float)

    n_land = max(np.sum((grid_arr != 10) & (grid_arr != 5)), 1)
    settle_density = n_settlements_total / n_land
    has_port_neighbor = (dist_to_port_settlement <= 5).astype(float)

    # Lookup table predictions as features (6 values per cell)
    lookup_preds = np.zeros((H, W, NUM_CLASSES))
    for y in range(H):
        for x in range(W):
            t = int(grid_arr[y, x])
            if t == 10:
                lookup_preds[y, x] = [1.0, 0, 0, 0, 0, 0]
            elif t == 5:
                lookup_preds[y, x] = [0, 0, 0, 0, 0, 1.0]
            else:
                lookup_preds[y, x] = lookup_predict_cell(
                    vbin, t, int(dist_to_settlement[y, x]),
                    int(coastal[y, x]), int(n_settle_r3[y, x]),
                    int(n_forest_r2[y, x])
                )

    # Build feature matrix: spatial features + lookup table predictions
    features = np.column_stack([
        is_plains.ravel(),                          # 0
        is_settlement.ravel(),                      # 1
        is_ruin.ravel(),                            # 2
        is_forest.ravel(),                          # 3
        is_port.ravel(),                            # 4
        dist_to_settlement.ravel().astype(float),   # 5
        coastal.ravel().astype(float),              # 6
        n_settle_r1.ravel().astype(float),          # 7
        n_settle_r2.ravel().astype(float),          # 8
        n_settle_r3.ravel().astype(float),          # 9
        n_settle_r5.ravel().astype(float),          # 10
        n_settle_r8.ravel().astype(float),          # 11
        n_forest_r1.ravel().astype(float),          # 12
        n_forest_r2.ravel().astype(float),          # 13
        n_forest_r3.ravel().astype(float),          # 14
        n_ruin_r2.ravel().astype(float),            # 15
        np.full(H*W, settle_density),               # 16
        np.full(H*W, vitality),                     # 17
        has_port_neighbor.ravel(),                   # 18
        dist_to_port_settlement.ravel().astype(float).clip(0, 40),  # 19
        dist_to_forest.ravel().astype(float).clip(0, 40),           # 20
        dist_to_ocean.ravel().astype(float).clip(0, 40),            # 21
        dist_to_mountain.ravel().astype(float).clip(0, 40),         # 22
        np.full(H*W, float(n_settlements_total)),                   # 23
        landmass_size_map.ravel().astype(float),                    # 24
        dist_to_ruin.ravel().astype(float).clip(0, 40),             # 25
        # Lookup table predictions as features (help model learn corrections)
        lookup_preds.reshape(-1, NUM_CLASSES)[:, 0],  # 26: lookup_p_empty
        lookup_preds.reshape(-1, NUM_CLASSES)[:, 1],  # 27: lookup_p_settlement
        lookup_preds.reshape(-1, NUM_CLASSES)[:, 2],  # 28: lookup_p_port
        lookup_preds.reshape(-1, NUM_CLASSES)[:, 3],  # 29: lookup_p_ruin
        lookup_preds.reshape(-1, NUM_CLASSES)[:, 4],  # 30: lookup_p_forest
        lookup_preds.reshape(-1, NUM_CLASSES)[:, 5],  # 31: lookup_p_mountain
    ])

    is_dynamic = np.array([(grid_arr[y, x] not in (10, 5))
                           for y in range(H) for x in range(W)])

    return features, is_dynamic, lookup_preds.reshape(-1, NUM_CLASSES)


# === DATA FETCHING ===

def fetch_all_data():
    """Fetch all ground truth data from API."""
    if DATA_CACHE_PATH.exists():
        print("Loading cached data...")
        with open(DATA_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    client = Client()
    print("Fetching rounds...")
    rounds = client.get("/my-rounds")
    completed = [r for r in rounds if r.get("status") == "completed"]
    print(f"Found {len(rounds)} rounds, {len(completed)} completed")

    all_data = []
    for rnd in completed:
        round_id = rnd["id"]
        round_num = rnd.get("round_number", "?")
        print(f"\nRound {round_num}:")

        try:
            details = client.get(f"/rounds/{round_id}")
        except Exception as e:
            print(f"  Error: {e}")
            continue

        initial_states = details.get("initial_states", [])
        if not initial_states:
            print("  No initial states")
            continue

        for seed_idx in range(len(initial_states)):
            try:
                gt_data = client.get(f"/analysis/{round_id}/{seed_idx}")
            except Exception as e:
                print(f"  Seed {seed_idx}: Error: {e}")
                continue

            ground_truth = gt_data.get("ground_truth")
            if ground_truth is None:
                continue

            state = initial_states[seed_idx]
            grid = state["grid"]
            settlements = state["settlements"]

            gt_arr = np.array(ground_truth)
            n_init = len(settlements)
            if n_init > 0:
                alive = sum(1 for s in settlements if np.argmax(gt_arr[s["y"], s["x"]]) in (1, 2))
                vitality = alive / n_init
            else:
                vitality = 0.0

            all_data.append({
                "round_id": round_id,
                "round_num": round_num,
                "seed_idx": seed_idx,
                "grid": grid,
                "settlements": settlements,
                "ground_truth": gt_arr,
                "vitality": vitality,
            })
            print(f"  Seed {seed_idx}: v={vitality:.3f}, {n_init} settle")
            time.sleep(0.05)

    print(f"\nTotal: {len(all_data)} datasets")
    with open(DATA_CACHE_PATH, "wb") as f:
        pickle.dump(all_data, f)
    return all_data


# === SCORING ===

def weighted_kl(gt, pred):
    gt = np.array(gt, dtype=float)
    pred = np.array(pred, dtype=float)
    gt_safe = np.clip(gt, 1e-12, 1.0)
    pred_safe = np.clip(pred, 1e-12, 1.0)
    cell_kl = np.sum(gt_safe * (np.log(gt_safe) - np.log(pred_safe)), axis=-1)
    cell_entropy = -np.sum(gt_safe * np.log(gt_safe), axis=-1)
    tw = cell_entropy.sum()
    if tw <= 0:
        return float(cell_kl.mean())
    return float((cell_kl * cell_entropy).sum() / tw)


def score_prediction(gt, pred):
    return 100.0 * np.exp(-3.0 * weighted_kl(gt, pred))


# === MODEL ===

def apply_floor(pred, eps=EPSILON):
    pred = np.clip(pred, eps, None)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    return pred


def prepare_training_data(all_data):
    """Extract features and targets."""
    X_list, y_list, lookup_list = [], [], []
    round_nums = []

    for item in all_data:
        features, is_dynamic, lookup_preds = extract_features_batch(
            item["grid"], item["settlements"], item["vitality"]
        )
        gt = item["ground_truth"].reshape(-1, NUM_CLASSES)

        X_list.append(features[is_dynamic])
        y_list.append(gt[is_dynamic])
        lookup_list.append(lookup_preds[is_dynamic])
        round_nums.extend([item["round_num"]] * is_dynamic.sum())

    X = np.vstack(X_list)
    y = np.vstack(y_list)
    lookups = np.vstack(lookup_list)
    round_nums = np.array(round_nums)

    print(f"Training data: {X.shape[0]} cells, {X.shape[1]} features")
    return X, y, lookups, round_nums


def train_xgb(X, y, round_nums, leave_out=None):
    """Train XGBoost model."""
    from xgboost import XGBRegressor
    from sklearn.multioutput import MultiOutputRegressor
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    if leave_out is not None:
        mask = round_nums != leave_out
        X_train, y_train = X[mask], y[mask]
    else:
        X_train, y_train = X, y

    xgb = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.6,
        colsample_bytree=0.6,
        min_child_weight=100,
        reg_alpha=2.0,
        reg_lambda=10.0,
        gamma=1.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )

    model = MultiOutputRegressor(xgb, n_jobs=1)
    model.fit(X_train, y_train)
    return model


def predict_with_model(model, grid, settlements, vitality, blend_alpha=0.3, eps=EPSILON):
    """
    Predict (H, W, 6) using ML model blended with lookup table.
    blend_alpha: weight for ML model (1-alpha for lookup table)
    Uses geometric mean blending (better for KL divergence).
    """
    grid_arr = np.array(grid, dtype=int)
    H, W = grid_arr.shape

    features, is_dynamic, lookup_preds = extract_features_batch(grid, settlements, vitality)

    pred = np.zeros((H * W, NUM_CLASSES), dtype=float)

    # Static cells
    for i in range(H * W):
        y, x = divmod(i, W)
        t = grid_arr[y, x]
        if t == 10:
            pred[i] = [1.0 - 5*eps, eps, eps, eps, eps, eps]
        elif t == 5:
            pred[i] = [eps, eps, eps, eps, eps, 1.0 - 5*eps]

    # Dynamic cells — blend ML + lookup
    if is_dynamic.sum() > 0:
        ml_raw = model.predict(features[is_dynamic])
        ml_raw = np.clip(ml_raw, eps, 1.0)

        lk = np.clip(lookup_preds[is_dynamic], eps, 1.0)

        # Geometric mean blending: p = ml^alpha * lookup^(1-alpha)
        combined = np.exp(
            blend_alpha * np.log(ml_raw) + (1 - blend_alpha) * np.log(lk)
        )
        pred[is_dynamic] = combined

    pred = pred.reshape(H, W, NUM_CLASSES)

    # Mountain = 0 for dynamic cells
    for yy in range(H):
        for xx in range(W):
            if grid_arr[yy, xx] not in (5, 10):
                pred[yy, xx, 5] = min(pred[yy, xx, 5], eps)

    pred = apply_floor(pred, eps)
    return pred


def find_optimal_blend(model, all_data, round_num):
    """Find optimal blend_alpha for a specific round."""
    round_data = [d for d in all_data if d["round_num"] == round_num]
    best_alpha = 0.0
    best_score = 0.0

    for alpha in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        scores = []
        for item in round_data:
            pred = predict_with_model(model, item["grid"], item["settlements"],
                                      item["vitality"], blend_alpha=alpha)
            scores.append(score_prediction(item["ground_truth"], pred))
        avg = np.mean(scores)
        if avg > best_score:
            best_score = avg
            best_alpha = alpha

    return best_alpha, best_score


# === MAIN PREDICT FUNCTION ===

_loaded_model = None

def regression_predict(grid, settlements, vitality):
    """
    Main predict function.
    Args:
        grid: 40x40 terrain grid
        settlements: list of {"x", "y", "has_port"} dicts
        vitality: float, settlement survival rate
    Returns: (40, 40, 6) numpy array
    """
    global _loaded_model
    if _loaded_model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"No model at {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            _loaded_model = pickle.load(f)

    model = _loaded_model["model"]
    alpha = _loaded_model.get("blend_alpha", 0.3)
    return predict_with_model(model, grid, settlements, vitality, blend_alpha=alpha)


# === LEAVE-ONE-ROUND-OUT CV ===

def run_cv(all_data, X, y, lookups, round_nums):
    unique_rounds = sorted(set(round_nums))
    print(f"\n{'='*70}")
    print(f"Leave-one-round-out CV ({len(unique_rounds)} rounds)")
    print(f"{'='*70}")

    results = {}

    # Also compute lookup-only scores for comparison
    print(f"\n{'Round':>6} | {'Lookup':>8} | ", end="")
    for a in [0.0, 0.1, 0.2, 0.3, 0.5]:
        print(f"{'α='+str(a):>8} | ", end="")
    print()
    print("-" * 70)

    for rnum in unique_rounds:
        model = train_xgb(X, y, round_nums, leave_out=rnum)
        round_data = [d for d in all_data if d["round_num"] == rnum]

        # Lookup-only score
        lk_scores = []
        for item in round_data:
            pred = predict_with_model(model, item["grid"], item["settlements"],
                                      item["vitality"], blend_alpha=0.0)
            lk_scores.append(score_prediction(item["ground_truth"], pred))
        lk_avg = np.mean(lk_scores)

        # Various blend alphas
        print(f"{rnum:>6} | {lk_avg:>7.1f}  | ", end="")
        best_alpha, best_score = 0.0, lk_avg
        for alpha in [0.0, 0.1, 0.2, 0.3, 0.5]:
            scores = []
            for item in round_data:
                pred = predict_with_model(model, item["grid"], item["settlements"],
                                          item["vitality"], blend_alpha=alpha)
                scores.append(score_prediction(item["ground_truth"], pred))
            avg = np.mean(scores)
            print(f"{avg:>7.1f}  | ", end="")
            if avg > best_score:
                best_score = avg
                best_alpha = alpha

        print(f"  best: α={best_alpha:.1f} → {best_score:.1f}")
        results[rnum] = {"best_alpha": best_alpha, "best_score": best_score, "lookup": lk_avg}

    print(f"\n{'='*70}")
    lookup_avg = np.mean([r["lookup"] for r in results.values()])
    best_avg = np.mean([r["best_score"] for r in results.values()])
    print(f"Lookup-only avg:  {lookup_avg:.1f}")
    print(f"Best-blend avg:   {best_avg:.1f}")
    print(f"Reference:        82.2 (current system)")

    # Find globally best alpha
    global_best_alpha = 0.0
    global_best_score = 0.0
    for alpha in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        total = 0
        for rnum in unique_rounds:
            model = train_xgb(X, y, round_nums, leave_out=rnum)
            round_data = [d for d in all_data if d["round_num"] == rnum]
            scores = []
            for item in round_data:
                pred = predict_with_model(model, item["grid"], item["settlements"],
                                          item["vitality"], blend_alpha=alpha)
                scores.append(score_prediction(item["ground_truth"], pred))
            total += np.mean(scores)
        avg = total / len(unique_rounds)
        print(f"  Global α={alpha:.2f}: avg={avg:.1f}")
        if avg > global_best_score:
            global_best_score = avg
            global_best_alpha = alpha

    print(f"\nBest global alpha: {global_best_alpha:.2f} → {global_best_score:.1f}")
    return results, global_best_alpha


# === MAIN ===

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    print("=" * 70)
    print("Astar Island — Gradient Boosted Regression Model")
    print("=" * 70)

    # 1. Fetch data
    all_data = fetch_all_data()
    if not all_data:
        print("No data!")
        sys.exit(1)

    # 2. Prepare features
    print("\nExtracting features...")
    X, y, lookups, round_nums = prepare_training_data(all_data)

    # 3. CV evaluation
    results, best_alpha = run_cv(all_data, X, y, lookups, round_nums)

    # 4. Train final model
    print(f"\nTraining final model (alpha={best_alpha:.2f})...")
    final_model = train_xgb(X, y, round_nums)

    # Save
    save_data = {"model": final_model, "blend_alpha": best_alpha}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(save_data, f)
    print(f"Saved to {MODEL_PATH}")

    # Verify
    print("\nVerification (train):")
    train_scores = []
    for item in all_data:
        pred = predict_with_model(final_model, item["grid"], item["settlements"],
                                  item["vitality"], blend_alpha=best_alpha)
        train_scores.append(score_prediction(item["ground_truth"], pred))
    print(f"  Mean: {np.mean(train_scores):.1f}")
    print("\nDone!")
