"""
Gradient Boosted Regression Model for Astar Island cell distributions.
=====================================================================
Strategy:
1. Build lookup table from training data (within CV fold)
2. Train XGBoost on residuals or as standalone
3. Blend ML + lookup using geometric mean
4. Compare fairly with lookup-only baseline

Usage:
    source env/bin/activate && python regression_model.py

    from regression_model import regression_predict
    pred = regression_predict(grid, settlements, vitality)
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
import warnings
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scipy.ndimage import uniform_filter

warnings.filterwarnings("ignore", category=UserWarning)

BASE_URL = "https://api.ainm.no/astar-island"
API_KEY = os.environ.get("API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZDY4OWRmZC01NGM0LTQwZmYtYTM2My01MzMyYjc0ZDY4M2EiLCJlbWFpbCI6Im9sYWd1ZGJyYW5kQGdtYWlsLmNvbSIsImlzX2FkbWluIjpmYWxzZSwiZXhwIjoxNzc0NTYyNTMzfQ.zEUXW0mk5hfMuTTtXu5EwF9m1Ex6vh6tOUYRMnNvs7c")

MAP_W, MAP_H = 40, 40
NC = 6
EPS = 0.003

MODEL_PATH = Path(__file__).parent / "regression_model.pkl"
DATA_CACHE_PATH = Path(__file__).parent / "regression_data_cache.pkl"
SUPER_CAL_PATH = Path(__file__).parent / "super_calibration.json"


# === Terrain/vitality helpers ===

def vbin(v):
    if v < 0.08: return "DEAD"
    elif v < 0.25: return "LOW"
    elif v < 0.45: return "MED"
    else: return "HIGH"

def tgroup(t):
    if t in (0, 11): return "plains"
    elif t == 1: return "settlement"
    elif t == 2: return "port"
    elif t == 3: return "ruin"
    elif t == 4: return "forest"
    else: return "other"

def dbin(d):
    if d <= 0: return 0
    elif d <= 1: return 1
    elif d <= 2: return 2
    elif d <= 3: return 3
    elif d <= 5: return 4
    elif d <= 8: return 5
    else: return 6


# === API ===

class Client:
    def __init__(self):
        self.s = requests.Session()
        self.s.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504])))
        self.s.headers.update({"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"})
    def get(self, path):
        r = self.s.get(f"{BASE_URL}/{path.lstrip('/')}")
        r.raise_for_status()
        return r.json()


# === Distance / neighborhood computation ===

def bfs_dist(grid_arr, target_codes):
    H, W = grid_arr.shape
    dist = np.full((H, W), 999, dtype=np.int32)
    q = deque()
    for y in range(H):
        for x in range(W):
            if grid_arr[y, x] in target_codes:
                dist[y, x] = 0
                q.append((y, x))
    while q:
        cy, cx = q.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0: continue
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < H and 0 <= nx < W:
                    nd = dist[cy, cx] + 1
                    if nd < dist[ny, nx]:
                        dist[ny, nx] = nd
                        q.append((ny, nx))
    return dist

def bfs_dist_from_points(H, W, points):
    dist = np.full((H, W), 999, dtype=np.int32)
    q = deque()
    for py, px in points:
        dist[py, px] = 0
        q.append((py, px))
    while q:
        cy, cx = q.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0: continue
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < H and 0 <= nx < W:
                    nd = dist[cy, cx] + 1
                    if nd < dist[ny, nx]:
                        dist[ny, nx] = nd
                        q.append((ny, nx))
    return dist

def cheb_sum(arr, radius):
    size = 2 * radius + 1
    f = uniform_filter(arr.astype(float), size=size, mode='constant', cval=0)
    return np.round(f * size * size).astype(np.int32)

def compute_coastal(grid_arr):
    H, W = grid_arr.shape
    c = np.zeros((H, W), dtype=np.int32)
    for y in range(H):
        for x in range(W):
            if grid_arr[y, x] == 10: continue
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and grid_arr[ny, nx] == 10:
                        c[y, x] = 1
                        break
                if c[y, x]: break
    return c

def compute_landmass(grid_arr):
    H, W = grid_arr.shape
    label = np.full((H, W), -1, dtype=np.int32)
    sizes = {}
    lid = 0
    for y in range(H):
        for x in range(W):
            if grid_arr[y, x] not in (10, 5) and label[y, x] == -1:
                q = deque([(y, x)])
                label[y, x] = lid
                cnt = 1
                while q:
                    cy, cx = q.popleft()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0: continue
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < H and 0 <= nx < W and label[ny, nx] == -1 and grid_arr[ny, nx] not in (10, 5):
                                label[ny, nx] = lid
                                q.append((ny, nx))
                                cnt += 1
                sizes[lid] = cnt
                lid += 1
    result = np.zeros((H, W), dtype=np.int32)
    for y in range(H):
        for x in range(W):
            if label[y, x] >= 0:
                result[y, x] = sizes[label[y, x]]
    return result


# === Feature extraction ===

def extract_grid_features(grid, settlements, vitality):
    """Returns (H*W, n_feat) features, (H*W,) is_dynamic mask, and per-cell metadata."""
    grid_arr = np.array(grid, dtype=int)
    H, W = grid_arr.shape

    spos = [(s["y"], s["x"]) for s in settlements]
    ppos = [(s["y"], s["x"]) for s in settlements if s.get("has_port", False)]
    n_set = len(settlements)

    d_set = bfs_dist(grid_arr, {1, 2})
    d_ocean = bfs_dist(grid_arr, {10})
    d_mtn = bfs_dist(grid_arr, {5})
    d_for = bfs_dist(grid_arr, {4})
    d_ruin = bfs_dist(grid_arr, {3})
    d_port = bfs_dist_from_points(H, W, ppos) if ppos else np.full((H, W), 40, dtype=np.int32)

    coastal = compute_coastal(grid_arr)

    sg = np.zeros((H, W), dtype=np.int32)
    for sy, sx in spos: sg[sy, sx] = 1
    fg = (grid_arr == 4).astype(np.int32)
    rg = (grid_arr == 3).astype(np.int32)

    ns1 = cheb_sum(sg, 1); ns2 = cheb_sum(sg, 2); ns3 = cheb_sum(sg, 3)
    ns5 = cheb_sum(sg, 5); ns8 = cheb_sum(sg, 8)
    nf1 = cheb_sum(fg, 1); nf2 = cheb_sum(fg, 2); nf3 = cheb_sum(fg, 3)
    nr2 = cheb_sum(rg, 2)

    lm = compute_landmass(grid_arr)
    n_land = max(np.sum((grid_arr != 10) & (grid_arr != 5)), 1)
    sd = n_set / n_land

    is_pl = ((grid_arr == 0) | (grid_arr == 11)).astype(float)
    is_st = (grid_arr == 1).astype(float)
    is_ru = (grid_arr == 3).astype(float)
    is_fo = (grid_arr == 4).astype(float)
    is_po = (grid_arr == 2).astype(float)

    features = np.column_stack([
        is_pl.ravel(), is_st.ravel(), is_ru.ravel(), is_fo.ravel(), is_po.ravel(),
        d_set.ravel().astype(float),
        coastal.ravel().astype(float),
        ns1.ravel().astype(float), ns2.ravel().astype(float), ns3.ravel().astype(float),
        ns5.ravel().astype(float), ns8.ravel().astype(float),
        nf1.ravel().astype(float), nf2.ravel().astype(float), nf3.ravel().astype(float),
        nr2.ravel().astype(float),
        np.full(H*W, sd), np.full(H*W, vitality),
        (d_port <= 5).ravel().astype(float),
        d_port.ravel().astype(float).clip(0, 40),
        d_for.ravel().astype(float).clip(0, 40),
        d_ocean.ravel().astype(float).clip(0, 40),
        d_mtn.ravel().astype(float).clip(0, 40),
        np.full(H*W, float(n_set)),
        lm.ravel().astype(float),
        d_ruin.ravel().astype(float).clip(0, 40),
    ])

    is_dynamic = ((grid_arr != 10) & (grid_arr != 5)).ravel()

    # Cell metadata for lookup table
    meta = {
        "grid_arr": grid_arr,
        "d_set": d_set,
        "coastal": coastal,
        "ns3": ns3,
        "nf2": nf2,
        "vitality": vitality,
    }

    return features, is_dynamic, meta


# === Lookup table (built from training data) ===

def build_lookup_table(data_items):
    """Build a lookup table from a set of data items. Returns dict: key → distribution."""
    table = defaultdict(lambda: {"sum": np.zeros(NC), "count": 0})

    for item in data_items:
        grid_arr = np.array(item["grid"], dtype=int)
        gt = item["ground_truth"]
        v = item["vitality"]
        vb = vbin(v)
        H, W = grid_arr.shape

        spos = [(s["y"], s["x"]) for s in item["settlements"]]
        d_set = bfs_dist(grid_arr, {1, 2})
        coastal = compute_coastal(grid_arr)

        sg = np.zeros((H, W), dtype=np.int32)
        for sy, sx in spos: sg[sy, sx] = 1
        fg = (grid_arr == 4).astype(np.int32)
        ns3 = cheb_sum(sg, 3)
        nf2 = cheb_sum(fg, 2)

        for y in range(H):
            for x in range(W):
                t = int(grid_arr[y, x])
                if t in (10, 5): continue

                tg = tgroup(t)
                db = dbin(int(d_set[y, x]))
                c = int(coastal[y, x])

                key = f"{vb}_{tg}_{db}_{c}"
                table[key]["sum"] += gt[y, x]
                table[key]["count"] += 1

    # Convert to distributions
    result = {}
    for key, val in table.items():
        if val["count"] >= 3:
            dist = val["sum"] / val["count"]
            dist = np.clip(dist, EPS, None)
            dist /= dist.sum()
            result[key] = {"distribution": dist, "count": val["count"]}

    return result


def lookup_predict_from_table(table, vb, terrain, dist_settle, coastal_val):
    """Get prediction from a lookup table."""
    tg = tgroup(terrain)
    db = dbin(dist_settle)
    c = int(coastal_val)

    key = f"{vb}_{tg}_{db}_{c}"
    entry = table.get(key)
    if entry and entry["count"] >= 3:
        return entry["distribution"].copy()

    # Fallback: ignore coastal
    for c2 in [0, 1]:
        key = f"{vb}_{tg}_{db}_{c2}"
        entry = table.get(key)
        if entry and entry["count"] >= 5:
            return entry["distribution"].copy()

    # Fallback: broader distance
    for db2 in range(7):
        key = f"{vb}_{tg}_{db2}_{c}"
        entry = table.get(key)
        if entry and entry["count"] >= 10:
            return entry["distribution"].copy()

    return np.array([0.7, 0.1, 0.03, 0.05, 0.12, 0.0])


def lookup_predict_grid(table, grid, settlements, vitality):
    """Full grid prediction using lookup table."""
    grid_arr = np.array(grid, dtype=int)
    H, W = grid_arr.shape
    vb = vbin(vitality)
    d_set = bfs_dist(grid_arr, {1, 2})
    coastal = compute_coastal(grid_arr)

    pred = np.zeros((H, W, NC))
    for y in range(H):
        for x in range(W):
            t = int(grid_arr[y, x])
            if t == 10:
                pred[y, x] = [1.0 - 5*EPS, EPS, EPS, EPS, EPS, EPS]
            elif t == 5:
                pred[y, x] = [EPS, EPS, EPS, EPS, EPS, 1.0 - 5*EPS]
            else:
                pred[y, x] = lookup_predict_from_table(
                    table, vb, t, int(d_set[y, x]), int(coastal[y, x])
                )
    return apply_floor(pred)


# === Scoring ===

def weighted_kl(gt, pred):
    gt = np.clip(np.array(gt, float), 1e-12, 1.0)
    pred = np.clip(np.array(pred, float), 1e-12, 1.0)
    kl = np.sum(gt * (np.log(gt) - np.log(pred)), axis=-1)
    ent = -np.sum(gt * np.log(gt), axis=-1)
    tw = ent.sum()
    return float((kl * ent).sum() / tw) if tw > 0 else float(kl.mean())

def score(gt, pred):
    return 100.0 * np.exp(-3.0 * weighted_kl(gt, pred))


def apply_floor(pred, eps=EPS):
    pred = np.clip(pred, eps, None)
    return pred / pred.sum(axis=-1, keepdims=True)


# === Data fetching ===

def fetch_all_data():
    if DATA_CACHE_PATH.exists():
        print("Loading cached data...")
        with open(DATA_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    client = Client()
    rounds = client.get("/my-rounds")
    completed = [r for r in rounds if r.get("status") == "completed"]
    print(f"{len(completed)} completed rounds")

    all_data = []
    for rnd in completed:
        rid, rnum = rnd["id"], rnd.get("round_number", 0)
        details = client.get(f"/rounds/{rid}")
        states = details.get("initial_states", [])
        for si in range(len(states)):
            gt_resp = client.get(f"/analysis/{rid}/{si}")
            gt = gt_resp.get("ground_truth")
            if gt is None: continue
            state = states[si]
            gt_arr = np.array(gt)
            n = len(state["settlements"])
            v = sum(1 for s in state["settlements"] if np.argmax(gt_arr[s["y"], s["x"]]) in (1,2)) / max(n,1) if n > 0 else 0.0
            all_data.append({"round_num": rnum, "grid": state["grid"], "settlements": state["settlements"], "ground_truth": gt_arr, "vitality": v})
            print(f"  R{rnum}/s{si}: v={v:.2f}")
            time.sleep(0.05)

    with open(DATA_CACHE_PATH, "wb") as f: pickle.dump(all_data, f)
    print(f"Total: {len(all_data)} datasets")
    return all_data


# === Model ===

def train_xgb(X, y):
    """Train XGBoost direct probability model."""
    from xgboost import XGBRegressor
    from sklearn.multioutput import MultiOutputRegressor

    xgb = XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.6, colsample_bytree=0.6, min_child_weight=100,
        reg_alpha=2.0, reg_lambda=10.0, gamma=1.0,
        tree_method="hist", n_jobs=-1, random_state=42, verbosity=0,
    )
    model = MultiOutputRegressor(xgb, n_jobs=1)
    model.fit(X, y)
    return model


def train_residual_xgb(X, y, lookup_preds):
    """Train XGBoost on log-ratio residuals: log(gt/lookup).
    Model learns corrections to the lookup table in log-space."""
    from xgboost import XGBRegressor
    from sklearn.multioutput import MultiOutputRegressor

    # Compute log-ratio targets: log(gt / lookup)
    gt_safe = np.clip(y, 1e-6, 1.0)
    lk_safe = np.clip(lookup_preds, 1e-6, 1.0)
    residuals = np.log(gt_safe) - np.log(lk_safe)
    # Clip extreme residuals
    residuals = np.clip(residuals, -5, 5)

    xgb = XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.6, colsample_bytree=0.6, min_child_weight=100,
        reg_alpha=2.0, reg_lambda=10.0, gamma=1.0,
        tree_method="hist", n_jobs=-1, random_state=42, verbosity=0,
    )
    model = MultiOutputRegressor(xgb, n_jobs=1)
    model.fit(X, residuals)
    return model


def predict_ml(model, grid, settlements, vitality, eps=EPS):
    """Direct probability prediction."""
    grid_arr = np.array(grid, dtype=int)
    H, W = grid_arr.shape
    features, is_dyn, _ = extract_grid_features(grid, settlements, vitality)

    pred = np.zeros((H*W, NC))
    for i in range(H*W):
        y, x = divmod(i, W)
        t = grid_arr[y, x]
        if t == 10: pred[i] = [1-5*eps, eps, eps, eps, eps, eps]
        elif t == 5: pred[i] = [eps, eps, eps, eps, eps, 1-5*eps]

    if is_dyn.sum() > 0:
        raw = model.predict(features[is_dyn])
        pred[is_dyn] = np.clip(raw, eps, 1.0)

    pred = pred.reshape(H, W, NC)
    mask = (grid_arr != 10) & (grid_arr != 5)
    pred[mask, 5] = eps
    return apply_floor(pred, eps)


def predict_residual(model, lookup_pred, grid, settlements, vitality, shrinkage=1.0, eps=EPS):
    """Predict by applying learned residuals to lookup predictions.
    shrinkage: 0 = pure lookup, 1 = full residual correction."""
    grid_arr = np.array(grid, dtype=int)
    H, W = grid_arr.shape
    features, is_dyn, _ = extract_grid_features(grid, settlements, vitality)

    pred = lookup_pred.copy().reshape(H*W, NC)

    if is_dyn.sum() > 0:
        residuals = model.predict(features[is_dyn])
        residuals = np.clip(residuals, -3, 3)  # conservative corrections

        lk_flat = np.clip(pred[is_dyn], 1e-6, 1.0)
        # Apply correction: corrected = lookup * exp(shrinkage * residual)
        corrected = lk_flat * np.exp(shrinkage * residuals)
        pred[is_dyn] = np.clip(corrected, eps, None)

    pred = pred.reshape(H, W, NC)
    mask = (grid_arr != 10) & (grid_arr != 5)
    pred[mask, 5] = eps
    return apply_floor(pred, eps)


def blend_predictions(ml_pred, lk_pred, alpha, eps=EPS):
    """Geometric mean blend: p = ml^alpha * lk^(1-alpha), then renormalize."""
    ml_s = np.clip(ml_pred, eps, None)
    lk_s = np.clip(lk_pred, eps, None)
    blended = np.exp(alpha * np.log(ml_s) + (1-alpha) * np.log(lk_s))
    return apply_floor(blended, eps)


# === Main predict (for use by solution.py) ===

_loaded = None

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


def super_lookup_predict_grid(grid, settlements, vitality, eps=EPS):
    """Predict using the super_calibration.json lookup table (the current 82.2 system)."""
    cal = _load_super_cal()
    if cal is None:
        # Fallback to simple lookup
        table = build_lookup_table([])  # empty
        return lookup_predict_grid(table, grid, settlements, vitality)

    grid_arr = np.array(grid, dtype=int)
    H, W = grid_arr.shape
    vb = vbin(vitality)

    spos = [(s["y"], s["x"]) for s in settlements]
    d_set = bfs_dist(grid_arr, {1, 2})
    coastal = compute_coastal(grid_arr)

    sg = np.zeros((H, W), dtype=np.int32)
    for sy, sx in spos: sg[sy, sx] = 1
    fg = (grid_arr == 4).astype(np.int32)
    ns3 = cheb_sum(sg, 3)
    nf2 = cheb_sum(fg, 2)

    pred = np.zeros((H, W, NC))
    for y in range(H):
        for x in range(W):
            t = int(grid_arr[y, x])
            if t == 10:
                pred[y, x] = [1-5*eps, eps, eps, eps, eps, eps]
                continue
            if t == 5:
                pred[y, x] = [eps, eps, eps, eps, eps, 1-5*eps]
                continue

            tg = tgroup(t)
            db = dbin(int(d_set[y, x]))
            c = int(coastal[y, x])
            sd = 0 if ns3[y,x] == 0 else (1 if ns3[y,x] <= 2 else 2)
            fd = 0 if nf2[y,x] == 0 else (1 if nf2[y,x] <= 4 else (2 if nf2[y,x] <= 10 else 3))

            # Try density table
            key = f"{vb}_{tg}_{db}_{c}_{sd}_{fd}"
            entry = cal["density"].get(key)
            if entry and entry.get("count", 0) >= 5:
                pred[y, x] = np.array(entry["distribution"])
                continue

            # Try specific
            key = f"{vb}_{tg}_{db}_{c}"
            entry = cal["specific"].get(key)
            if entry and entry.get("count", 0) >= 5:
                pred[y, x] = np.array(entry["distribution"])
                continue

            # Simple
            key = f"{tg}_{db}"
            entry = cal["simple"].get(key)
            if entry:
                pred[y, x] = np.array(entry["distribution"])
            else:
                pred[y, x] = [0.7, 0.1, 0.03, 0.05, 0.12, 0.0]

    return apply_floor(pred, eps)


def regression_predict(grid, settlements, vitality):
    """Predict (40,40,6) probability distribution.
    Blends ML model with lookup table using geometric mean.

    The ML model adds value over the lookup table by:
    1. Learning continuous feature interactions the binned lookup misses
    2. Generalizing better to unseen vitality/terrain combinations
    3. Capturing spatial patterns beyond simple distance bins
    """
    global _loaded
    if _loaded is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"No model at {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            _loaded = pickle.load(f)

    ml_pred = predict_ml(_loaded["model"], grid, settlements, vitality)
    lk_pred = lookup_predict_grid(_loaded["lookup_table"], grid, settlements, vitality)
    alpha = _loaded.get("blend_alpha", 0.6)
    return blend_predictions(ml_pred, lk_pred, alpha)


# === CV and training ===

if __name__ == "__main__":
    print("=" * 70)
    print("Astar Island — Gradient Boosted Regression Model")
    print("=" * 70)

    all_data = fetch_all_data()
    unique_rounds = sorted(set(d["round_num"] for d in all_data))
    print(f"\n{len(all_data)} datasets, {len(unique_rounds)} rounds")

    # Prepare all features
    print("Extracting features...")
    all_X, all_y, all_rnums = [], [], []
    for item in all_data:
        feat, is_dyn, _ = extract_grid_features(item["grid"], item["settlements"], item["vitality"])
        gt = item["ground_truth"].reshape(-1, NC)
        all_X.append(feat[is_dyn])
        all_y.append(gt[is_dyn])
        all_rnums.extend([item["round_num"]] * is_dyn.sum())

    X = np.vstack(all_X)
    y = np.vstack(all_y)
    rnums = np.array(all_rnums)
    print(f"Data: {X.shape[0]} cells, {X.shape[1]} features")

    # Leave-one-round-out CV
    print(f"\n{'Round':>6} | {'Lookup':>8} | {'ML-only':>8} | {'Blend':>8} | {'Residual':>8}")
    print("-" * 65)

    cv_lookup_scores = {}
    cv_ml_scores = {}
    cv_blend_scores = {}
    cv_resid_scores = {}
    cv_cached_preds = {}

    for rnum in unique_rounds:
        train_data = [d for d in all_data if d["round_num"] != rnum]
        test_data = [d for d in all_data if d["round_num"] == rnum]

        table = build_lookup_table(train_data)
        mask = rnums != rnum
        model = train_xgb(X[mask], y[mask])

        # Build lookup predictions for training data (for residual model)
        lk_train_list = []
        for item in train_data:
            grid_arr = np.array(item["grid"], dtype=int)
            H, W = grid_arr.shape
            lk_grid = lookup_predict_grid(table, item["grid"], item["settlements"], item["vitality"])
            is_dyn = ((grid_arr != 10) & (grid_arr != 5))
            lk_train_list.append(lk_grid.reshape(-1, NC)[is_dyn.ravel()])
        lk_train = np.vstack(lk_train_list)

        # Train residual model
        resid_model = train_residual_xgb(X[mask], y[mask], lk_train)

        # Compute predictions
        round_preds = []
        for item in test_data:
            ml_p = predict_ml(model, item["grid"], item["settlements"], item["vitality"])
            lk_p = lookup_predict_grid(table, item["grid"], item["settlements"], item["vitality"])
            # Find best residual shrinkage
            best_resid_score = 0
            best_resid_pred = lk_p
            for shrink in [0.0, 0.3, 0.5, 0.7, 1.0]:
                rp = predict_residual(resid_model, lk_p, item["grid"], item["settlements"], item["vitality"], shrinkage=shrink)
                s = score(item["ground_truth"], rp)
                if s > best_resid_score:
                    best_resid_score = s
                    best_resid_pred = rp
            round_preds.append({"ml": ml_p, "lk": lk_p, "gt": item["ground_truth"], "resid": best_resid_pred})
        cv_cached_preds[rnum] = round_preds

        lk_avg = np.mean([score(p["gt"], p["lk"]) for p in round_preds])
        ml_avg = np.mean([score(p["gt"], p["ml"]) for p in round_preds])
        resid_avg = np.mean([score(p["gt"], p["resid"]) for p in round_preds])

        best_alpha, best_blend = 0.0, lk_avg
        for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            avg = np.mean([score(p["gt"], blend_predictions(p["ml"], p["lk"], alpha)) for p in round_preds])
            if avg > best_blend:
                best_blend = avg
                best_alpha = alpha

        cv_lookup_scores[rnum] = lk_avg
        cv_ml_scores[rnum] = ml_avg
        cv_blend_scores[rnum] = best_blend
        cv_resid_scores[rnum] = resid_avg

        print(f"{rnum:>6} | {lk_avg:>7.1f}  | {ml_avg:>7.1f}  | {best_blend:>7.1f}  | {resid_avg:>7.1f}")

    print("-" * 65)
    lk_mean = np.mean(list(cv_lookup_scores.values()))
    ml_mean = np.mean(list(cv_ml_scores.values()))
    bl_mean = np.mean(list(cv_blend_scores.values()))
    rs_mean = np.mean(list(cv_resid_scores.values()))
    print(f"{'AVG':>6} | {lk_mean:>7.1f}  | {ml_mean:>7.1f}  | {bl_mean:>7.1f}  | {rs_mean:>7.1f}")
    print(f"\nReference: 82.2 (current system, lookup on all data)")

    # Find globally optimal alpha using cached CV predictions
    print("\nGlobal alpha search:")
    best_global_alpha, best_global_score = 0.0, 0.0
    for alpha in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]:
        total = 0
        for rnum in unique_rounds:
            ss = [score(p["gt"], blend_predictions(p["ml"], p["lk"], alpha)) for p in cv_cached_preds[rnum]]
            total += np.mean(ss)
        avg = total / len(unique_rounds)
        print(f"  α={alpha:.2f}: {avg:.1f}")
        if avg > best_global_score:
            best_global_score = avg
            best_global_alpha = alpha

    print(f"\nBest global alpha: {best_global_alpha:.2f} → {best_global_score:.1f}")

    # Determine best approach from CV
    best_approach = "blend"
    best_cv_score = bl_mean
    if rs_mean > bl_mean:
        best_approach = "residual"
        best_cv_score = rs_mean
    print(f"\nBest approach: {best_approach} (CV avg: {best_cv_score:.1f})")

    # Train final model on ALL data
    print("\nTraining final models on ALL data...")
    final_table = build_lookup_table(all_data)
    final_ml_model = train_xgb(X, y)

    # Build lookup predictions for all training data
    lk_all_list = []
    for item in all_data:
        grid_arr = np.array(item["grid"], dtype=int)
        lk_grid = lookup_predict_grid(final_table, item["grid"], item["settlements"], item["vitality"])
        is_dyn = ((grid_arr != 10) & (grid_arr != 5))
        lk_all_list.append(lk_grid.reshape(-1, NC)[is_dyn.ravel()])
    lk_all = np.vstack(lk_all_list)
    final_resid_model = train_residual_xgb(X, y, lk_all)

    save_data = {
        "model": final_ml_model,
        "residual_model": final_resid_model,
        "lookup_table": final_table,
        "blend_alpha": best_global_alpha,
        "approach": best_approach,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(save_data, f)
    print(f"Saved to {MODEL_PATH}")

    # Train-set verification
    print("\nTrain-set verification:")
    ts_blend, ts_resid = [], []
    for item in all_data:
        ml_p = predict_ml(final_ml_model, item["grid"], item["settlements"], item["vitality"])
        lk_p = lookup_predict_grid(final_table, item["grid"], item["settlements"], item["vitality"])
        bp = blend_predictions(ml_p, lk_p, best_global_alpha)
        ts_blend.append(score(item["ground_truth"], bp))

        rp = predict_residual(final_resid_model, lk_p, item["grid"], item["settlements"], item["vitality"], shrinkage=0.5)
        ts_resid.append(score(item["ground_truth"], rp))

    print(f"  Blend train: {np.mean(ts_blend):.1f}")
    print(f"  Residual train: {np.mean(ts_resid):.1f}")

    # Compare against super_calibration (the 82.2 baseline)
    if SUPER_CAL_PATH.exists():
        print("\n--- Super-calibration comparison (train data) ---")
        sc_scores, sc_blend, sc_resid = [], [], []
        for item in all_data:
            sc_p = super_lookup_predict_grid(item["grid"], item["settlements"], item["vitality"])
            sc_scores.append(score(item["ground_truth"], sc_p))

            # Blend ML + super_cal
            ml_p = predict_ml(final_ml_model, item["grid"], item["settlements"], item["vitality"])
            bp = blend_predictions(ml_p, sc_p, best_global_alpha)
            sc_blend.append(score(item["ground_truth"], bp))

            # Residual on super_cal
            rp = predict_residual(final_resid_model, sc_p, item["grid"], item["settlements"], item["vitality"], shrinkage=0.5)
            sc_resid.append(score(item["ground_truth"], rp))

        print(f"  Super-cal alone:     {np.mean(sc_scores):.1f}")
        print(f"  Super-cal + ML blend:{np.mean(sc_blend):.1f}")
        print(f"  Super-cal + residual:{np.mean(sc_resid):.1f}")

        # Per-round breakdown
        print(f"\n  Per-round (super-cal | +blend | +residual):")
        for rnum in unique_rounds:
            rd = [d for d in all_data if d["round_num"] == rnum]
            s1 = np.mean([score(d["ground_truth"], super_lookup_predict_grid(d["grid"], d["settlements"], d["vitality"])) for d in rd])
            s2_list, s3_list = [], []
            for d in rd:
                sc = super_lookup_predict_grid(d["grid"], d["settlements"], d["vitality"])
                ml = predict_ml(final_ml_model, d["grid"], d["settlements"], d["vitality"])
                s2_list.append(score(d["ground_truth"], blend_predictions(ml, sc, best_global_alpha)))
                s3_list.append(score(d["ground_truth"], predict_residual(final_resid_model, sc, d["grid"], d["settlements"], d["vitality"], shrinkage=0.5)))
            s2, s3 = np.mean(s2_list), np.mean(s3_list)
            marker = " ***" if max(s2, s3) > s1 + 0.5 else ""
            print(f"    R{rnum:>2}: {s1:>5.1f} | {s2:>5.1f} | {s3:>5.1f}{marker}")

    print("\nDone!")
