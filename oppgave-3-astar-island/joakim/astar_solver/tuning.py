"""History- and analysis-driven calibration helpers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .constants import GRID_SIZE, NUM_CLASSES, TERRAIN_TO_CLASS
from .probability import safe_normalize


@dataclass(slots=True)
class HistoryCalibrationProfile:
    """Small bundle of class-wise calibration adjustments."""

    class_bias: np.ndarray
    class_temperature: np.ndarray
    rounds_used: int
    seeds_used: int


def _grid_to_target(grid: np.ndarray) -> np.ndarray:
    """Convert an integer terrain grid into a one-hot target tensor."""
    target = np.zeros(grid.shape + (NUM_CLASSES,), dtype=float)
    for terrain_code, class_id in TERRAIN_TO_CLASS.items():
        target[grid == terrain_code, class_id] = 1.0
    unset = target.sum(axis=-1) <= 0.0
    if np.any(unset):
        target[unset] = 1.0 / NUM_CLASSES
    return safe_normalize(target, axis=-1)


def _as_target_tensor(value: Any) -> np.ndarray | None:
    """Try to interpret one payload value as a target tensor."""
    array = np.asarray(value)
    if array.shape == (GRID_SIZE, GRID_SIZE):
        return _grid_to_target(array.astype(int))
    if array.shape == (GRID_SIZE, GRID_SIZE, NUM_CLASSES):
        result = np.array(array, dtype=float, copy=True)
        return safe_normalize(np.clip(result, 1e-9, None), axis=-1)
    return None


def extract_target_tensor(payload: Any) -> np.ndarray | None:
    """Best-effort extraction of a target tensor from an analysis payload."""
    direct = _as_target_tensor(payload)
    if direct is not None:
        return direct

    if isinstance(payload, dict):
        likely_keys = (
            "ground_truth",
            "target",
            "truth",
            "final_distribution",
            "true_distribution",
            "distribution",
            "final_grid",
            "grid",
            "terrain",
        )
        for key in likely_keys:
            if key in payload:
                extracted = extract_target_tensor(payload[key])
                if extracted is not None:
                    return extracted
        for value in payload.values():
            extracted = extract_target_tensor(value)
            if extracted is not None:
                return extracted
    elif isinstance(payload, list):
        for item in payload:
            extracted = extract_target_tensor(item)
            if extracted is not None:
                return extracted
    return None


class HistoryCalibrationTuner:
    """Estimate class-wise calibration from saved rounds with labels."""

    def __init__(self, history_root: str | Path, logger: logging.Logger | None = None) -> None:
        self.history_root = Path(history_root)
        self.logger = logger or logging.getLogger(__name__)

    def fit(self, limit: int = 8) -> HistoryCalibrationProfile:
        """Aggregate available history and derive class-wise biases and temperatures."""
        if not self.history_root.exists():
            return self._empty_profile()

        pred_mass = np.zeros(NUM_CLASSES, dtype=float)
        target_mass = np.zeros(NUM_CLASSES, dtype=float)
        rounds_used = 0
        seeds_used = 0

        round_dirs = sorted(
            [path for path in self.history_root.iterdir() if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[: max(int(limit), 0)]

        for round_dir in round_dirs:
            manifest_path = round_dir / "manifest.json"
            if not manifest_path.exists():
                continue

            manifest = json.loads(manifest_path.read_text())
            targets = self._load_targets(round_dir, manifest)
            if not targets:
                continue

            round_used = False
            for seed_key, prediction_path in manifest.get("predictions", {}).items():
                seed_index = int(seed_key)
                target = targets.get(seed_index)
                if target is None:
                    continue
                prediction = np.load(round_dir / prediction_path)
                if prediction.shape != target.shape:
                    continue
                pred_mass += prediction.reshape(-1, NUM_CLASSES).mean(axis=0)
                target_mass += target.reshape(-1, NUM_CLASSES).mean(axis=0)
                seeds_used += 1
                round_used = True

            if round_used:
                rounds_used += 1

        if rounds_used == 0 or seeds_used == 0:
            return self._empty_profile()

        bias = np.log(np.clip(target_mass, 1e-6, None) / np.clip(pred_mass, 1e-6, None))
        bias = np.clip(bias, -0.75, 0.75)
        class_temperature = np.clip(
            np.sqrt(np.clip(pred_mass, 1e-6, None) / np.clip(target_mass, 1e-6, None)),
            0.70,
            1.45,
        )
        self.logger.info(
            "Loaded historical calibration rounds=%s seeds=%s bias=%s temp=%s",
            rounds_used,
            seeds_used,
            np.round(bias, 3).tolist(),
            np.round(class_temperature, 3).tolist(),
        )
        return HistoryCalibrationProfile(
            class_bias=bias,
            class_temperature=class_temperature,
            rounds_used=rounds_used,
            seeds_used=seeds_used,
        )

    def _load_targets(self, round_dir: Path, manifest: dict[str, Any]) -> dict[int, np.ndarray]:
        """Load target tensors from ground truth arrays or analysis payloads."""
        targets: dict[int, np.ndarray] = {}
        for seed_key, relative_path in manifest.get("ground_truth", {}).items():
            target = np.load(round_dir / relative_path)
            if target.shape == (GRID_SIZE, GRID_SIZE):
                target = _grid_to_target(target.astype(int))
            elif target.shape == (GRID_SIZE, GRID_SIZE, NUM_CLASSES):
                target = safe_normalize(np.clip(target.astype(float), 1e-9, None), axis=-1)
            else:
                continue
            targets[int(seed_key)] = target

        for seed_key, payload in manifest.get("analyses", {}).items():
            if int(seed_key) in targets:
                continue
            target = extract_target_tensor(payload)
            if target is not None:
                targets[int(seed_key)] = target
        return targets

    @staticmethod
    def _empty_profile() -> HistoryCalibrationProfile:
        return HistoryCalibrationProfile(
            class_bias=np.zeros(NUM_CLASSES, dtype=float),
            class_temperature=np.ones(NUM_CLASSES, dtype=float),
            rounds_used=0,
            seeds_used=0,
        )
