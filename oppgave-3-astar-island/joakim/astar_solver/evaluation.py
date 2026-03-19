"""Offline evaluation and calibration diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .probability import predictive_entropy


def kl_divergence(target: np.ndarray, prediction: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Cell-wise KL(target || prediction)."""
    safe_target = np.clip(target, eps, 1.0)
    safe_prediction = np.clip(prediction, eps, 1.0)
    return np.sum(safe_target * (np.log(safe_target) - np.log(safe_prediction)), axis=-1)


def weighted_kl(target: np.ndarray, prediction: np.ndarray) -> float:
    """Entropy-weighted KL divergence matching the competition objective."""
    cell_kl = kl_divergence(target, prediction)
    weights = predictive_entropy(target)
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        return float(cell_kl.mean())
    return float((cell_kl * weights).sum() / total_weight)


def negative_log_likelihood(target: np.ndarray, prediction: np.ndarray, eps: float = 1e-12) -> float:
    """Distributional NLL against a target probability tensor."""
    safe_prediction = np.clip(prediction, eps, 1.0)
    return float(-np.mean(np.sum(target * np.log(safe_prediction), axis=-1)))


def brier_score(target: np.ndarray, prediction: np.ndarray) -> float:
    """Multiclass Brier score."""
    return float(np.mean(np.sum((prediction - target) ** 2, axis=-1)))


@dataclass(slots=True)
class CalibrationDiagnostics:
    """Small bundle of calibration metrics."""

    nll: float
    brier: float
    ece: float
    weighted_kl_value: float


def expected_calibration_error(
    target: np.ndarray,
    prediction: np.ndarray,
    num_bins: int = 10,
) -> float:
    """Simple multiclass ECE using argmax confidence."""
    confidences = prediction.max(axis=-1).ravel()
    predicted_class = prediction.argmax(axis=-1).ravel()
    flat_target = target.reshape(-1, target.shape[-1])
    target_mass = flat_target[np.arange(flat_target.shape[0]), predicted_class]

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lower) & (confidences < upper if upper < 1.0 else confidences <= upper)
        if not np.any(mask):
            continue
        avg_conf = float(confidences[mask].mean())
        avg_acc = float(target_mass[mask].mean())
        ece += (mask.mean()) * abs(avg_conf - avg_acc)
    return float(ece)


def calibration_diagnostics(target: np.ndarray, prediction: np.ndarray) -> CalibrationDiagnostics:
    """Compute the main offline diagnostics in one call."""
    return CalibrationDiagnostics(
        nll=negative_log_likelihood(target, prediction),
        brier=brier_score(target, prediction),
        ece=expected_calibration_error(target, prediction),
        weighted_kl_value=weighted_kl(target, prediction),
    )
