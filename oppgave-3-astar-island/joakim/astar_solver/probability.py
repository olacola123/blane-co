"""Probability utilities used by prediction and evaluation."""

from __future__ import annotations

import numpy as np


def safe_normalize(
    values: np.ndarray,
    axis: int = -1,
    floor: float | None = None,
) -> np.ndarray:
    """Normalize a tensor along one axis, optionally flooring entries first."""
    result = np.array(values, dtype=float, copy=True)
    if floor is not None:
        result = np.maximum(result, floor)
    denom = result.sum(axis=axis, keepdims=True)
    zero_mask = denom <= 0.0
    if np.any(zero_mask):
        result = result + zero_mask.astype(float)
        denom = result.sum(axis=axis, keepdims=True)
    return result / denom


def apply_probability_floor(probabilities: np.ndarray, eps: float) -> np.ndarray:
    """Centralized probability flooring followed by renormalization."""
    normalized = safe_normalize(probabilities, axis=-1)
    num_classes = normalized.shape[-1]
    if eps < 0.0:
        raise ValueError("eps must be non-negative")
    if eps * num_classes >= 1.0:
        raise ValueError("eps is too large for the number of classes")
    return normalized * (1.0 - eps * num_classes) + eps


def temperature_scale(
    probabilities: np.ndarray,
    temperature: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Apply temperature scaling directly in probability space."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if np.isclose(temperature, 1.0):
        return np.array(probabilities, dtype=float, copy=True)
    logits = np.log(np.clip(probabilities, eps, 1.0))
    logits = logits / temperature
    logits -= np.max(logits, axis=-1, keepdims=True)
    scaled = np.exp(logits)
    return safe_normalize(scaled, axis=-1)


def ensemble_average(
    predictions: list[np.ndarray],
    floor: float,
) -> np.ndarray:
    """Average multiple probability tensors and re-validate numerically."""
    if not predictions:
        raise ValueError("predictions cannot be empty")
    stacked = np.stack(predictions, axis=0)
    averaged = stacked.mean(axis=0)
    return apply_probability_floor(averaged, eps=floor)


def predictive_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-cell entropy map."""
    probs = np.clip(probabilities, eps, 1.0)
    return -np.sum(probs * np.log(probs), axis=-1)


def validate_probability_tensor(
    probabilities: np.ndarray,
    atol: float = 1e-6,
) -> tuple[bool, list[str]]:
    """Check if the tensor is a valid probability field."""
    errors: list[str] = []
    if probabilities.ndim != 3:
        errors.append(f"expected 3 dimensions, got {probabilities.ndim}")
        return False, errors
    if np.any(probabilities < 0.0):
        errors.append("probabilities contain negative values")
    if np.any(probabilities > 1.0 + atol):
        errors.append("probabilities contain values above 1")
    sums = probabilities.sum(axis=-1)
    if not np.allclose(sums, 1.0, atol=atol):
        errors.append("probabilities do not sum to 1 along the class axis")
    if np.any(np.isnan(probabilities)):
        errors.append("probabilities contain NaN values")
    return not errors, errors
