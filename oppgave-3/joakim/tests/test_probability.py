"""Probability utility tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astar_solver.probability import (
    apply_probability_floor,
    calibrate_probabilities,
    temperature_scale,
    validate_probability_tensor,
)


class ProbabilityTests(unittest.TestCase):
    def test_probability_floor_removes_zeroes_and_renormalizes(self) -> None:
        probabilities = np.zeros((2, 2, 6), dtype=float)
        probabilities[..., 0] = 1.0

        floored = apply_probability_floor(probabilities, eps=0.01)

        self.assertTrue(np.all(floored >= 0.01 - 1e-9))
        self.assertTrue(np.allclose(floored.sum(axis=-1), 1.0))

    def test_temperature_scaling_preserves_simplex(self) -> None:
        probabilities = np.array([[[0.60, 0.15, 0.10, 0.08, 0.04, 0.03]]], dtype=float)

        scaled = temperature_scale(probabilities, temperature=1.7)

        valid, errors = validate_probability_tensor(scaled)
        self.assertTrue(valid, msg="; ".join(errors))

    def test_validation_flags_invalid_rows(self) -> None:
        probabilities = np.array([[[0.5, 0.5, 0.5, 0.0, 0.0, 0.0]]], dtype=float)
        valid, errors = validate_probability_tensor(probabilities)

        self.assertFalse(valid)
        self.assertTrue(any("sum to 1" in error for error in errors))

    def test_classwise_calibration_reweights_port_and_ruin(self) -> None:
        probabilities = np.array([[[0.30, 0.20, 0.20, 0.15, 0.10, 0.05]]], dtype=float)

        calibrated = calibrate_probabilities(
            probabilities,
            temperature=1.0,
            class_temperatures=(1.0, 1.0, 1.1, 1.1, 1.0, 1.0),
            class_bias=(0.0, 0.0, -0.4, -0.3, 0.0, 0.0),
        )

        self.assertLess(calibrated[0, 0, 2], probabilities[0, 0, 2])
        self.assertLess(calibrated[0, 0, 3], probabilities[0, 0, 3])
        self.assertTrue(np.allclose(calibrated.sum(axis=-1), 1.0))


if __name__ == "__main__":
    unittest.main()
