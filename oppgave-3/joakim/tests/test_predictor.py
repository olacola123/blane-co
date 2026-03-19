"""Predictor structure tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astar_solver.config import SolverConfig
from astar_solver.observations import RoundObservationStore
from astar_solver.predictor import ProbabilisticMapPredictor
from astar_solver.types import SeedState, Settlement


class PredictorTests(unittest.TestCase):
    def test_port_and_ruin_are_more_concentrated_on_plausible_cells(self) -> None:
        grid = np.full((40, 40), 10, dtype=int)
        grid[5:35, 5:35] = 11
        grid[10, 10] = 1
        grid[10, 11] = 3

        seed_state = SeedState(
            seed_index=0,
            grid=grid,
            settlements=[Settlement(x=10, y=10, has_port=False)],
            static_mask=np.isin(grid, [10, 5]),
        )
        store = RoundObservationStore([seed_state])
        predictor = ProbabilisticMapPredictor(SolverConfig())

        artifacts = predictor.predict_seed(seed_state=seed_state, round_store=store)
        probabilities = artifacts.probabilities

        coastal_port = probabilities[5, 10, 2]
        inland_port = probabilities[20, 20, 2]
        near_settlement_ruin = probabilities[10, 11, 3]
        far_inland_ruin = probabilities[24, 24, 3]

        self.assertGreater(coastal_port, inland_port)
        self.assertGreater(near_settlement_ruin, far_inland_ruin)


if __name__ == "__main__":
    unittest.main()
