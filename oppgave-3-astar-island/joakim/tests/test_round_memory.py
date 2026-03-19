"""Round memory aggregation tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astar_solver.observations import RoundObservationStore
from astar_solver.types import SeedState, Settlement


class RoundMemoryTests(unittest.TestCase):
    def test_round_memory_aggregates_viewports_across_seeds(self) -> None:
        grid_a = np.full((40, 40), 10, dtype=int)
        grid_a[5:15, 5:15] = 11
        grid_a[8, 8] = 1

        grid_b = np.full((40, 40), 10, dtype=int)
        grid_b[10:20, 10:20] = 11
        grid_b[12, 12] = 3

        seed_a = SeedState(
            seed_index=0,
            grid=grid_a,
            settlements=[Settlement(x=8, y=8, has_port=False)],
            static_mask=np.isin(grid_a, [10, 5]),
        )
        seed_b = SeedState(
            seed_index=1,
            grid=grid_b,
            settlements=[Settlement(x=12, y=12, has_port=False)],
            static_mask=np.isin(grid_b, [10, 5]),
        )

        store = RoundObservationStore([seed_a, seed_b])

        viewport_a = np.full((5, 5), 11, dtype=int)
        viewport_a[3, 3] = 2
        store.add_simulation_result(
            round_id="round-test",
            seed_index=0,
            payload={
                "grid": viewport_a.tolist(),
                "viewport": {"x": 5, "y": 5, "w": 5, "h": 5},
                "settlements": [
                    {
                        "x": 8,
                        "y": 8,
                        "population": 3.0,
                        "food": 1.5,
                        "wealth": 1.0,
                        "defense": 1.2,
                        "has_port": True,
                        "alive": True,
                        "owner_id": 1,
                    }
                ],
                "queries_used": 1,
                "queries_max": 50,
            },
        )

        viewport_b = np.full((5, 5), 11, dtype=int)
        viewport_b[2, 2] = 4
        store.add_simulation_result(
            round_id="round-test",
            seed_index=1,
            payload={
                "grid": viewport_b.tolist(),
                "viewport": {"x": 10, "y": 10, "w": 5, "h": 5},
                "settlements": [
                    {
                        "x": 12,
                        "y": 12,
                        "population": 1.0,
                        "food": 0.3,
                        "wealth": 0.2,
                        "defense": 0.4,
                        "has_port": False,
                        "alive": False,
                        "owner_id": 2,
                    }
                ],
                "queries_used": 2,
                "queries_max": 50,
            },
        )

        summary = store.build_summary()

        self.assertEqual(summary.num_viewports, 2)
        self.assertGreater(summary.coverage_ratio, 0.0)
        self.assertEqual(summary.observed_seed_fraction, 1.0)
        self.assertGreater(summary.reclaim_ratio, 0.0)
        self.assertGreater(store.get_seed_memory(0).observed[8, 8], 0)

    def test_observation_posterior_keeps_uncertainty_and_shared_prior(self) -> None:
        grid = np.full((40, 40), 11, dtype=int)
        grid[8, 8] = 1
        seed_a = SeedState(
            seed_index=0,
            grid=grid,
            settlements=[Settlement(x=8, y=8, has_port=False)],
            static_mask=np.isin(grid, [10, 5]),
        )
        seed_b = SeedState(
            seed_index=1,
            grid=grid.copy(),
            settlements=[Settlement(x=8, y=8, has_port=False)],
            static_mask=np.isin(grid, [10, 5]),
        )
        store = RoundObservationStore([seed_a, seed_b])

        store.add_simulation_result(
            round_id="round-test",
            seed_index=0,
            payload={
                "grid": [[2]],
                "viewport": {"x": 8, "y": 8, "w": 1, "h": 1},
                "settlements": [],
                "queries_used": 1,
                "queries_max": 50,
            },
        )

        shared_prior = store.terrain_conditioned_prior(seed_b)
        self.assertGreater(shared_prior[8, 8, 2], 0.05)

        prediction_prior = np.full((40, 40, 6), 1.0 / 6.0, dtype=float)
        posterior = store.get_seed_memory(0).posterior(prediction_prior, prior_strength=3.5)
        self.assertGreater(posterior[8, 8, 2], 0.25)
        self.assertLess(posterior[8, 8, 2], 0.70)
        self.assertTrue(np.allclose(posterior.sum(axis=-1), 1.0))


if __name__ == "__main__":
    unittest.main()
