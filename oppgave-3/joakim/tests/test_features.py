"""Feature extraction tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astar_solver.features import MapFeatureExtractor
from astar_solver.types import SeedState, Settlement


class FeatureExtractionTests(unittest.TestCase):
    def test_geography_features_capture_coastline_and_proximity(self) -> None:
        grid = np.full((40, 40), 10, dtype=int)
        grid[5:16, 5:16] = 11
        grid[8, 8] = 1
        grid[8, 9] = 4
        grid[10, 10] = 5

        state = SeedState(
            seed_index=0,
            grid=grid,
            settlements=[Settlement(x=8, y=8, has_port=False)],
            static_mask=np.isin(grid, [10, 5]),
        )
        features = MapFeatureExtractor().extract(state)

        self.assertGreater(features.channel("coastal_mask")[5, 10], 0.0)
        self.assertGreater(features.channel("forest_adj")[8, 8], 0.0)
        self.assertEqual(features.landmass_id[8, 8], features.landmass_id[14, 14])
        self.assertGreater(
            features.channel("settlement_proximity")[8, 8],
            features.channel("settlement_proximity")[14, 14],
        )


if __name__ == "__main__":
    unittest.main()
