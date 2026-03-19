"""Adaptive query strategy tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astar_solver.config import QueryConfig
from astar_solver.predictor import PredictionArtifacts
from astar_solver.query_strategy import HeuristicQuerySelector
from astar_solver.relations import RelationArtifacts
from astar_solver.types import RoundLatentState, SeedState, Settlement, Viewport


def _dummy_artifacts() -> PredictionArtifacts:
    base = np.ones((40, 40, 6), dtype=float) / 6.0
    entropy = np.full((40, 40), 0.1, dtype=float)
    dynamic = np.full((40, 40), 0.2, dtype=float)
    entropy[20, 20] = 1.2
    dynamic[20, 20] = 0.9
    frontier = np.zeros((40, 40), dtype=float)
    frontier[20, 20] = 1.0
    relations = RelationArtifacts(
        settlement_pressure=np.zeros((40, 40), dtype=float),
        port_pressure=np.zeros((40, 40), dtype=float),
        trade_access=frontier.copy(),
        conflict_risk=frontier.copy(),
        frontier_pressure=frontier.copy(),
        nodes=(),
        edge_count=0,
    )
    return PredictionArtifacts(
        probabilities=base,
        base_probabilities=base,
        entropy_map=entropy,
        dynamic_mass=dynamic,
        observation_weight=np.zeros((40, 40), dtype=float),
        features=None,  # type: ignore[arg-type]
        relations=relations,
        latent=RoundLatentState(),
    )


class QueryStrategyTests(unittest.TestCase):
    def test_coverage_stage_prefers_large_or_medium_viewports(self) -> None:
        selector = HeuristicQuerySelector(QueryConfig())
        seed_state = SeedState(
            seed_index=0,
            grid=np.full((40, 40), 11, dtype=int),
            settlements=[Settlement(x=20, y=20, has_port=False)],
        )
        candidate = selector.select_next(
            seed_state=seed_state,
            artifacts=_dummy_artifacts(),
            coverage=np.zeros((40, 40), dtype=float),
            queries_used_for_seed=0,
            stage="coverage",
        )

        self.assertIsNotNone(candidate)
        self.assertIn(candidate.viewport.w, {11, 15})
        self.assertEqual(candidate.stage, "coverage")

    def test_adaptive_stage_can_value_intentional_repeats(self) -> None:
        selector = HeuristicQuerySelector(QueryConfig())
        coverage = np.zeros((40, 40), dtype=float)
        coverage[16:25, 16:25] = 1.0

        candidate = selector._score_candidate(
            viewport=Viewport.centered(20, 20, 11),
            artifacts=_dummy_artifacts(),
            coverage=coverage,
            queries_used_for_seed=6,
            stage="adaptive",
            origin="repeat-test",
        )

        self.assertGreater(candidate.intentional_repeat_overlap, 0.0)
        self.assertGreater(candidate.repeat_value, 0.0)
        self.assertGreater(candidate.score, -5.0)


if __name__ == "__main__":
    unittest.main()
