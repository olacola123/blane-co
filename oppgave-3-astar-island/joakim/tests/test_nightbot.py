"""Nightbot integration tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nightbot
from astar_solver.constants import GRID_SIZE, INTERNAL_MOUNTAIN, INTERNAL_OCEAN, INTERNAL_PLAINS, NUM_CLASSES
from astar_solver.history import RoundDatasetStore
from astar_solver.observations import RoundObservationStore
from astar_solver.types import SeedState, Settlement


class _FakeAnalysisClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def get_analysis(self, round_id: str, seed_index: int) -> dict:
        return self.payload


class _TransientAnalysisError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status={status_code}")
        self.response = SimpleNamespace(status_code=status_code)


class _PartialAnalysisClient:
    def get_analysis(self, round_id: str, seed_index: int) -> dict:
        if seed_index == 0:
            grid = np.full((GRID_SIZE, GRID_SIZE), INTERNAL_PLAINS, dtype=int)
            return {"distribution": grid.tolist()}
        raise _TransientAnalysisError(404)


class _FakeRoundClient:
    def __init__(self, rounds: list[dict]) -> None:
        self.rounds = rounds

    def get_rounds(self) -> list[dict]:
        return list(self.rounds)

    def get_round(self, round_id: str) -> dict:
        return {"id": round_id}


class NightbotTests(unittest.TestCase):
    def test_get_new_round_skips_older_rounds_when_latest_is_already_known(self) -> None:
        client = _FakeRoundClient(
            [
                {"id": "round-2", "round_number": 2, "started_at": "2026-03-19T21:00:00+00:00"},
                {"id": "round-1", "round_number": 1, "started_at": "2026-03-19T18:00:00+00:00"},
            ]
        )

        result = nightbot.get_new_round(client, ["round-2"])

        self.assertIsNone(result)

    def test_bootstrap_state_from_history_marks_submitted_round_as_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            round_dir = root / "round-2"
            round_dir.mkdir(parents=True)
            (round_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "round_id": "round-2",
                        "saved_at_utc": "2026-03-20T00:00:00+00:00",
                        "initial_states": [{"seed_index": i} for i in range(5)],
                        "submission_responses": {str(i): {"ok": True} for i in range(5)},
                        "analyses": {},
                        "ground_truth": {},
                        "diagnostics": {},
                    }
                )
            )

            state = nightbot.bootstrap_state_from_history(
                RoundDatasetStore(root),
                {"solved_rounds": [], "pending_analysis": [], "round_scores": {}},
            )

            self.assertEqual(state["solved_rounds"], ["round-2"])
            self.assertEqual(len(state["pending_analysis"]), 1)
            self.assertEqual(state["pending_analysis"][0]["round_id"], "round-2")
            self.assertEqual(state["pending_analysis"][0]["seed_count"], 5)

    def test_fetch_and_score_analyses_persists_rich_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            grid = np.full((GRID_SIZE, GRID_SIZE), INTERNAL_PLAINS, dtype=int)
            grid[0, :] = INTERNAL_OCEAN
            grid[-1, :] = INTERNAL_MOUNTAIN
            seed_state = SeedState(
                seed_index=0,
                grid=grid,
                settlements=[Settlement(x=5, y=5, has_port=False)],
                static_mask=np.isin(grid, [INTERNAL_OCEAN, INTERNAL_MOUNTAIN]),
            )
            observation_store = RoundObservationStore([seed_state])
            history_store = RoundDatasetStore(root)

            prediction = np.zeros((GRID_SIZE, GRID_SIZE, NUM_CLASSES), dtype=float)
            prediction[..., 0] = 1.0
            history_store.save_round(
                round_id="round-1",
                round_metadata={"seeds": [{}]},
                seed_states=[seed_state],
                observation_store=observation_store,
                predictions={0: prediction},
            )

            diagnostics = nightbot.fetch_and_score_analyses(
                client=_FakeAnalysisClient({"distribution": grid.tolist()}),
                history_store=history_store,
                round_id="round-1",
                seed_count=1,
                timeout=0.01,
                poll_interval=0.0,
            )

            self.assertIsNotNone(diagnostics)
            assert diagnostics is not None
            self.assertIn("0", diagnostics["seeds"])
            self.assertIn("weighted_kl", diagnostics["seeds"]["0"])
            self.assertIn("classwise_ece", diagnostics["seeds"]["0"])
            self.assertIn("bucketed_kl", diagnostics["seeds"]["0"])

            manifest = json.loads((root / "round-1" / "manifest.json").read_text())
            self.assertIn("0", manifest["ground_truth"])
            self.assertIn("analysis", manifest["diagnostics"])
            self.assertIn("analysis_summary", manifest["diagnostics"])

    def test_fetch_and_score_analyses_marks_partial_round_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            grid = np.full((GRID_SIZE, GRID_SIZE), INTERNAL_PLAINS, dtype=int)
            seed_states = [
                SeedState(seed_index=i, grid=grid, settlements=[], static_mask=np.isin(grid, [INTERNAL_OCEAN, INTERNAL_MOUNTAIN]))
                for i in range(2)
            ]
            observation_store = RoundObservationStore(seed_states)
            history_store = RoundDatasetStore(root)
            prediction = np.zeros((GRID_SIZE, GRID_SIZE, NUM_CLASSES), dtype=float)
            prediction[..., 0] = 1.0
            history_store.save_round(
                round_id="round-1",
                round_metadata={"initial_states": [{}, {}]},
                seed_states=seed_states,
                observation_store=observation_store,
                predictions={0: prediction, 1: prediction},
            )

            diagnostics = nightbot.fetch_and_score_analyses(
                client=_PartialAnalysisClient(),
                history_store=history_store,
                round_id="round-1",
                seed_count=2,
                timeout=0.01,
                poll_interval=0.0,
            )

            self.assertIsNotNone(diagnostics)
            assert diagnostics is not None
            self.assertFalse(diagnostics["summary"]["analysis_complete"])
            self.assertEqual(diagnostics["summary"]["num_analysis_payloads"], 1)

    def test_run_nightbot_dry_run_does_not_persist_state(self) -> None:
        state = {"solved_rounds": [], "pending_analysis": [], "round_scores": {}}
        fake_client = object()

        with (
            patch.object(nightbot, "AstarClient", return_value=fake_client),
            patch.object(nightbot, "RoundDatasetStore"),
            patch.object(nightbot, "load_state", return_value=state),
            patch.object(nightbot, "bootstrap_state_from_history", return_value=state),
            patch.object(nightbot, "save_state") as save_state,
            patch.object(nightbot, "get_new_round", return_value=("round-1", {"seeds": [{}]})),
            patch.object(nightbot, "solve_round", return_value={}),
        ):
            nightbot.run_nightbot(poll_interval=0, analysis_wait=0, dry_run=True, max_rounds=1)

        self.assertEqual(state["solved_rounds"], [])
        self.assertEqual(state["pending_analysis"], [])
        save_state.assert_not_called()

    def test_run_nightbot_uses_analysis_wait_and_clears_pending_round(self) -> None:
        state = {"solved_rounds": [], "pending_analysis": [], "round_scores": {}}
        fake_client = object()
        diagnostics = {
            "round_id": "round-1",
            "seeds": {"0": {"weighted_kl": 0.12}},
            "summary": {"mean_weighted_kl": 0.12, "analysis_complete": True},
        }

        with (
            patch.object(nightbot, "AstarClient", return_value=fake_client),
            patch.object(nightbot, "RoundDatasetStore"),
            patch.object(nightbot, "load_state", return_value=state),
            patch.object(nightbot, "bootstrap_state_from_history", return_value=state),
            patch.object(nightbot, "save_state"),
            patch.object(nightbot, "get_new_round", return_value=("round-1", {"seeds": [{}]})),
            patch.object(nightbot, "solve_round", return_value={}),
            patch.object(nightbot, "fetch_and_score_analyses", return_value=diagnostics) as fetch_analysis,
            patch.object(nightbot, "log_diagnostics"),
        ):
            nightbot.run_nightbot(poll_interval=0, analysis_wait=123, dry_run=False, max_rounds=1)

        self.assertEqual(state["solved_rounds"], ["round-1"])
        self.assertEqual(state["pending_analysis"], [])
        self.assertEqual(state["round_scores"]["round-1"]["mean_weighted_kl"], 0.12)
        self.assertEqual(fetch_analysis.call_args.kwargs["timeout"], 123.0)

    def test_run_nightbot_keeps_round_pending_when_analysis_is_partial(self) -> None:
        state = {"solved_rounds": [], "pending_analysis": [], "round_scores": {}}
        fake_client = object()
        diagnostics = {
            "round_id": "round-1",
            "seeds": {"0": {"weighted_kl": 0.12}},
            "summary": {
                "mean_weighted_kl": 0.12,
                "analysis_complete": False,
                "num_analysis_payloads": 1,
                "expected_seed_count": 5,
            },
        }

        with (
            patch.object(nightbot, "AstarClient", return_value=fake_client),
            patch.object(nightbot, "RoundDatasetStore"),
            patch.object(nightbot, "load_state", return_value=state),
            patch.object(nightbot, "bootstrap_state_from_history", return_value=state),
            patch.object(nightbot, "save_state"),
            patch.object(nightbot, "get_new_round", return_value=("round-1", {"seeds": [{}, {}, {}, {}, {}]})),
            patch.object(nightbot, "solve_round", return_value={}),
            patch.object(nightbot, "fetch_and_score_analyses", return_value=diagnostics),
            patch.object(nightbot, "log_diagnostics"),
        ):
            nightbot.run_nightbot(poll_interval=0, analysis_wait=123, dry_run=False, max_rounds=1)

        self.assertEqual(state["solved_rounds"], ["round-1"])
        self.assertEqual(len(state["pending_analysis"]), 1)
        self.assertEqual(state["pending_analysis"][0]["round_id"], "round-1")
        self.assertEqual(state["round_scores"], {})


if __name__ == "__main__":
    unittest.main()
