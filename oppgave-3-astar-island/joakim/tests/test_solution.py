"""Solution entrypoint tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solution import _select_default_round_id


class SolutionTests(unittest.TestCase):
    def test_select_default_round_id_prefers_active_budget_round(self) -> None:
        rounds = [
            {"id": "round-6", "round_number": 6, "status": "active", "started_at": "2026-03-20T09:00:00+00:00"},
            {"id": "round-5", "round_number": 5, "status": "completed", "started_at": "2026-03-20T06:00:00+00:00"},
            {"id": "round-4", "round_number": 4, "status": "completed", "started_at": "2026-03-20T03:00:00+00:00"},
        ]
        budget = {"round_id": "round-6", "active": True}

        selected = _select_default_round_id(rounds, budget)

        self.assertEqual(selected, "round-6")

    def test_select_default_round_id_falls_back_to_highest_round_number(self) -> None:
        rounds = [
            {"id": "round-6", "round_number": 6, "status": "completed", "started_at": "2026-03-20T09:00:00+00:00"},
            {"id": "round-5", "round_number": 5, "status": "completed", "started_at": "2026-03-20T06:00:00+00:00"},
            {"id": "round-4", "round_number": 4, "status": "completed", "started_at": "2026-03-20T03:00:00+00:00"},
        ]

        selected = _select_default_round_id(rounds, budget=None)

        self.assertEqual(selected, "round-6")


if __name__ == "__main__":
    unittest.main()
