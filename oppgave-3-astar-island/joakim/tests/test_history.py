"""History storage tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astar_solver.history import RoundDatasetStore


class HistoryTests(unittest.TestCase):
    def test_update_round_analyses_merges_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            round_dir = root / "round-1"
            round_dir.mkdir(parents=True)
            manifest_path = round_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "round_id": "round-1",
                        "analyses": {"0": {"seed_index": 0, "status": "old"}},
                    }
                )
            )

            store = RoundDatasetStore(root)
            store.update_round_analyses("round-1", {1: {"seed_index": 1, "status": "new"}})

            updated = json.loads(manifest_path.read_text())
            self.assertIn("0", updated["analyses"])
            self.assertIn("1", updated["analyses"])
            self.assertEqual(updated["analyses"]["1"]["status"], "new")

    def test_update_round_analyses_extracts_ground_truth_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            round_dir = root / "round-1" / "arrays"
            round_dir.mkdir(parents=True)
            manifest_path = round_dir.parent / "manifest.json"
            manifest_path.write_text(json.dumps({"round_id": "round-1", "analyses": {}, "ground_truth": {}}))

            target_grid = np.full((40, 40), 11, dtype=int).tolist()
            store = RoundDatasetStore(root)
            store.update_round_analyses("round-1", {0: {"ground_truth": target_grid}})

            updated = json.loads(manifest_path.read_text())
            self.assertIn("0", updated["ground_truth"])
            saved = np.load(round_dir.parent / updated["ground_truth"]["0"])
            self.assertEqual(saved.shape, (40, 40, 6))

    def test_update_round_diagnostics_merges_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            round_dir = root / "round-1"
            round_dir.mkdir(parents=True)
            manifest_path = round_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "round_id": "round-1",
                        "diagnostics": {
                            "query": {"queries_per_seed": {"0": 6}},
                        },
                    }
                )
            )

            store = RoundDatasetStore(root)
            store.update_round_diagnostics(
                "round-1",
                {
                    "analysis": {"0": {"weighted_kl": 0.12}},
                    "analysis_summary": {"mean_weighted_kl": 0.12},
                },
            )

            updated = json.loads(manifest_path.read_text())
            self.assertIn("query", updated["diagnostics"])
            self.assertIn("analysis", updated["diagnostics"])
            self.assertIn("analysis_summary", updated["diagnostics"])


if __name__ == "__main__":
    unittest.main()
