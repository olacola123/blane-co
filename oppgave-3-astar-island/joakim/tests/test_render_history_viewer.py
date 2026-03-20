"""Tests for the local HTML history viewer."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from render_history_viewer import build_observation_counts, load_history_dataset


class RenderHistoryViewerTests(unittest.TestCase):
    def test_build_observation_counts_accumulates_overlapping_queries(self) -> None:
        manifest = {
            "initial_states": [{"seed_index": 0}],
            "queries": [
                {"seed_index": 0, "viewport": {"x": 1, "y": 1, "w": 2, "h": 2}},
                {"seed_index": 0, "viewport": {"x": 2, "y": 2, "w": 2, "h": 2}},
            ],
        }

        counts = build_observation_counts(manifest)

        self.assertEqual(int(counts[0][1, 1]), 1)
        self.assertEqual(int(counts[0][2, 2]), 2)
        self.assertEqual(int(counts[0][3, 3]), 1)

    def test_load_history_dataset_reads_prediction_and_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            round_dir = root / "round-1"
            arrays_dir = round_dir / "arrays"
            arrays_dir.mkdir(parents=True)

            initial = np.full((40, 40), 11, dtype=int)
            prediction = np.zeros((40, 40, 6), dtype=float)
            prediction[..., 0] = 0.95
            prediction[..., 1:] = 0.01
            ground_truth = np.zeros((40, 40, 6), dtype=float)
            ground_truth[..., 0] = 1.0

            np.save(arrays_dir / "seed_0_initial.npy", initial)
            np.save(arrays_dir / "seed_0_prediction.npy", prediction)
            np.save(arrays_dir / "seed_0_ground_truth.npy", ground_truth)

            manifest = {
                "round_id": "round-1",
                "saved_at_utc": "2026-03-20T00:00:00+00:00",
                "round_metadata": {
                    "id": "round-1",
                    "round_number": 1,
                    "status": "completed",
                    "started_at": "2026-03-20T00:00:00+00:00",
                    "event_date": "2026-03-20",
                },
                "initial_states": [
                    {
                        "seed_index": 0,
                        "grid_path": "arrays/seed_0_initial.npy",
                        "settlements": [],
                    }
                ],
                "queries": [
                    {
                        "seed_index": 0,
                        "viewport": {"x": 0, "y": 0, "w": 2, "h": 2},
                    }
                ],
                "predictions": {"0": "arrays/seed_0_prediction.npy"},
                "ground_truth": {"0": "arrays/seed_0_ground_truth.npy"},
                "submission_responses": {"0": {"score": 42.0}},
                "diagnostics": {"analysis_summary": {"mean_weighted_kl": 0.123}},
            }
            (round_dir / "manifest.json").write_text(json.dumps(manifest))

            dataset = load_history_dataset(root)

            self.assertEqual(len(dataset["rounds"]), 1)
            round_entry = dataset["rounds"][0]
            self.assertTrue(round_entry["has_ground_truth"])
            self.assertEqual(round_entry["label"], "Round 1")
            seed_entry = round_entry["seeds"][0]
            self.assertEqual(seed_entry["label"], "Seed 0 | score 42.0")
            self.assertEqual(seed_entry["query_count"], 1)
            self.assertEqual(seed_entry["covered_cells"], 4)
            self.assertAlmostEqual(seed_entry["prediction"][0][0][0], 0.95, places=6)
            self.assertAlmostEqual(seed_entry["ground_truth"][0][0][0], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
