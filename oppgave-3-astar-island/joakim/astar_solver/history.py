"""Local round-history storage for future training and analysis."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .observations import RoundObservationStore
from .types import SeedState, ViewportObservation


class RoundDatasetStore:
    """Persist round inputs, observations, predictions, and analysis artifacts."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_round(
        self,
        round_id: str,
        round_metadata: dict[str, Any],
        seed_states: list[SeedState],
        observation_store: RoundObservationStore,
        predictions: dict[int, np.ndarray] | None = None,
        submission_responses: dict[int, dict[str, Any]] | None = None,
        analyses: dict[int, dict[str, Any]] | None = None,
        ground_truth: dict[int, np.ndarray] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Path:
        """Write one round bundle to disk."""
        round_dir = self.root / round_id
        array_dir = round_dir / "arrays"
        array_dir.mkdir(parents=True, exist_ok=True)

        manifest: dict[str, Any] = {
            "round_id": round_id,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": config or {},
            "round_metadata": round_metadata,
            "initial_states": [],
            "queries": [],
            "predictions": {},
            "submission_responses": submission_responses or {},
            "analyses": analyses or {},
            "ground_truth": {},
        }

        for seed_state in seed_states:
            grid_path = Path("arrays") / f"seed_{seed_state.seed_index}_initial.npy"
            np.save(round_dir / grid_path, seed_state.grid)
            manifest["initial_states"].append(
                {
                    "seed_index": seed_state.seed_index,
                    "grid_path": str(grid_path),
                    "settlements": [
                        {
                            "x": settlement.x,
                            "y": settlement.y,
                            "has_port": settlement.has_port,
                        }
                        for settlement in seed_state.settlements
                    ],
                }
            )

        for index, observation in enumerate(observation_store.observations):
            manifest["queries"].append(self._save_observation(round_dir, index, observation))

        for seed_index, prediction in (predictions or {}).items():
            prediction_path = Path("arrays") / f"seed_{seed_index}_prediction.npy"
            np.save(round_dir / prediction_path, prediction)
            manifest["predictions"][str(seed_index)] = str(prediction_path)

        for seed_index, target in (ground_truth or {}).items():
            target_path = Path("arrays") / f"seed_{seed_index}_ground_truth.npy"
            np.save(round_dir / target_path, target)
            manifest["ground_truth"][str(seed_index)] = str(target_path)

        manifest_path = round_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return round_dir

    def load_round(self, round_id: str) -> dict[str, Any]:
        """Load a stored round bundle."""
        round_dir = self.root / round_id
        manifest = json.loads((round_dir / "manifest.json").read_text())
        loaded = {
            "manifest": manifest,
            "initial_grids": {},
            "observations": [],
            "predictions": {},
            "ground_truth": {},
        }

        for item in manifest.get("initial_states", []):
            loaded["initial_grids"][item["seed_index"]] = np.load(round_dir / item["grid_path"])

        for query in manifest.get("queries", []):
            query_record = dict(query)
            query_record["grid"] = np.load(round_dir / query["grid_path"])
            loaded["observations"].append(query_record)

        for seed_index, path in manifest.get("predictions", {}).items():
            loaded["predictions"][int(seed_index)] = np.load(round_dir / path)

        for seed_index, path in manifest.get("ground_truth", {}).items():
            loaded["ground_truth"][int(seed_index)] = np.load(round_dir / path)

        return loaded

    def update_round_analyses(self, round_id: str, analyses: dict[int, dict[str, Any]]) -> Path:
        """Merge fetched analysis payloads into an existing stored round."""
        round_dir = self.root / round_id
        manifest_path = round_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        existing = manifest.get("analyses", {})
        for seed_index, payload in analyses.items():
            existing[str(seed_index)] = payload
        manifest["analyses"] = existing
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return manifest_path

    def build_training_examples(self, round_id: str) -> list[dict[str, Any]]:
        """Turn one stored round into per-seed training examples when labels exist."""
        bundle = self.load_round(round_id)
        manifest = bundle["manifest"]
        if not bundle["ground_truth"]:
            return []

        queries_by_seed: dict[int, list[dict[str, Any]]] = {}
        for query in bundle["observations"]:
            queries_by_seed.setdefault(int(query["seed_index"]), []).append(query)

        examples = []
        for initial_state in manifest.get("initial_states", []):
            seed_index = int(initial_state["seed_index"])
            examples.append(
                {
                    "round_id": round_id,
                    "seed_index": seed_index,
                    "initial_grid": bundle["initial_grids"][seed_index],
                    "initial_settlements": initial_state["settlements"],
                    "queries": queries_by_seed.get(seed_index, []),
                    "prediction": bundle["predictions"].get(seed_index),
                    "ground_truth": bundle["ground_truth"].get(seed_index),
                }
            )
        return examples

    def _save_observation(
        self,
        round_dir: Path,
        index: int,
        observation: ViewportObservation,
    ) -> dict[str, Any]:
        grid_path = Path("arrays") / f"observation_{index}_grid.npy"
        np.save(round_dir / grid_path, observation.grid)
        return {
            "seed_index": observation.seed_index,
            "viewport": observation.viewport.as_dict(),
            "grid_path": str(grid_path),
            "queries_used": observation.queries_used,
            "queries_max": observation.queries_max,
            "settlements": [
                {
                    "x": settlement.x,
                    "y": settlement.y,
                    "population": settlement.population,
                    "food": settlement.food,
                    "wealth": settlement.wealth,
                    "defense": settlement.defense,
                    "has_port": settlement.has_port,
                    "alive": settlement.alive,
                    "owner_id": settlement.owner_id,
                }
                for settlement in observation.settlements
            ],
        }
