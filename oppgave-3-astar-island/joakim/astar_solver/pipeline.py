"""Shared round-solving pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict

from .config import SolverConfig
from .history import RoundDatasetStore
from .observations import RoundObservationStore
from .predictor import PredictionArtifacts, ProbabilisticMapPredictor
from .query_strategy import HeuristicQuerySelector
from .types import SeedState


class RoundSolver:
    """Solve an Astar Island round with shared observations and latent inference."""

    def __init__(self, client, config: SolverConfig | None = None):
        self.client = client
        self.config = config or SolverConfig()
        self.logger = logging.getLogger(__name__)
        self.predictor = ProbabilisticMapPredictor(self.config, logger=self.logger)
        self.query_selector = HeuristicQuerySelector(self.config.query, logger=self.logger)
        self.history_store = RoundDatasetStore(self.config.history_root)

    def solve_round(
        self,
        round_id: str,
        round_data: dict,
        queries_per_seed: int = 10,
        total_queries: int | None = None,
        submit: bool = True,
        dry_run: bool = False,
    ) -> dict[int, PredictionArtifacts]:
        """Run observation, prediction, submission, and storage for one round."""
        seed_payloads = round_data.get("seeds", round_data.get("initial_states", []))
        seed_states = [
            SeedState.from_round_data(seed_index=i, seed_data=seed_payload)
            for i, seed_payload in enumerate(seed_payloads)
        ]
        if not seed_states:
            raise ValueError("round data does not contain any seeds")

        seed_count = len(seed_states)
        feature_cache = {
            seed_state.seed_index: self.predictor.feature_extractor.extract(seed_state)
            for seed_state in seed_states
        }
        observation_store = RoundObservationStore(seed_states)
        planned_total_queries = total_queries or (queries_per_seed * seed_count)
        per_seed_limit = self._distribute_budget(planned_total_queries, seed_count)

        if not dry_run:
            self._collect_observations(
                round_id=round_id,
                seed_states=seed_states,
                feature_cache=feature_cache,
                observation_store=observation_store,
                per_seed_limit=per_seed_limit,
            )

        artifacts_by_seed: dict[int, PredictionArtifacts] = {}
        predictions = {}
        submission_responses: dict[int, dict] = {}
        analyses: dict[int, dict] = {}

        for seed_state in seed_states:
            artifacts = self.predictor.predict_seed(
                seed_state=seed_state,
                round_store=observation_store,
                features=feature_cache[seed_state.seed_index],
            )
            artifacts_by_seed[seed_state.seed_index] = artifacts
            predictions[seed_state.seed_index] = artifacts.probabilities

        if submit and not dry_run:
            for seed_state in seed_states:
                prediction = predictions[seed_state.seed_index].tolist()
                response = self.client.submit(round_id, seed_state.seed_index, prediction)
                submission_responses[seed_state.seed_index] = response
                self.logger.info(
                    "Submitted seed %s score=%s",
                    seed_state.seed_index,
                    response.get("score", "?"),
                )
                time.sleep(0.6)
            analyses = self.fetch_analyses(
                round_id=round_id,
                seed_indices=[seed_state.seed_index for seed_state in seed_states],
            )

        self.history_store.save_round(
            round_id=round_id,
            round_metadata=round_data,
            seed_states=seed_states,
            observation_store=observation_store,
            predictions=predictions,
            submission_responses=submission_responses,
            analyses=analyses,
            config={
                "queries_per_seed": queries_per_seed,
                "total_queries": planned_total_queries,
                "probability": asdict(self.config.probability),
                "query": asdict(self.config.query),
                "local_dynamics_passes": self.config.local_dynamics_passes,
                "observation_blend": self.config.observation_blend,
                "latent_strength": self.config.latent_strength,
            },
        )
        return artifacts_by_seed

    def fetch_analyses(
        self,
        round_id: str,
        seed_indices: list[int],
        timeout_seconds: float = 20.0,
        poll_interval_seconds: float = 2.0,
    ) -> dict[int, dict]:
        """Best-effort fetch of analysis payloads after submission."""
        analyses: dict[int, dict] = {}
        for seed_index in seed_indices:
            analysis = self._poll_analysis(
                round_id=round_id,
                seed_index=seed_index,
                timeout_seconds=timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
            )
            if analysis is not None:
                analyses[seed_index] = analysis
        return analyses

    def _poll_analysis(
        self,
        round_id: str,
        seed_index: int,
        timeout_seconds: float,
        poll_interval_seconds: float,
    ) -> dict | None:
        """Poll the analysis endpoint for one seed."""
        deadline = time.time() + timeout_seconds
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                analysis = self.client.get_analysis(round_id, seed_index)
                self.logger.info("Fetched analysis for seed %s", seed_index)
                return analysis
            except Exception as exc:
                last_error = exc
                response = getattr(exc, "response", None)
                status_code = getattr(response, "status_code", None)
                if status_code not in {404, 409, 425, 429}:
                    response_text = ""
                    if response is not None:
                        try:
                            response_text = response.text[:500]
                        except Exception:
                            response_text = "<unavailable>"
                    self.logger.warning(
                        "Analysis fetch failed for seed %s with status %s: %s body=%r",
                        seed_index,
                        status_code,
                        exc,
                        response_text,
                    )
                    return None
                time.sleep(poll_interval_seconds)
        self.logger.warning(
            "Analysis not available for seed %s within %.1fs: %s",
            seed_index,
            timeout_seconds,
            last_error,
        )
        return None

    def _collect_observations(
        self,
        round_id: str,
        seed_states: list[SeedState],
        feature_cache: dict[int, object],
        observation_store: RoundObservationStore,
        per_seed_limit: dict[int, int],
    ) -> None:
        used_per_seed = {seed_state.seed_index: 0 for seed_state in seed_states}
        pending_viewports = {seed_state.seed_index: [] for seed_state in seed_states}

        while any(used_per_seed[idx] < per_seed_limit[idx] for idx in used_per_seed):
            made_progress = False
            for seed_state in seed_states:
                seed_index = seed_state.seed_index
                if used_per_seed[seed_index] >= per_seed_limit[seed_index]:
                    continue

                artifacts = self.predictor.predict_seed(
                    seed_state=seed_state,
                    round_store=observation_store,
                    features=feature_cache[seed_index],
                )
                candidate = self.query_selector.select_next(
                    seed_state=seed_state,
                    artifacts=artifacts,
                    coverage=observation_store.get_seed_memory(seed_index).observed,
                    already_planned=pending_viewports[seed_index],
                )
                if candidate is None:
                    continue

                pending_viewports[seed_index].append(candidate.viewport)
                try:
                    result = self.client.simulate(
                        round_id,
                        seed_index,
                        candidate.viewport.x,
                        candidate.viewport.y,
                        candidate.viewport.w,
                        candidate.viewport.h,
                    )
                except Exception as exc:
                    self.logger.error("Simulation failed for seed %s: %s", seed_index, exc)
                    response = getattr(exc, "response", None)
                    if response is not None and response.status_code == 429:
                        time.sleep(2.0)
                    continue

                observation_store.add_simulation_result(round_id, seed_index, result)
                used_per_seed[seed_index] += 1
                pending_viewports[seed_index].clear()
                made_progress = True
                self.logger.info(
                    "Observed seed=%s query=%s/%s budget=%s/%s",
                    seed_index,
                    used_per_seed[seed_index],
                    per_seed_limit[seed_index],
                    result.get("queries_used", "?"),
                    result.get("queries_max", "?"),
                )
                time.sleep(0.25)

            if not made_progress:
                break

    @staticmethod
    def _distribute_budget(total_queries: int, seed_count: int) -> dict[int, int]:
        base = total_queries // seed_count
        remainder = total_queries % seed_count
        return {
            seed_index: base + (1 if seed_index < remainder else 0)
            for seed_index in range(seed_count)
        }
