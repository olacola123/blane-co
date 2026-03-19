"""Shared round-solving pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict

from .config import SolverConfig
from .evaluation import (
    bucketed_error_diagnostics,
    calibration_diagnostics,
    classwise_expected_calibration_error,
    prediction_field_diagnostics,
)
from .history import RoundDatasetStore
from .observations import RoundObservationStore
from .predictor import PredictionArtifacts, ProbabilisticMapPredictor
from .query_strategy import HeuristicQuerySelector
from .tuning import HistoryCalibrationTuner, extract_target_tensor
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
        if self.config.model.history_tuning_enabled:
            profile = HistoryCalibrationTuner(self.config.history_root, logger=self.logger).fit(
                limit=self.config.model.history_tuning_round_limit
            )
            self.predictor.set_history_calibration(
                class_bias=profile.class_bias,
                class_temperature=profile.class_temperature,
                rounds_used=profile.rounds_used,
            )

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
        minimum_queries_per_seed = self._minimum_queries_per_seed(
            total_queries=planned_total_queries,
            seed_count=seed_count,
        )
        query_diagnostics = self._empty_query_diagnostics(seed_states)

        if not dry_run:
            query_diagnostics = self._collect_observations(
                round_id=round_id,
                seed_states=seed_states,
                feature_cache=feature_cache,
                observation_store=observation_store,
                total_queries=planned_total_queries,
                minimum_queries_per_seed=minimum_queries_per_seed,
            )

        artifacts_by_seed: dict[int, PredictionArtifacts] = {}
        predictions = {}
        submission_responses: dict[int, dict] = {}
        analyses: dict[int, dict] = {}
        prediction_diagnostics: dict[str, dict] = {}

        for seed_state in seed_states:
            artifacts = self.predictor.predict_seed(
                seed_state=seed_state,
                round_store=observation_store,
                features=feature_cache[seed_state.seed_index],
            )
            artifacts_by_seed[seed_state.seed_index] = artifacts
            predictions[seed_state.seed_index] = artifacts.probabilities
            field_diagnostics = prediction_field_diagnostics(artifacts.probabilities)
            prediction_diagnostics[str(seed_state.seed_index)] = {
                "mean_max_prob": field_diagnostics.mean_max_probability,
                "mean_entropy": field_diagnostics.mean_entropy,
                "class_frequency": list(field_diagnostics.class_frequency),
            }

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

        ground_truth = {
            seed_index: target
            for seed_index, payload in analyses.items()
            if (target := extract_target_tensor(payload)) is not None
        }
        analysis_diagnostics = self._build_analysis_diagnostics(
            predictions=predictions,
            targets=ground_truth,
            feature_cache=feature_cache,
        )
        self.history_store.save_round(
            round_id=round_id,
            round_metadata=round_data,
            seed_states=seed_states,
            observation_store=observation_store,
            predictions=predictions,
            submission_responses=submission_responses,
            analyses=analyses,
            ground_truth=ground_truth,
            config={
                "queries_per_seed": queries_per_seed,
                "total_queries": planned_total_queries,
                "probability": asdict(self.config.probability),
                "model": asdict(self.config.model),
                "query": asdict(self.config.query),
                "local_dynamics_passes": self.config.local_dynamics_passes,
                "latent_strength": self.config.latent_strength,
            },
            diagnostics={
                "query": query_diagnostics,
                "prediction": prediction_diagnostics,
                "analysis": analysis_diagnostics,
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
        total_queries: int,
        minimum_queries_per_seed: int,
    ) -> dict[str, object]:
        used_per_seed = {seed_state.seed_index: 0 for seed_state in seed_states}
        diagnostics = self._empty_query_diagnostics(seed_states)
        minimum_targets = {
            seed_state.seed_index: minimum_queries_per_seed
            for seed_state in seed_states
        }
        total_used = 0

        while total_used < total_queries and any(
            used_per_seed[idx] < minimum_targets[idx] for idx in used_per_seed
        ):
            made_progress = False
            for seed_state in seed_states:
                seed_index = seed_state.seed_index
                if used_per_seed[seed_index] >= minimum_targets[seed_index]:
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
                    queries_used_for_seed=used_per_seed[seed_index],
                    stage="coverage",
                    global_progress=total_used / max(total_queries, 1),
                )
                if candidate is None:
                    continue

                if self._execute_query(
                    round_id=round_id,
                    seed_index=seed_index,
                    candidate=candidate,
                    observation_store=observation_store,
                    used_per_seed=used_per_seed,
                    diagnostics=diagnostics,
                    total_queries=total_queries,
                ):
                    total_used += 1
                    made_progress = True

            if not made_progress:
                break
        while total_used < total_queries:
            best_seed_state: SeedState | None = None
            best_candidate = None
            best_score = float("-inf")
            for seed_state in seed_states:
                seed_index = seed_state.seed_index
                artifacts = self.predictor.predict_seed(
                    seed_state=seed_state,
                    round_store=observation_store,
                    features=feature_cache[seed_index],
                )
                candidate = self.query_selector.select_next(
                    seed_state=seed_state,
                    artifacts=artifacts,
                    coverage=observation_store.get_seed_memory(seed_index).observed,
                    queries_used_for_seed=used_per_seed[seed_index],
                    stage="adaptive",
                    global_progress=total_used / max(total_queries, 1),
                )
                if candidate is None:
                    continue
                if candidate.score > best_score:
                    best_score = candidate.score
                    best_seed_state = seed_state
                    best_candidate = candidate

            if best_seed_state is None or best_candidate is None:
                break

            if not self._execute_query(
                round_id=round_id,
                seed_index=best_seed_state.seed_index,
                candidate=best_candidate,
                observation_store=observation_store,
                used_per_seed=used_per_seed,
                diagnostics=diagnostics,
                total_queries=total_queries,
            ):
                break
            total_used += 1

        self.logger.info(
            "Query summary total=%s per_seed=%s unique=%s overlap=%s deliberate=%.3f accidental=%.3f viewport_sizes=%s",
            total_used,
            diagnostics["queries_per_seed"],
            diagnostics["unique_coverage_by_seed"],
            diagnostics["overlap_by_seed"],
            diagnostics["deliberate_repeat_overlap"],
            diagnostics["accidental_overlap"],
            diagnostics["viewport_sizes"],
        )
        return diagnostics

    def _minimum_queries_per_seed(self, total_queries: int, seed_count: int) -> int:
        if seed_count <= 0 or total_queries <= 0:
            return 0
        return min(self.config.query.minimum_queries_per_seed, total_queries // seed_count)

    @staticmethod
    def _empty_query_diagnostics(seed_states: list[SeedState]) -> dict[str, object]:
        return {
            "queries_per_seed": {str(seed_state.seed_index): 0 for seed_state in seed_states},
            "viewport_sizes": {},
            "stages": {"coverage": 0, "adaptive": 0},
            "unique_coverage_by_seed": {str(seed_state.seed_index): 0.0 for seed_state in seed_states},
            "overlap_by_seed": {str(seed_state.seed_index): 0.0 for seed_state in seed_states},
            "deliberate_repeat_overlap": 0.0,
            "accidental_overlap": 0.0,
        }

    def _execute_query(
        self,
        round_id: str,
        seed_index: int,
        candidate,
        observation_store: RoundObservationStore,
        used_per_seed: dict[int, int],
        diagnostics: dict[str, object],
        total_queries: int,
    ) -> bool:
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
            return False

        observation_store.add_simulation_result(round_id, seed_index, result)
        used_per_seed[seed_index] += 1
        self._record_query_diagnostics(
            diagnostics=diagnostics,
            seed_index=seed_index,
            candidate=candidate,
            coverage=observation_store.get_seed_memory(seed_index).observed,
        )
        self.logger.info(
            "Observed seed=%s stage=%s query=%s/%s budget=%s/%s unique=%.3f overlap=%.3f deliberate=%.3f accidental=%.3f sizes=%s",
            seed_index,
            candidate.stage,
            sum(used_per_seed.values()),
            total_queries,
            result.get("queries_used", "?"),
            result.get("queries_max", "?"),
            diagnostics["unique_coverage_by_seed"][str(seed_index)],
            diagnostics["overlap_by_seed"][str(seed_index)],
            diagnostics["deliberate_repeat_overlap"],
            diagnostics["accidental_overlap"],
            diagnostics["viewport_sizes"],
        )
        time.sleep(0.25)
        return True

    def _record_query_diagnostics(
        self,
        diagnostics: dict[str, object],
        seed_index: int,
        candidate,
        coverage,
    ) -> None:
        seed_key = str(seed_index)
        diagnostics["queries_per_seed"][seed_key] += 1
        diagnostics["stages"][candidate.stage] += 1
        size_key = f"{candidate.viewport.w}x{candidate.viewport.h}"
        diagnostics["viewport_sizes"][size_key] = diagnostics["viewport_sizes"].get(size_key, 0) + 1
        diagnostics["deliberate_repeat_overlap"] += float(candidate.intentional_repeat_overlap)
        diagnostics["accidental_overlap"] += float(candidate.accidental_overlap)
        unique_coverage = float((coverage > 0).mean())
        overlap = float(max(float(coverage.sum()) - float((coverage > 0).sum()), 0.0) / coverage.size)
        diagnostics["unique_coverage_by_seed"][seed_key] = unique_coverage
        diagnostics["overlap_by_seed"][seed_key] = overlap

    def _build_analysis_diagnostics(
        self,
        predictions: dict[int, object],
        targets: dict[int, object],
        feature_cache: dict[int, object],
    ) -> dict[str, dict]:
        diagnostics: dict[str, dict] = {}
        for seed_index, target in targets.items():
            prediction = predictions.get(seed_index)
            if prediction is None:
                continue
            calibration = calibration_diagnostics(target, prediction)
            diagnostics[str(seed_index)] = {
                "nll": calibration.nll,
                "brier": calibration.brier,
                "ece": calibration.ece,
                "weighted_kl": calibration.weighted_kl_value,
                "classwise_ece": classwise_expected_calibration_error(target, prediction),
                "bucketed_kl": bucketed_error_diagnostics(target, prediction, feature_cache[seed_index]),
            }
        return diagnostics
