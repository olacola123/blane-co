"""Adaptive query scoring and viewport selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .config import QueryConfig
from .types import SeedState, Viewport

if False:  # pragma: no cover
    from .predictor import PredictionArtifacts


@dataclass(slots=True)
class QueryCandidate:
    """A scored viewport candidate with diagnostic breakdown."""

    viewport: Viewport
    score: float
    reason: str
    origin: str
    novelty: float
    information_gain: float
    frontier: float
    repeat_value: float
    accidental_overlap: float
    intentional_repeat_overlap: float
    stage: str


class HeuristicQuerySelector:
    """Rank viewports by coverage need, information value, and strategic repeats."""

    def __init__(self, config: QueryConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def select_next(
        self,
        seed_state: SeedState,
        artifacts: "PredictionArtifacts",
        coverage: np.ndarray,
        queries_used_for_seed: int,
        stage: str,
        global_progress: float = 0.0,
        already_planned: list[Viewport] | None = None,
    ) -> QueryCandidate | None:
        """Select the next best query for one seed and one planning stage."""
        candidates = self._generate_candidates(
            seed_state=seed_state,
            artifacts=artifacts,
            coverage=coverage,
            queries_used_for_seed=queries_used_for_seed,
            stage=stage,
            global_progress=global_progress,
        )
        if not candidates:
            return None

        temp_coverage = np.array(coverage, dtype=float, copy=True)
        for viewport in already_planned or []:
            temp_coverage[
                viewport.y : viewport.y + viewport.h,
                viewport.x : viewport.x + viewport.w,
            ] += 1.0

        best: QueryCandidate | None = None
        for origin, viewport in candidates:
            candidate = self._score_candidate(
                viewport=viewport,
                artifacts=artifacts,
                coverage=temp_coverage,
                queries_used_for_seed=queries_used_for_seed,
                stage=stage,
                origin=origin,
            )
            if best is None or candidate.score > best.score:
                best = candidate

        if best is not None:
            self.logger.info(
                "Selected viewport seed=%s stage=%s x=%s y=%s w=%s h=%s score=%.4f origin=%s reason=%s",
                seed_state.seed_index,
                best.stage,
                best.viewport.x,
                best.viewport.y,
                best.viewport.w,
                best.viewport.h,
                best.score,
                best.origin,
                best.reason,
            )
        return best

    def _generate_candidates(
        self,
        seed_state: SeedState,
        artifacts: "PredictionArtifacts",
        coverage: np.ndarray,
        queries_used_for_seed: int,
        stage: str,
        global_progress: float,
    ) -> list[tuple[str, Viewport]]:
        sizes = self._candidate_sizes(
            stage=stage,
            queries_used_for_seed=queries_used_for_seed,
            global_progress=global_progress,
        )
        candidates: list[tuple[str, Viewport]] = []
        seen: set[tuple[int, int, int, int]] = set()

        def add(center_x: int, center_y: int, size: int, origin: str) -> None:
            size = int(np.clip(size, self.config.min_viewport, self.config.max_viewport))
            viewport = Viewport.centered(center_x, center_y, size)
            key = (viewport.x, viewport.y, viewport.w, viewport.h)
            if key in seen:
                return
            seen.add(key)
            candidates.append((origin, viewport))

        for settlement in seed_state.settlements:
            for size in sizes:
                add(settlement.x, settlement.y, size, "seed-settlement")

        frontier = artifacts.dynamic_mass * (
            0.50 * artifacts.relations.frontier_pressure
            + 0.25 * artifacts.relations.conflict_risk
            + 0.25 * artifacts.relations.trade_access
        )
        coverage_need = np.exp(-coverage)
        hole_priority = coverage_need * (
            0.55 * artifacts.entropy_map + 0.25 * frontier + 0.20 * artifacts.dynamic_mass
        )
        repeat_window = np.clip(coverage, 0.0, self.config.repeat_target_coverage)
        repeat_priority = (repeat_window / max(self.config.repeat_target_coverage, 1e-6)) * (
            0.45 * artifacts.entropy_map + 0.35 * artifacts.dynamic_mass + 0.20 * frontier
        )

        anchor_maps: list[tuple[str, np.ndarray]] = [
            ("entropy", artifacts.entropy_map * (0.35 + 0.65 * coverage_need)),
            ("frontier", frontier * (0.35 + 0.65 * coverage_need)),
            ("holes", hole_priority),
        ]
        if stage == "adaptive":
            anchor_maps.append(("repeat", repeat_priority))

        for origin, score_map in anchor_maps:
            flat_indices = np.argsort(score_map.ravel())[::-1][: self.config.candidate_limit]
            width = score_map.shape[1]
            for index in flat_indices:
                y = int(index // width)
                x = int(index % width)
                for size in sizes:
                    add(x, y, size, origin)

        for size in sizes:
            add(artifacts.entropy_map.shape[1] // 2, artifacts.entropy_map.shape[0] // 2, size, "center")
        return candidates

    def _candidate_sizes(
        self,
        stage: str,
        queries_used_for_seed: int,
        global_progress: float,
    ) -> tuple[int, ...]:
        if stage == "coverage":
            if queries_used_for_seed <= 1:
                return (self.config.large_viewport, self.config.medium_viewport)
            return (self.config.large_viewport, self.config.medium_viewport, self.config.small_viewport)
        if global_progress >= 0.80:
            return (self.config.small_viewport, self.config.medium_viewport)
        return (self.config.medium_viewport, self.config.small_viewport, self.config.large_viewport)

    def _score_candidate(
        self,
        viewport: Viewport,
        artifacts: "PredictionArtifacts",
        coverage: np.ndarray,
        queries_used_for_seed: int,
        stage: str,
        origin: str,
    ) -> QueryCandidate:
        region = np.s_[viewport.y : viewport.y + viewport.h, viewport.x : viewport.x + viewport.w]
        region_coverage = coverage[region]
        uncertainty = float(artifacts.entropy_map[region].mean())
        dynamic = float(artifacts.dynamic_mass[region].mean())
        frontier = float(
            (
                artifacts.relations.frontier_pressure[region]
                + artifacts.relations.conflict_risk[region]
                + artifacts.relations.trade_access[region]
            ).mean()
            / 3.0
        )
        novelty = float(np.exp(-region_coverage).mean())
        overlap_fraction = float((region_coverage > 0.0).mean())
        repeat_window = np.clip(region_coverage, 0.0, self.config.repeat_target_coverage)
        repeat_support = float((repeat_window / max(self.config.repeat_target_coverage, 1e-6)).mean())
        repeat_value = repeat_support * (
            0.55 * uncertainty + 0.25 * dynamic + 0.20 * frontier
        )
        if stage == "coverage" or queries_used_for_seed < self.config.deliberate_repeat_after:
            repeat_value *= 0.25

        intentional_repeat_overlap = min(overlap_fraction, min(repeat_value, self.config.max_repeat_share))
        accidental_overlap = max(overlap_fraction - intentional_repeat_overlap, 0.0)
        excess_overlap = max(overlap_fraction - self.config.max_repeat_share, 0.0)
        information_gain = (
            0.50 * uncertainty * (0.45 + 0.55 * novelty)
            + 0.25 * dynamic
            + 0.25 * frontier
        )
        coverage_penalty = float(
            np.clip(region_coverage, 0.0, self.config.repeat_target_coverage).mean()
            / max(self.config.repeat_target_coverage, 1e-6)
        )

        novelty_scale = 1.20 if stage == "coverage" else 1.0
        repeat_scale = 0.25 if stage == "coverage" else 1.0
        score = (
            self.config.novelty_weight * novelty_scale * novelty
            + self.config.information_weight * information_gain
            + self.config.uncertainty_weight * uncertainty
            + self.config.dynamic_weight * dynamic
            + self.config.frontier_weight * frontier
            + self.config.repeat_value_weight * repeat_scale * repeat_value
            - self.config.coverage_penalty * coverage_penalty
            - self.config.duplicate_penalty * excess_overlap
            - self.config.accidental_overlap_penalty * accidental_overlap
        )
        if stage == "adaptive" and repeat_value > 0.55 * novelty:
            score += 0.12 * intentional_repeat_overlap

        reason = (
            f"nov={novelty:.3f} info={information_gain:.3f} unc={uncertainty:.3f} "
            f"front={frontier:.3f} rep={repeat_value:.3f} acc={accidental_overlap:.3f}"
        )
        return QueryCandidate(
            viewport=viewport,
            score=score,
            reason=reason,
            origin=origin,
            novelty=novelty,
            information_gain=information_gain,
            frontier=frontier,
            repeat_value=repeat_value,
            accidental_overlap=accidental_overlap,
            intentional_repeat_overlap=intentional_repeat_overlap,
            stage=stage,
        )
