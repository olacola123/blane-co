"""Modular query scoring and viewport selection."""

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
    """A scored viewport candidate."""

    viewport: Viewport
    score: float
    reason: str


class HeuristicQuerySelector:
    """Rank viewports by uncertainty, dynamic-zone prior, and current coverage."""

    def __init__(self, config: QueryConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def select_next(
        self,
        seed_state: SeedState,
        artifacts: "PredictionArtifacts",
        coverage: np.ndarray,
        already_planned: list[Viewport] | None = None,
    ) -> QueryCandidate | None:
        """Select the next best query for a seed."""
        candidates = self._generate_candidates(seed_state, artifacts)
        if not candidates:
            return None

        temp_coverage = np.array(coverage, dtype=float, copy=True)
        for viewport in already_planned or []:
            temp_coverage[
                viewport.y : viewport.y + viewport.h,
                viewport.x : viewport.x + viewport.w,
            ] += 1.0

        best: QueryCandidate | None = None
        for viewport in candidates:
            candidate = self._score_candidate(viewport, artifacts, temp_coverage)
            if best is None or candidate.score > best.score:
                best = candidate

        if best is not None:
            self.logger.info(
                "Selected viewport seed=%s x=%s y=%s w=%s h=%s score=%.4f reason=%s",
                seed_state.seed_index,
                best.viewport.x,
                best.viewport.y,
                best.viewport.w,
                best.viewport.h,
                best.score,
                best.reason,
            )
        return best

    def _generate_candidates(
        self,
        seed_state: SeedState,
        artifacts: "PredictionArtifacts",
    ) -> list[Viewport]:
        size = self.config.max_viewport
        candidates: list[Viewport] = []
        seen: set[tuple[int, int, int, int]] = set()

        def add(center_x: int, center_y: int) -> None:
            viewport = Viewport.centered(center_x, center_y, size)
            key = (viewport.x, viewport.y, viewport.w, viewport.h)
            if key in seen:
                return
            seen.add(key)
            candidates.append(viewport)

        for settlement in seed_state.settlements:
            add(settlement.x, settlement.y)

        entropy = artifacts.entropy_map
        frontier = artifacts.dynamic_mass * (
            0.5 * artifacts.relations.frontier_pressure
            + 0.3 * artifacts.relations.conflict_risk
            + 0.2 * artifacts.relations.trade_access
        )
        for score_map in (entropy, frontier):
            flat_indices = np.argsort(score_map.ravel())[::-1][: self.config.candidate_limit]
            width = score_map.shape[1]
            for index in flat_indices:
                y = int(index // width)
                x = int(index % width)
                add(x, y)

        add(entropy.shape[1] // 2, entropy.shape[0] // 2)
        return candidates

    def _score_candidate(
        self,
        viewport: Viewport,
        artifacts: "PredictionArtifacts",
        coverage: np.ndarray,
    ) -> QueryCandidate:
        region = np.s_[viewport.y : viewport.y + viewport.h, viewport.x : viewport.x + viewport.w]
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
        coverage_penalty = float(np.clip(coverage[region], 0.0, 3.0).mean() / 3.0)
        score = (
            self.config.uncertainty_weight * uncertainty
            + self.config.dynamic_weight * dynamic
            + self.config.frontier_weight * frontier
            - self.config.coverage_penalty * coverage_penalty
        )
        reason = (
            f"unc={uncertainty:.3f} dyn={dynamic:.3f} "
            f"frontier={frontier:.3f} cov={coverage_penalty:.3f}"
        )
        return QueryCandidate(viewport=viewport, score=score, reason=reason)
