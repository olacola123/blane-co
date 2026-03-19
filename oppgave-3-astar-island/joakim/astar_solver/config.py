"""Configuration dataclasses for the solver."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ProbabilityConfig:
    """Numerical controls for calibrated probabilistic outputs."""

    floor: float = 0.01
    temperature: float = 1.0


@dataclass(slots=True)
class QueryConfig:
    """Heuristic query selector configuration."""

    min_viewport: int = 5
    max_viewport: int = 15
    uncertainty_weight: float = 1.0
    dynamic_weight: float = 1.25
    frontier_weight: float = 0.85
    coverage_penalty: float = 0.6
    duplicate_penalty: float = 0.35
    candidate_limit: int = 48


@dataclass(slots=True)
class SolverConfig:
    """Top-level solver configuration."""

    probability: ProbabilityConfig = field(default_factory=ProbabilityConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    local_dynamics_passes: int = 2
    observation_blend: float = 0.85
    latent_strength: float = 0.65
    log_level: str = "INFO"
    history_root: str = "oppgave-3/joakim/history"
