"""Configuration dataclasses for the solver."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ProbabilityConfig:
    """Numerical controls for calibrated probabilistic outputs."""

    floor: float = 0.01
    temperature: float = 1.0
    enable_temperature_scaling: bool = True
    enable_class_calibration: bool = True
    class_bias: tuple[float, ...] = (
        0.18,
        0.0,
        -0.45,
        -0.35,
        -0.12,
        0.06,
    )
    class_temperature: tuple[float, ...] = (
        0.82,
        0.94,
        0.86,
        0.88,
        0.92,
        0.78,
    )


@dataclass(slots=True)
class ModelConfig:
    """Heuristic controls for geography priors, gates, and observation updates."""

    enable_structured_gates: bool = True
    enable_shared_round_priors: bool = True
    geography_prior_strength: float = 1.75
    shared_round_prior_strength: float = 0.85
    observation_prior_strength: float = 3.5
    settlement_influence_strength: float = 1.2
    history_tuning_enabled: bool = True
    history_tuning_weight: float = 0.65
    history_tuning_round_limit: int = 8


@dataclass(slots=True)
class QueryConfig:
    """Heuristic query selector configuration."""

    min_viewport: int = 5
    max_viewport: int = 15
    minimum_queries_per_seed: int = 6
    large_viewport: int = 15
    medium_viewport: int = 11
    small_viewport: int = 7
    novelty_weight: float = 1.35
    information_weight: float = 1.10
    uncertainty_weight: float = 1.0
    dynamic_weight: float = 1.25
    frontier_weight: float = 0.85
    repeat_value_weight: float = 0.55
    coverage_penalty: float = 0.35
    duplicate_penalty: float = 0.35
    accidental_overlap_penalty: float = 0.95
    deliberate_repeat_after: int = 4
    repeat_target_coverage: float = 2.0
    max_repeat_share: float = 0.22
    candidate_limit: int = 36


@dataclass(slots=True)
class SolverConfig:
    """Top-level solver configuration."""

    probability: ProbabilityConfig = field(default_factory=ProbabilityConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    local_dynamics_passes: int = 2
    latent_strength: float = 0.65
    log_level: str = "INFO"
    history_root: str = "oppgave-3/joakim/history"
