"""Modular Astar Island solver package for oppgave 3."""

from .config import SolverConfig

__all__ = ["AstarClient", "RoundSolver", "SolverConfig"]


def __getattr__(name: str):
    """Lazy package exports so utility modules work without optional deps."""
    if name == "AstarClient":
        from .api import AstarClient

        return AstarClient
    if name == "RoundSolver":
        from .pipeline import RoundSolver

        return RoundSolver
    raise AttributeError(name)
