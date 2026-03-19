"""
Oppgave 3 — Astar Island probabilistic round solver.

The implementation is intentionally modular:
    - geography feature extraction
    - shared round observation memory
    - pooled latent inference z_round
    - settlement relation maps
    - local dynamics refinement
    - calibrated probabilistic decoding
    - heuristic query selection
    - history storage for later training
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from astar_solver import SolverConfig
from astar_solver.constants import DEFAULT_TOTAL_QUERIES


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI definition."""
    parser = argparse.ArgumentParser(description="Astar Island probabilistic solver")
    parser.add_argument("--round", type=str, default=None, help="Round ID to solve")
    parser.add_argument(
        "--round-file",
        type=Path,
        default=None,
        help="Optional local JSON round file for offline debugging",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=10,
        help="Planned queries per seed when total queries are not set",
    )
    parser.add_argument(
        "--total-queries",
        type=int,
        default=None,
        help=f"Override total round query budget (default: {DEFAULT_TOTAL_QUERIES})",
    )
    parser.add_argument("--prob-floor", type=float, default=0.01, help="Probability floor before renormalization")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature scaling factor")
    parser.add_argument(
        "--history-root",
        type=str,
        default="oppgave-3/joakim/history",
        help="Where to persist round artifacts",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--no-submit", action="store_true", help="Predict without submitting")
    parser.add_argument(
        "--fetch-analysis-only",
        action="store_true",
        help="Fetch analysis for an already submitted round and update local history",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip simulate and submit calls; fetch round data unless --round-file is used",
    )
    return parser


def load_round_data(client, round_id: str | None, round_file: Path | None) -> tuple[str, dict, dict | None]:
    """Load round data either from disk or the live API."""
    if round_file is not None:
        payload = json.loads(round_file.read_text())
        inferred_round_id = round_id or payload.get("id") or payload.get("round_id") or round_file.stem
        return inferred_round_id, payload, None

    if client is None:
        raise ValueError("client is required when round data is not loaded from file")

    if not round_id:
        rounds = client.get_rounds()
        if not rounds:
            raise ValueError("no rounds available")
        latest_round = rounds[-1]
        round_id = latest_round["id"] if isinstance(latest_round, dict) else latest_round

    return round_id, client.get_round(round_id), client.get_budget()


def run(
    round_id: str | None = None,
    round_file: Path | None = None,
    queries_per_seed: int = 10,
    total_queries: int | None = None,
    prob_floor: float = 0.01,
    temperature: float = 1.0,
    history_root: str = "oppgave-3/joakim/history",
    submit: bool = True,
    fetch_analysis_only: bool = False,
    dry_run: bool = False,
    log_level: str = "INFO",
):
    """Main programmatic entrypoint."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = SolverConfig(
        probability=SolverConfig().probability,
        query=SolverConfig().query,
        local_dynamics_passes=SolverConfig().local_dynamics_passes,
        observation_blend=SolverConfig().observation_blend,
        latent_strength=SolverConfig().latent_strength,
        log_level=log_level.upper(),
        history_root=history_root,
    )
    config.probability.floor = prob_floor
    config.probability.temperature = temperature

    client = None
    if round_file is None:
        from astar_solver import AstarClient

        client = AstarClient()
    active_round_id, round_data, budget = load_round_data(client, round_id, round_file)

    print(f"Round: {active_round_id}")
    if budget is not None:
        print(f"Budget: {json.dumps(budget, indent=2)}")

    if client is None and not dry_run:
        raise ValueError("submit/simulate requires live API access; provide no --round-file or use --dry-run")

    from astar_solver import RoundSolver

    solver = RoundSolver(client=client, config=config)

    if fetch_analysis_only:
        seed_payloads = round_data.get("seeds", round_data.get("initial_states", []))
        analyses = solver.fetch_analyses(
            round_id=active_round_id,
            seed_indices=list(range(len(seed_payloads))),
        )
        if analyses:
            solver.history_store.update_round_analyses(active_round_id, analyses)
        print(json.dumps({"round_id": active_round_id, "analysis_seeds": sorted(analyses)}, indent=2))
        return analyses

    artifacts = solver.solve_round(
        round_id=active_round_id,
        round_data=round_data,
        queries_per_seed=queries_per_seed,
        total_queries=total_queries or DEFAULT_TOTAL_QUERIES,
        submit=submit and not dry_run,
        dry_run=dry_run,
    )

    summary = {
        seed_index: {
            "mean_entropy": float(artifact.entropy_map.mean()),
            "mean_dynamic_mass": float(artifact.dynamic_mass.mean()),
        }
        for seed_index, artifact in artifacts.items()
    }
    print(json.dumps(summary, indent=2))

    return {seed_index: artifact.probabilities for seed_index, artifact in artifacts.items()}


def main() -> None:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args()
    run(
        round_id=args.round,
        round_file=args.round_file,
        queries_per_seed=args.queries,
        total_queries=args.total_queries,
        prob_floor=args.prob_floor,
        temperature=args.temperature,
        history_root=args.history_root,
        submit=not args.no_submit,
        fetch_analysis_only=args.fetch_analysis_only,
        dry_run=args.dry_run,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
