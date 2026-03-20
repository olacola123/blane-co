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


def _default_round_sort_key(round_info) -> tuple[int, str]:
    """Prefer the numerically newest round, then newest timestamp."""
    if not isinstance(round_info, dict):
        return (-1, str(round_info))
    round_number = round_info.get("round_number")
    if not isinstance(round_number, int):
        round_number = -1
    tie_breaker = str(round_info.get("started_at") or round_info.get("event_date") or "")
    return (round_number, tie_breaker)


def _select_default_round_id(rounds: list[dict], budget: dict | None) -> str:
    """Pick the best default round ID from API metadata."""
    if isinstance(budget, dict) and budget.get("active") and budget.get("round_id"):
        budget_round_id = str(budget["round_id"])
        for item in rounds:
            if isinstance(item, dict) and str(item.get("id")) == budget_round_id:
                return budget_round_id

    active_rounds = [
        item
        for item in rounds
        if isinstance(item, dict) and str(item.get("status", "")).lower() == "active" and item.get("id")
    ]
    if active_rounds:
        return str(sorted(active_rounds, key=_default_round_sort_key, reverse=True)[0]["id"])

    newest_round = sorted(rounds, key=_default_round_sort_key, reverse=True)[0]
    return str(newest_round["id"] if isinstance(newest_round, dict) else newest_round)


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
        help="Budget hint per seed when total queries are not set",
    )
    parser.add_argument(
        "--total-queries",
        type=int,
        default=None,
        help=f"Override total round query budget (default: {DEFAULT_TOTAL_QUERIES})",
    )
    parser.add_argument("--prob-floor", type=float, default=0.01, help="Baseline probability floor before renormalization")
    parser.add_argument(
        "--sharp-prob-floor",
        type=float,
        default=0.001,
        help="Lower floor used for observed/static/high-confidence cells",
    )
    parser.add_argument(
        "--sharp-prob-threshold",
        type=float,
        default=0.985,
        help="Confidence threshold for switching to the sharp probability floor",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature scaling factor")
    parser.add_argument(
        "--history-root",
        type=str,
        default="oppgave-3-astar-island/joakim/history",
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
        budget = client.get_budget()
        round_id = _select_default_round_id(rounds, budget)

    return round_id, client.get_round(round_id), client.get_budget()


def run(
    round_id: str | None = None,
    round_file: Path | None = None,
    queries_per_seed: int = 10,
    total_queries: int | None = None,
    prob_floor: float = 0.01,
    sharp_prob_floor: float = 0.001,
    sharp_prob_threshold: float = 0.985,
    temperature: float = 1.0,
    history_root: str = "oppgave-3-astar-island/joakim/history",
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
        log_level=log_level.upper(),
        history_root=history_root,
    )
    config.probability.floor = prob_floor
    config.probability.sharp_floor = sharp_prob_floor
    config.probability.sharp_floor_threshold = sharp_prob_threshold
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
        sharp_prob_floor=args.sharp_prob_floor,
        sharp_prob_threshold=args.sharp_prob_threshold,
        temperature=args.temperature,
        history_root=args.history_root,
        submit=not args.no_submit,
        fetch_analysis_only=args.fetch_analysis_only,
        dry_run=args.dry_run,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
