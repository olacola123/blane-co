"""FastAPI endpoint for Astar Island — Cloud Run deployment."""

from __future__ import annotations

import logging
import os
import traceback

from fastapi import FastAPI

from astar_solver import AstarClient, RoundSolver, SolverConfig
from astar_solver.constants import DEFAULT_TOTAL_QUERIES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/solve")
async def solve(request: dict):
    """Solve an Astar Island round.

    The competition validators call this endpoint.  Accepts:
      - round_id (optional): specific round to solve; defaults to latest.
      - queries_per_seed (optional): queries per seed, default 10.
      - total_queries (optional): total query budget override.
      - prob_floor (optional): probability floor, default 0.01.
      - temperature (optional): temperature scaling, default 1.0.
    """
    try:
        round_id = request.get("round_id")
        queries_per_seed = int(request.get("queries_per_seed", 10))
        total_queries = request.get("total_queries")
        if total_queries is not None:
            total_queries = int(total_queries)
        prob_floor = float(request.get("prob_floor", 0.01))
        temperature = float(request.get("temperature", 1.0))

        config = SolverConfig(
            probability=SolverConfig().probability,
            query=SolverConfig().query,
            local_dynamics_passes=SolverConfig().local_dynamics_passes,
            observation_blend=SolverConfig().observation_blend,
            latent_strength=SolverConfig().latent_strength,
            log_level="INFO",
            history_root="/tmp/history",
        )
        config.probability.floor = prob_floor
        config.probability.temperature = temperature

        client = AstarClient(token=os.environ.get("API_KEY", ""))

        # Resolve round
        if not round_id:
            rounds = client.get_rounds()
            if not rounds:
                return {"status": "error", "message": "No rounds available"}
            latest = rounds[-1]
            round_id = latest["id"] if isinstance(latest, dict) else latest

        round_data = client.get_round(round_id)
        budget = client.get_budget()
        logger.info("Solving round %s — budget: %s", round_id, budget)

        solver = RoundSolver(client=client, config=config)
        artifacts = solver.solve_round(
            round_id=round_id,
            round_data=round_data,
            queries_per_seed=queries_per_seed,
            total_queries=total_queries or DEFAULT_TOTAL_QUERIES,
            submit=True,
        )

        summary = {
            str(seed_index): {
                "mean_entropy": float(artifact.entropy_map.mean()),
                "mean_dynamic_mass": float(artifact.dynamic_mass.mean()),
            }
            for seed_index, artifact in artifacts.items()
        }

        logger.info("Round %s solved: %s", round_id, summary)
        return {"status": "completed", "round_id": round_id, "seeds": summary}

    except Exception as exc:
        logger.error("Solve failed: %s\n%s", exc, traceback.format_exc())
        return {"status": "error", "message": str(exc)}
