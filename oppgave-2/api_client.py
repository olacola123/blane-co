"""
NM i AI 2026 — REST API Client Template
=========================================
Generisk API-klient for a hente oppgaver og sende inn losninger.

Slik fungerer API-scoring:
1. Hent oppgave/data fra API (GET)
2. Kjor din modell pa dataen
3. Send inn losning (POST)
4. Fol score tilbake

Bruk:
1. Sett BASE_URL og API_KEY
2. Tilpass endpoints og dataformat
3. Kjor: python api_client.py

TODO-markerte steder ma tilpasses for den spesifikke oppgaven.
"""

import json
import os
import time
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# === KONFIGURASJON ===

BASE_URL = os.environ.get("API_URL", "https://api.ainm.no")  # TODO: Riktig base URL
API_KEY = os.environ.get("API_KEY", "")  # TODO: Din API-nokkel


# === API-KLIENT ===

class APIClient:
    """
    Robust API-klient med retry-logikk og feilhandtering.
    """

    def __init__(self, base_url: str = BASE_URL, api_key: str = API_KEY):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

        # Session med retry-logikk
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.mount("http://", HTTPAdapter(max_retries=retries))

        # Standard headers
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

        # TODO: Tilpass autentisering
        if self.api_key:
            self.session.headers["Authorization"] = f"Bearer {self.api_key}"
            # Alternativ: self.session.headers["X-API-Key"] = self.api_key

    def get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Send GET-request."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def post(self, endpoint: str, data: Any) -> dict:
        """Send POST-request med JSON-data."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return response.json()

    def put(self, endpoint: str, data: Any) -> dict:
        """Send PUT-request med JSON-data."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.put(url, json=data)
        response.raise_for_status()
        return response.json()


# === OPPGAVE-SPESIFIKKE METODER ===

class CompetitionClient(APIClient):
    """
    Utvidet klient med metoder for typiske konkurranseoperasjoner.

    TODO: Tilpass alle metodene til oppgavens API-spesifikasjon.
    """

    def get_task(self, task_id: str = "") -> dict:
        """Hent oppgavebeskrivelse og data."""
        # TODO: Riktig endpoint
        return self.get(f"/tasks/{task_id}" if task_id else "/task")

    def get_data(self, task_id: str = "", split: str = "train") -> dict:
        """Hent trenings- eller testdata."""
        # TODO: Riktig endpoint
        return self.get(f"/tasks/{task_id}/data", params={"split": split})

    def submit_solution(self, task_id: str, solution: Any) -> dict:
        """
        Send inn losning og fol score.

        Args:
            task_id: ID for oppgaven
            solution: Losningen (format avhenger av oppgaven)

        Returns:
            Respons med score og feedback
        """
        # TODO: Tilpass payload-format
        payload = {
            "task_id": task_id,
            "solution": solution,
        }

        response = self.post(f"/tasks/{task_id}/submit", payload)

        score = response.get("score", response.get("result", {}).get("score", "ukjent"))
        print(f"  Submission for oppgave {task_id}: score = {score}")

        return response

    def get_leaderboard(self, task_id: str = "") -> list[dict]:
        """Hent leaderboard."""
        # TODO: Riktig endpoint
        endpoint = f"/tasks/{task_id}/leaderboard" if task_id else "/leaderboard"
        data = self.get(endpoint)
        return data.get("entries", data.get("leaderboard", data))

    def get_my_submissions(self, task_id: str = "") -> list[dict]:
        """Hent mine tidligere submissions."""
        # TODO: Riktig endpoint
        endpoint = f"/tasks/{task_id}/submissions" if task_id else "/submissions"
        return self.get(endpoint)

    def start_game(self, task_id: str, **kwargs) -> dict:
        """
        Start et nytt spill/forsok (for interaktive oppgaver).
        """
        # TODO: Tilpass til oppgavens startprosedyre
        return self.post(f"/tasks/{task_id}/start", kwargs)


# === ITERATIV SUBMISSION ===

def iterative_submit(
    client: CompetitionClient,
    task_id: str,
    solve_fn,  # Funksjon: (data) -> solution
    n_attempts: int = 5,
    delay: float = 2.0,
):
    """
    Iterativ forbedring: kjor losningen flere ganger og submit den beste.

    Nyttig for stokastiske metoder eller nol du vil prove ulike parametre.
    """
    print(f"\n=== Iterativ submission ({n_attempts} forsok) ===")

    # Hent data
    data = client.get_data(task_id)

    best_score = float("-inf")
    best_solution = None

    for attempt in range(n_attempts):
        print(f"\nForsok {attempt + 1}/{n_attempts}:")

        # Generer losning
        solution = solve_fn(data)

        # Submit
        result = client.submit_solution(task_id, solution)
        score = result.get("score", 0)

        if score > best_score:
            best_score = score
            best_solution = solution
            print(f"  NY BESTE: {score}")

        # Vent mellom submissions (respekter rate limits)
        if attempt < n_attempts - 1:
            time.sleep(delay)

    print(f"\nBeste score: {best_score}")
    return best_solution, best_score


# === BATCH-SUBMISSION ===

def batch_submit(
    client: CompetitionClient,
    task_id: str,
    solutions: list[Any],
    labels: Optional[list[str]] = None,
):
    """
    Submit flere losninger og sammenlign scores.

    Nyttig nol du vil teste ulike tilnærminger side om side.
    """
    print(f"\n=== Batch submission ({len(solutions)} losninger) ===")

    results = []
    for i, solution in enumerate(solutions):
        label = labels[i] if labels else f"Losning {i+1}"
        print(f"\n{label}:")

        result = client.submit_solution(task_id, solution)
        score = result.get("score", 0)
        results.append({"label": label, "score": score, "result": result})

        time.sleep(1)  # Rate limiting

    # Sorter etter score
    results.sort(key=lambda r: r["score"], reverse=True)

    print(f"\n=== Rangering ===")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['label']}: {r['score']}")

    return results


# === LEADERBOARD-MONITOR ===

def monitor_leaderboard(client: CompetitionClient, task_id: str = "", interval: int = 60):
    """
    Overvol leaderboard og vis endringer.

    Kjor i bakgrunnen for a holde oye med konkurrentene.
    """
    print(f"Overvoler leaderboard (oppdaterer hvert {interval}s, Ctrl+C for a stoppe)")

    prev_entries = {}

    try:
        while True:
            entries = client.get_leaderboard(task_id)

            print(f"\n[{time.strftime('%H:%M:%S')}] Leaderboard:")
            for i, entry in enumerate(entries[:10]):
                name = entry.get("name", entry.get("team", f"Lag {i+1}"))
                score = entry.get("score", 0)

                # Vis endring
                prev_score = prev_entries.get(name, score)
                change = score - prev_score
                change_str = f" (+{change:.1f})" if change > 0 else ""

                print(f"  {i+1:2d}. {name:20s} {score:8.1f}{change_str}")
                prev_entries[name] = score

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nOvervoking stoppet")


# === KJOR ===

if __name__ == "__main__":
    # Sjekk konfigurasjon
    if not API_KEY:
        print("ADVARSEL: API_KEY ikke satt!")
        print("Sett den med: export API_KEY='din-nokkel-her'\n")

    # Opprett klient
    client = CompetitionClient()

    # TODO: Endre til faktiske oppgave-IDer og losninger

    # Eksempel: Hent oppgave
    # task = client.get_task("task-1")
    # print(json.dumps(task, indent=2))

    # Eksempel: Submit losning
    # result = client.submit_solution("task-1", {"prediction": [1, 0, 1, 1, 0]})

    # Eksempel: Se leaderboard
    # entries = client.get_leaderboard()
    # for entry in entries[:10]:
    #     print(f"  {entry['name']}: {entry['score']}")

    # Eksempel: Iterativ submission
    # def my_solver(data):
    #     # TODO: Din losningslogikk her
    #     return {"prediction": [1, 0, 1]}
    #
    # best_solution, best_score = iterative_submit(client, "task-1", my_solver, n_attempts=5)

    print("API-klient klar. Tilpass endpoints og kjor!")
