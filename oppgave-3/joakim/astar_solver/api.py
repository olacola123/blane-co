"""REST client for the Astar Island endpoints."""

from __future__ import annotations

import os
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class AstarClient:
    """Small API wrapper for round retrieval, simulation, and submission."""

    BASE_URL = "https://api.ainm.no/astar-island"

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("API_KEY", "")
        if not self.token:
            raise ValueError("API_KEY not set. Use: export API_KEY='your-jwt-token'")

        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.token}",
            }
        )

    def get_rounds(self) -> list[dict]:
        response = self.session.get(f"{self.BASE_URL}/rounds")
        response.raise_for_status()
        return response.json()

    def get_round(self, round_id: str) -> dict:
        response = self.session.get(f"{self.BASE_URL}/rounds/{round_id}")
        response.raise_for_status()
        return response.json()

    def get_budget(self) -> dict:
        response = self.session.get(f"{self.BASE_URL}/budget")
        response.raise_for_status()
        return response.json()

    def simulate(self, round_id: str, seed_index: int, x: int, y: int, w: int, h: int) -> dict:
        payload = {
            "round_id": round_id,
            "seed_index": seed_index,
            "viewport_x": x,
            "viewport_y": y,
            "viewport_w": w,
            "viewport_h": h,
        }
        response = self.session.post(f"{self.BASE_URL}/simulate", json=payload)
        response.raise_for_status()
        return response.json()

    def submit(self, round_id: str, seed_index: int, prediction: list) -> dict:
        payload = {
            "round_id": round_id,
            "seed_index": seed_index,
            "prediction": prediction,
        }
        response = self.session.post(f"{self.BASE_URL}/submit", json=payload)
        response.raise_for_status()
        return response.json()

    def get_my_rounds(self) -> dict:
        response = self.session.get(f"{self.BASE_URL}/my-rounds")
        response.raise_for_status()
        return response.json()

    def get_analysis(self, round_id: str, seed_index: int) -> dict:
        response = self.session.get(f"{self.BASE_URL}/analysis/{round_id}/{seed_index}")
        response.raise_for_status()
        return response.json()
