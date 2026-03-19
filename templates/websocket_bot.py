"""
NM i AI 2026 — WebSocket Bot Template
=======================================
Generisk WebSocket-bot som kobler til en server, mottar state, sender actions.

Basert pa erfaring fra Grocery Bot-oppgaven.

Slik fungerer det:
1. Bot kobler til server via WebSocket
2. Server sender GAME STATE (JSON) hver runde
3. Bot analyserer state og velger ACTIONS
4. Bot sender actions tilbake til server
5. Server scorer og sender ny state

Bruk:
1. Sett WS_URL og API_KEY
2. Implementer decide_actions() med din strategi
3. Kjor: python websocket_bot.py

TODO-markerte steder ma tilpasses for den spesifikke oppgaven.
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import websockets


# === KONFIGURASJON ===

# TODO: Sett disse fra miljovariabler eller oppgave-spesifikasjonen
WS_URL = os.environ.get("WS_URL", "wss://api.ainm.no/ws")  # TODO: Riktig URL
API_KEY = os.environ.get("API_KEY", "")  # TODO: Din API-nokkel
GAME_ID = os.environ.get("GAME_ID", "")  # TODO: Spill-ID hvis pakrevd


# === GAME STATE ===

@dataclass
class GameState:
    """
    Representerer spill-tilstanden fra serveren.

    TODO: Tilpass feltene til oppgavens state-format.
    """
    round_number: int = 0
    max_rounds: int = 500
    score: float = 0.0

    # TODO: Legg til oppgave-spesifikke felt
    # Eksempler:
    # agents: list = field(default_factory=list)
    # grid: list = field(default_factory=list)
    # items: list = field(default_factory=list)
    # orders: list = field(default_factory=list)

    raw_data: dict = field(default_factory=dict)  # Ra JSON-data

    @classmethod
    def from_json(cls, data: dict) -> "GameState":
        """Parse JSON fra serveren til GameState."""
        return cls(
            round_number=data.get("round", data.get("tick", 0)),
            max_rounds=data.get("max_rounds", data.get("max_ticks", 500)),
            score=data.get("score", 0.0),
            raw_data=data,
        )


# === STRATEGI ===

def decide_actions(state: GameState) -> dict:
    """
    Hovedstrategien: analyser state og bestem actions.

    TODO: DETTE er der du implementerer logikken!

    Returnerer en dict som sendes til serveren som JSON.
    """
    actions = {}

    # TODO: Implementer strategien basert pa oppgaven
    # Eksempler:

    # --- For agent-basert oppgave (som grocery bot) ---
    # agents = state.raw_data.get("agents", [])
    # for agent in agents:
    #     agent_id = agent["id"]
    #     position = agent["position"]
    #     # Bestem handling basert pa posisjon, mal, etc.
    #     actions[agent_id] = {"action": "move_right"}

    # --- For turbasert spill ---
    # actions = {"move": "north", "attack": False}

    # --- For prediksjonsoppgave ---
    # actions = {"prediction": 42.0}

    return actions


# === WEBSOCKET-KLIENT ===

class WebSocketBot:
    """
    Generisk WebSocket-bot som handler kommunikasjon med serveren.
    """

    def __init__(self, url: str = WS_URL, api_key: str = API_KEY):
        self.url = url
        self.api_key = api_key
        self.ws: Optional[websockets.WebSocketClientProtocol] = None

        # Statistikk
        self.rounds_played = 0
        self.total_score = 0.0
        self.start_time = 0.0
        self.round_times: list[float] = []

    async def connect(self):
        """Koble til WebSocket-serveren."""
        # TODO: Tilpass headers/parametre til oppgavens autentisering
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Noen servere bruker query parameters i stedet
        url = self.url
        if GAME_ID:
            url = f"{url}?game_id={GAME_ID}"

        print(f"Kobler til {url}...")
        self.ws = await websockets.connect(
            url,
            extra_headers=headers,
            ping_interval=20,
            ping_timeout=60,
        )
        print("Tilkoblet!")

    async def send_join(self):
        """
        Send join/init-melding hvis serveren krever det.

        TODO: Tilpass til oppgavens protokoll.
        """
        join_message = {
            "type": "join",
            "api_key": self.api_key,
            # TODO: Legg til andre felt som trengs
        }
        await self.ws.send(json.dumps(join_message))
        print("Join-melding sendt")

    async def game_loop(self):
        """Hovedloop: motta state -> bestem actions -> send actions."""
        self.start_time = time.time()

        try:
            async for message in self.ws:
                round_start = time.time()

                # Parse melding
                data = json.loads(message)

                # TODO: Sjekk meldingstype
                msg_type = data.get("type", data.get("event", "state"))

                if msg_type == "error":
                    print(f"FEIL fra server: {data}")
                    continue

                if msg_type in ("game_over", "end", "finished"):
                    final_score = data.get("score", data.get("final_score", 0))
                    print(f"\n=== Spill ferdig! Sluttpoeng: {final_score} ===")
                    break

                if msg_type in ("state", "tick", "round", "update"):
                    state = GameState.from_json(data)

                    # Bestem og send actions
                    actions = decide_actions(state)
                    await self.ws.send(json.dumps(actions))

                    # Statistikk
                    round_time = (time.time() - round_start) * 1000
                    self.round_times.append(round_time)
                    self.rounds_played += 1

                    # Periodisk logging
                    if self.rounds_played % 50 == 0 or self.rounds_played == 1:
                        avg_time = sum(self.round_times[-50:]) / min(50, len(self.round_times))
                        print(f"  Runde {state.round_number}/{state.max_rounds} | "
                              f"Score: {state.score} | "
                              f"Snitt responstid: {avg_time:.1f}ms")

        except websockets.ConnectionClosed as e:
            print(f"Tilkobling lukket: {e}")
        except Exception as e:
            print(f"Feil i game loop: {e}")
            raise

    async def run(self):
        """Kjor boten: connect -> join -> game loop."""
        await self.connect()
        await self.send_join()
        await self.game_loop()

        # Oppsummering
        total_time = time.time() - self.start_time
        avg_round = sum(self.round_times) / max(1, len(self.round_times))
        print(f"\n=== Oppsummering ===")
        print(f"  Runder spilt: {self.rounds_played}")
        print(f"  Total tid: {total_time:.1f}s")
        print(f"  Snitt responstid: {avg_round:.1f}ms")
        print(f"  Maks responstid: {max(self.round_times, default=0):.1f}ms")

    async def disconnect(self):
        """Koble fra serveren."""
        if self.ws:
            await self.ws.close()
            print("Frakoblet")


# === MULTI-GAME RUNNER ===

async def run_multiple_games(n_games: int = 3):
    """
    Kjor boten flere ganger for a teste konsistens.
    """
    scores = []

    for i in range(n_games):
        print(f"\n{'='*40}")
        print(f"Spill {i+1}/{n_games}")
        print(f"{'='*40}")

        bot = WebSocketBot()
        try:
            await bot.run()
            # TODO: Hent sluttpoeng pa riktig mate
            # scores.append(final_score)
        except Exception as e:
            print(f"Feil i spill {i+1}: {e}")
        finally:
            await bot.disconnect()

    if scores:
        print(f"\n=== Resultater over {n_games} spill ===")
        print(f"  Snitt: {sum(scores)/len(scores):.1f}")
        print(f"  Beste: {max(scores)}")
        print(f"  Darligste: {min(scores)}")


# === KJOR ===

if __name__ == "__main__":
    # Sjekk at API-nokkel er satt
    if not API_KEY:
        print("ADVARSEL: API_KEY ikke satt!")
        print("Sett den med: export API_KEY='din-nokkel-her'")
        print("Kjorer likevel (noen servere trenger ikke nokkel)...\n")

    # Kjor boten
    asyncio.run(WebSocketBot().run())

    # Eller kjor flere spill:
    # asyncio.run(run_multiple_games(n_games=3))
