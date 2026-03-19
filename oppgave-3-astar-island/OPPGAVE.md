# Oppgave 3: Astar Island — Norse World Prediction

## Hva er dette?
Observer en black-box norrøn sivilisasjonssimulator gjennom et begrenset viewport, og prediker sannsynlighetsfordelinger for sluttilstanden over et 40×40 kart.

## Hvordan det fungerer
1. Runde starter med fast kart, skjulte parametere, 5 tilfeldige seeds
2. Kall `POST /simulate` med viewport-koordinater (maks 15×15 av 40×40 kart)
3. **50 queries totalt** per runde, delt på tvers av alle 5 seeds
4. Analyser observasjoner for å lære de skjulte reglene
5. Submit H×W×6 sannsynlighetstensor per seed
6. Scores via entropy-vektet KL divergence

## Verdenen — 8 terrengtyper → 6 prediksjonsklasser

| Intern kode | Terreng | Klasse | Beskrivelse |
|-------------|---------|--------|-------------|
| 10 | Ocean | 0 (Empty) | Upasserbart vann, kartkanter |
| 11 | Plains | 0 (Empty) | Flatt land, kan bygges på |
| 0 | Empty | 0 | Generisk tomt |
| 1 | Settlement | 1 | Aktiv norrøn bosetning |
| 2 | Port | 2 | Kystbosetning med havn |
| 3 | Ruin | 3 | Kollapset bosetning |
| 4 | Forest | 4 | Gir mat til naboceller |
| 5 | Mountain | 5 | Upasserbart, statisk |

## Kartgenerering (fra map seed, synlig for deg)
- Havkanter, fjorder skjærer innover, fjellkjeder via random walks
- Skogflekker, initielle bosetninger plassert på land med avstand mellom

## Simuleringslivssyklus (50 år, hver med faser)

### 1. Growth (Vekst)
- Matproduksjon, befolkningsvekst, havneutvikling, ekspansjon

### 2. Conflict (Konflikt)
- Raids, langskip utvider rekkevidde, desperate bosetninger raider mer

### 3. Trade (Handel)
- Havner handler hvis ikke i krig, genererer rikdom + mat, teknologidiffusjon

### 4. Winter (Vinter)
- Varierende alvorlighet, mattap, kollaps fra sult/raids/vinter → Ruins

### 5. Environment (Miljø)
- Ruiner gjenerobres av bosetninger eller skog, kystruiner → havner

## Bosetningsegenskaper
`position`, `population`, `food`, `wealth`, `defense`, `tech_level`, `port_status`, `longship_ownership`, `faction (owner_id)`

## API

**Base URL:** `https://api.ainm.no/astar-island/`
**Auth:** Cookie `access_token` JWT eller Bearer token (fra app.ainm.no login)

### Endepunkter

| Endepunkt | Metode | Beskrivelse |
|-----------|--------|-------------|
| `/rounds` | GET | List alle runder |
| `/rounds/{round_id}` | GET | Rundedetaljer + initial states for alle seeds |
| `/budget` | GET | Query-budsjett (maks 50) |
| `/simulate` | POST | Kjør observasjon (koster 1 query) |
| `/submit` | POST | Submit prediksjonstensor |
| `/my-rounds` | GET | Dine scores, rank, budsjett |
| `/my-predictions/{round_id}` | GET | Dine prediksjoner med argmax/confidence |
| `/analysis/{round_id}/{seed_index}` | GET | Post-runde ground truth sammenligning |
| `/leaderboard` | GET | Offentlig leaderboard |

### POST /simulate — Request
```json
{
  "round_id": "uuid",
  "seed_index": 0,
  "viewport_x": 0,
  "viewport_y": 0,
  "viewport_w": 15,
  "viewport_h": 15
}
```
- `seed_index`: 0–4
- `viewport_w` og `viewport_h`: 5–15

### POST /simulate — Response
```json
{
  "grid": [[...]],
  "settlements": [
    {
      "x": 0, "y": 0,
      "population": 2.8, "food": 0.4, "wealth": 0.7,
      "defense": 0.6, "has_port": true, "alive": true,
      "owner_id": 3
    }
  ],
  "viewport": {"x": 0, "y": 0, "w": 15, "h": 15},
  "width": 40,
  "height": 40,
  "queries_used": 24,
  "queries_max": 50
}
```

### POST /submit — Request
```json
{
  "round_id": "uuid",
  "seed_index": 0,
  "prediction": [[[p0, p1, p2, p3, p4, p5], ...], ...]
}
```
- H×W×6 tensor, hver celle summerer til 1.0 (±0.01 toleranse)
- Resubmitting overskriver tidligere prediksjon for den seeden

### GET /rounds/{round_id} — initial_states
Hver seed inneholder:
- `grid`: H×W terrengkoder
- `settlements`: `[{x, y, has_port, alive}]`
- **NB:** Kun posisjon + port eksponert, ikke interne stats

### Rate Limits
- `/simulate`: 5 req/sek per lag
- `/submit`: 2 req/sek per lag

## Scoring: Entropy-Weighted KL Divergence

### Formler
```
KL(p||q) = Σ pᵢ × log(pᵢ / qᵢ)
entropy(cell) = -Σ pᵢ × log(pᵢ)
weighted_kl = Σ entropy(cell) × KL(truth, pred) / Σ entropy(cell)
score = max(0, min(100, 100 × exp(-3 × weighted_kl)))
```

### Viktige detaljer
- Kun **dynamiske celler** teller (statiske ekskludert)
- Celler med høyere entropi vektes mer
- **100 = perfekt**, **0 = verst**
- **Per-runde score:** gjennomsnitt av 5 seeds. Manglende seed = 0.
- **Leaderboard:** beste `round_score × round_weight` på tvers av alle runder
- **Hot streak:** gjennomsnitt av siste 3 runder

### KRITISK: Aldri assign probability 0.0!
Hvis ground truth har non-zero men prediksjon er 0 → KL divergence → ∞ → ødelegger cellens score.

**Floor på 0.01, renormaliser:**
```python
prediction = np.maximum(prediction, 0.01)
prediction = prediction / prediction.sum(axis=-1, keepdims=True)
```

## Rundetiming
- `prediction_window_minutes` typisk 165 (2t 45min)
- Status-flyt: `pending → active → scoring → completed`

## Nøkkelkonsepter
- **Map seed**: bestemmer terreng (fast, synlig for deg)
- **Sim seed**: tilfeldig per query (ulikt hver gang)
- **Skjulte parametere**: styrer verdensoppførsel (like for alle seeds i en runde)
- **Uniform baseline** scorer typisk 1–5

## Quickstart

```python
import requests
import numpy as np

# Auth — hent token fra app.ainm.no
TOKEN = "din_token_her"
BASE = "https://api.ainm.no/astar-island"
headers = {"Authorization": f"Bearer {TOKEN}"}

# 1. Hent aktive runder
rounds = requests.get(f"{BASE}/rounds", headers=headers).json()
round_id = rounds[0]["id"]

# 2. Hent rundedetaljer med initial states
round_info = requests.get(f"{BASE}/rounds/{round_id}", headers=headers).json()
height = round_info["height"]  # 40
width = round_info["width"]    # 40

# 3. Observer med viewport
obs = requests.post(f"{BASE}/simulate", headers=headers, json={
    "round_id": round_id,
    "seed_index": 0,
    "viewport_x": 0,
    "viewport_y": 0,
    "viewport_w": 15,
    "viewport_h": 15
}).json()

print(f"Queries brukt: {obs['queries_used']}/{obs['queries_max']}")
print(f"Grid shape: {len(obs['grid'])}x{len(obs['grid'][0])}")
print(f"Settlements: {len(obs['settlements'])}")

# 4. Bygg prediksjon (uniform baseline)
prediction = np.ones((height, width, 6)) / 6.0

# Floor — ALDRI 0.0
prediction = np.maximum(prediction, 0.01)
prediction = prediction / prediction.sum(axis=-1, keepdims=True)

# 5. Submit
resp = requests.post(f"{BASE}/submit", headers=headers, json={
    "round_id": round_id,
    "seed_index": 0,
    "prediction": prediction.tolist()
})
print(f"Submit status: {resp.status_code}")

# 6. Sjekk score
my_rounds = requests.get(f"{BASE}/my-rounds", headers=headers).json()
print(my_rounds)
```

## Constraints oppsummert
| Constraint | Verdi |
|-----------|-------|
| Kart | 40×40 |
| Maks viewport | 15×15 |
| Queries per runde | 50 (delt på 5 seeds) |
| Seeds per runde | 5 |
| Prediksjonsklasser | 6 |
| Prediksjonsvindu | ~2t 45min |
| Maks ZIP | N/A (API-basert) |
| Min sannsynlighet | 0.01 (ALDRI 0.0) |
