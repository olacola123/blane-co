# Astar Island — Videre Plan for å Vinne

## Status per 21. mars kl 12:00

### Hva vi har bygget
- **Solver v6** (`solution.py`) — Bayesiansk posterior med 4-type detection
- **4 calibration-tabeller** fra 13 runder ground truth: DEAD, STABLE, BOOM_SPREAD, BOOM_CONC
- **6-fase pipeline**: safety submit → probe → type detect → observe → cross-seed → conditional resubmit
- **Prior-only backtest snitt: 83.0** (opp fra 40.7 med gammel kode)
- **Runde 14 submittet** med v6 — BOOM_CONC type, venter på score

### Nøkkelfiler
| Fil | Formål |
|-----|--------|
| `solution.py` | Hovedsolver v6 |
| `blending.py` | Continuous vitality-blending mellom type-tabeller |
| `calibration_4type.json` | 4-type calibration (DEAD/STABLE/BOOM_SPREAD/BOOM_CONC) |
| `calibration_data.json` | Original calibration (snitt av alle runder) |
| `calibrate_4type.py` | Script for å bygge 4-type tabeller |
| `full_test.py` | Full competition-simulering (alle runder × 5 seeds) |
| `backtest.py` | Enkel prior-only backtest |
| `nightbot.py` | Automatisk rundeløser (trenger oppdatering for v6) |
| `learning_state.json` | Alpha=6.0, alpha_locked=true |

### Scores per type (prior-only, oracle type)
| Type | Snitt | Runder |
|------|-------|--------|
| DEAD | 87.2 | R3, R8, R10 |
| STABLE | 87.3 | R4, R5, R9, R13 |
| BOOM_SPREAD | 84.1 | R1, R2, R6, R11 |
| BOOM_CONC | 63.3 | R7, R12 |

---

## Kjente Bugs (MÅ fikses)

### Bug 1: 4 queries kastes bort
`solve_round()` beregner `remaining_budget = queries_per_seed * n_seeds - 4` men fordeler `remaining // 5` per seed. Resten (1-4 queries) allokeres aldri.
**Fiks:** Fordel ekstra queries til seeds med flest settlements.

### Bug 2: Alpha bruker learning_state, ikke DEFAULT_ALPHA
`main()` leser `alpha = learning.get("alpha", DEFAULT_ALPHA)`. learning_state har `alpha=6.0` men `DEFAULT_ALPHA=3.5`. Disse bør synkroniseres.
**Fiks:** Sett `learning_state.json` alpha=3.5, eller fjern learning_state alpha-override.

### Bug 3: Nightbot ikke oppdatert for v6
`nightbot.py` importerer gamle funksjoner (`infer_round_type`, `adjust_priors_for_round`) som er fjernet/renamed i v6.
**Fiks:** Oppdater nightbot til å bruke v6 solve_round med type_tables.

---

## Forbedringer (prioritert)

### Prioritet 1: Fiks BOOM_CONC (63 → 75+)
BOOM_CONC er vår svakeste type. Calibration-taket er ~55 (per-celle variasjon for høy).
- **Observasjoner er eneste løsning** — alpha=0.85 ved 1 obs gir 54% vekt
- **Sjekk R14-score** — hvis den er 70+, fungerer strategien. Hvis <65, trenger vi endring.
- **Alternativ:** Bruk STABLE-tabell for BOOM_CONC (STABLE scorer 87 snitt, mer konservativ)

### Prioritet 2: Oppdater nightbot for v6
Nightboten kan kjøre runder automatisk. Trenger:
- Importer nye funksjoner (`classify_world_type`, `infer_vitality`)
- Last `calibration_4type.json`
- Bruk `solve_round()` med `type_tables` og `safety_submit=True`
- Legg til `importlib.reload(solution)` for hot-reload

### Prioritet 3: Rekalibrér etter R14
Når R14 er ferdig, legg den til calibration:
- Kjør `calibrate_4type.py` igjen (inkluderer R14)
- Oppdater `calibration_4type.json`
- Spesielt viktig hvis R14 er en ny BOOM_CONC — gir oss 3 datapunkter i stedet for 2

### Prioritet 4: Bruk alle 50 queries
Fiks bug 1 (query-allokering). 4 ekstra queries = ~50-100 ekstra observerte celler.

### Prioritet 5: Entropi-basert query-plassering
Nåværende: settlement-fokusert heatmap.
Bedre: etter probe-queries, beregn entropi per celle fra prior, observer høyest-entropi celler.
Joakims `query_strategy.py` har dette — kan stjeles.

---

## Strategi for gjenværende ~24 timer

### Automatisk drift
1. Start nightbot med v6-kode
2. Den løser runder automatisk (2t45m per runde = ~8-9 runder igjen)
3. Rekalibrér etter hver 2-3 nye runder

### Manuell forbedring mellom runder
1. Sjekk score etter hver runde
2. Hent ground truth → analyser feil
3. Juster tabeller/parametere → backtest → deploy

### Prioriter oppgave 2 (Tripletex)
Vi har 0 poeng på oppgave 2. Selv en baseline (30-40 normalisert) ville løfte totalscoren enormt.
**Noen bør jobbe på Tripletex parallelt.**

---

## Leaderboard-matte

### Hva vi trenger for topp 3
- #1 Matriks: 177.1 (Astar-leaderboard, weighted sum)
- Vår siste hot streak: 78.0
- For topp 3 hot streak: trenger ~85 per runde

### Per-runde mål
| Rundetype | Mål | Nåværende |
|-----------|-----|-----------|
| DEAD | 88+ | 87.2 ✓ |
| STABLE | 88+ | 87.3 ✓ |
| BOOM_SPREAD | 85+ | 84.1 (nær) |
| BOOM_CONC | 75+ | 63.3 (trenger obs) |

### Viktig: vektene øker
R14=1.98×, R15=2.08×, R20=2.65×. Senere runder teller MYE mer.
Én god runde med vekt 2.5 er verdt 2.5 runder med vekt 1.0.

---

## Teknisk arkitektur

```
solution.py (v6)
├── SeedObserver — Bayesiansk prior + observasjoner
│   ├── _prior_cache — forhåndsberegnet per celle fra calibration
│   ├── _floor_cache — klasse-spesifikke floors
│   ├── counts/observed — observasjonsteller
│   └── build_prediction() — posterior med alpha-decay + type-floor
├── solve_round() — 6-fase pipeline
│   ├── Fase 1: Prior-only safety submit
│   ├── Fase 2: Probe queries (4 stk) for type detection
│   ├── Fase 3: Rebuild priors med typed_table
│   ├── Fase 4: Observe alle seeds
│   ├── Fase 5: Kontinuerlig cross-seed learning
│   └── Fase 6: Conditional resubmit (blend hvis Δ>0.04)
├── classify_world_type() — n_settlements + vitality → 4 typer
├── infer_vitality() — continuous fra blending.py
└── get_prior() — calibration lookup med typed_table support

blending.py
├── get_blend_weights() — vitality → (w_dead, w_stable, w_boom)
├── get_blended_prior() — interpoler mellom type-tabeller
└── infer_vitality_continuous() — piecewise linear survival→vitality

calibration_4type.json
├── DEAD (27 keys) — fra R3, R8, R10
├── STABLE (25 keys) — fra R4, R5, R9, R13
├── BOOM_SPREAD (27 keys) — fra R1, R2, R6, R11
└── BOOM_CONC (27 keys) — fra R7, R12
```

## API-nøkler og tilgang
- API_KEY: i miljøvariabel (aldri hardkodet)
- Endpoint: https://api.ainm.no/astar-island
- Budget: 50 queries per runde, shared across 5 seeds
- Rate limit: 5 req/s simulate, 2 req/s submit
