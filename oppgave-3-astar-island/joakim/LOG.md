# Oppgave 3: Astar Island — Joakim

## Hva jeg har prøvd
| # | Tilnærming | Score | Beholdt? | Notater |
|---|-----------|-------|----------|---------|
| 1 | solution_diamond.py (Manhattan distance) | ~80 backtest | ✅ | Basert på Ola v8, Manhattan istedenfor Chebyshev |
| 2 | + Expansion modulation (urban/rural) | -0.12 | ❌ | Konsekvent verre, disabled |
| 3 | + Recalibrated 17 runder (85 seeds) | 80.18 | ✅ | Opp fra 79.26 med 16 runder |
| 4 | nightbot_diamond.py | N/A | ✅ | Ny nightbot med diamond solver |
| 5 | **FIX: opt_tables key format + 17r recal** | **86.66 HR** | ✅ | **+3.94 pts history_replay avg** — se detaljer under |
| 6 | **solution_v2.py: Pooled Empirical Model** | **89.58 HR** | ✅ | **+2.92 pts vs diamond** — ny arkitektur, se detaljer under |

## Nåværende strategi
**solution_v2.py** med Pooled Round-Specific Empirical Model:

### Nøkkelinnovasjon
Alle 5 seeds deler samme skjulte simulasjonsparametre. Celler med samme feature key `(terrain_group, dist_band, coastal)` har derfor identisk underliggende distribusjon uansett seed. Ved å poole observasjoner fra ALLE seeds etter feature key, får vi ~5x mer data per key enn per-seed analyse.

### Prediksjons-pipeline (per celle)
1. **Calibration prior**: Blended across world types (DEAD/STABLE/BOOM) via model_tables cascade (specific → medium → opt_tables → simple → fallback)
2. **Round-specific empirical**: Pooled fra alle seeds, cap 50% vekt, skalert med `emp_n / 100`
3. **Bayesian update**: Per-celle observasjoner med alpha=50 (nesten ingen effekt — pooled empirical dominerer)
4. **Floor + normalize**: Min 0.001, mountain impossible, port impossible inland

### Tunet parametre (grid search over 7 runder)
- `emp_max = 0.50` (maks vekt for empirisk data)
- `emp_scale = 100.0` (50 obs → 50% vekt)
- `alpha = 50.0` (per-celle obs er 1 stokastisk sample → noise, pooled empirisk er bedre)
- `min_emp_n = 5` (minimum observasjoner for å bruke empirisk)

### Backtest (7 runder history_replay, 7 runder prior_only)
- History replay avg: **89.58** (diamond: 86.66 → **+2.92 pts**)
- Per-runde vs diamond: R2 +6.50, R3 -2.12, R4 +0.53, R5 +0.86, R6 +4.08, R8 +0.08, R17 +2.98
- Vinner 6/7 runder
- Prior_only avg: 72.94

### Viktige funn
- **Per-celle Bayesian er noise**: Enkeltobservasjoner er stokastiske realisasjoner. Grid search viser at alpha=50+ (nesten null celle-vekt) slår alpha=8 med +1.8 pts. Den poolede empiriske modellen er langt mer robust.
- **50% empirisk cap er optimalt**: For aggressivt (75%) over-truster noisy data. For konservativt (30%) ignorerer nyttig data. 50% balanserer.
- **Skalering med sample count**: 5 obs → 5% vekt, 50 obs → 50% vekt. Unngår å bruke empirisk med lite data.

## Neste steg
1. Oppdater nightbot til å bruke solution_v2.py
2. Eventuelt: finjuster for DEAD-worlds (R3 taper -2.12 vs diamond)
3. Eventuelt: prøv å legge til scale_for_vitality post-processing

## Filer
- `solution_v2.py` — **NY HOVEDSOLVER** (Pooled Empirical Model, 89.58 HR avg)
- `solution_diamond.py` — Gammel solver (Manhattan distance, opt_tables, 86.66 HR avg)
- `sweep_v2.py` — Parameter sweep script for v2-tuning
- `calibrate_manhattan.py` — Bygger calibration-tabeller fra GT (API-basert)
- `calibrate_all_rounds.py` — Full recalibration fra joakim_data/ (offline, alle 17 runder)
- `backtest_diamond_full.py` — Hybrid backtest for diamond solver
- `nightbot_diamond.py` — Nattbot med diamond solver
- `calibration_manhattan.json` — Basic calibration (17 runder)
- `calibration_manhattan_4type.json` — 4-type calibration
- `calibration_manhattan_opt.json` — Optimerte tabeller (163 entries, world_type keys)
- `model_tables_17r.json` — Model tables fra 17 runder (for cascade lookup)
- `analyze_localization.py` — Analyser urban/rural patterns
