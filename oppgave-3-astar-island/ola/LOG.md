# Oppgave 3: Astar Island — Ola

## Hva jeg har prøvd
| # | Tilnærming | Score | Beholdt? | Notater |
|---|-----------|-------|----------|---------|
| 1 | Fallback priors (hardkodet) | ~45 | Nei | Baseline |
| 2 | Calibration fra 12 runder | ~55 | Delvis | calibration_data.json |
| 3 | 4-type calibration (DEAD/STABLE/BOOM_SPREAD/BOOM_CONC) | ~65 | Delvis | calibration_4type.json |
| 4 | Blending med continuous vitality | ~71 | Delvis | blending.py |
| 5 | **Optimerte tabeller** (wtype×terrain×dist_band×coastal, 70 seeds) | **82.2** | ✅ | calibration_optimized.json |
| 6 | Floor 0.005→0.003 | +0.8 | ✅ | Lav risiko |
| 7 | Forward simulator (Monte Carlo) | 0-10 | ❌ | Forest tar over, mekanikk feil |

## Nåværende strategi
**solution.py v7** med optimerte tabeller:
1. Laster `calibration_optimized.json` — 162 lookup-entries fra 14 runder × 5 seeds GT
2. Nøkler: `{world_type}_{terrain_group}_{dist_band}_{coastal}`
3. World types: DEAD / STABLE / BOOM (3-type, enklere enn 4-type)
4. Safety submit med STABLE (default), deretter probe 2 seeds for type detection
5. Rebuild priors med riktig type, observe, resubmit

**Backtest resultat (oracle world type):**
- Snitt: 82.2 (opp fra 71 med gammel kode)
- Beste: 91.9 (R4 STABLE)
- Dårligste: 59.7 (R12 BOOM_CONC surv=0.59)

## Kritisk funn
- **Observasjoner med <50 samples er SKADELIGE** — 1 stokastisk observasjon per celle gir score 0.3 vs prior-only 68.6
- Observasjoner bør KUN brukes for world-type detection (1-2 queries), IKKE direkte celle-oppdatering
- World-type detection er verdt +16 poeng over default
- Feil world-type kan koste -50 poeng

## Neste steg
1. Mer granulær world-type (BOOM_LOW vs BOOM_HIGH) for R12-type runder
2. Continuous vitality-interpolasjon mellom tabellene
3. Eventuelt: fikse simulator (krever reverse-engineering av spillregler)

## Filer
- `calibration_optimized.json` — VIKTIGST, de nye tabellene
- `solution.py` — v7 med opt_tables integrasjon
- `backtest.py` — oppdatert for å teste med opt_tables
- `simulator.py` — forward simulator (funker IKKE ennå)
