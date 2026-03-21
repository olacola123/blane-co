# Plan: Parametrisk modell for 90+ score

## Status nå (verifisert)
- Diskrete lookup-tabeller i `calibration_optimized.json` → snitt **82.2** (oracle type)
- DEAD: 88.4, STABLE: 87.8, BOOM: **76.5** ← hele problemet
- Verste: R12 (surv=0.60) = 61.1, R7 (surv=0.43) = 71.6, R14 (surv=0.52) = 77.0
- DEAD og STABLE er nesten på 90 allerede

## Rotårsak
BOOM-tabellen er gjennomsnitt av 7 runder med survival 0.35-0.60.
En runde med surv=0.60 bruker same tabell som surv=0.35 — helt feil.

## Løsning: Parametrisk modell
Kontinuerlig funksjon: `f(dist, terrain, survival_rate, coastal) → [6-class probs]`
Fit på alle 14×5=70 GT-datasett (~84.000 dynamiske celler).

### Steg
1. **Bygg modell**: parametrisk funksjon med ~20 params per terrain-type
2. **Fit**: scipy.optimize mot weighted KL (competition metric) over alle 70 datasett
3. **Valider**: leave-one-round-out cross-validation
4. **Integrer**: i solution.py
5. **Obs-strategi**: 2 queries → estimer survival rate → bruk modell

### Mål
- BOOM snitt: 76.5 → 84-88
- Totalt snitt: 82.2 → 87-90

## Verifiserte fakta
- Class order: [empty, settlement, port, ruin, forest, mountain]
- Terrain codes: 0=empty, 1=settle, 2=port, 3=ruin, 4=forest, 5=mountain, 10=ocean, 11=plains
- Ocean → alltid [1,0,0,0,0,0], Mountain → alltid [0,0,0,0,0,1]
- Floor: 0.003
- Seeds har FORSKJELLIGE grids innen same runde (~700 celler forskjell)
- Observations med <50 samples er skadelige for celle-oppdatering
- World-type detection verdt +16 poeng
