# Prompt for neste Claude Code-sesjon

Kopier og lim inn dette:

---

Vi jobber med NM i AI 2026 — Oppgave 3: Astar Island. Konkurransen avsluttes 22. mars kl 15:00.

## Les disse filene FØRST:
1. `oppgave-3-astar-island/ola/VIDERE_PLAN.md` — full statusoversikt, bugs, og prioriteringer
2. `oppgave-3-astar-island/ola/solution.py` — v6 solver (hovedkoden)
3. `oppgave-3-astar-island/OPPGAVE.md` — oppgavebeskrivelse
4. `CLAUDE.md` — konkurranse-regler

## Hva er gjort:
- Solver v6 med 4-type calibration (DEAD/STABLE/BOOM_SPREAD/BOOM_CONC)
- 6-fase pipeline: safety submit → probe → type detect → observe → cross-seed → conditional resubmit
- Prior-only backtest snitt: 83.0 (opp fra 40.7)
- Runde 14 submittet med v6, venter på/har fått score

## VIKTIG — regler:
- ALDRI kjør kode eller submit uten at jeg eksplisitt sier "kjør"
- API_KEY er i miljøvariabel, aldri hardkod den
- Backtest og full_test bruker INGEN queries (bare GET /analysis) — trygge å kjøre

## Prioritert arbeid:
1. Sjekk R14-score → analyser hva som funket/feilet
2. Fiks 3 bugs: query-allokering (4 bortkastet), alpha-mismatch, nightbot v6-oppdatering
3. Rekalibrér 4-type tabeller med nye runder
4. Oppdater nightbot for automatisk drift med v6
5. Kjør `full_test.py` for å verifisere endringer
6. Fokuser på BOOM_CONC-forbedring (svakeste type, 63 snitt)

## Nyttige kommandoer (IKKE kjør uten bekreftelse):
```bash
cd ~/vault/Prosjekter/NM\ i\ AI/hovedkonkurranse && source env/bin/activate
export API_KEY='...'

# Backtest (trygt, ingen queries)
python3 oppgave-3-astar-island/ola/full_test.py

# Rebuild calibration (trygt, bare GET-kall)
python3 oppgave-3-astar-island/ola/calibrate_4type.py

# Kjør live runde (BRUKER QUERIES — bare med bekreftelse)
python3 oppgave-3-astar-island/ola/solution.py

# Sjekk scores
curl -s -H "Authorization: Bearer $API_KEY" "https://api.ainm.no/astar-island/my-rounds" | python3 -c "import json,sys; [print(f'R{r[\"round_number\"]}: {r[\"round_score\"]} ({r[\"status\"]})') for r in sorted(json.load(sys.stdin), key=lambda x: x.get('round_number',0), reverse=True)[:5]]"
```
