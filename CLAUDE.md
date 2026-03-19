# NM i AI 2026 — Hovedkonkurranse

## Konkurransen
- 3 ukjente oppgaver, 19-22. mars 2026
- Oppgavetyper ukjent — kan være CV, NLP, tabular, RL, optimering, eller noe annet
- API-basert scoring (se `oppgave-*/api_client.py` som utgangspunkt)

## Laget
- **Ola** (olacola123) — Claude Code
- **Joakim** (joakimotto) — Claude Code
- **Mathea** (matheabrannstorph-commits) — Copilot
- Delt GitHub-repo. Alle pusher til main.

## Før du starter på en oppgave (VIKTIG)
1. `git pull`
2. Les `oppgave-X/OPPGAVE.md` — oppgavebeskrivelse og constraints
3. Les ALLE personlogger: `oppgave-X/ola/LOG.md`, `joakim/LOG.md`, `mathea/LOG.md`
4. Se `oppgave-X/scores.jsonl` — hvem har best score?
5. Sjekk den beste personens kode for inspirasjon

## Mappestruktur
```
oppgave-X/
  OPPGAVE.md       ← oppgavebeskrivelse (skrives én gang)
  api_client.py    ← delt API-klient
  scores.jsonl     ← scorelog (append-only, auto-merges)
  ola/
    LOG.md          ← Olas eksperiment-logg
    solution.py     ← Olas løsning
  joakim/
    LOG.md          ← Joakims eksperiment-logg
    solution.py
  mathea/
    LOG.md          ← Matheas eksperiment-logg
    solution.py
scripts/
  submit.sh        ← submit + logg + backup + commit + push
  scoreboard.py    ← genererer STATUS.md fra scores.jsonl
  copy-template.sh ← kopier template til din mappe
  sync.sh          ← hent andres arbeid (git pull)
templates/         ← ferdiglagde ML-templates
STATUS.md          ← auto-generert scoreboard
```

## Scripts
```bash
# Hent andres arbeid
bash scripts/sync.sh

# Kopier en template som utgangspunkt
bash scripts/copy-template.sh classifier 1 ola

# Logg en score (backup + commit + push automatisk)
bash scripts/submit.sh 1 ola 72.3 "XGBoost ensemble"

# Oppdater scoreboard manuelt
python3 scripts/scoreboard.py
```

## Eksperiment-logg
Etter hvert eksperiment — oppdater din `oppgave-X/<ditt-navn>/LOG.md`:
- Legg til rad i "Hva jeg har prøvd"-tabellen
- Oppdater "Nåværende strategi" og "Neste steg"
- Legg til "Funn" hvis noe kan hjelpe andre oppgaver

Bruk `bash scripts/submit.sh` for å logge score + commit + push i ett.

## Ressurser i repoet
- `templates/` — ferdiglagde Python-templates (rl_agent, rag_pipeline, classifier, segmentation, optimizer, websocket_bot, api_client)
- `oppgave-*/api_client.py` — API-klient klar til bruk i hver oppgave
- `setup.sh` — installer Python-miljø med alle ML-pakker

## Olas lokale ressurser (kun tilgjengelig på Olas maskin)
Disse ligger i vaulten, ikke i repoet. Les ved behov:
- `../NM i AI 2026 - Kampplan.md` — 4-dagers tidsplan, panic protocols, bytteprotokoll
- `../Meta-Strategi-Konkurransetips.md` — ML-tips, ensemble, AutoGluon, validation, hyperparams
- `../CV-Guide*.md`, `../RAG Guide*.md`, `../RL-Guide.md` — fagspesifikke guider
- `../Research - *.md` — competition tricks for CV, NLP, API submission

## Arbeidsregler
- Jobb autonomt — ikke spør Ola om tekniske valg, bare vis kort resultat
- Submit baseline FØRST, forbedre etterpå. Score på tavla > perfekt plan.
- Test inkrementelt: maks 2-3 endringer mellom submissions
- Secrets i miljøvariabler: `export API_KEY='...'`

## Overførbare lærdommer fra warm-up
1. **Enkelhet > kompleksitet** — enkel tilnærming slo alle avanserte varianter
2. **Automatisert søk > manuell tuning** — 1200 auto-iterasjoner fant det 12 manuelle ikke fant
3. **Submit tidlig** — ha noe på leaderboard før du optimerer
4. **Backups alltid** — submit.sh gjør dette automatisk
5. **Heuristikk har tak** — når score platåer etter 3+ forsøk, bytt tilnærming helt
6. **Verifiser antakelser empirisk** — les oppgaven, men TEST med faktiske kjøringer

## Python-miljø
```bash
source env/bin/activate  # eller: python3 -m venv env && source env/bin/activate
# setup.sh installerer torch, transformers, scikit-learn, chromadb, stable-baselines3, opencv, m.m.
```
