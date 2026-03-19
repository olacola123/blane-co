# NM i AI 2026 — Hovedkonkurranse

## Konkurransen
- 3 ukjente oppgaver, 19-22. mars 2026
- Oppgavetyper ukjent — kan være CV, NLP, tabular, RL, optimering, eller noe annet
- API-basert scoring (se `oppgave-*/api_client.py` som utgangspunkt)

## Laget
- **Ola** (olacola123) — Claude Code
- **Joakim** — Claude Code
- **Mathea** — Copilot
- Delt GitHub-repo. Alle pusher til main.

## Auto-push
Commit og push automatisk etter hver score-forbedring eller vesentlig endring.
Commit-melding: `oppgave-X: score Y, kort beskrivelse`

## Eksperiment-logg (VIKTIG)
Etter hvert eksperiment — oppdater `oppgave-X/CLAUDE.md`:
- Legg til i "Hva vi har prøvd": `tilnærming -> score -> beholdt/forkastet`
- Oppdater "Beste strategi" hvis ny best
- Legg til i "Funn" hvis noe kan hjelpe andre oppgaver
Commit+push dette sammen med koden. Da kan alle se hva som er prøvd, hva som funket, og hva som feilet — uten å spørre.

Før du starter på en oppgave: `git pull` og les oppgavens CLAUDE.md for å unngå å gjenta andres feil.

## Mappestruktur
- `oppgave-1/`, `oppgave-2/`, `oppgave-3/` — én mappe per oppgave, all kode her
- Hver oppgave har sin egen `CLAUDE.md` — oppdater med funn, strategi, hva som er prøvd
- `STATUS.md` — lagoversikt, scores, hvem jobber på hva

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
- `cp solution.py solution.py.best-X` etter hver forbedring — aldri mist beste versjon
- Secrets i miljøvariabler: `export API_KEY='...'`

## Overførbare lærdommer fra warm-up
1. **Enkelhet > kompleksitet** — enkel tilnærming slo alle avanserte varianter
2. **Automatisert søk > manuell tuning** — 1200 auto-iterasjoner fant det 12 manuelle ikke fant
3. **Submit tidlig** — ha noe på leaderboard før du optimerer
4. **Backups alltid** — navngi med score: `solution.py.best-85`
5. **Heuristikk har tak** — når score platåer etter 3+ forsøk, bytt tilnærming helt
6. **Verifiser antakelser empirisk** — les oppgaven, men TEST med faktiske kjøringer

## Python-miljø
```bash
source env/bin/activate  # eller: python3 -m venv env && source env/bin/activate
# setup.sh installerer torch, transformers, scikit-learn, chromadb, stable-baselines3, opencv, m.m.
```
