# NM i AI 2026 — Hovedkonkurranse

**Lag:** Ola, Joakim, Mathea | **Dato:** 19-22. mars 2026

## Quickstart

```bash
# 1. Klon og installer
git clone https://github.com/olacola123/nmiai-2026.git
cd nmiai-2026
bash setup.sh
source env/bin/activate

# 2. Sett API-nøkkel
export API_KEY='din-nøkkel-her'
```

## Arbeidsflyt

```bash
# Hent andres arbeid
bash scripts/sync.sh

# Kopier template som utgangspunkt
bash scripts/copy-template.sh classifier 1 ola

# Jobb i din mappe
cd oppgave-1/<ditt-navn>/

# Når du har en score — logger, backuper, committer og pusher automatisk:
bash scripts/submit.sh 1 ola 72.3 "XGBoost ensemble"
```

## Struktur

```
oppgave-X/
  OPPGAVE.md       ← oppgavebeskrivelse (fylles inn når oppgaven slippes)
  api_client.py    ← delt API-klient
  scores.jsonl     ← alle scores (append-only, ingen merge conflicts)
  ola/             ← Olas kode + LOG.md
  joakim/          ← Joakims kode + LOG.md
  mathea/          ← Matheas kode + LOG.md
scripts/           ← submit.sh, scoreboard.py, copy-template.sh, sync.sh
templates/         ← ferdiglagde ML-templates
STATUS.md          ← auto-generert scoreboard
```

## Regler

- **Les andres `LOG.md` før du starter** — unngå å gjenta feil
- **Submit baseline FØRST** — score på tavla > perfekt plan
- Commit-melding: `oppgave-X: score Y, kort beskrivelse`
- Secrets i miljøvariabler, ALDRI i kode
- Stjel det som funker fra andres mapper
