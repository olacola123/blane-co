# NM i AI 2026 — Hovedkonkurranse

**Lag:** Ola, Joakim, Mathea
**Dato:** 19-22. mars 2026

## Quickstart

```bash
# 1. Klon og installer
git clone https://github.com/olacola123/nmiai-2026.git
cd nmiai-2026
bash setup.sh

# 2. Aktiver miljø
source env/bin/activate

# 3. Sett API-nøkkel
export API_KEY='din-nøkkel-her'

# 4. Jobb i din mappe
cd oppgave-1/<ditt-navn>/
```

## Struktur

```
oppgave-X/
  CLAUDE.md        ← delt eksperiment-logg (ALLE oppdaterer denne)
  api_client.py    ← delt API-klient
  ola/             ← Olas kode
  joakim/          ← Joakims kode
  mathea/          ← Matheas kode
templates/         ← ferdiglagde ML-templates
STATUS.md          ← scores og hvem jobber på hva
```

## Arbeidsflyt

1. `git pull` før du starter
2. Les `oppgave-X/CLAUDE.md` — se hva andre har prøvd
3. Jobb i `oppgave-X/<ditt-navn>/`
4. Submit baseline FØRST, forbedre etterpå
5. Oppdater `oppgave-X/CLAUDE.md` med resultat
6. Commit + push etter hver score-forbedring

## Regler

- Commit-melding: `oppgave-X: score Y, kort beskrivelse`
- Backup beste løsning: `cp solution.py solution.py.best-85`
- Secrets i miljøvariabler, ALDRI i kode
- Les andres mapper for inspirasjon — stjel det som funker
