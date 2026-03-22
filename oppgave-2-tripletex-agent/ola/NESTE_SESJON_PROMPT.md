# Prompt for neste Claude Code sesjon

Kopier dette inn:

---

Jeg jobber med oppgave 2 (Tripletex AI Accounting Agent) i NM i AI. Mål: 90 poeng.

Les disse filene FØRST:
1. `oppgave-2-tripletex-agent/ola/PLAN.md` — vinnerplan
2. `oppgave-2-tripletex-agent/ola/TRACKER.md` — scores per task
3. `oppgave-2-tripletex-agent/ola/server.py` — nåværende kode

## Status etter sesjon 21. mars ~22:00

**Score: 61.72, #51** (men mange fixes deployet nå)

**Nåværende deploy: v21-fixed, gemini-2.0-flash, rev 00145**
Health: `https://tripletex-agent-609915262705.europe-north1.run.app/health`

## Fixes deployet av Claude denne sesjonen (rev 00138-00140):
- ✅ Fikset Claude 401-bug — hele codebase brukte gammel Claude SDK, nå Gemini
- ✅ Gemini 429 retry i extract_data (handlers.py)
- ✅ Spanish travel expense detection: `gastos de viaje` etc.
- ✅ French receipt detection: `justificatif` etc.
- ✅ Customer country: `{"id": 161}` i stedet for `"NO"`

Laget (Joakim/Mathea) deployede 5+ revisions etter (00141-00145, v21-fixed).
Sjekk git for å se hva de endret.

## Viktigste gaps som gjenstår (estimert):

| Task | Type | Nå | Max | Gap | Hva feiler |
|------|------|-----|-----|-----|------------|
| 20 | cost_analysis | 0.60 | 6.0 | +5.4 | Extraction feiler, fields-syntax |
| 29 | ukjent T3 | 1.64 | 6.0 | +4.4 | Ukjent type → feil handler |
| 11 | supplier_invoice | 0 | 4.0 | +4.0 | /incomingInvoice=403, voucher=0 |
| 24 | year_end | 2.25 | 6.0 | +3.75 | Checks 4+5 feiler |
| 22 | receipt_voucher | 0 | 6.0 | +3.0 | Trenger testing etter fix |
| 27 | ? T3 | 4.60 | 6.0 | +1.4 | 2 checks feiler |
| efficiency | alle T2 | ~2.0 | 4.0 | +2-3 | For mange GET-kall |

## Scoring-forklaring (bekreftet):
- Score = DIREKTE SUM av task-råscorer (ikke normalisert)
- T1 max: 2.0 | T2 max: 4.0 | T3 max: 6.0
- Estimert totalmax: ~116. For 90 trenger vi +28 fra 61.72.

## Deploy-kommando:
```bash
cd ~/vault/Prosjekter/NM\ i\ AI/hovedkonkurranse/oppgave-2-tripletex-agent/ola
gcloud run deploy tripletex-agent --source . --region europe-north1 --project ainm26osl-745 --allow-unauthenticated --timeout=300 --memory=512Mi --set-env-vars="GEMINI_API_KEY=AIzaSyAJwOUFybMfNr8VJjUBuzbeDEpQYFvD1Zc,GEMINI_MODEL=gemini-2.0-flash"
```

## Loggsjekk:
```bash
gcloud run services logs read tripletex-agent --project=ainm26osl-745 --region=europe-north1 --limit=100
```

Konkurransen slutter SØNDAG 22. mars kl 15:00.

---
