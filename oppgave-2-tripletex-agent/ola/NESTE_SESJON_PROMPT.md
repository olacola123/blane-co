# Prompt for neste Claude Code sesjon

Kopier dette inn:

---

Jeg jobber med oppgave 2 (Tripletex AI Accounting Agent) i NM i AI. Vi er #26 med 50.46 poeng, T3 er aktiv.

Les disse filene FØRST:
1. `oppgave-2-tripletex-agent/ola/PLAN.md` — full vinnerplan med alle tasks og status
2. `oppgave-2-tripletex-agent/ola/TASK_TRACKER.md` — detaljert task-tracker
3. `oppgave-2-tripletex-agent/ola/server.py` — v15 koden (system prompt starter linje 462)

Viktigste ting å fikse:
- **Task 22** (kvittering) scorer 0/10 — vatType fix i v15, trenger testing/debugging
- **Task 25** scorer 1.50 men Ninjas har 6.00 — finn ut hvorfor
- **Task 11/12** (salary/supplier inv) scorer 0 — har fikser men trenger å treffe dem
- **Efficiency** — mange T2 tasks scorer 2.00 men bør være 4.00 (færre API-kall)

Endpoint: `https://tripletex-agent-609915262705.europe-north1.run.app`
Deploy: `cd oppgave-2-tripletex-agent/ola && gcloud run deploy tripletex-agent --source . --region europe-north1 --project ainm26osl-745 --allow-unauthenticated --timeout=300 --memory=512Mi --set-env-vars="ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY"`

Sjekk loggene: `gcloud run services logs read tripletex-agent --project=ainm26osl-745 --region=europe-north1 --limit=100`

BETA endpoints (alltid 403): POST /incomingInvoice, PUT /project/{id}, POST /project/orderline
Salary-modul deaktivert → voucher 5000→2780 er riktig fallback.

Konkurransen slutter søndag kl 15:00. Hvert minutt teller. Submit hvert 90 sekund.

---
