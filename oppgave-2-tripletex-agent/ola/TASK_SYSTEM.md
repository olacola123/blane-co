# Tripletex Task System — Single Source of Truth

> Oppdatert: 21. mars 2026 16:30

## Vår situasjon
- **61.37 pts, #42 overall, #30 Tripletex (72.2)**
- **#1 Propulsion Optimizers: 85.01 pts Tripletex**
- **Gap: 23.6 poeng**
- 59/300 daily submissions brukt
- v16 deployed, Claude Sonnet 4 + Haiku

## Hva vi KAN se vs IKKE kan se

### Vi ser i dashboard:
- Task (X/Y) — score og maks checks
- Tidspunkt og varighet
- Check 1: passed/failed (ingen detaljer om HVA som sjekkes)

### Vi ser i Cloud Run logger:
- PROMPT-tekst (full oppgave)
- Detected task type
- Alle API-kall med status
- Errors og timing

### Vi kan IKKE se:
- Hvilken task-nummer (01-30) en run tilhører
- Hva hver check faktisk verifiserer
- Hvilke felt som mangler/feiler

## Metode for å identifisere tasks

**Steg 1**: Match dashboard-tidspunkt mot logger (CET = UTC+1)
**Steg 2**: Bruk antall checks (X/Y) som fingerprint — hver task-type har fast antall checks
**Steg 3**: Bruk score-mønster over tid for å bekrefte

## Verifisert task-type mapping

Disse er BEKREFTET via logg-matching:

| Task | Type | Checks | Vår score | #1 score | Gap | Verifisert? |
|------|------|--------|-----------|----------|-----|-------------|
| 01 | ? | ? | 1.50 | 1.50 | 0 | NEI |
| 02 | ? | ? | 2.00 | 2.00 | 0 | NEI |
| 03 | ? | ? | 2.00 | 2.00 | 0 | NEI |
| 04 | ? | ? | 2.00 | 2.00 | 0 | NEI |
| 05 | ? | ? | 1.33 | 1.33 | 0 | NEI |
| 06 | ? | ? | 1.25 | 1.50 | 0.25 | NEI |
| 07 | ? | ? | 2.00 | 2.00 | 0 | NEI |
| 08 | ? | ? | 1.20 | 1.50 | 0.30 | NEI |
| 09 | ? | ? | 2.33 | 3.00 | 0.67 | NEI |
| 10 | ? | ? | 2.40 | 2.67 | 0.27 | NEI |
| 11 | ? | ? | 0 | 4.00 | 4.00 | NEI |
| 12 | ? | ? | 0 | 0 | 0 | NEI |
| 13 | ? | ? | 2.40 | 2.50 | 0.10 | NEI |
| 14 | ? | ? | 4.00 | 4.00 | 0 | NEI |
| 15 | ? | ? | 2.44 | 2.80 | 0.36 | NEI |
| 16 | ? | ? | 2.40 | 3.00 | 0.60 | NEI |
| 17 | ? | ? | 3.20 | 3.50 | 0.30 | NEI |
| 18 | ? | ? | 4.00 | 4.00 | 0 | NEI |
| 19 | ? | ? | 2.05 | 2.45 | 0.40 | NEI |
| 20 | ? | ? | 0.60 | 6.00 | 5.40 | NEI |
| 21 | ? | ? | 1.93 | 2.36 | 0.43 | NEI |
| 22 | ? | ? | 0 | 0 | 0 | NEI |
| 23 | ? | ? | 0.60 | 0.60 | 0 | NEI |
| 24 | ? | ? | 2.25 | 6.00 | 3.75 | NEI |
| 25 | ? | ? | 4.50 | 5.25 | 0.75 | NEI |
| 26 | ? | ? | 3.75 | 3.75 | 0 | NEI |
| 27 | ? | ? | 4.60 | 6.00 | 1.40 | NEI |
| 28 | ? | ? | 1.20 | 1.50 | 0.30 | NEI |
| 29 | ? | ? | 1.64 | 6.00 | 4.36 | NEI |
| 30 | ? | ? | 1.80 | 1.80 | 0 | NEI |

## Task-type status (hva vi VET om vår agent per type)

### FUNGERER BRA (score matcher #1 eller nært)
| Type | Typisk score | Checks | Kjente problemer |
|------|-------------|--------|------------------|
| customer | ? | ? | Lite gap |
| product | ? | ? | OK |
| departments | ? | ? | OK |
| payment | 10/10 | 10 | PERFEKT |
| acct_dimension | 4.00 | ? | PERFEKT |
| invoice_send | ? | ? | OK |

### FUNGERER DELVIS (vi scorer men taper poeng)
| Type | Typisk score | Checks | Kjente problemer |
|------|-------------|--------|------------------|
| project_fixed | 6/8 | 8 | Check 2 feiler — ukjent felt |
| ledger_audit | 7.5/10 → 0/10 | 10 | Ustabilt! Noen ganger 75%, noen ganger 0% |
| cost_analysis | 4/10 | 10 | Bare 40% — feil analyse? |
| bank_recon | 2/10 | 10 | Bare 20% — matching feiler |
| fx_invoice | 10/10 | 10 | Funker noen ganger, errors andre |
| employee_pdf | ? | ? | Mangler felt |
| timesheet | ? | ? | Timer/rate beregning |
| year_end | ? | ? | Ukjent |
| month_end | ? | ? | Ukjent |

### FUNGERER IKKE (0 score)
| Type | Checks | Kjente problemer |
|------|--------|------------------|
| supplier_invoice | 0/8 | incomingInvoice→422 (feil feltnavn), voucher fallback scorer 0 |
| ? (task 11) | ? | Aldri identifisert |
| ? (task 22) | ? | Aldri identifisert |

## Observasjoner fra siste 7 runs (16:11-16:24 CET / 15:12-15:26 UTC)

| CET | Score | Type (fra logger) | Server-tid | Kall | Errors |
|-----|-------|-------------------|------------|------|--------|
| 16:11 | 7.5/10 | ledger_audit | 114s | 23 | 0 |
| 16:14 | 10/10 | fx_invoice | 28s | 4 | 1 |
| 16:15 | 6/8 | project_fixed | 113s | 10 | 0 |
| 16:18 | 4/10 | cost_analysis | 63s | 14 | 2 |
| 16:20 | 0/8 | supplier_invoice | 54s | 3 | 2 |
| 16:22 | 2/10 | bank_recon | 95s | 16 | 0 |
| 16:24 | 0/10 | ledger_audit | 96s | 25 | 0 |

### project_fixed: 6/8, Check 2 failed
- Check 1: passed
- Check 2: **failed**
- Check 3: passed
- Check 4: passed

## Prioritert fix-liste (sortert etter poeng-gap)

### 1. Task 20: cost_analysis (+5.40 gap)
- Vi: 0.60, #1: 6.00
- Siste run: 4/10 (40%)
- **Problem**: Ukjent — trenger logganalyse av hva vi faktisk gjør vs hva som forventes

### 2. Task 29: ukjent type (+4.36 gap)
- Vi: 1.64, #1: 6.00
- **Problem**: Vet ikke engang hvilken type dette er

### 3. Task 11: ukjent type (+4.00 gap)
- Vi: 0, #1: 4.00
- **Problem**: Aldri fått score. Vet ikke type.

### 4. Task 24: ukjent type (+3.75 gap)
- Vi: 2.25, #1: 6.00
- **Problem**: Vet ikke type

### 5. Task 27: ukjent type (+1.40 gap)
- Vi: 4.60, #1: 6.00
- **Problem**: Vet ikke type

## API-problemer (bekreftet)
- POST /incomingInvoice → 422: "unitPriceExcludingVatCurrency" eksisterer ikke i objektet
- POST /incomingInvoice nested structure → heller ikke riktig
- POST /salary/transaction → 422: "ikke registrert med arbeidsforhold i perioden"
- PUT /project/{id} → BETA (403)
- POST /project/orderline → BETA (403)

## Deployment
```bash
cd ~/vault/Prosjekter/NM\ i\ AI/hovedkonkurranse/oppgave-2-tripletex-agent/ola
gcloud run deploy tripletex-agent --source . --region europe-north1 --project ainm26osl-745 --allow-unauthenticated --timeout=300 --memory=512Mi --set-env-vars="ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY"
```

## Logger
```bash
gcloud run services logs read tripletex-agent --project=ainm26osl-745 --region=europe-north1 --limit=200
```
