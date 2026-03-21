# Tripletex Tracker — Single Source of Truth
> Oppdateres etter HVER batch submissions

## Score: 61.72 pts | Rank: #51 | Subs: 324

## Bekreftet Task → Type Mapping

| Task | Type | Checks | Score | #1 | Gap | Status | Hva feiler |
|------|------|--------|-------|-----|-----|--------|------------|
| 01 | ? | ? | 1.50 | 1.50 | 0 | OK | — |
| 02 | ? | 7? | 2.00 | 2.00 | 0 | OK | — |
| 03 | ? | ? | 2.00 | 2.00 | 0 | OK | — |
| 04 | customer | 2 | 2.00 | 2.00 | 0 | DELVIS | Check 2 failed (2/7=29%) — mangler felt |
| 05 | ? | ? | 1.33 | 1.33 | 0 | OK | — |
| 06 | ? | ? | 1.25 | 1.50 | 0.25 | DELVIS | — |
| 07 | ? | ? | 2.00 | 2.00 | 0 | OK | — |
| 08 | ? | ? | 1.33 | 1.50 | 0.17 | DELVIS | — |
| 09 | ? | ? | 2.33 | 3.00 | 0.67 | DELVIS | — |
| 10 | ? | ? | 2.40 | 2.67 | 0.27 | DELVIS | — |
| 11 | **supplier_invoice** | ? | **0** | 4.00 | **4.00** | FEIL | /incomingInvoice=403, voucher scorer 0 |
| 12 | ? | ? | **0** | 0 | 0 | FEIL | Begge team 0 |
| 13 | ? | ? | 2.40 | 2.50 | 0.10 | OK | — |
| 14 | acct_dimension | ? | 4.00 | 4.00 | 0 | OK | Fikset voucher-instruksjoner |
| 15 | ? | ? | 2.44 | 2.80 | 0.36 | DELVIS | — |
| 16 | ? | ? | 2.40 | 3.00 | 0.60 | DELVIS | — |
| 17 | ? | ? | 3.20 | 3.50 | 0.30 | DELVIS | — |
| 18 | payment | 10 | 4.00 | 4.00 | 0 | PERFEKT | — |
| 19 | ? | ? | 2.05 | 2.45 | 0.40 | DELVIS | — |
| 20 | cost_analysis | ? | **0.60** | 6.00 | **5.40** | FEIL | 55% errors, fields-syntax, manglende customer |
| 21 | ? | ? | 2.14 | 2.36 | 0.22 | DELVIS | — |
| 22 | **receipt_voucher** | ? | **0** | 0 | 0 | FEIL | Haiku=0 kall. Nå Sonnet, men detection feil (fr→unknown) |
| 23 | ? | ? | 0.60 | 0.60 | 0 | DÅRLIG | — |
| 24 | ? | 6? | 2.25 | 6.00 | **3.75** | DÅRLIG | year_end? check 4+5 failed |
| 25 | ? | ? | 4.50 | 5.25 | 0.75 | DELVIS | — |
| 26 | ? | ? | 3.75 | 3.75 | 0 | OK | — |
| 27 | ? | 3? | 4.60 | 6.00 | 1.40 | DELVIS | departments? 7/7=100% |
| 28 | ? | ? | 1.20 | 1.50 | 0.30 | DÅRLIG | — |
| 29 | ? | ? | 1.64 | 6.00 | **4.36** | DÅRLIG | Ukjent type |
| 30 | ? | ? | 1.80 | 1.80 | 0 | OK | — |

## Siste 5 runs (batch 18:17-18:18 CET)

| CET | Score | Checks | Type | Kall | Err | Notater |
|-----|-------|--------|------|------|-----|---------|
| 18:17 | 8/8 ✅ | 7/7 | payment | 2 | 0 | PERFEKT |
| 18:17 | 2/7 ❌ | 1/2 fail | customer | 1 | 0 | Check 2 fail — felt mangler |
| 18:17 | 6/10 | 4/6, c4+5 fail | year_end | 6 | 0 | 0 errors men 40% checks fail |
| 18:18 | 7/7 ✅ | 3/3 | departments | 4 | 0 | PERFEKT |
| 18:18 | 5/10 | 2/4, c3+4 fail | receipt(→unknown) | 11 | **11** | TOKEN EXPIRED — plattform-bug |

## Kjente problemer per task-type

### KRITISK (0 score eller store gap)

**supplier_invoice (Task 11)**:
- /incomingInvoice = alltid 403 i competition
- Voucher-fallback scorer 0 — scoring forventer kanskje incomingInvoice-objekt
- FIX DEPLOYET: Går rett til voucher nå, men trenger verifisering

**receipt_voucher (Task 22)**:
- Deteksjon feiler for franske prompts ("besoin de la depense" → unknown)
- Haiku ga 0 kall. Nå på Sonnet.
- Token-expired på siste run (plattform-issue, ikke vår feil)
- TRENGER FIX: detection regex for franske receipt-prompts

**cost_analysis (Task 20)**:
- 55% error rate: feil fields-syntaks, manglende customer på entitlement
- FIX DEPLOYET: parenteser i fields, company_id i entitlement
- Trenger verifisering

### DELVIS (scorer men taper poeng)

**year_end (Task 24?)**:
- 6/10 (60%), checks 4+5 feiler
- 0 API-errors men checks feiler → feil beregning/manglende felt
- Trenger: analysere hva check 4+5 sjekker

**customer (Task 04?)**:
- 2/7 (29%), check 2 feiler
- 0 errors men check feiler → mangler felt (invoiceEmail? country? phoneNumber?)

## API-fakta (verifisert)

| Endpoint | Status | Detaljer |
|----------|--------|----------|
| POST /incomingInvoice | **403** i competition | Riktige felt: amountInclVat, vatTypeId, accountId, externalId |
| POST /salary/transaction | 422 | "ikke registrert med arbeidsforhold" |
| PUT /project/{id} | 403 BETA | Bruk GET→PUT med version |
| POST /ledger/voucher | ✅ | postings-array, row+date required |
| POST /activity | ✅ | Trenger name + isGeneral. IKKE activityNumber! |
| GET /ledger/posting | ✅ | fields bruker PARENTESER: account(number,name) |

## Deploy-info
```bash
cd ~/vault/Prosjekter/NM\ i\ AI/hovedkonkurranse/oppgave-2-tripletex-agent/ola
gcloud run deploy tripletex-agent --source . --region europe-north1 --project ainm26osl-745 --allow-unauthenticated --timeout=300 --memory=512Mi --set-env-vars="ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY"
```

## Metode
1. Ola submitter 5 ganger
2. Sender leaderboard-screenshot FØR og ETTER
3. Claude matcher tidspunkter mot Cloud Run logger (CET = UTC+1)
4. Oppdaterer denne filen
5. Fikser kode → deploy → repeat
