# Tripletex Task Tracker — Verifisert 21. mars 23:xx

## Verifiserte oppgavetyper (fra Cloud Run logger, 139 prompts klassifisert)

### T1/T2 Tasks (19 typer bekreftet)

| Type | Antall sett | Vår oppskrift? | Kjente problemer |
|------|------------|----------------|------------------|
| CUSTOMER (m/adresse) | 7 | Ja | Mangler felt? ACR fikset med 12 tries |
| CUSTOMER (enkel) | 1 | Ja | OK |
| PRODUCT | 7 | Ja | OK, vatType viktig |
| INVOICE_SEND | 10 | Ja | OK |
| INVOICE_MULTI (3 linjer) | 8 | Ja | Ulike MVA-satser per linje |
| DEPARTMENTS (×3) | 7 | Ja | OK |
| EMPLOYEE | 9 | Ja | employment details kan mangle |
| SUPPLIER | 9 | Ja | ALLE team sliter — mangler 1 felt |
| ORDER | 8 | Ja | Vi feilet, andre OK — pris-fix i v12.8 |
| PROJECT | 10 | Ja | OK |
| PROJECT_FIXED | 7 | Ja | Efficiency gap til ACR |
| SALARY | 6 | Ja (voucher) | salary-modul=403, voucher fallback |
| SUPPL_INV (INV-xxxx) | 4 | Ja (voucher) | incomingInvoice=403, voucher fallback |
| ACCT_DIM | 4 | Ja | PERFEKT — alle team 4.00 |
| TIMESHEET | 5 | Ja | hours × rate beregning |
| TRAVEL | 6 | Ja | Per diem + costs |
| CREDIT_NOTE | 7 | Ja | OK |
| REVERSE_PAY | 3 | Ja | Negativ paidAmount |
| PAYMENT | 9 | Ja | PERFEKT — alle team 4.00 |

### T3 Tasks (8 nye typer bekreftet fra logger)

| Type | Antall sett | Vår oppskrift? | Beskrivelse |
|------|------------|----------------|-------------|
| EMP_PDF | 3 | Ja (fra v14) | Ansatt fra PDF-kontrakt — personnr, fødselsdato, dept, salary, % |
| RECEIPT | 2 | Ja (fra v14) | Kvittering → voucher med riktig konto/MVA/dept |
| SUPPL_INV_PDF | 2 | Delvis | Leverandørfaktura fra PDF → opprett supplier + voucher |
| **YEAR_END** | 3 | **NY v15** | Årsoppgjør: avskrivning, forskuddsbetalte, skatteavsetning |
| **BANK_RECON** | 3 | **NY v15** | Bankavsteming: match CSV mot åpne fakturaer |
| **REMINDER_FEE** | 1 | **NY v15** | Purregebyr + delfaktura + delbetaling |
| **LEDGER_AUDIT** | 1 | **NY v15** | Finn feil i hovedbok + korreksjonsbillag |
| **FX_INVOICE** | 1 | **NY v15** | Valutafaktura + kursforskjell (agio/disagio) |

### Ukjente / ikke-sett T3-typer (tasks 20, 29, 30 etc)
Vi har sett 27 unike typer. Maks 30 tasks finnes. Mulige gjenstående:
- DELETE_TRAVEL (slett reiseregning) — i prompt men 0 treff i logger
- CONTACT (kontaktperson) — i prompt men 0 treff i logger
- ADMIN_EMP (ansatt med admin-rettigheter) — i prompt men 0 treff i logger

## Viktig: Vi vet IKKE task-nummer-mapping
TASK_TRACKER v14 antok task 01=customer, 02=product osv — dette var GJETNING.
Plattformen tildeler tilfeldig. Vi ser ikke task-nummer i logger.

## v15 endringer
- 5 nye T3-oppskrifter: YEAR_END, BANK_RECON, REMINDER_FEE, LEDGER_AUDIT, FX_INVOICE
- Forbedret task-type detection (regex, 27+ typer vs 9 i v14)
- max_tokens: 2048 for komplekse tasks, 1024 for enkle
- max_iterations: 25 for komplekse, 12 for enkle
- 15 nye ledger-kontoer i prefetch (1200, 1209, 1230, 1240, 1250, 1700, 1710, 2900, 2920, 3400, 6010, 6020, 8060, 8160, 8700)
