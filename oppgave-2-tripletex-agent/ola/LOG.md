# Oppgave 2: Tripletex Agent — Ola

## Hva jeg har prøvd
| # | Tilnærming | Score | Beholdt? | Notater |
|---|-----------|-------|----------|---------|
| 1 | Regel-basert keyword matching | 0/7 | Nei | For begrenset, traff ikke oppgavetyper |
| 2 | Gemini 2.5 Flash + JSON-plan | 63-100% | Nei | Fragil JSON-parsing, placeholder-system |
| 3 | Gemini 2.5 Flash + function calling | 100% på 5+ typer | Ja | Gemini kaller API direkte, fikser feil selv |
| 4 | Claude Sonnet 4 + tool use | ? | Ja (v5) | Byttet fra Gemini, native JSON body |
| 5 | Claude Sonnet 4 + v6-v14 | ~50 pts | Ja | Iterativ forbedring, 19 oppskrifter, interceptors |
| 6 | Claude Sonnet 4 + v15 | ? | Ja | Full T3-støtte: year-end, bank recon, ledger audit, FX, reminder |

## Status
- Cloud Run: `https://tripletex-agent-609915262705.europe-north1.run.app`
- LLM: Claude Sonnet 4
- 134 runs, 657 API-kall, 29 errors (per 21. mars 23:xx)
- 27 av 30 oppgavetyper identifisert fra logger

## v15 (nåværende — deployed 21. mars ~23:xx)
1. **5 nye T3-oppskrifter**: YEAR_END, BANK_RECON, REMINDER_FEE, LEDGER_AUDIT, FX_INVOICE
2. **Forbedret task detection** — regex-basert, 27+ typer (opp fra 9)
3. **Dynamisk max_tokens** — 2048 for komplekse, 1024 for enkle
4. **Dynamisk max_iterations** — 25 for komplekse, 12 for enkle
5. **15 nye ledger-kontoer i prefetch** — for avskrivning, årsoppgjør, valuta

## Nåværende strategi
Claude Sonnet 4 med tool use: 4 tools (get/post/put/delete). System prompt med:
- 24+ step-by-step oppskrifter
- Massiv prefetch (48 parallelle kall): employees, departments, products, customers, suppliers, salary types, 30+ ledger-kontoer, rate categories, cost categories, payment types
- Interceptors: POST /supplier → /customer, incomingInvoice → voucher, dateOfBirth fix
- Native vision for bilder, pdfplumber for PDF

## Neste steg
1. Sjekk logger etter v15 — treffer vi nye T3-typer?
2. Optimaliser efficiency (færre prefetch-kall for enkle tasks?)
3. Test DELETE_TRAVEL, CONTACT, ADMIN_EMP oppgavene (0 treff så langt)

## Funn (viktig for alle)
- `userType` MÅ være "STANDARD" — "ADMINISTRATOR" godtas ikke av API
- `priceExcludingVatCurrency` ikke `priceExcludingVat`
- `postalAddress.addressLine1` ikke `address1`
- Employee krever department.id + email
- Employment krever at ansatt har dateOfBirth
- PUT med /:action bruker query params, ikke body
- GET /invoice krever invoiceDateFrom/To
- travelExpense: datoer i travelDetails-objekt, ikke flat
- Datoformat i prompts: "DD. EnglishMonth YYYY" — engelske månedsnavn selv i andre språk
- Claude håndterer bilder nativt via vision — trenger ikke separat OCR
- employmentType: "ORDINARY" bør inkluderes
- isCustomer:false MÅ settes eksplisitt for leverandører
- T3-oppgaver er MYE mer komplekse: årsoppgjør, bankavsteming, hovedbok-audit, valutafaktura
- /incomingInvoice og /supplierInvoice er BETA (403) — bruk voucher
- salary-modul deaktivert — bruk voucher (5000 debit, 2780 kredit)
