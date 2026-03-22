# Oppgave 2: Tripletex Agent — Ola

## Hva jeg har prøvd
| # | Tilnærming | Score | Beholdt? | Notater |
|---|-----------|-------|----------|---------|
| 1 | Regel-basert keyword matching | 0/7 | Nei | For begrenset, traff ikke oppgavetyper |
| 2 | Gemini 2.5 Flash + JSON-plan | 63-100% | Nei | Fragil JSON-parsing, placeholder-system |
| 3 | Gemini 2.5 Flash + function calling | 100% på 5+ typer | Ja | Gemini kaller API direkte, fikser feil selv |
| 4 | Claude Sonnet 4 + tool use | ? | Ja (v5) | Byttet fra Gemini, native JSON body |
| 5 | Claude Sonnet 4 + forbedret prompt | ? | Ja (v6) | Dynamisk dato, flere oppskrifter, bedre feilhåndtering |

## Status
- **Total Score: 8.1** | **Rank: #7 av 75** | **8/30 tasks løst** | **32 submissions**
- Cloud Run: `https://tripletex-agent-609915262705.europe-north1.run.app`
- LLM: Claude Sonnet 4 (byttet fra Gemini)

## v6 forbedringer (nåværende)
1. **Dynamisk dato** — system prompt bruker `date.today()` i stedet for hardkodet dato
2. **Separert employee-oppskrifter** — enkel ansatt vs. ansatt med startdato/employment
3. **employmentType: "ORDINARY"** — lagt til i employment-oppskrift (kan ha manglet)
4. **Bedre dato-parsing** — eksplisitte eksempler for alle formater med leading zeros
5. **Eksplisitt phoneNumber** — i supplier, customer, contact, employee oppskrifter
6. **isCustomer:false** — eksplisitt for leverandører
7. **Nye oppskrifter** — credit note, voucher/bilag, update resource, single department
8. **Request timeouts** — 30s per API-kall (hindrer hanging)
9. **Økt response-truncation** — 8000 chars (opp fra 5000)
10. **Bedre logging** — full prompt, filnavn, modellnavn

## Nåværende strategi
Claude Sonnet 4 med tool use (v6): 4 tools (get/post/put/delete), Claude ser API-svar og tilpasser seg. System prompt med:
- Dynamisk dato
- 18 step-by-step oppskrifter (opp fra 12)
- Flerspråklig dato-parsing guide
- Komplett field reference + error recovery
- Native vision for image attachments (ingen OCR-mellomsteg)

## Neste steg
1. Deploy v6 til Cloud Run
2. Submitte for Tier 2 (×2 multiplier)
3. Sjekk resultater — finne nye feilmønstre i logger
4. Vurder modellbytte (Opus?) for komplekse oppgaver

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
