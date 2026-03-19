# Oppgave 2: Tripletex Agent — Ola

## Hva jeg har prøvd
| # | Tilnærming | Score | Beholdt? | Notater |
|---|-----------|-------|----------|---------|
| 1 | Regel-basert keyword matching | 0/7 | Nei | For begrenset, traff ikke oppgavetyper |
| 2 | Gemini 2.5 Flash + JSON-plan | 63-100% | Nei | Fragil JSON-parsing, placeholder-system |
| 3 | Gemini 2.5 Flash + function calling | 100% på 5+ typer | Ja | Gemini kaller API direkte, fikser feil selv |

## Status
- **Total Score: 8.1** | **Rank: #7 av 75** | **8/30 tasks løst** | **32 submissions**
- Cloud Run: `https://tripletex-agent-609915262705.europe-north1.run.app`
- Daily limit nådd 19. mars

## Nåværende strategi
Gemini function calling (v3): 4 tool-funksjoner (get/post/put/delete), Gemini ser faktiske API-svar og tilpasser seg. System prompt med komplett API cheat sheet. Maks 25 remote calls per oppgave.

## Neste steg
1. Bytte til Claude API (Sonnet) for bedre tolkning
2. Submitte mye for å treffe alle 30 oppgavetyper
3. Tier 2 åpner fredag (x2 multiplier)
4. Sjekk logger: `curl -s https://tripletex-agent-609915262705.europe-north1.run.app/logs`

## Funn (viktig for alle)
- `userType` MÅ være "STANDARD" — "ADMINISTRATOR" godtas ikke av API
- `priceExcludingVatCurrency` ikke `priceExcludingVat`
- `postalAddress.addressLine1` ikke `address1`
- Employee krever department.id + email
- Employment krever at ansatt har dateOfBirth
- PUT med /:action bruker query params, ikke body
- GET /invoice krever invoiceDateFrom/To
- travelExpense: datoer i travelDetails-objekt, ikke flat
