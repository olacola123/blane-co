# Feilhistorikk — Ting vi IKKE skal gjøre igjen

## Deploy-feil
- ❌ Deploy uten `--set-env-vars` → API key forsvinner → 0 calls → 0 score
- ✅ Hardkodet fallback-key i koden + alltid `--set-env-vars` i deploy

## Interceptor-feil
- ❌ Blokkerte POST /incomingInvoice med fake 403 → aldri testet om det fungerer i competition sandbox
- ✅ Fjernet — la ekte API svare. Vi vet ikke hva som fungerer i fresh sandboxes.

## Deteksjonsfeil
- ❌ "Gehaltsrückstellung" matchet `gehalt` → salary, men var month-end → kjørte feil oppskrift
- ❌ "Totalkostnadene auka" (nynorsk) matchet ikke cost_analysis → falt gjennom til "project"
- ❌ "lettre d'offre" matchet ikke employee_pdf i v15
- ✅ Alle fikset med bedre regex + exclusions

## Voucher-feil
- ❌ Claude lager 3 postings med vatType:0 for supplier invoice → scoring forventer vatType:1 auto-split
- ✅ Interceptor auto-merger 3→2 postings med vatType:1

## dateOfBirth-feil
- ❌ PUT /employee for nyopprettet ansatt → dateOfBirth mangler → 422 → hele salary/project feiler
- ✅ Fallback: setter 1990-05-15 hvis ikke i context. Tracker nye ansatte i ctx.

## NoneType crash
- ❌ rate_categories prefetch crashet på None fromDate/toDate → 500 → hele requesten dør
- ✅ Fikset med `str(x or "")`

## Ting vi ALDRI har testet ordentlig (per 21. mars 15:00)
- Task 11 (salary): ALDRI fått en ekte salary-prompt gjennom uten crash/misdeteksjon
- Task 12 (supplier invoice): ALDRI prøvd uten interceptor-blokkering
- Task 22 (reminder fee): 2 av 3 treff drept av API key-feil
- Task 28 (receipt voucher): Ukjent — trenger logganalyse

## Hva vi trenger fra Ola
- Når task 11/12/22/28 treffer: SEND check-resultater (passed/failed per check)
- Klokkeslett så vi kan matche med logger
- Leaderboard-screenshot med oppdaterte scores
