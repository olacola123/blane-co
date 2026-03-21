# Tripletex Vinnerplan — Oppdatert 21. mars 13:00

## Status NÅ
- **50.46 poeng, #26, 28/30 tasks**
- **#1 Ninjas: 51.05** (men andre team har ikke kjørt T3 ennå — ledelsen vil endre seg)
- v15 deployed — raskere (50-90s vs 200-300s)
- T3 aktiv (×3 multiplier)

## Tasks med 0 score (MÅ FIKSES)
| Task | Type | Tries | Problem | Løsning |
|------|------|-------|---------|---------|
| 11 | Salary (tekst) | 10 | Aldri truffet med v14+ | Voucher 5000→2780 er klar. Trenger bare å treffe |
| 12 | Supplier invoice (tekst) | 10 | Aldri truffet med v14+ | Voucher med auto-MVA klar. Trenger bare å treffe |
| 22 | Receipt/kvittering? | 2 | 0/10 begge ganger | vatType id:1 fix i v15. Trenger testing |
| 28 | Supplier invoice PDF? | 1 | 0/10 | Mangler email på supplier + feil kontering? |

## Tasks med lav score (FORBEDRINGSPOTENSIAL)
| Task | Nå | Mål | Gap | Problem |
|------|-----|-----|-----|---------|
| 25 | 1.50 | 6.00 | +4.50 | Ninjas har 6.00! Hva gjør de annerledes? |
| 01 | 1.50 | 2.00+ | +0.50 | Customer m/adresse — mangler felt? |
| 05 | 1.33 | 2.00+ | +0.67 | Employee — pct fix deployet, trenger ny hit |
| 06 | 1.20 | 2.00+ | +0.80 | Supplier — language fix deployet, trenger ny hit |
| 08 | 1.20 | 2.00+ | +0.80 | Order — pris-fix, trenger ny hit |
| 19 | 1.77 | 3.00+ | +1.23 | PDF kontrakt — 13/22, mangler felt |
| 23 | 0.60 | 3.00+ | +2.40 | Standalone voucher — vatType feil? |
| 30 | 1.80 | 3.00+ | +1.20 | Årsoppgjør — 6/10 |

## Prioritert arbeidsliste

### Prio 1: Fix Task 22 (0/10 kvittering)
- Siste forsøk: USB-hub, 2 kall, 0 errors, men 0/10
- Hypotese: vatType id:1 fix i v15 kan fungere
- Alternativ: kanskje scoring sjekker spesifikke felter vi ikke setter
- **Aksjon:** Analysere loggene nøye, teste med sandbox

### Prio 2: Fix Task 25 (1.50 vs 6.00)
- Ninjas fikk 6.00 på 1. forsøk!
- Vi fikk bare 1.50
- **Aksjon:** Finn ut hvilken oppgavetype Task 25 er, analyser hva vi gjør feil

### Prio 3: Efficiency på alle T2-tasks (01-18)
- Mange tasks scorer 2.00-2.40 men ACR har 4.00
- Forskjellen er efficiency (færre API-kall)
- **Aksjon:** Reduser unødvendige kall, bruk pre-fetched data mer

### Prio 4: T3 task coverage
- Tasks 20, 23, 24, 26, 27, 29 trenger forbedring
- Mange nye oppgavetyper: årsoppgjør, bankavstemmig, feilretting, prosjektsyklus
- **Aksjon:** Legg til instruksjoner for nye task-typer

### Prio 5: Treffe Task 11 og 12
- 10 tries = systemet deprioriterer dem
- Men vi HAR fiksene (voucher 5000→2780, auto-MVA)
- **Aksjon:** Bare continue submitting

## Tekniske forbedringer å gjøre
1. **Prompt kan kuttes mer** — noen seksjoner er fortsatt verbose
2. **Pre-fetch flere kontoer** — vi har 13 ekstra, men noen mangler (6010, 6020, 8700, 2920 etc.)
3. **Response caching** — cache Claude's account lookups mellom iterasjoner
4. **Bedre PDF-parsing** — pdfplumber mister noen felt
5. **Parallel tool execution** — allerede implementert, men kan optimeres

## Hva scoring sjekker (bekreftet)
- Felt-for-felt verifisering av opprettede objekter
- Efficiency: færre kall = høyere bonus
- Feilrenhet: færre 4xx errors = høyere bonus
- Best score ever = permanent (dårlige runs senker ikke)
- Rekalkuering hver 12. time mot benchmarks

## Viktige API-fakta
- BETA endpoints → alltid 403 (POST /incomingInvoice, PUT /project/{id}, etc.)
- Salary-modul deaktivert i competition sandboxes → voucher fallback
- vatType id:1 = inngående MVA 25% (auto-splitter til 2710)
- vatType id:3 = utgående MVA 25% (for salg/faktura)
- Ny session token PER submission — token dør etter "completed"

## Deploy-kommando
```bash
cd oppgave-2-tripletex-agent/ola
gcloud run deploy tripletex-agent --source . --region europe-north1 --project ainm26osl-745 --allow-unauthenticated --timeout=300 --memory=512Mi --set-env-vars="ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY"
```

## Endpoint URL
`https://tripletex-agent-609915262705.europe-north1.run.app`
