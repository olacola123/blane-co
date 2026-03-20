# Astar Island: Solver Context For Future AI Chats

Dette dokumentet oppsummerer hva som er prøvd i `oppgave-3-astar-island/joakim`, hvilke problemer som ble observert i nattkjøringene 19.-20. mars 2026, hvilke kodeendringer som er gjort etter analysene, og hvilke åpne spørsmål som fortsatt bør undersøkes.

## Hvor du skal starte

- Les denne filen først.
- Les så [predictor.py](/Users/joakim_ruud/Documents/koding/nmiai-2026/oppgave-3-astar-island/joakim/astar_solver/predictor.py), [query_strategy.py](/Users/joakim_ruud/Documents/koding/nmiai-2026/oppgave-3-astar-island/joakim/astar_solver/query_strategy.py), [nightbot.py](/Users/joakim_ruud/Documents/koding/nmiai-2026/oppgave-3-astar-island/joakim/nightbot.py) og historikkmanifestene under [history](/Users/joakim_ruud/Documents/koding/nmiai-2026/oppgave-3-astar-island/joakim/history).
- Hvis du skal evaluere resultater, bruk lagrede manifester og arrays før du gjetter.

## Faktiske historiske funn

Lokale runder med analyser tilgjengelig:

- Runde 2: mean score ca. `34.19`
- Runde 3: mean score ca. `21.87`
- Runde 4: mean score ca. `46.29`

Viktig om runde 1:

- Runde 1 var ikke en gyldig baseline lokalt.
- Historikken viser `0` queries, ingen submissions og ingen analyser.
- Loggen viser først `400` på `simulate`, så `429` på `submit`.

De viktigste problemene som ble identifisert fra runde 2-4:

- Modellen overpredikerte `empty` og underpredikerte `forest` på dynamiske celler.
- Query-strategien brukte mesteparten av budsjettet på bred dekning, men svært lite på re-observasjon av viktige områder.
- Observerte celler fikk for svak effekt i den endelige prediksjonen.
- Historikk-kalibrering slo for hardt ut med få analyserte runder.
- Runde-latenten skilte svakt mellom runder med ganske ulik observasjonsprofil.

## Endringer som allerede var gjort før denne chatten

Følgende var allerede endret i workspace da denne chatten startet:

- `forest`/`empty`-balansen var justert i den statiske og dynamiske delen av prediktoren.
- Historikk-kalibrering var i praksis slått av med `HISTORY_WEIGHT = 0.0`.
- Nightbot hadde fått bedre bootstrapping fra historikk og rikere lagring av analyser/diagnostikk.
- Query-strategien hadde fått en enkel revisit-bonus.

## Endringer gjort i denne chatten

### 0. Høy-konfidens celler kan nå være mye skarpere enn 95/1/1/1/1/1

Problem:

- Med global `probability.floor = 0.01` ble selv sikre celler presset ned til omtrent `95.2%` toppsannsynlighet.
- Det kostet unødvendige poeng på statiske, observerte eller ekstremt høy-konfidens celler.

Løsning:

- Solveren bruker fortsatt en vanlig basis-floor for usikre celler.
- I tillegg brukes nå `sharp_floor = 0.001` på:
  - observerte celler
  - statiske hav-/fjellceller
  - andre celler med svært høy toppsannsynlighet
- Dette gjør at sikre celler kan havne rundt `99.5%` i stedet for rundt `95%`, uten å gjøre hele kartet mer aggressivt.

Berørte filer:

- [config.py](/Users/joakim_ruud/Documents/koding/nmiai-2026/oppgave-3-astar-island/joakim/astar_solver/config.py)
- [probability.py](/Users/joakim_ruud/Documents/koding/nmiai-2026/oppgave-3-astar-island/joakim/astar_solver/probability.py)
- [predictor.py](/Users/joakim_ruud/Documents/koding/nmiai-2026/oppgave-3-astar-island/joakim/astar_solver/predictor.py)
- [solution.py](/Users/joakim_ruud/Documents/koding/nmiai-2026/oppgave-3-astar-island/joakim/solution.py)

### 1. Delvise analyser håndteres nå som delvise, ikke ferdige

Problem:

- Nightbot kunne tidligere markere en runde som ferdig analysert så snart én seed-analyse kom tilbake.
- Det gjorde at senere analyser kunne utebli fra lokal historikk og state.

Løsning:

- `fetch_and_score_analyses()` bygger nå diagnostikk fra alle lagrede analyser i manifestet, ikke bare siste batch.
- Diagnostikk-sammendraget inneholder nå:
  - `expected_seed_count`
  - `num_analysis_payloads`
  - `num_ground_truth_tensors`
  - `analysis_complete`
- `pending_analysis` fjernes først når `analysis_complete == True`.
- `round_scores` oppdateres bare når en runde faktisk er komplett analysert.

Berørte filer:

- [nightbot.py](/Users/joakim_ruud/Documents/koding/nmiai-2026/oppgave-3-astar-island/joakim/nightbot.py)
- [history.py](/Users/joakim_ruud/Documents/koding/nmiai-2026/oppgave-3-astar-island/joakim/astar_solver/history.py)

### 2. Observerte celler får nå sterkere effekt enn før

Problem:

- Tidligere blending ga i praksis observerte celler mindre effekt enn den gamle posterioren alene.
- Det var motsatt av målet.

Løsning:

- `SeedObservationGrid.posterior()` aksepterer nå enten skalar eller kart for `prior_strength`.
- Prediktoren bruker nå count-avhengig pseudo-count-styrke:
  - `3.5` for uobserverte celler
  - `0.85` etter 1 observasjon
  - `0.35` etter 2 observasjoner
  - `0.15` etter 3+ observasjoner
- Global kalibrering brukes nå før observasjonsoppdatering, ikke etterpå.
- Den endelige observerte prediksjonen returnerer nå posterior direkte, uten å blandes tilbake mot prioren etter observasjon.

Praktisk effekt:

- Én observasjon kan nå faktisk vippe en celle bort fra en sterk `empty`-prior.
- Re-observasjon gjør at samme celle raskt blir mye mer evidensdrevet.

Berørte filer:

- [observations.py](/Users/joakim_ruud/Documents/koding/nmiai-2026/oppgave-3-astar-island/joakim/astar_solver/observations.py)
- [predictor.py](/Users/joakim_ruud/Documents/koding/nmiai-2026/oppgave-3-astar-island/joakim/astar_solver/predictor.py)

### 3. Query-strategien søker nå mer eksplisitt etter revisit-kandidater

Problem:

- Tidligere revisit-støtte var for svak og for implisitt.
- Koden hadde ingen tydelig prioritering av celler som var observert akkurat én gang.

Løsning:

- Adaptive query-generering har nå et eksplisitt `revisit`-anker.
- Kandidatscoring regner nå både `repeat_value` og `revisit_value`.
- `revisit_value` er høyest i regioner som er observert minst én gang, men fortsatt under ønsket coverage.

Berørt fil:

- [query_strategy.py](/Users/joakim_ruud/Documents/koding/nmiai-2026/oppgave-3-astar-island/joakim/astar_solver/query_strategy.py)

### 4. Nightbot kan nå importeres i tester uten `requests`

Problem:

- `nightbot.py` importerte tidligere `AstarClient` på modulnivå.
- Det gjorde at testimport feilet i miljøer uten `requests`.

Løsning:

- `AstarClient` opprettes nå lazy via helper, og importeres ikke lenger på modulnivå.
- Tester kan patch'e `nightbot.AstarClient` uten å laste HTTP-avhengigheter først.

Berørt fil:

- [nightbot.py](/Users/joakim_ruud/Documents/koding/nmiai-2026/oppgave-3-astar-island/joakim/nightbot.py)

## Tester lagt til eller oppdatert

Oppdaterte / nye tester:

- [test_history.py](/Users/joakim_ruud/Documents/koding/nmiai-2026/oppgave-3-astar-island/joakim/tests/test_history.py)
- [test_predictor.py](/Users/joakim_ruud/Documents/koding/nmiai-2026/oppgave-3-astar-island/joakim/tests/test_predictor.py)
- [test_query_strategy.py](/Users/joakim_ruud/Documents/koding/nmiai-2026/oppgave-3-astar-island/joakim/tests/test_query_strategy.py)
- [test_nightbot.py](/Users/joakim_ruud/Documents/koding/nmiai-2026/oppgave-3-astar-island/joakim/tests/test_nightbot.py)

Dekker nå blant annet:

- delvis analyse som skal forbli pending
- komplett analyse som skal rydde pending state
- observerte celler som skal kunne overstyre tom-prior
- adaptive revisit-score som skal være bedre enn overdekket region

## Hva som ser ut til å ha fungert

- Å slå av historikk-kalibrering midlertidig var riktig.
- Å styrke `forest` i static/dynamic-prediksjon var riktig retning.
- Å gjøre nightbot mer robust mot restart og historikkbootstrap var riktig.
- Å skille mellom komplett og delvis analyse er viktig for å stole på lokale målinger.

## Hva som fortsatt ikke er bevist å fungere

- Vi har ikke ennå nye konkurranseresultater som bekrefter at revisits faktisk øker score.
- Vi vet heller ikke om ny observasjonsvekt er optimalt kalibrert; den er bare klart mer logisk enn forrige versjon.
- Runde-latenten er fortsatt ganske heuristisk.
- Query-strategien er fortsatt heuristisk, ikke læringsbasert.

## Åpne spørsmål for neste AI-chat

- Får vi nå faktisk høyere repeat-share i practice på nattkjøringer?
- Øker `forest`-recall uten at `settlement` eller `port` kollapser?
- Bør `observation_prior_strength` gjøres konfigurerbar per count i stedet for hardkodet stige?
- Bør adaptive-budsjettet deles eksplisitt i `coverage_budget` og `revisit_budget`?
- Bør vi måle egne offline-metrikker på bare dynamiske celler i diagnostikken, ikke bare helkart?
- Bør latent-inferensen trenes eller erstattes med enklere direkte regler fra observerte frekvenser?

## Praktiske kommandoer

Hvis miljøet har dependencies installert:

```bash
./.venv/bin/python -m unittest \
  oppgave-3-astar-island/joakim/tests/test_history.py \
  oppgave-3-astar-island/joakim/tests/test_predictor.py \
  oppgave-3-astar-island/joakim/tests/test_query_strategy.py \
  oppgave-3-astar-island/joakim/tests/test_nightbot.py
```

Hvis du vil analysere en lagret runde manuelt:

- åpne `history/<round-id>/manifest.json`
- se `diagnostics.query`, `diagnostics.prediction`, `diagnostics.analysis`, `diagnostics.analysis_summary`
- sammenlign `arrays/seed_*_prediction.npy` og `arrays/seed_*_ground_truth.npy`

## Viktigste råd til neste AI-chat

- Ikke bruk runde 1 som normal sammenligningsrunde; den var et feilet kjør.
- Ikke anta at nightbot-state alene er sannheten; historikkmanifestene er viktigst.
- Ikke foreslå nye store modellendringer før nye runder har bekreftet effekten av:
  - sterkere observasjonsbruk
  - revisit-prioritering
  - deaktivert history calibration
