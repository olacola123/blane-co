# Oppgave 3: Astar Island — Norse World Prediction

## Beskrivelse
Observer en black-box norrøn sivilisasjonssimulator gjennom et begrenset viewport og prediker terrengfordelingen over et 40x40 kart etter 50 simulerte år.

## Kjernen
Simuleringen er stokastisk — identiske startbetingelser gir ulike utfall. Du mottar:
- **50 totale queries** per runde, delt på 5 seeds
- **15x15 maks viewport** per query (avdekker kun et lite vindu av fullt kart)
- Oppgave: Bygg sannsynlighetsfordelinger for terrengklasser over hele kartet

## Terrengtyper → 6 prediksjonsklasser
| Terrengtype | Klasse |
|-------------|--------|
| Empty/Ocean/Plains | 0 |
| Settlements | 1 |
| Ports | 2 |
| Ruins | 3 |
| Forests | 4 |
| Mountains | 5 |

## Simuleringsmekanikk
- 8 terrengtyper mapper til 6 prediksjonsklasser
- 50-års livssyklus med vekst, konflikt, handel, vinter, og miljøfaser
- Stokastiske elementer: settlementutvidelse, raidingmønstre, faksjonsskifter, kollaps/recovery

## API-endepunkter
**Base URL**: `https://api.ainm.no/astar-island/`

| Endepunkt | Metode | Beskrivelse |
|-----------|--------|-------------|
| `/astar-island/rounds` | GET | List aktive runder |
| `/astar-island/rounds/{round_id}` | GET | Hent initial kart-state for alle seeds |
| `/astar-island/simulate` | POST | Kjør én stokastisk query med viewport-koordinater |
| `/astar-island/submit` | POST | Submit W×H×6 sannsynlighetstensor per seed |
| `/astar-island/budget` | GET | Sjekk gjenværende query-budsjett |

## Scoring: Entropy-Weighted KL Divergence

**Formel**: `Score = 100 × exp(-KL_entropy_weighted)`

### Ground Truth
Arrangørene pre-beregner sannsynlighetsfordelinger ved å kjøre simuleringer hundrevis av ganger med sanne skjulte parametre.

### KL Divergence (per celle)
`KL(p||q) = Σ pᵢ × log(pᵢ / qᵢ)`

### Entropy-vekting
- Statiske celler (hav alltid hav, fjell endres aldri) bidrar null
- Kun dynamiske celler teller, vektet etter entropi
- Celler med usikre utfall har høyere vekt

### KRITISK: Minimum-sannsynlighet
**ALDRI assign probability 0.0 til noen klasse!**
Hvis ground truth har non-zero men prediksjon er 0 → KL divergence → uendelig → ødelegger den cellens score.
**Anbefalt**: Sett minimum floor på 0.01 per klasse, deretter renormaliser.

### Score-range: 0 (verst) til 100 (perfekt)

## Submission-format
3D array `prediction[y][x][class]`:
- Ytre dimensjon: H rader (høyde)
- Midtre dimensjon: W kolonner (bredde)
- Indre dimensjon: 6 klassesannsynligheter (MÅ summere til 1.0 ± 0.01 toleranse)

Alle sannsynligheter non-negative. Resubmission overskriver tidligere prediksjoner for samme seed.

## Constraints
- **50 queries totalt** per runde på tvers av alle 5 seeds
- **15x15 maks viewport** per query
- **40x40 fullt kartstørrelse**
- Må submitte prediksjoner for alle seeds (usubmittede seeds = score 0)
- Prediksjonsvindu typisk 2 timer 45 minutter etter runde starter

## Strategi-tips
- Identifiser skjulte parametre som styrer verdensoppførsel via viewport-observasjoner
- Bygg probabilistiske modeller som ekstrapolerer begrensede observasjoner til fullt kart
- Balanser query-allokering mellom seeds
- Regn med stokastisk varians i simuleringer
- Start med uniform prior (1/6 per klasse) som baseline
- Bruk 10 queries per seed (50/5)
