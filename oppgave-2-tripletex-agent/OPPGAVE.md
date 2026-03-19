# Oppgave 2: Tripletex — AI Accounting Agent

## Beskrivelse
Bygg en AI-agent som utfører regnskapsoppgaver via Tripletex API. Agenten mottar oppgavebeskrivelser på ulike språk, gjør API-kall for å utføre operasjoner, og scores på korrekthet og effektivitet.

## Hvordan det fungerer
1. Submit en HTTPS endpoint-URL til plattformen
2. Systemet provisjonerer en fersk Tripletex sandbox-konto per submission
3. En tilfeldig regnskapsoppgave sendes til ditt `/solve`-endpoint
4. Agenten tolker prompten, prosesserer vedlegg (PDFs/bilder)
5. Agenten kaller Tripletex API via autentisert proxy
6. Resultater verifiseres felt-for-felt mot forventede verdier
7. Score oppdateres på rolling leaderboard

## Scoring
- **Score-range**: 0.0 – 6.0 (perfekt Tier 3 + best effektivitet)
- **Felt-for-felt korrekthet**: points_earned / max_points (0.0–1.0)
- **Tier-multiplikator**: Tier 1 (x1.0), Tier 2 (x2.0), Tier 3 (x3.0)
- **Effektivitetsbonus**: Kun ved perfekt korrekthet (1.0). Basert på API-kall-effektivitet vs best kjente løsning. 4xx-feil reduserer bonus. Kan opptil doble tier-scoren.
- **Best score per task**: All-time best trackes. Dårlige runs senker aldri score. Rekalkueres hver 12. time.

## Oppgavetyper
- **30 forskjellige regnskapsoppgaver**
- **56 varianter per oppgave** (7 språk x 8 datasett)
- **Språk**: Norsk, Engelsk, Spansk, Portugisisk, Nynorsk, Tysk, Fransk

### Kategorier
- **Ansatte**: Opprett ansatte, sett roller, oppdater kontaktinfo
- **Kunder & Produkter**: Registrer kunder, opprett produkter
- **Fakturering**: Opprett fakturaer, registrer betalinger, kreditnotaer
- **Reiseregninger**: Registrer eller slett reiseregninger
- **Prosjekter**: Opprett prosjekter knyttet til kunder
- **Korreksjoner**: Slett eller reverser feil
- **Avdelinger**: Opprett avdelinger, aktiver regnskapsmoduler

## Endpoint-spesifikasjon

**Metode**: POST
**Content-Type**: application/json
**Timeout**: 300 sekunder

### Request-format
```json
{
  "prompt": "Oppgavebeskrivelse på ulike språk",
  "base_url": "https://api.proxy.tripletex.dev",
  "session_token": "auth_token",
  "files": [{"filename": "...", "content_base64": "..."}]
}
```

### Response-format
```json
{
  "status": "completed"
}
```

### Autentisering mot Tripletex API
- **Basic Auth**: Username: `0`, Password: `session_token`
- **Valgfri API Key**: Sendes som Bearer token header for endpoint-beskyttelse

## Tripletex API
- **Base URL**: `https://api.proxy.tripletex.dev`
- **Docs**: `https://kkpqfuj-amager.tripletex.dev/v2-docs/`
- **REST API v2**

### Eksempler
```
GET /employee?fields=id,firstName,lastName,*
Authorization: Basic 0:{session_token}
```

```
POST /customer
Content-Type: application/json
{"name": "Customer Name", "organizationNumber": "..."}
```

```
POST /invoice
{"customerId": 123, "lines": [...], "invoiceDate": "..."}
```

## Constraints
- Sandbox utløper 31. mars 2026
- Hvert team deler én sandbox
- Norske tegn (æ, ø, å) fungerer i requests
- Alle kall logges for debugging
- Hver submission = fersk konto (blank)

## Strategi-tips
- Planlegg før du kaller — unngå trial-and-error
- Minimer GET-kall; valider inputs først
- Les feilmeldinger for retry-veiledning
- Start med enkle Tier 1-oppgaver (single API call)
- Submit URL til: https://app.ainm.no/submit/tripletex
