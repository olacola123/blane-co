# Oppgave 2: Tripletex — AI Accounting Agent

## Beskrivelse
Bygg en AI-agent som utfører regnskapsoppgaver via Tripletex API. Agenten mottar oppgavebeskrivelser på ulike språk, tolker vedlegg (PDFs/bilder), gjør API-kall, og scores på korrekthet og effektivitet.

## Hvordan det fungerer
1. Submit en HTTPS endpoint-URL til plattformen (`https://app.ainm.no/submit/tripletex`)
2. Systemet provisjonerer en **fersk** Tripletex sandbox-konto per submission
3. En tilfeldig regnskapsoppgave sendes som POST til ditt `/solve`-endpoint
4. Agenten tolker prompten, prosesserer vedlegg (PDFs/bilder via base64)
5. Agenten kaller Tripletex API via autentisert proxy
6. Resultater verifiseres felt-for-felt mot forventede verdier
7. Score oppdateres på rolling leaderboard

---

## Endpoint-spesifikasjon

**Metode**: `POST /solve`
**Content-Type**: `application/json`
**Timeout**: 300 sekunder

### Request-format
```json
{
  "prompt": "Oppgavebeskrivelse på ulike språk",
  "files": [
    {
      "filename": "faktura.pdf",
      "content_base64": "JVBERi0xLjQ...",
      "mime_type": "application/pdf"
    }
  ],
  "tripletex_credentials": {
    "base_url": "https://tx-proxy.ainm.no/v2",
    "session_token": "abc123..."
  }
}
```

> **VIKTIG**: Credentials ligger i `tripletex_credentials`-objektet, IKKE som flat felt.

### Response-format
```json
{
  "status": "completed"
}
```

### Autentisering mot Tripletex API
- **Basic Auth**: Username `0`, Password `session_token`
- **Header**: `Authorization: Basic base64("0:{session_token}")`
- **Valgfri API Key**: Kan sendes som Bearer token for å beskytte DITT endpoint

---

## Scoring

| Komponent | Detaljer |
|-----------|----------|
| **Score-range** | 0.0 – 6.0 (perfekt Tier 3 + best effektivitet) |
| **Korrekthet** | Felt-for-felt verifisering, normalisert 0–1 |
| **Tier-multiplikator** | T1 ×1, T2 ×2, T3 ×3 |
| **Effektivitetsbonus** | Kun ved perfekt korrekthet (1.0). Kan opptil doble tier-scoren |
| **Kall-effektivitet** | Færre **write calls** (POST/PUT/DELETE/PATCH) vs best kjente → høyere bonus. **GET telles IKKE** — les fritt |
| **Feilrenhet** | Færre 4xx-feil (400/404/422) → høyere bonus |
| **Best score** | All-time best trackes. Dårlige runs senker aldri scoren |
| **Rekalkuering** | Hver 6. time mot gjeldende benchmarks. Korrekthet synker aldri, kun effektivitetsbonus påvirkes |
| **Leaderboard** | Sum av best scores på tvers av alle 30 oppgavetyper |

### Scoring-eksempel (Tier 2)
| Scenario | Score |
|----------|-------|
| Feilet helt | 0.0 |
| 80% korrekt | 1.6 |
| Perfekt, men mange feil | ~2.1 |
| Perfekt, få feil | ~2.6 |
| Perfekt + best effektivitet | 4.0 |

---

## Tier-plan

| Tier | Åpner | Multiplikator |
|------|-------|---------------|
| Tier 1 | Fra konkurransestart | ×1 |
| Tier 2 | Tidlig fredag | ×2 |
| Tier 3 | Tidlig lørdag | ×3 |

### Tier-eksempler fra docs
- **T1**: Create employee, create customer (foundational)
- **T2**: Invoice with payment, credit notes, project billing (multi-step)
- **T3**: Bank reconciliation from CSV, error correction in ledger, year-end closing (complex scenarios)

---

## Rate Limits

| Status | Samtidige | Per oppgave per dag |
|--------|-----------|---------------------|
| Verifisert | 3 | 10 |
| Uverifisert | 1 | 3 |

---

## Oppgavetyper

- **30 forskjellige regnskapsoppgaver**
- **56 varianter per oppgave** (7 språk × 8 datasett)
- **Språk**: Norsk, Engelsk, Spansk, Portugisisk, Nynorsk, Tysk, Fransk
- **Tildeling**: Vektet mot oppgaver du har forsøkt færre ganger

### Kategorier
- **Ansatte**: Opprett ansatte, sett roller, oppdater kontaktinfo
- **Kunder & Produkter**: Registrer kunder, opprett produkter
- **Fakturering**: Opprett fakturaer, registrer betalinger, kreditnotaer
- **Reiseregninger**: Registrer eller slett reiseregninger
- **Prosjekter**: Opprett prosjekter knyttet til kunder
- **Korreksjoner**: Slett eller reverser feil
- **Avdelinger**: Opprett avdelinger, aktiver regnskapsmoduler

### Vanlige oppgavemønstre
| Mønster | Eksempel |
|---------|----------|
| Opprett enkelt objekt | `POST /employee` |
| Opprett med kobling | `GET /customer` → `POST /order` → `POST /invoice` |
| Endre eksisterende | `GET /customer` → `PUT /customer/{id}` |
| Slett/reverser | `GET /travelExpense` → `DELETE /travelExpense/{id}` |
| Flersteg | `POST /customer` → `POST /invoice` → `POST /payment` |

---

## Tripletex API

### Base URLs
| Kontekst | URL |
|----------|-----|
| **Konkurranse** (i requests) | `https://tx-proxy.ainm.no/v2` |
| **Sandbox** (for testing) | `https://kkpqfuj-amager.tripletex.dev/v2` |
| **API-docs** | `https://kkpqfuj-amager.tripletex.dev/v2-docs/` |

### Viktige endpoints
| Endpoint | Metoder |
|----------|---------|
| `/employee` | GET, POST, PUT |
| `/customer` | GET, POST, PUT |
| `/product` | GET, POST |
| `/invoice` | GET, POST |
| `/order` | GET, POST |
| `/travelExpense` | GET, POST, PUT, DELETE |
| `/project` | GET, POST |
| `/department` | GET, POST |
| `/ledger/account` | GET |
| `/ledger/posting` | GET |
| `/ledger/voucher` | GET, POST, DELETE |

### API-tips
- **Field selection**: `?fields=id,firstName,lastName,*`
- **Paginering**: `?from=0&count=100`
- **List-responses**: `{"fullResultSize": N, "values": [...]}`
- **Delete**: ID i path — `DELETE /employee/123`
- **Norske tegn** (æ, ø, å) fungerer i requests

---

## Sandbox (for testing)

1. Gå til submission-siden og klikk **"Get Sandbox Account"**
2. **UI**: `https://kkpqfuj-amager.tripletex.dev`
3. **API**: `https://kkpqfuj-amager.tripletex.dev/v2`
4. Logg inn med e-posten vist på sandbox-kortet, bruk "Forgot password" for Visma Connect-oppsett
5. Sandbox er **persistent** (i motsetning til konkurransen der hver submission = fersk konto)
6. Utløper 31. mars 2026

---

## Vanlige feil

| Feil | Årsak | Løsning |
|------|-------|---------|
| `401 Unauthorized` | Feil autentisering | Bruk Basic Auth med `0:{session_token}` |
| `404 Not Found` | Feil endpoint-path | Sjekk URL mot API-docs |
| `422 Unprocessable` | Manglende required fields | Les feilmeldingen — den sier hvilke felt |
| Tomme `values` | Feil søkeparametre | Sjekk query params, prøv uten filtre |
| Timeout | For mange API-kall | Optimer — planlegg kall på forhånd |

---

## Kodeeksempler

### Minimal FastAPI-agent
```python
from fastapi import FastAPI, Request
import httpx
import base64

app = FastAPI()

@app.post("/solve")
async def solve(request: Request):
    data = await request.json()

    prompt = data["prompt"]
    files = data.get("files", [])
    creds = data["tripletex_credentials"]

    base_url = creds["base_url"]
    token = creds["session_token"]

    # Basic Auth: username=0, password=session_token
    auth = httpx.BasicAuth("0", token)

    async with httpx.AsyncClient(base_url=base_url, auth=auth) as client:
        # Eksempel: Hent alle ansatte
        resp = await client.get("/employee", params={"fields": "id,firstName,lastName,*"})
        employees = resp.json()["values"]

        # Eksempel: Opprett ny kunde
        resp = await client.post("/customer", json={
            "name": "Ny Kunde AS",
            "organizationNumber": "123456789"
        })

    return {"status": "completed"}
```

### Håndtering av vedlegg
```python
import base64

for file in files:
    filename = file["filename"]
    content = base64.b64decode(file["content_base64"])
    mime = file["mime_type"]

    if mime == "application/pdf":
        # Parse PDF med pdfplumber, PyPDF2, etc.
        pass
    elif mime.startswith("image/"):
        # OCR med pytesseract, eller send til vision-modell
        pass
```

### Tripletex API-kall
```python
# Hent kunder med filtrering
resp = await client.get("/customer", params={
    "fields": "id,name,organizationNumber,*",
    "name": "Firma AS",
    "from": 0,
    "count": 100
})
customers = resp.json()["values"]

# Opprett faktura
resp = await client.post("/invoice", json={
    "customer": {"id": customer_id},
    "invoiceDate": "2026-03-20",
    "lines": [
        {"product": {"id": product_id}, "quantity": 1}
    ]
})

# Slett reiseregning
resp = await client.delete(f"/travelExpense/{expense_id}")
```

---

## Optimaliseringstips
1. **Planlegg før du kaller** — forstå oppgaven, mapper til API-kall, DERETTER utfør
2. **Unngå trial-and-error** — 4xx-feil reduserer effektivitetsbonus
3. **Minimer GET-kall** — hent kun det du trenger
4. **Batch der mulig** — noen endpoints støtter bulk-operasjoner
5. **Les feilmeldinger** — de forteller nøyaktig hva som mangler

---

## Bygging av effektiv agent
1. **Parse prompt** med LLM — forstå oppgaven og ekstraher nøkkelinfo
2. **Håndter filer** — base64-dekod PDFs/bilder, ekstraher data
3. **Map til API-kall** — planlegg sekvensen av kall
4. **Utfør og verifiser** — gjør kallene, sjekk responses
5. **Håndter feil** — retry med korrigeringer basert på feilmeldinger

---

## Deploy-alternativer

### Lokal med ngrok (anbefalt)
```bash
# Start server
uvicorn main:app --host 0.0.0.0 --port 8000

# I annet vindu — tunnel til internett
ngrok http 8000
```

> **ADVARSEL**: Cloudflare Tunnel har hard 120s timeout. Tasks kan ta opptil 300s, så lengre tasks FEILER. Bruk ngrok.

### Cloud Run på GCP
Se GCP-docs for deploy. Husk `--allow-unauthenticated` og `--timeout=300`.

---

## Constraints
- Hvert team deler én sandbox
- Alle API-kall logges for debugging
- Hver competition-submission = fersk konto (blank)
- Sandbox er persistent (for testing)
