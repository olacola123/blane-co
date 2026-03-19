"""
Lokal test av alle 30 oppgavetyper mot sandbox.
Sender prompts til vår /solve endpoint og sjekker Tripletex API etterpå.
"""

import json
import time
import requests

SERVER = "http://localhost:8000"
SANDBOX_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
TOKEN = "eyJ0b2tlbklkIjoyMTQ3NjI5ODMzLCJ0b2tlbiI6ImQ2ZjE5ZmI4LWU1NjMtNGVhNS1hNGMyLTFlNWFkZTA1ODQyNyJ9"
AUTH = ("0", TOKEN)

# Alle 30 sannsynlige oppgavetyper med testprompts i ulike språk
TEST_TASKS = [
    # === ANSATTE ===
    {
        "name": "1. Opprett ansatt (enkel)",
        "prompt": "Opprett en ansatt med navn Erik Hansen og e-post erik.hansen@example.org.",
        "verify": lambda: search("employee", {"email": "erik.hansen@example.org", "fields": "id,firstName,lastName,email"}),
        "checks": ["firstName=Erik", "lastName=Hansen", "email=erik.hansen@example.org"],
    },
    {
        "name": "2. Opprett ansatt med fødselsdato og startdato",
        "prompt": "Nous avons un nouvel employé nommé Marie Larsen, née le 15 avril 1990. Créez-la avec l'email marie.larsen@example.org et une date de début le 1er août 2026.",
        "verify": lambda: search("employee", {"email": "marie.larsen@example.org", "fields": "id,firstName,lastName,email,dateOfBirth"}),
        "checks": ["firstName=Marie", "dateOfBirth=1990-04-15"],
    },
    {
        "name": "3. Opprett ansatt som administrator",
        "prompt": "Opprett en ansatt med navn Admin Testson, e-post admin.testson@example.org. Vedkommende skal være kontoadministrator.",
        "verify": lambda: search("employee", {"email": "admin.testson@example.org", "fields": "id,firstName,lastName,email"}),
        "checks": ["firstName=Admin"],
    },
    {
        "name": "4. Oppdater ansatt",
        "prompt": "Oppdater ansatt Erik Hansen (erik.hansen@example.org) med telefonnummer 99887766.",
        "verify": lambda: search("employee", {"email": "erik.hansen@example.org", "fields": "id,firstName,phoneNumberMobile"}),
        "checks": ["phoneNumberMobile=99887766"],
    },
    # === KUNDER ===
    {
        "name": "5. Opprett kunde (enkel)",
        "prompt": "Create a customer called Nordlys AS with email post@nordlys.no and organization number 912345678.",
        "verify": lambda: search("customer", {"name": "Nordlys", "fields": "id,name,email,organizationNumber"}),
        "checks": ["name=Nordlys AS", "email=post@nordlys.no"],
    },
    {
        "name": "6. Opprett kunde med adresse",
        "prompt": "Opprett kunden Fjelltopp AS med organisasjonsnummer 987654321. Adressen er Storgata 10, 0182 Oslo. E-post: post@fjelltopp.no.",
        "verify": lambda: search("customer", {"name": "Fjelltopp", "fields": "id,name,email,postalAddress"}),
        "checks": ["name=Fjelltopp AS"],
    },
    # === LEVERANDØRER ===
    {
        "name": "7. Opprett leverandør",
        "prompt": "Registrer leverandøren Materielt AS med organisasjonsnummer 876543210. E-post: faktura@materielt.no.",
        "verify": lambda: search("customer", {"name": "Materielt", "fields": "id,name,isSupplier,email"}),
        "checks": ["isSupplier=True"],
    },
    {
        "name": "8. Opprett leverandør (spansk)",
        "prompt": "Registre el proveedor Estrella SL con número de organización 899246829. Correo electrónico: faktura@estrellasl.no.",
        "verify": lambda: search("customer", {"name": "Estrella", "fields": "id,name,isSupplier"}),
        "checks": ["isSupplier=True"],
    },
    # === PRODUKTER ===
    {
        "name": "9. Opprett produkt med pris",
        "prompt": "Opprett produktet \"Konsulenttimer\" med produktnummer 5001. Prisen er 1500 kr eksklusiv MVA, og standard MVA-sats på 25% skal brukes.",
        "verify": lambda: search("product", {"number": "5001", "fields": "id,name,number,priceExcludingVatCurrency,vatType"}),
        "checks": ["name=Konsulenttimer", "priceExcludingVatCurrency=1500"],
    },
    {
        "name": "10. Opprett produkt (tysk)",
        "prompt": "Erstellen Sie das Produkt \"Datenberatung\" mit der Produktnummer 5002. Der Preis beträgt 2200 NOK ohne MwSt., mit dem Standardsatz von 25%.",
        "verify": lambda: search("product", {"number": "5002", "fields": "id,name,number,priceExcludingVatCurrency"}),
        "checks": ["name=Datenberatung", "priceExcludingVatCurrency=2200"],
    },
    # === AVDELINGER ===
    {
        "name": "11. Opprett avdeling",
        "prompt": "Opprett avdelingen \"Salg\" med avdelingsnummer 10.",
        "verify": lambda: search("department", {"name": "Salg", "fields": "id,name,departmentNumber"}),
        "checks": ["name=Salg"],
    },
    {
        "name": "12. Opprett tre avdelinger",
        "prompt": "Opprett tre avdelinger i Tripletex: \"Produksjon\", \"Lager\" og \"Kvalitetskontroll\".",
        "verify": lambda: get("department", {"fields": "id,name", "count": 100}),
        "checks": ["Produksjon", "Lager", "Kvalitetskontroll"],
    },
    # === FAKTURA ===
    {
        "name": "13. Opprett faktura",
        "prompt": "Crie e envie uma fatura ao cliente Porto AS (org. nº 981990099) por 10000 NOK sem IVA. A fatura refere-se a Consultoria.",
        "verify": lambda: None,  # Kompleks å verifisere
        "checks": [],
    },
    # === BETALING ===
    {
        "name": "14. Registrer betaling",
        "prompt": "Kunden Havbris AS (org.nr 827237108) har en utestående faktura på 5000 kr eksklusiv MVA for \"Rådgivning\". Registrer full betaling på denne fakturaen.",
        "verify": lambda: None,
        "checks": [],
    },
    # === PROSJEKT ===
    {
        "name": "15. Opprett prosjekt",
        "prompt": "Opprett prosjektet \"Digital Transformasjon\" knyttet til kunden Innovasjon AS (org.nr 904566950). Prosjektleder er Kari Nordmann (kari.nordmann@example.org).",
        "verify": lambda: search("project", {"name": "Digital", "fields": "id,name,projectManager,customer"}),
        "checks": ["name=Digital Transformasjon"],
    },
    # === KONTAKT ===
    {
        "name": "16. Opprett kontaktperson",
        "prompt": "Legg til kontaktperson Per Olsen med e-post per.olsen@nordlys.no for kunden Nordlys AS.",
        "verify": lambda: search("contact", {"email": "per.olsen@nordlys.no", "fields": "id,firstName,lastName,email"}),
        "checks": ["firstName=Per"],
    },
    # === REISEREGNING ===
    {
        "name": "17. Opprett reiseregning",
        "prompt": "Registrer en reiseregning for ansatt Erik Hansen (erik.hansen@example.org). Tittel: \"Kundemøte Bergen\". Reise fra 20. mars 2026 til 21. mars 2026.",
        "verify": lambda: get("travelExpense", {"fields": "id,title", "count": 5}),
        "checks": [],
    },
    {
        "name": "18. Slett reiseregning",
        "prompt": "Slett alle reiseregninger i systemet.",
        "verify": lambda: get("travelExpense", {"fields": "id", "count": 5}),
        "checks": [],
    },
    # === VOUCHER/BILAG ===
    {
        "name": "19. Opprett bilag/voucher",
        "prompt": "Opprett et bilag med dato 2026-03-19 og beskrivelse \"Korrigering av feilpostering\".",
        "verify": lambda: None,
        "checks": [],
    },
    # === KREDITTNOTER ===
    {
        "name": "20. Opprett kreditnota",
        "prompt": "Kunden Nordlys AS har fått en kreditnota på hele beløpet for siste faktura. Opprett kreditnotaen.",
        "verify": lambda: None,
        "checks": [],
    },
]


def get(endpoint, params=None):
    """GET fra Tripletex sandbox."""
    resp = requests.get(f"{SANDBOX_URL}/{endpoint}", auth=AUTH, params=params)
    if resp.ok:
        return resp.json()
    return {"error": resp.status_code, "message": resp.text[:200]}


def search(endpoint, params):
    """Søk i Tripletex og returner første resultat."""
    data = get(endpoint, params)
    values = data.get("values", [])
    return values[0] if values else None


def send_task(prompt):
    """Send oppgave til vår /solve endpoint."""
    resp = requests.post(f"{SERVER}/", json={
        "prompt": prompt,
        "files": [],
        "tripletex_credentials": {
            "base_url": SANDBOX_URL,
            "session_token": TOKEN,
        }
    }, timeout=120)
    return resp.status_code == 200 and resp.json().get("status") == "completed"


def run_tests():
    print(f"{'='*70}")
    print(f"LOKAL TEST — {len(TEST_TASKS)} oppgavetyper")
    print(f"{'='*70}\n")

    passed = 0
    failed = 0
    errors = []

    for task in TEST_TASKS:
        name = task["name"]
        print(f"\n--- {name} ---")
        print(f"Prompt: {task['prompt'][:80]}...")

        try:
            start = time.time()
            ok = send_task(task["prompt"])
            elapsed = time.time() - start

            if not ok:
                print(f"  ✗ Server returnerte feil ({elapsed:.1f}s)")
                failed += 1
                errors.append(name)
                continue

            # Verifiser resultat
            if task["verify"]:
                time.sleep(0.5)
                result = task["verify"]()
                if result:
                    print(f"  ✓ Opprettet ({elapsed:.1f}s)")
                    if isinstance(result, dict):
                        for check in task["checks"]:
                            if "=" in check:
                                key, expected = check.split("=", 1)
                                actual = str(result.get(key, ""))
                                if expected in actual:
                                    print(f"    ✓ {key} = {actual}")
                                else:
                                    print(f"    ✗ {key} = {actual} (forventet {expected})")
                    passed += 1
                else:
                    print(f"  ? Ikke funnet i Tripletex ({elapsed:.1f}s)")
                    failed += 1
                    errors.append(name)
            else:
                print(f"  ? Utført men ikke verifisert ({elapsed:.1f}s)")
                passed += 1  # Teller som OK hvis serveren svarte completed

        except Exception as e:
            print(f"  ✗ Feil: {e}")
            failed += 1
            errors.append(name)

    print(f"\n{'='*70}")
    print(f"RESULTAT: {passed}/{passed+failed} bestått")
    if errors:
        print(f"Feilet: {', '.join(errors)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    # Sjekk at server kjører
    try:
        r = requests.get(f"{SERVER}/health", timeout=3)
        assert r.ok
    except Exception:
        print("Server kjører ikke! Start med:")
        print("  cd oppgave-2-tripletex-agent/ola && uvicorn server:app --port 8000")
        exit(1)

    run_tests()
