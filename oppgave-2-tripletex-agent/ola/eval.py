"""
Evaluering av Tripletex-agenten mot sandbox.
Returnerer score: antall vellykkede tasks / totalt.
Bruker unike IDer per kjøring for å unngå kollisjon med persistent sandbox.
"""

import json
import random
import string
import sys
import time
import requests

SERVER = "http://localhost:8000"
SANDBOX_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
TOKEN = "eyJ0b2tlbklkIjoyMTQ3NjI5ODMzLCJ0b2tlbiI6ImQ2ZjE5ZmI4LWU1NjMtNGVhNS1hNGMyLTFlNWFkZTA1ODQyNyJ9"
AUTH = ("0", TOKEN)

def uid():
    return 'zq' + ''.join(random.choices(string.ascii_lowercase, k=6))

def tx_get(endpoint, params=None):
    resp = requests.get(f"{SANDBOX_URL}/{endpoint}", auth=AUTH, params=params, timeout=10)
    return resp.json() if resp.ok else {}

def tx_search(endpoint, params):
    data = tx_get(endpoint, params)
    values = data.get("values", [])
    return values[0] if values else None

def tx_find(endpoint, field, substring, extra_fields=""):
    """Search all entities and find one matching substring in field."""
    fields = f"id,{field}" + (f",{extra_fields}" if extra_fields else "")
    data = tx_get(endpoint, {"fields": fields, "count": 100})
    for v in data.get("values", []):
        if substring.lower() in str(v.get(field, "")).lower():
            return v
    return None

def send_task(prompt):
    try:
        resp = requests.post(f"{SERVER}/solve", json={
            "prompt": prompt,
            "files": [],
            "tripletex_credentials": {
                "base_url": SANDBOX_URL,
                "session_token": TOKEN,
            }
        }, timeout=120)
        return resp.status_code == 200
    except Exception as e:
        print(f"    ERROR: {e}")
        return False

def run_eval():
    u = uid()
    org = str(random.randint(800000000, 899999999))  # 9 digits exactly
    org2 = str(random.randint(800000000, 899999999))
    org3 = str(random.randint(800000000, 899999999))
    org4 = str(random.randint(800000000, 899999999))
    org5 = str(random.randint(800000000, 899999999))
    org6 = str(random.randint(800000000, 899999999))
    prodnum = str(random.randint(10000, 99999))
    depnum = random.randint(100, 999)

    tasks = [
        # 1. Enkel ansatt
        {
            "name": "employee_simple",
            "prompt": f"Opprett en ansatt med navn Test{u} Hansen og e-post test{u}@example.org.",
            "verify": lambda: tx_search("employee", {"email": f"test{u}@example.org", "fields": "id,firstName,lastName,email"}),
            "check": lambda r: r and "firstName" in r,
        },
        # 2. Ansatt med fødselsdato + startdato
        {
            "name": "employee_employment",
            "prompt": f"Vi har en ny ansatt som heter Anna{u} Berg, født 15. juni 1992. Opprett vedkommende som ansatt med e-post anna{u}@example.org og startdato 1. august 2026.",
            "verify": lambda: tx_search("employee", {"email": f"anna{u}@example.org", "fields": "id,firstName,dateOfBirth"}),
            "check": lambda r: r and r.get("dateOfBirth") == "1992-06-15",
        },
        # 3. Kunde med adresse
        {
            "name": "customer_address",
            "prompt": f"Opprett kunden Fjord{u} AS med organisasjonsnummer {org}. Adressen er Storgata 10, 0182 Oslo. E-post: post@fjord{u}.no.",
            "verify": lambda: tx_find("customer", "name", f"Fjord{u}", "email,organizationNumber"),
            "check": lambda r: r and r.get("organizationNumber") == org,
        },
        # 4. Leverandør
        {
            "name": "supplier",
            "prompt": f"Registrer leverandøren Lever{u} AS med organisasjonsnummer {org2}. E-post: faktura@lever{u}.no.",
            "verify": lambda: tx_find("customer", "name", f"Lever{u}", "isSupplier,email"),
            "check": lambda r: r and r.get("isSupplier") == True,
        },
        # 5. Produkt med MVA
        {
            "name": "product",
            "prompt": f"Opprett produktet \"Tjeneste{u}\" med produktnummer {prodnum}. Prisen er 5000 kr eksklusiv MVA, og standard MVA-sats på 25% skal brukes.",
            "verify": lambda: tx_search("product", {"number": prodnum, "fields": "id,name,number,priceExcludingVatCurrency,priceIncludingVatCurrency"}),
            "check": lambda r: r and abs(float(r.get("priceExcludingVatCurrency", 0)) - 5000.0) < 1,
        },
        # 6. Tre avdelinger
        {
            "name": "departments",
            "prompt": f"Opprett tre avdelinger i Tripletex: \"Salg{u}\", \"Drift{u}\" og \"HR{u}\".",
            "verify": lambda: tx_get("department", {"fields": "id,name", "count": 100}),
            "check": lambda r: r and any(f"Salg{u}" in str(v.get("name","")) for v in r.get("values", [])),
        },
        # 7. Prosjekt
        {
            "name": "project",
            "prompt": f"Opprett prosjektet \"Prosjekt {u}\" knyttet til kunden Klient{u} AS (org.nr {org3}). Prosjektleder er Leder{u} Olsen (leder{u}@example.org).",
            "verify": lambda: tx_search("project", {"name": f"Prosjekt {u}", "fields": "id,name"}),
            "check": lambda r: r and f"Prosjekt {u}" in str(r.get("name", "")),
        },
        # 8. Faktura
        {
            "name": "invoice",
            "prompt": f"Opprett og send en faktura til kunden Faktura{u} AS (org.nr {org4}) på 10000 NOK eksklusiv MVA for \"Konsulenttimer\".",
            "verify": lambda: tx_get("invoice", {"fields": "id,customer", "invoiceDateFrom": "2026-01-01", "invoiceDateTo": "2026-12-31", "count": 100}),
            "check": lambda r: r and len(r.get("values", [])) > 0,
        },
        # 9. Betaling (bygger på faktura — lager ny kjede)
        {
            "name": "payment",
            "prompt": f"Kunden Betal{u} AS (org.nr {org5}) har ein uteståande faktura på 20000 kr eksklusiv MVA for \"Rådgivning\". Registrer full betaling på denne fakturaen.",
            "verify": lambda: tx_get("invoice", {"fields": "id,amount", "invoiceDateFrom": "2026-01-01", "invoiceDateTo": "2026-12-31", "count": 100}),
            "check": lambda r: r and len(r.get("values", [])) > 0,
        },
        # 10. Kontaktperson
        {
            "name": "contact",
            "prompt": f"Opprett kunden Kontakt{u} AS, og legg til kontaktperson Per{u} Olsen med e-post per{u}@kontakt.no.",
            "verify": lambda: tx_search("contact", {"email": f"per{u}@kontakt.no", "fields": "id,firstName,lastName,email"}),
            "check": lambda r: r and r.get("email") == f"per{u}@kontakt.no",
        },
        # 11. Leverandør (spansk)
        {
            "name": "supplier_es",
            "prompt": f"Registre el proveedor Estrella{u} SL con número de organización {org6}. Correo electrónico: faktura@estrella{u}.no.",
            "verify": lambda: tx_find("customer", "name", f"Estrella{u}", "isSupplier"),
            "check": lambda r: r and r.get("isSupplier") == True,
        },
        # 12. Ansatt (portugisisk)
        {
            "name": "employee_pt",
            "prompt": f"Temos um novo funcionário chamado João{u} Silva, nascido em 5 de setembro de 1985. Crie-o como funcionário com o e-mail joao{u}@example.org e data de início 8 de agosto de 2026.",
            "verify": lambda: tx_search("employee", {"email": f"joao{u}@example.org", "fields": "id,firstName,dateOfBirth"}),
            "check": lambda r: r and r.get("dateOfBirth") == "1985-09-05",
        },
    ]

    passed = 0
    failed = 0
    results = []

    for task in tasks:
        name = task["name"]
        sys.stdout.write(f"  {name}: ")
        sys.stdout.flush()

        start = time.time()
        ok = send_task(task["prompt"])
        elapsed = time.time() - start

        if not ok:
            print(f"FAIL (server error, {elapsed:.1f}s)")
            failed += 1
            results.append((name, "FAIL", 0))
            continue

        time.sleep(0.5)
        try:
            result = task["verify"]()
            if task["check"](result):
                print(f"PASS ({elapsed:.1f}s)")
                passed += 1
                results.append((name, "PASS", elapsed))
            else:
                print(f"FAIL (verification, {elapsed:.1f}s)")
                failed += 1
                results.append((name, "FAIL", elapsed))
        except Exception as e:
            print(f"FAIL (verify error: {e}, {elapsed:.1f}s)")
            failed += 1
            results.append((name, "FAIL", elapsed))

    total = passed + failed
    score = passed / total * 100 if total > 0 else 0
    print(f"\n  SCORE: {passed}/{total} ({score:.0f}%)")
    return passed, total, results


if __name__ == "__main__":
    try:
        r = requests.get(f"{SERVER}/health", timeout=3)
        assert r.ok
    except Exception:
        print("Server kjører ikke! Start med: uvicorn server:app --port 8000")
        sys.exit(1)

    passed, total, results = run_eval()
    # Print final score for autoresearch parsing
    print(f"\nFINAL_SCORE={passed}/{total}")
