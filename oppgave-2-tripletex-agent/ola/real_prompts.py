"""
Faktiske prompts mottatt fra konkurransen — kjør på nytt for å forbedre.
"""

import json
import time
import requests

SERVER = "http://localhost:8000"
SANDBOX_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
TOKEN = "eyJ0b2tlbklkIjoyMTQ3NjI5ODMzLCJ0b2tlbiI6ImQ2ZjE5ZmI4LWU1NjMtNGVhNS1hNGMyLTFlNWFkZTA1ODQyNyJ9"

# Alle ekte prompts fra konkurransen (fra loggene)
REAL_PROMPTS = [
    # 0% — betalingsoppgaver
    {
        "prompt": "O cliente Floresta Lda (org. nº 906739542) tem uma fatura pendente de 6800 NOK sem IVA por \"Consultoria de dados\". Registe o pagamento total desta fatura.",
        "score": "0/7 (0%)",
        "type": "payment",
    },
    {
        "prompt": "El cliente Viento SL (org. nº 954808483) hat eine offene Rechnung über 47600 NOK ohne MwSt. für \"Systementwicklung\". Registrieren Sie die vollständige Zahlung dieser Rechnung.",
        "score": "?",
        "type": "payment",
    },
    {
        "prompt": "Kunden Strandvik AS (org.nr 827237108) har ein uteståande faktura på 30500 kr eksklusiv MVA for \"Skylagring\". Registrer full betaling på denne fakturaen.",
        "score": "2/7 (29%)",
        "type": "payment",
    },
    # 0% — prosjekt
    {
        "prompt": "Opprett prosjektet \"Migrasjon Vestfjord\" knytt til kunden Vestfjord AS (org.nr 887727872). Prosjektleiar er Liv Stølsvik (liv.stlsvik@example.org).",
        "score": "0/7 (0%)",
        "type": "project",
    },
    {
        "prompt": "Créez le projet \"Migration Lumière\" lié au client Lumière SARL (nº org. 849572458). Le chef de projet est Nathan Dubois (nathan.dubois@example.org).",
        "score": "0/7 (0%)",
        "type": "project",
    },
    {
        "prompt": "Crea el proyecto \"Migración Dorada\" vinculado al cliente Dorada SL (org. nº 904566950). El director del proyecto es Carlos Torres (carlos.torres@example.org).",
        "score": "0/7 (0%)",
        "type": "project",
    },
    # 63-86% — ansatte
    {
        "prompt": "Me har ein ny tilsett som heiter Geir Stølsvik, fødd 6. March 1990. Opprett vedkomande som tilsett med e-post geir.stlsvik@example.org og startdato 14. November 2026.",
        "score": "5/8 (63%)",
        "type": "employee",
    },
    {
        "prompt": "Temos um novo funcionário chamado João Rodrigues, nascido em 5. September 1980. Crie-o como funcionário com o e-mail joao.rodrigues@example.org e data de início 8. August 2026.",
        "score": "6/8 (75%)",
        "type": "employee",
    },
    {
        "prompt": "Vi har en ny ansatt som heter Astrid Johansen, født 10. June 1989. Opprett vedkommende som ansatt med e-post astrid.johansen@example.org og startdato 25. October 2026.",
        "score": "?",
        "type": "employee",
    },
    # 86% — kunde med adresse
    {
        "prompt": "Crea el cliente Dorada SL con número de organización 823073917. La dirección es Kirkegata 46, 4006 Stavanger. Correo: post@dorada.no.",
        "score": "6/7 (86%)",
        "type": "customer_address",
    },
    {
        "prompt": "Erstellen Sie den Kunden Brückentor GmbH mit der Organisationsnummer 846392599. Die Adresse ist Industriveien 129, 4611 Kristiansand. E-Mail: post@bruckentor.no.",
        "score": "?",
        "type": "customer_address",
    },
    {
        "prompt": "Opprett kunden Strandvik AS med organisasjonsnummer 808795132. Adressa er Parkveien 146, 7010 Trondheim. E-post: post@strandvik.no.",
        "score": "8/8 (100%)",
        "type": "customer_address",
    },
    # 86% — leverandør
    {
        "prompt": "Registrer leverandøren Skogheim AS med organisasjonsnummer 993130494. E-post: faktura@skogheim.no.",
        "score": "6/7 (86%)",
        "type": "supplier",
    },
    {
        "prompt": "Registre el proveedor Estrella SL con número de organización 899246829. Correo electrónico: faktura@estrellasl.no.",
        "score": "6/7 (86%)",
        "type": "supplier",
    },
    # 100% — produkt
    {
        "prompt": "Opprett produktet \"Konsulenttimer\" med produktnummer 9497. Prisen er 17300 kr eksklusiv MVA, og standard MVA-sats på 25 % skal brukes.",
        "score": "7/7 (100%)",
        "type": "product",
    },
    {
        "prompt": "Erstellen Sie das Produkt \"Datenberatung\" mit der Produktnummer 5524. Der Preis beträgt 22550 NOK ohne MwSt., mit dem Standardsatz von 25 %.",
        "score": "?",
        "type": "product",
    },
    {
        "prompt": "Créez le produit \"Sessão de formação\" avec le numéro de produit 6378. Le prix est de 37050 NOK hors TVA, avec le taux standard de 25 %.",
        "score": "7/7 (100%)",
        "type": "product",
    },
    # 100% — avdelinger
    {
        "prompt": "Opprett tre avdelingar i Tripletex: \"Produksjon\", \"Lager\" og \"Kvalitetskontroll\".",
        "score": "8/8 (100%)",
        "type": "departments",
    },
    {
        "prompt": "Erstellen Sie drei Abteilungen in Tripletex: \"Administrasjon\", \"Kundeservice\" und \"Markedsføring\".",
        "score": "?",
        "type": "departments",
    },
    {
        "prompt": "Opprett tre avdelingar i Tripletex: \"Markedsføring\", \"Administrasjon\" og \"Regnskap\".",
        "score": "?",
        "type": "departments",
    },
    # 100% — leverandør (registrer)
    {
        "prompt": "Registre el proveedor Dorada SL con número de organización 958363060. Correo electrónico: faktura@doradasl.no.",
        "score": "?",
        "type": "supplier",
    },
    # Faktura
    {
        "prompt": "Crie e envie uma fatura ao cliente Porto Alegre Lda (org. nº 981990099) por 46900 NOK sem IVA. A fatura refere-se a Serviço de rede.",
        "score": "0/7 (0%)",
        "type": "invoice",
    },
]


def send_task(prompt):
    resp = requests.post(f"{SERVER}/", json={
        "prompt": prompt,
        "files": [],
        "tripletex_credentials": {
            "base_url": SANDBOX_URL,
            "session_token": TOKEN,
        }
    }, timeout=120)
    return resp.status_code == 200


def run():
    # Grupper etter type og kjør de som feilet
    failed_types = {}
    for p in REAL_PROMPTS:
        score = p.get("score", "?")
        if "100%" not in score and score != "?":
            t = p["type"]
            if t not in failed_types:
                failed_types[t] = []
            failed_types[t].append(p)

    print(f"Oppgavetyper som feilet: {list(failed_types.keys())}")
    print(f"Totalt {sum(len(v) for v in failed_types.values())} prompts å teste\n")

    for task_type, prompts in failed_types.items():
        print(f"\n{'='*60}")
        print(f"TYPE: {task_type}")
        print(f"{'='*60}")

        for p in prompts:
            print(f"\nPrompt: {p['prompt'][:80]}...")
            print(f"Forrige score: {p['score']}")

            start = time.time()
            ok = send_task(p["prompt"])
            elapsed = time.time() - start

            if ok:
                print(f"✓ Utført ({elapsed:.1f}s)")
            else:
                print(f"✗ Feilet ({elapsed:.1f}s)")


if __name__ == "__main__":
    try:
        r = requests.get(f"{SERVER}/health", timeout=3)
        assert r.ok
    except Exception:
        print("Server kjører ikke!")
        exit(1)
    run()
