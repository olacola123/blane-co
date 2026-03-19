"""
Simuler hva competition-serveren sender til /solve.
Tester mot sandbox uten å starte FastAPI-serveren.
"""

import json
import sys
sys.path.insert(0, ".")
from server import TripletexClient, detect_task_type, TASK_HANDLERS

BASE_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
TOKEN = "eyJ0b2tlbklkIjoyMTQ3NjI5ODMzLCJ0b2tlbiI6ImQ2ZjE5ZmI4LWU1NjMtNGVhNS1hNGMyLTFlNWFkZTA1ODQyNyJ9"

# Testoppgaver som ligner competition-prompts
TEST_PROMPTS = [
    "Opprett en ansatt med navn Ola Nordmann, ola@example.org. Han skal være kontoadministrator.",
    "Create an employee named Erik Hansen with email erik@hansen.no",
    "Opprett en kunde kalt Bergen Consulting AS med e-post post@bergen.no",
    "Create a customer called Oslo Tech AS with email hello@oslotech.no",
    "Opprett et produkt Konsulenttime med pris 1200",
]

tx = TripletexClient(BASE_URL, TOKEN)

for i, prompt in enumerate(TEST_PROMPTS):
    print(f"\n{'='*60}")
    print(f"Test {i+1}: {prompt[:80]}...")

    task_type = detect_task_type(prompt)
    print(f"Detektert type: {task_type}")

    handler = TASK_HANDLERS.get(task_type)
    if handler:
        try:
            handler(tx, prompt)
            print("✓ OK")
        except Exception as e:
            print(f"✗ Feil: {e}")
    else:
        print(f"✗ Ukjent oppgavetype")

print(f"\n{'='*60}")
print("Ferdig! Sjekk Tripletex UI for å se hva som ble opprettet.")
