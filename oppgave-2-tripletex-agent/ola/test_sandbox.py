"""
Test sandbox-tilgang mot Tripletex API.
Kjør: python test_sandbox.py
"""

import os
import requests

BASE_URL = os.environ.get("TRIPLETEX_URL", "https://kkpqfuj-amager.tripletex.dev/v2")
TOKEN = os.environ.get("TRIPLETEX_TOKEN", "")

if not TOKEN:
    print("Sett TRIPLETEX_TOKEN først:")
    print("  export TRIPLETEX_TOKEN='eyJ0b2tlb...'")
    exit(1)

auth = ("0", TOKEN)

print("=== Tester Tripletex Sandbox ===\n")

# Test 1: List ansatte
print("1. Henter ansatte...")
resp = requests.get(f"{BASE_URL}/employee", auth=auth, params={"fields": "id,firstName,lastName,email"})
print(f"   Status: {resp.status_code}")
if resp.ok:
    data = resp.json()
    print(f"   Ansatte funnet: {data.get('fullResultSize', len(data.get('values', [])))}")
    for emp in data.get("values", [])[:3]:
        print(f"   - {emp.get('firstName', '?')} {emp.get('lastName', '?')} (ID: {emp.get('id')})")

# Test 2: List kunder
print("\n2. Henter kunder...")
resp = requests.get(f"{BASE_URL}/customer", auth=auth, params={"fields": "id,name,email"})
print(f"   Status: {resp.status_code}")
if resp.ok:
    data = resp.json()
    print(f"   Kunder funnet: {data.get('fullResultSize', len(data.get('values', [])))}")

# Test 3: Opprett test-kunde
print("\n3. Oppretter test-kunde...")
resp = requests.post(f"{BASE_URL}/customer", auth=auth, json={
    "name": "Test AS",
    "email": "test@example.com",
    "isCustomer": True,
})
print(f"   Status: {resp.status_code}")
if resp.ok:
    customer = resp.json().get("value", {})
    print(f"   Opprettet kunde ID: {customer.get('id')}")
elif resp.status_code == 422:
    print(f"   Validering feilet: {resp.text[:200]}")

# Test 4: Sjekk tilgjengelige endepunkter
print("\n4. Sjekker ledger/account...")
resp = requests.get(f"{BASE_URL}/ledger/account", auth=auth, params={"from": 0, "count": 3, "fields": "id,number,name"})
print(f"   Status: {resp.status_code}")
if resp.ok:
    data = resp.json()
    print(f"   Kontoer funnet: {data.get('fullResultSize', '?')}")
    for acc in data.get("values", [])[:3]:
        print(f"   - {acc.get('number')}: {acc.get('name')}")

print("\n=== Sandbox-test ferdig ===")
