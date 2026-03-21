"""
Tripletex AI Accounting Agent — NM i AI 2026
Mathea / v16 — Clean architecture with task detection + targeted recipes

ARCHITECTURE:
  1. TOOLS          — 4 API tools (GET/POST/PUT/DELETE)
  2. TASK DETECT    — classify task type from prompt keywords (any language)
  3. PRE-FETCH      — parallel GETs (GET calls are FREE for efficiency score!)
  4. SYSTEM PROMPT  — with task-specific recipe injected
  5. AGENT LOOP     — Claude with tools, iteration limit per task complexity
  6. /solve         — main endpoint

SCORING REMINDER:
  score = correctness × tier_multiplier × efficiency_bonus
  - GET calls: FREE (don't count against efficiency)
  - Write calls (POST/PUT/DELETE): minimize these!
  - 4xx errors: penalize efficiency bonus
  - Perfect correctness required for efficiency bonus to apply
  → Pre-fetch everything via GET, then make exactly the right write calls

CURRENT STATUS (from PLAN.md, 21. mars):
  50.46 points, #26, 28/30 tasks covered
  Task 25 (credit note T3): 1.50 vs Ninjas' 6.00  ← biggest easy win
  Task 22 (receipt/kvittering T3): 0/10            ← account issue?
  Task 12 (supplier invoice T2): 0                 ← voucher not hitting
"""

import base64
import concurrent.futures
import io
import json
import logging
import os
import re
import time
import traceback
from datetime import date, timedelta
from pathlib import Path

import anthropic
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("agent")

app = FastAPI()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ═══════════════════════════════════════════════════════════════
# SECTION 1: TOOLS
# GET calls are FREE for efficiency. POST/PUT/DELETE are scored.
# ═══════════════════════════════════════════════════════════════

TOOLS = [
    {
        "name": "tripletex_get",
        "description": "GET request to Tripletex API. FREE — does not count against efficiency. Use freely.",
        "input_schema": {
            "type": "object",
            "properties": {
                "endpoint": {"type": "string", "description": "API path, e.g. /employee or /invoice"},
                "query_params": {"type": "string", "description": "Query string, e.g. 'fields=id,name&count=10'", "default": ""},
            },
            "required": ["endpoint"],
        },
    },
    {
        "name": "tripletex_post",
        "description": "POST request to create a resource. COUNTS against efficiency — get it right first try!",
        "input_schema": {
            "type": "object",
            "properties": {
                "endpoint": {"type": "string"},
                "body": {"type": "object"},
                "query_params": {"type": "string", "default": ""},
            },
            "required": ["endpoint", "body"],
        },
    },
    {
        "name": "tripletex_put",
        "description": "PUT to update or trigger action. COUNTS against efficiency. For /:action endpoints, use query_params NOT body.",
        "input_schema": {
            "type": "object",
            "properties": {
                "endpoint": {"type": "string"},
                "body": {"type": "object", "default": {}},
                "query_params": {"type": "string", "default": ""},
            },
            "required": ["endpoint"],
        },
    },
    {
        "name": "tripletex_delete",
        "description": "DELETE a resource. COUNTS against efficiency.",
        "input_schema": {
            "type": "object",
            "properties": {
                "endpoint": {"type": "string", "description": "e.g. /travelExpense/789"},
            },
            "required": ["endpoint"],
        },
    },
]


# ═══════════════════════════════════════════════════════════════
# SECTION 2: TASK DETECTION
# Classify task type from prompt (works across all 7 languages).
# Used to: set max iterations, inject task-specific recipe hint.
# ═══════════════════════════════════════════════════════════════

# Max write-call iterations per task type (fewer = higher efficiency bonus)
TASK_CONFIG = {
    # (task_type): (max_iterations, description)
    "credit_note":        (6,  "Tier 3 — find existing invoice → PUT /:createCreditNote"),
    "receipt_voucher":    (6,  "Tier 3 — POST /ledger/voucher with 2 postings"),
    "salary":             (12, "Tier 2/3 — salary transaction or voucher fallback"),
    "supplier_invoice":   (10, "Tier 2 — POST /ledger/voucher with 3 postings"),
    "travel_expense":     (10, "Tier 2 — POST /travelExpense + /perDiemCompensation"),
    "delete_travel":      (8,  "Tier 3 — state machine: unapprove → delete"),
    "employee_pdf":       (12, "Tier 3 — extract from PDF, create employee + employment"),
    "update_employee":    (8,  "Tier 3 — PUT existing employee"),
    "contact_person":     (8,  "Tier 3 — find/create customer → POST /contact"),
    "year_end":           (15, "Tier 3 — complex multi-step"),
    "voucher":            (8,  "Tier 3 — standalone voucher"),
    "payment":            (8,  "Tier 2 — find existing invoice → PUT /:payment"),
    "reverse_payment":    (8,  "Tier 2 — find invoice → PUT /:payment with negative"),
    "accounting_dim":     (10, "Tier 2 — POST /ledger/accountingDimensionName + Value"),
    "invoice":            (10, "Tier 2 — customer + product + order + orderline + invoice"),
    "invoice_payment":    (12, "Tier 2 — full invoice flow + payment"),
    "project":            (10, "Tier 1/2 — employee + entitlements + customer + project"),
    "timesheet_invoice":  (14, "Tier 2 — project + timesheet + invoice"),
    "order":              (10, "Tier 2 — customer + product + order + orderlines"),
    "employee":           (8,  "Tier 1 — POST /employee + /employment + /employment/details"),
    "customer":           (6,  "Tier 1 — POST /customer"),
    "supplier":           (6,  "Tier 1 — POST /customer with isSupplier:true"),
    "product":            (5,  "Tier 1 — POST /product"),
    "department":         (6,  "Tier 1 — POST /department × N"),
    "bank_reconciliation":(12, "Tier 3 — bank reconciliation"),
    "update_customer":    (6,  "Tier 3 — PUT existing customer"),
    "unknown":            (15, "Unknown — use full budget"),
}

def detect_task_type(prompt: str) -> str:
    """Classify task type from prompt keywords (multilingual)."""
    p = prompt.lower()

    # Credit note — MUST check before invoice (contains "faktura")
    if any(w in p for w in [
        "kreditnota", "credit note", "nota de crédito", "note de crédit",
        "gutschrift", "kreditera", "opphev faktura", "annuller faktura", "avskriv faktura",
        "creditnota", "nota crédito",
    ]):
        return "credit_note"

    # Payment reversal
    if any(w in p for w in ["tilbakefør", "reverse", "revertir", "inverser", "stornieren", "reverter"]):
        if any(w in p for w in ["betaling", "payment", "pago", "paiement", "zahlung"]):
            return "reverse_payment"

    # Receipt / expense voucher (from image or text)
    if any(w in p for w in [
        "kvittering", "receipt", "recibo", "quittance", "quittung", "reçu",
        "expense", "utlegg", "bon ", "usb", "hub", "kjøp", "purchase",
    ]):
        return "receipt_voucher"

    # Salary / payroll
    if any(w in p for w in [
        "lønn", "lønning", "payroll", "payslip", "salary", "nómina", "salário",
        "salaire", "gehalt", "lohn", "grunnlønn", "bonus",
    ]):
        return "salary"

    # Supplier invoice
    if any(w in p for w in [
        "leverandørfaktura", "supplier invoice", "factura del proveedor",
        "fatura do fornecedor", "facture fournisseur", "lieferantenrechnung",
        "inkomende faktura", "incoming invoice", "inv-",
    ]):
        return "supplier_invoice"

    # Delete travel expense
    if any(w in p for w in ["slett", "delete", "eliminar", "supprimer", "löschen", "eliminar", "excluir"]):
        if any(w in p for w in ["reiseregning", "travel expense", "reisekosten", "note de frais", "despesa de viagem"]):
            return "delete_travel"

    # Travel expense
    if any(w in p for w in [
        "reiseregning", "travel expense", "despesa de viagem",
        "reisekostenabrechnung", "note de frais", "dietas", "reiseutgift",
    ]):
        return "travel_expense"

    # PDF contract / employee from PDF
    if any(f in p for f in [".pdf", "kontrakt", "arbeidskontrakt", "employment contract",
                              "arbeitsvertrag", "contrato", "contrat"]):
        return "employee_pdf"

    # Update employee
    if any(w in p for w in ["oppdater", "update", "actualizar", "mettre à jour", "aktualisieren",
                              "endre", "change", "modify"]):
        if any(w in p for w in ["ansatt", "employee", "empleado", "employé", "mitarbeiter", "funcionário"]):
            return "update_employee"

    # Contact person
    if any(w in p for w in [
        "kontaktperson", "contact person", "persona de contacto",
        "personne de contact", "ansprechpartner", "pessoa de contato",
    ]):
        return "contact_person"

    # Year-end / accounting
    if any(w in p for w in [
        "årsoppgjør", "year-end", "cierre anual", "bilan annuel",
        "jahresabschluss", "avskrivning", "depreciation", "avskriv",
        "bankavstemmig", "bank reconciliation",
    ]):
        return "year_end"

    # Standalone voucher
    if any(w in p for w in ["bilag", "voucher", "asiento", "écriture comptable", "buchung"]):
        if not any(w in p for w in ["faktura", "invoice", "lønn", "salary"]):
            return "voucher"

    # Payment (register payment for existing invoice)
    if any(w in p for w in [
        "betaling", "payment", "pago", "paiement", "zahlung", "pagamento",
        "registrer betaling", "register payment", "betale",
    ]):
        return "payment"

    # Accounting dimensions
    if any(w in p for w in [
        "regnskapsdimensjon", "accounting dimension", "dimensión contable",
        "dimension comptable", "kostnadsbærer",
    ]):
        return "accounting_dim"

    # Invoice (check before customer — invoice includes customer creation)
    if any(w in p for w in [
        "faktura", "invoice", "factura", "facture", "rechnung", "fatura",
    ]):
        if any(w in p for w in [
            "betaling", "payment", "pago", "paiement", "zahlung", "betale",
        ]):
            return "invoice_payment"
        return "invoice"

    # Order
    if any(w in p for w in ["ordre", "order", "pedido", "commande", "bestellung", "bestilling"]):
        return "order"

    # Timesheet + project invoice
    if any(w in p for w in ["timesheet", "timer", "hours", "horas", "heures", "stunden", "timen"]):
        return "timesheet_invoice"

    # Project
    if any(w in p for w in ["prosjekt", "project", "proyecto", "projet", "projekt"]):
        return "project"

    # Employee
    if any(w in p for w in [
        "ansatt", "employee", "empleado", "employé", "mitarbeiter", "funcionário",
        "medarbeider", "tilsett", "tilsatt",
    ]):
        return "employee"

    # Customer
    if any(w in p for w in ["kunde", "customer", "cliente", "client", "klient", "kunden"]):
        return "customer"

    # Supplier
    if any(w in p for w in [
        "leverandør", "supplier", "proveedor", "fournisseur", "lieferant", "fornecedor",
    ]):
        return "supplier"

    # Product
    if any(w in p for w in ["produkt", "product", "producto", "produit", "artikel"]):
        return "product"

    # Department
    if any(w in p for w in ["avdeling", "department", "departamento", "département", "abteilung"]):
        return "department"

    # Bank reconciliation
    if any(w in p for w in [
        "bankavstemmig", "bank reconciliation", "reconcilia", "bankutskrift",
        "bankkonto", "bank statement", "kontoavstemming",
    ]):
        return "bank_reconciliation"

    # Update customer
    if any(w in p for w in ["oppdater", "update", "actualizar", "mettre à jour", "aktualisieren", "endre"]):
        if any(w in p for w in ["kunde", "customer", "cliente", "client", "klient"]):
            return "update_customer"

    # Year-end / depreciation / annual closing
    if any(w in p for w in [
        "årsoppgjør", "year-end", "year end", "cierre", "bilan annuel",
        "jahresabschluss", "avskrivning", "depreciation", "avskriv",
        "bankavstemmig", "bank reconciliation", "nedskrivning",
    ]):
        return "year_end"

    return "unknown"


# ═══════════════════════════════════════════════════════════════
# SECTION 3: API EXECUTION + INTERCEPTORS
# Intercepts prevent known bad patterns before they hit the API.
# ═══════════════════════════════════════════════════════════════

def execute_tool(name: str, inp: dict, session: requests.Session,
                 base_url: str, trace: list, ctx: dict = None) -> str:
    endpoint = inp.get("endpoint", "")
    qp = inp.get("query_params", "") or ""
    params = dict(p.split("=", 1) for p in qp.split("&") if "=" in p) if qp else None
    body = inp.get("body")

    # ── INTERCEPT: /supplier → /customer with isSupplier:true ──
    if name == "tripletex_post" and endpoint.strip("/") == "supplier":
        endpoint = "/customer"
        if body:
            body.setdefault("isSupplier", True)
            body.setdefault("isCustomer", False)
            body.setdefault("language", "NO")
            if body.get("email"):
                body.setdefault("invoiceEmail", body["email"])
            if body.get("organizationNumber"):
                body["organizationNumber"] = body["organizationNumber"].replace(" ", "")
        log.info("Intercepted POST /supplier → POST /customer isSupplier:true")

    # ── INTERCEPT: /customer — ensure required fields ──
    if name == "tripletex_post" and endpoint.strip("/") == "customer" and body:
        body.setdefault("phoneNumber", "")
        body.setdefault("language", "NO")
        if body.get("email"):
            body.setdefault("invoiceEmail", body["email"])

    # ── INTERCEPT: /incomingInvoice or /supplierInvoice → BETA, block it ──
    if name == "tripletex_post" and any(x in endpoint for x in ["incomingInvoice", "supplierInvoice"]):
        log.info(f"Blocked BETA endpoint {endpoint} → return voucher hint")
        trace.append({"tool": name, "endpoint": endpoint, "status": 403, "error": "BETA"})
        return json.dumps({
            "status": 403,
            "message": "BETA endpoint — use voucher: POST /ledger/voucher with expense(debit,vatType:1) + 2400(credit,supplier). See SUPPLIER INVOICE recipe."
        })

    # ── INTERCEPT: PUT /employee — preserve real dateOfBirth ──
    if name == "tripletex_put" and "/employee/" in endpoint and body and ctx:
        if body.get("dateOfBirth") in ("1985-01-01", "1990-05-15", None, ""):
            emp_id_str = endpoint.rstrip("/").split("/")[-1]
            try:
                emp_id = int(emp_id_str)
                for e in ctx.get("employees", []):
                    if e.get("id") == emp_id and e.get("dateOfBirth") and e["dateOfBirth"] not in ("1985-01-01", "1990-05-15"):
                        body["dateOfBirth"] = e["dateOfBirth"]
                        log.info(f"Fixed dateOfBirth → {e['dateOfBirth']} for emp {emp_id}")
                        break
            except (ValueError, TypeError):
                pass

    # ── INTERCEPT: POST /employee — set realistic dateOfBirth if missing ──
    if name == "tripletex_post" and endpoint.strip("/") == "employee" and body:
        if body.get("dateOfBirth") in (None, "", "1985-01-01"):
            body["dateOfBirth"] = "1990-05-15"

    # ── INTERCEPT: PUT /project — don't hardcode project number ──
    if name == "tripletex_put" and "/project/" in endpoint and body:
        if body.get("number") in ("1", "2", 1, 2):
            body.pop("number", None)
            log.info("Removed hardcoded project number from PUT")

    url = f"{base_url}/{endpoint.lstrip('/')}"
    try:
        if name == "tripletex_get":
            resp = session.get(url, params=params, timeout=30)
        elif name == "tripletex_post":
            resp = session.post(url, json=body, params=params, timeout=30)
        elif name == "tripletex_put":
            if body and body != {}:
                resp = session.put(url, json=body, params=params, timeout=30)
            else:
                resp = session.put(url, params=params, timeout=30)
        elif name == "tripletex_delete":
            resp = session.delete(url, timeout=30)
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})
    except (requests.Timeout, requests.ConnectionError) as e:
        log.error(f"{name} {endpoint} → {e}")
        trace.append({"tool": name, "endpoint": endpoint, "status": 0, "error": str(e)[:200]})
        return json.dumps({"error": str(e)[:200]})

    status = resp.status_code
    trace.append({
        "tool": name, "endpoint": endpoint, "status": status,
        "error": resp.text[:300] if status >= 400 else None,
    })
    if status >= 400:
        log.error(f"{name} {endpoint} → {status}: {resp.text[:300]}")
    else:
        log.info(f"{name} {endpoint} → {status}")

    try:
        result = resp.json()
    except Exception:
        result = {"raw": resp.text[:500]}

    text = json.dumps(result, ensure_ascii=False)
    if len(text) > 2000:
        if isinstance(result, dict) and "values" in result:
            result["values"] = result["values"][:5]
            result["_truncated"] = True
            result["_note"] = "Use query params to filter instead of browsing all results"
            text = json.dumps(result, ensure_ascii=False)
        if len(text) > 2000:
            text = text[:2000]
    return text


# ═══════════════════════════════════════════════════════════════
# SECTION 4: PRE-FETCH (parallel GETs — all FREE for efficiency)
# The more context we fetch here, the fewer tool calls Claude needs.
# ═══════════════════════════════════════════════════════════════

def prefetch_context(session: requests.Session, base_url: str) -> dict:
    ctx = {}

    def fetch(name, endpoint, params):
        try:
            r = session.get(f"{base_url}/{endpoint}", params=params, timeout=12)
            if r.status_code == 200:
                return name, r.json().get("values", r.json().get("value", []))
        except Exception as e:
            log.warning(f"Prefetch {name}: {e}")
        return name, []

    fetches = [
        ("employees",       "employee",              {"fields": "id,firstName,lastName,email,dateOfBirth,userType,version,department", "count": "100"}),
        ("departments",     "department",             {"fields": "id,name,departmentNumber", "count": "20"}),
        ("divisions",       "division",               {"fields": "id,name", "count": "10"}),
        ("activities",      "activity",               {"fields": "id,name", "count": "30"}),
        ("payment_types",   "invoice/paymentType",    {"fields": "id,description", "count": "10"}),
        ("products",        "product",                {"fields": "id,name,number,priceExcludingVatCurrency", "count": "100"}),
        ("customers",       "customer",               {"fields": "id,name,organizationNumber,email", "count": "100"}),
        ("suppliers",       "supplier",               {"fields": "id,name,organizationNumber,email", "count": "100"}),
        ("cost_categories", "travelExpense/costCategory",  {"fields": "id,description", "count": "50"}),
        ("travel_pay_types","travelExpense/paymentType",   {"fields": "id,description", "count": "10"}),
        ("rate_categories", "travelExpense/rateCategory",  {"fields": "id,name,type,fromDate,toDate", "count": "500"}),
        # Salary types
        ("sal_2000", "salary/type", {"number": "2000", "fields": "id,number,name"}),
        ("sal_2002", "salary/type", {"number": "2002", "fields": "id,number,name"}),
        ("sal_2001", "salary/type", {"number": "2001", "fields": "id,number,name"}),
        # Ledger accounts — all common ones (GET is FREE!)
        ("acct_1500", "ledger/account", {"number": "1500", "fields": "id,number,name"}),
        ("acct_1920", "ledger/account", {"number": "1920", "fields": "id,number,name"}),
        ("acct_2400", "ledger/account", {"number": "2400", "fields": "id,number,name"}),
        ("acct_2710", "ledger/account", {"number": "2710", "fields": "id,number,name"}),
        ("acct_2780", "ledger/account", {"number": "2780", "fields": "id,number,name"}),
        ("acct_3000", "ledger/account", {"number": "3000", "fields": "id,number,name"}),
        ("acct_5000", "ledger/account", {"number": "5000", "fields": "id,number,name"}),
        ("acct_6300", "ledger/account", {"number": "6300", "fields": "id,number,name"}),  # Husleie/leie
        ("acct_6340", "ledger/account", {"number": "6340", "fields": "id,number,name"}),  # Lys/varme
        ("acct_6500", "ledger/account", {"number": "6500", "fields": "id,number,name"}),  # Kontorrekvisita
        ("acct_6700", "ledger/account", {"number": "6700", "fields": "id,number,name"}),  # Reklame
        ("acct_6800", "ledger/account", {"number": "6800", "fields": "id,number,name"}),  # IKT/data-utstyr ← FIX Task 22
        ("acct_6900", "ledger/account", {"number": "6900", "fields": "id,number,name"}),  # Telefon
        ("acct_7100", "ledger/account", {"number": "7100", "fields": "id,number,name"}),  # Bil
        ("acct_7140", "ledger/account", {"number": "7140", "fields": "id,number,name"}),  # Reise
        ("acct_7300", "ledger/account", {"number": "7300", "fields": "id,number,name"}),  # Representasjon
        ("acct_7350", "ledger/account", {"number": "7350", "fields": "id,number,name"}),  # Representasjon mat
        ("acct_7400", "ledger/account", {"number": "7400", "fields": "id,number,name"}),  # Gaver
        ("acct_7700", "ledger/account", {"number": "7700", "fields": "id,number,name"}),  # Annen driftskostnad
        ("acct_8700", "ledger/account", {"number": "8700", "fields": "id,number,name"}),  # Skattekostnad
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        for name, values in ex.map(lambda f: fetch(*f), fetches):
            if name == "employees":
                ctx["employees"] = values if isinstance(values, list) else []
            elif name == "departments":
                ctx["departments"] = values if isinstance(values, list) else []
                if ctx["departments"]:
                    ctx["default_department_id"] = ctx["departments"][0]["id"]
            elif name == "divisions":
                ctx["divisions"] = values if isinstance(values, list) else []
                if ctx["divisions"]:
                    ctx["default_division_id"] = ctx["divisions"][0]["id"]
            elif name == "activities":
                ctx["activities"] = values if isinstance(values, list) else []
            elif name == "payment_types":
                ctx["payment_types"] = values if isinstance(values, list) else []
            elif name == "products":
                ctx["products"] = values if isinstance(values, list) else []
            elif name == "customers":
                ctx["customers"] = values if isinstance(values, list) else []
            elif name == "suppliers":
                ctx["suppliers"] = values if isinstance(values, list) else []
            elif name == "cost_categories":
                ctx["cost_categories"] = values if isinstance(values, list) else []
            elif name == "travel_pay_types":
                ctx["travel_pay_types"] = values if isinstance(values, list) else []
            elif name == "rate_categories":
                vals = values if isinstance(values, list) else []
                for rc in vals:
                    n = str(rc.get("name") or "")
                    fr = str(rc.get("fromDate") or "")
                    to_d = str(rc.get("toDate") or "")
                    if fr >= "2026" and to_d >= "2026" and "innland" in n.lower():
                        if "Overnatting" in n and "per_diem_overnight_id" not in ctx:
                            ctx["per_diem_overnight_id"] = rc["id"]
                            log.info(f"Per diem overnight: id={rc['id']}")
                        elif "Dag" in n and "per_diem_day_id" not in ctx:
                            ctx["per_diem_day_id"] = rc["id"]
                            log.info(f"Per diem day: id={rc['id']}")
            elif name.startswith("sal_"):
                num = name.split("_")[1]
                if values and isinstance(values, list) and values:
                    ctx.setdefault("salary_types", {})[num] = {
                        "id": values[0]["id"],
                        "number": num,
                        "name": values[0].get("name", ""),
                    }
            elif name.startswith("acct_"):
                num = name.split("_")[1]
                vals = values if isinstance(values, list) else []
                if vals:
                    ctx.setdefault("ledger_accounts", {})[num] = vals[0]["id"]

    # Fetch employments for pre-fetched employees (parallel, FREE)
    if ctx.get("employees"):
        def fetch_employment(emp):
            try:
                r = session.get(
                    f"{base_url}/employee/employment",
                    params={"employeeId": emp["id"], "fields": "id,startDate,division", "count": "5"},
                    timeout=10,
                )
                if r.status_code == 200:
                    return emp["id"], r.json().get("values", [])
            except Exception:
                pass
            return emp["id"], []

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            for emp_id, empls in ex.map(fetch_employment, ctx["employees"]):
                for e in ctx["employees"]:
                    if e["id"] == emp_id:
                        e["_employments"] = empls
                        break

    log.info(f"Pre-fetch done: {len(ctx.get('employees', []))} employees, "
             f"{len(ctx.get('departments', []))} depts, "
             f"accounts={list(ctx.get('ledger_accounts', {}).keys())}")
    return ctx


def format_context(ctx: dict) -> str:
    """Format pre-fetched data for Claude."""
    lines = []

    if ctx.get("employees"):
        lines.append("EXISTING EMPLOYEES (use these IDs directly — do NOT GET /employee again):")
        for e in ctx["employees"]:
            dept = e.get("department", {}) or {}
            has_emp = bool(e.get("_employments"))
            lines.append(
                f"  id={e['id']} v={e.get('version',0)} | {e['firstName']} {e['lastName']} "
                f"| {e['email']} | dob={e.get('dateOfBirth','?')} | type={e.get('userType','?')} "
                f"| dept={dept.get('id','?')} | hasEmployment={'YES' if has_emp else 'NO'}"
            )

    if ctx.get("departments"):
        used_nums = [d.get("departmentNumber", 0) for d in ctx["departments"]]
        dept_list = ["{name}(id={id},num={num})".format(name=d["name"], id=d["id"], num=d.get("departmentNumber", "?")) for d in ctx["departments"]]
        lines.append("EXISTING DEPARTMENTS: " + str(dept_list))
        lines.append("  → Next dept numbers: {}, {}, {}".format(max(used_nums)+1, max(used_nums)+2, max(used_nums)+3))
    else:
        lines.append("EXISTING DEPARTMENTS: none — use departmentNumber=1,2,3 for new ones")

    if ctx.get("divisions"):
        div_list = ["{name}(id={id})".format(name=d["name"], id=d["id"]) for d in ctx["divisions"]]
        lines.append("DIVISIONS: " + str(div_list))
    else:
        lines.append("DIVISIONS: none (create one if needed for employment)")

    if ctx.get("activities"):
        act_list = ["{name}(id={id})".format(name=a["name"], id=a["id"]) for a in ctx["activities"][:5]]
        lines.append("ACTIVITIES: " + str(act_list))

    if ctx.get("payment_types"):
        pt_list = ["{desc}(id={id})".format(desc=p.get("description", "?"), id=p["id"]) for p in ctx["payment_types"]]
        lines.append("INVOICE PAYMENT TYPES: " + str(pt_list))

    if ctx.get("products"):
        prod_list = ["{name}(id={id},num={num})".format(name=p["name"], id=p["id"], num=p.get("number", "?")) for p in ctx["products"][:5]]
        lines.append("EXISTING PRODUCTS: " + str(prod_list))

    if ctx.get("customers"):
        cust_list = ["{name}(id={id})".format(name=c["name"], id=c["id"]) for c in ctx["customers"][:5]]
        lines.append("EXISTING CUSTOMERS: " + str(cust_list))

    if ctx.get("suppliers"):
        supp_list = ["{name}(id={id})".format(name=s["name"], id=s["id"]) for s in ctx["suppliers"][:5]]
        lines.append("EXISTING SUPPLIERS: " + str(supp_list))

    if ctx.get("ledger_accounts"):
        lines.append("LEDGER ACCOUNT IDs (use directly — do NOT GET /ledger/account again):")
        acct_names = {
            "1500": "Kundefordringer", "1920": "Bankinnskudd", "2400": "Leverandørgjeld",
            "2710": "Inng.MVA 25%", "2780": "Skyldig lønn", "3000": "Salgsinntekt",
            "5000": "Lønn", "6300": "Husleie", "6340": "Lys/varme", "6500": "Kontorrekvisita",
            "6700": "Reklame", "6800": "IKT/data-utstyr", "6900": "Telefon", "7100": "Bil",
            "7140": "Reise", "7300": "Representasjon", "7350": "Rep.mat", "7400": "Gaver",
            "7700": "Annen drift", "8700": "Skatt",
        }
        for num, acc_id in sorted(ctx["ledger_accounts"].items()):
            lines.append("  {} ({}): id={}".format(num, acct_names.get(num, "?"), acc_id))

    if ctx.get("salary_types"):
        lines.append("SALARY TYPES: " + str(ctx["salary_types"]))

    if ctx.get("per_diem_overnight_id") or ctx.get("per_diem_day_id"):
        lines.append("PER DIEM: overnight_id={}  day_id={}".format(
            ctx.get("per_diem_overnight_id", "?"), ctx.get("per_diem_day_id", "?")))

    if ctx.get("cost_categories"):
        cc_list = ["{desc}(id={id})".format(desc=c.get("description", "?"), id=c["id"]) for c in ctx["cost_categories"][:5]]
        lines.append("TRAVEL COST CATEGORIES: " + str(cc_list))

    if ctx.get("travel_pay_types"):
        tpt_list = ["{desc}(id={id})".format(desc=p.get("description", "?"), id=p["id"]) for p in ctx["travel_pay_types"]]
        lines.append("TRAVEL PAYMENT TYPES: " + str(tpt_list))

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# SECTION 5: SYSTEM PROMPT
# Core rules + per-task recipes.
# EFFICIENCY PRINCIPLE: Every POST/PUT/DELETE counts. Plan first!
# ═══════════════════════════════════════════════════════════════

TASK_RECIPES = {
    "credit_note": """
── CREDIT NOTE (T3 — 2 write calls max!) ──
SCENARIO A: Task says "create credit note" for an EXISTING invoice (most common in T3):
  1. GET /invoice?invoiceDateFrom=2020-01-01&invoiceDateTo=2027-01-01&fields=id,invoiceNumber,amount,customer (FREE)
  2. PUT /invoice/ID/:createCreditNote query_params: date={TODAY}&sendToCustomer=false
  → TOTAL: 1 GET (free) + 1 PUT = PERFECT EFFICIENCY

SCENARIO B: Task asks you to create a full invoice AND credit note:
  1. POST /customer
  2. POST /product
  3. POST /order
  4. POST /order/orderline
  5. POST /invoice query_params: sendToCustomer=false
  6. PUT /invoice/ID/:createCreditNote query_params: date={TODAY}&sendToCustomer=false

DO NOT create new customer/product/order/invoice if the invoice already exists!
""",

    "receipt_voucher": """
── RECEIPT / EXPENSE VOUCHER (T3) ──
ACCOUNT MAPPING (use IDs from LEDGER ACCOUNT IDs above):
  6300 = Husleie/leie lokaler
  6340 = Lys, varme, vann
  6500 = Kontorrekvisita, forbruksmateriell
  6700 = Reklame, markedsføring
  6800 = IKT-utstyr, datautstyr, elektronikk (USB-hub, laptop, osv.)  ← VIKTIG
  6900 = Telefon, internett
  7100 = Bil, drivstoff
  7140 = Reise, overnatting
  7300 = Representasjon, kundemøter
  7350 = Representasjon mat/drikke
  7700 = Annen driftskostnad

POST /ledger/voucher with EXACTLY 2 postings:
  {
    "date": "RECEIPT_DATE",  ← use date from receipt, NOT today!
    "description": "Kvittering [item] [supplier]",
    "postings": [
      {
        "account": {"id": EXPENSE_ACCT_ID},
        "amountGross": TOTAL_INCL_VAT,
        "amountGrossCurrency": TOTAL_INCL_VAT,
        "date": "RECEIPT_DATE",
        "row": 1,
        "vatType": {"id": 1},     ← id:1 = inngående 25% MVA (auto-splits to 2710!)
        "department": {"id": DEPT_ID}  ← include if department mentioned
      },
      {
        "account": {"id": ACCT_1920_ID},  ← bank account (payment from bank)
        "amountGross": -TOTAL_INCL_VAT,
        "amountGrossCurrency": -TOTAL_INCL_VAT,
        "date": "RECEIPT_DATE",
        "row": 2,
        "vatType": {"id": 0}
      }
    ]
  }
  Food/restaurant: use vatType id:11 (15% MVA) instead of id:1
  No VAT (tax exempt purchase): use vatType id:0 on both postings
""",

    "salary": """
── SALARY / PAYROLL (T2/T3) ──
Step 1: Find employee by email in EXISTING EMPLOYEES (do NOT GET /employee)
Step 2: PUT /employee/ID to update name if needed (keep dateOfBirth FROM CONTEXT!)
Step 3: If hasEmployment=NO → POST /division + /employee/employment + /employee/employment/details
Step 4: Try POST /salary/transaction query_params: generateTaxDeduction=true
  Body: {date:TODAY, month:MM, year:YYYY, payslips:[{
    employee:{id:EMP_ID}, date:TODAY, year:YYYY, month:MM,
    specifications:[
      {salaryType:{id:SAL_2000_ID}, rate:BASE, count:1, amount:BASE},
      {salaryType:{id:SAL_2002_ID}, rate:BONUS, count:1, amount:BONUS}
    ]
  }]}
  If → 201: STOP! Do NOT create manual voucher (duplicate!)
  If → 403: fall back to voucher IMMEDIATELY:
    POST /ledger/voucher with accounts 5000 (debit) + 2780 (credit)
    NEVER use 1920 for salary credit — ALWAYS 2780!
""",

    "supplier_invoice": """
── SUPPLIER INVOICE (T2 — voucher only!) ──
/incomingInvoice and /supplierInvoice are BETA → always 403. Use voucher!

Step 1: Create supplier: POST /customer {name, organizationNumber, email, phoneNumber, isCustomer:false, isSupplier:true, language:"NO"}

Step 2: Calculate amounts (invoice total is INCL VAT unless "ekskl. MVA"/"excl. VAT"):
  totalInclVAT = stated amount
  netExclVAT = round(totalInclVAT / 1.25, 2)
  vatAmount = round(totalInclVAT - netExclVAT, 2)

Step 3: POST /ledger/voucher with 3 EXPLICIT postings:
  {
    "date": TODAY,
    "description": "Leverandørfaktura [INV_NUM] [SUPPLIER]",
    "postings": [
      {account:{id:EXPENSE_ACCT_ID}, amountGross:netExclVAT,   amountGrossCurrency:netExclVAT,   date:TODAY, row:1, vatType:{id:0}},
      {account:{id:ACCT_2710_ID},    amountGross:vatAmount,     amountGrossCurrency:vatAmount,     date:TODAY, row:2, vatType:{id:0}},
      {account:{id:ACCT_2400_ID},    amountGross:-totalInclVAT, amountGrossCurrency:-totalInclVAT, date:TODAY, row:3, vatType:{id:0}, supplier:{id:SUPPLIER_ID}}
    ]
  }
  → 3 postings balance to zero: netExclVAT + vatAmount - totalInclVAT = 0 ✓
  Choose expense account from type: 6800=tech/IKT, 6500=office, 6300=rent, etc.
""",

    "payment": """
── PAYMENT — register payment for existing invoice ──
1. GET /invoice?invoiceDateFrom=2020-01-01&invoiceDateTo=2027-01-01&fields=id,invoiceNumber,amount,amountOutstanding,customer (FREE)
2. Find the invoice matching the customer/amount in the prompt
3. PUT /invoice/ID/:payment query_params: paymentDate=TODAY&paymentTypeId=ID&paidAmount=AMOUNT_INCL_VAT
   - paidAmount MUST include VAT (amount × 1.25 if prompt says excl. MVA)
   - paymentTypeId from INVOICE PAYMENT TYPES in context
""",

    "reverse_payment": """
── REVERSE PAYMENT ──
1. GET /invoice?invoiceDateFrom=2020-01-01&invoiceDateTo=2027-01-01&fields=id,invoiceNumber,amount,customer (FREE)
2. PUT /invoice/ID/:payment query_params: paymentDate=TODAY&paymentTypeId=ID&paidAmount=-AMOUNT (negative!)
""",

    "delete_travel": """
── DELETE TRAVEL EXPENSE — state machine ──
1. GET /travelExpense?fields=id,status,title (FREE)
2. If status=APPROVED → PUT /travelExpense/ID/:undeliver first, then PUT /travelExpense/ID/:unapprove
   If status=DELIVERED → PUT /travelExpense/ID/:unapprove
   If status=OPEN → skip state changes
3. DELETE /travelExpense/ID
""",

    "employee": """
── EMPLOYEE — check for existing first! ──
If email in EXISTING EMPLOYEES:
  PUT /employee/ID {firstName, lastName, email, userType, department:{id:X}, dateOfBirth:FROM_CONTEXT, version:V}
Else:
  POST /employee {firstName, lastName, email, userType:"STANDARD", department:{id:DEFAULT_DEPT_ID}, dateOfBirth, employeeNumber}
  POST /division (if no divisions exist) {name:"Hovedkontor", startDate:TODAY, municipality:{id:262}, organizationNumber:"996757435"}
  POST /employee/employment {employee:{id:X}, startDate:DATE_FROM_PROMPT, division:{id:DIV_ID}, isMainEmployer:true, taxDeductionCode:"loennFraHovedarbeidsgiver"}
  POST /employee/employment/details {employment:{id:X}, date:TODAY, employmentType:"ORDINARY", employmentForm:"PERMANENT", remunerationType:"MONTHLY_WAGE", workingHoursScheme:"NOT_SHIFT", percentageOfFullTimeEquivalent:100.0}
""",

    "update_employee": """
── UPDATE EMPLOYEE (T3 — 1-2 write calls!) ──
Find employee by email in EXISTING EMPLOYEES — do NOT GET /employee again!
Required fields for PUT: {id, version, firstName, lastName, email, dateOfBirth:FROM_CONTEXT, userType:FROM_CONTEXT, department:{id:FROM_CONTEXT}}
→ ALWAYS keep dateOfBirth exactly from context, NEVER change it
→ ALWAYS keep version from context
→ If updating salary/position: also PUT /employee/employment/ID with updated fields

EFFICIENCY: 1-2 write calls max
  - Find employee in pre-fetched list → id, version, current fields
  - PUT /employee/ID with ALL required fields including unchanged ones
""",

    "contact_person": """
── CONTACT PERSON (T3) ──
1. Find existing customer by name or orgNr in EXISTING CUSTOMERS — do NOT GET /customer
   If not found: POST /customer {name, organizationNumber, email, phoneNumber, isCustomer:true}
2. POST /contact {firstName, lastName, email, customer:{id:CUSTOMER_ID}, phoneNumberMobile:PHONE_IF_GIVEN}
   Required: firstName, lastName, customer
   Optional: email, phoneNumberMobile, description, position

EFFICIENCY: 1-2 write calls (1 if customer exists, 2 if need to create)
""",

    "voucher": """
── STANDALONE VOUCHER (T3) ──
POST /ledger/voucher:
{
  "date": "YYYY-MM-DD",  ← use date from prompt, NOT today unless unspecified
  "description": "exact description from prompt",
  "postings": [
    {account:{id:DEBIT_ACCT_ID}, amountGross:AMT, amountGrossCurrency:AMT, date:"YYYY-MM-DD", row:1, vatType:{id:0}},
    {account:{id:CREDIT_ACCT_ID}, amountGross:-AMT, amountGrossCurrency:-AMT, date:"YYYY-MM-DD", row:2, vatType:{id:0}}
  ]
}
→ BOTH postings must have `date` field
→ Postings MUST balance to 0 (debit + credit = 0)
→ Use account IDs from LEDGER ACCOUNT IDs in context — do NOT GET /ledger/account
→ If VAT involved: use vatType:{id:1} on expense side (Tripletex auto-handles 2710)
""",

    "update_customer": """
── UPDATE CUSTOMER (T3 — 1 write call!) ──
Find customer by name/orgNr in EXISTING CUSTOMERS — do NOT GET /customer
GET /customer/ID?fields=* to get all current fields including version
PUT /customer/ID with ALL fields (keep everything unchanged, update only what prompt says):
  {id, version, name, organizationNumber, email, invoiceEmail, phoneNumber,
   isCustomer, isSupplier, language, postalAddress:{addressLine1, postalCode, city}}
→ Keep version from GET. NEVER omit version on PUT.
""",

    "bank_reconciliation": """
── BANK RECONCILIATION (T3) ──
Step 1: GET /ledger/account?number=1920&fields=id,version,bankAccountNumber (FREE)
  → This is the bank account. Note its ID.
Step 2: GET /bank/reconciliation/>last?accountId=ACCT_ID&fields=* (FREE)
  → Find last reconciliation or create new period
Step 3: POST /bank/reconciliation:
  {account:{id:ACCT_ID}, accountingPeriod:{id:PERIOD_ID}, type:"MANUAL",
   bankAccountClosingBalanceCurrency:CLOSING_BALANCE}
Step 4: POST /bank/reconciliation/match (match transactions to postings)
  Or: PUT /bank/reconciliation/match/:suggest (auto-suggest matches)
→ If task gives bank statement file: POST /bank/statement/import first
""",

    "year_end": """
── YEAR-END / ÅRSOPPGJØR (T3) ──
Common operations:
1. Depreciation (avskrivning): POST /ledger/voucher with:
   - Debit 6000-6099 (depreciation expense)
   - Credit 1200-1299 (accumulated depreciation on asset)
2. Closing entries: move profit/loss to equity accounts
3. Tax provision: POST /ledger/voucher with:
   - Debit 8700 (skattekostnad) — account id from context
   - Credit 2920 (betalbar skatt)
4. Check for bank reconciliation if mentioned

READ THE PROMPT CAREFULLY — year-end tasks vary widely. Use accounts from context.
For avskrivning: amount = (cost / useful_life_years) per year, or % of book value.
""",
}

def build_system_prompt(today: str, due: str, company_id, task_type: str) -> str:
    task_desc = TASK_CONFIG.get(task_type, TASK_CONFIG["unknown"])[1]
    task_recipe = TASK_RECIPES.get(task_type, "").replace("{TODAY}", today)

    return f"""You are a Tripletex accounting API agent. Complete accounting tasks by calling the API efficiently.

TODAY: {today}
DEFAULT_DUE_DATE: {due}
COMPANY_ID: {company_id or "call GET /token/session/>whoAmI to find it"}
DETECTED TASK: {task_type} — {task_desc}

══ EFFICIENCY RULES (critical for score!) ══
• GET calls are FREE — use them freely to read data
• POST/PUT/DELETE COST efficiency points — minimize these!
• 4xx errors PENALIZE efficiency — validate before writing
• Pre-fetched context below replaces most GET calls

══ CRITICAL RULES ══
1. NEVER call forbidden GETs (data already in context):
   /employee, /customer, /supplier, /product, /department, /division,
   /activity, /salary/type, /ledger/account, /invoice/paymentType,
   /travelExpense/costCategory, /travelExpense/paymentType, /travelExpense/rateCategory
2. EXACT values from prompt — never change names, numbers, dates, amounts
3. Dates → YYYY-MM-DD with leading zeros
4. Existing employee emails → PUT (never create duplicate), keep dateOfBirth FROM CONTEXT
5. On 422 → read validationMessages, fix field. On 403 → use fallback immediately
6. PUT /:action endpoints → parameters in query_params NOT body
7. Never set "id"/"version" in POST body
8. Prices are EXCLUDING VAT unless prompt says "inkl. MVA"/"TTC"/"con IVA"/"mit MwSt"
9. Do NOT invent data not in the prompt (phone, address, etc.)

══ LANGUAGES ══
Prompts in NO, EN, ES, PT, NN, DE, FR. Parse dates in any language → YYYY-MM-DD.
Months: januar/january/enero/janvier/januar/janeiro=01, februar/february/febrero/février/februar/fevereiro=02,
mars/march/marzo/mars/März/março=03, april=04, mai/may/mayo/mai/Mai/maio=05, juni/june/junio/juin/Juni/junho=06,
juli/july/julio/juillet/Juli/julho=07, august=08, september=09, oktober/october/octubre/octobre/Oktober/outubro=10,
november=11, desember/december/diciembre/décembre/Dezember/dezembro=12

══ TASK-SPECIFIC RECIPE ══{task_recipe if task_recipe else " Follow standard patterns below."}

══ STANDARD PATTERNS ══

── CUSTOMER ──
POST /customer {{name, organizationNumber, email, phoneNumber:"", isCustomer:true,
  postalAddress:{{addressLine1, postalCode, city}}}}  ← postalAddress only if address in prompt!

── SUPPLIER ──
POST /customer {{name, organizationNumber, email, phoneNumber, isCustomer:false, isSupplier:true}}
(NOT POST /supplier — use /customer with isSupplier:true)

── PRODUCT ──
POST /product {{name, number:"STRING", priceExcludingVatCurrency:X,
  priceIncludingVatCurrency:X*1.25, vatType:{{id:3}}}}
VAT IDs: 3=25%, 31=15%, 32=12%, 5=0%(inside MVA zone), 6=0%(outside)

── INVOICE ──
EFFICIENT PATH (fewer calls = higher efficiency bonus!):
1. POST /customer (skip if customer in EXISTING CUSTOMERS context!)
2. POST /product (skip if product exists!)
3. POST /order {{customer:{{id:X}}, orderDate:TODAY, deliveryDate:TODAY, isPrioritizeAmountsIncludingVat:false,
     orderLines:[{{product:{{id:Y}}, count:1, unitPriceExcludingVatCurrency:PRICE, vatType:{{id:3}}}}]}}
   ← orderLines CAN be nested in POST /order → saves 1 extra call!
4. PUT /order/ID/:invoice query_params: invoiceDate=TODAY&sendToCustomer=false
   ← saves 1 call vs POST /invoice!

── INVOICE + PAYMENT (combine into 1 step!) ──
4. PUT /order/ID/:invoice query_params: invoiceDate=TODAY&sendToCustomer=false&paymentTypeId=ID&paidAmount=TOTAL_INCL_VAT
   paidAmount = price × 1.25 (includes VAT). paymentTypeId from INVOICE PAYMENT TYPES in context.
   ← This creates AND pays the invoice in ONE call!

── PROJECT ──
1. PUT existing employee to EXTENDED (or POST new with userType:"EXTENDED")
2. POST /employee/entitlement {{employee:{{id:X}}, entitlementId:45, customer:{{id:COMPANY_ID}}}}
3. POST /employee/entitlement {{employee:{{id:X}}, entitlementId:10, customer:{{id:COMPANY_ID}}}}
4. POST /customer (if needed)
5. POST /project {{name, startDate:TODAY, projectManager:{{id:X}}, customer:{{id:Y}}}}
   Do NOT set "number" — let it auto-generate

── DEPARTMENT ──
POST /department {{name:"NAME", departmentNumber:NEXT_UNIQUE_INT}}
departmentNumber must be unique — use next numbers from context

── ACCOUNTING DIMENSIONS ──
POST /ledger/accountingDimensionName {{dimensionName:"NAME", description:"DESC"}} → get dimensionIndex
POST /ledger/accountingDimensionValue {{displayName:"NAME", number:1, dimensionIndex:X}}

── TRAVEL EXPENSE ──
POST /travelExpense {{employee:{{id:X}}, title:"...", travelDetails:{{departureDate:"YYYY-MM-DD",
  returnDate:"YYYY-MM-DD", departureFrom:"CITY", destination:"CITY", purpose:"..."}}}}
Per diem: POST /travelExpense/perDiemCompensation {{travelExpense:{{id:X}},
  rateCategory:{{id:FROM_CONTEXT}}, count:DAYS, rate:DAILY_RATE,
  overnightAccommodation:"HOTEL", location:"CITY"}}

── VOUCHER ──
POST /ledger/voucher {{date:"YYYY-MM-DD", description:"...", postings:[
  {{account:{{id:X}}, amountGross:AMT, amountGrossCurrency:AMT, date:"YYYY-MM-DD", row:1, vatType:{{id:0}}}},
  {{account:{{id:Y}}, amountGross:-AMT, amountGrossCurrency:-AMT, date:"YYYY-MM-DD", row:2, vatType:{{id:0}}}}
]}}
Customer accounts (1500): add customer:{{id:X}}. Supplier accounts (2400): add supplier:{{id:X}}.

── CONTACT PERSON ──
1. Find/create customer (GET from context or POST)
2. POST /contact {{firstName, lastName, email, customer:{{id:X}}, phoneNumberMobile:PHONE_IF_GIVEN}}

── PDF CONTRACT EMPLOYEE ──
Extract: firstName, lastName, email, dateOfBirth, employeeNumber, nationalIdentityNumber,
  department, startDate, salary, percentageOfFullTimeEquivalent
Use vision to read images. Then follow EMPLOYEE recipe above.

── ADMIN EMPLOYEE ──
POST /employee {{userType:"EXTENDED"}} then POST /employee/entitlement ×2 (entitlementId 45 + 10)

══ BETA ENDPOINTS (always 403 — never call!) ══
POST /incomingInvoice, PUT /project/{{id}}, POST /project/orderline,
POST /project/participant, DELETE /project

══ ERRORS ══
422 → read validationMessages → fix exactly that field
403 → skip, use fallback
409 → GET fresh version → retry PUT
Duplicate email/number → GET existing, use its ID"""


# ═══════════════════════════════════════════════════════════════
# SECTION 6: FILE PROCESSING
# ═══════════════════════════════════════════════════════════════

def extract_pdf_text(data: bytes) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            parts = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    parts.append(text)
                for table in page.extract_tables():
                    for row in table:
                        if row:
                            parts.append(" | ".join(str(c) for c in row if c))
            if parts:
                return "\n".join(parts)[:6000]
    except Exception as e:
        log.warning(f"pdfplumber failed: {e}")
    text = data.decode("latin-1", errors="ignore")
    readable = re.findall(r'[\w\s@.,;:!?/\\()-]{4,}', text)
    return " ".join(readable)[:4000]


def process_files(files: list) -> tuple[list[str], list[dict]]:
    text_parts, image_blocks = [], []
    for f in files:
        raw = base64.b64decode(f["content_base64"])
        fname = f["filename"]
        mime = f.get("mime_type", "")
        if mime.startswith("text") or fname.endswith((".csv", ".json", ".txt")):
            try:
                text_parts.append(f"File '{fname}':\n{raw.decode('utf-8')[:8000]}")
            except UnicodeDecodeError:
                text_parts.append(f"File '{fname}': [binary]")
        elif mime == "application/pdf" or fname.endswith(".pdf"):
            extracted = extract_pdf_text(raw)
            if extracted.strip():
                text_parts.append(f"PDF '{fname}':\n{extracted}")
        elif mime.startswith("image/"):
            image_blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": f["content_base64"]},
            })
            text_parts.append(f"[Image '{fname}' attached — read it visually]")
    return text_parts, image_blocks


# ═══════════════════════════════════════════════════════════════
# SECTION 7: MAIN ENDPOINT
# ═══════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "version": "v16", "model": CLAUDE_MODEL}


@app.post("/solve")
@app.post("/")
async def solve(request: Request):
    start_time = time.time()
    body = await request.json()
    prompt = body["prompt"]
    files = body.get("files", [])
    creds = body["tripletex_credentials"]

    log.info("=" * 60)
    log.info(f"PROMPT: {prompt[:200]}")
    log.info(f"FILES:  {[f['filename'] for f in files]}")

    # ── Detect task type ──
    task_type = detect_task_type(prompt)
    max_iter, task_desc = TASK_CONFIG.get(task_type, TASK_CONFIG["unknown"])
    log.info(f"TASK: {task_type} — {task_desc} — max_iter={max_iter}")

    # ── Setup session ──
    base_url = creds["base_url"].rstrip("/")
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.auth = ("0", creds["session_token"])
    session.headers.update({"Content-Type": "application/json", "Accept": "application/json"})

    # ── Get company ID ──
    company_id = None
    try:
        r = session.get(f"{base_url}/token/session/>whoAmI", timeout=10)
        if r.status_code == 200:
            company_id = r.json().get("value", {}).get("company", {}).get("id")
            log.info(f"Company: {company_id}")
    except Exception as e:
        log.warning(f"whoAmI: {e}")

    # ── Register bank account on 1920 (needed for invoices) ──
    try:
        r = session.get(f"{base_url}/ledger/account",
                        params={"number": "1920", "fields": "id,version,bankAccountNumber"}, timeout=10)
        if r.status_code == 200:
            accts = r.json().get("values", [])
            if accts and not accts[0].get("bankAccountNumber"):
                a = accts[0]
                session.put(f"{base_url}/ledger/account/{a['id']}", json={
                    "id": a["id"], "version": a["version"], "number": 1920,
                    "name": "Bankinnskudd", "bankAccountNumber": "12345678903", "isBankAccount": True,
                }, timeout=10)
                log.info("Bank account set on 1920")
    except Exception as e:
        log.warning(f"Bank setup: {e}")

    # ── Pre-fetch context (all parallel GETs — FREE for efficiency!) ──
    ctx = prefetch_context(session, base_url)
    ctx_text = format_context(ctx)

    # ── Process files ──
    text_parts, image_blocks = process_files(files)
    file_text = "\n---\n".join(text_parts)

    # ── Build messages ──
    today = date.today().isoformat()
    due = (date.today() + timedelta(days=30)).isoformat()
    system_prompt = build_system_prompt(today, due, company_id, task_type)

    user_text = f"Complete this accounting task:\n\n{prompt}"
    if file_text:
        user_text += f"\n\nAttached files:\n{file_text}"
    user_text += f"\n\n═══ PRE-FETCHED SANDBOX DATA ═══\n{ctx_text}"

    user_content = []
    user_content.extend(image_blocks)
    user_content.append({"type": "text", "text": user_text})

    messages = [{"role": "user", "content": user_content}]
    trace = []

    # ── Agent loop ──
    iteration = 0
    while iteration < max_iter:
        iteration += 1
        elapsed = time.time() - start_time
        if elapsed > 240:  # 4 min safety margin
            log.warning(f"Time limit at {elapsed:.0f}s — stopping")
            break

        try:
            response = claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4096,
                system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
                tools=TOOLS,
                messages=messages,
                temperature=0.0,
            )
        except Exception as e:
            log.error(f"Claude error: {e}")
            break

        stop_reason = response.stop_reason
        log.info(f"Iter {iteration}: stop={stop_reason}, blocks={len(response.content)}")

        if stop_reason == "end_turn":
            log.info("Claude finished.")
            break

        if stop_reason != "tool_use":
            log.warning(f"Unexpected stop: {stop_reason}")
            break

        # Execute tool calls
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = execute_tool(block.name, block.input, session, base_url, trace, ctx)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        if not tool_results:
            break

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    elapsed = time.time() - start_time
    write_calls = sum(1 for t in trace if t["tool"] in ("tripletex_post", "tripletex_put", "tripletex_delete"))
    errors = sum(1 for t in trace if (t.get("status") or 0) >= 400)

    log.info(f"Done in {elapsed:.1f}s — {iteration} iters, {write_calls} writes, {errors} errors")
    log.info(f"Trace: {json.dumps(trace, ensure_ascii=False)}")

    return JSONResponse({"status": "completed"})
