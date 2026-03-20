"""
Tripletex AI Accounting Agent — NM i AI 2026
v12: Pre-fetched context + stronger email handling + optimized prompts.
"""

import base64
import io
import json
import logging
import os
import re
import time
import traceback
from datetime import date, timedelta, datetime, timezone
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

LOG_PATH = Path(os.environ.get("SUBMISSION_LOG", "/tmp/tripletex_submissions.jsonl"))


# ═══════════════════════════════════════════════
# TOOLS — Claude calls these to interact with Tripletex API
# ═══════════════════════════════════════════════

TOOLS = [
    {
        "name": "tripletex_get",
        "description": "GET request to Tripletex API. Returns JSON response.",
        "input_schema": {
            "type": "object",
            "properties": {
                "endpoint": {"type": "string", "description": "API path, e.g. /employee or /department"},
                "query_params": {"type": "string", "description": "Query string, e.g. 'fields=id,name&count=10'. Empty string if none.", "default": ""},
            },
            "required": ["endpoint"],
        },
    },
    {
        "name": "tripletex_post",
        "description": "POST request to create a resource. Returns JSON response with created object.",
        "input_schema": {
            "type": "object",
            "properties": {
                "endpoint": {"type": "string", "description": "API path, e.g. /employee"},
                "body": {"type": "object", "description": "JSON body to send"},
                "query_params": {"type": "string", "description": "Query string. Empty string if none.", "default": ""},
            },
            "required": ["endpoint", "body"],
        },
    },
    {
        "name": "tripletex_put",
        "description": "PUT request to update a resource or trigger an action. For actions like /:payment, use query_params for parameters (NOT body).",
        "input_schema": {
            "type": "object",
            "properties": {
                "endpoint": {"type": "string", "description": "API path, e.g. /employee/123 or /invoice/456/:payment"},
                "body": {"type": "object", "description": "JSON body (for updates). Empty object {} for actions.", "default": {}},
                "query_params": {"type": "string", "description": "Query string. For actions like /:payment, put params here.", "default": ""},
            },
            "required": ["endpoint"],
        },
    },
    {
        "name": "tripletex_delete",
        "description": "DELETE request to remove a resource.",
        "input_schema": {
            "type": "object",
            "properties": {
                "endpoint": {"type": "string", "description": "API path with ID, e.g. /travelExpense/789"},
            },
            "required": ["endpoint"],
        },
    },
]


# ═══════════════════════════════════════════════
# API EXECUTION
# ═══════════════════════════════════════════════

def execute_tool(name: str, input_data: dict, session: requests.Session, base_url: str, trace: list, ctx: dict = None) -> str:
    """Execute a Tripletex API tool call. Returns truncated JSON string."""
    endpoint = input_data.get("endpoint", "")
    qp = input_data.get("query_params", "") or ""
    params = dict(p.split("=", 1) for p in qp.split("&") if "=" in p) if qp else None
    body = input_data.get("body")

    # Intercept: redirect POST /supplier → POST /customer with isSupplier:true
    if name == "tripletex_post" and endpoint.strip("/") == "supplier":
        endpoint = "/customer"
        if body:
            body["isSupplier"] = True
            if "isCustomer" not in body:
                body["isCustomer"] = False
            if body.get("organizationNumber"):
                body["organizationNumber"] = body["organizationNumber"].replace(" ", "")
            # Ensure phoneNumber is set (even empty) — scoring may check it
            if "phoneNumber" not in body:
                body["phoneNumber"] = ""
        log.info(f"Redirected POST /supplier → POST /customer with isSupplier:true")

    # Intercept: POST /customer — also ensure phoneNumber for regular customers
    if name == "tripletex_post" and endpoint.strip("/") == "customer" and body:
        if "phoneNumber" not in body:
            body["phoneNumber"] = ""

    # Intercept: POST /incomingInvoice → skip (always 403), force voucher fallback
    if name == "tripletex_post" and "incomingInvoice" in endpoint:
        log.info("Intercepted POST /incomingInvoice → returning 403 to force voucher fallback")
        trace.append({"tool": name, "endpoint": endpoint, "status": 403, "error": "Module unavailable"})
        return json.dumps({"status": 403, "message": "incomingInvoice module not available. Use voucher fallback with accounts 5000/2710/2400."})

    # Intercept: PUT /employee — preserve pre-fetched dateOfBirth if Claude uses "1985-01-01"
    if name == "tripletex_put" and "/employee/" in endpoint and body and ctx:
        if body.get("dateOfBirth") == "1985-01-01":
            emp_id_str = endpoint.rstrip("/").split("/")[-1]
            try:
                emp_id = int(emp_id_str)
                for e in ctx.get("employees", []):
                    if e.get("id") == emp_id and e.get("dateOfBirth") and e["dateOfBirth"] != "1985-01-01":
                        body["dateOfBirth"] = e["dateOfBirth"]
                        log.info(f"Fixed dateOfBirth: 1985-01-01 → {e['dateOfBirth']} for employee {emp_id}")
                        break
            except (ValueError, TypeError):
                pass

    # Intercept: POST /employee — ensure realistic dateOfBirth
    if name == "tripletex_post" and endpoint.strip("/") == "employee" and body:
        if body.get("dateOfBirth") in (None, "1985-01-01", ""):
            body["dateOfBirth"] = "1990-05-15"

    # Intercept: PUT /project — don't overwrite auto-generated number with "2"
    if name == "tripletex_put" and "/project/" in endpoint and body:
        if body.get("number") in ("2", "1", 2, 1):
            body.pop("number", None)
            log.info(f"Removed hardcoded project number from PUT")
            log.info(f"Fixed POST /employee dateOfBirth to 1990-05-15")

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
        log.error(f"{name} {endpoint} → connection error: {e}")
        trace.append({"tool": name, "endpoint": endpoint, "status": 0, "error": str(e)[:200]})
        return json.dumps({"error": str(e)[:200]})

    status_code = resp.status_code
    trace.append({
        "tool": name,
        "endpoint": endpoint,
        "status": status_code,
        "error": resp.text[:300] if status_code >= 400 else None,
    })

    if status_code >= 400:
        log.error(f"{name} {endpoint} → {status_code}: {resp.text[:300]}")
    else:
        log.info(f"{name} {endpoint} → {status_code}")

    try:
        result = resp.json()
    except Exception:
        result = {"raw": resp.text[:500]}

    # Truncate large responses to save tokens
    text = json.dumps(result, ensure_ascii=False)
    if len(text) > 3000:
        if isinstance(result, dict) and "values" in result:
            result["values"] = result["values"][:10]
            result["_truncated"] = True
            result["_note"] = "Use query params to filter (e.g. number=2002) instead of browsing"
            text = json.dumps(result, ensure_ascii=False)
        if len(text) > 3000:
            text = text[:3000]
    return text


# ═══════════════════════════════════════════════
# PRE-FETCH CONTEXT
# ═══════════════════════════════════════════════

def prefetch_context(session: requests.Session, base_url: str) -> dict:
    """Pre-fetch commonly needed data in parallel to reduce latency."""
    import concurrent.futures
    ctx = {}

    def fetch(name, endpoint, params):
        try:
            resp = session.get(f"{base_url}/{endpoint}", params=params, timeout=10)
            if resp.status_code == 200:
                return name, resp.json().get("values", [])
        except Exception as e:
            log.warning(f"Prefetch {name}: {e}")
        return name, []

    # Run all fetches in parallel
    fetches = [
        ("employees", "employee", {"fields": "id,firstName,lastName,email,dateOfBirth,userType,version,department", "count": "100"}),
        ("departments", "department", {"fields": "id,name,departmentNumber", "count": "10"}),
        ("divisions", "division", {"fields": "id,name", "count": "5"}),
        ("activities", "activity", {"fields": "id,name", "count": "20"}),
        ("payment_types", "invoice/paymentType", {"fields": "id,description", "count": "5"}),
        ("products", "product", {"fields": "id,name,number,priceExcludingVatCurrency", "count": "100"}),
        ("customers", "customer", {"fields": "id,name,organizationNumber", "count": "100"}),
        ("suppliers", "supplier", {"fields": "id,name,organizationNumber", "count": "100"}),
        ("salary_2000", "salary/type", {"number": "2000", "fields": "id,number,name"}),
        ("salary_2001", "salary/type", {"number": "2001", "fields": "id,number,name"}),
        ("salary_2002", "salary/type", {"number": "2002", "fields": "id,number,name"}),
        ("salary_2005", "salary/type", {"number": "2005", "fields": "id,number,name"}),
        ("rate_categories", "travelExpense/rateCategory", {"fields": "id,name,type,fromDate,toDate", "count": "500"}),
        ("acct_5000", "ledger/account", {"number": "5000", "fields": "id,number,name"}),
        ("acct_2780", "ledger/account", {"number": "2780", "fields": "id,number,name"}),
        ("acct_1920", "ledger/account", {"number": "1920", "fields": "id,number,name"}),
        ("acct_2710", "ledger/account", {"number": "2710", "fields": "id,number,name"}),
        ("acct_2400", "ledger/account", {"number": "2400", "fields": "id,number,name"}),
        ("cost_categories", "travelExpense/costCategory", {"fields": "id,description", "count": "50"}),
        ("travel_payment_types", "travelExpense/paymentType", {"fields": "id,description", "count": "5"}),
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch, name, ep, params): name for name, ep, params in fetches}
        for future in concurrent.futures.as_completed(futures):
            try:
                name, values = future.result()
            except Exception as e:
                log.warning(f"Prefetch future failed: {e}")
                continue
            if name == "employees":
                ctx["employees"] = values
                log.info(f"Pre-fetched {len(values)} employees")
            elif name == "departments":
                ctx["departments"] = values
                if values:
                    ctx["default_department_id"] = values[0]["id"]
                log.info(f"Pre-fetched {len(values)} departments")
            elif name == "divisions":
                ctx["divisions"] = values
                if values:
                    ctx["default_division_id"] = values[0]["id"]
            elif name == "activities":
                ctx["activities"] = values
            elif name == "payment_types":
                ctx["payment_types"] = values
            elif name == "products":
                ctx["products"] = values
                log.info(f"Pre-fetched {len(values)} products")
            elif name == "customers":
                ctx["customers"] = values
                log.info(f"Pre-fetched {len(values)} customers")
            elif name == "suppliers":
                ctx["suppliers"] = values
                log.info(f"Pre-fetched {len(values)} suppliers")
            elif name == "rate_categories":
                # Find 2026-valid per diem categories (overnight + day trip)
                try:
                    for rc in values:
                        n = str(rc.get("name") or "")
                        fr = str(rc.get("fromDate") or "")
                        to = str(rc.get("toDate") or "")
                        if fr >= "2026" and to >= "2026" and "innland" in n:
                            if "Overnatting" in n and "per_diem_overnight_id" not in ctx:
                                ctx["per_diem_overnight_id"] = rc["id"]
                                ctx["per_diem_rate_category_id"] = rc["id"]  # backward compat
                                log.info(f"Per diem overnight: id={rc['id']} name={n}")
                            elif "Dag" in n and "per_diem_day_id" not in ctx:
                                ctx["per_diem_day_id"] = rc["id"]
                                log.info(f"Per diem day trip: id={rc['id']} name={n}")
                except Exception as e:
                    log.warning(f"Rate category search failed: {e}")
            elif name.startswith("salary_"):
                num = name.split("_")[1]
                if values:
                    if "salary_types" not in ctx:
                        ctx["salary_types"] = {}
                    ctx["salary_types"][num] = {"id": values[0]["id"], "name": values[0].get("name", ""), "number": num}
            elif name == "cost_categories":
                ctx["cost_categories"] = values
            elif name == "travel_payment_types":
                ctx["travel_payment_types"] = values
            elif name.startswith("acct_"):
                acct_num = name.split("_")[1]
                if values:
                    if "ledger_accounts" not in ctx:
                        ctx["ledger_accounts"] = {}
                    ctx["ledger_accounts"][acct_num] = values[0]["id"]

    if ctx.get("salary_types"):
        log.info(f"Pre-fetched salary types: {list(ctx['salary_types'].keys())}")
    if ctx.get("ledger_accounts"):
        log.info(f"Pre-fetched ledger accounts: {ctx['ledger_accounts']}")

    # Post-fetch: check employments for each employee (fast, parallel)
    if ctx.get("employees"):
        def fetch_employment(emp):
            try:
                resp = session.get(f"{base_url}/employee/employment",
                    params={"employeeId": emp["id"], "fields": "id,startDate,division", "count": "5"}, timeout=10)
                if resp.status_code == 200:
                    empls = resp.json().get("values", [])
                    return emp["id"], empls
            except Exception:
                pass
            return emp["id"], []

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            empl_results = list(ex.map(fetch_employment, ctx["employees"]))
        for emp_id, empls in empl_results:
            for e in ctx["employees"]:
                if e["id"] == emp_id:
                    e["_employments"] = empls
                    break
        log.info(f"Pre-fetched employments for {len(ctx['employees'])} employees")

    return ctx


def format_prefetched_context(ctx: dict) -> str:
    """Format pre-fetched data for the user message."""
    parts = []

    if ctx.get("employees"):
        emp_lines = []
        for e in ctx["employees"]:
            dept_id = ""
            if isinstance(e.get("department"), dict):
                dept_id = e["department"].get("id", "")
            empls = e.get("_employments", [])
            empl_info = ""
            if empls:
                empl = empls[0]
                div_id = ""
                if isinstance(empl.get("division"), dict):
                    div_id = empl["division"].get("id", "")
                empl_info = f" hasEmployment=YES employmentId={empl['id']} divisionId={div_id}"
            else:
                empl_info = " hasEmployment=NO"
            emp_lines.append(f"  id={e['id']} email={e.get('email','')} name={e.get('firstName','')} {e.get('lastName','')} dateOfBirth={e.get('dateOfBirth','')} userType={e.get('userType','')} version={e.get('version','')} dept={dept_id}{empl_info}")
        parts.append("EXISTING EMPLOYEES (use PUT to update if email matches — KEEP existing dateOfBirth!):\n" + "\n".join(emp_lines))

    if ctx.get("default_department_id"):
        parts.append(f"DEFAULT DEPARTMENT ID: {ctx['default_department_id']}")

    if ctx.get("default_division_id"):
        parts.append(f"DEFAULT DIVISION ID: {ctx['default_division_id']}")
    else:
        parts.append("NO DIVISIONS EXIST — create one if needed for employment")

    if ctx.get("activities"):
        act_lines = [f"  id={a['id']} name={a.get('name','')}" for a in ctx["activities"]]
        parts.append("AVAILABLE ACTIVITIES:\n" + "\n".join(act_lines))

    if ctx.get("departments"):
        dept_nums = [str(d.get("departmentNumber", "")) for d in ctx["departments"] if d.get("departmentNumber")]
        if dept_nums:
            parts.append(f"EXISTING DEPARTMENT NUMBERS: {', '.join(dept_nums)} (use a different number for new departments)")

    if ctx.get("salary_types"):
        st_lines = [f"  number={v['number']} id={v['id']} name={v['name']}" for v in ctx["salary_types"].values()]
        parts.append("SALARY TYPES (use these exact IDs — do NOT GET /salary/type yourself):\n" + "\n".join(st_lines))

    if ctx.get("payment_types"):
        pt_lines = [f"  id={p['id']} description={p.get('description','')}" for p in ctx["payment_types"]]
        parts.append("INVOICE PAYMENT TYPES:\n" + "\n".join(pt_lines))

    if ctx.get("products"):
        prod_lines = [f"  id={p['id']} number={p.get('number','')} name={p.get('name','')} priceExVat={p.get('priceExcludingVatCurrency','')}" for p in ctx["products"][:20]]
        parts.append("EXISTING PRODUCTS (use existing if number/name matches — do NOT re-create):\n" + "\n".join(prod_lines))

    if ctx.get("customers"):
        cust_lines = [f"  id={c['id']} name={c.get('name','')} orgNr={c.get('organizationNumber','')}" for c in ctx["customers"][:20]]
        parts.append("EXISTING CUSTOMERS (use existing if name/orgNr matches — do NOT re-create):\n" + "\n".join(cust_lines))

    if ctx.get("suppliers"):
        sup_lines = [f"  id={s['id']} name={s.get('name','')} orgNr={s.get('organizationNumber','')}" for s in ctx["suppliers"][:20]]
        parts.append("EXISTING SUPPLIERS (use existing if name/orgNr matches — do NOT re-create):\n" + "\n".join(sup_lines))

    if ctx.get("per_diem_overnight_id") or ctx.get("per_diem_day_id"):
        pd_lines = []
        if ctx.get("per_diem_overnight_id"):
            pd_lines.append(f"  Overnight (overnatting innland): {ctx['per_diem_overnight_id']} — for trips WITH overnight stay")
        if ctx.get("per_diem_day_id"):
            pd_lines.append(f"  Day trip (dagsreise innland): {ctx['per_diem_day_id']} — for day trips WITHOUT overnight")
        parts.append("PER DIEM RATE CATEGORY IDs (use in POST /travelExpense/perDiemCompensation):\n" + "\n".join(pd_lines))
    elif ctx.get("per_diem_rate_category_id"):
        parts.append(f"PER DIEM RATE CATEGORY ID (for 2026): {ctx['per_diem_rate_category_id']} — use this in POST /travelExpense/perDiemCompensation")

    if ctx.get("ledger_accounts"):
        accts = ctx["ledger_accounts"]
        lines = [f"  {num}: id={aid}" for num, aid in sorted(accts.items())]
        parts.append("LEDGER ACCOUNT IDS (pre-fetched — use directly, do NOT GET /ledger/account yourself):\n" + "\n".join(lines))

    if ctx.get("cost_categories"):
        cc_lines = [f"  id={c['id']} desc={c.get('description','')}" for c in ctx["cost_categories"][:15]]
        parts.append("TRAVEL EXPENSE COST CATEGORIES (pre-fetched — do NOT GET /travelExpense/costCategory yourself):\n" + "\n".join(cc_lines))

    if ctx.get("travel_payment_types"):
        tp_lines = [f"  id={t['id']} desc={t.get('description','')}" for t in ctx["travel_payment_types"][:5]]
        parts.append("TRAVEL EXPENSE PAYMENT TYPES (pre-fetched — do NOT GET /travelExpense/paymentType yourself):\n" + "\n".join(tp_lines))

    return "\n\n".join(parts)


# ═══════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════

def build_system_prompt(company_id: int | None) -> str:
    today = date.today().isoformat()
    due = (date.today() + timedelta(days=30)).isoformat()

    return f"""You are a Tripletex accounting API agent. Complete the given accounting task by calling the API.

TODAY: {today}
DEFAULT_DUE_DATE: {due}
COMPANY_ID: {company_id or "unknown — call GET /token/session/>whoAmI to find it"}

CRITICAL RULES:
1. Pre-populated employees exist. If email ALREADY EXISTS in context → PUT /employee/ID with name from prompt + dateOfBirth FROM CONTEXT (keep existing, never use "1985-01-01"). Never create with modified email.
2. Use EXACT values from prompt — never change names, numbers, dates, amounts, emails.
3. Dates MUST be YYYY-MM-DD with leading zeros.
4. EFFICIENCY IS SCORED! Minimize API calls — NEVER make GETs for data that is already in the PRE-FETCHED CONTEXT below. Use IDs directly from context for: employees, departments, divisions, activities, payment types, products, customers, suppliers, salary types, ledger accounts, cost categories, travel payment types, per diem rate categories.
5. On 422: read validationMessages, fix the field. On 403: skip immediately, use fallback.
6. For PUT actions (/:payment, /:createCreditNote): parameters in query_params, NOT body.
7. Never set "id"/"version" in POST. Use department/division/activity IDs from context directly.
8. Per diem / daily allowance in travel expenses → use /travelExpense/perDiemCompensation (NOT /travelExpense/cost!)
9. Salary: if /salary/transaction → 201, STOP. Do NOT also create manual voucher (causes duplicate postings).
10. Prices in prompts are EXCLUDING VAT unless explicitly stated as "inkl. MVA" / "TTC" / "con IVA incluido".
11. Do NOT invent data not in the prompt (e.g. departureFrom, phone numbers). Only use values explicitly stated.

LANGUAGES: Prompts come in NO, EN, ES, PT, NN, DE, FR. Parse dates in any language to YYYY-MM-DD.

═══ EMPLOYEE EMAIL HANDLING (MOST IMPORTANT) ═══
Pre-populated employees exist in every sandbox. When creating an employee:
1. CHECK the EXISTING EMPLOYEES list in the context below
2. If the email from the task matches an existing employee → PUT /employee/ID with ALL fields from the prompt + version from context
3. If no match → POST /employee as normal
4. NEVER modify the email address. NEVER add "2" or any suffix.
5. When doing PUT, include: firstName, lastName, email (unchanged!), userType, department, dateOfBirth (use "1985-01-01" if not in prompt), phoneNumberMobile (if in prompt), version

═══ ENDPOINT REFERENCE ═══

GET /token/session/>whoAmI → company ID

── EMPLOYEE ──
POST /employee {{firstName, lastName, email, userType:"STANDARD", department:{{id:X}}, phoneNumberMobile, dateOfBirth}}
PUT /employee/ID {{firstName, lastName, email, userType, department:{{id:X}}, dateOfBirth, version:V}}
- userType: "STANDARD" (normal), "EXTENDED" (for project managers/admins), "NO_ACCESS"
- NEVER use "ADMINISTRATOR" as userType
- dateOfBirth required if you need to create employment
- Employee address uses "address" field: {{addressLine1, postalCode, city}} (NOT postalAddress like customer!)

── EMPLOYMENT (for start date / salary tasks) ──
Step 1: Create/update employee (with dateOfBirth!)
Step 2: POST /employee/employment {{employee:{{id:X}}, startDate:"YYYY-MM-DD", division:{{id:DIVISION_ID}}, isMainEmployer:true, taxDeductionCode:"loennFraHovedarbeidsgiver"}}
  If no division exists: POST /division {{name:"Hovedkontor", startDate:"{today}", municipality:{{id:262}}, organizationNumber:"996757435"}}
Step 3 (if salary/details needed): POST /employee/employment/details {{employment:{{id:X}}, date:"{today}", employmentType:"ORDINARY", employmentForm:"PERMANENT", remunerationType:"MONTHLY_WAGE", workingHoursScheme:"NOT_SHIFT", percentageOfFullTimeEquivalent:100}}

── ADMIN EMPLOYEE ──
POST /employee {{userType:"EXTENDED"}} → POST /employee/entitlement {{employee:{{id:X}}, entitlementId:1, customer:{{id:{company_id or 'COMPANY_ID'}}}}}

── CUSTOMER ──
POST /customer {{name, organizationNumber, email, phoneNumber, isCustomer:true, postalAddress:{{addressLine1, postalCode, city}}}}
- postalAddress.addressLine1 (NOT address1!)
- phoneNumber (NOT phoneNumberMobile for customers)

── SUPPLIER ──
POST /customer {{name, organizationNumber, email, phoneNumber, isCustomer:false, isSupplier:true, postalAddress:{{addressLine1, postalCode, city}}}}
- IMPORTANT: Create suppliers via POST /customer with isSupplier:true (NOT POST /supplier!)
- This ensures the supplier is visible on both /customer and /supplier endpoints

── CONTACT PERSON ──
POST /customer (create customer first) → POST /contact {{firstName, lastName, email, phoneNumberMobile, customer:{{id:X}}}}

── PRODUCT ──
POST /product {{name, number:"STRING!", priceExcludingVatCurrency:X, priceIncludingVatCurrency:Y, vatType:{{id:Z}}}}
- number is STRING (e.g. "7898"), not integer
- priceExcludingVatCurrency (NOT priceExcludingVat!)
- ALWAYS set vatType explicitly! Default is id 6 (0%) which is wrong for most products
- VAT IDs: 3=25%, 31=15%, 32=12%, 5=0% (innenfor MVA), 6=0% (utenfor)
- Calculate: priceIncludingVatCurrency = priceExcludingVatCurrency × (1 + vatPct/100)

── DEPARTMENT ──
POST /department {{name, departmentNumber:INT}}
- departmentNumber must be unique integer — use numbers from EXISTING DEPARTMENT NUMBERS in context +1, +2, etc. Do NOT GET /department yourself!
- For 3 departments: just POST 3 times with unique numbers. No GET needed.

── INVOICE (multi-step) ──
Bank account is pre-registered. Steps:
1. POST /customer
2. POST /product (ALWAYS set vatType:{{id:3}} for 25% MVA unless specified otherwise)
   - If product number already exists: GET /product?number=XXXX&fields=id,name,number,priceExcludingVatCurrency to find the existing one
3. POST /order {{customer:{{id:X}}, deliveryDate:"{today}", orderDate:"{today}", isPrioritizeAmountsIncludingVat:false}}
4. POST /order/orderline {{order:{{id:X}}, product:{{id:Y}}, count:N, unitPriceExcludingVatCurrency:PRICE, vatType:{{id:3}}}}
   - ALWAYS set vatType on orderline too! It does NOT inherit from product
   - unitPriceExcludingVatCurrency is REQUIRED on orderline
5. POST /invoice {{invoiceDate:"{today}", invoiceDueDate:"{due}", orders:[{{id:X}}]}} query_params: sendToCustomer=false

PRICES: Always EXCLUDING VAT unless "inkl. MVA"/"TTC"/"con IVA incluido"/"mit MwSt" is stated.
paidAmount on /:payment = TOTAL INCLUDING VAT (orderline sum × 1.25)
GET /invoice REQUIRES: invoiceDateFrom & invoiceDateTo params.

── INVOICE + PAYMENT ──
After creating invoice:
6. Use payment type ID from the INVOICE PAYMENT TYPES in context (do NOT GET /invoice/paymentType again)
7. PUT /invoice/ID/:payment query_params: paymentDate={today}&paymentTypeId=ID&paidAmount=TOTAL_INCL_VAT
- paidAmount MUST include VAT!

── REVERSE PAYMENT ──
1. GET /invoice?invoiceDateFrom=2020-01-01&invoiceDateTo=2027-01-01&fields=id,invoiceNumber,amount,amountOutstanding,customer
2. PUT /invoice/ID/:payment query_params: paymentDate={today}&paymentTypeId=ID&paidAmount=-AMOUNT (negative to reverse)

── CREDIT NOTE ──
After creating invoice:
6. PUT /invoice/ID/:createCreditNote query_params: date={today}&sendToCustomer=false

── PROJECT ──
1. Create employee with userType:"EXTENDED" (or PUT existing to EXTENDED)
2. POST /employee/entitlement {{employee:{{id:X}}, entitlementId:45, customer:{{id:{company_id or 'COMPANY_ID'}}}}}
3. POST /employee/entitlement {{employee:{{id:X}}, entitlementId:10, customer:{{id:{company_id or 'COMPANY_ID'}}}}}
4. POST /customer (if needed)
5. POST /project {{name, startDate:"{today}", projectManager:{{id:X}}, customer:{{id:Y}}}}
- Do NOT set "number" — let auto-generate to avoid conflicts

── PROJECT + FIXED PRICE ──
After creating project:
6. GET /project/ID?fields=id,version,number
7. PUT /project/ID {{name, number:KEEP_ORIGINAL_NUMBER, startDate, isFixedPrice:true, fixedprice:AMOUNT, version:V, projectManager:{{id:X}}, customer:{{id:Y}}}}
   IMPORTANT: Use the ORIGINAL number from step 6 GET response — do NOT change it to "2" or any other value!

── TIMESHEET + PROJECT INVOICE ──
Register hours and invoice for a project:
1. Create employee (EXTENDED), entitlements, customer, project (as above — do all in minimum calls)
2. Use activity ID from the AVAILABLE ACTIVITIES in context (do NOT call /activity again)
3. POST /timesheet/entry {{project:{{id:X}}, activity:{{id:Y}}, employee:{{id:Z}}, date:"{today}", hours:N}}
4. Create invoice: total = hours × hourlyRate
   a. POST /product {{name:"Hours", number:"HRS1", priceExcludingVatCurrency:TOTAL, priceIncludingVatCurrency:TOTAL*1.25, vatType:{{id:3}}}}
   b. POST /order {{customer:{{id:Y}}, project:{{id:Z}}, deliveryDate:"{today}", orderDate:"{today}", isPrioritizeAmountsIncludingVat:false}}
   c. POST /order/orderline {{order:{{id:X}}, product:{{id:Y}}, count:1, unitPriceExcludingVatCurrency:TOTAL, vatType:{{id:3}}}}
   d. POST /invoice {{invoiceDate:"{today}", invoiceDueDate:"{due}", orders:[{{id:X}}]}} query_params: sendToCustomer=false
   IMPORTANT: ALWAYS set vatType:{{id:3}} on BOTH product AND orderline!
NOTE: /project/projectActivity does NOT have a "name" field. Use /activity for activities.
NOTE: /projectInvoice does NOT exist. Always use order → orderline → invoice flow.

── PROJECT + PARTIAL INVOICE ──
After fixed price project:
8. POST /product {{name:"Delbetaling", number:"DEL1", priceExcludingVatCurrency:PARTIAL_AMOUNT, priceIncludingVatCurrency:PARTIAL_AMOUNT*1.25, vatType:{{id:3}}}}
   - PARTIAL_AMOUNT = fixedprice × percentage (e.g. 250000 × 0.25 = 62500)
9. POST /order {{customer:{{id:Y}}, project:{{id:Z}}, deliveryDate:"{today}", orderDate:"{today}", isPrioritizeAmountsIncludingVat:false}}
10. POST /order/orderline {{order:{{id:X}}, product:{{id:Y}}, count:1, unitPriceExcludingVatCurrency:PARTIAL_AMOUNT, vatType:{{id:3}}}}
    CRITICAL: vatType:{{id:3}} is REQUIRED on orderline — without it, invoice has 0% VAT!
11. POST /invoice {{invoiceDate:"{today}", invoiceDueDate:"{due}", orders:[{{id:X}}]}} query_params: sendToCustomer=false

── TRAVEL EXPENSE ──
1. Create/update employee (if needed)
2. POST /travelExpense {{employee:{{id:X}}, title:"...", travelDetails:{{departureDate:"YYYY-MM-DD", returnDate:"YYYY-MM-DD", departureFrom:"CITY_FROM_PROMPT", destination:"CITY_FROM_PROMPT", purpose:"PURPOSE_FROM_PROMPT"}}}}
- Dates go INSIDE travelDetails, NOT flat!
- travelDetails with departureFrom/destination/purpose = type 0 (travel)
- ONLY set departureFrom/destination if mentioned in prompt — do NOT invent cities
3. For per diem / daily allowance ("dagpenger", "diett", "ajudas de custo", "per diem", "daily rate"):
   Use PER DIEM RATE CATEGORY ID from context (pre-fetched, do NOT search yourself).
   POST /travelExpense/perDiemCompensation {{travelExpense:{{id:X}}, rateCategory:{{id:RATE_CATEGORY_ID_FROM_CONTEXT}}, count:NUM_DAYS, rate:DAILY_RATE, overnightAccommodation:"HOTEL", location:"DESTINATION_CITY"}}
   - DAILY_RATE = the exact daily rate from prompt (e.g. 800)
   - count = number of days from prompt
   - location = destination city from prompt
   - Do NOT use /travelExpense/cost for per diem! Use /travelExpense/perDiemCompensation!
4. For regular expenses (flight, taxi, hotel costs etc.):
   Use TRAVEL EXPENSE COST CATEGORIES and PAYMENT TYPES from context (pre-fetched, do NOT GET them yourself!).
   POST /travelExpense/cost {{travelExpense:{{id:X}}, costCategory:{{id:Y}}, paymentType:{{id:Z}}, amountCurrencyIncVat:AMOUNT, currency:{{id:1}}, date:"YYYY-MM-DD"}}
   - Use "costCategory" NOT "category"
   - Use "comments" NOT "description" for cost line text
   - Match costCategory by description: "Fly", "Hotell", "Taxi", "Mat", "Parkering", etc.

── DELETE TRAVEL EXPENSE ──
1. GET /travelExpense?fields=id,title&count=100
2. DELETE /travelExpense/ID (for each matching)

── SALARY / PAYROLL ──
Step 1: Find employee by email in EXISTING EMPLOYEES context.
   - PUT /employee/ID with firstName, lastName from prompt + dateOfBirth FROM CONTEXT (NEVER use "1985-01-01"!) + version from context
Step 2: Check if employee already has employment (hasEmployment=YES in context).
   - If YES → SKIP step 3, go straight to step 4
   - If NO → create division + employment + details
Step 3 (only if hasEmployment=NO): POST /division + POST /employee/employment + POST /employee/employment/details
Step 4: Try salary/transaction first:
   POST /salary/transaction query_params: generateTaxDeduction=true
   Body: {{date:"{today}", month:{date.today().month}, year:{date.today().year}, payslips:[{{
     employee:{{id:EMP_ID}}, date:"{today}", year:{date.today().year}, month:{date.today().month},
     specifications:[
       {{salaryType:{{id:SALARY_TYPE_2000_ID}}, rate:BASE_SALARY, count:1, amount:BASE_SALARY}},
       {{salaryType:{{id:SALARY_TYPE_2002_ID}}, rate:BONUS, count:1, amount:BONUS}}
     ]
   }}]}}
   - If salary/transaction → 201, STOP. Do NOT also create a manual voucher!
   - If salary/transaction → 403, fall back to voucher IMMEDIATELY (do NOT retry):
     POST /ledger/voucher with postings:
     row 1: account 5000 (ACCT_5000_ID from context), amountGross = BASE_SALARY + BONUS (debit)
     row 2: account 2780 (ACCT_2780_ID from context), amountGross = -(BASE_SALARY + BONUS) (credit)
     Use LEDGER ACCOUNT IDS from context. Account 2780 = Skyldig lønn (NOT 1920 Bank!)

── VOUCHER / JOURNAL ENTRY ──
1. GET /ledger/account?fields=id,number,name&number=NNNN (search by account number)
2. POST /ledger/voucher {{date:"{today}", description:"...", postings:[
     {{account:{{id:DEBIT_ACCT}}, amountGross:AMOUNT, amountGrossCurrency:AMOUNT, date:"{today}", row:1, vatType:{{id:0}}}},
     {{account:{{id:CREDIT_ACCT}}, amountGross:-AMOUNT, amountGrossCurrency:-AMOUNT, date:"{today}", row:2, vatType:{{id:0}}}}
   ]}}
- Postings MUST balance (sum of amountGross = 0)
- Use amountGross (positive=debit, negative=credit), NOT debitAmount/creditAmount
- "row" field is REQUIRED on each posting (1, 2, 3...)
- Customer accounts (1500-1599) require customer:{{id:X}} in the posting
- Supplier accounts (2400): allowed when you include supplier:{{id:X}} on the posting (required for supplier invoice vouchers)
- vatType {{id:0}} = no VAT handling (for manual journal entries)

── ACCOUNTING DIMENSIONS ──
1. POST /ledger/accountingDimensionName {{dimensionName:"...", description:"..."}}
   Response contains dimensionIndex (auto-assigned)
2. POST /ledger/accountingDimensionValue {{displayName:"...", number:1, dimensionIndex:X}} (for each value)
3. Link voucher posting to dimension: freeAccountingDimension1:{{id:VALUE_ID}} (for dimensionIndex 1)
   NEVER use "dimensions", "dimension1", or "accountingDimensionValue" on postings

── SUPPLIER INVOICE / INCOMING INVOICE ──
1. POST /customer {{name, organizationNumber, email, phoneNumber, isCustomer:false, isSupplier:true}}
2. GET /ledger/account?number=EXPENSE_ACCT&fields=id (only if NOT in LEDGER ACCOUNT IDS context)
3. Create voucher with 3 postings (calculate: excl = incl/1.25, vat = incl - excl):
   POST /ledger/voucher {{date:"{today}", description:"...", postings:[
     {{account:{{id:EXPENSE_ACCT_ID}}, amountGross:AMOUNT_EXCL_VAT, amountGrossCurrency:AMOUNT_EXCL_VAT, date:"{today}", row:1, vatType:{{id:0}}}},
     {{account:{{id:ACCT_2710_ID}}, amountGross:VAT_AMOUNT, amountGrossCurrency:VAT_AMOUNT, date:"{today}", row:2, vatType:{{id:0}}}},
     {{account:{{id:ACCT_2400_ID}}, amountGross:-AMOUNT_INCL_VAT, amountGrossCurrency:-AMOUNT_INCL_VAT, date:"{today}", row:3, vatType:{{id:0}}, supplier:{{id:SUPPLIER_ID}}}}
   ]}}
   Use ACCT_2710_ID and ACCT_2400_ID from LEDGER ACCOUNT IDS in context.

── UPDATE EXISTING RESOURCE ──
1. GET /resource?fields=id,name,version (search)
2. GET /resource/ID?fields=* (get full object with version)
3. PUT /resource/ID {{...updated fields..., version:V}}

── DELETE RESOURCE ──
1. GET /resource?fields=id,name (search)
2. DELETE /resource/ID

═══ ERROR HANDLING ═══
- 422: Read validationMessages, fix the field. Product duplicate → GET /product?number=XXXX, use existing ID.
- 403: Skip immediately, use fallback approach.
- 409: GET fresh version, retry PUT.
- Duplicate email/number: GET existing, use its ID. NEVER modify email."""


# ═══════════════════════════════════════════════
# FILE PROCESSING
# ═══════════════════════════════════════════════

def extract_pdf_text(data: bytes) -> str:
    """Extract text from PDF using pdfplumber."""
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
    # Fallback: extract readable strings from raw bytes
    text = data.decode("latin-1", errors="ignore")
    readable = re.findall(r'[\w\s@.,;:!?/\\()-]{4,}', text)
    return " ".join(readable)[:4000]


def process_files(files: list) -> tuple[list[str], list[dict]]:
    """Process attached files. Returns (text_parts, image_blocks)."""
    text_parts = []
    image_blocks = []

    for f in files:
        raw_b64 = f["content_base64"]
        data = base64.b64decode(raw_b64)
        filename = f["filename"]
        mime = f.get("mime_type", "")

        if mime.startswith("text") or filename.endswith((".csv", ".json", ".txt")):
            try:
                text_parts.append(f"File '{filename}':\n{data.decode('utf-8')[:8000]}")
            except UnicodeDecodeError:
                text_parts.append(f"File '{filename}': [binary data]")
        elif mime == "application/pdf" or filename.endswith(".pdf"):
            extracted = extract_pdf_text(data)
            if extracted.strip():
                text_parts.append(f"PDF '{filename}':\n{extracted}")
        elif mime.startswith("image/"):
            image_blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": raw_b64},
            })
            text_parts.append(f"[Image '{filename}' attached — visible to you]")

    return text_parts, image_blocks


# ═══════════════════════════════════════════════
# MAIN ENDPOINT
# ═══════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "version": "v12", "model": CLAUDE_MODEL}


@app.post("/solve")
@app.post("/")
async def solve(request: Request):
    start_time = time.time()
    body = await request.json()
    prompt = body["prompt"]
    files = body.get("files", [])
    creds = body["tripletex_credentials"]

    log.info("=" * 60)
    log.info(f"PROMPT: {prompt}")
    log.info(f"FILES: {[f['filename'] for f in files]}")

    # Setup session with larger connection pool
    base_url = creds["base_url"].rstrip("/")
    adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.auth = ("0", creds["session_token"])
    session.headers.update({"Content-Type": "application/json", "Accept": "application/json"})

    # Validate token + get company ID
    company_id = None
    try:
        resp = session.get(f"{base_url}/token/session/>whoAmI", timeout=10)
        if resp.status_code == 200:
            company_id = resp.json().get("value", {}).get("company", {}).get("id")
            log.info(f"Token OK. Company: {company_id}")
        else:
            log.error(f"Token validation failed: {resp.status_code}")
    except Exception as e:
        log.warning(f"Token check error: {e}")

    # Pre-setup: Register bank account so invoices can be created
    try:
        resp = session.get(f"{base_url}/ledger/account", params={"number": "1920", "fields": "id,version,bankAccountNumber"}, timeout=10)
        if resp.status_code == 200:
            accounts = resp.json().get("values", [])
            if accounts and not accounts[0].get("bankAccountNumber"):
                acct = accounts[0]
                session.put(f"{base_url}/ledger/account/{acct['id']}", json={
                    "id": acct["id"], "version": acct["version"],
                    "number": 1920, "name": "Bankinnskudd",
                    "bankAccountNumber": "12345678903", "isBankAccount": True,
                }, timeout=10)
                log.info("Bank account registered on 1920")
    except Exception as e:
        log.warning(f"Bank setup: {e}")

    # Pre-fetch context (employees, departments, divisions, activities)
    ctx = prefetch_context(session, base_url)
    ctx_text = format_prefetched_context(ctx)

    # Process files
    text_parts, image_blocks = process_files(files)
    file_text = "\n---\n".join(text_parts) if text_parts else ""

    # Build user message with pre-fetched context
    user_text = f"Complete this accounting task:\n\n{prompt}"
    if file_text:
        user_text += f"\n\nAttached files:\n{file_text}"
    if ctx_text:
        user_text += f"\n\n═══ PRE-FETCHED SANDBOX DATA ═══\n{ctx_text}"

    content_blocks = [{"type": "text", "text": user_text}]
    content_blocks.extend(image_blocks)

    # Build system prompt
    system_prompt = build_system_prompt(company_id)

    # Agent loop
    messages = [{"role": "user", "content": content_blocks}]
    trace = []
    deadline = start_time + 270  # 30s buffer before 300s timeout

    for iteration in range(20):
        if time.time() > deadline:
            log.warning(f"Deadline reached at iteration {iteration}")
            break

        try:
            response = claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=2048,
                system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
                tools=TOOLS,
                messages=messages,
                temperature=0.0,
            )
        except Exception as e:
            log.error(f"Claude API error: {e}")
            break

        if response.stop_reason == "tool_use":
            tool_blocks = [b for b in response.content if b.type == "tool_use"]

            # Execute tools in parallel if multiple
            if len(tool_blocks) > 1:
                import concurrent.futures
                def run_tool(block):
                    log.info(f"Tool: {block.name}({json.dumps(block.input, ensure_ascii=False)[:800]})")
                    result = execute_tool(block.name, block.input, session, base_url, trace, ctx)
                    return {"type": "tool_result", "tool_use_id": block.id, "content": result}
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                    tool_results = list(ex.map(run_tool, tool_blocks))
            else:
                tool_results = []
                for block in tool_blocks:
                    log.info(f"Tool: {block.name}({json.dumps(block.input, ensure_ascii=False)[:800]})")
                    result = execute_tool(block.name, block.input, session, base_url, trace, ctx)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            # Safety: too many errors
            error_count = sum(1 for t in trace if t.get("status", 200) >= 400)
            if error_count >= 15:
                log.warning(f"Too many errors ({error_count}), stopping")
                break
        else:
            # Claude is done
            log.info(f"Claude finished at iteration {iteration}")
            break

    elapsed = time.time() - start_time
    total_calls = len(trace)
    total_errors = sum(1 for t in trace if t.get("status", 200) >= 400)
    log.info(f"=== DONE {elapsed:.1f}s calls={total_calls} errors={total_errors} ===")

    # Log run
    _log_run(prompt, files, trace, elapsed)

    return JSONResponse({"status": "completed"})


def _log_run(prompt, files, trace, elapsed):
    """Append run details to JSONL log."""
    try:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": CLAUDE_MODEL,
            "prompt": prompt[:500],
            "files": [f["filename"] for f in files] if files and isinstance(files[0], dict) else [],
            "api_calls": len(trace),
            "api_errors": sum(1 for t in trace if t.get("status", 200) >= 400),
            "elapsed_seconds": round(elapsed, 1),
            "trace": trace,
        }
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


@app.get("/logs")
def get_logs():
    """View recent submission logs."""
    if not LOG_PATH.exists():
        return {"runs": [], "count": 0}
    lines = LOG_PATH.read_text().strip().split("\n")
    entries = [json.loads(l) for l in lines if l.strip()]
    return {"runs": entries[-50:], "count": len(entries)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
