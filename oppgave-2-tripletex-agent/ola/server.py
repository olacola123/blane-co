"""
Tripletex AI Accounting Agent — NM i AI 2026
v50-agent: Pure agent architecture. No hardcoded handlers.
Claude sees full sandbox data, reasons about it, makes correct API calls.
"""

import base64
import io
import json
import logging
import os
import re
import time
import traceback
import hashlib
import concurrent.futures
from datetime import date, timedelta, datetime, timezone
from pathlib import Path

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("agent")

app = FastAPI()

# ── LLM Config ──
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")

# ── GCS Logging ──
GCS_BUCKET = "ainm26osl-745-tripletex-logs"
GCS_LOG_BLOB = "submissions.jsonl"
LOG_PATH = Path(os.environ.get("SUBMISSION_LOG", "/tmp/tripletex_submissions.jsonl"))


def _gcs_client():
    try:
        from google.cloud import storage
        return storage.Client()
    except Exception:
        return None


def _gcs_append_log(entry_json: str):
    try:
        client = _gcs_client()
        if not client:
            return
        bucket = client.bucket(GCS_BUCKET)
        if not bucket.exists():
            bucket.location = "EUROPE-NORTH1"
            client.create_bucket(bucket)
        blob = bucket.blob(GCS_LOG_BLOB)
        existing = ""
        if blob.exists():
            existing = blob.download_as_text()
        blob.upload_from_string(existing + entry_json + "\n", content_type="text/plain")
    except Exception as e:
        log.warning(f"GCS log write failed: {e}")


def _gcs_read_logs() -> list:
    try:
        client = _gcs_client()
        if not client:
            return []
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(GCS_LOG_BLOB)
        if not blob.exists():
            return []
        lines = blob.download_as_text().strip().split("\n")
        return [json.loads(l) for l in lines if l.strip()]
    except Exception:
        return []


# ═══════════════════════════════════════════════
# TOOLS — Claude calls these to interact with Tripletex API
# ═══════════════════════════════════════════════

CLAUDE_TOOLS = [
    {
        "name": "tripletex_get",
        "description": "GET request to Tripletex API. Returns JSON response. GET is FREE — use it liberally to understand sandbox state before writing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "endpoint": {"type": "string", "description": "API path, e.g. /employee or /department"},
                "query_params": {"type": "string", "description": "Query string, e.g. 'fields=id,name&count=100'. Use parentheses for nested fields: account(number,name) NOT account.number"},
            },
            "required": ["endpoint"],
        },
    },
    {
        "name": "tripletex_post",
        "description": "POST request to create a resource. Returns JSON response with created object. Each POST counts toward efficiency score.",
        "input_schema": {
            "type": "object",
            "properties": {
                "endpoint": {"type": "string", "description": "API path, e.g. /employee"},
                "body": {"type": "object", "description": "JSON body to send with all required fields"},
                "query_params": {"type": "string", "description": "Query string. Empty string if none."},
            },
            "required": ["endpoint", "body"],
        },
    },
    {
        "name": "tripletex_put",
        "description": "PUT request to update a resource or trigger an action. For actions like /:payment or /:createCreditNote, pass parameters in query_params (NOT body). Each PUT counts toward efficiency score.",
        "input_schema": {
            "type": "object",
            "properties": {
                "endpoint": {"type": "string", "description": "API path, e.g. /employee/123 or /invoice/456/:payment"},
                "body": {"type": "object", "description": "JSON body (for updates). Empty object {} for action endpoints."},
                "query_params": {"type": "string", "description": "Query string. For /:payment: paymentDate=2026-03-22&paidAmount=1000&paymentTypeId=1"},
            },
            "required": ["endpoint"],
        },
    },
    {
        "name": "tripletex_delete",
        "description": "DELETE request to remove a resource. Each DELETE counts toward efficiency score.",
        "input_schema": {
            "type": "object",
            "properties": {
                "endpoint": {"type": "string", "description": "API path with ID, e.g. /travelExpense/789"},
            },
            "required": ["endpoint"],
        },
    },
]


_anthropic_client = None

def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic_client


def claude_generate(system_prompt, messages, tools=None, max_tokens=2048):
    """Call Claude API with tool use support."""
    client = _get_anthropic_client()

    kwargs = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools

    for attempt in range(3):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError:
            wait = min(30, 2 ** attempt + 2)
            log.warning(f"Claude 429, waiting {wait}s (attempt {attempt+1})")
            time.sleep(wait)
        except anthropic.APIError as e:
            log.error(f"Claude API error: {e}")
            return None
        except Exception as e:
            log.error(f"Claude request error: {e}")
            return None
    return None


def claude_extract(system_prompt: str, user_text: str, image_blocks: list = None, max_tokens=1024) -> str:
    """Single Claude call for data extraction. Returns raw text response."""
    client = _get_anthropic_client()

    content = []
    if image_blocks:
        for img in image_blocks:
            if "inline_data" in img:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img["inline_data"]["mime_type"],
                        "data": img["inline_data"]["data"],
                    },
                })
    content.append({"type": "text", "text": user_text})

    for attempt in range(3):
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=max_tokens,
                system=system_prompt or "Extract data as requested. Return ONLY valid JSON.",
                messages=[{"role": "user", "content": content}],
            )
            return response.content[0].text
        except anthropic.RateLimitError:
            wait = min(30, 2 ** attempt + 2)
            time.sleep(wait)
        except Exception as e:
            log.error(f"Claude extract error: {e}")
            return ""
    return ""


# ═══════════════════════════════════════════════
# API EXECUTION — minimal interceptors, no truncation on GET
# ═══════════════════════════════════════════════

def execute_tool(name: str, input_data: dict, session: requests.Session, base_url: str, trace: list, ctx: dict = None) -> str:
    """Execute a Tripletex API tool call. Full response for GET, generous limit for writes."""
    endpoint = input_data.get("endpoint", "")
    qp = input_data.get("query_params", "") or ""
    params = dict(p.split("=", 1) for p in qp.split("&") if "=" in p) if qp else None
    body = input_data.get("body")

    # ── Essential API compatibility interceptors ──

    # Redirect POST /supplier → POST /customer with isSupplier:true
    if name == "tripletex_post" and endpoint.strip("/") == "supplier":
        endpoint = "/customer"
        if body:
            body["isSupplier"] = True
            if "isCustomer" not in body:
                body["isCustomer"] = False
            if body.get("organizationNumber"):
                body["organizationNumber"] = body["organizationNumber"].replace(" ", "")
            if "phoneNumber" not in body:
                body["phoneNumber"] = ""
            if "language" not in body:
                body["language"] = "NO"
            if "invoiceEmail" not in body and body.get("email"):
                body["invoiceEmail"] = body["email"]
            if "supplierNumber" not in body:
                body["supplierNumber"] = 0
            if "accountManager" not in body and ctx and ctx.get("employees"):
                body["accountManager"] = {"id": ctx["employees"][0]["id"]}
        log.info("Redirected POST /supplier → POST /customer with isSupplier:true")

    # Ensure phoneNumber + language on customer POST
    if name == "tripletex_post" and endpoint.strip("/") == "customer" and body:
        if "phoneNumber" not in body:
            body["phoneNumber"] = ""
        if "language" not in body:
            body["language"] = "NO"
        if "invoiceEmail" not in body and body.get("email"):
            body["invoiceEmail"] = body["email"]
        if not body.get("email") and body.get("isSupplier"):
            body["email"] = f"faktura@{body.get('name', 'supplier').lower().replace(' ', '')[:20]}.no"
            body["invoiceEmail"] = body["email"]

    # Block hallucinated endpoints
    fake_endpoints = ("/voucher", "/journalEntry", "/generalLedgerEntry", "/ledger/entry",
                      "/generalLedgerVoucher", "/purchaseInvoice", "/timeSheet",
                      "/projectCost", "/projectHour", "/ledger/paymentTypeIn")
    if endpoint.strip("/") in [e.strip("/") for e in fake_endpoints]:
        log.info(f"BLOCKED hallucinated endpoint {endpoint}")
        trace.append({"tool": name, "endpoint": endpoint, "status": 404, "error": "not found"})
        return json.dumps({"error": f"404 Not Found. The endpoint {endpoint} does not exist. Use POST /ledger/voucher for vouchers.", "status": 404})

    # Fix field dots → parentheses for GET
    if name == "tripletex_get" and params and "fields" in (params or {}):
        fields = params["fields"]
        if "." in fields:
            parts = fields.split(",")
            groups = {}
            result = []
            for p in parts:
                if "." in p:
                    parent, child = p.split(".", 1)
                    if parent not in groups:
                        groups[parent] = []
                    groups[parent].append(child)
                else:
                    result.append(p)
            for parent, children in groups.items():
                result.append(f"{parent}({','.join(children)})")
            params["fields"] = ",".join(result)

    # Auto-inject date range on GET /invoice
    if name == "tripletex_get" and endpoint.strip("/") == "invoice":
        if params is None:
            params = {}
        if "invoiceDateFrom" not in params:
            params["invoiceDateFrom"] = "2020-01-01"
            params["invoiceDateTo"] = "2027-01-01"

    # Fix /supplier/invoice → /supplierInvoice
    if name == "tripletex_get" and endpoint.strip("/") == "supplier/invoice":
        endpoint = "/supplierInvoice"

    # Fix quantity → count on orderline
    if name == "tripletex_post" and "order/orderline" in endpoint and body:
        if "quantity" in body:
            body["count"] = body.pop("quantity")

    # Fix address → postalAddress on customer
    if name == "tripletex_post" and endpoint.strip("/") == "customer" and body:
        if "address" in body and "postalAddress" not in body:
            body["postalAddress"] = body.pop("address")

    # Fix dueDate → invoiceDueDate on invoice
    if name == "tripletex_post" and endpoint.strip("/") == "invoice" and body:
        if "dueDate" in body and "invoiceDueDate" not in body:
            body["invoiceDueDate"] = body.pop("dueDate")

    # Strip activityNumber from activity POST (doesn't exist)
    if name == "tripletex_post" and endpoint.strip("/") == "activity" and body:
        body.pop("activityNumber", None)
        if "isGeneral" not in body:
            body["isGeneral"] = True

    # Fix voucher postings: ensure row and date on each posting
    if name == "tripletex_post" and "ledger/voucher" in endpoint and body:
        voucher_date = body.get("date", date.today().isoformat())
        for i, posting in enumerate(body.get("postings", [])):
            if "row" not in posting:
                posting["row"] = i + 1
            if "date" not in posting:
                posting["date"] = voucher_date

    # Fix employee POST: ensure required fields
    if name == "tripletex_post" and endpoint.strip("/") == "employee" and body:
        if "dateOfBirth" not in body or not body["dateOfBirth"]:
            body["dateOfBirth"] = "1990-01-01"

    # Fix product vatType: look up correct ID from prefetched vat_types
    if name in ("tripletex_post", "tripletex_put") and "product" in endpoint and body:
        vt = body.get("vatType")
        if isinstance(vt, dict) and ctx and ctx.get("vat_types"):
            vt_id = vt.get("id")
            # Check if this vatType ID actually exists in sandbox
            valid_ids = {v["id"] for v in ctx["vat_types"]}
            if vt_id not in valid_ids:
                # Try to find matching outgoing 25% VAT type
                for v in ctx["vat_types"]:
                    if v.get("percentage") == 25 and "utgående" in str(v.get("name", "")).lower():
                        body["vatType"] = {"id": v["id"]}
                        log.info(f"Fixed vatType {vt_id} → {v['id']} ({v.get('name')})")
                        break

    # Fix order POST: ensure deliveryDate
    if name == "tripletex_post" and endpoint.strip("/") == "order" and body:
        if "deliveryDate" not in body:
            body["deliveryDate"] = date.today().isoformat()
        if "orderDate" not in body:
            body["orderDate"] = date.today().isoformat()

    url = f"{base_url}/{endpoint.lstrip('/')}"

    try:
        if name == "tripletex_get":
            resp = session.get(url, params=params, timeout=30)
        elif name == "tripletex_post":
            resp = session.post(url, json=body, params=params, timeout=30)
            # Track new employees in context
            if resp.status_code == 201 and endpoint.strip("/") == "employee" and ctx is not None:
                try:
                    new_emp = resp.json().get("value", {})
                    if new_emp.get("id"):
                        if "employees" not in ctx:
                            ctx["employees"] = []
                        ctx["employees"].append(new_emp)
                except Exception:
                    pass
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
        result = {"raw": resp.text[:2000]}

    text = json.dumps(result, ensure_ascii=False)

    # Generous truncation — GET responses get full view, writes get 6K
    if name == "tripletex_get":
        # GET is free — let agent see data. Truncate only extreme cases.
        if len(text) > 15000:
            if isinstance(result, dict) and "values" in result:
                result["values"] = result["values"][:30]
                result["_truncated"] = True
                result["_total_available"] = result.get("count", "?")
                text = json.dumps(result, ensure_ascii=False)
            if len(text) > 15000:
                text = text[:15000] + "\n... [truncated]"
    else:
        if len(text) > 6000:
            text = text[:6000]

    return text


# ═══════════════════════════════════════════════
# PRE-FETCH CONTEXT — read sandbox state before agent starts
# ═══════════════════════════════════════════════

def prefetch_context(session: requests.Session, base_url: str, task_type: str = "unknown") -> dict:
    """Pre-fetch sandbox data. GET is free — fetch everything we might need."""
    ctx = {}

    def fetch(name, endpoint, params):
        try:
            resp = session.get(f"{base_url}/{endpoint}", params=params, timeout=10)
            if resp.status_code == 200:
                return name, resp.json().get("values", [])
        except Exception as e:
            log.warning(f"Prefetch {name}: {e}")
        return name, []

    # ── Always fetch these (GET is free) ──
    fetches = [
        ("employees", "employee", {"fields": "id,firstName,lastName,email,dateOfBirth,userType,version,department", "count": "100"}),
        ("departments", "department", {"fields": "id,name,departmentNumber", "count": "20"}),
        ("divisions", "division", {"fields": "id,name", "count": "5"}),
        ("activities", "activity", {"fields": "id,name,isGeneral", "count": "50"}),
        ("payment_types", "invoice/paymentType", {"fields": "id,description", "count": "10"}),
        ("products", "product", {"fields": "id,name,number,priceExcludingVatCurrency,vatType", "count": "100"}),
        ("customers", "customer", {"fields": "id,name,organizationNumber,email,isSupplier,isCustomer", "count": "100"}),
        ("suppliers", "supplier", {"fields": "id,name,organizationNumber", "count": "100"}),
        ("currencies", "currency", {"fields": "id,code", "count": "20"}),
        ("vat_types", "ledger/vatType", {"fields": "id,number,name,percentage", "count": "50"}),
    ]

    # Account lookups — all common accounts
    acct_numbers = [1200, 1209, 1210, 1219, 1230, 1240, 1250, 1500, 1700, 1710, 1920,
                    2000, 2050, 2400, 2600, 2710, 2780, 2800, 2900, 2920, 2940, 2960,
                    3000, 3400, 4000, 4300,
                    5000, 5400,
                    6010, 6020, 6100, 6200, 6300, 6340, 6350, 6500, 6540, 6700, 6800, 6900,
                    7000, 7100, 7140, 7300, 7350, 7400, 7500, 7700,
                    8060, 8160, 8300, 8700, 8900]
    for n in acct_numbers:
        fetches.append((f"acct_{n}", "ledger/account", {"number": str(n), "fields": "id,number,name"}))

    # Salary types
    for n in (2000, 2001, 2002, 2005):
        fetches.append((f"salary_{n}", "salary/type", {"number": str(n), "fields": "id,number,name"}))

    # Travel expense data (if relevant)
    if task_type in ("travel_expense", "delete_travel", "unknown"):
        fetches.extend([
            ("cost_categories", "travelExpense/costCategory", {"fields": "id,description", "count": "50"}),
            ("travel_payment_types", "travelExpense/paymentType", {"fields": "id,description", "count": "5"}),
            ("rate_categories", "travelExpense/rateCategory", {"fields": "id,name,type,fromDate,toDate", "count": "500"}),
        ])

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch, name, ep, params): name for name, ep, params in fetches}
        for future in concurrent.futures.as_completed(futures):
            try:
                name, values = future.result()
            except Exception as e:
                log.warning(f"Prefetch future failed: {e}")
                continue

            if name == "employees":
                ctx["employees"] = values
            elif name == "departments":
                ctx["departments"] = values
                if values:
                    ctx["default_department_id"] = values[0]["id"]
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
            elif name == "customers":
                ctx["customers"] = values
            elif name == "suppliers":
                ctx["suppliers"] = values
            elif name == "currencies":
                ctx["currencies"] = {c.get("code"): c.get("id") for c in values if c.get("code")}
            elif name == "vat_types":
                ctx["vat_types"] = values
            elif name == "rate_categories":
                try:
                    for rc in values:
                        n = str(rc.get("name") or "")
                        fr = str(rc.get("fromDate") or "")
                        to = str(rc.get("toDate") or "")
                        if fr >= "2026" and to >= "2026" and "innland" in n.lower():
                            if "overnatting" in n.lower() and "per_diem_overnight_id" not in ctx:
                                ctx["per_diem_overnight_id"] = rc["id"]
                            elif "dag" in n.lower() and "per_diem_day_id" not in ctx:
                                ctx["per_diem_day_id"] = rc["id"]
                except Exception:
                    pass
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

    # Fetch employments for employees
    if ctx.get("employees"):
        def fetch_employment(emp):
            try:
                resp = session.get(f"{base_url}/employee/employment",
                    params={"employeeId": emp["id"], "fields": "id,startDate,division", "count": "5"}, timeout=10)
                if resp.status_code == 200:
                    return emp["id"], resp.json().get("values", [])
            except Exception:
                pass
            return emp["id"], []

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            empl_results = list(ex.map(fetch_employment, ctx["employees"]))
        for emp_id, empls in empl_results:
            for e in ctx["employees"]:
                if e["id"] == emp_id:
                    e["_employments"] = empls
                    break

    log.info(f"Prefetched: {len(ctx.get('employees', []))} employees, {len(ctx.get('departments', []))} depts, "
             f"{len(ctx.get('ledger_accounts', {}))} accounts, {len(ctx.get('vat_types', []))} vatTypes, "
             f"{len(ctx.get('customers', []))} customers, {len(ctx.get('products', []))} products")

    return ctx


def format_prefetched_context(ctx: dict) -> str:
    """Format pre-fetched sandbox data for the agent. This is the agent's eyes into the sandbox."""
    parts = []

    # VAT Types — build clear summary for agent
    if ctx.get("vat_types"):
        common = {}
        for v in ctx["vat_types"]:
            vid = v.get('id', '')
            vname = (v.get('name') or '').lower()
            pct = str(v.get('percentage', ''))
            if '25' in pct and any(w in vname for w in ('utgående', 'outgoing', 'ut ')):
                common['outgoing_25'] = vid
            elif '25' in pct and any(w in vname for w in ('inngående', 'incoming', 'inn', 'inng')):
                common['ingoing_25'] = vid
            elif '15' in pct and any(w in vname for w in ('utgående', 'outgoing', 'ut ')):
                common['outgoing_15'] = vid
            elif '15' in pct and any(w in vname for w in ('inngående', 'incoming', 'inn', 'inng')):
                common['ingoing_15'] = vid
            elif pct in ('0', '0.0') and any(w in vname for w in ('fritatt', 'exempt', 'avgiftsfri')):
                common['exempt_0'] = vid
            elif pct in ('0', '0.0') and any(w in vname for w in ('utenfor', 'outside')):
                common['outside_0'] = vid
            elif pct in ('0', '0.0') and any(w in vname for w in ('ingen', 'none', 'no vat', 'no tax')):
                common['none'] = vid
            elif v.get('number') == 0 or vid == 0:
                common['none'] = vid

        summary = ["═══ VAT TYPE IDs (use these EXACT ids in vatType:{id:X}) ═══"]
        if 'outgoing_25' in common: summary.append(f"  OUTGOING 25% (for products, invoices, sales): vatType:{{id:{common['outgoing_25']}}}")
        if 'ingoing_25' in common: summary.append(f"  INGOING 25% (for supplier invoices, receipts, expenses): vatType:{{id:{common['ingoing_25']}}}")
        if 'outgoing_15' in common: summary.append(f"  OUTGOING 15% food: vatType:{{id:{common['outgoing_15']}}}")
        if 'ingoing_15' in common: summary.append(f"  INGOING 15% food (receipt for food): vatType:{{id:{common['ingoing_15']}}}")
        if 'exempt_0' in common: summary.append(f"  EXEMPT 0% (purregebyr, fritatt): vatType:{{id:{common['exempt_0']}}}")
        if 'outside_0' in common: summary.append(f"  OUTSIDE 0% (utenfor MVA): vatType:{{id:{common['outside_0']}}}")
        if 'none' in common: summary.append(f"  NO VAT (balance postings, bank): vatType:{{id:{common['none']}}}")
        parts.append("\n".join(summary))
        ctx["_vat_summary"] = common

    # Ledger accounts
    if ctx.get("ledger_accounts"):
        accts = ctx["ledger_accounts"]
        lines = [f"  {num}: id={aid}" for num, aid in sorted(accts.items(), key=lambda x: int(x[0]))]
        parts.append("LEDGER ACCOUNT IDS (use directly — never GET /ledger/account for these):\n" + "\n".join(lines))

    # Employees
    if ctx.get("employees"):
        emp_lines = []
        for e in ctx["employees"]:
            dept_id = ""
            if isinstance(e.get("department"), dict):
                dept_id = e["department"].get("id", "")
            empls = e.get("_employments", [])
            if empls:
                empl = empls[0]
                div_id = empl.get("division", {}).get("id", "") if isinstance(empl.get("division"), dict) else ""
                empl_info = f" hasEmployment=YES employmentId={empl['id']} divisionId={div_id}"
            else:
                empl_info = " hasEmployment=NO"
            emp_lines.append(f"  id={e['id']} email={e.get('email','')} name={e.get('firstName','')} {e.get('lastName','')} "
                           f"dateOfBirth={e.get('dateOfBirth','')} userType={e.get('userType','')} version={e.get('version','')} "
                           f"dept={dept_id}{empl_info}")
        parts.append("EXISTING EMPLOYEES:\n" + "\n".join(emp_lines))

    # Departments
    if ctx.get("departments"):
        dept_lines = [f"  id={d['id']} name={d.get('name','')} number={d.get('departmentNumber','')}" for d in ctx["departments"]]
        parts.append("EXISTING DEPARTMENTS:\n" + "\n".join(dept_lines))

    if ctx.get("default_division_id"):
        parts.append(f"DEFAULT DIVISION ID: {ctx['default_division_id']}")
    else:
        parts.append("NO DIVISIONS EXIST — you must create one for employment")

    # Activities
    if ctx.get("activities"):
        act_lines = [f"  id={a['id']} name={a.get('name','')}" for a in ctx["activities"]]
        parts.append("AVAILABLE ACTIVITIES:\n" + "\n".join(act_lines))

    # Products
    if ctx.get("products"):
        prod_lines = [f"  id={p['id']} number={p.get('number','')} name={p.get('name','')} priceExVat={p.get('priceExcludingVatCurrency','')}" for p in ctx["products"][:30]]
        parts.append("EXISTING PRODUCTS (reuse if name/number matches):\n" + "\n".join(prod_lines))

    # Customers
    if ctx.get("customers"):
        cust_lines = [f"  id={c['id']} name={c.get('name','')} orgNr={c.get('organizationNumber','')} isSupplier={c.get('isSupplier',False)}" for c in ctx["customers"][:30]]
        parts.append("EXISTING CUSTOMERS (reuse if name/orgNr matches):\n" + "\n".join(cust_lines))

    # Suppliers
    if ctx.get("suppliers"):
        sup_lines = [f"  id={s['id']} name={s.get('name','')} orgNr={s.get('organizationNumber','')}" for s in ctx["suppliers"][:30]]
        parts.append("EXISTING SUPPLIERS:\n" + "\n".join(sup_lines))

    # Salary types
    if ctx.get("salary_types"):
        st_lines = [f"  number={v['number']} id={v['id']} name={v['name']}" for v in ctx["salary_types"].values()]
        parts.append("SALARY TYPES:\n" + "\n".join(st_lines))

    # Payment types
    if ctx.get("payment_types"):
        pt_lines = [f"  id={p['id']} description={p.get('description','')}" for p in ctx["payment_types"]]
        parts.append("INVOICE PAYMENT TYPES:\n" + "\n".join(pt_lines))

    # Currencies
    if ctx.get("currencies"):
        curr_lines = [f"  {code}: id={cid}" for code, cid in sorted(ctx["currencies"].items())]
        parts.append("CURRENCIES:\n" + "\n".join(curr_lines))

    # Travel-specific
    if ctx.get("per_diem_overnight_id") or ctx.get("per_diem_day_id"):
        pd_lines = []
        if ctx.get("per_diem_overnight_id"):
            pd_lines.append(f"  Overnight: {ctx['per_diem_overnight_id']}")
        if ctx.get("per_diem_day_id"):
            pd_lines.append(f"  Day trip: {ctx['per_diem_day_id']}")
        parts.append("PER DIEM RATE CATEGORY IDs:\n" + "\n".join(pd_lines))

    if ctx.get("cost_categories"):
        cc_lines = [f"  id={c['id']} desc={c.get('description','')}" for c in ctx["cost_categories"][:20]]
        parts.append("TRAVEL COST CATEGORIES:\n" + "\n".join(cc_lines))

    if ctx.get("travel_payment_types"):
        tp_lines = [f"  id={t['id']} desc={t.get('description','')}" for t in ctx["travel_payment_types"]]
        parts.append("TRAVEL PAYMENT TYPES:\n" + "\n".join(tp_lines))

    return "\n\n".join(parts)


# ═══════════════════════════════════════════════
# SYSTEM PROMPT — one clean prompt, task-specific recipes
# ═══════════════════════════════════════════════

def build_system_prompt(company_id: int | None, task_type: str = "unknown") -> str:
    today = date.today().isoformat()
    due = (date.today() + timedelta(days=30)).isoformat()

    core = f"""You are a Tripletex accounting API agent. Complete the given task by calling the API.

TODAY: {today}  DUE: {due}  COMPANY_ID: {company_id or "unknown"}

CRITICAL RULES:
1. Use IDs from PRE-FETCHED SANDBOX DATA. Never hardcode IDs — they change per sandbox.
2. GET is FREE (no scoring penalty). Use GET to look up anything you need. POST/PUT/DELETE count.
3. Use EXACT values from the prompt. Dates MUST be YYYY-MM-DD.
4. On 422: read validationMessages, fix the field, retry. On 403: use alternative approach.
5. PUT actions (/:payment, /:createCreditNote): params in query_params, NOT body.
6. Never set "id" or "version" in POST. Include "version" in PUT.
7. Batch independent calls in ONE response — each iteration costs ~5s.
8. Prices are EXCLUDING VAT unless prompt says "inkl. MVA"/"TTC"/"con IVA incluido".
9. userType: "STANDARD" | "EXTENDED" | "NO_ACCESS". Never "ADMINISTRATOR".
10. Use parentheses for nested fields in GET: account(number,name) NOT account.number

LANGUAGES: NO, EN, ES, PT, NN, DE, FR. Parse dates/amounts in any language.

SUPPLIER CREATION: Always POST /customer with isSupplier:true (NOT POST /supplier — it doesn't work).

VOUCHER FORMAT:
POST /ledger/voucher {{date, description, postings:[
  {{account:{{id:X}}, amountGross:AMT, amountGrossCurrency:AMT, date:"YYYY-MM-DD", row:1, vatType:{{id:VT}}}},
  {{account:{{id:Y}}, amountGross:-AMT, amountGrossCurrency:-AMT, date:"YYYY-MM-DD", row:2, vatType:{{id:NO_VAT_ID_FROM_CONTEXT}}}}
]}}
Postings MUST balance to 0. row and date REQUIRED on each.
For zero-VAT postings: use the "Ingen avgiftsbehandling" vatType ID from PRE-FETCHED DATA (never hardcode id:0!).
Customer accounts (1500) need customer:{{id:X}}. Supplier accounts (2400) need supplier:{{id:X}}.

ENDPOINTS THAT DON'T EXIST: /voucher, /journalEntry, /generalLedgerEntry, /purchaseInvoice, /timeSheet.
Use /ledger/voucher for ALL vouchers/journal entries."""

    recipes = _get_recipes(task_type, today, due, company_id)

    return core + "\n\n" + recipes


def _get_recipes(task_type: str, today: str, due: str, company_id) -> str:
    """Task-specific guidance. References sandbox data instead of hardcoding."""

    if task_type == "employee":
        return f"""═══ EMPLOYEE TASK ═══
1. Match email against EXISTING EMPLOYEES → PUT /employee/ID (include version! keep dateOfBirth!)
2. Or POST /employee {{firstName, lastName, email, userType:"STANDARD", department:{{id:FROM_CONTEXT}}, dateOfBirth, employeeNumber}}
3. POST /division {{name:"Hovedkontor", startDate:"{today}", municipality:{{id:LOOKUP_VIA_GET}}, organizationNumber:"FROM_PROMPT"}}
   GET /municipality to find correct municipality ID! Never hardcode.
4. POST /employee/employment {{employee:{{id:X}}, startDate:"FROM_PROMPT", division:{{id:DIV}}, isMainEmployer:true, taxDeductionCode:"loennFraHovedarbeidsgiver"}}
5. POST /employee/employment/details {{employment:{{id:X}}, date:"{today}", employmentType:"ORDINARY", employmentForm:"PERMANENT", remunerationType:"MONTHLY_WAGE", workingHoursScheme:"NOT_SHIFT", percentageOfFullTimeEquivalent:100.0}}"""

    elif task_type == "employee_pdf":
        return f"""═══ EMPLOYEE FROM PDF/CONTRACT ═══
Extract ALL details from the PDF. EVERY field is scored!
1. POST /employee {{firstName, lastName, email (from PDF or firstname.lastname@example.org), dateOfBirth, employeeNumber, nationalIdentityNumber:"11-DIGIT", userType:"STANDARD", department:{{id:FROM_CONTEXT}}}}
2. POST /division (if needed) → POST /employee/employment → POST /employment/details with percentageOfFullTimeEquivalent from PDF
3. GET /employee/employment/occupationCode?code=DIGITS_FROM_PDF → PUT /employee/employment/ID with occupationCode:{{id:OCC_ID}}
4. POST /employee/standardTime {{employee:{{id:X}}, fromDate:START_DATE, hoursPerDay:7.5×(pct/100)}}
5. If salary mentioned: monthly = annual/12. POST /salary/transaction or paySlip.
CRITICAL: nationalIdentityNumber + occupationCode are SCORED!"""

    elif task_type == "customer":
        return """═══ CUSTOMER TASK ═══
POST /customer {name, organizationNumber, email, phoneNumber, isCustomer:true, postalAddress:{addressLine1, postalCode, city}}
postalAddress.addressLine1 (NOT address1!). phoneNumber (NOT phoneNumberMobile).
Reuse existing customer if name/orgNr matches (check EXISTING CUSTOMERS)."""

    elif task_type == "supplier":
        return """═══ SUPPLIER TASK ═══
POST /customer {name, organizationNumber, email, phoneNumber:"", isCustomer:false, isSupplier:true, postalAddress:{addressLine1, postalCode, city}}
Always via POST /customer with isSupplier:true — NOT POST /supplier!"""

    elif task_type == "product":
        return """═══ PRODUCT TASK ═══
POST /product {name, number:"STRING!", priceExcludingVatCurrency:X, priceIncludingVatCurrency:Y, vatType:{id:FROM_CONTEXT}}
number must be STRING. Look up vatType ID from VAT TYPES in context (e.g. 25% outgoing)."""

    elif task_type == "departments":
        return """═══ DEPARTMENTS TASK ═══
POST /department {name, departmentNumber:INT}
departmentNumber must be unique. Check EXISTING DEPARTMENTS for used numbers."""

    elif task_type in ("invoice_send", "invoice_multi"):
        return f"""═══ INVOICE TASK ═══
1. Find/create customer (check EXISTING CUSTOMERS first)
2. POST /product — number as STRING, vatType from context, set BOTH priceExcludingVatCurrency AND priceIncludingVatCurrency
3. POST /order {{customer:{{id:X}}, deliveryDate:"{today}", orderDate:"{today}", isPrioritizeAmountsIncludingVat:false}}
4. POST /order/orderline — ALWAYS set vatType AND unitPriceExcludingVatCurrency on EACH line
5. POST /invoice {{invoiceDate:"{today}", invoiceDueDate:"{due}", orders:[{{id:X}}]}} query_params: sendToCustomer=false
For multi-line: one product + one orderline per product line. Different vatType per line if needed."""

    elif task_type == "order":
        return f"""═══ ORDER TASK ═══
1. Find/create customer + products
2. POST /order → POST /order/orderline per product
3. PUT /order/ID/:invoice query_params: invoiceDate={today}&sendToCustomer=false
4. PUT /invoice/ID/:payment query_params: paymentDate={today}&paymentTypeId=ID&paidAmount=TOTAL_INCL_VAT"""

    elif task_type == "payment":
        return f"""═══ PAYMENT TASK ═══
1. GET /invoice?invoiceDateFrom=2020-01-01&invoiceDateTo=2027-01-01 — find the invoice by amount/customer
2. PUT /invoice/ID/:payment query_params: paymentDate={today}&paymentTypeId=PAYMENT_TYPE_ID&paidAmount=AMOUNT"""

    elif task_type == "reverse_payment":
        return f"""═══ REVERSE PAYMENT TASK ═══
1. GET /invoice — find the invoice
2. PUT /invoice/ID/:payment with paidAmount=-AMOUNT (negative!) to reverse"""

    elif task_type == "credit_note":
        return f"""═══ CREDIT NOTE TASK ═══
1. First CREATE the original invoice if it doesn't exist (customer→product→order→orderline→invoice)
2. PUT /invoice/ID/:createCreditNote query_params: date={today}&sendToCustomer=false
If "allerede kreditert" error: find another matching invoice."""

    elif task_type == "supplier_invoice" or task_type == "supplier_invoice_pdf":
        return f"""═══ SUPPLIER INVOICE TASK ═══
1. Find/create supplier (POST /customer with isSupplier:true)
2. POST /ledger/voucher — go straight to voucher (do NOT try /supplierInvoice or /incomingInvoice):
   Debit: expense account (from prompt context), amountGross=TOTAL_INCL_VAT, vatType={{id:INGOING_25%_FROM_CONTEXT}} (auto-splits MVA)
   Credit: account 2400, amountGross=-TOTAL_INCL_VAT, vatType={{id:0/none}}, supplier:{{id:SUPPLIER_ID}}
   Use the INVOICE DATE from prompt — not today!
Expense account mapping: 6540=office/kontor, 6340=IT/data, 6500=supplies, 7100=vehicle, 7140=travel, 6800=other"""

    elif task_type == "receipt_voucher":
        return f"""═══ RECEIPT VOUCHER TASK ═══
POST /ledger/voucher:
  Debit: expense account by WHAT WAS BOUGHT (not department!), vatType=ingoing 25% from context, department:{{id:DEPT_ID}}
  Credit: account 1920 (bank), vatType={{id:0}}
Account mapping by purchase type:
  6340=IT/USB/PC/laptop/electronics, 7140=travel/tog/fly/taxi, 7100=vehicle/bensin/parkering
  6500=office supplies/papir, 6540=furniture/møbler, 7350=entertainment/restaurant/gave
  6900=telefon, 6300=rent/husleie, 7700=maintenance, 6800=other
Food items: use INGOING 15% vatType from context (not 25%)!
Use RECEIPT DATE from prompt, not today!"""

    elif task_type == "salary":
        return f"""═══ SALARY TASK ═══
1. Find employee by email in EXISTING EMPLOYEES. PUT with name+dateOfBirth from context.
2. If hasEmployment=NO: create division+employment+details. If YES: skip to 3.
3. POST /employee/standardTime {{employee:{{id:X}}, fromDate:"{today}", hoursPerDay:7.5}} (ignore 422)
4. Try POST /salary/paySlip with employee, date, year, month, and specifications inline.
   If paySlip returns 404/403: fall back to manual voucher:
   POST /ledger/voucher with:
     Debit 5000 (lønn): gross salary amount
     Credit 2780 (skattetrekk): tax deduction (~30%)
     Credit 2960 (feriepenger): holiday pay accrual (10.2% of gross, 12% if over 60)
     Credit 1920 (bank): net payment (gross - tax)
     Debit 5400 (arbeidsgiveravgift): 14.1% of gross
     Credit 2780 (skyldig AGA): 14.1% of gross
5. After successful paySlip: PUT /:calculate → PUT /:createPayment → STOP."""

    elif task_type == "travel_expense":
        return f"""═══ TRAVEL EXPENSE TASK ═══
1. Find/create employee with employment
2. POST /travelExpense {{employee:{{id:X}}, title:"...", travelDetails:{{departureDate, returnDate, departureFrom, destination, purpose}}}}
3. Per diem: POST /travelExpense/perDiemCompensation {{travelExpense:{{id:X}}, rateCategory:{{id:FROM_CONTEXT}}, count:DAYS, overnightAccommodation:"HOTEL", location:"CITY"}}
4. Costs: POST /travelExpense/cost {{travelExpense:{{id:X}}, costCategory:{{id:FROM_CONTEXT}}, paymentType:{{id:FROM_CONTEXT}}, amountCurrencyIncVat:AMT, currency:{{id:1}}, date:"DATE"}}"""

    elif task_type == "delete_travel":
        return """═══ DELETE TRAVEL EXPENSE ═══
GET /travelExpense → if APPROVED: PUT /travelExpense/ID/:unapprove → DELETE /travelExpense/ID"""

    elif task_type == "bank_recon":
        return f"""═══ BANK RECONCILIATION ═══
1. Parse CSV: date, amount, reference per transaction
2. GET /invoice?invoiceDateFrom=2020-01-01&invoiceDateTo=2027-01-01&fields=id,invoiceNumber,amount,amountOutstanding,customer(id,name)
3. Incoming payments (positive): match by amount/ref → PUT /invoice/ID/:payment
4. Outgoing payments (negative): POST /ledger/voucher — debit 2400 (with supplier), credit 1920
   Look up account IDs from LEDGER ACCOUNT IDS — never hardcode id:0!
5. Handle partial payments. Match by reference/name if exact amount doesn't match."""

    elif task_type == "ledger_audit":
        return """═══ LEDGER AUDIT ═══
The prompt TELLS you the errors. You do NOT need to find them — go straight to corrections.
For each error: POST /ledger/voucher with corrective postings:
  Wrong account: debit correct, credit wrong (reverse the error)
  Duplicate: reverse it (swap signs)
  Missing VAT: debit expense with ingoing VAT type (auto-splits), credit same amount vatType:0
  Wrong amount: reverse wrong, post correct
Look up any account IDs you need via GET /ledger/account?number=XXXX.
Each correction = separate voucher with description: "Korreksjon: [reason]" """

    elif task_type == "year_end":
        return f"""═══ YEAR-END CLOSING ═══
1. Depreciation: SEPARATE voucher per asset. debit 6010/6020, credit 1209/1219. Linear = cost/years.
2. Prepaid reversal: debit expense(6300/7140), credit 1700.
3. Salary accrual: debit 5000, credit 2900.
4. Tax 22%: GET /ledger/posting for full year → calc profit → debit 8700, credit 2920.
   Calculate tax AFTER depreciation+prepaid vouchers!
5. Profit distribution: debit 8900, credit 2050.
Look up ALL account IDs from LEDGER ACCOUNT IDS in context."""

    elif task_type == "month_end":
        return """═══ MONTHLY CLOSING ═══
1. Prepaid accrual: debit expense, credit 1710. Monthly = total/months.
2. Monthly depreciation: annual/12 per asset.
3. Additional accruals as specified in prompt.
All account IDs from context."""

    elif task_type == "reminder_fee":
        return f"""═══ REMINDER FEE ═══
1. GET invoices, find overdue (amountOutstanding>0, past due)
2. Reminder voucher: debit 1500, credit 3400, amount=35 (purregebyr)
3. Create reminder invoice: product "Purregebyr" 35kr, vatType=outgoing 0% exempt from context
4. Partial payment on overdue: PUT /invoice/ID/:payment"""

    elif task_type == "fx_invoice":
        return f"""═══ FX / CURRENCY INVOICE ═══
1. Find the customer and invoice from prompt
2. Pay the invoice: PUT /invoice/ID/:payment
3. Calculate exchange rate difference: diff = amount × (newRate - oldRate)
   GAIN (diff > 0): debit 1500, credit 8060
   LOSS (diff < 0): debit 8160, credit 1500
4. POST /ledger/voucher for the FX difference. Use account 1500 (kundefordringer)."""

    elif task_type == "project" or task_type == "project_fixed":
        extra = ""
        if task_type == "project_fixed":
            extra = "\nAfter creating: GET /project/ID → PUT /project/ID with isFixedPrice:true, fixedprice:AMT, version:V. KEEP original number!"
        return f"""═══ PROJECT TASK ═══
1. Make employee EXTENDED: PUT /employee/ID with userType:"EXTENDED" + version
2. POST /employee/entitlement {{employee:{{id:X}}, entitlementId:45, customer:{{id:{company_id or 'COMPANY_ID'}}}}}
   POST /employee/entitlement {{employee:{{id:X}}, entitlementId:10, customer:{{id:{company_id or 'COMPANY_ID'}}}}}
3. Find/create customer
4. POST /project {{name, startDate:"{today}", projectManager:{{id:EMP}}, customer:{{id:CUST}}}} — no "number"!{extra}"""

    elif task_type == "project_lifecycle":
        return f"""═══ PROJECT LIFECYCLE ═══
Full cycle: employees → customer → project → hours → supplier cost → customer invoice.
1. For EACH employee: PUT to EXTENDED + entitlements(45+10).
2. POST /customer (or use existing)
3. POST /project
4. GET /project/ID?fields=id,projectActivities(id,activity(id,name)) to find valid activity IDs
5. POST /timesheet/entry per employee {{project:{{id:P}}, activity:{{id:FROM_PROJECT_ACTIVITIES}}, employee:{{id:E}}, date:"{today}", hours:N}}
6. Supplier cost: POST /ledger/voucher debit 4300(vatType:ingoing25%), credit 2400(supplier)
7. Customer invoice: product→order→orderline→invoice. vatType=outgoing 25% on both."""

    elif task_type == "cost_analysis":
        return f"""═══ COST ANALYSIS ═══
1. GET /ledger/posting?dateFrom=2026-01-01&dateTo=2026-01-31&fields=id,amount,account(number,name)&count=1000
2. GET /ledger/posting?dateFrom=2026-02-01&dateTo=2026-02-28&fields=...&count=1000
3. Group by account, sum per month, find 3 accounts with largest increase
4. Make employee EXTENDED + entitlements (45+10, customer:{{id:{company_id or 'COMPANY'}}})
5. For each of 3 accounts: POST /project + POST /activity {{name, activityType:"GENERAL_ACTIVITY", isGeneral:true}}
   CRITICAL: activityType is REQUIRED!"""

    elif task_type == "acct_dimension":
        return f"""═══ ACCOUNTING DIMENSIONS ═══
1. POST /ledger/accountingDimensionName {{dimensionName, description}} → get dimensionIndex
2. POST /ledger/accountingDimensionValue {{displayName, number:1, dimensionIndex:X}} per value
3. POST /ledger/voucher to link dimension: use freeAccountingDimension1:{{id:VALUE_ID}} on posting"""

    elif task_type == "contact_person":
        return """═══ CONTACT PERSON ═══
1. Find/create customer
2. POST /contact {firstName, lastName, email, customer:{id:CUST_ID}}"""

    elif task_type == "timesheet":
        return f"""═══ TIMESHEET TASK ═══
1. Setup employee + project with entitlements
2. Use activity ID from AVAILABLE ACTIVITIES context
3. POST /timesheet/entry {{project:{{id:P}}, activity:{{id:A}}, employee:{{id:E}}, date:"{today}", hours:N}}
4. Create invoice: product(priceExVat=hours×rate) → order → orderline → invoice"""

    else:
        return """═══ UNKNOWN TASK ═══
Read the prompt carefully. Use GET to explore the sandbox before making any writes.
Common patterns:
- Customer/employee/product: POST to create
- Invoice: customer → product → order → orderline → invoice
- Voucher: POST /ledger/voucher with balanced postings
- Supplier: POST /customer with isSupplier:true
Always check EXISTING data before creating new resources."""


# ═══════════════════════════════════════════════
# FILE PROCESSING
# ═══════════════════════════════════════════════

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
                return "\n".join(parts)[:8000]
    except Exception as e:
        log.warning(f"pdfplumber failed: {e}")
    text = data.decode("latin-1", errors="ignore")
    readable = re.findall(r'[\w\s@.,;:!?/\\()-]{4,}', text)
    return " ".join(readable)[:4000]


def process_files(files: list) -> tuple[list[str], list[dict]]:
    text_parts = []
    image_blocks = []
    for f in files:
        raw_b64 = f["content_base64"]
        data = base64.b64decode(raw_b64)
        filename = f["filename"]
        mime = f.get("mime_type", "")
        if mime.startswith("text") or filename.endswith((".csv", ".json", ".txt")):
            try:
                text_parts.append(f"File '{filename}':\n{data.decode('utf-8')[:16000]}")
            except UnicodeDecodeError:
                text_parts.append(f"File '{filename}': [binary data]")
        elif mime == "application/pdf" or filename.endswith(".pdf"):
            extracted = extract_pdf_text(data)
            if extracted.strip():
                text_parts.append(f"PDF '{filename}':\n{extracted}")
            else:
                text_parts.append(f"[PDF '{filename}' — text extraction failed, content sent as image]")
            # Send PDF pages as JPEG images for Claude vision
            try:
                import pdfplumber
                with pdfplumber.open(io.BytesIO(data)) as pdf:
                    for i, page in enumerate(pdf.pages[:3]):
                        try:
                            img = page.to_image(resolution=150).original
                            buf = io.BytesIO()
                            # Always convert to RGB before JPEG (handles RGBA/P/LA/CMYK modes)
                            img_rgb = img.convert("RGB") if hasattr(img, "convert") else img
                            img_rgb.save(buf, format="JPEG", quality=85)
                            img_data = buf.getvalue()
                            if img_data:
                                img_b64 = base64.b64encode(img_data).decode()
                                image_blocks.append({
                                    "inline_data": {"mime_type": "image/jpeg", "data": img_b64},
                                })
                                log.info(f"PDF page {i}: {len(img_data)} bytes as JPEG")
                        except Exception as e_page:
                            log.warning(f"PDF page {i} render failed: {e_page}")
                text_parts.append(f"[PDF '{filename}' pages rendered as images — visible to you]")
            except Exception as e:
                log.warning(f"PDF to image failed: {e}")
        elif mime.startswith("image/"):
            _valid = {"image/jpeg", "image/png", "image/gif", "image/webp"}
            if mime in _valid:
                image_blocks.append({"inline_data": {"mime_type": mime, "data": raw_b64}})
                text_parts.append(f"[Image '{filename}' attached — visible to you]")
            else:
                # Try converting unsupported format to JPEG
                try:
                    from PIL import Image as PILImage
                    img = PILImage.open(io.BytesIO(base64.b64decode(raw_b64))).convert("RGB")
                    buf2 = io.BytesIO()
                    img.save(buf2, format="JPEG", quality=85)
                    img_b64 = base64.b64encode(buf2.getvalue()).decode()
                    image_blocks.append({"inline_data": {"mime_type": "image/jpeg", "data": img_b64}})
                    text_parts.append(f"[Image '{filename}' converted to JPEG — visible to you]")
                except Exception:
                    text_parts.append(f"[Image '{filename}' format {mime!r} not supported, skipped]")
    return text_parts, image_blocks


# ═══════════════════════════════════════════════
# TASK TYPE DETECTION
# ═══════════════════════════════════════════════

def detect_task_type(prompt: str) -> str:
    import unicodedata
    p = unicodedata.normalize("NFKD", prompt.lower())
    p = p.replace("ø", "o").replace("å", "a").replace("æ", "ae").replace("ü", "u").replace("ö", "o").replace("ä", "a")
    p = p.encode("ascii", "ignore").decode("ascii")

    if re.search(r'(slett|delete|supprim|elimin).*(reise|travel|viagem|frais)|(reise|travel|viagem|frais).*(slett|delete|supprim|elimin)', p):
        return "delete_travel"
    if re.search(r'manedsavslut|manadsavslut|manavslutn|monatsabschluss|month.end.clos|cloture.mensuel|cierre.mensual|encerramento.mensal|monthly.closing', p):
        return "month_end"
    if re.search(r'avskriv|depreciation|abschreibung|arsoppgjor|year.end|encerramento.anual|cloture.annuelle|forenkl.*arsoppgj', p):
        return "year_end"
    if re.search(r'reconcil|bankavsteming|bankavstemming|extracto.bancario|avstem.*bank|concilia.*extracto|bank.statement|bankutskrift|relev.+bancaire|rapprochez.*relev|kontoauszug|extrato.banc', p):
        return "bank_recon"
    if re.search(r'erros.*livro|errors.*ledger|feil.*hovedbok|errores.*libro|fehler.*hauptbuch|erreurs.*grand.livre|erros.*razao|feil.*bilag|errors.*voucher', p):
        return "ledger_audit"
    if re.search(r'ciclo.de.vida|prosjektsykl|project.lifecycle|hele.prosjekt|full.project.cycle|prosjektsyklusen|ciclo.*completo.*projeto', p):
        return "project_lifecycle"
    if re.search(r'costos.*aumentaron|costs.*increased|utgift.*okt|identifique.*gastos|identify.*expense.*increase|analice.*libro.*mayor|analys.*ledger.*identify|totalkostnad.*auka|kostnad.*okt.*januar|despesas.*aumentaram|couts.*augmente|kosten.*gestiegen|finn.*tre.*kostnadskonto|find.*three.*expense.*account|identifique.*tres.*contas', p):
        return "cost_analysis"
    if re.search(r'cambio|exchange.rate|wechselkurs|tipo.de.cambio|taux.de.change|kursforskjell|agio|disagio|\beur\b.*nok.*rate|\beur\b.*taxa|\beur\b.*kurs', p):
        return "fx_invoice"
    if re.search(r'purregebyr|reminder.fee|lembrete.*taxa|mahngebuh?r|taxa.de.lembrete|fatura.vencida.*taxa|overdue.*reminder|forfalt.*faktura.*gebyr|facture.*impayee.*frais|factura.*vencida.*cargo|uberfallige.*rechnung|uberfalliger?.*rechnung', p):
        return "reminder_fee"
    if re.search(r'arbeidskontrakt|employment.contract|contrato.de.trabaj|contrato.de.trabalh|arbeitsvertrag|contrat.de.travail|carta.de.oferta|offer.letter|lettre.d.offre|tilbudsbrev|tilbodsbrev|onboarding|integra.+compl|incorpora.+compl|funcionario.*tripletex|crie.*funcionario', p):
        return "employee_pdf"
    if re.search(r'kvittering|receipt|quittung|recibo|recu|necesitamos.*gasto|gasto.*recibo|besoin.*depense|depense.*recu|ce recu|dieser quittung|dette kvittering|this receipt|deste recibo|este recibo|comptabiliser.*depense|enregistrer.*depense|depense.*recu|justificatif|cette.*depense|depense.*achat|achat.*nota.*fiscal|nota.*fiscal|bon.*caisse|kassabon|scontrino|kassakvitto|esta depensa|este gasto|diese ausgabe|questa spesa', p) and not re.search(r'inv-\d+', p) and not re.search(r'reise|viaje|viagem|voyage|dienstreise|deplacement|travel', p):
        return "receipt_voucher"
    if re.search(r'(lieferantenrechnung|supplier.invoice|factura.+proveedor|fatura.+fornecedor).*(pdf|beigefugt|attached|adjunt|anexo)', p):
        return "supplier_invoice_pdf"
    if re.search(r'lonn|lon\b|payroll|nomina|gehalt|salario|salary|folha.de.pagamento', p):
        if not re.search(r'monatsabschluss|manedsavslut|manadsavslut|manavslutn|month.end|arsoppgjor|year.end|cloture|encerramento|cierre|ruckstellung|avsetjing|avsetting|accrual|periodiser|contrato.*trabalh|contrato.*trabaj|arbeidskontrakt|employment.contract|arbeitsvertrag|contrat.de.travail|offer.letter|lettre.d.offre|tilbudsbrev|onboarding|pdf.anexo|pdf.ci.joint|vedlagt.pdf|attached.pdf', p):
            return "salary"
    if re.search(r'inv-\d+', p):
        return "supplier_invoice"
    if re.search(r'reiseregning|travel.expense|despesa.de.viagem|reisekostenabrechnung|note.de.frais|gastos.de.viaje|nota.*gastos.*viaje|nota.*viaje|informe.*gastos|dietas|viaticos|deplacement|dienstreise', p):
        return "travel_expense"
    if re.search(r'kontaktperson|contact.person|persona.de.contacto|pessoa.de.contato|ansprechpartner', p):
        return "contact_person"
    if re.search(r'reklamert|reclamou|reclamado|complained|reclame', p):
        return "credit_note"
    if re.search(r'betaling.*returnert|payment.*returned|pago.*devuelto|pagamento.*dev|zahlung.*zuruck|paiement.*retourne', p):
        return "reverse_payment"
    if re.search(r'fastpris|fixed.price|prix.forfaitaire|precio.fijo|preco.fixo|festpreis', p):
        return "project_fixed"
    if re.search(r'rekneskapsdimensjon|regnskapsdimensjon|accounting.dimension|dimension.compt|buchhaltungs.imension|dimensao', p):
        return "acct_dimension"
    if re.search(r'registrer?\s+\d+\s*(tim|hour|hora|stund|heur)|registe\s+\d+|log\s+\d+\s+hour|enregistrez\s+\d+\s+heur', p):
        return "timesheet"
    if re.search(r'utestaende|outstanding|pendiente|pendente|offene.*rechnung|impayee|utestaande|paiement.*facture|facture.*impayee', p):
        return "payment"
    if re.search(r'tre.*avdeling|three.*department|tres.*departam|tres.*departam|drei.*abteilung|trois.*departement', p):
        return "departments"
    if re.search(r'opprett.*prosjekt|create.*project|crie.*projeto|crea.*proyecto|erstellen.*projekt|creez.*projet', p):
        return "project"
    if re.search(r'opprett.*ordre|create.*order|crie.*commande|crie.*encomenda|cria.*pedido|erstellen.*auftrag|creez.*commande', p):
        return "order"
    if re.search(r'tre.*produktlinj|three.*product.line|tres.*linha|tres.*linea|drei.*produkt|trois.*ligne|com tres.*produto|con tres.*producto|mit drei.*produkt|avec trois.*produit|with three.*product', p):
        return "invoice_multi"
    if re.search(r'opprett.*send.*faktura|create.*send.*invoice|crie.*envie|crea.*env.*factura|erstellen.*senden.*rechnung|creez.*envoyez', p):
        return "invoice_send"
    if re.search(r'ny.*ansatt|opprett.*ansatt|new.*employee|novo.*funcion|nuevo.*empleado|neuen.*mitarbeiter|nouvel.*employe|ny.*tilsett|nouveau.*employe|nova.*funcion|crie.*empregado|crear.*empleado|erstellen.*mitarbeiter', p):
        return "employee"
    if re.search(r'registr.*leverand|regist.*liefer|regist.*fornecedor|regist.*proveedor|enregistr.*fournisseur', p):
        return "supplier"
    if re.search(r'opprett.*produkt|create.*product|cr[ée][ez].*produi[ts]|crie.*produto|crea.*producto|erstellen.*produkt|enregistr.*produit', p):
        return "product"
    if re.search(r'opprett.*kunde|create.*customer|crie.*cliente|crea.*cliente|erstellen.*kund|creez.*client|enregistr.*client', p):
        return "customer"
    return "unknown"


COMPLEX_TASKS = {"supplier_invoice_pdf", "year_end", "bank_recon", "ledger_audit",
                 "fx_invoice", "reminder_fee", "employee_pdf", "project_lifecycle",
                 "cost_analysis", "month_end", "timesheet", "salary",
                 "project_lifecycle", "order"}


# ═══════════════════════════════════════════════
# MAIN ENDPOINT — pure agent, no handlers
# ═══════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "version": "v51-agent", "model": CLAUDE_MODEL}


@app.post("/solve")
@app.post("/")
async def solve(request: Request):
    start_time = time.time()
    try:
        body = await request.json()
        return await _solve_inner(body, start_time)
    except Exception as e:
        log.error(f"Top-level error after {time.time()-start_time:.1f}s: {e}")
        log.error(traceback.format_exc())
        return JSONResponse({"status": "completed"})


async def _solve_inner(body, start_time):
    prompt = body["prompt"]
    files = body.get("files", [])
    creds = body["tripletex_credentials"]

    log.info("=" * 60)
    log.info(f"PROMPT: {prompt[:200]}")
    log.info(f"FILES: {[f['filename'] for f in files]}")

    task_type = detect_task_type(prompt)
    log.info(f"Detected task type: {task_type}")

    max_iterations = 20 if task_type in COMPLEX_TASKS else 12

    # Setup session
    base_url = creds["base_url"].rstrip("/")
    adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.auth = ("0", creds["session_token"])
    session.headers.update({"Content-Type": "application/json", "Accept": "application/json"})

    # Get company ID
    company_id = None
    try:
        resp = session.get(f"{base_url}/token/session/>whoAmI", timeout=10)
        if resp.status_code == 200:
            company_id = resp.json().get("value", {}).get("company", {}).get("id")
            log.info(f"Company: {company_id}")
    except Exception as e:
        log.warning(f"Token check: {e}")

    # Pre-setup bank account
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
    except Exception:
        pass

    # Pre-fetch sandbox state (GET is free!)
    ctx = prefetch_context(session, base_url, task_type)
    if company_id:
        ctx["_company_id"] = company_id
    ctx_text = format_prefetched_context(ctx)

    # Process files
    text_parts, image_blocks = process_files(files)
    file_text = "\n---\n".join(text_parts) if text_parts else ""

    # Build user message
    user_text = f"Complete this accounting task:\n\n{prompt}"
    if file_text:
        user_text += f"\n\nAttached files:\n{file_text}"
    user_text += f"\n\n═══ PRE-FETCHED SANDBOX DATA (use these IDs!) ═══\n{ctx_text}"

    system_prompt = build_system_prompt(company_id, task_type)
    trace = []
    deadline = start_time + 250

    # Build message content
    _VALID_IMG_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
    content = []
    for img in image_blocks:
        if "inline_data" in img:
            mt = img["inline_data"].get("mime_type", "")
            if mt not in _VALID_IMG_TYPES:
                log.warning(f"Skipping image with invalid media_type: {mt!r}")
                continue
            img_data = img["inline_data"].get("data", "")
            if not img_data:
                log.warning(f"Skipping image with empty data (type={mt})")
                continue
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mt,
                    "data": img_data,
                },
            })
    content.append({"type": "text", "text": user_text})

    messages = [{"role": "user", "content": content}]

    # ═══ AGENT LOOP ═══
    for iteration in range(max_iterations):
        if time.time() > deadline:
            log.warning(f"Deadline at iteration {iteration}")
            break
        remaining = deadline - time.time()
        if remaining < 15:
            log.warning(f"Only {remaining:.0f}s left, stopping")
            break

        # Trim: keep first message + last 14 exchanges
        if len(messages) > 15:
            messages = [messages[0]] + messages[-14:]

        response = claude_generate(system_prompt, messages, CLAUDE_TOOLS, 2048)
        if not response:
            log.error("Claude returned no response")
            break

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b.text for b in response.content if hasattr(b, 'text')]
        if text_blocks:
            log.info(f"Claude text: {' '.join(text_blocks)[:500]}")

        if not tool_uses:
            log.info(f"Agent finished at iteration {iteration} (stop_reason={response.stop_reason})")
            break

        messages.append({"role": "assistant", "content": response.content})

        # Execute tools (parallel if multiple)
        tool_results = []
        if len(tool_uses) > 1:
            def run_tool(tu):
                log.info(f"Tool: {tu.name}({json.dumps(tu.input, ensure_ascii=False)[:800]})")
                result = execute_tool(tu.name, tu.input, session, base_url, trace, ctx)
                result_str = str(result)
                remaining_time = deadline - time.time()
                remaining_iters = max_iterations - iteration - 1
                if remaining_time < 60 or remaining_iters <= 3:
                    result_str += f"\n[URGENT: {remaining_time:.0f}s and {remaining_iters} iterations left. Finish NOW.]"
                return {"type": "tool_result", "tool_use_id": tu.id, "content": result_str}
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                tool_results = list(ex.map(run_tool, tool_uses))
        else:
            for tu in tool_uses:
                log.info(f"Tool: {tu.name}({json.dumps(tu.input, ensure_ascii=False)[:800]})")
                result = execute_tool(tu.name, tu.input, session, base_url, trace, ctx)
                result_str = str(result)
                # Add urgency signal near deadline
                remaining_time = deadline - time.time()
                remaining_iters = max_iterations - iteration - 1
                if remaining_time < 60 or remaining_iters <= 3:
                    result_str += f"\n[URGENT: {remaining_time:.0f}s and {remaining_iters} iterations left. Finish NOW.]"
                tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": result_str})

        messages.append({"role": "user", "content": tool_results})

        error_count = sum(1 for t in trace if t.get("status", 200) >= 400)
        if error_count >= 12:
            log.warning(f"Too many errors ({error_count}), stopping")
            break

    elapsed = time.time() - start_time
    total_calls = len(trace)
    total_errors = sum(1 for t in trace if t.get("status", 200) >= 400)
    log.info(f"=== DONE {elapsed:.1f}s type={task_type} calls={total_calls} errors={total_errors} ===")

    _log_run(prompt, files, trace, elapsed, task_type)
    return JSONResponse({"status": "completed"})


def _prompt_fingerprint(prompt: str) -> str:
    return hashlib.md5(prompt.strip().encode()).hexdigest()[:12]


def _log_run(prompt, files, trace, elapsed, task_type="unknown"):
    try:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "v51-agent",
            "model": CLAUDE_MODEL,
            "task_type": task_type,
            "prompt_fingerprint": _prompt_fingerprint(prompt),
            "prompt": prompt[:1000],
            "files": [f["filename"] for f in files] if files and isinstance(files[0], dict) else [],
            "api_calls": len(trace),
            "api_errors": sum(1 for t in trace if t.get("status", 200) >= 400),
            "elapsed_seconds": round(elapsed, 1),
            "trace": trace,
        }
        entry_json = json.dumps(entry, ensure_ascii=False)
        with open(LOG_PATH, "a") as f:
            f.write(entry_json + "\n")
        _gcs_append_log(entry_json)
    except Exception:
        pass


# ═══════════════════════════════════════════════
# DASHBOARD & API ENDPOINTS (unchanged — reads from GCS)
# ═══════════════════════════════════════════════

@app.get("/logs")
def get_logs():
    gcs_entries = _gcs_read_logs()
    if gcs_entries:
        return {"runs": gcs_entries[-50:], "count": len(gcs_entries), "source": "gcs"}
    if not LOG_PATH.exists():
        return {"runs": [], "count": 0}
    lines = LOG_PATH.read_text().strip().split("\n")
    entries = [json.loads(l) for l in lines if l.strip()]
    return {"runs": entries[-50:], "count": len(entries), "source": "local"}


@app.get("/taskmap")
def get_taskmap():
    entries = _gcs_read_logs()
    if not entries and LOG_PATH.exists():
        lines = LOG_PATH.read_text().strip().split("\n")
        entries = [json.loads(l) for l in lines if l.strip()]
    tasks = {}
    for e in entries:
        fp = e.get("prompt_fingerprint", "unknown")
        if fp not in tasks:
            tasks[fp] = {"fingerprint": fp, "task_type": e.get("task_type", "unknown"),
                         "prompt_snippet": e.get("prompt", "")[:200], "runs": 0,
                         "best_calls": 999, "worst_calls": 0, "best_errors": 999,
                         "latest_timestamp": "", "history": []}
        t = tasks[fp]
        t["runs"] += 1
        calls = e.get("api_calls", 0)
        errors = e.get("api_errors", 0)
        t["best_calls"] = min(t["best_calls"], calls)
        t["worst_calls"] = max(t["worst_calls"], calls)
        t["best_errors"] = min(t["best_errors"], errors)
        t["latest_timestamp"] = e.get("timestamp", "")
        t["history"].append({"timestamp": e.get("timestamp", ""), "version": e.get("version", "?"),
                             "calls": calls, "errors": errors, "elapsed": e.get("elapsed_seconds", 0)})
    return {"unique_tasks": len(tasks), "tasks": sorted(tasks.values(), key=lambda x: x["task_type"])}


@app.post("/scores")
async def post_scores(request: Request):
    try:
        body = await request.json()
        scores = body.get("scores", {})
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(),
                 "scores": {str(k): v for k, v in scores.items()},
                 "total": sum(scores.values())}
        entry_json = json.dumps(entry, ensure_ascii=False)
        try:
            client = _gcs_client()
            if client:
                bucket = client.bucket(GCS_BUCKET)
                blob = bucket.blob("scores.jsonl")
                existing = blob.download_as_text() if blob.exists() else ""
                blob.upload_from_string(existing + entry_json + "\n", content_type="text/plain")
        except Exception as e:
            log.warning(f"GCS score write failed: {e}")
        return {"status": "saved", "total": sum(scores.values())}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/scores")
def get_scores():
    try:
        client = _gcs_client()
        if not client:
            return {"history": []}
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob("scores.jsonl")
        if not blob.exists():
            return {"history": []}
        lines = blob.download_as_text().strip().split("\n")
        return {"history": [json.loads(l) for l in lines if l.strip()]}
    except Exception:
        return {"history": []}


@app.post("/tag")
async def tag_task(request: Request):
    try:
        body = await request.json()
        client = _gcs_client()
        if not client:
            return JSONResponse({"error": "GCS not available"}, status_code=500)
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob("task_tags.json")
        tags = json.loads(blob.download_as_text()) if blob.exists() else {}
        if "fingerprint" in body:
            tags[body["fingerprint"]] = body["task_number"]
        elif "match" in body:
            entries = _gcs_read_logs()
            matched = 0
            for e in entries:
                if body["match"].lower() in e.get("prompt", "").lower():
                    tags[e["prompt_fingerprint"]] = body["task_number"]
                    matched += 1
            if matched == 0:
                return {"status": "no_match"}
        blob.upload_from_string(json.dumps(tags), content_type="application/json")
        return {"status": "tagged", "tags": tags}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


def _get_task_tags() -> dict:
    try:
        client = _gcs_client()
        if not client:
            return {}
        blob = client.bucket(GCS_BUCKET).blob("task_tags.json")
        return json.loads(blob.download_as_text()) if blob.exists() else {}
    except Exception:
        return {}


def _get_task_knowledge() -> dict:
    try:
        client = _gcs_client()
        if not client:
            return {}
        blob = client.bucket(GCS_BUCKET).blob("task_knowledge.json")
        return json.loads(blob.download_as_text()) if blob.exists() else {}
    except Exception:
        return {}


def _save_task_knowledge(knowledge: dict):
    try:
        client = _gcs_client()
        if not client:
            return
        blob = client.bucket(GCS_BUCKET).blob("task_knowledge.json")
        blob.upload_from_string(json.dumps(knowledge, ensure_ascii=False, indent=2), content_type="application/json")
    except Exception as e:
        log.warning(f"GCS knowledge write failed: {e}")


@app.post("/knowledge")
async def post_knowledge(request: Request):
    try:
        body = await request.json()
        tn = str(body["task_number"])
        knowledge = _get_task_knowledge()
        if tn not in knowledge:
            knowledge[tn] = {"type": "", "experiments": [], "best_score": 0, "checks": "", "notes": ""}
        k = knowledge[tn]
        for field in ("type", "checks", "notes"):
            if field in body:
                k[field] = body[field]
        if "score" in body:
            k["best_score"] = max(k.get("best_score", 0), body["score"])
            k["latest_score"] = body["score"]
        if "experiment" in body:
            k["experiments"].append(body["experiment"])
        _save_task_knowledge(knowledge)
        return {"status": "saved", "task": tn, "knowledge": k}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/knowledge")
def get_knowledge():
    return _get_task_knowledge()


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    score_history = get_scores().get("history", [])
    latest_scores = score_history[-1] if score_history else {"scores": {}, "total": 0, "timestamp": "---"}
    prev_scores = score_history[-2] if len(score_history) >= 2 else None
    tags = _get_task_tags()
    reverse_tags = {v: k for k, v in tags.items()}
    all_logs = _gcs_read_logs()
    knowledge = _get_task_knowledge()

    cards_html = ""
    total = 0
    for i in range(1, 31):
        s = float(latest_scores["scores"].get(str(i), 0))
        total += s
        delta = ""
        if prev_scores:
            prev = float(prev_scores["scores"].get(str(i), 0))
            diff = s - prev
            if diff > 0.01:
                delta = f'<span style="color:#22c55e;font-size:0.75em">+{diff:.2f}</span>'
            elif diff < -0.01:
                delta = f'<span style="color:#ef4444;font-size:0.75em">{diff:.2f}</span>'
        color = "#ef4444" if s == 0 else "#f97316" if s < 2 else "#eab308" if s < 3.5 else "#22c55e" if s < 5 else "#06b6d4"
        k = knowledge.get(str(i), {})
        type_name = k.get("type", "")
        cards_html += f'<div style="background:#1e293b;border-radius:8px;padding:10px;text-align:center;border-top:3px solid {color}"><div style="font-size:0.7em;color:#94a3b8">Task {i:02d}</div><div style="font-size:1.6em;font-weight:700;color:{color}">{s:.2f}</div>{delta}<div style="font-size:0.6em;color:#64748b">{type_name}</div></div>'

    html = f'''<!DOCTYPE html><html><head><meta charset="utf-8"><title>Tripletex v50</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="60">
<style>*{{margin:0;padding:0;box-sizing:border-box}}body{{background:#0f172a;color:#e2e8f0;font-family:system-ui;padding:16px;max-width:1200px;margin:0 auto}}</style></head><body>
<h1 style="text-align:center;margin-bottom:8px">Tripletex Agent v50</h1>
<div style="text-align:center;font-size:2.8em;font-weight:800;color:#06b6d4;margin:8px 0">{total:.2f}</div>
<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin:16px 0">{cards_html}</div>
<p style="text-align:center;color:#64748b;font-size:0.8em">{latest_scores.get("timestamp","---")[:19]}</p>
</body></html>'''
    return HTMLResponse(html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
