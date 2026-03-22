"""
Tripletex AI Accounting Agent — NM i AI 2026
v23-fixes: cost_analysis activityType fix, project_lifecycle GET project activities,
              salary employee fallback, credit_note better handling.
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

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("agent")

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# Claude API config
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")
USE_CLAUDE = os.environ.get("USE_CLAUDE", "true").lower() == "true"

# Vertex AI config (uses GCP service account — no API key needed on Cloud Run)
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "ainm26osl-745")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = "gemini-2.0-flash-001"  # Stable Vertex AI model name

_gcp_token: str | None = None
_gcp_token_expiry: float = 0.0


def _get_gcp_token() -> str | None:
    """Get GCP access token from metadata server (available on Cloud Run)."""
    global _gcp_token, _gcp_token_expiry
    now = time.time()
    if _gcp_token and now < _gcp_token_expiry - 60:
        return _gcp_token
    try:
        resp = requests.get(
            "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
            headers={"Metadata-Flavor": "Google"},
            timeout=3,
        )
        if resp.status_code == 200:
            data = resp.json()
            _gcp_token = data["access_token"]
            _gcp_token_expiry = now + data.get("expires_in", 3600)
            log.info("Got GCP token via metadata server (Vertex AI available)")
            return _gcp_token
    except Exception:
        pass  # Not on GCP / metadata not available
    return None
GEMINI_FALLBACK_MODEL = "gemini-2.0-flash"  # always-free fallback if primary quota exhausted

LOG_PATH = Path(os.environ.get("SUBMISSION_LOG", "/tmp/tripletex_submissions.jsonl"))
import hashlib

# ── Persistent GCS logging ──
GCS_BUCKET = "ainm26osl-745-tripletex-logs"
GCS_LOG_BLOB = "submissions.jsonl"

def _gcs_client():
    try:
        from google.cloud import storage
        return storage.Client()
    except Exception:
        return None

def _gcs_append_log(entry_json: str):
    """Append a log entry to GCS (persistent across deploys)."""
    try:
        client = _gcs_client()
        if not client:
            return
        bucket = client.bucket(GCS_BUCKET)
        # Create bucket if not exists
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
    """Read all log entries from GCS."""
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

GEMINI_TOOLS = [{"function_declarations": [
    {
        "name": "tripletex_get",
        "description": "GET request to Tripletex API. Returns JSON response.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "endpoint": {"type": "STRING", "description": "API path, e.g. /employee or /department"},
                "query_params": {"type": "STRING", "description": "Query string, e.g. 'fields=id,name&count=10'. Empty string if none."},
            },
            "required": ["endpoint"],
        },
    },
    {
        "name": "tripletex_post",
        "description": "POST request to create a resource. Returns JSON response with created object.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "endpoint": {"type": "STRING", "description": "API path, e.g. /employee"},
                "body": {"type": "OBJECT", "description": "JSON body to send with all required fields"},
                "query_params": {"type": "STRING", "description": "Query string. Empty string if none."},
            },
            "required": ["endpoint", "body"],
        },
    },
    {
        "name": "tripletex_put",
        "description": "PUT request to update a resource or trigger an action. For actions like /:payment, use query_params for parameters (NOT body).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "endpoint": {"type": "STRING", "description": "API path, e.g. /employee/123 or /invoice/456/:payment"},
                "body": {"type": "OBJECT", "description": "JSON body (for updates). Empty object {} for actions."},
                "query_params": {"type": "STRING", "description": "Query string. For actions like /:payment, put params here."},
            },
            "required": ["endpoint"],
        },
    },
    {
        "name": "tripletex_delete",
        "description": "DELETE request to remove a resource.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "endpoint": {"type": "STRING", "description": "API path with ID, e.g. /travelExpense/789"},
            },
            "required": ["endpoint"],
        },
    },
]}]


# ═══════════════════════════════════════════════
# CLAUDE TOOLS — same capabilities, Anthropic format
# ═══════════════════════════════════════════════

CLAUDE_TOOLS = [
    {
        "name": "tripletex_get",
        "description": "GET request to Tripletex API. Returns JSON response.",
        "input_schema": {
            "type": "object",
            "properties": {
                "endpoint": {"type": "string", "description": "API path, e.g. /employee or /department"},
                "query_params": {"type": "string", "description": "Query string, e.g. 'fields=id,name&count=10'. Empty string if none."},
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
                "body": {"type": "object", "description": "JSON body to send with all required fields"},
                "query_params": {"type": "string", "description": "Query string. Empty string if none."},
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
                "body": {"type": "object", "description": "JSON body (for updates). Empty object {} for actions."},
                "query_params": {"type": "string", "description": "Query string. For actions like /:payment, put params here."},
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


def claude_generate(system_prompt, messages, tools=None, max_tokens=1536):
    """Call Claude API with tool use support. Returns parsed response."""
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

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
            response = client.messages.create(**kwargs)
            return response
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
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

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
            log.warning(f"Claude extract 429, waiting {wait}s")
            time.sleep(wait)
        except Exception as e:
            log.error(f"Claude extract error: {e}")
            return ""
    return ""


def gemini_generate(model, system_prompt, messages, tools=None, max_tokens=1536):
    """Call Gemini/Vertex AI API. Tries Vertex AI first (service account), falls back to API key."""
    body = {
        "contents": messages,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.0,
        },
    }
    if system_prompt:
        body["system_instruction"] = {"parts": [{"text": system_prompt}]}
    if tools:
        body["tools"] = tools
        body["tool_config"] = {"function_calling_config": {"mode": "AUTO"}}

    # Try Vertex AI first (uses GCP service account — no API key needed)
    token = _get_gcp_token()
    if token:
        url = (f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/v1/projects/{VERTEX_PROJECT}"
               f"/locations/{VERTEX_LOCATION}/publishers/google/models/{VERTEX_MODEL}:generateContent")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        for attempt in range(3):
            try:
                resp = requests.post(url, json=body, headers=headers, timeout=60)
                if resp.status_code == 429:
                    wait = min(30, 2 ** attempt + 2)
                    log.warning(f"Vertex AI 429, waiting {wait}s (attempt {attempt+1})")
                    time.sleep(wait)
                    continue
                if resp.status_code == 401:
                    global _gcp_token
                    _gcp_token = None  # Force token refresh on next call
                    log.warning("Vertex AI 401 — token expired, clearing cache")
                    break
                if resp.status_code != 200:
                    log.error(f"Vertex AI {resp.status_code}: {resp.text[:300]}")
                    break
                return resp.json()
            except Exception as e:
                log.error(f"Vertex AI request error: {e}")
                break

    # Fallback: direct Gemini API with API key
    if not GEMINI_API_KEY:
        log.error("No Gemini API key and Vertex AI unavailable")
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    for attempt in range(3):
        try:
            resp = requests.post(url, json=body, timeout=60)
            if resp.status_code == 429:
                wait = min(30, 2 ** attempt + 2)
                log.warning(f"Gemini 429 rate limit, waiting {wait}s (attempt {attempt+1})")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                log.error(f"Gemini API {resp.status_code}: {resp.text[:500]}")
                return None
            return resp.json()
        except Exception as e:
            log.error(f"Gemini request error: {e}")
            return None
    return None


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
            if "phoneNumber" not in body:
                body["phoneNumber"] = ""
            if "language" not in body:
                body["language"] = "NO"
            if "invoiceEmail" not in body and body.get("email"):
                body["invoiceEmail"] = body["email"]
            # Ensure supplierNumber is set (scoring checks this!)
            if "supplierNumber" not in body:
                body["supplierNumber"] = 0  # auto-generate
            # Ensure accountManager is not null
            if "accountManager" not in body:
                if ctx and ctx.get("employees"):
                    body["accountManager"] = {"id": ctx["employees"][0]["id"]}
        log.info(f"Redirected POST /supplier → POST /customer with isSupplier:true")

    # Intercept: POST /customer — ensure phoneNumber + language for regular customers AND suppliers
    if name == "tripletex_post" and endpoint.strip("/") == "customer" and body:
        if "phoneNumber" not in body:
            body["phoneNumber"] = ""
        if "language" not in body:
            body["language"] = "NO"
        if "invoiceEmail" not in body and body.get("email"):
            body["invoiceEmail"] = body["email"]
        # Ensure email is set even for suppliers created via /customer
        if not body.get("email") and body.get("isSupplier"):
            body["email"] = f"faktura@{body.get('name', 'supplier').lower().replace(' ', '').replace('gmbh', '').replace('as', '').replace('ltd', '')}.no"
            body["invoiceEmail"] = body["email"]
            log.info(f"Auto-generated supplier email: {body['email']}")

    # Intercept: POST /incomingInvoice — fix field names to match actual API spec
    # Claude often uses wrong field names (unitPriceExcludingVatCurrency, vatType:{id:X})
    # Correct fields: accountId (flat int), amountInclVat, vatTypeId (flat int)
    if name == "tripletex_post" and "incomingInvoice" in endpoint and body:
        order_lines = body.get("orderLines", [])
        for ol in order_lines:
            # Fix vatType: {id: X} → vatTypeId: X
            if "vatType" in ol and isinstance(ol["vatType"], dict):
                ol["vatTypeId"] = ol.pop("vatType")["id"]
            # Fix account: {id: X} → accountId: X
            if "account" in ol and isinstance(ol["account"], dict):
                ol["accountId"] = ol.pop("account")["id"]
            # Fix unitPriceExcludingVatCurrency / unitPrice → amountInclVat
            excl_vat = ol.pop("unitPriceExcludingVatCurrency", None) or ol.pop("unitPriceExcludingVat", None) or ol.pop("unitPrice", None)
            if excl_vat and "amountInclVat" not in ol:
                # Calculate incl VAT based on vatTypeId
                vat_id = ol.get("vatTypeId", 1)
                vat_pct = {1: 0.25, 11: 0.15, 0: 0}.get(vat_id, 0.25)
                ol["amountInclVat"] = round(float(excl_vat) * (1 + vat_pct), 2)
                log.info(f"Fixed incomingInvoice orderLine: {excl_vat} exVat → {ol['amountInclVat']} inclVat")
            # Ensure accountId from context if missing
            if "accountId" not in ol and ctx:
                # Try to find expense account from prompt context
                accts = ctx.get("ledger_accounts", {})
                # Default to 6500 (office supplies) if nothing specified
                for default_acct in ("6500", "6300", "6800"):
                    if default_acct in accts:
                        ol["accountId"] = accts[default_acct]
                        log.info(f"Auto-assigned accountId from {default_acct}")
                        break
            # Remove fields that don't exist in ExternalWrite API
            for bad_field in ("unitCostPrice", "grossAmount"):
                ol.pop(bad_field, None)
            # Ensure row is set
            if "row" not in ol:
                ol["row"] = order_lines.index(ol) + 1
            # Ensure externalId is set (API requires it)
            if "externalId" not in ol:
                ol["externalId"] = str(order_lines.index(ol) + 1)
        log.info(f"Fixed incomingInvoice body: {json.dumps(body, ensure_ascii=False)[:500]}")

    # Intercept: PUT /employee — ALWAYS ensure dateOfBirth is present and correct
    if name == "tripletex_put" and "/employee/" in endpoint and ":/" not in endpoint and body:
        emp_id_str = endpoint.rstrip("/").split("/")[-1]
        try:
            emp_id = int(emp_id_str)
            found = False
            if ctx:
                for e in ctx.get("employees", []):
                    if e.get("id") == emp_id:
                        found = True
                        real_dob = e.get("dateOfBirth")
                        if real_dob and real_dob != "1985-01-01":
                            if not body.get("dateOfBirth") or body.get("dateOfBirth") == "1985-01-01":
                                body["dateOfBirth"] = real_dob
                                log.info(f"Injected dateOfBirth={real_dob} for employee {emp_id}")
                        elif not body.get("dateOfBirth"):
                            body["dateOfBirth"] = "1990-05-15"
                        if not body.get("version") and e.get("version"):
                            body["version"] = e["version"]
                        break
            # FALLBACK: employee not in prefetched context (newly created during this run)
            if not found and not body.get("dateOfBirth"):
                body["dateOfBirth"] = "1990-05-15"
                log.info(f"Fallback dateOfBirth=1990-05-15 for new employee {emp_id}")
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

    # Intercept: Block manual salary voucher if /salary/transaction already succeeded
    if name == "tripletex_post" and endpoint.strip("/") == "ledger/voucher" and body and ctx:
        if ctx.get("_salary_transaction_done"):
            desc = str(body.get("description", "")).lower()
            postings = body.get("postings", [])
            # Check if this looks like a salary voucher (has 5000 account or mentions lønn/salary)
            is_salary_voucher = False
            acct_5000_id = ctx.get("ledger_accounts", {}).get("5000")
            for p in postings:
                acct_id = p.get("account", {}).get("id")
                if acct_id and acct_5000_id and acct_id == acct_5000_id:
                    is_salary_voucher = True
            if "lønn" in desc or "salary" in desc or "gehalt" in desc or "nómina" in desc or "salário" in desc or is_salary_voucher:
                log.info("BLOCKED duplicate salary voucher — /salary/transaction already succeeded!")
                trace.append({"tool": name, "endpoint": endpoint, "status": 200, "error": None})
                return json.dumps({"status": "already_done", "message": "Salary was already processed via /salary/transaction. No manual voucher needed. Task is COMPLETE — stop now."})

    # Intercept: POST /ledger/voucher — fix manual VAT splitting
    # Claude sometimes creates 3 postings with vatType:0 instead of 2 postings with vatType:1
    # This auto-corrects: merge the expense+VAT postings into one with vatType:1
    if name == "tripletex_post" and endpoint.strip("/") == "ledger/voucher" and body and ctx:
        postings = body.get("postings", [])
        accts = ctx.get("ledger_accounts", {})
        acct_2710_id = accts.get("2710")
        if len(postings) == 3 and acct_2710_id:
            # Check if one posting is to 2710 (MVA) with vatType:0 — this is the manual split pattern
            vat_posting_idx = None
            expense_posting_idx = None
            supplier_posting_idx = None
            for i, p in enumerate(postings):
                acct_id = p.get("account", {}).get("id")
                vat_type_id = p.get("vatType", {}).get("id")
                if acct_id == acct_2710_id and vat_type_id == 0:
                    vat_posting_idx = i
                elif acct_id == accts.get("2400"):
                    supplier_posting_idx = i
                else:
                    expense_posting_idx = i

            if vat_posting_idx is not None and expense_posting_idx is not None and supplier_posting_idx is not None:
                # Merge: expense gets gross amount (net + VAT), vatType becomes 1 (auto-split)
                exp = postings[expense_posting_idx]
                vat = postings[vat_posting_idx]
                sup = postings[supplier_posting_idx]
                merged_gross = float(exp.get("amountGross", 0)) + float(vat.get("amountGross", 0))
                exp["amountGross"] = merged_gross
                exp["amountGrossCurrency"] = merged_gross
                exp["vatType"] = {"id": 1}  # Auto-split inbound VAT 25%
                # Update supplier posting to match
                sup["amountGross"] = -merged_gross
                sup["amountGrossCurrency"] = -merged_gross
                # Remove the manual 2710 posting
                body["postings"] = [exp, sup]
                # Fix row numbers
                body["postings"][0]["row"] = 1
                body["postings"][1]["row"] = 2
                log.info(f"Fixed voucher: merged 3 manual-VAT postings → 2 auto-VAT postings (gross={merged_gross})")

    # ── UNIVERSAL INTERCEPTORS (fix common Claude mistakes) ──

    # Block nonexistent endpoints Claude hallucinates
    fake_endpoints = ("/voucher", "/journalEntry", "/generalLedgerEntry", "/ledger/entry",
                      "/generalLedgerVoucher", "/purchaseInvoice", "/timeSheet",
                      "/projectCost", "/projectHour", "/ledger/paymentTypeIn")
    if endpoint.strip("/") in [e.strip("/") for e in fake_endpoints]:
        log.info(f"BLOCKED hallucinated endpoint {endpoint}")
        trace.append({"tool": name, "endpoint": endpoint, "status": 404, "error": "not found"})
        return json.dumps({"error": f"404 Not Found. Use POST /ledger/voucher with 'postings' array for vouchers.", "status": 404})

    # Fix quantity → count on orderline
    if name == "tripletex_post" and "order/orderline" in endpoint and body:
        if "quantity" in body:
            body["count"] = body.pop("quantity")
            log.info("Fixed orderline: quantity → count")

    # Fix address → postalAddress on customer
    if name == "tripletex_post" and endpoint.strip("/") == "customer" and body:
        if "address" in body and "postalAddress" not in body:
            body["postalAddress"] = body.pop("address")
            log.info("Fixed customer: address → postalAddress")

    # Fix dueDate → invoiceDueDate on invoice
    if name == "tripletex_post" and endpoint.strip("/") == "invoice" and body:
        if "dueDate" in body and "invoiceDueDate" not in body:
            body["invoiceDueDate"] = body.pop("dueDate")
        if "lines" in body and "orders" not in body:
            body["orders"] = body.pop("lines")

    # Strip dateOfBirth from employment details (it belongs on /employee, not /employment/details)
    if name == "tripletex_put" and "/employee/employment/details/" in endpoint and body:
        if "dateOfBirth" in body:
            body.pop("dateOfBirth")
            log.info("Stripped dateOfBirth from employment details PUT")

    # Auto-inject customer on entitlement if missing
    if name == "tripletex_post" and "employee/entitlement" in endpoint and body and ctx:
        if not body.get("customer") or (isinstance(body.get("customer"), dict) and not body["customer"].get("id")):
            company_id = ctx.get("_company_id")
            if company_id:
                body["customer"] = {"id": company_id}
                log.info(f"Auto-injected customer id={company_id} on entitlement")

    # Fix activity: strip activityNumber (doesn't exist), ensure isGeneral
    if name == "tripletex_post" and endpoint.strip("/") == "activity" and body:
        body.pop("activityNumber", None)
        if "isGeneral" not in body:
            body["isGeneral"] = True

    # Fix fields with dots → parentheses for GET requests
    if name == "tripletex_get" and params and "fields" in (params or {}):
        fields = params["fields"]
        if "." in fields:
            # account.number,account.name → account(number,name)
            import re as _re
            def fix_fields(f):
                parts = f.split(",")
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
                return ",".join(result)
            params["fields"] = fix_fields(fields)
            log.info(f"Fixed fields dots→parentheses: {fields} → {params['fields']}")

    # Auto-inject invoiceDateFrom/To on GET /invoice if missing
    if name == "tripletex_get" and endpoint.strip("/") == "invoice":
        if params is None:
            params = {}
        if "invoiceDateFrom" not in params:
            params["invoiceDateFrom"] = "2020-01-01"
            params["invoiceDateTo"] = "2027-01-01"
            log.info("Auto-injected invoice date range")
        # Strip invalid InvoiceDTO fields
        if "fields" in params:
            valid_invoice_fields = {"id", "invoiceNumber", "amount", "amountOutstanding", "customer",
                                    "invoiceDate", "invoiceDueDate", "currency", "isCreditNote",
                                    "orders", "ehfSendStatus", "version"}
            requested = [f.strip() for f in params["fields"].split(",")]
            cleaned = [f for f in requested if f in valid_invoice_fields or "(" in f]
            if len(cleaned) < len(requested):
                log.info(f"Stripped invalid invoice fields: {set(requested) - set(cleaned)}")
                params["fields"] = ",".join(cleaned) if cleaned else "id,invoiceNumber,amount,amountOutstanding,customer"

    # Fix /supplier/invoice → /supplierInvoice
    if name == "tripletex_get" and endpoint.strip("/") == "supplier/invoice":
        endpoint = "/supplierInvoice"
        log.info("Fixed endpoint: /supplier/invoice → /supplierInvoice")

    url = f"{base_url}/{endpoint.lstrip('/')}"

    try:
        if name == "tripletex_get":
            resp = session.get(url, params=params, timeout=30)
        elif name == "tripletex_post":
            resp = session.post(url, json=body, params=params, timeout=30)
            # Track newly created employees so PUT dateOfBirth works later
            if resp.status_code == 201 and endpoint.strip("/") == "employee" and ctx is not None:
                try:
                    new_emp = resp.json().get("value", {})
                    if new_emp.get("id"):
                        if "employees" not in ctx:
                            ctx["employees"] = []
                        ctx["employees"].append(new_emp)
                        log.info(f"Tracked new employee id={new_emp['id']} dateOfBirth={new_emp.get('dateOfBirth')}")
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
        # Track salary/payslip success to prevent duplicate vouchers
        if ("salary/transaction" in endpoint or "salary/paySlip" in endpoint) and status_code in (200, 201) and ctx is not None:
            ctx["_salary_transaction_done"] = True
            log.info("Salary payslip succeeded — blocking future manual salary vouchers")

    try:
        result = resp.json()
    except Exception:
        result = {"raw": resp.text[:500]}

    # Truncate large responses to save tokens
    text = json.dumps(result, ensure_ascii=False)
    if len(text) > 1200:
        if isinstance(result, dict) and "values" in result:
            result["values"] = result["values"][:3]
            result["_truncated"] = True
            text = json.dumps(result, ensure_ascii=False)
        if len(text) > 1200:
            text = text[:1200]
    return text


# ═══════════════════════════════════════════════
# PRE-FETCH CONTEXT
# ═══════════════════════════════════════════════

def prefetch_context(session: requests.Session, base_url: str, task_type: str = "unknown") -> dict:
    """Pre-fetch commonly needed data in parallel to reduce latency.

    Task-specific prefetch: only fetch what's needed for the detected task type.
    """
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

    # ── Data group definitions ──
    F_EMP = ("employees", "employee", {"fields": "id,firstName,lastName,email,dateOfBirth,userType,version,department", "count": "100"})
    F_DEPT = ("departments", "department", {"fields": "id,name,departmentNumber", "count": "10"})
    F_DIV = ("divisions", "division", {"fields": "id,name", "count": "5"})
    F_ACT = ("activities", "activity", {"fields": "id,name", "count": "20"})
    F_PT = ("payment_types", "invoice/paymentType", {"fields": "id,description", "count": "5"})
    F_PROD = ("products", "product", {"fields": "id,name,number,priceExcludingVatCurrency", "count": "100"})
    F_CUST = ("customers", "customer", {"fields": "id,name,organizationNumber", "count": "100"})
    F_SUPP = ("suppliers", "supplier", {"fields": "id,name,organizationNumber", "count": "100"})
    F_CURR = ("currencies", "currency", {"fields": "id,code", "count": "20"})
    F_CC = ("cost_categories", "travelExpense/costCategory", {"fields": "id,description", "count": "50"})
    F_TPT = ("travel_payment_types", "travelExpense/paymentType", {"fields": "id,description", "count": "5"})
    F_RC = ("rate_categories", "travelExpense/rateCategory", {"fields": "id,name,type,fromDate,toDate", "count": "500"})

    SALARY_F = [("salary_%s" % n, "salary/type", {"number": str(n), "fields": "id,number,name"}) for n in (2000, 2001, 2002, 2005)]
    COMMON_ACCTS = [("acct_%s" % n, "ledger/account", {"number": str(n), "fields": "id,number,name"}) for n in (1920, 2400, 2710, 3000, 1500)]
    EXPENSE_ACCTS = [("acct_%s" % n, "ledger/account", {"number": str(n), "fields": "id,number,name"}) for n in (5000, 2780, 7100, 7140, 7350, 6300, 6340, 6500, 6700, 6800, 6900, 7300, 7400, 7700, 2600)]
    YE_ACCTS = [("acct_%s" % n, "ledger/account", {"number": str(n), "fields": "id,number,name"}) for n in (1200, 1209, 1210, 1230, 1240, 1250, 1700, 1710, 1950, 2000, 2050, 2800, 2900, 2920, 3400, 4000, 4300, 6010, 6020, 6100, 6200, 6540, 7000, 7500, 8060, 8160, 8300, 8700, 8900)]

    # ── Task-specific prefetch ──
    W_EMP = {"employee", "employee_pdf", "salary", "travel_expense", "delete_travel",
             "month_end", "year_end", "bank_recon", "ledger_audit", "project_lifecycle",
             "cost_analysis", "timesheet", "acct_dimension", "unknown"}
    W_PROD = {"product", "invoice_send", "invoice_multi", "fx_invoice", "reminder_fee",
              "payment", "credit_note", "order", "project_lifecycle", "unknown"}
    W_CUST = {"customer", "contact_person", "invoice_send", "invoice_multi", "fx_invoice",
              "reminder_fee", "payment", "credit_note", "order", "project_lifecycle", "unknown"}
    W_SUPP = {"supplier", "supplier_invoice", "supplier_invoice_pdf", "project_lifecycle", "unknown"}
    W_SAL = {"salary", "employee_pdf", "month_end", "unknown"}
    W_TRAV = {"travel_expense", "delete_travel", "unknown"}
    W_ACCT = {"supplier_invoice", "supplier_invoice_pdf", "receipt_voucher", "invoice_send",
              "invoice_multi", "fx_invoice", "reminder_fee", "salary", "payment",
              "credit_note", "reverse_payment", "cost_analysis", "bank_recon", "ledger_audit", "project_lifecycle", "unknown"}
    W_YE = {"year_end", "month_end", "cost_analysis", "ledger_audit", "bank_recon", "unknown"}
    W_MINIMAL = {"departments", "project", "project_fixed"}

    fetches = []
    if task_type in W_MINIMAL:
        fetches.append(F_DEPT)
        if task_type in ("project", "project_fixed"):
            fetches.extend([F_EMP, F_ACT])
        log.info(f"Minimal prefetch for {task_type}: {len(fetches)} calls")
    else:
        fetches.extend([F_DEPT, F_DIV])
        if task_type in W_EMP: fetches.append(F_EMP)
        if task_type in W_PROD: fetches.append(F_PROD)
        if task_type in W_CUST: fetches.append(F_CUST)
        if task_type in W_SUPP: fetches.append(F_SUPP)
        if task_type in W_SAL: fetches.extend(SALARY_F)
        if task_type in W_TRAV: fetches.extend([F_RC, F_CC, F_TPT])
        if task_type in W_CUST or task_type in W_PROD: fetches.extend([F_PT, F_CURR])
        if task_type in W_ACCT: fetches.extend(COMMON_ACCTS + EXPENSE_ACCTS)
        if task_type in W_YE: fetches.extend(YE_ACCTS)
        if task_type in ("employee", "employee_pdf", "project_lifecycle", "unknown"): fetches.append(F_ACT)
        log.info(f"Task-specific prefetch for {task_type}: {len(fetches)} calls (was always 48+)")

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
            elif name == "currencies":
                ctx["currencies"] = {c.get("code"): c.get("id") for c in values if c.get("code")}
                log.info(f"Pre-fetched currencies: {list(ctx['currencies'].keys())}")
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

    # Post-fetch: check employments for each employee (only for employee-related tasks)
    NEEDS_EMPLOYMENTS = {"employee", "employee_pdf", "salary", "unknown"}
    if ctx.get("employees") and task_type in NEEDS_EMPLOYMENTS:
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

    if ctx.get("currencies"):
        curr_lines = [f"  {code}: id={cid}" for code, cid in sorted(ctx["currencies"].items())]
        parts.append("CURRENCY IDS (use directly for FX invoices — do NOT GET /currency):\n" + "\n".join(curr_lines))

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

def build_system_prompt(company_id: int | None, task_type: str = "unknown") -> str:
    today = date.today().isoformat()
    due = (date.today() + timedelta(days=30)).isoformat()

    header = f"""You are a Tripletex accounting API agent. Complete the given accounting task by calling the API.

TODAY: {today}  DUE: {due}  COMPANY_ID: {company_id or "unknown"}

RULES:
1. If email matches EXISTING EMPLOYEES → PUT /employee/ID (keep dateOfBirth FROM CONTEXT). Never create with modified email.
2. Use EXACT values from prompt. Dates MUST be YYYY-MM-DD.
3. EFFICIENCY: Only POST/PUT/DELETE count! GET is FREE. Minimize write calls. Use pre-fetched data.
4. On 422: read validationMessages, fix. On 403: skip, use fallback. On 409: GET fresh version, retry.
5. PUT actions (/:payment, /:createCreditNote): params in query_params, NOT body.
6. Never set "id"/"version" in POST. Prices EXCLUDING VAT unless "inkl. MVA"/"TTC"/"con IVA incluido".
7. BATCH independent calls in ONE response. Each iteration ≈ 10s. Fewer = faster.
8. PLAN FIRST: mentally plan ALL steps, group independent ones.
9. Salary: use payslip flow (POST paySlip → PUT :calculate → PUT :createPayment). STOP after createPayment. No manual voucher.
10. Per diem → /travelExpense/perDiemCompensation (NOT /cost).
11. userType: STANDARD | EXTENDED | NO_ACCESS. NEVER "ADMINISTRATOR".

LANGUAGES: NO, EN, ES, PT, NN, DE, FR. Parse dates in any language.
VAT IDs: 0=none, 1=ingoing25%, 3=outgoing25%, 5=outgoing0%(exempt), 6=outgoing0%(outside), 11=ingoing15%(food), 31=outgoing15%(food), 32=outgoing12%

═══ BETA (always 403) ═══
PUT /project/{{id}}, POST /project/orderline, DELETE /project"""

    # Task-specific recipes — only include what's needed
    recipes = _get_recipes(task_type, today, due, company_id)

    error_handling = """
═══ ERROR HANDLING ═══
422: fix field. Product dup → GET /product?number=X, use ID. 403: skip, fallback. 409: GET version, retry. Dup email → GET existing, use ID."""

    return header + "\n" + recipes + "\n" + error_handling


def _get_recipes(task_type: str, today: str, due: str, company_id) -> str:
    """Return only the recipes needed for the detected task type."""

    # Shared sub-recipes
    R_EMPLOYEE = f"""═══ EMPLOYEE ═══
Match email → PUT /employee/ID (with version). Else POST /employee.
POST /employee {{{{firstName, lastName, email, userType:"STANDARD", department:{{{{id:X}}}}, dateOfBirth, employeeNumber}}}}
PUT /employee/ID {{{{firstName, lastName, email, userType, department:{{{{id:X}}}}, dateOfBirth, version:V}}}}
Address: use "address" (NOT postalAddress)."""

    R_EMPLOYMENT = f"""── EMPLOYMENT ──
POST /division {{{{name:"Hovedkontor", startDate:"{today}", municipality:{{{{id:262}}}}, organizationNumber:"996757435"}}}}
POST /employee/employment {{{{employee:{{{{id:X}}}}, startDate:"YYYY-MM-DD", division:{{{{id:DIV}}}}, isMainEmployer:true, taxDeductionCode:"loennFraHovedarbeidsgiver"}}}}
POST /employee/employment/details {{{{employment:{{{{id:X}}}}, date:"{today}", employmentType:"ORDINARY", employmentForm:"PERMANENT", remunerationType:"MONTHLY_WAGE", workingHoursScheme:"NOT_SHIFT", percentageOfFullTimeEquivalent:100.0}}}}"""

    R_PRODUCT = f"""── PRODUCT ──
POST /product {{{{name, number:"STRING!", priceExcludingVatCurrency:X, priceIncludingVatCurrency:Y, vatType:{{{{id:Z}}}}}}}}
priceExcludingVatCurrency (NOT priceExcludingVat!). ALWAYS set vatType. priceInclVat = exVat × (1+vat/100)."""

    R_INVOICE_FLOW = f"""── INVOICE FLOW ──
Batch: [POST customer + POST product] → [POST order] → [POST orderline] → [POST invoice]
POST /order {{{{customer:{{{{id:X}}}}, deliveryDate:"{today}", orderDate:"{today}", isPrioritizeAmountsIncludingVat:false}}}}
POST /order/orderline: ALWAYS set vatType + unitPriceExcludingVatCurrency on EACH line.
POST /invoice {{{{invoiceDate:"{today}", invoiceDueDate:"{due}", orders:[{{{{id:X}}}}]}}}} query_params: sendToCustomer=false
paidAmount on /:payment = TOTAL INCLUDING VAT. GET /invoice REQUIRES invoiceDateFrom & invoiceDateTo."""

    R_VOUCHER = f"""── VOUCHER ──
POST /ledger/voucher {{{{date:"YYYY-MM-DD", description:"...", postings:[
  {{{{account:{{{{id:X}}}}, amountGross:AMT, amountGrossCurrency:AMT, date:"YYYY-MM-DD", row:1, vatType:{{{{id:0}}}}}}}},
  {{{{account:{{{{id:Y}}}}, amountGross:-AMT, amountGrossCurrency:-AMT, date:"YYYY-MM-DD", row:2, vatType:{{{{id:0}}}}}}}}
]}}}}
Postings MUST balance to 0. row+date REQUIRED. Customer accts(1500s) need customer:{{{{id:X}}}}. Supplier accts(2400) need supplier:{{{{id:X}}}}."""

    R_PROJECT_BASE = f"""── PROJECT ──
Batch: [PUT employee EXTENDED + POST customer] → [entitlement×2 + POST project]
POST /employee/entitlement {{{{entitlementId:45}}}} + {{{{entitlementId:10}}}} — batch!
POST /project {{{{name, startDate:"{today}", projectManager:{{{{id:X}}}}, customer:{{{{id:Y}}}}}}}} — no "number"!"""

    # Map task types to their recipes
    if task_type == "customer":
        return f"""{R_EMPLOYEE}
═══ CUSTOMER ═══
POST /customer {{{{name, organizationNumber, email, phoneNumber, isCustomer:true, postalAddress:{{{{addressLine1, postalCode, city}}}}}}}}
postalAddress.addressLine1 (NOT address1!). phoneNumber (NOT phoneNumberMobile)."""

    elif task_type == "supplier":
        return f"""═══ SUPPLIER ═══
POST /customer {{{{name, organizationNumber, email, phoneNumber, isCustomer:false, isSupplier:true, postalAddress:{{{{addressLine1, postalCode, city}}}}}}}}
Create via POST /customer with isSupplier:true (NOT POST /supplier!)."""

    elif task_type == "product":
        return R_PRODUCT

    elif task_type == "departments":
        return f"""═══ DEPARTMENTS ═══
POST /department {{{{name, departmentNumber:INT}}}}
departmentNumber must be unique. Use EXISTING DEPARTMENT NUMBERS +1, +2. For 3 depts: POST 3 times."""

    elif task_type == "employee":
        return R_EMPLOYEE + "\n" + R_EMPLOYMENT

    elif task_type == "employee_pdf":
        return f"""{R_EMPLOYEE}
{R_EMPLOYMENT}
═══ PDF CONTRACT ═══
Extract ALL from PDF. EVERY field scored!
1. POST /employee {{{{firstName, lastName, email:"firstname.lastname@example.org", dateOfBirth, employeeNumber, nationalIdentityNumber:"11-DIGIT", userType:"STANDARD", department:{{{{id:X}}}}}}}}
   CRITICAL: email MUST be valid format! Use the email from the PDF. If no email in PDF, use firstname.lastname@example.org
   If POST fails with email validation error, retry WITHOUT email field.
2. POST /division (if needed) → POST /employment
3. POST /employment/details {{{{percentageOfFullTimeEquivalent:FROM_PDF}}}}
4. GET /employee/employment/occupationCode?code=DIGITS → PUT /employee/employment/ID with occupationCode:{{{{id:OCC_ID}}}}
5. POST /employee/standardTime {{{{employee:{{{{id:X}}}}, fromDate:START, hoursPerDay:7.5×pct/100}}}}
6. If salary: monthly = annual/12
CRITICAL: nationalIdentityNumber + occupationCode are SCORED!"""

    elif task_type == "invoice_send":
        return f"""{R_PRODUCT}
{R_INVOICE_FLOW}"""

    elif task_type == "invoice_multi":
        return f"""{R_PRODUCT}
{R_INVOICE_FLOW}
═══ MULTI-LINE ═══
POST multiple products with CORRECT vatType per line. 25%→id:3, 15%(food)→id:31, 0%(exempt)→id:5.
One orderline per product, ALWAYS set vatType + unitPriceExcludingVatCurrency."""

    elif task_type == "order":
        return f"""{R_PRODUCT}
{R_INVOICE_FLOW}
═══ ORDER ═══
Batch: [customer + products] → [order] → [orderlines] → [PUT order/:invoice] → [PUT invoice/:payment]
PUT /order/ID/:invoice query_params: invoiceDate={today}&sendToCustomer=false
PUT /invoice/ID/:payment: paidAmount = sum(prices) × 1.25"""

    elif task_type == "payment":
        return f"""═══ PAYMENT ═══
1. GET /invoice?invoiceDateFrom=2020-01-01&invoiceDateTo=2027-01-01&fields=id,invoiceNumber,amount,amountOutstanding,customer
2. PUT /invoice/ID/:payment query_params: paymentDate={today}&paymentTypeId=ID&paidAmount=AMOUNT"""

    elif task_type == "reverse_payment":
        return f"""═══ REVERSE PAYMENT ═══
1. GET /invoice?invoiceDateFrom=2020-01-01&invoiceDateTo=2027-01-01&fields=id,invoiceNumber,amount,amountOutstanding,customer
2. PUT /invoice/ID/:payment query_params: paymentDate={today}&paymentTypeId=ID&paidAmount=-AMOUNT (negative!)"""

    elif task_type == "credit_note":
        return f"""{R_PRODUCT}
{R_INVOICE_FLOW}
═══ CREDIT NOTE ═══
1. Find the invoice: GET /invoice?invoiceDateFrom=2020-01-01&invoiceDateTo=2027-01-01&fields=id,invoiceNumber,amount,amountOutstanding,customer
2. Create credit note: PUT /invoice/ID/:createCreditNote query_params: date={today}&sendToCustomer=false
If "Hele eller deler av fakturaen er kreditert" error: the invoice was already credited. Try finding another matching invoice.
IMPORTANT: First create the ORIGINAL invoice (customer+product+order+orderline+invoice), THEN credit it!"""

    elif task_type == "project":
        return f"""{R_EMPLOYEE}
{R_PROJECT_BASE}"""

    elif task_type == "project_fixed":
        return f"""{R_EMPLOYEE}
{R_PROJECT_BASE}
── FIXED PRICE ──
After project: GET /project/ID?fields=id,version,number → PUT /project/ID {{{{isFixedPrice:true, fixedprice:AMT, version:V, number:KEEP_ORIGINAL}}}}"""

    elif task_type == "timesheet":
        return f"""{R_EMPLOYEE}
{R_PROJECT_BASE}
{R_PRODUCT}
{R_INVOICE_FLOW}
═══ TIMESHEET ═══
Use activity ID from AVAILABLE ACTIVITIES context.
POST /timesheet/entry {{{{project:{{{{id:P}}}}, activity:{{{{id:A}}}}, employee:{{{{id:E}}}}, date:"{today}", hours:N}}}}
TOTAL = hours × rate. Product priceExVat=TOTAL. Orderline unitPriceExVat=TOTAL. vatType:{{{{id:3}}}} on BOTH!"""

    elif task_type == "travel_expense":
        return f"""{R_EMPLOYEE}
═══ TRAVEL EXPENSE ═══
POST /travelExpense {{{{employee:{{{{id:X}}}}, title:"...", travelDetails:{{{{departureDate, returnDate, departureFrom, destination, purpose}}}}}}}}
Per diem: POST /travelExpense/perDiemCompensation {{{{travelExpense:{{{{id:X}}}}, rateCategory:{{{{id:FROM_CONTEXT}}}}, count:DAYS, rate:RATE, overnightAccommodation:"HOTEL", location:"CITY"}}}}
Costs: POST /travelExpense/cost {{{{travelExpense:{{{{id:X}}}}, costCategory:{{{{id:Y}}}}, paymentType:{{{{id:Z}}}}, amountCurrencyIncVat:AMT, currency:{{{{id:1}}}}, date:"YYYY-MM-DD"}}}}"""

    elif task_type == "delete_travel":
        return f"""═══ DELETE TRAVEL ═══
GET /travelExpense → if APPROVED: PUT /:unapprove?id=X → DELETE /travelExpense/ID"""

    elif task_type == "salary":
        return f"""{R_EMPLOYEE}
{R_EMPLOYMENT}
{R_VOUCHER}
═══ SALARY ═══
Batch: [PUT employee + POST division] → [employment + standardTime] → [payslip flow]
1. Find employee by email. PUT with name+dateOfBirth FROM CONTEXT.
2. If hasEmployment=NO: create division+employment+details. If YES: skip to 3.
3. POST /employee/standardTime {{{{employee:{{{{id:X}}}}, fromDate:"{today}", hoursPerDay:7.5}}}} (ignore 422)
4. POST /salary/paySlip — create payslip with employee, date, month, year, and specifications inline:
   Body: {{{{employee:{{{{id:X}}}}, date:"{today}", year:{date.today().year}, month:{date.today().month}, specifications:[{{{{salaryType:{{{{id:TYPE_2000}}}}, rate:BASE, count:1, amount:BASE}}}}, {{{{salaryType:{{{{id:TYPE_2002}}}}, rate:BONUS, count:1, amount:BONUS}}}}]}}}}
5. PUT /salary/paySlip/{{{{ID}}}}/:calculate — calculates tax deductions
6. PUT /salary/paySlip/{{{{ID}}}}/:createPayment — finalizes payroll
   After step 6 → STOP! Task is COMPLETE."""

    elif task_type == "supplier_invoice":
        return f"""{R_VOUCHER}
═══ SUPPLIER INVOICE ═══
1. POST /supplier (via /customer with isSupplier:true, include email+phoneNumber)
2. Go STRAIGHT to voucher (do NOT try /incomingInvoice — always 403 in competition!):
   POST /ledger/voucher {{{{date:"INVOICE_DATE_FROM_PROMPT", description:"Leverandørfaktura [INV_NUMBER] [SUPPLIER_NAME]", postings:[
     {{{{account:{{{{id:EXPENSE_ACCT_ID}}}}, amountGross:AMOUNT_INCL_VAT, amountGrossCurrency:AMOUNT_INCL_VAT, date:"INVOICE_DATE", row:1, vatType:{{{{id:1}}}}}}}},
     {{{{account:{{{{id:ACCT_2400_ID}}}}, amountGross:-AMOUNT_INCL_VAT, amountGrossCurrency:-AMOUNT_INCL_VAT, date:"INVOICE_DATE", row:2, vatType:{{{{id:0}}}}, supplier:{{{{id:SUPPLIER_ID}}}}}}}}
   ]}}}}
   - amountGross = TOTAL INCLUDING VAT on BOTH postings. vatType:1 auto-splits MVA.
   - Expense acct from prompt: 6540=kontor/office, 6340=IT, 6500=rekvisita, 7300=service, 7100=bil, 7140=reise, 6800=annet
   - Use INVOICE DATE from prompt (not today!)"""

    elif task_type == "supplier_invoice_pdf":
        return f"""{R_VOUCHER}
═══ SUPPLIER INVOICE FROM PDF ═══
Extract from PDF: supplier name/org, invoice number, date, amounts, description.
1. POST /supplier (via /customer with isSupplier:true, include email!)
2. Go STRAIGHT to voucher (do NOT try /incomingInvoice — always 403!):
   POST /ledger/voucher {{{{date:"INVOICE_DATE", description:"Leverandørfaktura [INV_NUMBER] [SUPPLIER_NAME]", postings:[
     {{{{account:{{{{id:EXPENSE_ACCT_ID}}}}, amountGross:AMOUNT_INCL_VAT, amountGrossCurrency:AMOUNT_INCL_VAT, date:"INVOICE_DATE", row:1, vatType:{{{{id:1}}}}}}}},
     {{{{account:{{{{id:ACCT_2400_ID}}}}, amountGross:-AMOUNT_INCL_VAT, amountGrossCurrency:-AMOUNT_INCL_VAT, date:"INVOICE_DATE", row:2, vatType:{{{{id:0}}}}, supplier:{{{{id:SUPPLIER_ID}}}}}}}}
   ]}}}}
Expense mapping: 6340=IT, 6500=office/rekvisita, 6800=other, 7100=vehicle, 7140=travel, 6540=kontor"""

    elif task_type == "receipt_voucher":
        return f"""{R_VOUCHER}
═══ RECEIPT VOUCHER ═══
Account mapping: 6500=kontorrekvisita, 7350=representasjon, 7100=bil, 7140=reise, 6900=telefon, 6300=leie, 6340=IT/data, 6350=strøm, 6540=inventar/møbler, 7700=drift
POST /ledger/voucher: debit EXPENSE(vatType:{{{{id:1}}}}, department:{{{{id:DEPT}}}}), credit 1920(vatType:{{{{id:0}}}}).
vatType 1=ingoing25%. Food: vatType 11(ingoing15%). Use RECEIPT DATE not today!"""

    elif task_type == "acct_dimension":
        return f"""{R_VOUCHER}
═══ ACCOUNTING DIMENSIONS ═══
1. POST /ledger/accountingDimensionName {{{{dimensionName:"...", description:"..."}}}} → get dimensionIndex from response
2. POST /ledger/accountingDimensionValue {{{{displayName:"Value1", number:1, dimensionIndex:X}}}}
3. POST /ledger/accountingDimensionValue {{{{displayName:"Value2", number:2, dimensionIndex:X}}}}
4. POST /ledger/voucher to link dimension to account:
   {{{{date:"{today}", description:"...", postings:[
     {{{{account:{{{{id:ACCT_ID}}}}, amountGross:AMOUNT, amountGrossCurrency:AMOUNT, date:"{today}", row:1, vatType:{{{{id:0}}}}, freeAccountingDimension1:{{{{id:VALUE_ID}}}}}}}},
     {{{{account:{{{{id:BANK_1920}}}}, amountGross:-AMOUNT, amountGrossCurrency:-AMOUNT, date:"{today}", row:2, vatType:{{{{id:0}}}}}}}}
   ]}}}}
CRITICAL: Use POST /ledger/voucher with postings array. NOT /voucher, /journalEntry, /generalLedgerEntry (those don't exist!)."""

    elif task_type == "contact_person":
        return f"""═══ CONTACT PERSON ═══
1. Find/create customer (GET /customer or POST /customer)
2. POST /contact {{{{firstName, lastName, email, customer:{{{{id:CUST_ID}}}}}}}}
phoneNumberMobile only if in prompt. No title/role fields."""

    elif task_type == "project_lifecycle":
        return f"""{R_EMPLOYEE}
{R_EMPLOYMENT}
{R_PROJECT_BASE}
{R_PRODUCT}
{R_INVOICE_FLOW}
{R_VOUCHER}
═══ PROJECT LIFECYCLE ═══
Full cycle: employees → customer → project → hours → supplier cost → customer invoice.
1. For EACH employee: PUT to EXTENDED + entitlements(45+10). Ensure employment.
2. POST /customer (use existing if org matches)
3. POST /project. If budget: GET→PUT with isFixedPrice+fixedprice, KEEP original number.
4. POST /timesheet/entry per employee {{{{project:{{{{id:PROJ}}}}, activity:{{{{id:ACT}}}}, employee:{{{{id:EMP}}}}, date:"{today}", hours:N}}}}
   CRITICAL: GET /project/ID?fields=id,projectActivities(id,activity(id,name)) to find valid activity IDs for the project!
   Use projectActivities[0].activity.id (NOT a random activity from context!)
5. Supplier cost: voucher debit expense(vatType:1) credit 2400(supplier). Use correct expense acct (4300=subcontractor).
6. Customer invoice: product→order→orderline→invoice. vatType:{{{{id:3}}}} on both product+orderline."""

    elif task_type == "cost_analysis":
        return f"""{R_EMPLOYEE}
{R_PROJECT_BASE}
═══ COST ANALYSIS ═══
1. GET /ledger/posting?dateFrom=2026-01-01&dateTo=2026-01-31&fields=id,amount,account(number,name)&count=1000
   NOTE: Use account(number,name) with PARENTHESES, NOT account.number (dots give 400 error!)
2. GET /ledger/posting?dateFrom=2026-02-01&dateTo=2026-02-28&fields=id,amount,account(number,name)&count=1000
3. Group by account number, sum amounts per month, calc increase (feb-jan)
4. Find 3 accounts with largest increase
5. Make employee EXTENDED: PUT /employee/ID {{{{userType:"EXTENDED",...}}}}
6. POST /employee/entitlement {{{{employee:{{{{id:EMP}}}}, entitlementId:45, customer:{{{{id:{company_id or 'COMPANY_ID'}}}}}}}}}
   POST /employee/entitlement {{{{employee:{{{{id:EMP}}}}, entitlementId:10, customer:{{{{id:{company_id or 'COMPANY_ID'}}}}}}}}}
   CRITICAL: customer field is REQUIRED — use company_id from context!
7. For each of 3 accounts: POST /project {{{{name:ACCOUNT_NAME, startDate:"{today}", projectManager:{{{{id:EMP}}}}, isInternal:true}}}}
8. For each project: POST /activity {{{{name:ACCOUNT_NAME, activityType:"GENERAL_ACTIVITY", isGeneral:true}}}}
   CRITICAL: activityType is REQUIRED — always use "GENERAL_ACTIVITY"!"""

    elif task_type == "year_end":
        return f"""{R_VOUCHER}
═══ YEAR-END ═══
1. Lookup ledger accounts from context or GET /ledger/account?number=XXXX
2. Depreciation per asset: voucher debit 6010/6020, credit 1209/1219/etc. Linear = cost/years. SEPARATE voucher per asset!
3. Prepaid reversal: voucher debit expense(6300/7140), credit 1700. Amount from prompt.
4. Tax 22%: GET ledger postings full year → calc revenue-expenses-depreciation → voucher debit 8700, credit 2920.
   CRITICAL: calc tax AFTER depreciation+prepaid!
5. Salary accrual: voucher debit 5000, credit 2900."""

    elif task_type == "month_end":
        return f"""{R_VOUCHER}
═══ MONTHLY CLOSING ═══
1. Prepaid accrual: voucher debit expense, credit 1710. Monthly = total/months.
2. Monthly depreciation: annual/12 per asset.
3. Trial balance: GET postings for month, verify total=0.
4. Additional accruals as specified."""

    elif task_type == "bank_recon":
        return f"""{R_VOUCHER}
═══ BANK RECONCILIATION ═══
1. Parse CSV: date, amount, reference per transaction
2. GET /invoice?invoiceDateFrom=2020-01-01&invoiceDateTo=2027-01-01&fields=id,invoiceNumber,amount,amountOutstanding,customer
3. Incoming payments: match by amount/ref → PUT /invoice/ID/:payment
4. Outgoing: voucher debit 2400, credit 1920
5. Handle partial payments. Match by ref/name if amount doesn't match exactly."""

    elif task_type == "reminder_fee":
        return f"""{R_PRODUCT}
{R_INVOICE_FLOW}
{R_VOUCHER}
═══ REMINDER FEE ═══
1. GET invoices, find overdue (amountOutstanding>0, dueDate<today)
2. Reminder voucher: debit 1500, credit 3400, amount=35
3. Reminder invoice: product "Purregebyr" 35kr vatType:{{{{id:6}}}}(0%) → order→orderline→invoice
4. Partial payment on overdue: PUT /invoice/ID/:payment paidAmount=PARTIAL"""

    elif task_type == "ledger_audit":
        return f"""{R_VOUCHER}
═══ LEDGER AUDIT ═══
The prompt TELLS you the 4 errors (wrong account, duplicate, missing VAT, wrong amount). You do NOT need to find them.
1. GET /ledger/account for any accounts not in pre-fetched context (e.g. 6390, 6590, 6860)
2. For EACH error, immediately POST /ledger/voucher with corrective postings:
   - Wrong account: debit correct acct, credit wrong acct (reverse the error)
   - Duplicate: reverse it (swap debit/credit signs)
   - Missing VAT: debit expense with vatType:{{{{id:1}}}} (auto-splits to 2710), credit same amount vatType:{{{{id:0}}}}
   - Wrong amount: reverse wrong amount, post correct amount
3. Each correction = separate POST /ledger/voucher, description: "Korreksjon: [reason]"
CRITICAL: Do NOT waste iterations browsing vouchers! The prompt already has all error details. Go straight to corrections!"""

    elif task_type == "fx_invoice":
        return f"""{R_VOUCHER}
═══ FX INVOICE ═══
1. Find/create customer
2. GET invoices, find the one mentioned
3. PUT /invoice/ID/:payment paidAmount=INVOICE_AMOUNT (from GET, NOT recalculated)
4. FX voucher: diff = EUR×new_rate - EUR×old_rate
   GAIN: debit 1500, credit 8060
   LOSS: debit 8160, credit 1500
   Use 1500 (kundefordringer) NOT 1920! Currency IDs from context."""

    else:
        # Unknown — include core recipes only (employee, customer, product, invoice, voucher, project)
        return f"""{R_EMPLOYEE}
{R_EMPLOYMENT}
{R_PRODUCT}
{R_INVOICE_FLOW}
{R_VOUCHER}
{R_PROJECT_BASE}
── SUPPLIER ──
POST /customer with isSupplier:true, isCustomer:false.
── CUSTOMER ──
POST /customer {{{{name, organizationNumber, email, phoneNumber, isCustomer:true, postalAddress:{{{{addressLine1, postalCode, city}}}}}}}}
── DEPARTMENT ──
POST /department {{{{name, departmentNumber:INT}}}} — unique numbers.
── ACCT DIMENSIONS ──
POST /ledger/accountingDimensionName → POST /ledger/accountingDimensionValue → freeAccountingDimension1:{{{{id:X}}}}"""


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
                "inline_data": {"mime_type": mime, "data": raw_b64},
            })
            text_parts.append(f"[Image '{filename}' attached — visible to you]")

    return text_parts, image_blocks


# ═══════════════════════════════════════════════
# TASK TYPE DETECTION
# ═══════════════════════════════════════════════

def detect_task_type(prompt: str) -> str:
    """Detect task type from prompt for logging and iteration limits."""
    import unicodedata
    # Normalize: replace accented chars with ASCII equivalents, then strip remaining non-ASCII
    p = unicodedata.normalize("NFKD", prompt.lower())
    p = p.replace("ø", "o").replace("å", "a").replace("æ", "ae").replace("ü", "u").replace("ö", "o").replace("ä", "a")
    p = p.encode("ascii", "ignore").decode("ascii")
    # Order matters — more specific patterns first
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
    # Salary — but NOT if it's actually month-end/year-end with salary accrual
    if re.search(r'lonn|lon\b|payroll|nomina|gehalt|salario|salary', p):
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
    if re.search(r'opprett.*ordre|create.*order|crie.*commande|cria.*pedido|erstellen.*auftrag|creez.*commande', p):
        return "order"
    if re.search(r'tre.*produktlinj|three.*product.line|tres.*linha|tres.*linea|drei.*produkt|trois.*ligne|com tres.*produto|con tres.*producto|mit drei.*produkt|avec trois.*produit|with three.*product', p):
        return "invoice_multi"
    if re.search(r'opprett.*send.*faktura|create.*send.*invoice|crie.*envie|crea.*env.*factura|erstellen.*senden.*rechnung|creez.*envoyez', p):
        return "invoice_send"
    if re.search(r'ny.*ansatt|new.*employee.*born|novo.*funcion|nuevo.*empleado|neuen.*mitarbeiter|nouvel.*employe|ny.*tilsett|nouveau.*employe|nova.*funcion', p):
        return "employee"
    if re.search(r'registr.*leverand|regist.*liefer|regist.*fornecedor|regist.*proveedor|enregistr.*fournisseur', p):
        return "supplier"
    if re.search(r'opprett.*produkt|create.*product|cr[ée][ez].*produi[ts]|crea.*producto|erstellen.*produkt|enregistr.*produit', p):
        return "product"
    if re.search(r'opprett.*kunde|create.*customer|crie.*cliente|crea.*cliente|erstellen.*kund|creez.*client|enregistr.*client', p):
        return "customer"
    return "unknown"


COMPLEX_TASKS = {"supplier_invoice_pdf", "year_end", "bank_recon", "ledger_audit",
                 "fx_invoice", "reminder_fee", "employee_pdf", "project_lifecycle",
                 "cost_analysis", "month_end", "timesheet"}


# ═══════════════════════════════════════════════
# MAIN ENDPOINT
# ═══════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "version": "v23-fixes", "model": CLAUDE_MODEL if USE_CLAUDE else GEMINI_MODEL, "llm": "claude" if USE_CLAUDE else "gemini"}


@app.post("/solve")
@app.post("/")
async def solve(request: Request):
    start_time = time.time()
    try:
        body = await request.json()
        return await _solve_inner(body, start_time)
    except Exception as e:
        log.error(f"Top-level error after {time.time()-start_time:.1f}s: {e}")
        return JSONResponse({"status": "completed"})


async def _solve_inner(body, start_time):
    prompt = body["prompt"]
    files = body.get("files", [])
    creds = body["tripletex_credentials"]

    log.info("=" * 60)
    log.info(f"PROMPT: {prompt}")
    log.info(f"FILES: {[f['filename'] for f in files]}")

    # Detect task type for logging and iteration limits
    task_type = detect_task_type(prompt)
    log.info(f"Detected task type: {task_type}")

    # Set iteration limit based on task type
    if task_type in COMPLEX_TASKS:
        max_iterations = 18  # Complex tasks — reduced from 25
    else:
        max_iterations = 10  # Simple tasks — finish fast

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
    ctx = prefetch_context(session, base_url, task_type)
    if company_id:
        ctx["_company_id"] = company_id
    ctx_text = format_prefetched_context(ctx)

    # Process files
    text_parts, image_blocks = process_files(files)
    file_text = "\n---\n".join(text_parts) if text_parts else ""

    # ═══ DETERMINISTIC HANDLERS — try these FIRST ═══
    try:
        from handlers import DETERMINISTIC_HANDLERS, solve_deterministic
        if task_type in DETERMINISTIC_HANDLERS:
            log.info(f"Using DETERMINISTIC handler for {task_type}")
            full_prompt = prompt
            if file_text:
                full_prompt += f"\n\nAttached files:\n{file_text}"
            trace = solve_deterministic(
                task_type, full_prompt, [], session, base_url, ctx,
                GEMINI_API_KEY, GEMINI_MODEL, start_time, image_blocks
            )
            if trace:
                elapsed = time.time() - start_time
                total_calls = len(trace)
                total_errors = sum(1 for t in trace if t.get("status", 200) >= 400)
                log.info(f"=== DONE {elapsed:.1f}s type={task_type} DETERMINISTIC calls={total_calls} errors={total_errors} fp={_prompt_fingerprint(prompt)} ===")
                _log_run(prompt, files, trace, elapsed, task_type)
                return JSONResponse({"status": "completed"})
            log.warning(f"Deterministic returned empty trace for {task_type}, falling back to LLM agent")
    except Exception as e:
        log.error(f"Deterministic handler failed for {task_type}: {e}")
        log.error(traceback.format_exc())
        log.info(f"Falling back to LLM agent for {task_type}")

    # ═══ LLM AGENT FALLBACK — for unknown tasks or if deterministic fails ═══

    # Build user message
    user_text = f"Complete this accounting task:\n\n{prompt}"
    if file_text:
        user_text += f"\n\nAttached files:\n{file_text}"
    if ctx_text:
        user_text += f"\n\n═══ PRE-FETCHED SANDBOX DATA ═══\n{ctx_text}"

    system_prompt = build_system_prompt(company_id, task_type)
    trace = []
    deadline = start_time + 250  # 50s buffer before 300s timeout
    tok = 1536 if task_type in COMPLEX_TASKS else 768

    if USE_CLAUDE and ANTHROPIC_API_KEY:
        # ═══ CLAUDE AGENT LOOP ═══
        log.info("Using Claude agent loop")

        # Build Claude message content
        content = []
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

        messages = [{"role": "user", "content": content}]

        for iteration in range(max_iterations):
            if time.time() > deadline:
                log.warning(f"Deadline reached at iteration {iteration}")
                break
            remaining = deadline - time.time()
            if remaining < 15:
                log.warning(f"Only {remaining:.0f}s left, stopping at iteration {iteration}")
                break

            # Trim old messages (keep first user + last 12)
            if len(messages) > 13:
                messages = [messages[0]] + messages[-12:]

            response = claude_generate(system_prompt, messages, CLAUDE_TOOLS, tok)
            if not response:
                log.error("Claude returned no response")
                break

            # Check for tool use
            tool_uses = [b for b in response.content if b.type == "tool_use"]

            if tool_uses:
                # Add assistant response to conversation
                messages.append({"role": "assistant", "content": response.content})

                # Execute tools
                tool_results = []
                if len(tool_uses) > 1:
                    import concurrent.futures
                    def run_claude_tool(tu):
                        log.info(f"Tool: {tu.name}({json.dumps(tu.input, ensure_ascii=False)[:800]})")
                        result = execute_tool(tu.name, tu.input, session, base_url, trace, ctx)
                        return {"type": "tool_result", "tool_use_id": tu.id, "content": str(result)[:4000]}
                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                        tool_results = list(ex.map(run_claude_tool, tool_uses))
                else:
                    for tu in tool_uses:
                        log.info(f"Tool: {tu.name}({json.dumps(tu.input, ensure_ascii=False)[:800]})")
                        result = execute_tool(tu.name, tu.input, session, base_url, trace, ctx)
                        result_str = str(result)[:4000]
                        # Add urgency signal
                        remaining_time = deadline - time.time()
                        remaining_iters = max_iterations - iteration - 1
                        if remaining_time < 60 or remaining_iters <= 3:
                            result_str += f"\n[URGENT: {remaining_time:.0f}s and {remaining_iters} iterations left. Batch ALL remaining calls NOW or stop.]"
                        tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": result_str})

                messages.append({"role": "user", "content": tool_results})

                # Safety: too many errors
                error_count = sum(1 for t in trace if t.get("status", 200) >= 400)
                if error_count >= 15:
                    log.warning(f"Too many errors ({error_count}), stopping")
                    break
            else:
                log.info(f"Claude finished at iteration {iteration}")
                break

    else:
        # ═══ GEMINI AGENT LOOP (fallback) ═══
        log.info("Using Gemini agent loop")

        content_parts = [{"text": user_text}]
        content_parts.extend(image_blocks)
        messages = [{"role": "user", "parts": content_parts}]

        for iteration in range(max_iterations):
            if time.time() > deadline:
                log.warning(f"Deadline reached at iteration {iteration}")
                break

            remaining = deadline - time.time()
            if remaining < 15:
                log.warning(f"Only {remaining:.0f}s left, stopping at iteration {iteration}")
                break

            if len(messages) > 13:
                messages = [messages[0]] + messages[-12:]

            resp_json = gemini_generate(GEMINI_MODEL, system_prompt, messages, GEMINI_TOOLS, tok)
            if not resp_json or "candidates" not in resp_json:
                log.error(f"Gemini returned no candidates: {json.dumps(resp_json or {})[:500]}")
                break

            candidate = resp_json["candidates"][0]
            parts = candidate.get("content", {}).get("parts", [])

            function_calls = [p for p in parts if "functionCall" in p]

            if function_calls:
                messages.append({"role": "model", "parts": parts})

                fn_responses = []
                if len(function_calls) > 1:
                    import concurrent.futures
                    def run_tool(fc_part):
                        fc = fc_part["functionCall"]
                        log.info(f"Tool: {fc['name']}({json.dumps(fc.get('args', {}), ensure_ascii=False)[:800]})")
                        result = execute_tool(fc["name"], fc.get("args", {}), session, base_url, trace, ctx)
                        return {"functionResponse": {"name": fc["name"], "response": {"result": result}}}
                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                        fn_responses = list(ex.map(run_tool, function_calls))
                else:
                    for fc_part in function_calls:
                        fc = fc_part["functionCall"]
                        log.info(f"Tool: {fc['name']}({json.dumps(fc.get('args', {}), ensure_ascii=False)[:800]})")
                        result = execute_tool(fc["name"], fc.get("args", {}), session, base_url, trace, ctx)
                        fn_responses.append({"functionResponse": {"name": fc["name"], "response": {"result": result}}})

                remaining_time = deadline - time.time()
                remaining_iters = max_iterations - iteration - 1
                if remaining_time < 60 or remaining_iters <= 3:
                    last_resp = fn_responses[-1]["functionResponse"]["response"]
                    last_resp["result"] = str(last_resp["result"]) + f"\n[URGENT: {remaining_time:.0f}s and {remaining_iters} iterations left. Batch ALL remaining calls NOW or stop.]"

                messages.append({"role": "user", "parts": fn_responses})

                error_count = sum(1 for t in trace if t.get("status", 200) >= 400)
                if error_count >= 15:
                    log.warning(f"Too many errors ({error_count}), stopping")
                    break
            else:
                log.info(f"Gemini finished at iteration {iteration}")
                break

    elapsed = time.time() - start_time
    total_calls = len(trace)
    total_errors = sum(1 for t in trace if t.get("status", 200) >= 400)
    log.info(f"=== DONE {elapsed:.1f}s type={task_type} calls={total_calls} errors={total_errors} fp={_prompt_fingerprint(prompt)} ===")

    # Log run
    _log_run(prompt, files, trace, elapsed, task_type)

    return JSONResponse({"status": "completed"})


def _prompt_fingerprint(prompt: str) -> str:
    """Create a stable fingerprint for a prompt to identify recurring tasks."""
    return hashlib.md5(prompt.strip().encode()).hexdigest()[:12]

def _log_run(prompt, files, trace, elapsed, task_type="unknown"):
    """Append run details to JSONL log."""
    try:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "v33-claude",
            "model": CLAUDE_MODEL if USE_CLAUDE else GEMINI_MODEL,
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
        # Persist to GCS (survives deploys)
        _gcs_append_log(entry_json)
    except Exception:
        pass


@app.get("/logs")
def get_logs():
    """View recent submission logs (persistent via GCS)."""
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
    """Aggregated view: group all runs by prompt fingerprint.
    Shows unique tasks with their type, run count, latest trace, and prompt snippet."""
    entries = _gcs_read_logs()
    if not entries:
        if LOG_PATH.exists():
            lines = LOG_PATH.read_text().strip().split("\n")
            entries = [json.loads(l) for l in lines if l.strip()]

    tasks = {}
    for e in entries:
        fp = e.get("prompt_fingerprint", "unknown")
        if fp not in tasks:
            tasks[fp] = {
                "fingerprint": fp,
                "task_type": e.get("task_type", "unknown"),
                "prompt_snippet": e.get("prompt", "")[:200],
                "runs": 0,
                "best_calls": 999,
                "worst_calls": 0,
                "best_errors": 999,
                "latest_timestamp": "",
                "history": [],
            }
        t = tasks[fp]
        t["runs"] += 1
        calls = e.get("api_calls", 0)
        errors = e.get("api_errors", 0)
        t["best_calls"] = min(t["best_calls"], calls)
        t["worst_calls"] = max(t["worst_calls"], calls)
        t["best_errors"] = min(t["best_errors"], errors)
        t["latest_timestamp"] = e.get("timestamp", "")
        t["history"].append({
            "timestamp": e.get("timestamp", ""),
            "version": e.get("version", "?"),
            "calls": calls,
            "errors": errors,
            "elapsed": e.get("elapsed_seconds", 0),
        })

    # Sort by task type for readability
    sorted_tasks = sorted(tasks.values(), key=lambda x: x["task_type"])
    return {"unique_tasks": len(sorted_tasks), "tasks": sorted_tasks}


@app.post("/scores")
async def post_scores(request: Request):
    """POST task scores from leaderboard. Body: {"scores": {1: 2.0, 2: 3.5, ...}}
    Persists to GCS so we can track score changes over time."""
    try:
        body = await request.json()
        scores = body.get("scores", {})
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scores": {str(k): v for k, v in scores.items()},
            "total": sum(scores.values()),
        }
        entry_json = json.dumps(entry, ensure_ascii=False)
        try:
            client = _gcs_client()
            if client:
                bucket = client.bucket(GCS_BUCKET)
                if not bucket.exists():
                    bucket.location = "EUROPE-NORTH1"
                    client.create_bucket(bucket)
                blob = bucket.blob("scores.jsonl")
                existing = ""
                if blob.exists():
                    existing = blob.download_as_text()
                blob.upload_from_string(existing + entry_json + "\n", content_type="text/plain")
        except Exception as e:
            log.warning(f"GCS score write failed: {e}")
        return {"status": "saved", "total": sum(scores.values())}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/scores")
def get_scores():
    """Get score history over time."""
    try:
        client = _gcs_client()
        if not client:
            return {"history": []}
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob("scores.jsonl")
        if not blob.exists():
            return {"history": []}
        lines = blob.download_as_text().strip().split("\n")
        entries = [json.loads(l) for l in lines if l.strip()]
        return {"history": entries}
    except Exception:
        return {"history": []}


@app.post("/tag")
async def tag_task(request: Request):
    """Tag a fingerprint with a task number. Body: {"fingerprint": "abc123", "task_number": 4}
    Or tag by matching prompt text: {"match": "keyword in prompt", "task_number": 4}"""
    try:
        body = await request.json()
        client = _gcs_client()
        if not client:
            return JSONResponse({"error": "GCS not available"}, status_code=500)
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob("task_tags.json")
        tags = {}
        if blob.exists():
            tags = json.loads(blob.download_as_text())

        if "fingerprint" in body:
            tags[body["fingerprint"]] = body["task_number"]
        elif "match" in body:
            # Find fingerprints matching this text in logs
            entries = _gcs_read_logs()
            matched = 0
            for e in entries:
                if body["match"].lower() in e.get("prompt", "").lower():
                    tags[e["prompt_fingerprint"]] = body["task_number"]
                    matched += 1
            if matched == 0:
                return {"status": "no_match", "search": body["match"]}

        blob.upload_from_string(json.dumps(tags), content_type="application/json")
        return {"status": "tagged", "tags": tags}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


def _get_task_tags() -> dict:
    """Get fingerprint → task_number mapping from GCS."""
    try:
        client = _gcs_client()
        if not client:
            return {}
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob("task_tags.json")
        if not blob.exists():
            return {}
        return json.loads(blob.download_as_text())
    except Exception:
        return {}


def _get_task_knowledge() -> dict:
    """Get task knowledge base from GCS. {task_number: {type, checks, notes, experiments}}"""
    try:
        client = _gcs_client()
        if not client:
            return {}
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob("task_knowledge.json")
        if not blob.exists():
            return {}
        return json.loads(blob.download_as_text())
    except Exception:
        return {}


def _save_task_knowledge(knowledge: dict):
    """Save task knowledge base to GCS."""
    try:
        client = _gcs_client()
        if not client:
            return
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob("task_knowledge.json")
        blob.upload_from_string(json.dumps(knowledge, ensure_ascii=False, indent=2), content_type="application/json")
    except Exception as e:
        log.warning(f"GCS knowledge write failed: {e}")


@app.post("/knowledge")
async def post_knowledge(request: Request):
    """Update task knowledge. Body: {"task_number": 1, "type": "employee", "checks_total": 7, "checks_passed": 7, "score": 1.50, "notes": "...", "experiments": [...]}"""
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
    """Get all task knowledge."""
    return _get_task_knowledge()


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Visual dashboard with task scores, run logs, and API trace details."""
    score_history = get_scores().get("history", [])
    latest_scores = score_history[-1] if score_history else {"scores": {}, "total": 0, "timestamp": "---"}
    prev_scores = score_history[-2] if len(score_history) >= 2 else None
    tags = _get_task_tags()
    reverse_tags = {v: k for k, v in tags.items()}
    all_logs = _gcs_read_logs()
    knowledge = _get_task_knowledge()

    # Build task cards with knowledge
    cards_html = ""
    total = 0
    total_delta = 0
    for i in range(1, 31):
        s = float(latest_scores["scores"].get(str(i), 0))
        total += s
        delta = ""
        if prev_scores:
            prev = float(prev_scores["scores"].get(str(i), 0))
            diff = s - prev
            if diff > 0.01:
                delta = f'<span class="delta up">+{diff:.2f}</span>'
                total_delta += diff
            elif diff < -0.01:
                delta = f'<span class="delta down">{diff:.2f}</span>'
                total_delta += diff

        if s == 0:
            color = "#ef4444"
        elif s < 2.0:
            color = "#f97316"
        elif s < 3.5:
            color = "#eab308"
        elif s < 5.0:
            color = "#22c55e"
        else:
            color = "#06b6d4"

        # Get knowledge for this task
        k = knowledge.get(str(i), {})
        type_name = k.get("type", "")
        if not type_name:
            fp = reverse_tags.get(i) or reverse_tags.get(str(i))
            if fp:
                for e in all_logs:
                    if e.get("prompt_fingerprint") == fp:
                        type_name = e.get("task_type", "")
                        break

        type_html = f'<div class="task-type">{type_name}</div>' if type_name else ''
        checks_html = f'<div class="task-checks">{k["checks"]}</div>' if k.get("checks") else ''
        notes_html = f'<div class="task-notes" title="{k.get("notes","")}">{k.get("notes","")[:30]}</div>' if k.get("notes") else ''
        exp_count = len(k.get("experiments", []))
        exp_html = f'<div class="task-exp">{exp_count} exp</div>' if exp_count > 0 else ''

        cards_html += f'''
        <div class="card" style="border-top: 3px solid {color}">
            <div class="task-num">Task {i:02d}</div>
            <div class="score" style="color: {color}">{s:.2f}</div>
            {delta}
            {type_html}
            {checks_html}
            {notes_html}
            {exp_html}
        </div>'''

    total_delta_html = ""
    if prev_scores:
        if total_delta > 0:
            total_delta_html = f'<span class="delta up" style="font-size:1.2em">+{total_delta:.2f}</span>'
        elif total_delta < 0:
            total_delta_html = f'<span class="delta down" style="font-size:1.2em">{total_delta:.2f}</span>'

    # Recent runs — last 20, newest first
    recent_runs = sorted(all_logs, key=lambda x: x.get("timestamp", ""), reverse=True)[:20]
    runs_html = ""
    for r in recent_runs:
        fp = r.get("prompt_fingerprint", "?")
        task_num = tags.get(fp, "?")
        task_type = r.get("task_type", "?")
        calls = r.get("api_calls", 0)
        errors = r.get("api_errors", 0)
        elapsed = r.get("elapsed_seconds", 0)
        ts = r.get("timestamp", "")[:19].replace("T", " ")
        ver = r.get("version", "?")
        prompt_snip = r.get("prompt", "")[:120].replace("<", "&lt;").replace('"', "&quot;")

        # Trace summary — show API endpoints called and their status
        trace = r.get("trace", [])
        trace_summary = ""
        if trace:
            status_counts = {}
            endpoints = []
            for t in trace[:15]:  # first 15 calls
                method = t.get("method", "?")
                path = t.get("path", t.get("url", "?"))
                # Shorten path
                if isinstance(path, str) and "/v2/" in path:
                    path = path.split("/v2/")[-1]
                elif isinstance(path, str) and "tripletex" in path:
                    path = path.split(".dev/v2/")[-1] if ".dev/v2/" in path else path.split(".no/v2/")[-1] if ".no/v2/" in path else path
                status = t.get("status", "?")
                endpoints.append(f"{method} {path[:40]} → {status}")
                key = f"{status}"
                status_counts[key] = status_counts.get(key, 0) + 1
            status_str = " ".join(f'<span class="{"st-ok" if str(k).startswith("2") else "st-err"}">{k}:{v}</span>' for k, v in sorted(status_counts.items()))
            trace_detail = "\\n".join(endpoints)
            trace_summary = f'<div class="trace-status">{status_str}</div>'
            trace_summary += f'<details><summary>Vis {len(trace)} API-kall</summary><pre class="trace-pre">{chr(10).join(endpoints)}</pre></details>'

        error_class = ' class="run-error"' if errors > 0 else ""
        task_label = f"<b>T{task_num}</b>" if task_num != "?" else f'<span class="untagged" title="{fp}">#{fp[:8]}</span>'

        runs_html += f'''
        <div class="run-card"{error_class}>
            <div class="run-header">
                <span class="run-task">{task_label}</span>
                <span class="run-type">{task_type}</span>
                <span class="run-stats">{calls} calls, {errors} err, {elapsed}s</span>
                <span class="run-time">{ts}</span>
                <span class="run-ver">{ver}</span>
            </div>
            <div class="run-prompt">{prompt_snip}</div>
            {trace_summary}
        </div>'''

    # Score history
    history_html = ""
    if score_history:
        for idx, h in enumerate(score_history):
            pct = (h["total"] / 180) * 100
            ts = h["timestamp"][:16].replace("T", " ")
            history_html += f'''
            <div class="bar-row">
                <div class="bar-label">#{idx+1}</div>
                <div class="bar" style="width:{pct}%"></div>
                <div class="bar-val">{h["total"]:.2f} &middot; {ts}</div>
            </div>'''
    else:
        history_html = '<p style="color:#94a3b8">Ingen score-historikk ennå.</p>'

    html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Tripletex Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="60">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0f172a; color:#e2e8f0; font-family:-apple-system,system-ui,sans-serif; padding:16px; max-width:1200px; margin:0 auto; }}
h1 {{ text-align:center; margin-bottom:4px; font-size:1.4em; }}
.subtitle {{ text-align:center; color:#94a3b8; margin-bottom:16px; font-size:0.85em; }}
.total {{ text-align:center; font-size:2.8em; font-weight:800; color:#06b6d4; margin:8px 0; }}
.grid {{ display:grid; grid-template-columns:repeat(6,1fr); gap:8px; margin-bottom:24px; }}
@media(max-width:800px) {{ .grid {{ grid-template-columns:repeat(3,1fr); }} }}
.card {{ background:#1e293b; border-radius:8px; padding:10px; text-align:center; }}
.task-num {{ font-size:0.7em; color:#94a3b8; }}
.task-type {{ font-size:0.65em; color:#64748b; margin-top:2px; }}
.task-checks {{ font-size:0.6em; color:#94a3b8; }}
.task-notes {{ font-size:0.55em; color:#475569; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.task-exp {{ font-size:0.55em; color:#06b6d4; }}
.score {{ font-size:1.6em; font-weight:700; }}
.delta {{ font-size:0.75em; display:block; margin-top:1px; }}
.delta.up {{ color:#22c55e; }}
.delta.down {{ color:#ef4444; }}
h2 {{ margin:24px 0 8px; font-size:1.1em; color:#06b6d4; }}
.legend {{ display:flex; gap:12px; justify-content:center; margin:10px 0; font-size:0.75em; flex-wrap:wrap; }}
.legend span {{ display:flex; align-items:center; gap:3px; }}
.dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}
.history {{ background:#1e293b; border-radius:8px; padding:12px; margin:8px 0; }}
.bar-row {{ display:flex; align-items:center; gap:6px; margin:3px 0; }}
.bar-label {{ width:40px; text-align:right; font-size:0.75em; color:#94a3b8; }}
.bar {{ height:16px; border-radius:3px; background:#06b6d4; min-width:2px; }}
.bar-val {{ font-size:0.75em; color:#94a3b8; }}
.run-card {{ background:#1e293b; border-radius:8px; padding:10px; margin:6px 0; border-left:3px solid #334155; }}
.run-card.run-error {{ border-left-color:#ef4444; }}
.run-header {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; font-size:0.8em; }}
.run-task {{ color:#06b6d4; font-size:1.1em; min-width:40px; }}
.run-type {{ background:#334155; padding:2px 8px; border-radius:4px; color:#94a3b8; }}
.run-stats {{ color:#94a3b8; }}
.run-time {{ color:#64748b; margin-left:auto; }}
.run-ver {{ color:#475569; font-size:0.8em; }}
.run-prompt {{ font-size:0.75em; color:#64748b; margin-top:4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.trace-status {{ margin-top:4px; font-size:0.75em; }}
.st-ok {{ color:#22c55e; margin-right:6px; }}
.st-err {{ color:#ef4444; margin-right:6px; }}
details {{ margin-top:4px; font-size:0.75em; }}
summary {{ color:#64748b; cursor:pointer; }}
.trace-pre {{ background:#0f172a; padding:8px; border-radius:4px; margin-top:4px; overflow-x:auto; font-size:0.85em; color:#94a3b8; white-space:pre-wrap; }}
.untagged {{ color:#f97316; cursor:help; }}
.refresh {{ text-align:center; margin:16px 0; }}
.refresh a {{ color:#06b6d4; text-decoration:none; background:#1e293b; padding:6px 16px; border-radius:6px; font-size:0.85em; }}
</style></head><body>
<h1>Tripletex Agent</h1>
<p class="subtitle">v25 &middot; {latest_scores.get("timestamp", "---")[:19]} &middot; Auto-refresh 60s</p>
<div class="total">{total:.2f} {total_delta_html}</div>
<div class="legend">
    <span><span class="dot" style="background:#ef4444"></span> 0</span>
    <span><span class="dot" style="background:#f97316"></span> &lt;2</span>
    <span><span class="dot" style="background:#eab308"></span> 2-3.5</span>
    <span><span class="dot" style="background:#22c55e"></span> 3.5-5</span>
    <span><span class="dot" style="background:#06b6d4"></span> 5+</span>
</div>
<div class="grid">{cards_html}</div>

<h2>Siste kjøringer</h2>
{runs_html if runs_html else '<p style="color:#94a3b8">Ingen kjøringer ennå. Submit tasks først.</p>'}

<h2>Score-historikk</h2>
<div class="history">{history_html}</div>

<div class="refresh"><a href="/dashboard">Oppdater nå</a></div>
</body></html>'''

    return HTMLResponse(html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
