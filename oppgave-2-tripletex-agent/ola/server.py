"""
Tripletex AI Accounting Agent — NM i AI 2026
v3: Gemini Function Calling — modellen kaller Tripletex API direkte.
"""

import base64
import json
import logging
import os
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import requests
from google import genai
from google.genai.types import GenerateContentConfig, AutomaticFunctionCallingConfig, Tool, FunctionDeclaration, Schema
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("agent")

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

SUBMISSION_LOG_PATH = Path(os.environ.get("SUBMISSION_LOG", "/tmp/tripletex_submissions.jsonl"))

# Global state per request (set in solve())
_tx_session = None
_tx_base_url = ""
_call_count = 0
_error_count = 0


# === TRIPLETEX TOOL FUNCTIONS (called by Gemini) ===

def tripletex_get(endpoint: str, query_params: str = "") -> str:
    """Make a GET request to the Tripletex API.

    Args:
        endpoint: API endpoint path, e.g. '/employee', '/customer', '/department'
        query_params: URL query string, e.g. 'fields=id,name&count=10' or 'name=Acme&fields=id,name'

    Returns:
        JSON response as string with the API result
    """
    global _call_count, _error_count
    _call_count += 1
    url = f"{_tx_base_url}/{endpoint.lstrip('/')}"
    params = dict(p.split("=", 1) for p in query_params.split("&") if "=" in p) if query_params else None
    resp = _tx_session.get(url, params=params)
    if resp.status_code >= 400:
        _error_count += 1
        log.error(f"GET {endpoint} → {resp.status_code}: {resp.text[:200]}")
    else:
        log.info(f"GET {endpoint} → {resp.status_code}")
    return resp.text[:4000]


def tripletex_post(endpoint: str, body_json: str) -> str:
    """Make a POST request to the Tripletex API to create a resource.

    Args:
        endpoint: API endpoint path, e.g. '/employee', '/customer', '/product', '/order', '/invoice'
        body_json: JSON string with the request body, e.g. '{"name": "Test AS", "isCustomer": true}'

    Returns:
        JSON response as string with the created resource (including its id)
    """
    global _call_count, _error_count
    _call_count += 1
    url = f"{_tx_base_url}/{endpoint.lstrip('/')}"
    try:
        data = json.loads(body_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON in body_json"})
    resp = _tx_session.post(url, json=data)
    if resp.status_code >= 400:
        _error_count += 1
        log.error(f"POST {endpoint} → {resp.status_code}: {resp.text[:200]}")
    else:
        log.info(f"POST {endpoint} → {resp.status_code}")
    return resp.text[:4000]


def tripletex_put(endpoint: str, body_json: str = "", query_params: str = "") -> str:
    """Make a PUT request to the Tripletex API to update a resource or execute an action.

    For regular updates: use body_json with the updated fields.
    For actions (endpoints with : like /:payment, /:approve): use query_params instead of body.

    Args:
        endpoint: API endpoint path, e.g. '/employee/123' or '/invoice/456/:payment'
        body_json: JSON string with request body (for regular updates). Leave empty for action endpoints.
        query_params: Query parameters string (for action endpoints like /:payment), e.g. 'paymentDate=2026-03-19&paymentTypeId=123&paidAmount=1250.0'

    Returns:
        JSON response as string
    """
    global _call_count, _error_count
    _call_count += 1
    url = f"{_tx_base_url}/{endpoint.lstrip('/')}"
    params = dict(p.split("=", 1) for p in query_params.split("&") if "=" in p) if query_params else None

    if "/:" in endpoint:
        # Action endpoint — use query params
        resp = _tx_session.put(url, params=params)
    else:
        try:
            data = json.loads(body_json) if body_json else {}
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON in body_json"})
        resp = _tx_session.put(url, json=data, params=params)

    if resp.status_code >= 400:
        _error_count += 1
        log.error(f"PUT {endpoint} → {resp.status_code}: {resp.text[:200]}")
    else:
        log.info(f"PUT {endpoint} → {resp.status_code}")
    return resp.text[:4000] if resp.text else '{"status": "ok"}'


def tripletex_delete(endpoint: str) -> str:
    """Make a DELETE request to the Tripletex API to remove a resource.

    Args:
        endpoint: API endpoint path with ID, e.g. '/travelExpense/123'

    Returns:
        JSON response confirming deletion
    """
    global _call_count, _error_count
    _call_count += 1
    url = f"{_tx_base_url}/{endpoint.lstrip('/')}"
    resp = _tx_session.delete(url)
    if resp.status_code >= 400:
        _error_count += 1
        log.error(f"DELETE {endpoint} → {resp.status_code}: {resp.text[:200]}")
    else:
        log.info(f"DELETE {endpoint} → {resp.status_code}")
    return resp.text[:2000] if resp.text else '{"deleted": true}'


# === SYSTEM PROMPT ===

SYSTEM_PROMPT = """You are a Tripletex accounting API agent. You receive accounting task prompts and must complete them by calling the Tripletex API using the provided tool functions.

CRITICAL RULES:
- Call the tool functions to execute API operations. You have: tripletex_get, tripletex_post, tripletex_put, tripletex_delete
- The Tripletex account starts EMPTY each time. Create all prerequisites before referencing them.
- EXTRACT EVERY DETAIL from the prompt: names, emails, dates, phone numbers, org numbers, addresses, amounts, roles. Missing fields = failed score.
- Dates format: YYYY-MM-DD. Today: 2026-03-19.
- Read API responses carefully. Use actual IDs from responses in subsequent calls.
- If a call fails, read the error message and fix your approach. Do NOT repeat the same failing call.
- MINIMIZE API calls — efficiency is scored. Plan ahead, don't trial-and-error.
- When done, just say "Task completed."

=== API CHEAT SHEET (exact field names) ===

GET /department?fields=id,name&count=1 → ALWAYS call first to get department ID for employees

POST /employee:
  REQUIRED: firstName, lastName, email, userType ("STANDARD"), department ({"id": N})
  OPTIONAL: dateOfBirth ("YYYY-MM-DD"), phoneNumberMobile, phoneNumberHome, phoneNumberWork
  NOTE: userType MUST be "STANDARD" (not "ADMINISTRATOR" — that value is rejected by API)
  For admin role: create with userType="STANDARD", then set admin via PUT /employee/{id} or entitlements

POST /employee/employment:
  REQUIRED: employee ({"id": N}), startDate ("YYYY-MM-DD")
  Employee MUST have dateOfBirth set. Do NOT include employmentType field.

POST /customer:
  REQUIRED: name
  OPTIONAL: email, organizationNumber, isCustomer (true), isSupplier (true for suppliers), phoneNumber, postalAddress ({"addressLine1":"..","postalCode":"..","city":".."})
  For supplier (leverandør/proveedor/fornecedor/fournisseur/Lieferant): isSupplier=true

POST /product:
  REQUIRED: name
  OPTIONAL: number (str), priceExcludingVatCurrency (float), priceIncludingVatCurrency (float), vatType ({"id":3})
  WRONG NAMES: priceExcludingVat → USE priceExcludingVatCurrency
  vatType id 3 = 25% MVA. priceIncludingVatCurrency = priceExcludingVatCurrency × 1.25

POST /order: customer ({"id":N}), deliveryDate, orderDate
POST /order/orderline: order ({"id":N}), product ({"id":N}), count (float)
POST /invoice: invoiceDate, invoiceDueDate, orders ([{"id":N}])
  NOTE: Invoice requires orderlines on the order first. Create order, then orderline, then invoice.

PUT /invoice/{id}/:payment → USE query_params: "paymentDate=YYYY-MM-DD&paymentTypeId=ID&paidAmount=TOTAL_WITH_VAT"
  First GET /invoice/paymentType?fields=id,description&count=5 to find paymentTypeId

POST /project: name, number (str), startDate, projectManager ({"id":N}), optionally customer ({"id":N})
POST /department: name, departmentNumber (int)
POST /travelExpense: employee ({"id":N}), title, travelDetails ({"departureDate":"YYYY-MM-DD","returnDate":"YYYY-MM-DD"})
  NOTE: dates go inside travelDetails object, NOT as flat fields. departureDate/returnDate do NOT exist as top-level fields.
POST /contact: firstName, lastName, customer ({"id":N}), optionally email

For DELETE: first GET to find the ID, then DELETE /endpoint/{id}
For UPDATE: first GET with ?fields=id,...,version, then PUT /endpoint/{id} with version field included

=== COMMON PATTERNS ===
Invoice chain: POST /customer → POST /product → POST /order → POST /order/orderline → POST /invoice
Payment: invoice chain + GET /invoice/paymentType + PUT /invoice/{id}/:payment
Employee with start date: GET /department → POST /employee (with dateOfBirth!) → POST /employee/employment
Project: GET /department → POST /employee → POST /customer → POST /project

=== ERROR RECOVERY ===
- "email already exists" → GET /employee?email=X&fields=id,firstName,lastName to find existing ID. Use that ID.
- "field does not exist" → wrong field name, check cheat sheet above
- "Feltet må fylles ut" → required field missing, add it
- "Produktnummeret er i bruk" → use a different product number
- "Faktura kan ikke opprettes" → likely missing bank account or orderlines. Ensure order has orderlines before creating invoice.
- "projectManager.id Oppgitt ansatt..." → the employee needs the correct access rights. Create employee with userType="STANDARD".
- "Verdien er ikke av korrekt type" for userType → ONLY use "STANDARD". Never "ADMINISTRATOR", "ADMIN", etc.
- "Numm..." for departmentNumber → each department must have a UNIQUE number. Use incrementing numbers (1, 2, 3...).
- NEVER retry with the exact same data that caused an error — always change something.
- GET /invoice REQUIRES invoiceDateFrom and invoiceDateTo as query params."""


# === ENDPOINTS ===

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/solve")
@app.post("/")
async def solve(request: Request):
    global _tx_session, _tx_base_url, _call_count, _error_count
    start_time = time.time()

    body = await request.json()
    prompt = body["prompt"]
    files = body.get("files", [])
    creds = body["tripletex_credentials"]

    log.info(f"{'='*60}")
    log.info(f"PROMPT: {prompt[:500]}")

    # Setup Tripletex session
    _tx_base_url = creds["base_url"].rstrip("/")
    _tx_session = requests.Session()
    _tx_session.auth = ("0", creds["session_token"])
    _tx_session.headers.update({"Content-Type": "application/json", "Accept": "application/json"})
    _call_count = 0
    _error_count = 0

    # Decode vedlegg
    file_contents = []
    for f in files:
        data = base64.b64decode(f["content_base64"])
        filename = f["filename"]
        filepath = Path(f"/tmp/{filename}")
        filepath.write_bytes(data)
        mime = f.get("mime_type", "")

        if mime.startswith("text") or filename.endswith((".csv", ".json", ".txt")):
            try:
                file_contents.append(f"File '{filename}':\n{data.decode('utf-8')[:3000]}")
            except UnicodeDecodeError:
                file_contents.append(f"File '{filename}': [binary, {len(data)} bytes]")
        elif mime == "application/pdf" or filename.endswith(".pdf"):
            # Raw text extraction from PDF
            text = data.decode("latin-1", errors="ignore")
            readable = re.findall(r'[\w\s@.,;:!?/\\()-]{4,}', text)
            extracted = " ".join(readable)[:2000]
            if extracted.strip():
                file_contents.append(f"PDF '{filename}':\n{extracted}")
            else:
                file_contents.append(f"PDF '{filename}': [{len(data)} bytes]")
        elif mime.startswith("image/"):
            file_contents.append(f"Image '{filename}': [{mime}, {len(data)} bytes — use info from prompt text]")
        else:
            file_contents.append(f"File '{filename}': [{mime}, {len(data)} bytes]")

    # Build user message
    user_msg = f"Complete this accounting task:\n\n{prompt}"
    if file_contents:
        user_msg += "\n\nAttached files:\n" + "\n---\n".join(file_contents)

    try:
        # Call Gemini with function calling — it will call Tripletex API directly
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_msg,
            config=GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[tripletex_get, tripletex_post, tripletex_put, tripletex_delete],
                automatic_function_calling=AutomaticFunctionCallingConfig(
                    maximum_remote_calls=25,
                ),
                temperature=0.1,
                max_output_tokens=2048,
            ),
        )
        log.info(f"Gemini response: {response.text[:200] if response.text else 'no text'}")
        log.info(f"API calls: {_call_count}, errors: {_error_count}")

    except Exception as e:
        log.error(f"Agent error: {e}")
        log.error(traceback.format_exc())

    elapsed = time.time() - start_time
    log.info(f"=== DONE in {elapsed:.1f}s (calls={_call_count}, errors={_error_count}) ===")

    # Log submission
    try:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt_preview": prompt[:200],
            "files_count": len(files),
            "api_calls": _call_count,
            "api_errors": _error_count,
            "elapsed_seconds": round(elapsed, 1),
        }
        with open(SUBMISSION_LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

    return JSONResponse({"status": "completed"})


@app.get("/logs")
def get_logs():
    if not SUBMISSION_LOG_PATH.exists():
        return {"logs": [], "count": 0}
    lines = SUBMISSION_LOG_PATH.read_text().strip().split("\n")
    entries = [json.loads(l) for l in lines[-50:] if l.strip()]
    return {"logs": entries, "count": len(entries)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
