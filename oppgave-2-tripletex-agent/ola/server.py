"""
Tripletex AI Accounting Agent — NM i AI 2026
Bruker Gemini via Google GenAI SDK (gratis API).
Med automatisk feilretting og retry.
"""

import base64
import json
import logging
import os
import re
import traceback
from pathlib import Path

import requests
from google import genai
from google.genai.types import GenerateContentConfig
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("agent")

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyC5BmlT5_sMtRgHUsrmb0MwOfEr2UCq6xI")
client = genai.Client(api_key=GEMINI_API_KEY)


# === TRIPLETEX CLIENT ===

class TripletexClient:
    def __init__(self, base_url: str, session_token: str):
        self.base_url = base_url.rstrip("/")
        self.auth = ("0", session_token)
        self.session = requests.Session()
        self.session.auth = self.auth
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    def _request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        resp = self.session.request(method, url, **kwargs)
        if resp.status_code >= 400:
            log.error(f"{method} {endpoint} → {resp.status_code}: {resp.text[:300]}")
        else:
            log.info(f"{method} {endpoint} → {resp.status_code}")
        return resp

    def safe_request(self, method: str, endpoint: str, data=None, params=None) -> dict:
        try:
            if method == "GET":
                resp = self._request("GET", endpoint, params=params)
            elif method == "POST":
                resp = self._request("POST", endpoint, json=data or {}, params=params)
            elif method == "PUT":
                # PUT with :action (e.g. /:payment) uses query params, not body
                if "/:payment" in endpoint or ":approve" in endpoint or ":deliver" in endpoint:
                    # Merge body fields into params for action endpoints
                    action_params = dict(params or {})
                    if data:
                        action_params.update(data)
                    resp = self._request("PUT", endpoint, params=action_params)
                else:
                    resp = self._request("PUT", endpoint, json=data or {}, params=params)
            elif method == "DELETE":
                resp = self._request("DELETE", endpoint)
            else:
                return {"error": True, "message": f"Unknown method {method}"}
            if resp.status_code >= 400:
                return {"error": True, "status_code": resp.status_code, "message": resp.text[:500]}
            if resp.status_code == 204:
                return {"deleted": True}
            return resp.json()
        except Exception as e:
            return {"error": True, "message": str(e)}


# === SYSTEM PROMPT ===

SYSTEM_PROMPT = """You are a Tripletex accounting API agent. You receive a task prompt and must return a JSON array of API calls to execute in order.

CRITICAL RULES:
- Return ONLY a JSON array. No explanation, no markdown, no code fences.
- Each element: {"method": "GET|POST|PUT|DELETE", "endpoint": "/path", "params": {}, "body": {}}
- Use {prev_N_id} to reference the ID from result N (0-indexed).
- Date format: YYYY-MM-DD. Today is 2026-03-19.
- The Tripletex account starts EMPTY each time. Create all prerequisites first.
- EXTRACT EVERY DETAIL from the prompt: names, emails, dates, phone numbers, org numbers, addresses, amounts, roles. Missing fields = failed score.

EMPLOYEE:
- POST /employee REQUIRES: firstName, lastName, userType, department.id
- ALWAYS start with GET /department?fields=id,name&count=1 to get department ID
- userType: "STANDARD" or "ADMINISTRATOR" (for admin/kontoadministrator)
- Optional fields: email, dateOfBirth (YYYY-MM-DD), phoneNumberMobile
- IMPORTANT: If prompt mentions birth date, ALWAYS include dateOfBirth in employee creation
- IMPORTANT: If prompt mentions start date, FIRST create employee WITH dateOfBirth, THEN POST /employee/employment with {"employee":{"id":{prev_N_id}},"startDate":"YYYY-MM-DD"}
  Employment REQUIRES the employee to have dateOfBirth set. Do NOT include employmentType field.

CUSTOMER:
- POST /customer → {"name":"..","email":"..","isCustomer":true}
- Include organizationNumber if given
- Address goes in postalAddress: {"postalAddress":{"addressLine1":"..","postalCode":"..","city":".."}}
- NOT address1/address2 — always use postalAddress object

SUPPLIER (leverandør/proveedor/fornecedor/fournisseur/Lieferant):
- POST /customer → {"name":"..","email":"..","isSupplier":true}
- Same endpoint as customer but with isSupplier:true instead of isCustomer:true

PRODUCT:
- POST /product → {"name":"..","number":"1001"}
- Price field: priceExcludingVatCurrency (NOT priceExcludingVat)
- With 25% VAT: {"priceExcludingVatCurrency":100,"priceIncludingVatCurrency":125,"vatType":{"id":3}}
- vatType id 3 = 25% MVA (standard Norwegian)

ORDER + INVOICE:
- POST /order → {"customer":{"id":N},"deliveryDate":"2026-03-19","orderDate":"2026-03-19"}
- POST /order/orderline → {"order":{"id":N},"product":{"id":N},"count":1}
- POST /invoice → {"invoiceDate":"2026-03-19","invoiceDueDate":"2026-04-19","orders":[{"id":N}]}

PAYMENT:
- First GET /invoice/paymentType?fields=id,description&count=5 to find payment type ID
- Then PUT /invoice/{invoice_id}/:payment?paymentDate=2026-03-19&paymentTypeId=ID&paidAmount=TOTAL_WITH_VAT
- Payment uses PUT with QUERY PARAMS, not body

PROJECT:
- Need projectManager (employee) and optionally customer
- 1) GET /department, 2) POST /employee, 3) POST /customer if needed, 4) POST /project
- POST /project REQUIRES: name, number, startDate, projectManager.id
- Example: {"name":"..","number":"1001","startDate":"2026-03-19","projectManager":{"id":{prev_1_id}}}
- If employee creation fails with "email already exists", you MUST use GET /employee?email=X&fields=id,firstName,lastName to find the existing employee ID. Then use that ID for projectManager.
- NEVER retry creating an employee with the same email — always search for existing one instead.

DEPARTMENT:
- POST /department → {"name":"..","departmentNumber":N}

TRAVEL EXPENSE:
- POST /travelExpense → {"employee":{"id":N},"title":"..","departureDate":"2026-03-19","returnDate":"2026-03-19"}
- DELETE: GET /travelExpense first, then DELETE /travelExpense/{id}

CONTACT:
- POST /contact → {"firstName":"..","lastName":"..","email":"..","customer":{"id":N}}

SEARCH:
- GET /customer?name=X&fields=id,name,email&count=100
- GET /employee?firstName=X&fields=id,firstName,lastName,email,version&count=100

Return ONLY the JSON array."""


RETRY_PROMPT = """The previous API plan had errors. Here is what happened:

Original task: {prompt}

Previous plan and results:
{execution_log}

Fix the plan. Common fixes:
- Field doesn't exist: check the correct field name (e.g. priceExcludingVatCurrency not priceExcludingVat, postalAddress not address1)
- Email already exists: use a different email or GET existing employee
- Validation failed: read the error message for which field is wrong
- 404 Not Found: wrong endpoint path
- ID reference failed: if a previous step failed, the ID is not available — restructure the plan

Return a NEW complete JSON array of API calls to execute from scratch (account is still in current state with some things already created). Only include the remaining/fixed steps."""


# === LLM ===

def ask_gemini(prompt: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1,
            max_output_tokens=4096,
        ),
    )
    return response.text


def ask_gemini_retry(original_prompt: str, execution_log: str) -> str:
    retry_msg = RETRY_PROMPT.format(prompt=original_prompt, execution_log=execution_log)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=retry_msg,
        config=GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1,
            max_output_tokens=4096,
        ),
    )
    return response.text


def parse_plan(llm_text: str) -> list[dict]:
    cleaned = re.sub(r"```json?\s*", "", llm_text)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()
    return json.loads(cleaned)


def resolve_refs(value, results: list[dict]):
    if isinstance(value, str):
        for i, result in enumerate(results):
            placeholder = f"{{prev_{i}_id}}"
            if placeholder in value:
                rid = None
                if isinstance(result, dict):
                    if "value" in result and isinstance(result["value"], dict):
                        rid = result["value"].get("id")
                    elif "values" in result and result["values"]:
                        rid = result["values"][0].get("id")
                    elif "id" in result:
                        rid = result["id"]
                if rid is not None:
                    value = value.replace(placeholder, str(rid))
        try:
            return int(value)
        except (ValueError, TypeError):
            return value
    elif isinstance(value, dict):
        return {k: resolve_refs(v, results) for k, v in value.items()}
    elif isinstance(value, list):
        return [resolve_refs(v, results) for v in value]
    return value


def execute_plan(tx: TripletexClient, plan: list[dict]) -> tuple[list[dict], list[str]]:
    """Utfør plan. Returnerer (results, error_log)."""
    results = []
    error_log = []
    for i, step in enumerate(plan):
        method = step.get("method", "GET").upper()
        endpoint = resolve_refs(step.get("endpoint", ""), results)
        params = resolve_refs(step.get("params", {}), results)
        body = resolve_refs(step.get("body", {}), results)

        log.info(f"Step {i}: {method} {endpoint}")
        result = tx.safe_request(method, str(endpoint), data=body, params=params if params else None)
        results.append(result)

        if isinstance(result, dict) and result.get("error"):
            err_msg = f"Step {i}: {method} {endpoint} → ERROR {result.get('status_code', '?')}: {result.get('message', '')[:300]}"
            error_log.append(err_msg)
            log.warning(err_msg[:200])
        else:
            error_log.append(f"Step {i}: {method} {endpoint} → OK (id={result.get('value', {}).get('id', '?') if isinstance(result, dict) and 'value' in result else '?'})")

    return results, error_log


# === ENDPOINTS ===

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/solve")
@app.post("/")
async def solve(request: Request):
    body = await request.json()
    prompt = body["prompt"]
    files = body.get("files", [])
    creds = body["tripletex_credentials"]

    log.info(f"{'='*60}")
    log.info(f"PROMPT: {prompt[:500]}")

    tx = TripletexClient(creds["base_url"], creds["session_token"])

    # Decode vedlegg
    file_contents = []
    for f in files:
        data = base64.b64decode(f["content_base64"])
        filename = f["filename"]
        Path(f"/tmp/{filename}").write_bytes(data)
        if f.get("mime_type", "").startswith("text") or filename.endswith((".csv", ".json", ".txt")):
            try:
                file_contents.append(f"File '{filename}':\n{data.decode('utf-8')[:3000]}")
            except UnicodeDecodeError:
                file_contents.append(f"File '{filename}': [binary, {len(data)} bytes]")
        else:
            file_contents.append(f"File '{filename}': [{f.get('mime_type', 'unknown')}, {len(data)} bytes]")

    user_prompt = f"Task prompt:\n{prompt}"
    if file_contents:
        user_prompt += "\n\nAttached files:\n" + "\n---\n".join(file_contents)

    try:
        # Round 1: Initial plan
        llm_response = ask_gemini(user_prompt)
        log.info(f"LLM plan: {llm_response[:500]}")
        plan = parse_plan(llm_response)
        log.info(f"Parsed {len(plan)} steps")

        results, error_log = execute_plan(tx, plan)
        has_errors = any("ERROR" in e for e in error_log)

        # Round 2: Automatic retry if errors
        if has_errors:
            log.info("=== RETRY: Fixing errors ===")
            execution_summary = "\n".join(error_log)
            retry_response = ask_gemini_retry(prompt, execution_summary)
            log.info(f"Retry plan: {retry_response[:500]}")

            try:
                retry_plan = parse_plan(retry_response)
                log.info(f"Retry: {len(retry_plan)} steps")
                retry_results, retry_errors = execute_plan(tx, retry_plan)
                results.extend(retry_results)
                error_log.extend(retry_errors)

                # Round 3: One more retry if still errors
                still_errors = any("ERROR" in e for e in retry_errors)
                if still_errors:
                    log.info("=== RETRY 2: Final attempt ===")
                    full_log = "\n".join(error_log)
                    final_response = ask_gemini_retry(prompt, full_log)
                    try:
                        final_plan = parse_plan(final_response)
                        log.info(f"Final retry: {len(final_plan)} steps")
                        execute_plan(tx, final_plan)
                    except Exception:
                        log.warning("Final retry parse failed")
            except Exception:
                log.warning("Retry parse failed")

        log.info(f"=== DONE ===")

    except Exception as e:
        log.error(f"Agent error: {e}")
        log.error(traceback.format_exc())

    return JSONResponse({"status": "completed"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
