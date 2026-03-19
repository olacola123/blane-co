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

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
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
                # PUT with :action (e.g. /:payment, /:approve, /:deliver, /:send) uses query params, not body
                if re.search(r'/:', endpoint):
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

SYSTEM_PROMPT = """You are a Tripletex accounting API agent. Return a JSON array of API calls to execute.

RULES:
- Return ONLY a JSON array. No explanation, no markdown, no code fences.
- Each element: {"method": "GET|POST|PUT|DELETE", "endpoint": "/path", "params": {}, "body": {}}
- Use {prev_N_id} to reference the ID from result N (0-indexed).
- Dates: YYYY-MM-DD. Today: 2026-03-19.
- Account starts EMPTY. Create prerequisites first.
- EXTRACT EVERY DETAIL: names, emails, dates, phones, org numbers, addresses, amounts, roles. Missing = failed.

=== API CHEAT SHEET (exact field names from Tripletex API) ===

POST /employee:
  REQUIRED: firstName(str), lastName(str), email(str), userType("STANDARD"|"ADMINISTRATOR"), department({"id":N})
  OPTIONAL: dateOfBirth("YYYY-MM-DD"), phoneNumberMobile(str), phoneNumberHome(str), phoneNumberWork(str), bankAccountNumber(str), nationalIdentityNumber(str), comments(str), address({"addressLine1":"..","postalCode":"..","city":".."})
  ALWAYS get department first: GET /department?fields=id,name&count=1
  If birth date in prompt → include dateOfBirth
  If start date in prompt → ALSO create POST /employee/employment (see below)

POST /employee/employment:
  REQUIRED: employee({"id":N}), startDate("YYYY-MM-DD")
  OPTIONAL: endDate, employmentId(str)
  NOTE: Employee MUST have dateOfBirth set first. Do NOT include employmentType.

POST /customer:
  REQUIRED: name(str)
  OPTIONAL: email(str), organizationNumber(str), isCustomer(bool), isSupplier(bool), phoneNumber(str), phoneNumberMobile(str), postalAddress({"addressLine1":"..","postalCode":"..","city":".."}), physicalAddress(same), invoiceEmail(str), language(str)
  For customer: isCustomer=true. For supplier: isSupplier=true (same endpoint!)

POST /product:
  REQUIRED: name(str)
  OPTIONAL: number(str), priceExcludingVatCurrency(float), priceIncludingVatCurrency(float), vatType({"id":3}), costExcludingVatCurrency(float), description(str)
  WRONG FIELD NAMES: priceExcludingVat, price → USE priceExcludingVatCurrency
  vatType id 3 = 25% MVA. priceIncludingVat = priceExcluding × 1.25

POST /order:
  REQUIRED: customer({"id":N}), deliveryDate("YYYY-MM-DD"), orderDate("YYYY-MM-DD")
  OPTIONAL: project({"id":N}), department({"id":N}), reference(str), deliveryComment(str)

POST /order/orderline:
  REQUIRED: order({"id":N}), product({"id":N}), count(float)
  OPTIONAL: unitPriceExcludingVatCurrency(float), unitPriceIncludingVatCurrency(float), discount(float), description(str), vatType({"id":N})

POST /invoice:
  REQUIRED: invoiceDate("YYYY-MM-DD"), invoiceDueDate("YYYY-MM-DD"), orders([{"id":N}])
  OPTIONAL: invoiceComment(str)

PUT /invoice/{id}/:payment (query params, NOT body):
  REQUIRED: paymentDate("YYYY-MM-DD"), paymentTypeId(int), paidAmount(float)
  Get paymentTypeId from: GET /invoice/paymentType?fields=id,description&count=5
  paidAmount = total INCLUDING VAT

POST /project:
  REQUIRED: name(str), number(str), startDate("YYYY-MM-DD"), projectManager({"id":N})
  OPTIONAL: customer({"id":N}), department({"id":N}), endDate(str), description(str)

POST /department:
  REQUIRED: name(str), departmentNumber(int)

POST /travelExpense:
  REQUIRED: employee({"id":N}), title(str), departureDate("YYYY-MM-DD"), returnDate("YYYY-MM-DD")
  DELETE: GET /travelExpense first, then DELETE /travelExpense/{id}

POST /contact:
  REQUIRED: firstName(str), lastName(str), customer({"id":N})
  OPTIONAL: email(str), phoneNumberMobile(str)

POST /ledger/voucher:
  For journal entries/corrections

SEARCH:
  GET /customer?name=X&fields=id,name,email&count=100
  GET /employee?firstName=X&fields=id,firstName,lastName,email,version&count=100
  GET /employee?email=X&fields=id,firstName,lastName
  GET /invoice?invoiceDateFrom=X&invoiceDateTo=X&fields=id,amount&count=100

=== TASK PATTERNS ===
- Create employee → GET /department, POST /employee
- Create customer → POST /customer with isCustomer:true
- Create supplier → POST /customer with isSupplier:true
- Create product → POST /product
- Create invoice → POST /customer, POST /product, POST /order, POST /order/orderline, POST /invoice
- Register payment → (create invoice chain), GET /invoice/paymentType, PUT /invoice/{id}/:payment
- Create project → GET /department, POST /employee, POST /customer, POST /project
- Create department(s) → POST /department (one per department)
- Create travel expense → GET /department, POST /employee, POST /travelExpense
- Delete travel expense → GET /travelExpense, DELETE /travelExpense/{id}
- Update employee → GET /employee?fields=id,firstName,lastName,email,dateOfBirth,version, PUT /employee/{id} (MUST include version AND dateOfBirth in body!)
- Create contact → POST /customer, POST /contact

=== ERROR RECOVERY ===
- "email already exists" → GET /employee?email=X to find existing ID. NEVER retry same email.
- "field does not exist" → check cheat sheet for correct field name
- "Feltet må fylles ut" → required field missing, add it
- "Produktnummeret er i bruk" → use a different number

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
        filepath = Path(f"/tmp/{filename}")
        filepath.write_bytes(data)
        mime = f.get("mime_type", "")

        if mime.startswith("text") or filename.endswith((".csv", ".json", ".txt")):
            try:
                file_contents.append(f"File '{filename}':\n{data.decode('utf-8')[:3000]}")
            except UnicodeDecodeError:
                file_contents.append(f"File '{filename}': [binary, {len(data)} bytes]")
        elif mime == "application/pdf" or filename.endswith(".pdf"):
            # Extract text from PDF
            try:
                import subprocess
                result = subprocess.run(
                    ["python3", "-c", f"import fitz; doc=fitz.open('{filepath}'); print('\\n'.join(p.get_text() for p in doc))"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0 and result.stdout.strip():
                    file_contents.append(f"PDF '{filename}':\n{result.stdout[:3000]}")
                else:
                    # Fallback: try raw text extraction
                    text = data.decode("latin-1", errors="ignore")
                    # Extract readable strings from PDF binary
                    readable = re.findall(r'[\w\s@.,;:!?/\\()-]{4,}', text)
                    extracted = " ".join(readable)[:2000]
                    file_contents.append(f"PDF '{filename}' (raw extract):\n{extracted}")
            except Exception:
                file_contents.append(f"PDF '{filename}': [{len(data)} bytes, could not extract text]")
        elif mime.startswith("image/") or filename.endswith((".png", ".jpg", ".jpeg")):
            file_contents.append(f"Image '{filename}': [{mime}, {len(data)} bytes — cannot read image content, use info from prompt text]")
        else:
            file_contents.append(f"File '{filename}': [{mime}, {len(data)} bytes]")

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
            retry_response = ask_gemini_retry(user_prompt, execution_summary)
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
                    final_response = ask_gemini_retry(user_prompt, full_log)
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
