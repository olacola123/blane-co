"""
Tripletex AI Accounting Agent — NM i AI 2026
Bruker Gemini via Google GenAI SDK (gratis API).
"""

import base64
import json
import logging
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

# Gemini API direkte (gratis tier)
import os
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
                resp = self._request("POST", endpoint, json=data or {})
            elif method == "PUT":
                resp = self._request("PUT", endpoint, json=data or {})
            elif method == "DELETE":
                resp = self._request("DELETE", endpoint)
            else:
                return {"error": f"Unknown method {method}"}
            if resp.status_code >= 400:
                return {"error": True, "status_code": resp.status_code, "message": resp.text[:500]}
            if resp.status_code == 204:
                return {"deleted": True}
            return resp.json()
        except Exception as e:
            return {"error": True, "message": str(e)}


# === SYSTEM PROMPT ===

SYSTEM_PROMPT = """You are a Tripletex accounting API agent. You receive a task prompt and must return a JSON array of API calls to execute.

RULES:
- Return ONLY a JSON array. No explanation, no markdown code fences.
- Each element: {"method": "GET|POST|PUT|DELETE", "endpoint": "/path", "params": {}, "body": {}}
- Use "params" for GET query parameters, "body" for POST/PUT JSON body.
- Use {prev_N_id} to reference the ID from the Nth result (0-indexed). Example: {prev_0_id}
- Date format: YYYY-MM-DD. Use 2026-03-19 as today.
- The account starts EMPTY. Create prerequisites before referencing them.

IMPORTANT FOR EMPLOYEES:
- POST /employee REQUIRES: firstName, lastName, userType, department.id
- First GET /department to find existing department ID
- userType: "STANDARD" or "ADMINISTRATOR" (for admin role)
- Include ALL fields mentioned in the prompt: dateOfBirth (YYYY-MM-DD), email, phoneNumberMobile, etc.
- For start date, create an employment: POST /employee/employment with {"employee":{"id":N},"startDate":"YYYY-MM-DD","employmentType":{"id":1}}
- After creating employee, if prompt mentions start date, create employment record too

EXTRACT EVERY DETAIL from the prompt. Include dateOfBirth, phone numbers, addresses, roles, etc. Missing fields = failed checks.

COMMON ENDPOINTS:
- GET /department?fields=id,name&count=10 → find departments
- POST /employee → {"firstName":"..","lastName":"..","email":"..","dateOfBirth":"YYYY-MM-DD","userType":"STANDARD","department":{"id": N}}
- POST /customer → {"name":"..","email":"..","isCustomer":true}
- POST /product → {"name":"..","number":"1001"}
- POST /order → {"customer":{"id":N},"deliveryDate":"2026-03-19","orderDate":"2026-03-19"}
- POST /order/orderline → {"order":{"id":N},"product":{"id":N},"count":1}
- POST /invoice → {"invoiceDate":"2026-03-19","invoiceDueDate":"2026-04-19","orders":[{"id":N}]}
- POST /project → {"name":"..","number":"1001","projectManager":{"id":N}}
- POST /department → {"name":"..","departmentNumber":N}
- POST /travelExpense → {"employee":{"id":N},"title":"..","departureDate":"2026-03-19","returnDate":"2026-03-19"}
- DELETE /travelExpense/{id}
- PUT /employee/{id} → update fields (include id and version)
- GET /customer?name=X&fields=id,name,email&count=100
- GET /employee?firstName=X&fields=id,firstName,lastName,email,version&count=100
- POST /contact → {"firstName":"..","lastName":"..","email":"..","customer":{"id":N}}

TASK PATTERNS:
- "Create employee" → 1) GET /department, 2) POST /employee with department.id from step 1
- "Create customer" → POST /customer
- "Create invoice" → 1) POST /customer, 2) POST /product, 3) POST /order with customer id, 4) POST /order/orderline with order+product ids, 5) POST /invoice with order id
- "Delete travel expense" → 1) GET /travelExpense, 2) DELETE /travelExpense/{id}
- "Register payment" → 1) POST /customer, 2) POST /product, 3) POST /order, 4) POST /order/orderline, 5) POST /invoice, 6) POST /invoice/{id}/:createPayment or similar
- "Create project" → 1) GET /employee?count=1, 2) POST /project with projectManager

Return ONLY the JSON array. No other text."""


# === LLM ===

def ask_gemini(prompt: str, file_contents: list[str] | None = None) -> str:
    user_msg = f"Task prompt:\n{prompt}"
    if file_contents:
        user_msg += "\n\nAttached files:\n" + "\n---\n".join(file_contents)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_msg,
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


def execute_plan(tx: TripletexClient, plan: list[dict]) -> list[dict]:
    results = []
    for i, step in enumerate(plan):
        method = step.get("method", "GET").upper()
        endpoint = resolve_refs(step.get("endpoint", ""), results)
        params = resolve_refs(step.get("params", {}), results)
        body = resolve_refs(step.get("body", {}), results)

        log.info(f"Step {i}: {method} {endpoint}")
        result = tx.safe_request(method, str(endpoint), data=body, params=params if params else None)
        results.append(result)

        if isinstance(result, dict) and result.get("error"):
            log.warning(f"Step {i} failed: {result.get('message', '')[:200]}")

    return results


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
    log.info(f"PROMPT: {prompt[:300]}")

    tx = TripletexClient(creds["base_url"], creds["session_token"])

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

    try:
        llm_response = ask_gemini(prompt, file_contents if file_contents else None)
        log.info(f"LLM plan: {llm_response[:500]}")

        plan = parse_plan(llm_response)
        log.info(f"Parsed {len(plan)} steps")

        results = execute_plan(tx, plan)
        log.info(f"Done: {len(results)} steps executed")

    except Exception as e:
        log.error(f"Agent error: {e}")
        log.error(traceback.format_exc())

    return JSONResponse({"status": "completed"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
