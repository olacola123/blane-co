"""
Tripletex AI Accounting Agent — NM i AI 2026
v10: Hybrid — LLM extracts fields, Python executes deterministic API calls.
Fallback to LLM agent loop for unknown task types.
"""

import base64
import io
import json
import logging
import math
import os
import re
import time
import traceback
from contextvars import ContextVar
from datetime import date, datetime, timedelta, timezone
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

SUBMISSION_LOG_PATH = Path(os.environ.get("SUBMISSION_LOG", "/tmp/tripletex_submissions.jsonl"))
_ctx_var: ContextVar[dict] = ContextVar("ctx")

def get_today():
    return date.today().isoformat()

def get_due_date():
    return (date.today() + timedelta(days=30)).isoformat()


# ═══════════════════════════════════════════════
# API HELPERS — deterministic, no LLM
# ═══════════════════════════════════════════════

def api_call(method: str, endpoint: str, body=None, params=None):
    """Make a Tripletex API call. Returns (status_code, response_dict)."""
    ctx = _ctx_var.get()
    ctx["call_count"] += 1
    session = ctx["session"]
    url = f"{ctx['base_url']}/{endpoint.lstrip('/')}"

    try:
        if method == "GET":
            resp = session.get(url, params=params, timeout=30)
        elif method == "POST":
            resp = session.post(url, json=body, params=params, timeout=30)
        elif method == "PUT":
            if body:
                resp = session.put(url, json=body, params=params, timeout=30)
            else:
                resp = session.put(url, params=params, timeout=30)
        elif method == "DELETE":
            resp = session.delete(url, timeout=30)
        else:
            return 400, {"error": f"Unknown method {method}"}
    except (requests.Timeout, requests.ConnectionError) as e:
        ctx["error_count"] += 1
        log.error(f"{method} {endpoint} → connection error: {e}")
        return 0, {"error": str(e)[:200]}

    ctx["api_trace"].append({
        "method": method,
        "endpoint": endpoint,
        "status": resp.status_code,
        "error": resp.text[:200] if resp.status_code >= 400 else None,
    })

    if resp.status_code >= 400:
        ctx["error_count"] += 1
        log.error(f"{method} {endpoint} → {resp.status_code}: {resp.text[:300]}")
    else:
        log.info(f"{method} {endpoint} → {resp.status_code}")

    try:
        return resp.status_code, resp.json()
    except Exception:
        return resp.status_code, {}


def get_id(resp_data):
    """Extract ID from API response."""
    if isinstance(resp_data, dict):
        val = resp_data.get("value", resp_data)
        if isinstance(val, dict):
            return val.get("id")
    return None


def get_department_id():
    """Get first department ID."""
    status, data = api_call("GET", "/department", params={"fields": "id,name", "count": "1"})
    values = data.get("values", [])
    if values:
        return values[0]["id"]
    return None


def get_company_id():
    """Get company ID — use cached value from token validation if available."""
    ctx = _ctx_var.get()
    if ctx.get("company_id"):
        return ctx["company_id"]
    status, data = api_call("GET", "/token/session/>whoAmI")
    cid = data.get("value", {}).get("company", {}).get("id")
    ctx["company_id"] = cid
    return cid


def vat_id_for_pct(pct):
    """Map VAT percentage to Tripletex vatType ID."""
    if pct is None or pct == 25:
        return 3
    if pct == 15:
        return 31
    if pct == 12:
        return 32
    if pct == 0:
        return 5
    return 3  # default 25%


def price_incl_vat(price_excl, vat_pct):
    """Calculate price including VAT."""
    if vat_pct is None:
        vat_pct = 25
    return round(price_excl * (1 + vat_pct / 100), 2)


# ═══════════════════════════════════════════════
# EXTRACTION PROMPT
# ═══════════════════════════════════════════════

EXTRACTION_PROMPT = """You are a data extraction assistant. Extract ALL fields from the accounting task below into structured JSON.

TASK TYPES (pick the most specific one):
- employee: Create employee (simple)
- employee_start: Create employee with employment start date
- employee_admin: Create employee as administrator
- customer: Create customer
- supplier: Create supplier
- product: Create product
- department: Create department(s)
- contact: Create contact person for a customer
- invoice: Create invoice (possibly with payment)
- credit_note: Create invoice then credit note
- project: Create project (possibly with invoice)
- travel_expense: Create travel expense
- delete_travel: Delete travel expense(s)
- salary: Run payroll / salary
- supplier_invoice: Register incoming supplier invoice
- voucher: Create accounting voucher / journal entry
- accounting_dimension: Create accounting dimensions
- update: Update existing resource
- delete: Delete a resource
- unknown: Doesn't match any above

RESPOND WITH ONLY valid JSON, no other text:
{
  "task_type": "one of the types above",
  "customer_name": "string or null",
  "customer_org_number": "string or null",
  "customer_email": "string or null",
  "customer_phone": "string or null",
  "customer_address_line": "string or null",
  "customer_postal_code": "string or null",
  "customer_city": "string or null",
  "is_supplier": false,
  "employee_first_name": "string or null",
  "employee_last_name": "string or null",
  "employee_email": "string or null",
  "employee_phone": "string or null",
  "employee_date_of_birth": "YYYY-MM-DD or null",
  "employee_start_date": "YYYY-MM-DD or null",
  "is_admin": false,
  "products": [{"name": "str", "number": "str", "price_excl_vat": 0, "vat_pct": 25, "count": 1}],
  "departments": [{"name": "str", "number": "str or null"}],
  "contact_first_name": "string or null",
  "contact_last_name": "string or null",
  "contact_email": "string or null",
  "contact_phone": "string or null",
  "project_name": "string or null",
  "project_number": "string or null",
  "project_fixed_price": null,
  "project_partial_invoice_amount": null,
  "register_payment": false,
  "create_credit_note": false,
  "travel_title": "string or null",
  "travel_departure_date": "YYYY-MM-DD or null",
  "travel_return_date": "YYYY-MM-DD or null",
  "travel_costs": [{"description": "str", "amount": 0}],
  "salary_amount": null,
  "salary_bonus": null,
  "supplier_invoice_number": "string or null",
  "supplier_invoice_amount": null,
  "supplier_invoice_account_number": null,
  "voucher_description": "string or null",
  "voucher_amount": null,
  "voucher_debit_account": null,
  "voucher_credit_account": null,
  "dimension_name": "string or null",
  "dimension_values": [],
  "update_entity_type": "string or null",
  "update_search_term": "string or null",
  "update_fields": {},
  "delete_entity_type": "string or null",
  "delete_search_term": "string or null",
  "invoice_date": "YYYY-MM-DD or null",
  "due_date": "YYYY-MM-DD or null",
  "raw_extra_fields": {}
}

CRITICAL RULES:
- Dates MUST be YYYY-MM-DD with leading zeros. "6. March 1990" → "1990-03-06"
- Product numbers from prompt are STRINGS: "7898" not 7898
- Extract EVERY detail. Missing field = lost points.
- Month names: março=March, setembro=September, agosto=August, outubro=October, avril=April, juin=June, juillet=July, enero=January, febrero=February, mai=May, novembre=November, décembre=December
- If prompt mentions "betaling/payment/Zahlung/pagamento/pago" → set register_payment=true
- If prompt mentions "kreditnota/credit note/Gutschrift" → set create_credit_note=true
- If prompt mentions "administrator/admin/kontoadministrator" → set is_admin=true
- If prompt mentions "startdato/start date/Anfangsdatum/fecha de inicio" → task_type="employee_start"
- If prompt mentions "fastpris/fixed price" AND project → include project_fixed_price
- If prompt mentions "delbetaling/partial" → include project_partial_invoice_amount"""


def extract_fields(prompt: str, file_text: str = "", image_blocks: list = None) -> dict:
    """Use Claude to extract structured fields from the prompt and any images."""
    user_msg = f"TASK:\n{prompt}"
    if file_text:
        user_msg += f"\n\nATTACHED FILES:\n{file_text}"

    # Build content with text + images
    content = [{"type": "text", "text": user_msg}]
    if image_blocks:
        content.extend(image_blocks)

    response = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=[{"type": "text", "text": EXTRACTION_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
    )

    text = "".join(b.text for b in response.content if hasattr(b, "text"))
    # Extract JSON from response
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```\w*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        log.error(f"Failed to parse extraction: {text[:500]}")
        return {"task_type": "unknown"}


# ═══════════════════════════════════════════════
# DETERMINISTIC HANDLERS
# ═══════════════════════════════════════════════

def handle_employee(f):
    dept_id = get_department_id()
    body = {
        "firstName": f.get("employee_first_name", ""),
        "lastName": f.get("employee_last_name", ""),
        "email": f.get("employee_email", ""),
        "userType": "STANDARD",
        "department": {"id": dept_id},
    }
    if f.get("employee_date_of_birth"):
        body["dateOfBirth"] = f["employee_date_of_birth"]
    if f.get("employee_phone"):
        body["phoneNumberMobile"] = f["employee_phone"]
    status, data = api_call("POST", "/employee", body)
    return get_id(data)


def handle_employee_start(f):
    dept_id = get_department_id()
    body = {
        "firstName": f.get("employee_first_name", ""),
        "lastName": f.get("employee_last_name", ""),
        "email": f.get("employee_email", ""),
        "dateOfBirth": f.get("employee_date_of_birth", "1990-01-01"),
        "userType": "STANDARD",
        "department": {"id": dept_id},
    }
    if f.get("employee_phone"):
        body["phoneNumberMobile"] = f["employee_phone"]
    status, data = api_call("POST", "/employee", body)
    emp_id = get_id(data)
    if emp_id:
        api_call("POST", "/employee/employment", {
            "employee": {"id": emp_id},
            "startDate": f.get("employee_start_date", get_today()),
            "employmentType": "ORDINARY",
        })
    return emp_id


def handle_employee_admin(f):
    dept_id = get_department_id()
    body = {
        "firstName": f.get("employee_first_name", ""),
        "lastName": f.get("employee_last_name", ""),
        "email": f.get("employee_email", ""),
        "userType": "EXTENDED",
        "department": {"id": dept_id},
    }
    if f.get("employee_date_of_birth"):
        body["dateOfBirth"] = f["employee_date_of_birth"]
    if f.get("employee_phone"):
        body["phoneNumberMobile"] = f["employee_phone"]
    status, data = api_call("POST", "/employee", body)
    emp_id = get_id(data)
    if emp_id:
        company_id = get_company_id()
        if company_id:
            api_call("POST", "/employee/entitlement", {
                "employee": {"id": emp_id},
                "entitlementId": 1,
                "customer": {"id": company_id},
            })
    return emp_id


def handle_customer(f):
    body = {
        "name": f.get("customer_name", ""),
        "isCustomer": True,
    }
    if f.get("customer_org_number"):
        body["organizationNumber"] = f["customer_org_number"]
    if f.get("customer_email"):
        body["email"] = f["customer_email"]
    if f.get("customer_phone"):
        body["phoneNumber"] = f["customer_phone"]
    if f.get("customer_address_line"):
        body["postalAddress"] = {
            "addressLine1": f["customer_address_line"],
            "postalCode": f.get("customer_postal_code", ""),
            "city": f.get("customer_city", ""),
        }
    if f.get("is_supplier"):
        body["isSupplier"] = True
        body["isCustomer"] = False
    status, data = api_call("POST", "/customer", body)
    return get_id(data)


def handle_supplier(f):
    f["is_supplier"] = True
    return handle_customer(f)


def handle_product(f):
    products = f.get("products", [])
    ids = []
    for p in products:
        vat_pct = p.get("vat_pct", 25)
        price = p.get("price_excl_vat", 0)
        body = {
            "name": p.get("name", "Product"),
            "number": str(p.get("number", str(len(ids) + 1))),
            "priceExcludingVatCurrency": price,
            "priceIncludingVatCurrency": price_incl_vat(price, vat_pct),
            "vatType": {"id": vat_id_for_pct(vat_pct)},
        }
        status, data = api_call("POST", "/product", body)
        pid = get_id(data)
        if pid:
            ids.append(pid)
    return ids


def handle_department(f):
    departments = f.get("departments", [])
    ids = []
    for i, d in enumerate(departments):
        try:
            dept_num = int(d.get("number", 501 + i)) if d.get("number") else 501 + i
        except (ValueError, TypeError):
            dept_num = 501 + i
        body = {
            "name": d.get("name", f"Department {i+1}"),
            "departmentNumber": dept_num,
        }
        status, data = api_call("POST", "/department", body)
        did = get_id(data)
        if did:
            ids.append(did)
    return ids


def handle_contact(f):
    # Create customer first
    cust_id = handle_customer(f)
    if not cust_id:
        return None
    body = {
        "firstName": f.get("contact_first_name", ""),
        "lastName": f.get("contact_last_name", ""),
        "email": f.get("contact_email", ""),
        "customer": {"id": cust_id},
    }
    if f.get("contact_phone"):
        body["phoneNumberMobile"] = f["contact_phone"]
    status, data = api_call("POST", "/contact", body)
    return get_id(data)


def _create_invoice_chain(f):
    """Create customer → products → order → orderlines → invoice. Returns invoice_id and total_incl_vat."""
    # Customer
    cust_id = handle_customer(f)
    if not cust_id:
        return None, 0

    # Products
    products = f.get("products", [])
    product_ids = []
    total_incl = 0
    for p in products:
        vat_pct = p.get("vat_pct", 25)
        price = p.get("price_excl_vat", 0)
        incl = price_incl_vat(price, vat_pct)
        count = p.get("count", 1)
        total_incl += incl * count

        body = {
            "name": p.get("name", "Product"),
            "number": str(p.get("number", str(len(product_ids) + 1))),
            "priceExcludingVatCurrency": price,
            "priceIncludingVatCurrency": incl,
            "vatType": {"id": vat_id_for_pct(vat_pct)},
        }
        status, data = api_call("POST", "/product", body)
        pid = get_id(data)
        if pid:
            product_ids.append((pid, count))

    # Order
    inv_date = f.get("invoice_date") or get_today()
    status, data = api_call("POST", "/order", {
        "customer": {"id": cust_id},
        "deliveryDate": inv_date,
        "orderDate": inv_date,
        "isPrioritizeAmountsIncludingVat": False,
    })
    order_id = get_id(data)
    if not order_id:
        return None, 0

    # Order lines
    for pid, count in product_ids:
        api_call("POST", "/order/orderline", {
            "order": {"id": order_id},
            "product": {"id": pid},
            "count": count,
        })

    # Invoice
    due = f.get("due_date") or get_due_date()
    status, data = api_call("POST", "/invoice",
        body={"invoiceDate": inv_date, "invoiceDueDate": due, "orders": [{"id": order_id}]},
        params={"sendToCustomer": "false"})
    invoice_id = get_id(data)
    return invoice_id, total_incl


def handle_invoice(f):
    invoice_id, total_incl = _create_invoice_chain(f)
    if invoice_id and f.get("register_payment"):
        # Get payment type
        status, data = api_call("GET", "/invoice/paymentType", params={"fields": "id,description", "count": "5"})
        pt_id = None
        for v in data.get("values", []):
            pt_id = v.get("id")
            break
        if pt_id:
            api_call("PUT", f"/invoice/{invoice_id}/:payment", params={
                "paymentDate": get_today(),
                "paymentTypeId": str(pt_id),
                "paidAmount": str(total_incl),
            })
    return invoice_id


def handle_credit_note(f):
    invoice_id, _ = _create_invoice_chain(f)
    if invoice_id:
        api_call("PUT", f"/invoice/{invoice_id}/:createCreditNote", params={
            "date": get_today(),
            "sendToCustomer": "false",
        })
    return invoice_id


def handle_project(f):
    dept_id = get_department_id()

    # Project manager (must be EXTENDED)
    pm_body = {
        "firstName": f.get("employee_first_name", "PM"),
        "lastName": f.get("employee_last_name", "Manager"),
        "email": f.get("employee_email", "pm@example.com"),
        "userType": "EXTENDED",
        "department": {"id": dept_id},
    }
    if f.get("employee_date_of_birth"):
        pm_body["dateOfBirth"] = f["employee_date_of_birth"]
    status, data = api_call("POST", "/employee", pm_body)
    emp_id = get_id(data)

    if not emp_id:
        # Try to find existing employee
        status, data = api_call("GET", "/employee", params={
            "email": f.get("employee_email", ""),
            "fields": "id,version,userType",
        })
        values = data.get("values", [])
        if values:
            emp_id = values[0]["id"]

    if not emp_id:
        return None

    company_id = get_company_id()
    if company_id:
        api_call("POST", "/employee/entitlement", {
            "employee": {"id": emp_id},
            "entitlementId": 45,
            "customer": {"id": company_id},
        })
        api_call("POST", "/employee/entitlement", {
            "employee": {"id": emp_id},
            "entitlementId": 10,
            "customer": {"id": company_id},
        })

    # Customer
    cust_id = handle_customer(f)

    # Project
    proj_num = f.get("project_number") or "1"
    status, data = api_call("POST", "/project", {
        "name": f.get("project_name", "Project"),
        "number": str(proj_num),
        "startDate": get_today(),
        "projectManager": {"id": emp_id},
        "customer": {"id": cust_id},
    })
    proj_id = get_id(data)

    # Retry with different number if conflict
    if not proj_id and status == 422:
        for n in range(2, 10):
            status, data = api_call("POST", "/project", {
                "name": f.get("project_name", "Project"),
                "number": str(n),
                "startDate": get_today(),
                "projectManager": {"id": emp_id},
                "customer": {"id": cust_id},
            })
            proj_id = get_id(data)
            if proj_id:
                break

    # Fixed price + partial invoice
    if proj_id and f.get("project_fixed_price"):
        status, data = api_call("GET", f"/project/{proj_id}", params={"fields": "id,version"})
        version = data.get("value", {}).get("version", 1)
        api_call("PUT", f"/project/{proj_id}", body={
            "name": f.get("project_name", "Project"),
            "number": str(proj_num),
            "startDate": get_today(),
            "isFixedPrice": True,
            "fixedprice": f["project_fixed_price"],
            "version": version,
            "projectManager": {"id": emp_id},
            "customer": {"id": cust_id},
        })

        if f.get("project_partial_invoice_amount"):
            partial = f["project_partial_invoice_amount"]
            status, pdata = api_call("POST", "/product", {
                "name": "Delbetaling",
                "number": "DL01",
                "priceExcludingVatCurrency": partial,
                "priceIncludingVatCurrency": price_incl_vat(partial, 25),
                "vatType": {"id": 3},
            })
            prod_id = get_id(pdata)
            if prod_id:
                status, odata = api_call("POST", "/order", {
                    "customer": {"id": cust_id},
                    "project": {"id": proj_id},
                    "deliveryDate": get_today(),
                    "orderDate": get_today(),
                    "isPrioritizeAmountsIncludingVat": False,
                })
                order_id = get_id(odata)
                if order_id:
                    api_call("POST", "/order/orderline", {
                        "order": {"id": order_id},
                        "product": {"id": prod_id},
                        "count": 1,
                    })
                    api_call("POST", "/invoice",
                        body={"invoiceDate": get_today(), "invoiceDueDate": get_due_date(), "orders": [{"id": order_id}]},
                        params={"sendToCustomer": "false"})

    return proj_id


def handle_travel_expense(f):
    dept_id = get_department_id()
    emp_body = {
        "firstName": f.get("employee_first_name", ""),
        "lastName": f.get("employee_last_name", ""),
        "email": f.get("employee_email", ""),
        "userType": "STANDARD",
        "department": {"id": dept_id},
    }
    if f.get("employee_date_of_birth"):
        emp_body["dateOfBirth"] = f["employee_date_of_birth"]
    status, data = api_call("POST", "/employee", emp_body)
    emp_id = get_id(data)
    if not emp_id:
        status, data = api_call("GET", "/employee", params={"email": f.get("employee_email", ""), "fields": "id"})
        values = data.get("values", [])
        if values:
            emp_id = values[0]["id"]
    if not emp_id:
        return None

    te_body = {
        "employee": {"id": emp_id},
        "title": f.get("travel_title", "Travel"),
        "travelDetails": {
            "departureDate": f.get("travel_departure_date", get_today()),
            "returnDate": f.get("travel_return_date", get_today()),
        },
    }
    status, data = api_call("POST", "/travelExpense", te_body)
    te_id = get_id(data)

    # Add costs
    costs = f.get("travel_costs", [])
    if costs and te_id:
        status, cat_data = api_call("GET", "/travelExpense/costCategory", params={"fields": "id,description", "count": "20"})
        status, pt_data = api_call("GET", "/travelExpense/paymentType", params={"fields": "id,description", "count": "5"})
        cat_id = cat_data.get("values", [{}])[0].get("id") if cat_data.get("values") else None
        pt_id = pt_data.get("values", [{}])[0].get("id") if pt_data.get("values") else None

        if cat_id and pt_id:
            for cost in costs:
                api_call("POST", "/travelExpense/cost", {
                    "travelExpense": {"id": te_id},
                    "costCategory": {"id": cat_id},
                    "paymentType": {"id": pt_id},
                    "amountCurrencyIncVat": cost.get("amount", 0),
                    "currency": {"id": 1},
                })

    return te_id


def handle_delete_travel(f):
    status, data = api_call("GET", "/travelExpense", params={"fields": "id,title", "count": "100"})
    for te in data.get("values", []):
        api_call("DELETE", f"/travelExpense/{te['id']}")


def handle_salary(f):
    dept_id = get_department_id()
    emp_body = {
        "firstName": f.get("employee_first_name", ""),
        "lastName": f.get("employee_last_name", ""),
        "email": f.get("employee_email", ""),
        "dateOfBirth": f.get("employee_date_of_birth", "1990-01-01"),
        "userType": "STANDARD",
        "department": {"id": dept_id},
    }
    status, data = api_call("POST", "/employee", emp_body)
    emp_id = get_id(data)
    if not emp_id:
        status, data = api_call("GET", "/employee", params={"email": f.get("employee_email", ""), "fields": "id"})
        values = data.get("values", [])
        if values:
            emp_id = values[0]["id"]
    if not emp_id:
        return None

    api_call("POST", "/employee/employment", {
        "employee": {"id": emp_id},
        "startDate": "2026-01-01",
        "employmentType": "ORDINARY",
    })

    status, st_data = api_call("GET", "/salary/type", params={"fields": "id,number,name", "count": "50"})
    salary_types = {str(st.get("number", "")): st["id"] for st in st_data.get("values", [])}

    specs = []
    if f.get("salary_amount"):
        fastlonn_id = salary_types.get("2000")
        if fastlonn_id:
            specs.append({"type": {"id": fastlonn_id}, "rate": f["salary_amount"], "count": 1})
    if f.get("salary_bonus"):
        bonus_id = salary_types.get("2002")
        if bonus_id:
            specs.append({"type": {"id": bonus_id}, "rate": f["salary_bonus"], "count": 1})

    if specs:
        api_call("POST", "/salary/transaction", {
            "month": date.today().month,
            "year": date.today().year,
            "payslips": [{"employee": {"id": emp_id}, "specifications": specs}],
        })


def handle_supplier_invoice(f):
    # incomingInvoice is 403 in competition — module not available
    # Still create the supplier so we get partial credit for that
    sup_body = {
        "name": f.get("customer_name", ""),
        "isSupplier": True,
        "isCustomer": False,
    }
    if f.get("customer_org_number"):
        sup_body["organizationNumber"] = f["customer_org_number"]
    if f.get("customer_email"):
        sup_body["email"] = f["customer_email"]
    api_call("POST", "/customer", sup_body)

    # Get supplier ID
    status, data = api_call("GET", "/supplier", params={
        "name": f.get("customer_name", ""),
        "fields": "id,name",
    })
    supplier_id = None
    for v in data.get("values", []):
        supplier_id = v.get("id")
        break

    if not supplier_id:
        return None

    # Get account
    acct_num = f.get("supplier_invoice_account_number")
    acct_id = None
    if acct_num:
        status, data = api_call("GET", "/ledger/account", params={
            "number": str(acct_num),
            "fields": "id,number,name",
            "count": "1",
        })
        values = data.get("values", [])
        if values:
            acct_id = values[0]["id"]

    # Get VAT type
    status, vat_data = api_call("GET", "/ledger/vatType", params={
        "fields": "id,number,name,percentage",
        "count": "50",
    })
    vat_type_id = 1  # default
    for vt in vat_data.get("values", []):
        if vt.get("percentage") == 25:
            vat_type_id = vt["id"]
            break

    amount = f.get("supplier_invoice_amount", 0)
    inv_body = {
        "invoiceHeader": {
            "vendorId": supplier_id,
            "invoiceNumber": f.get("supplier_invoice_number", "INV-001"),
            "invoiceDate": get_today(),
            "dueDate": get_due_date(),
            "invoiceAmount": amount,
            "description": f.get("voucher_description", "Supplier invoice"),
        },
        "orderLines": [{
            "accountId": acct_id,
            "description": "Line 1",
            "amountInclVat": amount,
            "vatTypeId": vat_type_id,
            "externalId": "line1",
            "count": 1,
            "row": 1,
        }],
    }
    status, data = api_call("POST", "/incomingInvoice", inv_body)
    return get_id(data)


def handle_voucher(f):
    debit_acct = f.get("voucher_debit_account")
    credit_acct = f.get("voucher_credit_account")

    # Look up accounts
    debit_id = credit_id = None
    if debit_acct:
        status, data = api_call("GET", "/ledger/account", params={"number": str(debit_acct), "fields": "id,number,name", "count": "1"})
        values = data.get("values", [])
        if values:
            debit_id = values[0]["id"]
    if credit_acct:
        status, data = api_call("GET", "/ledger/account", params={"number": str(credit_acct), "fields": "id,number,name", "count": "1"})
        values = data.get("values", [])
        if values:
            credit_id = values[0]["id"]

    if not debit_id or not credit_id:
        # Get all accounts and use first two
        status, data = api_call("GET", "/ledger/account", params={"fields": "id,number,name", "count": "100"})
        accounts = data.get("values", [])
        if len(accounts) >= 2:
            if not debit_id:
                debit_id = accounts[0]["id"]
            if not credit_id:
                credit_id = accounts[1]["id"]

    amount = f.get("voucher_amount", 0)
    api_call("POST", "/ledger/voucher", {
        "date": get_today(),
        "description": f.get("voucher_description", "Journal entry"),
        "postings": [
            {"row": 1, "date": get_today(), "amountGross": amount, "amountGrossCurrency": amount, "account": {"id": debit_id}, "vatType": {"id": 0}},
            {"row": 2, "date": get_today(), "amountGross": -amount, "amountGrossCurrency": -amount, "account": {"id": credit_id}, "vatType": {"id": 0}},
        ],
    })


def handle_accounting_dimension(f):
    # Create dimension
    status, data = api_call("POST", "/ledger/accountingDimensionName", {
        "dimensionName": f.get("dimension_name", "Dimension"),
        "description": f.get("dimension_name", "Dimension"),
    })
    dim_index = data.get("value", {}).get("dimensionIndex")

    # Create values
    for i, val in enumerate(f.get("dimension_values", [])):
        name = val if isinstance(val, str) else val.get("name", f"Value {i+1}")
        api_call("POST", "/ledger/accountingDimensionValue", {
            "displayName": name,
            "number": i + 1,
            "dimensionIndex": dim_index,
        })


# Handler dispatch
HANDLERS = {
    "employee": handle_employee,
    "employee_start": handle_employee_start,
    "employee_admin": handle_employee_admin,
    "customer": handle_customer,
    "supplier": handle_supplier,
    "product": handle_product,
    "department": handle_department,
    "contact": handle_contact,
    "invoice": handle_invoice,
    "credit_note": handle_credit_note,
    "project": handle_project,
    "travel_expense": handle_travel_expense,
    "delete_travel": handle_delete_travel,
    "salary": handle_salary,
    "supplier_invoice": handle_supplier_invoice,
    "voucher": handle_voucher,
    "accounting_dimension": handle_accounting_dimension,
}


# ═══════════════════════════════════════════════
# LLM FALLBACK (v9 approach for unknown tasks)
# ═══════════════════════════════════════════════

TOOLS = [
    {"name": "tripletex_get", "description": "GET request.", "input_schema": {"type": "object", "properties": {"endpoint": {"type": "string"}, "query_params": {"type": "string", "default": ""}}, "required": ["endpoint"]}},
    {"name": "tripletex_post", "description": "POST to create.", "input_schema": {"type": "object", "properties": {"endpoint": {"type": "string"}, "body": {"type": "object"}, "query_params": {"type": "string", "default": ""}}, "required": ["endpoint", "body"]}},
    {"name": "tripletex_put", "description": "PUT to update/action.", "input_schema": {"type": "object", "properties": {"endpoint": {"type": "string"}, "body": {"type": "object", "default": {}}, "query_params": {"type": "string", "default": ""}}, "required": ["endpoint"]}},
    {"name": "tripletex_delete", "description": "DELETE.", "input_schema": {"type": "object", "properties": {"endpoint": {"type": "string"}}, "required": ["endpoint"]}},
]


def execute_tool(name, input_data):
    endpoint = input_data.get("endpoint", "")
    qp = input_data.get("query_params", "")
    params = dict(p.split("=", 1) for p in qp.split("&") if "=" in p) if qp else None
    body = input_data.get("body")

    if name == "tripletex_get":
        s, d = api_call("GET", endpoint, params=params)
    elif name == "tripletex_post":
        s, d = api_call("POST", endpoint, body=body, params=params)
    elif name == "tripletex_put":
        if "/:" in endpoint:
            s, d = api_call("PUT", endpoint, params=params)
        else:
            s, d = api_call("PUT", endpoint, body=body, params=params)
    elif name == "tripletex_delete":
        s, d = api_call("DELETE", endpoint)
    else:
        return json.dumps({"error": "unknown tool"})

    return json.dumps(d, ensure_ascii=False)[:8000]


def run_llm_fallback(prompt, file_contents, image_blocks, start_time):
    """v9-style LLM agent loop for unknown tasks."""
    system_prompt = build_system_prompt_v9()

    text_msg = f"Complete this accounting task:\n\n{prompt}"
    if file_contents:
        text_msg += "\n\nAttached files:\n" + "\n---\n".join(file_contents)
    content_blocks = [{"type": "text", "text": text_msg}]
    content_blocks.extend(image_blocks)

    messages = [{"role": "user", "content": content_blocks}]
    deadline = start_time + 270

    for iteration in range(20):
        if time.time() > deadline:
            break
        response = claude_client.messages.create(
            model=CLAUDE_MODEL, max_tokens=4096,
            system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
            tools=TOOLS, messages=messages, temperature=0.0,
        )
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            ctx = _ctx_var.get()
            if ctx["error_count"] >= 12:
                break
        else:
            break


# Inline v9 system prompt for fallback
def build_system_prompt_v9():
    today = date.today().isoformat()
    due_date = (date.today() + timedelta(days=30)).isoformat()
    return f"""You are a Tripletex accounting API agent. Complete the task by calling the API.
TODAY: {today}. DEFAULT_DUE_DATE: {due_date}
RULES: Extract ALL fields. Account is EMPTY. Dates YYYY-MM-DD. Use IDs from responses. Fix 4xx errors. Minimize calls. STOP after last step.
USE EXACT VALUES from prompt — never change product numbers, names, or identifiers.

ENDPOINTS:
/employee: POST (firstName, lastName, email, userType:"STANDARD", department.id, dateOfBirth, phoneNumberMobile)
/employee/employment: POST (employee.id, startDate, employmentType:"ORDINARY")
/employee/entitlement: POST (employee.id, entitlementId:1=ADMIN/10=PM/45=CREATE_PROJECT, customer.id)
/customer: POST (name, organizationNumber, email, phoneNumber, isCustomer, isSupplier, postalAddress)
/contact: POST (firstName, lastName, email, phoneNumberMobile, customer.id)
/product: POST (name, number, priceExcludingVatCurrency, priceIncludingVatCurrency, vatType.id) VAT: 3=25%, 31=15%, 32=12%, 5=0%
/department: POST (name, departmentNumber)
/order: POST (customer.id, deliveryDate, orderDate, isPrioritizeAmountsIncludingVat:false)
/order/orderline: POST (order.id, product.id, count)
/invoice: POST (invoiceDate, invoiceDueDate, orders[]) query_params sendToCustomer=false
/invoice/ID/:payment: PUT query_params paymentDate,paymentTypeId,paidAmount
/invoice/ID/:createCreditNote: PUT query_params date,sendToCustomer=false
/invoice/paymentType: GET
/project: POST (name, number, startDate, projectManager.id, customer.id)
/travelExpense: POST (employee.id, title, travelDetails.departureDate/returnDate) DELETE by ID
/travelExpense/cost: POST (travelExpense.id, costCategory.id, paymentType.id, amountCurrencyIncVat, currency.id=1)
/salary/transaction: POST (month, year, payslips[employee.id, specifications[type.id, rate, count]])
/salary/type: GET (2000=Fastlønn, 2002=Bonus)
/supplier: GET
/incomingInvoice: POST (invoiceHeader.vendorId, orderLines with flat accountId,vatTypeId,externalId)
/ledger/voucher: POST (date, description, postings must balance)
/ledger/account: GET
/ledger/vatType: GET
/ledger/accountingDimensionName: POST (dimensionName, description)
/ledger/accountingDimensionValue: POST (displayName, number, dimensionIndex)
/token/session/>whoAmI: GET"""


# ═══════════════════════════════════════════════
# PDF EXTRACTION
# ═══════════════════════════════════════════════

def extract_pdf_text(data: bytes) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
                for table in page.extract_tables():
                    for row in table:
                        if row:
                            pages.append(" | ".join(str(c) for c in row if c))
            if pages:
                return "\n".join(pages)[:6000]
    except Exception as e:
        log.warning(f"pdfplumber: {e}")
    text = data.decode("latin-1", errors="ignore")
    readable = re.findall(r'[\w\s@.,;:!?/\\()-]{4,}', text)
    return " ".join(readable)[:4000]


# ═══════════════════════════════════════════════
# MAIN ENDPOINT
# ═══════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "version": "v10", "model": CLAUDE_MODEL}


@app.post("/solve")
@app.post("/")
async def solve(request: Request):
    start_time = time.time()
    body = await request.json()
    prompt = body["prompt"]
    files = body.get("files", [])
    creds = body["tripletex_credentials"]

    log.info(f"{'='*60}")
    log.info(f"PROMPT: {prompt}")
    log.info(f"FILES: {[f['filename'] for f in files]}")

    # Setup session
    ctx = {
        "base_url": creds["base_url"].rstrip("/"),
        "session": requests.Session(),
        "call_count": 0,
        "error_count": 0,
        "api_trace": [],
    }
    ctx["session"].auth = ("0", creds["session_token"])
    ctx["session"].headers.update({"Content-Type": "application/json", "Accept": "application/json"})
    _ctx_var.set(ctx)

    # Validate token and cache company_id
    try:
        test_resp = ctx["session"].get(f"{ctx['base_url']}/token/session/>whoAmI", timeout=10)
        if test_resp.status_code == 403:
            log.error("TOKEN INVALID")
            _log_run("TOKEN_INVALID", prompt, files, ctx, time.time() - start_time)
            return JSONResponse({"status": "completed"})
        ctx["company_id"] = test_resp.json().get("value", {}).get("company", {}).get("id")
        log.info(f"Token OK. Company: {ctx['company_id']}")
    except Exception as e:
        log.warning(f"Token check: {e}")

    # Process files
    file_contents = []
    image_blocks = []
    for f in files:
        raw_b64 = f["content_base64"]
        data = base64.b64decode(raw_b64)
        filename = f["filename"]
        mime = f.get("mime_type", "")
        if mime.startswith("text") or filename.endswith((".csv", ".json", ".txt")):
            try:
                file_contents.append(f"File '{filename}':\n{data.decode('utf-8')[:8000]}")
            except UnicodeDecodeError:
                file_contents.append(f"File '{filename}': [binary]")
        elif mime == "application/pdf" or filename.endswith(".pdf"):
            extracted = extract_pdf_text(data)
            if extracted.strip():
                file_contents.append(f"PDF '{filename}':\n{extracted}")
        elif mime.startswith("image/"):
            image_blocks.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": raw_b64}})
            file_contents.append(f"[Image '{filename}']")

    file_text = "\n---\n".join(file_contents) if file_contents else ""

    # STEP 1: Extract fields with LLM
    try:
        fields = extract_fields(prompt, file_text, image_blocks)
        task_type = fields.get("task_type", "unknown")
        log.info(f"EXTRACTED: type={task_type}, fields={json.dumps(fields, ensure_ascii=False)[:500]}")
    except Exception as e:
        log.error(f"Extraction failed: {e}")
        task_type = "unknown"
        fields = {}

    # STEP 2: Execute deterministic handler or fallback
    handler = HANDLERS.get(task_type)
    if handler:
        try:
            log.info(f"DETERMINISTIC: {task_type}")
            handler(fields)
        except Exception as e:
            log.error(f"Handler {task_type} failed: {e}")
            log.error(traceback.format_exc())
    else:
        log.info(f"FALLBACK LLM: {task_type}")
        try:
            run_llm_fallback(prompt, file_contents, image_blocks, start_time)
        except Exception as e:
            log.error(f"LLM fallback failed: {e}")
            log.error(traceback.format_exc())

    elapsed = time.time() - start_time
    log.info(f"=== DONE {elapsed:.1f}s type={task_type} calls={ctx['call_count']} errors={ctx['error_count']} ===")
    _log_run(task_type, prompt, files, ctx, elapsed)
    return JSONResponse({"status": "completed"})


def _log_run(task_type, prompt, files, ctx, elapsed):
    try:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": CLAUDE_MODEL, "task_type": task_type,
            "prompt": prompt,
            "files": [f["filename"] for f in files] if isinstance(files, list) and files and isinstance(files[0], dict) else files,
            "api_calls": ctx["call_count"], "api_errors": ctx["error_count"],
            "elapsed_seconds": round(elapsed, 1), "api_trace": ctx["api_trace"],
        }
        with open(SUBMISSION_LOG_PATH, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


@app.get("/logs")
def get_logs():
    if not SUBMISSION_LOG_PATH.exists():
        return {"runs": [], "count": 0}
    lines = SUBMISSION_LOG_PATH.read_text().strip().split("\n")
    entries = [json.loads(l) for l in lines if l.strip()]
    return {"runs": entries[-50:], "count": len(entries)}


@app.get("/runs/{index}")
def get_run(index: int):
    if not SUBMISSION_LOG_PATH.exists():
        return {"error": "No runs"}
    lines = SUBMISSION_LOG_PATH.read_text().strip().split("\n")
    entries = [json.loads(l) for l in lines if l.strip()]
    if 0 <= index < len(entries):
        return entries[index]
    return {"error": f"Run {index} not found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
