"""
Deterministic handler system for Tripletex AI Accounting Agent.
Replaces LLM-decides-API-calls with: Claude extracts data -> Python makes exact API calls.
"""

import json
import logging
import re
import time
from datetime import date, timedelta

log = logging.getLogger("agent")

# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

def _api_post(session, base_url, endpoint, body, trace, params=None):
    url = f"{base_url}/{endpoint.lstrip('/')}"
    try:
        resp = session.post(url, json=body, params=params, timeout=30)
        entry = {"tool": "POST", "endpoint": endpoint, "status": resp.status_code,
                 "error": resp.text[:300] if resp.status_code >= 400 else None}
        trace.append(entry)
        if resp.status_code >= 400:
            log.error(f"POST {endpoint} -> {resp.status_code}: {resp.text[:300]}")
        else:
            log.info(f"POST {endpoint} -> {resp.status_code}")
        try:
            return resp.json(), resp.status_code
        except Exception:
            return {"raw": resp.text[:500]}, resp.status_code
    except Exception as e:
        trace.append({"tool": "POST", "endpoint": endpoint, "status": 0, "error": str(e)[:200]})
        log.error(f"POST {endpoint} error: {e}")
        return {"error": str(e)[:200]}, 0


def _api_get(session, base_url, endpoint, params, trace):
    url = f"{base_url}/{endpoint.lstrip('/')}"
    try:
        resp = session.get(url, params=params, timeout=30)
        entry = {"tool": "GET", "endpoint": endpoint, "status": resp.status_code,
                 "error": resp.text[:300] if resp.status_code >= 400 else None}
        trace.append(entry)
        if resp.status_code < 400:
            log.info(f"GET {endpoint} -> {resp.status_code}")
        else:
            log.error(f"GET {endpoint} -> {resp.status_code}: {resp.text[:300]}")
        try:
            return resp.json(), resp.status_code
        except Exception:
            return {"raw": resp.text[:500]}, resp.status_code
    except Exception as e:
        trace.append({"tool": "GET", "endpoint": endpoint, "status": 0, "error": str(e)[:200]})
        return {"error": str(e)[:200]}, 0


def _api_put(session, base_url, endpoint, trace, body=None, params=None):
    url = f"{base_url}/{endpoint.lstrip('/')}"
    try:
        if body and body != {}:
            resp = session.put(url, json=body, params=params, timeout=30)
        else:
            resp = session.put(url, params=params, timeout=30)
        entry = {"tool": "PUT", "endpoint": endpoint, "status": resp.status_code,
                 "error": resp.text[:300] if resp.status_code >= 400 else None}
        trace.append(entry)
        if resp.status_code >= 400:
            log.error(f"PUT {endpoint} -> {resp.status_code}: {resp.text[:300]}")
        else:
            log.info(f"PUT {endpoint} -> {resp.status_code}")
        try:
            return resp.json(), resp.status_code
        except Exception:
            return {"raw": resp.text[:500]}, resp.status_code
    except Exception as e:
        trace.append({"tool": "PUT", "endpoint": endpoint, "status": 0, "error": str(e)[:200]})
        return {"error": str(e)[:200]}, 0


def _api_delete(session, base_url, endpoint, trace):
    url = f"{base_url}/{endpoint.lstrip('/')}"
    try:
        resp = session.delete(url, timeout=30)
        entry = {"tool": "DELETE", "endpoint": endpoint, "status": resp.status_code,
                 "error": resp.text[:300] if resp.status_code >= 400 else None}
        trace.append(entry)
        return resp.status_code
    except Exception as e:
        trace.append({"tool": "DELETE", "endpoint": endpoint, "status": 0, "error": str(e)[:200]})
        return 0


def _val(d, key, default=None):
    """Get value from dict, return default if missing or empty string."""
    v = d.get(key, default)
    if v == "" or v is None:
        return default
    return v


def _today():
    return date.today().isoformat()


def _due(days=30):
    return (date.today() + timedelta(days=days)).isoformat()


def _find_employee_by_email(ctx, email):
    """Find existing employee by email in prefetched context."""
    if not email or not ctx.get("employees"):
        return None
    email_lower = email.lower().strip()
    for e in ctx["employees"]:
        if str(e.get("email", "")).lower().strip() == email_lower:
            return e
    return None


def _find_customer_by_name(ctx, name):
    """Find existing customer by name in prefetched context."""
    if not name or not ctx.get("customers"):
        return None
    name_lower = name.lower().strip()
    for c in ctx["customers"]:
        if str(c.get("name", "")).lower().strip() == name_lower:
            return c
    return None


def _find_supplier_by_name(ctx, name):
    """Find existing supplier by name in prefetched context."""
    if not name or not ctx.get("suppliers"):
        return None
    name_lower = name.lower().strip()
    for s in ctx["suppliers"]:
        if str(s.get("name", "")).lower().strip() == name_lower:
            return s
    return None


def _get_acct_id(ctx, acct_num):
    """Get ledger account ID from prefetched context."""
    return ctx.get("ledger_accounts", {}).get(str(acct_num))


def _get_payment_type_id(ctx):
    """Get first available payment type ID."""
    pts = ctx.get("payment_types", [])
    if pts:
        return pts[0]["id"]
    return None


def _parse_amount(val):
    """Parse amount from various formats to float."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        # Remove currency symbols and whitespace
        cleaned = re.sub(r'[^\d,.\-]', '', val.strip())
        # Handle comma as decimal separator
        if ',' in cleaned and '.' in cleaned:
            # 1.234,56 format
            cleaned = cleaned.replace('.', '').replace(',', '.')
        elif ',' in cleaned:
            # Could be 1234,56 or 1,234
            parts = cleaned.split(',')
            if len(parts[-1]) == 2:
                cleaned = cleaned.replace(',', '.')
            else:
                cleaned = cleaned.replace(',', '')
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0


# ═══════════════════════════════════════════════════════════
# EXTRACTION PROMPTS PER TASK TYPE
# ═══════════════════════════════════════════════════════════

EXTRACTION_PROMPTS = {
    "customer": """Extract from this accounting task. Return ONLY valid JSON:
{"name": "company name", "orgNumber": "org number digits only or empty string", "email": "email", "phone": "phone or empty string", "addressLine1": "street address", "postalCode": "postal code", "city": "city name"}""",

    "supplier": """Extract from this accounting task. Return ONLY valid JSON:
{"name": "supplier company name", "orgNumber": "org number digits only or empty string", "email": "email or empty string", "phone": "phone or empty string", "addressLine1": "street address", "postalCode": "postal code", "city": "city name"}""",

    "departments": """Extract from this accounting task. Return ONLY valid JSON:
{"departments": [{"name": "department name", "number": "department number as string"}]}
List ALL departments to create.""",

    "product": """Extract from this accounting task. Return ONLY valid JSON:
{"name": "product name", "number": "product number as string", "priceExVat": 0.00, "priceInclVat": 0.00, "vatPercent": 25, "description": "description or empty string"}
Prices are EXCLUDING VAT unless explicitly stated "inkl. MVA"/"TTC"/"con IVA incluido"/"inkl. MwSt".""",

    "payment": """Extract from this accounting task. Return ONLY valid JSON:
{"customerName": "customer name or empty", "invoiceNumber": "invoice number if mentioned or empty", "amount": 0.00, "paymentDate": "YYYY-MM-DD"}
Amount should be the full invoice amount INCLUDING VAT.""",

    "reverse_payment": """Extract from this accounting task. Return ONLY valid JSON:
{"customerName": "customer name or empty", "invoiceNumber": "invoice number if mentioned or empty", "amount": 0.00, "paymentDate": "YYYY-MM-DD", "reason": "reason for reversal"}
Amount is POSITIVE (will be negated in the API call).""",

    "contact_person": """Extract from this accounting task. Return ONLY valid JSON:
{"customerName": "company name the contact belongs to", "firstName": "contact first name", "lastName": "contact last name", "email": "email or empty string", "phone": "phone or empty string"}""",

    "acct_dimension": """Extract from this accounting task. Return ONLY valid JSON:
{"dimensionName": "name of the accounting dimension", "description": "description", "values": [{"name": "value name", "number": "value number as string"}], "voucherDate": "YYYY-MM-DD or empty", "voucherDescription": "voucher description or empty", "accountNumber": "ledger account number or empty", "amount": 0.00}
List ALL dimension values to create.""",

    "invoice_send": """Extract from this accounting task. Return ONLY valid JSON:
{"customerName": "customer company name", "customerOrgNumber": "org number or empty", "customerEmail": "email", "customerPhone": "phone or empty", "customerAddress": "street address or empty", "customerPostalCode": "postal code or empty", "customerCity": "city or empty", "productName": "product name", "productNumber": "product number as string", "priceExVat": 0.00, "vatPercent": 25, "quantity": 1, "invoiceDate": "YYYY-MM-DD", "dueDate": "YYYY-MM-DD", "description": "product description or empty"}
Prices EXCLUDING VAT unless explicitly stated otherwise.""",

    "invoice_multi": """Extract from this accounting task. Return ONLY valid JSON:
{"customerName": "customer company name", "customerOrgNumber": "org number or empty", "customerEmail": "email", "customerPhone": "phone or empty", "customerAddress": "street address or empty", "customerPostalCode": "postal code or empty", "customerCity": "city or empty", "products": [{"name": "product name", "number": "product number as string", "priceExVat": 0.00, "vatPercent": 25, "quantity": 1}], "invoiceDate": "YYYY-MM-DD", "dueDate": "YYYY-MM-DD"}
Prices EXCLUDING VAT unless explicitly stated otherwise. vatPercent: 25 for standard, 15 for food, 0 for exempt.""",

    "order": """Extract from this accounting task. Return ONLY valid JSON:
{"customerName": "customer company name", "customerOrgNumber": "org number or empty", "customerEmail": "email", "customerPhone": "phone or empty", "customerAddress": "street address or empty", "customerPostalCode": "postal code or empty", "customerCity": "city or empty", "products": [{"name": "product name", "number": "product number as string", "priceExVat": 0.00, "vatPercent": 25, "quantity": 1}], "orderDate": "YYYY-MM-DD", "deliveryDate": "YYYY-MM-DD", "payFull": true}
payFull: true if the order should be invoiced and paid immediately.""",

    "credit_note": """Extract from this accounting task. Return ONLY valid JSON:
{"customerName": "customer company name", "customerOrgNumber": "org number or empty", "customerEmail": "email", "customerPhone": "phone or empty", "customerAddress": "street address or empty", "customerPostalCode": "postal code or empty", "customerCity": "city or empty", "productName": "product name", "productNumber": "product number as string", "priceExVat": 0.00, "vatPercent": 25, "quantity": 1, "invoiceDate": "YYYY-MM-DD", "dueDate": "YYYY-MM-DD", "creditNoteDate": "YYYY-MM-DD", "reason": "reason for credit note"}""",

    "employee": """Extract from this accounting task. Return ONLY valid JSON:
{"firstName": "first name", "lastName": "last name", "email": "email", "dateOfBirth": "YYYY-MM-DD", "employeeNumber": "employee number as string or empty", "startDate": "YYYY-MM-DD or empty", "userType": "STANDARD or EXTENDED or NO_ACCESS", "addressLine1": "street address or empty", "postalCode": "postal code or empty", "city": "city or empty", "nationalIdentityNumber": "11-digit national ID or empty"}
userType defaults to STANDARD. startDate is employment start date.""",

    "supplier_invoice": """Extract from this accounting task. Return ONLY valid JSON:
{"supplierName": "supplier company name", "supplierOrgNumber": "org number or empty", "supplierEmail": "email or empty", "supplierPhone": "phone or empty", "supplierAddress": "street address or empty", "supplierPostalCode": "postal code or empty", "supplierCity": "city or empty", "invoiceNumber": "invoice number", "invoiceDate": "YYYY-MM-DD", "amountInclVat": 0.00, "amountExVat": 0.00, "vatPercent": 25, "description": "what was purchased", "expenseAccount": "expense account number like 6500", "isFood": false}
isFood: true if food/beverage (15% VAT). expenseAccount: 6540=office, 6340=IT, 6500=supplies, 7300=service, 7100=vehicle, 7140=travel, 6800=other.""",

    "receipt_voucher": """Extract from this receipt/accounting task. Return ONLY valid JSON:
{"description": "what was purchased", "amountInclVat": 0.00, "vatPercent": 25, "receiptDate": "YYYY-MM-DD", "expenseAccount": "expense account number", "isFood": false, "departmentName": "department name from prompt or empty string"}
expenseAccount: 6500=office supplies, 7350=entertainment, 7100=vehicle, 7140=travel, 6900=phone, 6300=rent, 6340=utilities, 7700=operations, 6800=other.
isFood: true if food (15% VAT instead of 25%).""",

    "salary": """Extract from this accounting task. Return ONLY valid JSON:
{"employeeEmail": "email to find employee", "baseSalary": 0.00, "bonus": 0.00, "salaryDate": "YYYY-MM-DD or empty", "month": 0, "year": 0}
baseSalary is the monthly base salary. bonus is additional bonus amount (0 if none).""",

    "year_end": """Extract from this accounting task. Return ONLY valid JSON:
{"assets": [{"name": "asset description", "accountNumber": "asset account like 1200", "depreciationAccountNumber": "depreciation expense account like 6010", "accumulatedDepAccountNumber": "accumulated depreciation account like 1209", "originalCost": 0.00, "usefulLifeYears": 0, "depreciationAmount": 0.00}],
"prepaidExpenses": [{"description": "prepaid item", "amount": 0.00, "expenseAccount": "expense account number", "prepaidAccount": "1700 or 1710"}],
"salaryAccrual": {"amount": 0.00, "description": "salary accrual description or empty"},
"taxRate": 0.22,
"voucherDate": "YYYY-MM-DD"}
depreciationAmount: if not given, calculate as originalCost / usefulLifeYears.""",

    "cost_analysis": """Extract from this accounting task. Return ONLY valid JSON:
{"employeeEmail": "email of employee to assign as project manager", "month1Start": "YYYY-MM-DD", "month1End": "YYYY-MM-DD", "month2Start": "YYYY-MM-DD", "month2End": "YYYY-MM-DD", "numAccountsToFind": 3}
month1 is the earlier month, month2 is the later month to compare costs.""",

    "project": """Extract from this accounting task. Return ONLY valid JSON:
{"projectName": "project name", "customerName": "customer company name or empty", "customerOrgNumber": "org number or empty", "customerEmail": "email or empty", "startDate": "YYYY-MM-DD", "endDate": "YYYY-MM-DD or empty", "description": "project description or empty", "employeeEmail": "project manager email or empty"}""",

    "project_fixed": """Extract from this accounting task. Return ONLY valid JSON:
{"projectName": "project name", "customerName": "customer company name or empty", "customerOrgNumber": "org number or empty", "customerEmail": "email or empty", "startDate": "YYYY-MM-DD", "endDate": "YYYY-MM-DD or empty", "fixedPrice": 0.00, "description": "project description or empty", "employeeEmail": "project manager email or empty", "partialInvoicePercent": 0}
partialInvoicePercent: percentage for partial invoice (0 if none mentioned).""",

    "timesheet": """Extract from this accounting task. Return ONLY valid JSON:
{"projectName": "project name", "customerName": "customer company name or empty", "customerOrgNumber": "org number or empty", "customerEmail": "email or empty", "employeeEmail": "employee email", "hours": 0.0, "hourlyRate": 0.00, "date": "YYYY-MM-DD", "description": "activity description or empty"}""",
}


# ═══════════════════════════════════════════════════════════
# EXTRACT FUNCTION
# ═══════════════════════════════════════════════════════════

def extract_data(claude_client, model, prompt: str, files: list, task_type: str, image_blocks: list = None) -> dict:
    """Make ONE Claude call with a task-specific extraction prompt. Returns structured dict."""
    extraction_prompt = EXTRACTION_PROMPTS.get(task_type)
    if not extraction_prompt:
        extraction_prompt = f"Extract all relevant data from this accounting task. Return ONLY valid JSON with all fields needed."

    # Build content blocks
    content = []
    if image_blocks:
        content.extend(image_blocks)

    file_text = ""
    if files:
        for f in files:
            import base64, io
            data = base64.b64decode(f["content_base64"])
            filename = f["filename"]
            mime = f.get("mime_type", "")
            if mime.startswith("text") or filename.endswith((".csv", ".json", ".txt")):
                try:
                    file_text += f"\nFile '{filename}':\n{data.decode('utf-8')[:8000]}"
                except UnicodeDecodeError:
                    pass
            elif mime == "application/pdf" or filename.endswith(".pdf"):
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
                            file_text += f"\nPDF '{filename}':\n" + "\n".join(parts)[:6000]
                except Exception as e:
                    log.warning(f"PDF extraction failed: {e}")
            elif mime.startswith("image/"):
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime, "data": f["content_base64"]},
                })

    user_text = f"""{extraction_prompt}

TASK:
{prompt}"""
    if file_text:
        user_text += f"\n\nATTACHED FILES:\n{file_text}"

    content.append({"type": "text", "text": user_text})

    try:
        response = claude_client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
        )
        text = response.content[0].text.strip()
        # Extract JSON from response (handle markdown code blocks)
        if "```" in text:
            match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
            if match:
                text = match.group(1).strip()
        data = json.loads(text)
        log.info(f"Extracted data for {task_type}: {json.dumps(data, ensure_ascii=False)[:500]}")
        return data
    except json.JSONDecodeError as e:
        log.error(f"JSON parse error in extraction: {e}, text: {text[:300]}")
        # Try to find JSON in the text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return {}
    except Exception as e:
        log.error(f"Extraction failed: {e}")
        return {}


# ═══════════════════════════════════════════════════════════
# HANDLER: CUSTOMER
# ═══════════════════════════════════════════════════════════

def handle_customer(data, session, base_url, ctx):
    trace = []
    body = {
        "name": data.get("name", ""),
        "email": _val(data, "email", ""),
        "phoneNumber": _val(data, "phone", ""),
        "isCustomer": True,
        "language": "NO",
        "postalAddress": {
            "addressLine1": _val(data, "addressLine1", ""),
            "postalCode": _val(data, "postalCode", ""),
            "city": _val(data, "city", ""),
        },
    }
    org = _val(data, "orgNumber", "")
    if org:
        body["organizationNumber"] = org.replace(" ", "")
    if body.get("email"):
        body["invoiceEmail"] = body["email"]

    # Check if customer already exists
    existing = _find_customer_by_name(ctx, data.get("name"))
    if existing:
        log.info(f"Customer already exists: id={existing['id']}")
        return trace

    _api_post(session, base_url, "/customer", body, trace)
    return trace


# ═══════════════════════════════════════════════════════════
# HANDLER: SUPPLIER
# ═══════════════════════════════════════════════════════════

def handle_supplier(data, session, base_url, ctx):
    trace = []
    body = {
        "name": data.get("name", ""),
        "email": _val(data, "email", ""),
        "phoneNumber": _val(data, "phone", ""),
        "isCustomer": False,
        "isSupplier": True,
        "supplierNumber": 0,
        "language": "NO",
        "postalAddress": {
            "addressLine1": _val(data, "addressLine1", ""),
            "postalCode": _val(data, "postalCode", ""),
            "city": _val(data, "city", ""),
        },
    }
    org = _val(data, "orgNumber", "")
    if org:
        body["organizationNumber"] = org.replace(" ", "")
    if not body.get("email"):
        safe_name = re.sub(r'[^a-z]', '', data.get("name", "supplier").lower())[:20]
        body["email"] = f"faktura@{safe_name}.no"
    body["invoiceEmail"] = body["email"]

    # Account manager
    if ctx.get("employees"):
        body["accountManager"] = {"id": ctx["employees"][0]["id"]}

    existing = _find_supplier_by_name(ctx, data.get("name"))
    if existing:
        log.info(f"Supplier already exists: id={existing['id']}")
        return trace

    _api_post(session, base_url, "/customer", body, trace)
    return trace


# ═══════════════════════════════════════════════════════════
# HANDLER: DEPARTMENTS
# ═══════════════════════════════════════════════════════════

def handle_departments(data, session, base_url, ctx):
    trace = []
    departments = data.get("departments", [])
    existing_numbers = set()
    for d in ctx.get("departments", []):
        num = d.get("departmentNumber")
        if num:
            existing_numbers.add(str(num))

    for dept in departments:
        num = str(dept.get("number", ""))
        # If number conflicts, pick next available
        if num in existing_numbers:
            n = int(num) if num.isdigit() else 1
            while str(n) in existing_numbers:
                n += 1
            num = str(n)

        body = {"name": dept["name"], "departmentNumber": num}
        _api_post(session, base_url, "/department", body, trace)
        existing_numbers.add(num)
    return trace


# ═══════════════════════════════════════════════════════════
# HANDLER: PRODUCT
# ═══════════════════════════════════════════════════════════

def handle_product(data, session, base_url, ctx):
    trace = []
    vat_pct = data.get("vatPercent", 25)
    vat_type_id = {25: 3, 15: 31, 12: 32, 0: 5}.get(vat_pct, 3)
    price_ex = _parse_amount(data.get("priceExVat", 0))
    price_incl = _parse_amount(data.get("priceInclVat", 0))
    if price_incl and not price_ex:
        price_ex = round(price_incl / (1 + vat_pct / 100), 2)
    elif price_ex and not price_incl:
        price_incl = round(price_ex * (1 + vat_pct / 100), 2)

    body = {
        "name": data.get("name", ""),
        "number": str(data.get("number", "1")),
        "priceExcludingVatCurrency": price_ex,
        "priceIncludingVatCurrency": price_incl,
        "vatType": {"id": vat_type_id},
    }
    desc = _val(data, "description")
    if desc:
        body["description"] = desc

    _api_post(session, base_url, "/product", body, trace)
    return trace


# ═══════════════════════════════════════════════════════════
# HANDLER: PAYMENT
# ═══════════════════════════════════════════════════════════

def handle_payment(data, session, base_url, ctx):
    trace = []
    # Step 1: GET invoices
    result, status = _api_get(session, base_url, "/invoice", {
        "invoiceDateFrom": "2020-01-01",
        "invoiceDateTo": "2027-01-01",
        "fields": "id,invoiceNumber,amount,amountOutstanding,customer",
    }, trace)

    invoices = result.get("values", [])
    if not invoices:
        log.error("No invoices found")
        return trace

    # Find the right invoice
    target = None
    inv_num = _val(data, "invoiceNumber", "")
    cust_name = _val(data, "customerName", "")
    amount = _parse_amount(data.get("amount", 0))

    for inv in invoices:
        if inv_num and str(inv.get("invoiceNumber")) == str(inv_num):
            target = inv
            break
    if not target and cust_name:
        cust_lower = cust_name.lower()
        for inv in invoices:
            c = inv.get("customer", {})
            if cust_lower in str(c.get("name", "")).lower():
                target = inv
                break
    if not target and amount:
        for inv in invoices:
            if abs(float(inv.get("amountOutstanding", 0)) - amount) < 1.0:
                target = inv
                break
    if not target and invoices:
        # Pick first with outstanding amount
        for inv in invoices:
            if float(inv.get("amountOutstanding", 0)) > 0:
                target = inv
                break
        if not target:
            target = invoices[0]

    # Step 2: PUT /:payment
    inv_id = target["id"]
    pay_amount = amount if amount else float(target.get("amountOutstanding", target.get("amount", 0)))
    pt_id = _get_payment_type_id(ctx)
    pay_date = _val(data, "paymentDate", _today())

    params = {
        "paymentDate": pay_date,
        "paymentTypeId": str(pt_id) if pt_id else "1",
        "paidAmount": str(pay_amount),
    }
    _api_put(session, base_url, f"/invoice/{inv_id}/:payment", trace, params=params)
    return trace


# ═══════════════════════════════════════════════════════════
# HANDLER: REVERSE PAYMENT
# ═══════════════════════════════════════════════════════════

def handle_reverse_payment(data, session, base_url, ctx):
    trace = []
    # Step 1: GET invoices
    result, status = _api_get(session, base_url, "/invoice", {
        "invoiceDateFrom": "2020-01-01",
        "invoiceDateTo": "2027-01-01",
        "fields": "id,invoiceNumber,amount,amountOutstanding,customer",
    }, trace)

    invoices = result.get("values", [])
    if not invoices:
        return trace

    # Find the right invoice (same logic as payment)
    target = None
    inv_num = _val(data, "invoiceNumber", "")
    cust_name = _val(data, "customerName", "")
    amount = _parse_amount(data.get("amount", 0))

    for inv in invoices:
        if inv_num and str(inv.get("invoiceNumber")) == str(inv_num):
            target = inv
            break
    if not target and cust_name:
        cust_lower = cust_name.lower()
        for inv in invoices:
            c = inv.get("customer", {})
            if cust_lower in str(c.get("name", "")).lower():
                target = inv
                break
    if not target and invoices:
        target = invoices[0]

    if not target:
        return trace

    inv_id = target["id"]
    pay_amount = amount if amount else abs(float(target.get("amount", 0)))
    pt_id = _get_payment_type_id(ctx)
    pay_date = _val(data, "paymentDate", _today())

    # Negative amount for reversal
    params = {
        "paymentDate": pay_date,
        "paymentTypeId": str(pt_id) if pt_id else "1",
        "paidAmount": str(-pay_amount),
    }
    _api_put(session, base_url, f"/invoice/{inv_id}/:payment", trace, params=params)
    return trace


# ═══════════════════════════════════════════════════════════
# HANDLER: CONTACT PERSON
# ═══════════════════════════════════════════════════════════

def handle_contact_person(data, session, base_url, ctx):
    trace = []
    cust_name = data.get("customerName", "")
    cust_id = None

    # Find existing customer
    existing = _find_customer_by_name(ctx, cust_name)
    if existing:
        cust_id = existing["id"]
    else:
        # Search via API
        result, status = _api_get(session, base_url, "/customer", {
            "fields": "id,name", "count": "100"
        }, trace)
        for c in result.get("values", []):
            if cust_name.lower() in str(c.get("name", "")).lower():
                cust_id = c["id"]
                break
        if not cust_id:
            # Create customer
            cust_body = {
                "name": cust_name,
                "email": data.get("email", f"info@{re.sub(r'[^a-z]', '', cust_name.lower())[:15]}.no"),
                "phoneNumber": "",
                "isCustomer": True,
                "language": "NO",
            }
            result, status = _api_post(session, base_url, "/customer", cust_body, trace)
            if status == 201:
                cust_id = result.get("value", {}).get("id")

    if not cust_id:
        log.error("Could not find or create customer for contact person")
        return trace

    # Create contact
    contact_body = {
        "firstName": data.get("firstName", ""),
        "lastName": data.get("lastName", ""),
        "email": _val(data, "email", ""),
        "customer": {"id": cust_id},
    }
    phone = _val(data, "phone")
    if phone:
        contact_body["phoneNumberMobile"] = phone

    _api_post(session, base_url, "/contact", contact_body, trace)
    return trace


# ═══════════════════════════════════════════════════════════
# HANDLER: ACCOUNTING DIMENSION
# ═══════════════════════════════════════════════════════════

def handle_acct_dimension(data, session, base_url, ctx):
    trace = []

    # Step 1: Create dimension name
    dim_body = {
        "dimensionName": data.get("dimensionName", ""),
        "description": _val(data, "description", data.get("dimensionName", "")),
    }
    result, status = _api_post(session, base_url, "/ledger/accountingDimensionName", dim_body, trace)
    dim_index = None
    if status == 201:
        dim_index = result.get("value", {}).get("dimensionIndex")

    if not dim_index:
        # Try to get the dimension index from listing
        result2, _ = _api_get(session, base_url, "/ledger/accountingDimensionName", {"fields": "id,dimensionName,dimensionIndex"}, trace)
        for d in result2.get("values", []):
            if d.get("dimensionName") == data.get("dimensionName"):
                dim_index = d.get("dimensionIndex")
                break
        if not dim_index:
            dim_index = 1  # fallback

    # Step 2: Create dimension values
    value_ids = []
    for val in data.get("values", []):
        val_body = {
            "displayName": val.get("name", ""),
            "number": str(val.get("number", "")),
            "dimensionIndex": dim_index,
        }
        result, status = _api_post(session, base_url, "/ledger/accountingDimensionValue", val_body, trace)
        if status == 201:
            value_ids.append(result.get("value", {}).get("id"))

    # Step 3: Create voucher linking dimension if amount and account specified
    acct_num = _val(data, "accountNumber")
    amount = _parse_amount(data.get("amount", 0))
    if acct_num and amount and value_ids:
        acct_id = _get_acct_id(ctx, acct_num)
        bank_id = _get_acct_id(ctx, "1920")
        if acct_id and bank_id:
            voucher_date = _val(data, "voucherDate", _today())
            dim_field = f"freeAccountingDimension{dim_index}" if dim_index and dim_index <= 3 else "freeAccountingDimension1"
            postings = [
                {
                    "account": {"id": acct_id},
                    "amountGross": amount,
                    "amountGrossCurrency": amount,
                    "date": voucher_date,
                    "row": 1,
                    "vatType": {"id": 0},
                    dim_field: {"id": value_ids[0]},
                },
                {
                    "account": {"id": bank_id},
                    "amountGross": -amount,
                    "amountGrossCurrency": -amount,
                    "date": voucher_date,
                    "row": 2,
                    "vatType": {"id": 0},
                },
            ]
            voucher_body = {
                "date": voucher_date,
                "description": _val(data, "voucherDescription", data.get("dimensionName", "Accounting dimension voucher")),
                "postings": postings,
            }
            _api_post(session, base_url, "/ledger/voucher", voucher_body, trace, params={"sendToLedger": "true"})

    return trace


# ═══════════════════════════════════════════════════════════
# HELPER: CREATE CUSTOMER + PRODUCT + ORDER + ORDERLINE + INVOICE
# ═══════════════════════════════════════════════════════════

def _create_customer_if_needed(data, session, base_url, ctx, trace, prefix="customer"):
    """Create customer if not exists. Returns customer_id."""
    cust_name = data.get(f"{prefix}Name", data.get("customerName", ""))
    existing = _find_customer_by_name(ctx, cust_name)
    if existing:
        return existing["id"]

    body = {
        "name": cust_name,
        "email": _val(data, f"{prefix}Email", _val(data, "customerEmail", "")),
        "phoneNumber": _val(data, f"{prefix}Phone", _val(data, "customerPhone", "")),
        "isCustomer": True,
        "language": "NO",
    }
    org = _val(data, f"{prefix}OrgNumber", _val(data, "customerOrgNumber", ""))
    if org:
        body["organizationNumber"] = org.replace(" ", "")
    addr = _val(data, f"{prefix}Address", _val(data, "customerAddress", ""))
    postal = _val(data, f"{prefix}PostalCode", _val(data, "customerPostalCode", ""))
    city = _val(data, f"{prefix}City", _val(data, "customerCity", ""))
    if addr or postal or city:
        body["postalAddress"] = {
            "addressLine1": addr or "",
            "postalCode": postal or "",
            "city": city or "",
        }
    if body.get("email"):
        body["invoiceEmail"] = body["email"]

    result, status = _api_post(session, base_url, "/customer", body, trace)
    if status == 201:
        return result.get("value", {}).get("id")
    return None


def _create_product(name, number, price_ex_vat, vat_pct, session, base_url, trace, description=""):
    """Create a product. Returns product_id."""
    vat_type_id = {25: 3, 15: 31, 12: 32, 0: 5}.get(vat_pct, 3)
    price_incl = round(price_ex_vat * (1 + vat_pct / 100), 2)
    body = {
        "name": name,
        "number": str(number),
        "priceExcludingVatCurrency": price_ex_vat,
        "priceIncludingVatCurrency": price_incl,
        "vatType": {"id": vat_type_id},
    }
    if description:
        body["description"] = description

    result, status = _api_post(session, base_url, "/product", body, trace)
    if status == 201:
        return result.get("value", {}).get("id")
    # If 422 (duplicate), try to find existing
    if status == 422:
        r2, s2 = _api_get(session, base_url, "/product", {"number": str(number), "fields": "id"}, trace)
        vals = r2.get("values", [])
        if vals:
            return vals[0]["id"]
    return None


def _create_invoice_flow(customer_id, products_info, session, base_url, ctx, trace,
                          invoice_date=None, due_date=None, send_to_customer=False):
    """Create order -> orderlines -> invoice. Returns invoice_id.
    products_info: list of dicts with keys: product_id, price_ex_vat, vat_pct, quantity
    """
    today = _today()
    inv_date = invoice_date or today
    d_date = due_date or _due(30)

    # Create order
    order_body = {
        "customer": {"id": customer_id},
        "deliveryDate": today,
        "orderDate": today,
        "isPrioritizeAmountsIncludingVat": False,
    }
    result, status = _api_post(session, base_url, "/order", order_body, trace)
    if status != 201:
        return None
    order_id = result.get("value", {}).get("id")
    if not order_id:
        return None

    # Create orderlines
    for p in products_info:
        vat_type_id = {25: 3, 15: 31, 12: 32, 0: 5}.get(p.get("vat_pct", 25), 3)
        ol_body = {
            "order": {"id": order_id},
            "product": {"id": p["product_id"]},
            "count": p.get("quantity", 1),
            "unitPriceExcludingVatCurrency": p["price_ex_vat"],
            "vatType": {"id": vat_type_id},
        }
        desc = p.get("description")
        if desc:
            ol_body["description"] = desc
        _api_post(session, base_url, "/order/orderline", ol_body, trace)

    # Create invoice
    invoice_body = {
        "invoiceDate": inv_date,
        "invoiceDueDate": d_date,
        "orders": [{"id": order_id}],
    }
    send_param = "true" if send_to_customer else "false"
    result, status = _api_post(session, base_url, "/invoice", invoice_body, trace,
                                params={"sendToCustomer": send_param})
    if status == 201:
        return result.get("value", {}).get("id")
    return None


# ═══════════════════════════════════════════════════════════
# HANDLER: INVOICE SEND
# ═══════════════════════════════════════════════════════════

def handle_invoice_send(data, session, base_url, ctx):
    trace = []

    # Step 1: Create customer
    cust_id = _create_customer_if_needed(data, session, base_url, ctx, trace)
    if not cust_id:
        return trace

    # Step 2: Create product
    price_ex = _parse_amount(data.get("priceExVat", 0))
    vat_pct = data.get("vatPercent", 25)
    prod_id = _create_product(
        data.get("productName", "Product"),
        data.get("productNumber", "1"),
        price_ex, vat_pct, session, base_url, trace,
        description=_val(data, "description", ""),
    )
    if not prod_id:
        return trace

    # Step 3: Invoice flow
    _create_invoice_flow(
        cust_id,
        [{"product_id": prod_id, "price_ex_vat": price_ex, "vat_pct": vat_pct, "quantity": data.get("quantity", 1)}],
        session, base_url, ctx, trace,
        invoice_date=_val(data, "invoiceDate", _today()),
        due_date=_val(data, "dueDate", _due()),
    )
    return trace


# ═══════════════════════════════════════════════════════════
# HANDLER: INVOICE MULTI
# ═══════════════════════════════════════════════════════════

def handle_invoice_multi(data, session, base_url, ctx):
    trace = []

    # Step 1: Create customer
    cust_id = _create_customer_if_needed(data, session, base_url, ctx, trace)
    if not cust_id:
        return trace

    # Step 2: Create products
    products_info = []
    for i, p in enumerate(data.get("products", [])):
        price_ex = _parse_amount(p.get("priceExVat", 0))
        vat_pct = p.get("vatPercent", 25)
        prod_id = _create_product(
            p.get("name", f"Product {i+1}"),
            p.get("number", str(i+1)),
            price_ex, vat_pct, session, base_url, trace,
        )
        if prod_id:
            products_info.append({
                "product_id": prod_id,
                "price_ex_vat": price_ex,
                "vat_pct": vat_pct,
                "quantity": p.get("quantity", 1),
            })

    if not products_info:
        return trace

    # Step 3: Invoice flow
    _create_invoice_flow(
        cust_id, products_info, session, base_url, ctx, trace,
        invoice_date=_val(data, "invoiceDate", _today()),
        due_date=_val(data, "dueDate", _due()),
    )
    return trace


# ═══════════════════════════════════════════════════════════
# HANDLER: ORDER
# ═══════════════════════════════════════════════════════════

def handle_order(data, session, base_url, ctx):
    trace = []

    # Step 1: Create customer
    cust_id = _create_customer_if_needed(data, session, base_url, ctx, trace)
    if not cust_id:
        return trace

    # Step 2: Create products
    products_info = []
    for i, p in enumerate(data.get("products", [])):
        price_ex = _parse_amount(p.get("priceExVat", 0))
        vat_pct = p.get("vatPercent", 25)
        prod_id = _create_product(
            p.get("name", f"Product {i+1}"),
            p.get("number", str(i+1)),
            price_ex, vat_pct, session, base_url, trace,
        )
        if prod_id:
            products_info.append({
                "product_id": prod_id,
                "price_ex_vat": price_ex,
                "vat_pct": vat_pct,
                "quantity": p.get("quantity", 1),
            })

    if not products_info:
        return trace

    today = _today()

    # Step 3: Create order
    order_body = {
        "customer": {"id": cust_id},
        "deliveryDate": _val(data, "deliveryDate", today),
        "orderDate": _val(data, "orderDate", today),
        "isPrioritizeAmountsIncludingVat": False,
    }
    result, status = _api_post(session, base_url, "/order", order_body, trace)
    if status != 201:
        return trace
    order_id = result.get("value", {}).get("id")

    # Step 4: Create orderlines
    for p in products_info:
        vat_type_id = {25: 3, 15: 31, 12: 32, 0: 5}.get(p.get("vat_pct", 25), 3)
        ol_body = {
            "order": {"id": order_id},
            "product": {"id": p["product_id"]},
            "count": p.get("quantity", 1),
            "unitPriceExcludingVatCurrency": p["price_ex_vat"],
            "vatType": {"id": vat_type_id},
        }
        _api_post(session, base_url, "/order/orderline", ol_body, trace)

    # Step 5: Invoice from order
    params = {
        "invoiceDate": today,
        "sendToCustomer": "false",
    }
    result, status = _api_put(session, base_url, f"/order/{order_id}/:invoice", trace, params=params)
    invoice_id = None
    if status in (200, 201):
        invoice_id = result.get("value", {}).get("id")

    # Step 6: Pay if requested
    if data.get("payFull") and invoice_id:
        total_incl = sum(
            p["price_ex_vat"] * p.get("quantity", 1) * (1 + p.get("vat_pct", 25) / 100)
            for p in products_info
        )
        pt_id = _get_payment_type_id(ctx)
        pay_params = {
            "paymentDate": today,
            "paymentTypeId": str(pt_id) if pt_id else "1",
            "paidAmount": str(round(total_incl, 2)),
        }
        _api_put(session, base_url, f"/invoice/{invoice_id}/:payment", trace, params=pay_params)

    return trace


# ═══════════════════════════════════════════════════════════
# HANDLER: CREDIT NOTE
# ═══════════════════════════════════════════════════════════

def handle_credit_note(data, session, base_url, ctx):
    trace = []

    # Step 1: Create customer
    cust_id = _create_customer_if_needed(data, session, base_url, ctx, trace)
    if not cust_id:
        return trace

    # Step 2: Create product
    price_ex = _parse_amount(data.get("priceExVat", 0))
    vat_pct = data.get("vatPercent", 25)
    prod_id = _create_product(
        data.get("productName", "Product"),
        data.get("productNumber", "1"),
        price_ex, vat_pct, session, base_url, trace,
    )
    if not prod_id:
        return trace

    # Step 3: Create invoice
    invoice_id = _create_invoice_flow(
        cust_id,
        [{"product_id": prod_id, "price_ex_vat": price_ex, "vat_pct": vat_pct, "quantity": data.get("quantity", 1)}],
        session, base_url, ctx, trace,
        invoice_date=_val(data, "invoiceDate", _today()),
        due_date=_val(data, "dueDate", _due()),
    )

    if not invoice_id:
        return trace

    # Step 4: Create credit note
    cn_date = _val(data, "creditNoteDate", _today())
    cn_params = {
        "date": cn_date,
        "sendToCustomer": "false",
    }
    comment = _val(data, "reason")
    if comment:
        cn_params["comment"] = comment

    _api_put(session, base_url, f"/invoice/{invoice_id}/:createCreditNote", trace, params=cn_params)
    return trace


# ═══════════════════════════════════════════════════════════
# HANDLER: EMPLOYEE
# ═══════════════════════════════════════════════════════════

def _ensure_division(session, base_url, ctx, trace):
    """Ensure a division exists. Returns division_id."""
    if ctx.get("default_division_id"):
        return ctx["default_division_id"]

    # Create one
    div_body = {
        "name": "Hovedkontor",
        "startDate": _today(),
        "municipality": {"id": 262},
        "organizationNumber": "996757435",
    }
    result, status = _api_post(session, base_url, "/division", div_body, trace)
    if status == 201:
        div_id = result.get("value", {}).get("id")
        ctx["default_division_id"] = div_id
        return div_id
    return None


def handle_employee(data, session, base_url, ctx):
    trace = []
    today = _today()

    email = data.get("email", "")
    existing = _find_employee_by_email(ctx, email)

    dept_id = ctx.get("default_department_id")

    if existing:
        # PUT to update
        emp_id = existing["id"]
        put_body = {
            "id": emp_id,
            "firstName": data.get("firstName", existing.get("firstName", "")),
            "lastName": data.get("lastName", existing.get("lastName", "")),
            "email": email,
            "dateOfBirth": existing.get("dateOfBirth", "1990-05-15"),
            "version": existing.get("version"),
        }
        user_type = _val(data, "userType", "STANDARD")
        if user_type.upper() == "ADMINISTRATOR":
            user_type = "EXTENDED"
        put_body["userType"] = user_type
        if dept_id:
            put_body["department"] = {"id": dept_id}
        addr = _val(data, "addressLine1")
        if addr:
            put_body["address"] = {
                "addressLine1": addr,
                "postalCode": _val(data, "postalCode", ""),
                "city": _val(data, "city", ""),
            }
        nid = _val(data, "nationalIdentityNumber")
        if nid:
            put_body["nationalIdentityNumber"] = nid
        emp_num = _val(data, "employeeNumber")
        if emp_num:
            put_body["employeeNumber"] = str(emp_num)

        _api_put(session, base_url, f"/employee/{emp_id}", trace, body=put_body)

        # Check if employment exists
        has_employment = bool(existing.get("_employments"))
        if not has_employment:
            div_id = _ensure_division(session, base_url, ctx, trace)
            start_date = _val(data, "startDate", today)
            if div_id:
                empl_body = {
                    "employee": {"id": emp_id},
                    "startDate": start_date,
                    "division": {"id": div_id},
                    "isMainEmployer": True,
                    "taxDeductionCode": "loennFraHovedarbeidsgiver",
                }
                result, status = _api_post(session, base_url, "/employee/employment", empl_body, trace)
                if status == 201:
                    empl_id = result.get("value", {}).get("id")
                    if empl_id:
                        det_body = {
                            "employment": {"id": empl_id},
                            "date": start_date,
                            "employmentType": "ORDINARY",
                            "employmentForm": "PERMANENT",
                            "remunerationType": "MONTHLY_WAGE",
                            "workingHoursScheme": "NOT_SHIFT",
                            "percentageOfFullTimeEquivalent": 100.0,
                        }
                        _api_post(session, base_url, "/employee/employment/details", det_body, trace)
    else:
        # POST new employee
        dob = _val(data, "dateOfBirth", "1990-05-15")
        user_type = _val(data, "userType", "STANDARD")
        if user_type.upper() == "ADMINISTRATOR":
            user_type = "EXTENDED"
        post_body = {
            "firstName": data.get("firstName", ""),
            "lastName": data.get("lastName", ""),
            "email": email,
            "dateOfBirth": dob,
            "userType": user_type,
        }
        if dept_id:
            post_body["department"] = {"id": dept_id}
        emp_num = _val(data, "employeeNumber")
        if emp_num:
            post_body["employeeNumber"] = str(emp_num)
        nid = _val(data, "nationalIdentityNumber")
        if nid:
            post_body["nationalIdentityNumber"] = nid
        addr = _val(data, "addressLine1")
        if addr:
            post_body["address"] = {
                "addressLine1": addr,
                "postalCode": _val(data, "postalCode", ""),
                "city": _val(data, "city", ""),
            }

        result, status = _api_post(session, base_url, "/employee", post_body, trace)
        if status != 201:
            return trace
        emp_id = result.get("value", {}).get("id")
        if not emp_id:
            return trace

        # Create division + employment + details
        div_id = _ensure_division(session, base_url, ctx, trace)
        start_date = _val(data, "startDate", today)
        if div_id:
            empl_body = {
                "employee": {"id": emp_id},
                "startDate": start_date,
                "division": {"id": div_id},
                "isMainEmployer": True,
                "taxDeductionCode": "loennFraHovedarbeidsgiver",
            }
            result2, status2 = _api_post(session, base_url, "/employee/employment", empl_body, trace)
            if status2 == 201:
                empl_id = result2.get("value", {}).get("id")
                if empl_id:
                    det_body = {
                        "employment": {"id": empl_id},
                        "date": start_date,
                        "employmentType": "ORDINARY",
                        "employmentForm": "PERMANENT",
                        "remunerationType": "MONTHLY_WAGE",
                        "workingHoursScheme": "NOT_SHIFT",
                        "percentageOfFullTimeEquivalent": 100.0,
                    }
                    _api_post(session, base_url, "/employee/employment/details", det_body, trace)

    return trace


# ═══════════════════════════════════════════════════════════
# HANDLER: SUPPLIER INVOICE
# ═══════════════════════════════════════════════════════════

def _create_supplier_if_needed(data, session, base_url, ctx, trace):
    """Create supplier (via /customer with isSupplier). Returns supplier_id."""
    sup_name = data.get("supplierName", "")
    existing = _find_supplier_by_name(ctx, sup_name)
    if existing:
        return existing["id"]

    email = _val(data, "supplierEmail", "")
    if not email:
        safe_name = re.sub(r'[^a-z]', '', sup_name.lower())[:20]
        email = f"faktura@{safe_name}.no"

    body = {
        "name": sup_name,
        "email": email,
        "invoiceEmail": email,
        "phoneNumber": _val(data, "supplierPhone", ""),
        "isCustomer": False,
        "isSupplier": True,
        "supplierNumber": 0,
        "language": "NO",
    }
    org = _val(data, "supplierOrgNumber", "")
    if org:
        body["organizationNumber"] = org.replace(" ", "")
    addr = _val(data, "supplierAddress")
    postal = _val(data, "supplierPostalCode")
    city = _val(data, "supplierCity")
    if addr or postal or city:
        body["postalAddress"] = {
            "addressLine1": addr or "",
            "postalCode": postal or "",
            "city": city or "",
        }
    if ctx.get("employees"):
        body["accountManager"] = {"id": ctx["employees"][0]["id"]}

    result, status = _api_post(session, base_url, "/customer", body, trace)
    if status == 201:
        return result.get("value", {}).get("id")
    return None


def handle_supplier_invoice(data, session, base_url, ctx):
    trace = []

    # Step 1: Create supplier
    sup_id = _create_supplier_if_needed(data, session, base_url, ctx, trace)
    if not sup_id:
        return trace

    # Step 2: Determine expense account
    exp_acct_num = _val(data, "expenseAccount", "6500")
    exp_acct_id = _get_acct_id(ctx, exp_acct_num)
    if not exp_acct_id:
        # Fetch it
        result, _ = _api_get(session, base_url, "/ledger/account",
                             {"number": exp_acct_num, "fields": "id"}, trace)
        vals = result.get("values", [])
        if vals:
            exp_acct_id = vals[0]["id"]

    acct_2400_id = _get_acct_id(ctx, "2400")
    if not acct_2400_id:
        result, _ = _api_get(session, base_url, "/ledger/account",
                             {"number": "2400", "fields": "id"}, trace)
        vals = result.get("values", [])
        if vals:
            acct_2400_id = vals[0]["id"]

    if not exp_acct_id or not acct_2400_id:
        log.error("Missing ledger account IDs for supplier invoice voucher")
        return trace

    # Step 3: Create voucher
    amount_incl = _parse_amount(data.get("amountInclVat", 0))
    amount_ex = _parse_amount(data.get("amountExVat", 0))
    vat_pct = data.get("vatPercent", 25)

    if amount_incl <= 0 and amount_ex > 0:
        amount_incl = round(amount_ex * (1 + vat_pct / 100), 2)
    elif amount_incl <= 0:
        amount_incl = amount_ex  # fallback

    is_food = data.get("isFood", False)
    # vatType:1 = ingoing 25%, vatType:11 = ingoing 15% (food)
    expense_vat_type = 11 if is_food else 1

    inv_date = _val(data, "invoiceDate", _today())
    inv_num = _val(data, "invoiceNumber", "")
    sup_name = data.get("supplierName", "")

    voucher_body = {
        "date": inv_date,
        "description": f"Leverandorfaktura {inv_num} {sup_name}".strip(),
        "postings": [
            {
                "account": {"id": exp_acct_id},
                "amountGross": amount_incl,
                "amountGrossCurrency": amount_incl,
                "date": inv_date,
                "row": 1,
                "vatType": {"id": expense_vat_type},
            },
            {
                "account": {"id": acct_2400_id},
                "amountGross": -amount_incl,
                "amountGrossCurrency": -amount_incl,
                "date": inv_date,
                "row": 2,
                "vatType": {"id": 0},
                "supplier": {"id": sup_id},
            },
        ],
    }

    _api_post(session, base_url, "/ledger/voucher", voucher_body, trace, params={"sendToLedger": "true"})
    return trace


# ═══════════════════════════════════════════════════════════
# HANDLER: RECEIPT VOUCHER
# ═══════════════════════════════════════════════════════════

def handle_receipt_voucher(data, session, base_url, ctx):
    trace = []

    exp_acct_num = _val(data, "expenseAccount", "6500")
    exp_acct_id = _get_acct_id(ctx, exp_acct_num)
    if not exp_acct_id:
        result, _ = _api_get(session, base_url, "/ledger/account",
                             {"number": exp_acct_num, "fields": "id"}, trace)
        vals = result.get("values", [])
        if vals:
            exp_acct_id = vals[0]["id"]

    bank_id = _get_acct_id(ctx, "1920")
    if not bank_id:
        result, _ = _api_get(session, base_url, "/ledger/account",
                             {"number": "1920", "fields": "id"}, trace)
        vals = result.get("values", [])
        if vals:
            bank_id = vals[0]["id"]

    if not exp_acct_id or not bank_id:
        log.error("Missing ledger account IDs for receipt voucher")
        return trace

    amount_incl = _parse_amount(data.get("amountInclVat", 0))
    receipt_date = _val(data, "receiptDate", _today())
    is_food = data.get("isFood", False)
    expense_vat_type = 11 if is_food else 1

    # Look up department by name if ID not given
    dept_id = data.get("departmentId")
    dept_name = data.get("departmentName", "")
    if not dept_id and dept_name:
        for d in ctx.get("departments", []):
            if d.get("name", "").lower() == dept_name.lower():
                dept_id = d["id"]
                break
    if not dept_id:
        dept_id = ctx.get("default_department_id")

    expense_posting = {
        "account": {"id": exp_acct_id},
        "amountGross": amount_incl,
        "amountGrossCurrency": amount_incl,
        "date": receipt_date,
        "row": 1,
        "vatType": {"id": expense_vat_type},
    }
    if dept_id:
        expense_posting["department"] = {"id": dept_id}

    voucher_body = {
        "date": receipt_date,
        "description": data.get("description", "Receipt"),
        "postings": [
            expense_posting,
            {
                "account": {"id": bank_id},
                "amountGross": -amount_incl,
                "amountGrossCurrency": -amount_incl,
                "date": receipt_date,
                "row": 2,
                "vatType": {"id": 0},
            },
        ],
    }

    _api_post(session, base_url, "/ledger/voucher", voucher_body, trace, params={"sendToLedger": "true"})
    return trace


# ═══════════════════════════════════════════════════════════
# HANDLER: SALARY
# ═══════════════════════════════════════════════════════════

def handle_salary(data, session, base_url, ctx):
    trace = []
    today = _today()

    # Step 1: Find employee
    email = data.get("employeeEmail", "")
    emp = _find_employee_by_email(ctx, email)
    if not emp:
        log.error(f"Employee not found by email: {email}")
        return trace

    emp_id = emp["id"]

    # Ensure employee has employment
    has_employment = bool(emp.get("_employments"))
    if not has_employment:
        div_id = _ensure_division(session, base_url, ctx, trace)
        if div_id:
            empl_body = {
                "employee": {"id": emp_id},
                "startDate": today,
                "division": {"id": div_id},
                "isMainEmployer": True,
                "taxDeductionCode": "loennFraHovedarbeidsgiver",
            }
            result, status = _api_post(session, base_url, "/employee/employment", empl_body, trace)
            if status == 201:
                empl_id = result.get("value", {}).get("id")
                if empl_id:
                    det_body = {
                        "employment": {"id": empl_id},
                        "date": today,
                        "employmentType": "ORDINARY",
                        "employmentForm": "PERMANENT",
                        "remunerationType": "MONTHLY_WAGE",
                        "workingHoursScheme": "NOT_SHIFT",
                        "percentageOfFullTimeEquivalent": 100.0,
                    }
                    _api_post(session, base_url, "/employee/employment/details", det_body, trace)

    # Step 2: Standard time (ignore 422 if exists)
    std_body = {
        "employee": {"id": emp_id},
        "fromDate": today,
        "hoursPerDay": 7.5,
    }
    _api_post(session, base_url, "/employee/standardTime", std_body, trace)

    # Step 3: Try salary/transaction
    base_salary = _parse_amount(data.get("baseSalary", 0))
    bonus = _parse_amount(data.get("bonus", 0))
    sal_month = data.get("month", date.today().month)
    sal_year = data.get("year", date.today().year)
    sal_date = _val(data, "salaryDate", today)

    salary_types = ctx.get("salary_types", {})
    type_2000 = salary_types.get("2000", {})
    type_2002 = salary_types.get("2002", {})

    specifications = []
    if type_2000.get("id") and base_salary > 0:
        specifications.append({
            "salaryType": {"id": type_2000["id"]},
            "rate": base_salary,
            "count": 1,
            "amount": base_salary,
        })
    if type_2002.get("id") and bonus > 0:
        specifications.append({
            "salaryType": {"id": type_2002["id"]},
            "rate": bonus,
            "count": 1,
            "amount": bonus,
        })

    if specifications:
        sal_body = {
            "date": sal_date,
            "month": sal_month,
            "year": sal_year,
            "payslips": [{
                "employee": {"id": emp_id},
                "date": sal_date,
                "year": sal_year,
                "month": sal_month,
                "specifications": specifications,
            }],
        }
        result, status = _api_post(session, base_url, "/salary/transaction", sal_body, trace,
                                    params={"generateTaxDeduction": "true"})
        if status == 201:
            ctx["_salary_transaction_done"] = True
            return trace

    # Step 4: Voucher fallback
    total = base_salary + bonus
    if total <= 0:
        return trace

    acct_5000 = _get_acct_id(ctx, "5000")
    acct_2780 = _get_acct_id(ctx, "2780")

    if not acct_5000 or not acct_2780:
        # Fetch them
        for num in ("5000", "2780"):
            if not _get_acct_id(ctx, num):
                r, _ = _api_get(session, base_url, "/ledger/account", {"number": num, "fields": "id"}, trace)
                vals = r.get("values", [])
                if vals:
                    if "ledger_accounts" not in ctx:
                        ctx["ledger_accounts"] = {}
                    ctx["ledger_accounts"][num] = vals[0]["id"]
        acct_5000 = _get_acct_id(ctx, "5000")
        acct_2780 = _get_acct_id(ctx, "2780")

    if acct_5000 and acct_2780:
        voucher_body = {
            "date": sal_date,
            "description": f"Lonn {sal_month}/{sal_year}",
            "postings": [
                {
                    "account": {"id": acct_5000},
                    "amountGross": total,
                    "amountGrossCurrency": total,
                    "date": sal_date,
                    "row": 1,
                    "vatType": {"id": 0},
                },
                {
                    "account": {"id": acct_2780},
                    "amountGross": -total,
                    "amountGrossCurrency": -total,
                    "date": sal_date,
                    "row": 2,
                    "vatType": {"id": 0},
                },
            ],
        }
        _api_post(session, base_url, "/ledger/voucher", voucher_body, trace, params={"sendToLedger": "true"})

    return trace


# ═══════════════════════════════════════════════════════════
# HANDLER: YEAR END
# ═══════════════════════════════════════════════════════════

def handle_year_end(data, session, base_url, ctx):
    trace = []
    voucher_date = _val(data, "voucherDate", _today())

    def get_acct(num):
        """Get account ID, fetching if needed."""
        aid = _get_acct_id(ctx, str(num))
        if aid:
            return aid
        r, _ = _api_get(session, base_url, "/ledger/account", {"number": str(num), "fields": "id"}, trace)
        vals = r.get("values", [])
        if vals:
            if "ledger_accounts" not in ctx:
                ctx["ledger_accounts"] = {}
            ctx["ledger_accounts"][str(num)] = vals[0]["id"]
            return vals[0]["id"]
        return None

    total_depreciation = 0.0

    # Step 1: Depreciation vouchers (one per asset)
    for asset in data.get("assets", []):
        dep_amount = _parse_amount(asset.get("depreciationAmount", 0))
        if dep_amount <= 0:
            cost = _parse_amount(asset.get("originalCost", 0))
            years = asset.get("usefulLifeYears", 1)
            if years > 0:
                dep_amount = round(cost / years, 2)

        if dep_amount <= 0:
            continue

        total_depreciation += dep_amount

        dep_exp_acct = get_acct(asset.get("depreciationAccountNumber", "6010"))
        accum_dep_acct = get_acct(asset.get("accumulatedDepAccountNumber", "1209"))

        if dep_exp_acct and accum_dep_acct:
            v_body = {
                "date": voucher_date,
                "description": f"Avskrivning {asset.get('name', '')}".strip(),
                "postings": [
                    {
                        "account": {"id": dep_exp_acct},
                        "amountGross": dep_amount,
                        "amountGrossCurrency": dep_amount,
                        "date": voucher_date,
                        "row": 1,
                        "vatType": {"id": 0},
                    },
                    {
                        "account": {"id": accum_dep_acct},
                        "amountGross": -dep_amount,
                        "amountGrossCurrency": -dep_amount,
                        "date": voucher_date,
                        "row": 2,
                        "vatType": {"id": 0},
                    },
                ],
            }
            _api_post(session, base_url, "/ledger/voucher", v_body, trace, params={"sendToLedger": "true"})

    # Step 2: Prepaid expense reversals
    total_prepaid = 0.0
    for prepaid in data.get("prepaidExpenses", []):
        amt = _parse_amount(prepaid.get("amount", 0))
        if amt <= 0:
            continue
        total_prepaid += amt

        exp_acct = get_acct(prepaid.get("expenseAccount", "6300"))
        prepaid_acct = get_acct(prepaid.get("prepaidAccount", "1700"))

        if exp_acct and prepaid_acct:
            v_body = {
                "date": voucher_date,
                "description": f"Periodisering {prepaid.get('description', '')}".strip(),
                "postings": [
                    {
                        "account": {"id": exp_acct},
                        "amountGross": amt,
                        "amountGrossCurrency": amt,
                        "date": voucher_date,
                        "row": 1,
                        "vatType": {"id": 0},
                    },
                    {
                        "account": {"id": prepaid_acct},
                        "amountGross": -amt,
                        "amountGrossCurrency": -amt,
                        "date": voucher_date,
                        "row": 2,
                        "vatType": {"id": 0},
                    },
                ],
            }
            _api_post(session, base_url, "/ledger/voucher", v_body, trace, params={"sendToLedger": "true"})

    # Step 3: Salary accrual
    sal_accrual = data.get("salaryAccrual", {})
    sal_amt = _parse_amount(sal_accrual.get("amount", 0))
    if sal_amt > 0:
        acct_5000 = get_acct("5000")
        acct_2900 = get_acct("2900")
        if acct_5000 and acct_2900:
            v_body = {
                "date": voucher_date,
                "description": sal_accrual.get("description", "Lonnsavsetning"),
                "postings": [
                    {
                        "account": {"id": acct_5000},
                        "amountGross": sal_amt,
                        "amountGrossCurrency": sal_amt,
                        "date": voucher_date,
                        "row": 1,
                        "vatType": {"id": 0},
                    },
                    {
                        "account": {"id": acct_2900},
                        "amountGross": -sal_amt,
                        "amountGrossCurrency": -sal_amt,
                        "date": voucher_date,
                        "row": 2,
                        "vatType": {"id": 0},
                    },
                ],
            }
            _api_post(session, base_url, "/ledger/voucher", v_body, trace, params={"sendToLedger": "true"})

    # Step 4: Tax calculation
    # GET ledger postings for full year to calculate revenue - expenses
    year = voucher_date[:4]
    result, status = _api_get(session, base_url, "/ledger/posting", {
        "dateFrom": f"{year}-01-01",
        "dateTo": f"{year}-12-31",
        "fields": "id,amount,account(number,name)",
        "count": "5000",
    }, trace)

    postings = result.get("values", [])
    revenue = 0.0
    expenses = 0.0
    for p in postings:
        acct = p.get("account", {})
        acct_num = str(acct.get("number", ""))
        amt = float(p.get("amount", 0))
        if acct_num.startswith("3"):  # Revenue accounts
            revenue += amt
        elif acct_num.startswith(("4", "5", "6", "7")):  # Expense accounts
            expenses += amt

    # Revenue is typically negative (credit), expenses positive (debit)
    # The GET results are BEFORE our new vouchers, so we need to ADD our new costs
    # total_depreciation, total_prepaid, sal_amt are costs we just posted (not yet in ledger)
    profit = -(revenue) - expenses - total_depreciation - total_prepaid - sal_amt
    tax_rate = data.get("taxRate", 0.22)
    tax = round(max(0, profit * tax_rate), 2)

    if tax > 0:
        acct_8700 = get_acct("8700")
        acct_2920 = get_acct("2920")
        if not acct_2920:
            acct_2920 = get_acct("2050")  # fallback
        if acct_8700 and acct_2920:
            v_body = {
                "date": voucher_date,
                "description": f"Skattekostnad {year}",
                "postings": [
                    {
                        "account": {"id": acct_8700},
                        "amountGross": tax,
                        "amountGrossCurrency": tax,
                        "date": voucher_date,
                        "row": 1,
                        "vatType": {"id": 0},
                    },
                    {
                        "account": {"id": acct_2920},
                        "amountGross": -tax,
                        "amountGrossCurrency": -tax,
                        "date": voucher_date,
                        "row": 2,
                        "vatType": {"id": 0},
                    },
                ],
            }
            _api_post(session, base_url, "/ledger/voucher", v_body, trace, params={"sendToLedger": "true"})

    return trace


# ═══════════════════════════════════════════════════════════
# HANDLER: COST ANALYSIS
# ═══════════════════════════════════════════════════════════

def handle_cost_analysis(data, session, base_url, ctx):
    trace = []
    today = _today()

    # Step 1: GET postings for both months
    m1_start = data.get("month1Start", "2026-01-01")
    m1_end = data.get("month1End", "2026-01-31")
    m2_start = data.get("month2Start", "2026-02-01")
    m2_end = data.get("month2End", "2026-02-28")

    r1, _ = _api_get(session, base_url, "/ledger/posting", {
        "dateFrom": m1_start, "dateTo": m1_end,
        "fields": "id,amount,account(number,name)", "count": "1000",
    }, trace)

    r2, _ = _api_get(session, base_url, "/ledger/posting", {
        "dateFrom": m2_start, "dateTo": m2_end,
        "fields": "id,amount,account(number,name)", "count": "1000",
    }, trace)

    # Step 2: Group by account, sum amounts
    def sum_by_account(postings_data):
        sums = {}
        names = {}
        for p in postings_data.get("values", []):
            acct = p.get("account", {})
            num = str(acct.get("number", ""))
            name = acct.get("name", "")
            if not num:
                continue
            # Only expense accounts (4xxx-7xxx)
            if num and num[0] in ("4", "5", "6", "7"):
                sums[num] = sums.get(num, 0) + float(p.get("amount", 0))
                names[num] = name
        return sums, names

    m1_sums, m1_names = sum_by_account(r1)
    m2_sums, m2_names = sum_by_account(r2)

    # Step 3: Calculate increases
    increases = []
    all_accounts = set(m1_sums.keys()) | set(m2_sums.keys())
    for acct_num in all_accounts:
        m1_val = m1_sums.get(acct_num, 0)
        m2_val = m2_sums.get(acct_num, 0)
        increase = m2_val - m1_val
        name = m2_names.get(acct_num, m1_names.get(acct_num, acct_num))
        increases.append((acct_num, name, increase))

    increases.sort(key=lambda x: x[2], reverse=True)
    top_n = data.get("numAccountsToFind", 3)
    top_accounts = increases[:top_n]

    log.info(f"Top {top_n} cost increases: {top_accounts}")

    # Step 4: Make employee EXTENDED
    email = data.get("employeeEmail", "")
    emp = _find_employee_by_email(ctx, email)
    if not emp:
        # Use first employee
        if ctx.get("employees"):
            emp = ctx["employees"][0]

    if emp:
        emp_id = emp["id"]
        put_body = {
            "id": emp_id,
            "firstName": emp.get("firstName", ""),
            "lastName": emp.get("lastName", ""),
            "email": emp.get("email", ""),
            "dateOfBirth": emp.get("dateOfBirth", "1990-05-15"),
            "userType": "EXTENDED",
            "version": emp.get("version"),
        }
        _api_put(session, base_url, f"/employee/{emp_id}", trace, body=put_body)

        # Step 5: Grant entitlements
        company_id = ctx.get("_company_id")
        if company_id:
            for ent_id in (45, 10):
                _api_put(session, base_url, "/employee/entitlement/:grantEntitlementsByTemplate",
                         trace, params={"employeeId": str(emp_id), "template": "ALL_PRIVILEGES"})

        # Step 6: Create projects for top accounts
        for acct_num, acct_name, increase in top_accounts:
            proj_body = {
                "name": acct_name if acct_name else f"Kostnadsanalyse {acct_num}",
                "startDate": today,
                "projectManager": {"id": emp_id},
                "isInternal": True,
            }
            result, status = _api_post(session, base_url, "/project", proj_body, trace)

            # Create activity
            act_body = {
                "name": acct_name if acct_name else f"Analyse {acct_num}",
                "activityType": "GENERAL_ACTIVITY",
                "isGeneral": True,
                "isProjectActivity": False,
            }
            _api_post(session, base_url, "/activity", act_body, trace)

    return trace


# ═══════════════════════════════════════════════════════════
# HANDLER: PROJECT
# ═══════════════════════════════════════════════════════════

def _ensure_employee_extended(emp, session, base_url, ctx, trace):
    """Ensure employee is EXTENDED and has entitlements. Returns emp_id."""
    emp_id = emp["id"]
    if emp.get("userType") != "EXTENDED":
        put_body = {
            "id": emp_id,
            "firstName": emp.get("firstName", ""),
            "lastName": emp.get("lastName", ""),
            "email": emp.get("email", ""),
            "dateOfBirth": emp.get("dateOfBirth", "1990-05-15"),
            "userType": "EXTENDED",
            "version": emp.get("version"),
        }
        _api_put(session, base_url, f"/employee/{emp_id}", trace, body=put_body)

    # Grant entitlements
    _api_put(session, base_url, "/employee/entitlement/:grantEntitlementsByTemplate",
             trace, params={"employeeId": str(emp_id), "template": "ALL_PRIVILEGES"})

    return emp_id


def handle_project(data, session, base_url, ctx):
    trace = []

    # Find/setup employee as project manager
    email = _val(data, "employeeEmail", "")
    emp = _find_employee_by_email(ctx, email) if email else None
    if not emp and ctx.get("employees"):
        emp = ctx["employees"][0]

    if not emp:
        log.error("No employee found for project manager")
        return trace

    emp_id = _ensure_employee_extended(emp, session, base_url, ctx, trace)

    # Create customer if specified
    cust_id = None
    cust_name = _val(data, "customerName")
    if cust_name:
        cust_id = _create_customer_if_needed(data, session, base_url, ctx, trace)

    # Create project
    proj_body = {
        "name": data.get("projectName", "Project"),
        "startDate": _val(data, "startDate", _today()),
        "projectManager": {"id": emp_id},
    }
    if cust_id:
        proj_body["customer"] = {"id": cust_id}
    end_date = _val(data, "endDate")
    if end_date:
        proj_body["endDate"] = end_date
    desc = _val(data, "description")
    if desc:
        proj_body["description"] = desc

    _api_post(session, base_url, "/project", proj_body, trace)
    return trace


# ═══════════════════════════════════════════════════════════
# HANDLER: PROJECT FIXED PRICE
# ═══════════════════════════════════════════════════════════

def handle_project_fixed(data, session, base_url, ctx):
    trace = []

    # Find/setup employee
    email = _val(data, "employeeEmail", "")
    emp = _find_employee_by_email(ctx, email) if email else None
    if not emp and ctx.get("employees"):
        emp = ctx["employees"][0]

    if not emp:
        return trace

    emp_id = _ensure_employee_extended(emp, session, base_url, ctx, trace)

    # Create customer if specified
    cust_id = None
    cust_name = _val(data, "customerName")
    if cust_name:
        cust_id = _create_customer_if_needed(data, session, base_url, ctx, trace)

    # Create project
    proj_body = {
        "name": data.get("projectName", "Project"),
        "startDate": _val(data, "startDate", _today()),
        "projectManager": {"id": emp_id},
    }
    if cust_id:
        proj_body["customer"] = {"id": cust_id}

    result, status = _api_post(session, base_url, "/project", proj_body, trace)
    if status != 201:
        return trace

    proj_id = result.get("value", {}).get("id")
    proj_version = result.get("value", {}).get("version")
    proj_number = result.get("value", {}).get("number")

    if not proj_id:
        return trace

    # GET project to get version
    r2, _ = _api_get(session, base_url, f"/project/{proj_id}", {"fields": "id,version,number"}, trace)
    if r2.get("value"):
        proj_version = r2["value"].get("version", proj_version)
        proj_number = r2["value"].get("number", proj_number)

    # PUT to set fixed price
    fixed_price = _parse_amount(data.get("fixedPrice", 0))
    put_body = {
        "id": proj_id,
        "name": data.get("projectName", "Project"),
        "startDate": _val(data, "startDate", _today()),
        "projectManager": {"id": emp_id},
        "isFixedPrice": True,
        "fixedprice": fixed_price,
        "version": proj_version,
    }
    if proj_number:
        put_body["number"] = str(proj_number)
    if cust_id:
        put_body["customer"] = {"id": cust_id}

    _api_put(session, base_url, f"/project/{proj_id}", trace, body=put_body)

    # Create partial invoice if requested
    pct = data.get("partialInvoicePercent", 0)
    if pct > 0 and fixed_price > 0 and cust_id:
        partial_amount = round(fixed_price * pct / 100, 2)
        # Create product for partial invoice
        prod_id = _create_product(
            f"Delfaktura {data.get('projectName', 'Prosjekt')}",
            f"PROJ-{proj_id}",
            partial_amount, 25, session, base_url, trace,
        )
        if prod_id:
            _create_invoice_flow(
                cust_id,
                [{"product_id": prod_id, "price_ex_vat": partial_amount, "vat_pct": 25, "quantity": 1}],
                session, base_url, ctx, trace,
            )

    return trace


# ═══════════════════════════════════════════════════════════
# HANDLER: TIMESHEET
# ═══════════════════════════════════════════════════════════

def handle_timesheet(data, session, base_url, ctx):
    trace = []
    today = _today()

    # Find/setup employee
    email = _val(data, "employeeEmail", "")
    emp = _find_employee_by_email(ctx, email) if email else None
    if not emp and ctx.get("employees"):
        emp = ctx["employees"][0]

    if not emp:
        return trace

    emp_id = _ensure_employee_extended(emp, session, base_url, ctx, trace)

    # Create customer if specified
    cust_id = None
    cust_name = _val(data, "customerName")
    if cust_name:
        cust_id = _create_customer_if_needed(data, session, base_url, ctx, trace)

    # Create project
    proj_body = {
        "name": data.get("projectName", "Project"),
        "startDate": today,
        "projectManager": {"id": emp_id},
    }
    if cust_id:
        proj_body["customer"] = {"id": cust_id}

    result, status = _api_post(session, base_url, "/project", proj_body, trace)
    proj_id = None
    if status == 201:
        proj_id = result.get("value", {}).get("id")

    if not proj_id:
        return trace

    # Find activity
    activity_id = None
    activities = ctx.get("activities", [])
    if activities:
        activity_id = activities[0]["id"]
    else:
        # Create activity
        act_body = {"name": _val(data, "description", "Arbeid"), "activityType": "GENERAL_ACTIVITY", "isGeneral": True}
        r, s = _api_post(session, base_url, "/activity", act_body, trace)
        if s == 201:
            activity_id = r.get("value", {}).get("id")

    if not activity_id:
        return trace

    # Create timesheet entry
    hours = _parse_amount(data.get("hours", 0))
    entry_date = _val(data, "date", today)
    ts_body = {
        "project": {"id": proj_id},
        "activity": {"id": activity_id},
        "employee": {"id": emp_id},
        "date": entry_date,
        "hours": hours,
    }
    _api_post(session, base_url, "/timesheet/entry", ts_body, trace)

    # Create invoice for hours if hourly rate and customer
    hourly_rate = _parse_amount(data.get("hourlyRate", 0))
    if hourly_rate > 0 and hours > 0 and cust_id:
        total = round(hours * hourly_rate, 2)
        prod_id = _create_product(
            _val(data, "description", "Konsulenttimer"),
            f"TIME-{proj_id}",
            total, 25, session, base_url, trace,
        )
        if prod_id:
            _create_invoice_flow(
                cust_id,
                [{"product_id": prod_id, "price_ex_vat": total, "vat_pct": 25, "quantity": 1}],
                session, base_url, ctx, trace,
            )

    return trace


# ═══════════════════════════════════════════════════════════
# HANDLER REGISTRY + SOLVE FUNCTION
# ═══════════════════════════════════════════════════════════

DETERMINISTIC_HANDLERS = {
    "customer": handle_customer,
    "supplier": handle_supplier,
    "departments": handle_departments,
    "product": handle_product,
    "payment": handle_payment,
    "reverse_payment": handle_reverse_payment,
    "contact_person": handle_contact_person,
    "acct_dimension": handle_acct_dimension,
    "invoice_send": handle_invoice_send,
    "invoice_multi": handle_invoice_multi,
    "order": handle_order,
    "credit_note": handle_credit_note,
    "employee": handle_employee,
    "supplier_invoice": handle_supplier_invoice,
    "receipt_voucher": handle_receipt_voucher,
    "salary": handle_salary,
    "year_end": handle_year_end,
    "cost_analysis": handle_cost_analysis,
    "project": handle_project,
    "project_fixed": handle_project_fixed,
    "timesheet": handle_timesheet,
}


def solve_deterministic(task_type, prompt, files, session, base_url, ctx,
                         claude_client, model, start_time, image_blocks=None):
    """Main entry point: extract data with Claude, then run deterministic handler."""
    log.info(f"solve_deterministic: task_type={task_type}")

    # Extract structured data
    data = extract_data(claude_client, model, prompt, files, task_type, image_blocks)
    if not data:
        log.error(f"Extraction returned empty data for {task_type}")
        return []

    # Run handler
    handler = DETERMINISTIC_HANDLERS.get(task_type)
    if not handler:
        log.error(f"No deterministic handler for task_type={task_type}")
        return []

    trace = handler(data, session, base_url, ctx)
    log.info(f"Deterministic handler {task_type} completed: {len(trace)} API calls")
    return trace
