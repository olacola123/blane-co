"""
Deterministic handler system for Tripletex AI Accounting Agent.
Replaces LLM-decides-API-calls with: Claude extracts data -> Python makes exact API calls.
"""

import json
import logging
import os
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
{"dimensionName": "name of the accounting dimension", "description": "description", "values": [{"name": "value name", "number": "value number as string"}], "voucherDate": "YYYY-MM-DD", "voucherDescription": "voucher description", "accountNumber": "4-digit ledger account number like 6540", "amount": 44100.00, "dimensionValueName": "the dimension value name to link the voucher to"}
List ALL dimension values to create.
CRITICAL — these 3 fields are REQUIRED, extract them from the task text (any language):
- accountNumber: the 4-digit ledger account (e.g. "6540", "6590"). Look for "conta", "account", "konto" followed by a number.
- amount: the monetary amount in NOK (e.g. 44100). Look for "por X NOK", "for X NOK", "på X NOK". MUST be > 0, never 0.
- dimensionValueName: which dimension value the voucher should be linked to (e.g. "Offentlig"). Look for "vinculado ao", "linked to", "knyttet til".""",

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
{"customerName": "customer company name", "customerOrgNumber": "org number or empty", "customerEmail": "email", "customerPhone": "phone or empty", "customerAddress": "street address or empty", "customerPostalCode": "postal code or empty", "customerCity": "city or empty", "productName": "product name", "productNumber": "product number as string", "priceExVat": 0.00, "vatPercent": 25, "quantity": 1, "invoiceDate": "YYYY-MM-DD", "dueDate": "YYYY-MM-DD", "creditNoteDate": "YYYY-MM-DD", "reason": "reason for credit note", "invoiceNumber": "invoice number if mentioned or empty"}
The task usually refers to an EXISTING invoice that must be credited. Extract the customer name and org number to find it.""",

    "employee": """Extract from this accounting task. Return ONLY valid JSON:
{"firstName": "first name", "lastName": "last name", "email": "email", "dateOfBirth": "YYYY-MM-DD", "employeeNumber": "employee number as string or empty", "startDate": "YYYY-MM-DD or empty", "userType": "STANDARD or EXTENDED or NO_ACCESS", "addressLine1": "street address or empty", "postalCode": "postal code or empty", "city": "city or empty", "nationalIdentityNumber": "11-digit national ID or empty"}
userType defaults to STANDARD. startDate is employment start date.""",

    "supplier_invoice": """Extract from this accounting task. Return ONLY valid JSON:
{"supplierName": "supplier company name", "supplierOrgNumber": "org number or empty", "supplierEmail": "email or empty", "supplierPhone": "phone or empty", "supplierAddress": "street address or empty", "supplierPostalCode": "postal code or empty", "supplierCity": "city or empty", "invoiceNumber": "invoice number", "invoiceDate": "YYYY-MM-DD", "amountInclVat": 0.00, "amountExVat": 0.00, "vatPercent": 25, "description": "what was purchased", "expenseAccount": "expense account number like 6500", "isFood": false}
isFood: true if food/beverage (15% VAT). expenseAccount: 6540=office, 6340=IT, 6500=supplies, 7300=service, 7100=vehicle, 7140=travel, 6800=other.""",

    "receipt_voucher": """Extract from this receipt/accounting task. Return ONLY valid JSON:
{"description": "what was purchased", "amountInclVat": 0.00, "vatPercent": 25, "receiptDate": "YYYY-MM-DD", "expenseAccount": "expense account number", "isFood": false, "departmentName": "department name from prompt or empty string"}
expenseAccount: 6340=IT/datautstyr/computer/elektronikk/USB/laptop/telefon/programvare/lisenser, 6500=office supplies/kontorrekvisita/papir/skriver/penner, 7350=entertainment/representasjon/mat/restaurant, 7100=vehicle/bil/transport/bensin/parkering, 7140=travel/reise/hotell/fly, 6900=phone/mobiltelefon/abonnement, 6300=rent/leie/husleie, 6350=electricity/strøm/energi, 7700=operations/drift, 6800=other/andre driftskostnader, 6540=office equipment/kontormøbler/inventar/oppbevaringsboks/hyller/møbler/renhold.
isFood: true if food/beverage/restaurant (15% VAT instead of 25%).
IMPORTANT: Read the PDF carefully for the EXACT amount including VAT, the EXACT date, and classify the purchase correctly.""",

    "salary": """Extract from this accounting task. Return ONLY valid JSON:
{"employeeEmail": "email to find employee", "baseSalary": 0.00, "bonus": 0.00, "salaryDate": "YYYY-MM-DD or empty", "month": 0, "year": 0}
baseSalary is the monthly base salary. bonus is additional bonus amount (0 if none).""",

    "year_end": """Extract from this year-end closing task. Return ONLY valid JSON.
CRITICAL: Extract EVERY asset mentioned for depreciation — there are usually 3+ assets. Each asset MUST be a separate entry in the assets array.
{"assets": [{"name": "IT-utstyr", "accountNumber": "1210", "depreciationAccountNumber": "6010", "accumulatedDepAccountNumber": "1209", "originalCost": 108700, "usefulLifeYears": 5, "depreciationAmount": 21740}],
"prepaidExpenses": [{"description": "prepaid item", "amount": 53150, "expenseAccount": "6300", "prepaidAccount": "1700"}],
"salaryAccrual": {"amount": 0.00, "description": "salary accrual description or empty"},
"taxRate": 0.22,
"voucherDate": "YYYY-MM-DD",
"equityAccount": "2050",
"includeResultDisposition": true}
RULES:
- Extract ALL assets (IT-utstyr/datautstyr=1210, kontormaskiner=1200, kjøretøy/biler=1230, inventar=1240, bygg=1250). Each one SEPARATE in the array.
- depreciationAmount: calculate as originalCost / usefulLifeYears (linear). Round to 2 decimals.
- depreciationAccountNumber: usually 6010 (avskrivning). accumulatedDepAccountNumber: usually 1209.
- prepaidExpenses: reverse prepaid from 1700/1710 to expense account.
- taxRate: 0.22 (22% Norwegian corporate tax) unless task says otherwise.
- equityAccount: 2050 (default) or 2080 if specified.
- includeResultDisposition: true for year-end closing.""",

    "ledger_audit": """Extract ALL errors described in this accounting task. Return ONLY valid JSON:
{"errors": [{"type": "wrong_account|duplicate|missing_vat|wrong_amount", "description": "what the error is", "amount": 0.00, "wrongAccount": "account number with the error", "correctAccount": "correct account number", "correctAmount": 0.00, "date": "YYYY-MM-DD", "vatRate": 25, "expenseAccount": "expense account number for missing_vat errors"}]}
type must be one of: wrong_account, duplicate, missing_vat, wrong_amount.
For missing_vat: amount is the EXCL VAT amount. vatRate is the VAT percentage (usually 25). expenseAccount is the expense account (e.g. 6300, 7000) — NOT the 2710 account.
Extract ALL errors mentioned.""",

    "cost_analysis": """Extract from this accounting task. Return ONLY valid JSON:
{"employeeEmail": "email of employee to assign as project manager", "month1Start": "YYYY-MM-DD", "month1End": "YYYY-MM-DD", "month2Start": "YYYY-MM-DD", "month2End": "YYYY-MM-DD", "numAccountsToFind": 3}
month1 is the earlier month, month2 is the later month to compare costs.""",

    "project": """Extract from this accounting task. Return ONLY valid JSON:
{"projectName": "project name", "customerName": "customer company name or empty", "customerOrgNumber": "org number or empty", "customerEmail": "email or empty", "startDate": "YYYY-MM-DD", "endDate": "YYYY-MM-DD or empty", "description": "project description or empty", "employeeEmail": "project manager email or empty"}""",

    "project_fixed": """Extract from this accounting task. Return ONLY valid JSON:
{"projectName": "project name", "customerName": "customer company name or empty", "customerOrgNumber": "org number or empty", "customerEmail": "email or empty", "startDate": "YYYY-MM-DD", "endDate": "YYYY-MM-DD or empty", "fixedPrice": 0.00, "description": "project description or empty", "employeeEmail": "project manager email or empty", "partialInvoicePercent": 0}
fixedPrice: the fixed price amount for the project. partialInvoicePercent: percentage for partial/milestone invoice (e.g. 33 for 33%). Extract the milestone/partial percentage as integer.""",

    "timesheet": """Extract from this accounting task. Return ONLY valid JSON:
{"projectName": "project name", "customerName": "customer company name or empty", "customerOrgNumber": "org number or empty", "customerEmail": "email or empty", "employeeEmail": "employee email", "hours": 0.0, "hourlyRate": 0.00, "date": "YYYY-MM-DD", "description": "activity description or empty"}""",

    "travel_expense": """Extract from this accounting task. Return ONLY valid JSON:
{"employeeEmail": "employee email", "title": "travel expense title/description", "departureDate": "YYYY-MM-DD", "returnDate": "YYYY-MM-DD", "departureFrom": "departure city/location", "destination": "destination city/location", "purpose": "purpose of travel", "overnightStay": false, "perDiemRate": 0.00, "costs": [{"description": "cost description", "amount": 0.00, "date": "YYYY-MM-DD", "category": "transport/food/accommodation/other"}]}
overnightStay: true if there was overnight accommodation. perDiemRate: the daily per diem rate if mentioned (0 if not specified, will default to 800).""",

    "delete_travel": """Extract from this accounting task. Return ONLY valid JSON:
{"employeeEmail": "employee email or empty", "travelExpenseTitle": "title or identifier of travel expense to delete or empty", "travelExpenseId": 0}
Extract any identifier that can help find the travel expense to delete.""",

    "bank_recon": """Extract from this accounting task. Return ONLY valid JSON:
{"transactions": [{"date": "YYYY-MM-DD", "amount": 0.00, "reference": "reference/invoice number or empty", "description": "transaction description", "counterparty": "name of payer/payee or empty"}], "dateFrom": "YYYY-MM-DD", "dateTo": "YYYY-MM-DD"}
Parse ALL transactions from CSV/bank statement. Positive amounts = incoming payments, negative = outgoing payments.
CRITICAL: Extract the invoice/reference number from each transaction carefully. Look for patterns like "Faktura 1234", "Fakt. 1234", "Inv 1234", "Ref 1234", or standalone numbers in the description. Put the numeric part in "reference".""",

    "project_lifecycle": """Extract from this accounting task. Return ONLY valid JSON:
{"projectName": "project name", "budget": 0.00, "customerName": "customer company name", "customerOrgNumber": "org number or empty", "customerEmail": "email or empty", "employees": [{"email": "employee email", "hours": 0.0, "hourlyRate": 0.00, "date": "YYYY-MM-DD"}], "supplierCosts": [{"supplierName": "supplier name", "amount": 0.00, "description": "what was purchased", "expenseAccount": "expense account number like 6500", "vatPercent": 25}], "invoiceProducts": [{"name": "product name", "number": "product number", "priceExVat": 0.00, "vatPercent": 25, "quantity": 1}]}
budget: the project budget amount (0 if not mentioned). Extract ALL employees with their hours, supplier costs, and invoice products.""",

    "fx_invoice": """Extract from this accounting task. Return ONLY valid JSON:
{"customerName": "customer name", "customerOrgNumber": "org number or empty", "customerEmail": "email or empty", "invoiceNumber": "invoice number or empty", "invoiceAmount": 0.00, "currency": "currency code like USD/EUR", "oldRate": 0.00, "newRate": 0.00, "paymentDate": "YYYY-MM-DD"}
oldRate: exchange rate when invoice was created. newRate: exchange rate at payment time.""",

    "reminder_fee": """Extract from this accounting task. Return ONLY valid JSON:
{"customerName": "customer name", "overdueAmount": 0.00, "reminderFee": 35, "invoiceNumber": "overdue invoice number or empty", "reminderDate": "YYYY-MM-DD", "partialPaymentAmount": 0.00}
reminderFee defaults to 35 NOK. partialPaymentAmount: if the task mentions a partial payment on the overdue invoice, put the amount here (0 if none).""",

    "month_end": """Extract from this accounting task. Return ONLY valid JSON:
{"month": 3, "year": 2026, "voucherDate": "YYYY-MM-DD",
"assets": [{"name": "asset description", "originalCost": 0.00, "usefulLifeYears": 0, "annualDepreciation": 0.00, "depreciationAccountNumber": "6010", "accumulatedDepAccountNumber": "1209"}],
"prepaidExpenses": [{"description": "prepaid item", "monthlyAmount": 0.00, "expenseAccount": "7700", "prepaidAccount": "1710"}],
"salaryAccruals": [{"description": "salary accrual", "amount": 0.00, "expenseAccount": "5000", "liabilityAccount": "2900"}],
"otherAccruals": [{"description": "other accrual", "amount": 0.00, "debitAccount": "account number", "creditAccount": "account number"}]}

CRITICAL RULES FOR ACCOUNT NUMBERS:
- ALL account fields MUST be NUMERIC 4-digit account numbers (e.g. "1700", "6030", "5000", "2900"). NEVER use text names like "charges", "expenses", "salaires", "dépenses".
- If the task says "vers charges" / "to expenses" / "dépenses" / "gastos" without a specific account number, use "7700" (general operating expenses).
- Common mappings: charges/expenses/dépenses = "7700", salary expenses/charges salariales = "5000", depreciation/amortissement = "6010" or "6030" (use number from task if given), accumulated depreciation = "1029" or "1209", prepaid/compte 1700/1710 = "1700" or "1710", accrued salaries/salaires à payer = "2900".
- If the task explicitly gives an account number (like "compte 6030"), always use THAT number.

For assets: if annualDepreciation not given, calculate as originalCost / usefulLifeYears. Monthly = annual / 12.
For prepaid: monthlyAmount is the per-month amount (already divided).
For salary accruals: debit expense (e.g. 5000), credit liability (e.g. 2900/2910/2920).
Include ALL items mentioned in the task — depreciation, prepaid reversals, salary accruals, other provisions.""",

    "employee_pdf": """Extract from this PDF/document about an employee. Return ONLY valid JSON:
{"firstName": "first name", "lastName": "last name", "email": "email", "dateOfBirth": "YYYY-MM-DD", "nationalIdentityNumber": "11-digit national ID or empty", "occupationCode": "occupation code like 2310 or empty", "startDate": "YYYY-MM-DD", "salary": 0.00, "percentageOfFullTime": 100.0, "employeeNumber": "employee number as string or empty", "userType": "STANDARD or EXTENDED", "departmentName": "department/avdeling name from contract or empty"}
CRITICAL: nationalIdentityNumber is the 11-digit fødselsnummer/personnummer. salary is the annual salary (årslønn). If monthly salary given, multiply by 12. percentageOfFullTime is stillingsprosent (e.g. 80 means 80%). Extract ALL fields from the attached PDF document.""",

    "supplier_invoice_pdf": """Extract from this PDF/document about a supplier invoice. Return ONLY valid JSON:
{"supplierName": "supplier company name", "supplierOrgNumber": "org number or empty", "supplierEmail": "email or empty", "supplierPhone": "phone or empty", "supplierAddress": "street address or empty", "supplierPostalCode": "postal code or empty", "supplierCity": "city or empty", "invoiceNumber": "invoice number", "invoiceDate": "YYYY-MM-DD", "amountInclVat": 0.00, "amountExVat": 0.00, "vatPercent": 25, "description": "what was purchased", "expenseAccount": "expense account number like 6500", "isFood": false}
Extract from the attached PDF document. isFood: true if food/beverage (15% VAT). expenseAccount: 6540=office, 6340=IT, 6500=supplies, 7300=service, 7100=vehicle, 7140=travel, 6800=other.""",
}


# ═══════════════════════════════════════════════════════════
# EXTRACT FUNCTION
# ═══════════════════════════════════════════════════════════

_handler_gcp_token: dict = {"token": None, "expiry": 0.0}


def _get_handler_gcp_token():
    """Get GCP access token from metadata server (for Cloud Run)."""
    import time as _t
    import requests as _req
    now = _t.time()
    if _handler_gcp_token["token"] and now < _handler_gcp_token["expiry"] - 60:
        return _handler_gcp_token["token"]
    try:
        resp = _req.get(
            "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
            headers={"Metadata-Flavor": "Google"},
            timeout=3,
        )
        if resp.status_code == 200:
            data = resp.json()
            _handler_gcp_token["token"] = data["access_token"]
            _handler_gcp_token["expiry"] = now + data.get("expires_in", 3600)
            return _handler_gcp_token["token"]
    except Exception:
        pass
    return None


def extract_data(api_key, model, prompt: str, files: list, task_type: str, image_blocks: list = None) -> dict:
    """Make ONE LLM call with a task-specific extraction prompt. Returns structured dict.
    Tries Claude first, falls back to Vertex AI / Gemini API key."""
    import requests as _requests

    extraction_prompt = EXTRACTION_PROMPTS.get(task_type)
    if not extraction_prompt:
        extraction_prompt = "Extract all relevant data from this accounting task. Return ONLY valid JSON with all fields needed."

    # Process files into text
    file_text = ""
    gemini_image_parts = []  # For Gemini fallback
    claude_image_parts = []  # For Claude
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
                        pdf_parts = []
                        for page in pdf.pages:
                            text_content = page.extract_text()
                            if text_content:
                                pdf_parts.append(text_content)
                            for table in page.extract_tables():
                                for row in table:
                                    if row:
                                        pdf_parts.append(" | ".join(str(c) for c in row if c))
                        if pdf_parts:
                            file_text += f"\nPDF '{filename}':\n" + "\n".join(pdf_parts)[:6000]
                except Exception as e:
                    log.warning(f"PDF extraction failed: {e}")
            elif mime.startswith("image/"):
                gemini_image_parts.append({
                    "inline_data": {"mime_type": mime, "data": f["content_base64"]},
                })
                claude_image_parts.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime, "data": f["content_base64"]},
                })

    # Also add pre-processed image blocks
    if image_blocks:
        for img in image_blocks:
            if "inline_data" in img:
                gemini_image_parts.append(img)
                claude_image_parts.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": img["inline_data"]["mime_type"], "data": img["inline_data"]["data"]},
                })

    user_text = f"""{extraction_prompt}

TASK:
{prompt}"""
    if file_text:
        user_text += f"\n\nATTACHED FILES:\n{file_text}"

    import time as _time
    text = None

    # ═══ Try Claude first ═══
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    use_claude = os.environ.get("USE_CLAUDE", "true").lower() == "true"
    if use_claude and anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            claude_model = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")

            content = list(claude_image_parts)  # images first
            content.append({"type": "text", "text": user_text})

            for _attempt in range(3):
                try:
                    response = client.messages.create(
                        model=claude_model,
                        max_tokens=1024,
                        system="Extract data as requested. Return ONLY valid JSON, no markdown, no explanation.",
                        messages=[{"role": "user", "content": content}],
                    )
                    text = response.content[0].text.strip()
                    log.info(f"Extraction using Claude ({claude_model})")
                    break
                except anthropic.RateLimitError:
                    wait = min(30, 2 ** _attempt + 2)
                    log.warning(f"Claude extraction 429, retrying in {wait}s")
                    _time.sleep(wait)
                except Exception as e:
                    log.error(f"Claude extraction error: {e}")
                    break
        except ImportError:
            log.warning("anthropic package not available, falling back to Gemini")

    # ═══ Fallback: Vertex AI / Gemini ═══
    if text is None:
        parts = list(gemini_image_parts)
        parts.append({"text": user_text})
        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"maxOutputTokens": 1024, "temperature": 0.0},
        }
        resp = None

        gcp_token = _get_handler_gcp_token()
        if gcp_token:
            vertex_url = ("https://us-central1-aiplatform.googleapis.com/v1/projects/ainm26osl-745"
                          "/locations/us-central1/publishers/google/models/gemini-2.0-flash-001:generateContent")
            headers = {"Authorization": f"Bearer {gcp_token}", "Content-Type": "application/json"}
            for _attempt in range(3):
                try:
                    resp = _requests.post(vertex_url, json=body, headers=headers, timeout=60)
                    if resp.status_code == 429:
                        wait = min(30, 2 ** _attempt + 2)
                        log.warning(f"Vertex extraction 429, retrying in {wait}s")
                        _time.sleep(wait)
                        resp = None
                        continue
                    if resp.status_code == 401:
                        _handler_gcp_token["token"] = None
                        resp = None
                        break
                    if resp.status_code != 200:
                        log.error(f"Vertex extraction error: {resp.status_code} {resp.text[:300]}")
                        resp = None
                        break
                    break
                except Exception as e:
                    log.error(f"Vertex extraction request error: {e}")
                    resp = None
                    break

        if resp is None and api_key:
            FALLBACK_MODEL = "gemini-2.0-flash"
            models_to_try = [model] if model == FALLBACK_MODEL else [model, FALLBACK_MODEL]
            for try_model in models_to_try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{try_model}:generateContent?key={api_key}"
                body_retry = {
                    "contents": [{"role": "user", "parts": parts}],
                    "generationConfig": {"maxOutputTokens": 1024, "temperature": 0.0},
                }
                success = False
                for _attempt in range(2):
                    try:
                        resp = _requests.post(url, json=body_retry, timeout=60)
                        if resp.status_code == 429:
                            if _attempt == 0:
                                log.warning(f"Gemini {try_model} 429, waiting 5s...")
                                _time.sleep(5)
                                continue
                            resp = None
                            break
                        if resp.status_code != 200:
                            log.error(f"Gemini {try_model} error: {resp.status_code} {resp.text[:300]}")
                            resp = None
                            break
                        success = True
                        log.info(f"Extraction using model: {try_model}")
                        break
                    except Exception as e:
                        log.error(f"Extraction request error ({try_model}): {e}")
                        resp = None
                        break
                if success:
                    break

        if resp is None:
            log.error("All extraction APIs failed")
            return {}
        try:
            resp_json = resp.json()
            text = resp_json["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            log.error(f"Failed to parse Gemini response: {e}")
            return {}

    # ═══ Parse JSON from text ═══
    try:
        text = text  # already set by Claude or Gemini path above
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
    postal = {
        "addressLine1": _val(data, "addressLine1", ""),
        "postalCode": _val(data, "postalCode", ""),
        "city": _val(data, "city", ""),
    }
    if postal["addressLine1"] or postal["postalCode"] or postal["city"]:
        postal["country"] = {"id": 161}  # Norway
    body = {
        "name": data.get("name", ""),
        "email": _val(data, "email", ""),
        "phoneNumber": _val(data, "phone", ""),
        "isCustomer": True,
        "language": "NO",
        "invoiceSendMethod": "EMAIL",
        "postalAddress": postal,
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
    value_ids = {}  # name -> id mapping
    for i, val in enumerate(data.get("values", []), start=1):
        raw_number = val.get("number", i)
        try:
            num = int(raw_number)
        except (TypeError, ValueError):
            num = i
        val_name = val.get("name", "")
        val_body = {
            "displayName": val_name,
            "number": num,
            "dimensionIndex": dim_index,
        }
        result, status = _api_post(session, base_url, "/ledger/accountingDimensionValue", val_body, trace)
        if status == 201:
            val_id = result.get("value", {}).get("id")
            if val_id:
                value_ids[val_name.lower()] = val_id

    # Step 3: Create voucher linking dimension if amount and account specified
    acct_num = _val(data, "accountNumber")
    amount = _parse_amount(data.get("amount", 0))
    if acct_num and amount and value_ids:
        acct_id = _get_acct_id(ctx, acct_num)
        bank_id = _get_acct_id(ctx, "1920")
        if acct_id and bank_id:
            voucher_date = _val(data, "voucherDate", _today())
            dim_field = f"freeAccountingDimension{dim_index}" if dim_index and dim_index <= 3 else "freeAccountingDimension1"

            # Find the right dimension value ID
            target_val_name = _val(data, "dimensionValueName", "")
            if target_val_name and target_val_name.lower() in value_ids:
                dim_val_id = value_ids[target_val_name.lower()]
            else:
                # Fallback: use first created value
                dim_val_id = list(value_ids.values())[0]

            postings = [
                {
                    "account": {"id": acct_id},
                    "amountGross": amount,
                    "amountGrossCurrency": amount,
                    "date": voucher_date,
                    "row": 1,
                    "vatType": {"id": 0},
                    dim_field: {"id": dim_val_id},
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
        "invoiceSendMethod": "EMAIL",
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


def _create_product(name, number, price_ex_vat, vat_pct, session, base_url, trace, description="", ctx=None):
    """Create a product. Returns product_id. Checks prefetch + GET before POST to avoid 422."""
    # Check prefetched products first (FREE — no API call)
    if ctx and ctx.get("products"):
        for p in ctx["products"]:
            if str(p.get("number", "")) == str(number):
                log.info(f"Found prefetched product {number}: id={p['id']}")
                return p["id"]
            if p.get("name", "").lower() == name.lower():
                log.info(f"Found prefetched product by name '{name}': id={p['id']}")
                return p["id"]

    # GET to check if exists (GET is FREE — doesn't count for efficiency)
    r, _ = _api_get(session, base_url, "/product", {"number": str(number), "fields": "id,name"}, trace)
    vals = r.get("values", [])
    if vals:
        log.info(f"Found existing product {number}: id={vals[0]['id']}")
        return vals[0]["id"]

    # POST new product
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
    # If 422 (duplicate) — shouldn't happen now but keep as safety
    if status == 422:
        r2, _ = _api_get(session, base_url, "/product", {"number": str(number), "fields": "id"}, trace)
        vals2 = r2.get("values", [])
        if vals2:
            return vals2[0]["id"]
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
        description=_val(data, "description", ""), ctx=ctx,
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
            price_ex, vat_pct, session, base_url, trace, ctx=ctx,
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
            price_ex, vat_pct, session, base_url, trace, ctx=ctx,
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

    # Step 1: Find the EXISTING invoice to credit
    # Search all invoices and match by customer name/org number
    result, status = _api_get(session, base_url, "/invoice", {
        "invoiceDateFrom": "2020-01-01",
        "invoiceDateTo": "2027-01-01",
        "fields": "id,invoiceNumber,amount,amountOutstanding,customer",
    }, trace)

    invoices = result.get("values", [])
    target = None
    cust_name = _val(data, "customerName", "")
    cust_org = _val(data, "customerOrgNumber", "")
    inv_num = _val(data, "invoiceNumber", "")
    price_ex = _parse_amount(data.get("priceExVat", 0))

    # Try matching by invoice number first
    if inv_num:
        for inv in invoices:
            if str(inv.get("invoiceNumber")) == str(inv_num):
                target = inv
                break

    # Try matching by customer org number
    if not target and cust_org:
        org_clean = cust_org.replace(" ", "")
        for inv in invoices:
            c = inv.get("customer", {})
            if str(c.get("organizationNumber", "")).replace(" ", "") == org_clean:
                target = inv
                break

    # Try matching by customer name
    if not target and cust_name:
        cust_lower = cust_name.lower()
        for inv in invoices:
            c = inv.get("customer", {})
            if cust_lower in str(c.get("name", "")).lower():
                target = inv
                break

    # Fallback: first invoice with outstanding amount
    if not target and invoices:
        for inv in invoices:
            if float(inv.get("amountOutstanding", 0)) > 0:
                target = inv
                break
        if not target:
            target = invoices[0]

    if target:
        # Found existing invoice — credit it directly
        invoice_id = target["id"]
        log.info(f"Credit note: found existing invoice {invoice_id} (num={target.get('invoiceNumber')})")
    else:
        # No existing invoice found — create one then credit it
        log.info("Credit note: no existing invoice found, creating new one")
        cust_id = _create_customer_if_needed(data, session, base_url, ctx, trace)
        if not cust_id:
            return trace

        vat_pct = data.get("vatPercent", 25)
        prod_id = _create_product(
            data.get("productName", "Product"),
            data.get("productNumber", "1"),
            price_ex, vat_pct, session, base_url, trace, ctx=ctx,
        )
        if not prod_id:
            return trace

        invoice_id = _create_invoice_flow(
            cust_id,
            [{"product_id": prod_id, "price_ex_vat": price_ex, "vat_pct": vat_pct, "quantity": data.get("quantity", 1)}],
            session, base_url, ctx, trace,
            invoice_date=_val(data, "invoiceDate", _today()),
            due_date=_val(data, "dueDate", _due()),
        )
        if not invoice_id:
            return trace

    # Step 2: Create credit note on the found/created invoice
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
        # v29: use prefetched division (no POST), but keep employment+details (checks 6+7 need them)
        div_id = ctx.get("default_division_id")  # Use prefetched, no POST
        if not div_id:
            div_id = _ensure_division(session, base_url, ctx, trace)  # Fallback: create if needed
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
        log.info(f"Found existing supplier: {existing.get('name')} id={existing['id']}")
        return existing["id"]
    log.info(f"Creating new supplier: {sup_name}")

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

    # Step 2: Determine expense account and 2400 account
    exp_acct_num = str(_val(data, "expenseAccount", "6500"))
    exp_acct_id = _get_acct_id(ctx, exp_acct_num)
    if not exp_acct_id:
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
        log.error("Missing ledger account IDs for supplier invoice")
        return trace

    # Step 3: Parse amounts
    amount_incl = _parse_amount(data.get("amountInclVat", 0))
    amount_ex = _parse_amount(data.get("amountExVat", 0))
    vat_pct = data.get("vatPercent", 25)

    if amount_incl <= 0 and amount_ex > 0:
        amount_incl = round(amount_ex * (1 + vat_pct / 100), 2)
    elif amount_incl <= 0:
        amount_incl = amount_ex  # fallback

    is_food = data.get("isFood", False)
    vat_pct_val = 15 if is_food else (vat_pct if vat_pct else 25)
    vat_rate = vat_pct_val / 100

    if amount_ex <= 0 and amount_incl > 0:
        amount_ex = round(amount_incl / (1 + vat_rate), 2)
    if amount_incl <= 0 and amount_ex > 0:
        amount_incl = round(amount_ex * (1 + vat_rate), 2)

    inv_date = _val(data, "invoiceDate") or _today()
    if not inv_date or len(inv_date) < 8:
        inv_date = _today()
    due_date = _val(data, "dueDate") or ""

    # Calculate due_date as invoiceDate + 30 days if not provided
    if not due_date or due_date == inv_date:
        due_date = (date.fromisoformat(inv_date) + timedelta(days=30)).isoformat()

    inv_num = _val(data, "invoiceNumber", "")
    sup_name = data.get("supplierName", "")

    # Determine vatType ID: 1=ingoing25%, 11=ingoing15%(food)
    # Some accounts (e.g. 7100 Bilgodtgjørelse) are locked to vatType 0
    VAT_LOCKED_ACCOUNTS = {"7100", "7101", "7130", "7140", "7141", "7142", "5000", "5001"}
    if exp_acct_num in VAT_LOCKED_ACCOUNTS:
        vat_type_id = 0
        log.info(f"Account {exp_acct_num} locked to vatType 0, using no-VAT posting")
    else:
        vat_type_id = 11 if is_food else 1

    # ── Step 4: Try POST /supplierInvoice ──
    supplier_invoice_body = {
        "invoiceNumber": inv_num,
        "invoiceDate": inv_date,
        "dueDate": due_date,
        "supplier": {"id": sup_id},
        "voucher": {
            "date": inv_date,
            "description": f"Leverandørfaktura {inv_num} {sup_name}".strip(),
        },
        "postings": [
            {
                "account": {"id": exp_acct_id},
                "amountGross": amount_incl,
                "amountGrossCurrency": amount_incl,
                "date": inv_date,
                "row": 1,
                "vatType": {"id": vat_type_id},
            },
            {
                "account": {"id": acct_2400_id},
                "amountGross": -amount_incl,
                "amountGrossCurrency": -amount_incl,
                "date": inv_date,
                "row": 2,
                "vatType": {"id": 0},
            },
        ],
    }

    result, status = _api_post(session, base_url, "/supplierInvoice", supplier_invoice_body, trace)

    # vatType 422 retry for supplierInvoice
    if status == 422 and vat_type_id != 0:
        log.info(f"supplierInvoice 422 with vatType {vat_type_id}, retrying with vatType 0")
        supplier_invoice_body["postings"][0]["vatType"] = {"id": 0}
        result, status = _api_post(session, base_url, "/supplierInvoice", supplier_invoice_body, trace)

    if status in (200, 201):
        log.info("supplierInvoice succeeded")
        return trace

    # ── Step 5: Fallback to POST /incomingInvoice ──
    if status in (403, 422):
        log.info(f"supplierInvoice returned {status}, trying /incomingInvoice")
        incoming_body = {
            "invoiceHeader": {
                "vendorId": sup_id,
                "invoiceNumber": inv_num,
                "invoiceDate": inv_date,
                "dueDate": due_date,
                "invoiceAmount": amount_incl,
                "description": f"Leverandørfaktura {inv_num} {sup_name}".strip(),
            },
            "orderLines": [
                {
                    "accountId": exp_acct_id,
                    "amountInclVat": amount_incl,
                    "vatTypeId": vat_type_id,
                    "description": data.get("description", "Leverandørfaktura"),
                    "row": 1,
                },
            ],
        }

        result, status = _api_post(session, base_url, "/incomingInvoice", incoming_body, trace,
                                   params={"sendTo": "ledger"})

        # vatType 422 retry for incomingInvoice
        if status == 422 and vat_type_id != 0:
            log.info(f"incomingInvoice 422 with vatType {vat_type_id}, retrying with vatType 0")
            incoming_body["orderLines"][0]["vatTypeId"] = 0
            result, status = _api_post(session, base_url, "/incomingInvoice", incoming_body, trace,
                                       params={"sendTo": "ledger"})

        if status in (200, 201):
            log.info("incomingInvoice succeeded")
            return trace

    # ── Step 6: Final fallback to POST /ledger/voucher with vendorInvoiceNumber ──
    log.info(f"Both supplierInvoice and incomingInvoice failed, falling back to /ledger/voucher")
    postings = [
        {
            "account": {"id": exp_acct_id},
            "amountGross": amount_incl,
            "amountGrossCurrency": amount_incl,
            "date": inv_date,
            "row": 1,
            "vatType": {"id": vat_type_id},
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
    ]

    voucher_body = {
        "date": inv_date,
        "description": f"Leverandørfaktura {inv_num} {sup_name}".strip(),
        "vendorInvoiceNumber": inv_num,
        "postings": postings,
    }

    result, status = _api_post(session, base_url, "/ledger/voucher", voucher_body, trace,
                               params={"sendToLedger": "true"})

    # If 422 due to vatType lock, retry with vatType 0
    if status == 422 and vat_type_id != 0:
        log.info(f"Voucher 422 with vatType {vat_type_id}, retrying with vatType 0")
        postings[0]["vatType"] = {"id": 0}
        voucher_body["postings"] = postings
        _api_post(session, base_url, "/ledger/voucher", voucher_body, trace,
                  params={"sendToLedger": "true"})

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
        # If not found in prefetch, search via API
        if not dept_id:
            try:
                result, _ = _api_get(session, base_url, "/department",
                                     {"name": dept_name, "fields": "id,name"}, trace)
                vals = result.get("values", [])
                if vals:
                    dept_id = vals[0]["id"]
                else:
                    # Create department if not found
                    dept_result = _api_post(session, base_url, "/department",
                                           {"name": dept_name, "departmentNumber": "1"}, trace)
                    if dept_result and isinstance(dept_result, dict):
                        dept_id = dept_result.get("value", {}).get("id") or dept_result.get("id")
            except Exception as e:
                log.warning(f"Department lookup/create failed: {e}")
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
    if not emp and ctx.get("employees"):
        # Fallback: try matching by name parts in email
        if email:
            name_parts = email.split("@")[0].replace(".", " ").lower().split()
            for e in ctx["employees"]:
                full = (e.get("firstName", "") + " " + e.get("lastName", "")).lower()
                if all(p in full for p in name_parts):
                    emp = e
                    log.info(f"Found employee by name match from email: {email} -> {e.get('firstName')} {e.get('lastName')}")
                    break
        if not emp:
            emp = ctx["employees"][0]
            log.warning(f"Employee not found by email: {email}, using first employee: {emp.get('firstName')} {emp.get('lastName')}")
    if not emp:
        log.error(f"No employees available at all")
        return trace

    emp_id = emp["id"]

    # Ensure dateOfBirth is set (required for employment)
    if not emp.get("dateOfBirth"):
        put_body = {
            "id": emp_id,
            "firstName": emp.get("firstName", ""),
            "lastName": emp.get("lastName", ""),
            "email": emp.get("email", ""),
            "dateOfBirth": "1990-05-15",
            "userType": emp.get("userType", "STANDARD"),
            "version": emp.get("version"),
        }
        _api_put(session, base_url, f"/employee/{emp_id}", trace, body=put_body)

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

    # Step 3: Create payslip with specifications
    base_salary = _parse_amount(data.get("baseSalary", 0))
    bonus = _parse_amount(data.get("bonus", 0))
    sal_month = int(data.get("month") or date.today().month)
    sal_year = int(data.get("year") or date.today().year)
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

    if not specifications:
        return trace

    # POST /salary/paySlip — create payslip with specs included (1 write call)
    payslip_body = {
        "employee": {"id": emp_id},
        "date": sal_date,
        "year": sal_year,
        "month": sal_month,
        "specifications": specifications,
    }
    result, status = _api_post(session, base_url, "/salary/paySlip", payslip_body, trace)

    if status in (200, 201):
        # PaySlip worked — calculate and finalize
        payslip_id = None
        val = result.get("value", result)
        if isinstance(val, dict):
            payslip_id = val.get("id")

        if payslip_id:
            _api_put(session, base_url, f"/salary/paySlip/{payslip_id}/:calculate", trace)
            _api_put(session, base_url, f"/salary/paySlip/{payslip_id}/:createPayment", trace)
            ctx["_salary_transaction_done"] = True
            return trace

    # PaySlip failed — try /salary/transaction as second attempt
    log.warning(f"PaySlip failed ({status}), trying /salary/transaction")
    total_salary = base_salary + bonus
    txn_body = {
        "employee": {"id": emp_id},
        "date": sal_date,
        "year": sal_year,
        "month": sal_month,
        "amount": total_salary,
        "description": f"Lønn {emp.get('firstName', '')} {emp.get('lastName', '')}",
    }
    txn_result, txn_status = _api_post(session, base_url, "/salary/transaction", txn_body, trace)
    if txn_status in (200, 201):
        ctx["_salary_transaction_done"] = True
        return trace

    # Both salary APIs failed — fallback to manual voucher per oppgavetekst
    # "Dersom lønns-API-et ikke fungerer, bokfør lønnsbilag manuelt"
    # Accounts: 5000 (lønnskostnad), 2780 (skattetrekk), 2000 (skyldige feriepenger), 1920 (bank)
    log.warning(f"salary/transaction also failed ({txn_status}), using manual voucher fallback")

    def _get_or_fetch(acct_num):
        aid = _get_acct_id(ctx, acct_num)
        if not aid:
            r, _ = _api_get(session, base_url, "/ledger/account", {"number": acct_num, "fields": "id"}, trace)
            vals = r.get("values", [])
            if vals:
                aid = vals[0]["id"]
        return aid

    acct_5000_id = _get_or_fetch("5000")  # Lønnskostnad (debit)
    acct_2780_id = _get_or_fetch("2780")  # Skattetrekk (credit)
    acct_2000_id = _get_or_fetch("2000")  # Skyldige feriepenger (credit)
    bank_id = _get_or_fetch("1920")       # Bank (credit — net pay)

    if not acct_5000_id or not bank_id:
        log.error(f"Cannot create salary voucher — missing accounts: 5000={acct_5000_id}, 1920={bank_id}")
        return trace

    # Rates per oppgavetekst: skattetrekk ~33%, feriepenger ~12%
    tax_rate = 0.33
    holiday_rate = 0.12
    tax_amount = round(total_salary * tax_rate, 2)
    holiday_amount = round(total_salary * holiday_rate, 2)
    net_salary = round(total_salary - tax_amount - holiday_amount, 2)

    emp_name = f"{emp.get('firstName', '')} {emp.get('lastName', '')}".strip()

    # Lønnsbilag: debit 5000, credit 2780 + 2000 + 1920 (must balance to zero)
    postings = [
        {
            "account": {"id": acct_5000_id},
            "amountGross": total_salary,
            "amountGrossCurrency": total_salary,
            "date": sal_date, "row": 1, "vatType": {"id": 0},
        },
    ]
    row = 2
    if acct_2780_id:
        postings.append({
            "account": {"id": acct_2780_id},
            "amountGross": -tax_amount,
            "amountGrossCurrency": -tax_amount,
            "date": sal_date, "row": row, "vatType": {"id": 0},
        })
        row += 1
    else:
        # No 2780 — add tax to net pay (credit to bank)
        net_salary = round(net_salary + tax_amount, 2)

    if acct_2000_id:
        postings.append({
            "account": {"id": acct_2000_id},
            "amountGross": -holiday_amount,
            "amountGrossCurrency": -holiday_amount,
            "date": sal_date, "row": row, "vatType": {"id": 0},
        })
        row += 1
    else:
        # No 2000 — add holiday pay to net pay (credit to bank)
        net_salary = round(net_salary + holiday_amount, 2)

    postings.append({
        "account": {"id": bank_id},
        "amountGross": -net_salary,
        "amountGrossCurrency": -net_salary,
        "date": sal_date, "row": row, "vatType": {"id": 0},
    })

    voucher1 = {
        "date": sal_date,
        "description": f"Lønn {emp_name} - {sal_month}/{sal_year}",
        "postings": postings,
    }
    _api_post(session, base_url, "/ledger/voucher", voucher1, trace, params={"sendToLedger": "true"})

    ctx["_salary_transaction_done"] = True
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

    # Pre-fetch accounts we'll need
    get_acct("8700")  # Tax expense
    get_acct("2920")  # Tax payable
    get_acct("2050")  # Tax payable fallback
    get_acct("8900")  # Result disposition
    get_acct("2050")  # Equity / retained earnings
    get_acct("2800")  # Equity fallback

    # Step 0: GET ledger postings FIRST (before creating vouchers) for accurate calculation
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
    financial = 0.0  # 8xxx accounts (financial income/expense)
    for p in postings:
        acct = p.get("account", {})
        acct_num = str(acct.get("number", ""))
        amt = float(p.get("amount", 0))
        if acct_num.startswith("3"):  # Revenue accounts
            revenue += amt
        elif acct_num.startswith(("4", "5", "6", "7")):  # Operating expense accounts
            expenses += amt
        elif acct_num.startswith("8") and not acct_num.startswith(("87", "89")):  # Financial income/expense (exclude tax 87xx and result disposition 89xx)
            financial += amt

    total_depreciation = 0.0

    assets_list = data.get("assets", [])
    log.info(f"Year-end: {len(assets_list)} assets to depreciate: {[a.get('name','?') for a in assets_list]}")

    # Step 1: Depreciation vouchers (one per asset)
    for asset in assets_list:
        dep_amount = _parse_amount(asset.get("depreciationAmount", 0))
        if dep_amount <= 0:
            cost = _parse_amount(asset.get("originalCost", 0))
            years = asset.get("usefulLifeYears", 1)
            if years > 0:
                dep_amount = round(cost / years, 2)

        if dep_amount <= 0:
            log.warning(f"Year-end: skipping asset {asset.get('name','?')} — dep_amount={dep_amount}")
            continue

        total_depreciation += dep_amount
        log.info(f"Year-end: asset={asset.get('name','?')} dep={dep_amount}")

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
    prepaid_list = data.get("prepaidExpenses", [])
    log.info(f"Year-end: {len(prepaid_list)} prepaid expenses to reverse")
    total_prepaid = 0.0
    for prepaid in prepaid_list:
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
        sal_debit_acct = get_acct(sal_accrual.get("expenseAccount", "5000"))
        sal_credit_acct = get_acct(sal_accrual.get("liabilityAccount", "2900"))
        if sal_debit_acct and sal_credit_acct:
            v_body = {
                "date": voucher_date,
                "description": sal_accrual.get("description", "Lonnsavsetning"),
                "postings": [
                    {
                        "account": {"id": sal_debit_acct},
                        "amountGross": sal_amt,
                        "amountGrossCurrency": sal_amt,
                        "date": voucher_date,
                        "row": 1,
                        "vatType": {"id": 0},
                    },
                    {
                        "account": {"id": sal_credit_acct},
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
    # Revenue is typically negative (credit), expenses positive (debit)
    # financial includes 8xxx (financial income negative, financial expense positive)
    # We fetched postings BEFORE creating vouchers, so we need to ADD our new costs
    profit = -(revenue) - expenses - financial - total_depreciation - total_prepaid - sal_amt
    tax_rate = data.get("taxRate", 0.22)
    tax = round(max(0, profit * tax_rate), 2)

    log.info(f"Year-end: revenue={revenue}, expenses={expenses}, financial={financial}, "
             f"dep={total_depreciation}, prepaid={total_prepaid}, sal={sal_amt}, "
             f"profit={profit}, tax={tax}")

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

    # Step 5: Result disposition voucher (debit 8900, credit equity)
    # Profit: debit 8900, credit 2050. Loss: debit 2050, credit 8900.
    net_profit = round(profit - tax, 2)
    log.info(f"Year-end: net_profit={net_profit}, includeResultDisposition={data.get('includeResultDisposition', True)}")
    if abs(net_profit) > 0.01 and data.get("includeResultDisposition", True):
        acct_8900 = get_acct("8900")
        acct_equity = get_acct(data.get("equityAccount", "2050"))
        if not acct_equity:
            acct_equity = get_acct("2800")
        if acct_8900 and acct_equity:
            v_body = {
                "date": voucher_date,
                "description": f"Resultatdisponering {year}",
                "postings": [
                    {
                        "account": {"id": acct_8900},
                        "amountGross": net_profit,
                        "amountGrossCurrency": net_profit,
                        "date": voucher_date,
                        "row": 1,
                        "vatType": {"id": 0},
                    },
                    {
                        "account": {"id": acct_equity},
                        "amountGross": -net_profit,
                        "amountGrossCurrency": -net_profit,
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
            "dateOfBirth": emp.get("dateOfBirth") or "1990-05-15",
            "userType": "EXTENDED",
            "version": emp.get("version"),
        }
        _api_put(session, base_url, f"/employee/{emp_id}", trace, body=put_body)

        # Step 5: Grant entitlements via POST /employee/entitlement
        company_id = ctx.get("_company_id")
        if company_id:
            for ent_id in (45, 10):
                ent_body = {
                    "employee": {"id": emp_id},
                    "entitlementId": ent_id,
                    "customer": {"id": company_id},
                }
                _api_post(session, base_url, "/employee/entitlement", ent_body, trace)

        # Step 6: Create projects for top accounts
        for acct_num, acct_name, increase in top_accounts:
            proj_body = {
                "name": acct_name if acct_name else f"Kostnadsanalyse {acct_num}",
                "startDate": today,
                "projectManager": {"id": emp_id},
                "isInternal": True,
            }
            result, status = _api_post(session, base_url, "/project", proj_body, trace)

            # Create activity linked to this project
            if status in (200, 201):
                proj_id = result.get("value", {}).get("id")
                act_body = {
                    "name": acct_name if acct_name else f"Analyse {acct_num}",
                    "activityType": "GENERAL_ACTIVITY",
                    "isGeneral": True,
                }
                r, s = _api_post(session, base_url, "/activity", act_body, trace)
                if s >= 400 and proj_id:
                    # Retry with PROJECT_GENERAL_ACTIVITY linked to project
                    act_body2 = {
                        "name": acct_name if acct_name else f"Analyse {acct_num}",
                        "activityType": "PROJECT_GENERAL_ACTIVITY",
                        "project": {"id": proj_id},
                    }
                    _api_post(session, base_url, "/activity", act_body2, trace)

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
            "dateOfBirth": emp.get("dateOfBirth") or "1990-05-15",
            "userType": "EXTENDED",
            "version": emp.get("version"),
        }
        _api_put(session, base_url, f"/employee/{emp_id}", trace, body=put_body)

    # Grant entitlements via POST /employee/entitlement
    company_id = ctx.get("_company_id")
    if company_id:
        for ent_id in (45, 10):
            ent_body = {
                "employee": {"id": emp_id},
                "entitlementId": ent_id,
                "customer": {"id": company_id},
            }
            _api_post(session, base_url, "/employee/entitlement", ent_body, trace)

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

    # Create project with fixed price set directly in POST
    fixed_price = _parse_amount(data.get("fixedPrice", 0))
    proj_body = {
        "name": data.get("projectName", "Project"),
        "startDate": _val(data, "startDate", _today()),
        "projectManager": {"id": emp_id},
        "isFixedPrice": True,
        "fixedprice": fixed_price,
    }
    end_date = _val(data, "endDate")
    if end_date:
        proj_body["endDate"] = end_date
    desc = _val(data, "description")
    if desc:
        proj_body["description"] = desc
    if cust_id:
        proj_body["customer"] = {"id": cust_id}

    result, status = _api_post(session, base_url, "/project", proj_body, trace)
    if status != 201:
        return trace

    proj_id = result.get("value", {}).get("id")

    if not proj_id:
        return trace

    # Create partial/milestone invoice if requested
    pct = data.get("partialInvoicePercent", 0)
    if pct > 0 and fixed_price > 0 and cust_id:
        partial_amount = round(fixed_price * pct / 100, 2)
        # Use numeric product number to avoid GET /product 422
        prod_number = str(10000 + (proj_id % 90000))
        # Create product for milestone invoice
        prod_id = _create_product(
            f"Milepæl {data.get('projectName', 'Prosjekt')}",
            prod_number,
            partial_amount, 25, session, base_url, trace, ctx=ctx,
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

    # Create a project-compatible activity (isProjectActivity=True)
    activity_id = None
    act_body = {"name": _val(data, "description", "Arbeid"), "isGeneral": True}
    r, s = _api_post(session, base_url, "/activity", act_body, trace)
    if s == 201:
        activity_id = r.get("value", {}).get("id")
    if not activity_id:
        # Fallback: try general activity with alternate name
        act_body2 = {"name": _val(data, "description", "Arbeid") + " Alt", "isGeneral": True}
        r2, s2 = _api_post(session, base_url, "/activity", act_body2, trace)
        if s2 == 201:
            activity_id = r2.get("value", {}).get("id")
    if not activity_id:
        # Last fallback: use pre-fetched activity
        activities = ctx.get("activities", [])
        if activities:
            activity_id = activities[0]["id"]

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
# HANDLER: TRAVEL EXPENSE
# ═══════════════════════════════════════════════════════════

def handle_travel_expense(data, session, base_url, ctx):
    trace = []
    today = _today()

    # Step 1: Find employee
    email = data.get("employeeEmail", "")
    emp = _find_employee_by_email(ctx, email)
    if not emp and ctx.get("employees"):
        emp = ctx["employees"][0]
    if not emp:
        log.error("No employee found for travel expense")
        return trace

    emp_id = emp["id"]

    # Ensure employee is EXTENDED + has employment
    _ensure_employee_extended(emp, session, base_url, ctx, trace)

    has_employment = bool(emp.get("_employments"))
    if not has_employment:
        div_id = _ensure_division(session, base_url, ctx, trace)
        if div_id:
            empl_body = {
                "employee": {"id": emp_id},
                "startDate": _val(data, "departureDate", today),
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
                        "date": _val(data, "departureDate", today),
                        "employmentType": "ORDINARY",
                        "employmentForm": "PERMANENT",
                        "remunerationType": "MONTHLY_WAGE",
                        "workingHoursScheme": "NOT_SHIFT",
                        "percentageOfFullTimeEquivalent": 100.0,
                    }
                    _api_post(session, base_url, "/employee/employment/details", det_body, trace)

    # Step 2: POST /travelExpense
    dep_date = _val(data, "departureDate", today)
    ret_date = _val(data, "returnDate", dep_date)
    travel_body = {
        "employee": {"id": emp_id},
        "title": _val(data, "title", "Reiseregning"),
        "travelDetails": {
            "departureDate": dep_date,
            "returnDate": ret_date,
            "departureFrom": _val(data, "departureFrom", ""),
            "destination": _val(data, "destination", ""),
            "purpose": _val(data, "purpose", _val(data, "title", "Reise")),
        },
    }
    result, status = _api_post(session, base_url, "/travelExpense", travel_body, trace)
    if status != 201:
        return trace

    travel_id = result.get("value", {}).get("id")
    if not travel_id:
        return trace

    # Step 3: Per diem compensation if overnight stay
    if data.get("overnightStay"):
        rate_cat_id = ctx.get("per_diem_overnight_id")
        if not rate_cat_id:
            rate_cat_id = ctx.get("per_diem_day_id")
        if rate_cat_id:
            # Calculate days from departure to return
            try:
                from datetime import datetime as _dt
                _dep_dt = _dt.strptime(dep_date, "%Y-%m-%d")
                _ret_dt = _dt.strptime(ret_date, "%Y-%m-%d")
                per_diem_days = (_ret_dt - _dep_dt).days + 1
            except Exception:
                per_diem_days = 1
            per_diem_rate = _parse_amount(data.get("perDiemRate", 800))
            pd_body = {
                "travelExpense": {"id": travel_id},
                "rateCategory": {"id": rate_cat_id},
                "count": per_diem_days,
                "rate": per_diem_rate,
                "overnightAccommodation": "HOTEL",
                "location": _val(data, "destination", "Norge"),
            }
            _api_post(session, base_url, "/travelExpense/perDiemCompensation", pd_body, trace)

    # Step 4: Costs
    pt_id = _get_payment_type_id(ctx)
    for cost in data.get("costs", []):
        amount = _parse_amount(cost.get("amount", 0))
        if amount <= 0:
            continue

        # Match cost category from prefetched context
        cc_id = None
        cat = cost.get("category", "other").lower()
        for cc in ctx.get("cost_categories", []):
            desc = str(cc.get("description", "")).lower()
            if cat in desc or desc in cat:
                cc_id = cc["id"]
                break
        if not cc_id and ctx.get("cost_categories"):
            cc_id = ctx["cost_categories"][0]["id"]

        cost_body = {
            "travelExpense": {"id": travel_id},
            "description": cost.get("description", cost.get("category", "Utgift")),
            "amountCurrencyIncVat": amount,
            "currency": {"id": 1},
            "date": cost.get("date", dep_date),
            "paymentType": {"id": pt_id} if pt_id else {"id": 1},
        }
        if cc_id:
            cost_body["costCategory"] = {"id": cc_id}
        _api_post(session, base_url, "/travelExpense/cost", cost_body, trace)

    return trace


# ═══════════════════════════════════════════════════════════
# HANDLER: DELETE TRAVEL EXPENSE
# ═══════════════════════════════════════════════════════════

def handle_delete_travel(data, session, base_url, ctx):
    trace = []

    # Step 1: GET travel expenses
    result, status = _api_get(session, base_url, "/travelExpense", {
        "fields": "id,title,status,employee(email)",
        "count": "100",
    }, trace)

    travels = result.get("values", [])
    if not travels:
        log.error("No travel expenses found")
        return trace

    # Step 2: Find the right one
    target = None
    travel_id_hint = data.get("travelExpenseId", 0)
    title_hint = _val(data, "travelExpenseTitle", "")
    email_hint = _val(data, "employeeEmail", "")

    if travel_id_hint:
        for t in travels:
            if t.get("id") == travel_id_hint:
                target = t
                break

    if not target and title_hint:
        title_lower = title_hint.lower()
        for t in travels:
            if title_lower in str(t.get("title", "")).lower():
                target = t
                break

    if not target and email_hint:
        email_lower = email_hint.lower()
        for t in travels:
            e = t.get("employee", {})
            if email_lower in str(e.get("email", "")).lower():
                target = t
                break

    if not target:
        target = travels[-1]

    travel_id = target["id"]
    travel_status = str(target.get("status", "")).upper()

    # Step 3: Unapprove if APPROVED
    if travel_status == "APPROVED":
        _api_put(session, base_url, f"/travelExpense/{travel_id}/:unapprove", trace)

    # Step 4: DELETE
    _api_delete(session, base_url, f"/travelExpense/{travel_id}", trace)

    return trace


# ═══════════════════════════════════════════════════════════
# HANDLER: BANK RECONCILIATION
# ═══════════════════════════════════════════════════════════

def handle_bank_recon(data, session, base_url, ctx):
    trace = []
    today = _today()

    transactions = data.get("transactions", [])
    if not transactions:
        log.error("No transactions to reconcile")
        return trace

    date_from = _val(data, "dateFrom", "2020-01-01")
    date_to = _val(data, "dateTo", "2027-01-01")

    # Step 1: GET customer invoices (for incoming payments matching)
    result, status = _api_get(session, base_url, "/invoice", {
        "invoiceDateFrom": date_from,
        "invoiceDateTo": date_to,
        "fields": "id,invoiceNumber,amount,amountOutstanding,customer",
        "count": "1000",
    }, trace)
    invoices = result.get("values", []) if status < 400 else []

    pt_id = _get_payment_type_id(ctx)

    # Track which invoices have been matched to avoid double-matching
    matched_inv_ids = set()

    for txn in transactions:
        amount = _parse_amount(txn.get("amount", 0))
        txn_date = txn.get("date", today)
        reference = str(txn.get("reference", "")).strip()
        description = str(txn.get("description", "")).strip()
        counterparty = str(txn.get("counterparty", "")).strip()

        if amount > 0:
            # ── Incoming payment: match to customer invoice → PUT /invoice/{id}/:payment ──
            matched = None

            # Extract potential invoice numbers from reference and description
            ref_numbers = set()
            if reference:
                ref_numbers.add(reference)
                # Also extract pure digit sequences
                nums = re.findall(r'\d+', reference)
                ref_numbers.update(nums)
            if description:
                nums = re.findall(r'(?:faktura|fakt\.?|inv(?:oice)?|ref\.?|facture|rechnung)\s*#?\s*(\d+)', description, re.IGNORECASE)
                ref_numbers.update(nums)
                # Also try standalone numbers in description
                if not nums:
                    standalone = re.findall(r'\b(\d{3,})\b', description)
                    ref_numbers.update(standalone)

            # Match 1: by invoice number matching reference
            if ref_numbers:
                for inv in invoices:
                    if inv["id"] in matched_inv_ids:
                        continue
                    inv_num = str(inv.get("invoiceNumber", ""))
                    if inv_num and inv_num in ref_numbers:
                        matched = inv
                        break

            # Match 2: by customer name matching counterparty/description
            if not matched and (counterparty or description):
                search_text = (counterparty + " " + description).lower()
                for inv in invoices:
                    if inv["id"] in matched_inv_ids:
                        continue
                    outstanding = float(inv.get("amountOutstanding", inv.get("amount", 0)))
                    if outstanding <= 0:
                        continue
                    cust = inv.get("customer", {})
                    cust_name = str(cust.get("name", "")).lower() if isinstance(cust, dict) else ""
                    if cust_name and len(cust_name) > 2 and cust_name in search_text:
                        matched = inv
                        break

            # Match 3: by exact amount outstanding
            if not matched:
                for inv in invoices:
                    if inv["id"] in matched_inv_ids:
                        continue
                    outstanding = float(inv.get("amountOutstanding", inv.get("amount", 0)))
                    if outstanding > 0 and abs(outstanding - amount) < 1.0:
                        matched = inv
                        break

            # Match 4: partial payment — find invoice with outstanding > payment amount
            if not matched:
                for inv in invoices:
                    if inv["id"] in matched_inv_ids:
                        continue
                    outstanding = float(inv.get("amountOutstanding", inv.get("amount", 0)))
                    if outstanding > 0 and amount < outstanding:
                        matched = inv
                        break

            # Match 5: any invoice with outstanding > 0
            if not matched:
                for inv in invoices:
                    if inv["id"] in matched_inv_ids:
                        continue
                    outstanding = float(inv.get("amountOutstanding", inv.get("amount", 0)))
                    if outstanding > 0:
                        matched = inv
                        break

            if matched:
                matched_inv_ids.add(matched["id"])
                inv_id = matched["id"]
                params = {
                    "paymentDate": txn_date,
                    "paymentTypeId": str(pt_id) if pt_id else "1",
                    "paidAmount": str(amount),
                }
                _api_put(session, base_url, f"/invoice/{inv_id}/:payment", trace, params=params)
            else:
                log.warning(f"No matching customer invoice for incoming payment {amount} ref={reference}")

        elif amount < 0:
            # ── Outgoing payment: voucher debit 2400 (leverandorgjeld), credit 1920 (bank) ──
            abs_amount = abs(amount)
            desc = description or reference or counterparty or "Utgaende betaling"
            voucher_body = {
                "date": txn_date,
                "description": desc[:200],
                "postings": [
                    {
                        "account": {"id": 0, "number": 2400},
                        "amountGross": abs_amount,
                        "amountGrossCurrency": abs_amount,
                        "date": txn_date,
                        "row": 1,
                        "vatType": {"id": 0},
                    },
                    {
                        "account": {"id": 0, "number": 1920},
                        "amountGross": -abs_amount,
                        "amountGrossCurrency": -abs_amount,
                        "date": txn_date,
                        "row": 2,
                        "vatType": {"id": 0},
                    },
                ],
            }
            _api_post(session, base_url, "/ledger/voucher", voucher_body, trace, params={"sendToLedger": "true"})

    return trace


# ═══════════════════════════════════════════════════════════
# HANDLER: PROJECT LIFECYCLE
# ═══════════════════════════════════════════════════════════

def handle_project_lifecycle(data, session, base_url, ctx):
    trace = []
    today = _today()

    # Step 1: Setup employees
    emp_objects = []
    for emp_data in data.get("employees", []):
        email = emp_data.get("email", "")
        emp = _find_employee_by_email(ctx, email)
        if not emp and ctx.get("employees"):
            emp = ctx["employees"][0]
        if emp:
            _ensure_employee_extended(emp, session, base_url, ctx, trace)
            has_employment = bool(emp.get("_employments"))
            if not has_employment:
                div_id = _ensure_division(session, base_url, ctx, trace)
                if div_id:
                    empl_body = {
                        "employee": {"id": emp["id"]},
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
            emp_objects.append({"emp": emp, "data": emp_data})

    if not emp_objects:
        log.error("No employees found for project lifecycle")
        return trace

    # Step 2: Create/find customer
    cust_id = _create_customer_if_needed(data, session, base_url, ctx, trace)

    # Step 3: Create project
    pm_id = emp_objects[0]["emp"]["id"]
    proj_body = {
        "name": data.get("projectName", "Project"),
        "startDate": today,
        "projectManager": {"id": pm_id},
    }
    if cust_id:
        proj_body["customer"] = {"id": cust_id}
    budget = _parse_amount(data.get("budget", 0))
    if budget > 0:
        proj_body["isFixedPrice"] = True
        proj_body["fixedprice"] = budget

    result, status = _api_post(session, base_url, "/project", proj_body, trace)
    proj_id = None
    if status == 201:
        proj_id = result.get("value", {}).get("id")

    if not proj_id:
        return trace

    # Step 4: Get/create activity for the project
    activity_id = None

    # First try to GET existing project activities
    proj_detail, ps = _api_get(session, base_url, f"/project/{proj_id}",
        {"fields": "id,projectActivities(id,activity(id,name))"}, trace)
    if ps == 200:
        pa_list = proj_detail.get("value", {}).get("projectActivities", [])
        if pa_list:
            activity_id = pa_list[0].get("activity", {}).get("id")
            log.info(f"Using existing project activity: {activity_id}")

    # If no existing activity, create one
    if not activity_id:
        act_body = {
            "name": "Prosjektarbeid",
            "activityType": "PROJECT_GENERAL_ACTIVITY",
            "project": {"id": proj_id},
        }
        r, s = _api_post(session, base_url, "/activity", act_body, trace)
        if s in (200, 201):
            activity_id = r.get("value", {}).get("id")

    # If still no activity, try GENERAL_ACTIVITY
    if not activity_id:
        act_body2 = {
            "name": "Prosjektarbeid",
            "activityType": "GENERAL_ACTIVITY",
            "isGeneral": True,
        }
        r2, s2 = _api_post(session, base_url, "/activity", act_body2, trace)
        if s2 in (200, 201):
            activity_id = r2.get("value", {}).get("id")

    # Step 5: Timesheet entries
    if activity_id:
        for emp_obj in emp_objects:
            hours = _parse_amount(emp_obj["data"].get("hours", 0))
            entry_date = _val(emp_obj["data"], "date", today)
            if hours > 0:
                ts_body = {
                    "project": {"id": proj_id},
                    "activity": {"id": activity_id},
                    "employee": {"id": emp_obj["emp"]["id"]},
                    "date": entry_date,
                    "hours": hours,
                }
                _api_post(session, base_url, "/timesheet/entry", ts_body, trace)

    # Step 6: Supplier cost vouchers
    def get_acct(num):
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

    for sc in data.get("supplierCosts", []):
        amount = _parse_amount(sc.get("amount", 0))
        if amount <= 0:
            continue

        # Create supplier if needed
        sup_id = None
        sup_name = sc.get("supplierName", "")
        if sup_name:
            sup_data = {"supplierName": sup_name, "supplierOrgNumber": sc.get("supplierOrgNumber", "")}
            sup_id = _create_supplier_if_needed(sup_data, session, base_url, ctx, trace)

        exp_acct_num = sc.get("expenseAccount", "6500")
        exp_acct_id = get_acct(exp_acct_num)
        acct_2400 = get_acct("2400")
        vat_pct = sc.get("vatPercent", 25)
        expense_vat_type = {25: 1, 15: 11, 0: 0}.get(vat_pct, 1)

        if exp_acct_id and acct_2400:
            credit_posting = {"account": {"id": acct_2400}, "amountGross": -amount, "amountGrossCurrency": -amount, "date": today, "row": 2, "vatType": {"id": 0}}
            if sup_id:
                credit_posting["supplier"] = {"id": sup_id}
            v_body = {
                "date": today,
                "description": sc.get("description", f"Leverandorkostnad {sup_name}"),
                "postings": [
                    {"account": {"id": exp_acct_id}, "amountGross": amount, "amountGrossCurrency": amount, "date": today, "row": 1, "vatType": {"id": expense_vat_type}},
                    credit_posting,
                ],
            }
            _api_post(session, base_url, "/ledger/voucher", v_body, trace, params={"sendToLedger": "true"})

    # Step 7: Customer invoice
    if cust_id and data.get("invoiceProducts"):
        products_info = []
        for i, p in enumerate(data.get("invoiceProducts", [])):
            price_ex = _parse_amount(p.get("priceExVat", 0))
            vat_pct = p.get("vatPercent", 25)
            prod_id = _create_product(
                p.get("name", f"Product {i+1}"),
                p.get("number", str(i+1)),
                price_ex, vat_pct, session, base_url, trace, ctx=ctx,
            )
            if prod_id:
                products_info.append({
                    "product_id": prod_id,
                    "price_ex_vat": price_ex,
                    "vat_pct": vat_pct,
                    "quantity": p.get("quantity", 1),
                })

        if products_info:
            _create_invoice_flow(cust_id, products_info, session, base_url, ctx, trace)

    return trace


# ═══════════════════════════════════════════════════════════
# HANDLER: FX INVOICE (FOREIGN CURRENCY)
# ═══════════════════════════════════════════════════════════

def handle_fx_invoice(data, session, base_url, ctx):
    trace = []
    today = _today()

    # Step 1: Find/create customer
    cust_id = _create_customer_if_needed(data, session, base_url, ctx, trace)

    # Step 2: Find the invoice
    result, status = _api_get(session, base_url, "/invoice", {
        "invoiceDateFrom": "2020-01-01",
        "invoiceDateTo": "2027-01-01",
        "fields": "id,invoiceNumber,amount,amountOutstanding,customer",
        "count": "200",
    }, trace)

    invoices = result.get("values", [])
    target = None
    inv_num = _val(data, "invoiceNumber", "")
    cust_name = _val(data, "customerName", "")
    amount = _parse_amount(data.get("invoiceAmount", 0))

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
        for inv in invoices:
            if float(inv.get("amountOutstanding", 0)) > 0:
                target = inv
                break

    # Step 3: Pay the invoice
    if target:
        inv_id = target["id"]
        pay_amount = float(target.get("amountOutstanding", target.get("amount", 0)))
        pt_id = _get_payment_type_id(ctx)
        pay_date = _val(data, "paymentDate", today)
        params = {
            "paymentDate": pay_date,
            "paymentTypeId": str(pt_id) if pt_id else "1",
            "paidAmount": str(pay_amount),
        }
        _api_put(session, base_url, f"/invoice/{inv_id}/:payment", trace, params=params)

    # Step 4: FX difference voucher
    old_rate = _parse_amount(data.get("oldRate", 0))
    new_rate = _parse_amount(data.get("newRate", 0))
    if old_rate > 0 and new_rate > 0 and amount > 0:
        amount_nok_old = round(amount * old_rate, 2)
        amount_nok_new = round(amount * new_rate, 2)
        fx_diff = round(amount_nok_new - amount_nok_old, 2)

        def get_acct(num):
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

        if fx_diff != 0:
            acct_1500 = get_acct("1500")
            pay_date = _val(data, "paymentDate", today)

            if fx_diff > 0:
                # FX gain: debit 1500, credit 8060
                acct_fx = get_acct("8060")
                if acct_1500 and acct_fx:
                    v_body = {
                        "date": pay_date,
                        "description": f"Valutagevinst {data.get('currency', 'FX')}",
                        "postings": [
                            {"account": {"id": acct_1500}, "amountGross": fx_diff, "amountGrossCurrency": fx_diff, "date": pay_date, "row": 1, "vatType": {"id": 0}, **({"customer": {"id": cust_id}} if cust_id else {})},
                            {"account": {"id": acct_fx}, "amountGross": -fx_diff, "amountGrossCurrency": -fx_diff, "date": pay_date, "row": 2, "vatType": {"id": 0}},
                        ],
                    }
                    _api_post(session, base_url, "/ledger/voucher", v_body, trace, params={"sendToLedger": "true"})
            else:
                # FX loss: debit 8160, credit 1500
                acct_fx = get_acct("8160")
                abs_diff = abs(fx_diff)
                if acct_1500 and acct_fx:
                    v_body = {
                        "date": pay_date,
                        "description": f"Valutatap {data.get('currency', 'FX')}",
                        "postings": [
                            {"account": {"id": acct_fx}, "amountGross": abs_diff, "amountGrossCurrency": abs_diff, "date": pay_date, "row": 1, "vatType": {"id": 0}},
                            {"account": {"id": acct_1500}, "amountGross": -abs_diff, "amountGrossCurrency": -abs_diff, "date": pay_date, "row": 2, "vatType": {"id": 0}, **({"customer": {"id": cust_id}} if cust_id else {})},
                        ],
                    }
                    _api_post(session, base_url, "/ledger/voucher", v_body, trace, params={"sendToLedger": "true"})

    return trace


# ═══════════════════════════════════════════════════════════
# HANDLER: REMINDER FEE
# ═══════════════════════════════════════════════════════════

def handle_reminder_fee(data, session, base_url, ctx):
    trace = []
    today = _today()

    # Step 1: Find overdue invoice
    result, status = _api_get(session, base_url, "/invoice", {
        "invoiceDateFrom": "2020-01-01",
        "invoiceDateTo": "2027-01-01",
        "fields": "id,invoiceNumber,amount,amountOutstanding,customer",
        "count": "200",
    }, trace)

    invoices = result.get("values", [])
    cust_name = _val(data, "customerName", "")
    inv_num = _val(data, "invoiceNumber", "")
    target = None

    for inv in invoices:
        if inv_num and str(inv.get("invoiceNumber")) == str(inv_num):
            target = inv
            break
    if not target and cust_name:
        cust_lower = cust_name.lower()
        for inv in invoices:
            c = inv.get("customer", {})
            outstanding = float(inv.get("amountOutstanding", 0))
            if cust_lower in str(c.get("name", "")).lower() and outstanding > 0:
                target = inv
                break
    if not target:
        for inv in invoices:
            if float(inv.get("amountOutstanding", 0)) > 0:
                target = inv
                break

    # Step 2: Reminder voucher: debit 1500, credit 3400
    reminder_fee = _parse_amount(data.get("reminderFee", 35))
    reminder_date = _val(data, "reminderDate", today)

    def get_acct(num):
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

    acct_1500 = get_acct("1500")
    acct_3400 = get_acct("3400")
    reminder_cust_id = target.get("customer", {}).get("id") if target else None
    if not reminder_cust_id:
        reminder_cust_id = _create_customer_if_needed(data, session, base_url, ctx, trace)
    if acct_1500 and acct_3400:
        posting_1500 = {"account": {"id": acct_1500}, "amountGross": reminder_fee, "amountGrossCurrency": reminder_fee, "date": reminder_date, "row": 1, "vatType": {"id": 0}}
        if reminder_cust_id:
            posting_1500["customer"] = {"id": reminder_cust_id}
        v_body = {
            "date": reminder_date,
            "description": f"Purregebyr {cust_name}",
            "postings": [
                posting_1500,
                {"account": {"id": acct_3400}, "amountGross": -reminder_fee, "amountGrossCurrency": -reminder_fee, "date": reminder_date, "row": 2, "vatType": {"id": 0}},
            ],
        }
        _api_post(session, base_url, "/ledger/voucher", v_body, trace, params={"sendToLedger": "true"})

    # Step 3: Create reminder invoice
    cust_id = None
    if target:
        cust_id = target.get("customer", {}).get("id")
    if not cust_id:
        cust_id = _create_customer_if_needed(data, session, base_url, ctx, trace)

    if cust_id:
        prod_id = _create_product("Purregebyr", "PURRE-1", reminder_fee, 0, session, base_url, trace, ctx=ctx)
        if prod_id:
            _create_invoice_flow(
                cust_id,
                [{"product_id": prod_id, "price_ex_vat": reminder_fee, "vat_pct": 0, "quantity": 1}],
                session, base_url, ctx, trace,
                invoice_date=reminder_date,
                due_date=_due(14),
            )

    # Step 4: Partial payment on overdue invoice if specified
    partial = _parse_amount(data.get("partialPaymentAmount", 0))
    if partial > 0 and target:
        pt_id = _get_payment_type_id(ctx)
        params = {
            "paymentDate": reminder_date,
            "paymentTypeId": str(pt_id) if pt_id else "1",
            "paidAmount": str(partial),
        }
        _api_put(session, base_url, f"/invoice/{target['id']}/:payment", trace, params=params)

    return trace


# ═══════════════════════════════════════════════════════════
# HANDLER: MONTH END
# ═══════════════════════════════════════════════════════════

def _sanitize_account_number(val, default="7700"):
    """Ensure account number is numeric. If LLM returned a text name, map to default."""
    if val is None:
        return default
    s = str(val).strip()
    # Already a valid numeric account number
    if s.isdigit() and len(s) >= 3:
        return s
    # Common text-to-account mappings (LLM sometimes returns names instead of numbers)
    text_map = {
        "charges": "7700", "expenses": "7700", "dépenses": "7700", "depenses": "7700",
        "gastos": "7700", "spese": "7700", "kosten": "7700", "utgifter": "7700",
        "kostnader": "7700", "driftskostnader": "7700", "operating expenses": "7700",
        "charges salariales": "5000", "salary expenses": "5000", "lønnskostnader": "5000",
        "salaires à payer": "2900", "accrued salaries": "2900", "påløpt lønn": "2900",
        "amortissement": "6010", "depreciation": "6010", "avskrivning": "6010",
        "accumulated depreciation": "1029", "amortissement cumulé": "1029",
    }
    lower = s.lower()
    if lower in text_map:
        return text_map[lower]
    return default


def handle_month_end(data, session, base_url, ctx):
    trace = []
    voucher_date = _val(data, "voucherDate", _today())

    def get_acct(num, default="7700"):
        num = _sanitize_account_number(num, default)
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

    # Step 1: Monthly depreciation per asset
    for asset in data.get("assets", []):
        annual_dep = _parse_amount(asset.get("annualDepreciation", 0))
        if annual_dep <= 0:
            # Calculate from originalCost / usefulLifeYears
            cost = _parse_amount(asset.get("originalCost", 0))
            years = asset.get("usefulLifeYears", 0)
            if cost > 0 and years > 0:
                annual_dep = cost / years
        monthly_dep = round(annual_dep / 12, 2)
        if monthly_dep <= 0:
            continue

        dep_acct = get_acct(asset.get("depreciationAccountNumber", "6010"), "6010")
        accum_acct = get_acct(asset.get("accumulatedDepAccountNumber", "1209"), "1209")

        if dep_acct and accum_acct:
            v_body = {
                "date": voucher_date,
                "description": f"Mnd avskrivning {asset.get('name', '')}".strip(),
                "postings": [
                    {"account": {"id": dep_acct}, "amountGross": monthly_dep, "amountGrossCurrency": monthly_dep, "date": voucher_date, "row": 1, "vatType": {"id": 0}},
                    {"account": {"id": accum_acct}, "amountGross": -monthly_dep, "amountGrossCurrency": -monthly_dep, "date": voucher_date, "row": 2, "vatType": {"id": 0}},
                ],
            }
            _api_post(session, base_url, "/ledger/voucher", v_body, trace, params={"sendToLedger": "true"})

    # Step 2: Prepaid expense accruals
    for prepaid in data.get("prepaidExpenses", []):
        monthly_amt = _parse_amount(prepaid.get("monthlyAmount", 0))
        if monthly_amt <= 0:
            # Fallback: totalAmount / months
            total = _parse_amount(prepaid.get("totalAmount", 0))
            months = prepaid.get("months", 12)
            if months <= 0:
                months = 12
            monthly_amt = round(total / months, 2)
        if monthly_amt <= 0:
            continue

        exp_acct = get_acct(prepaid.get("expenseAccount", "7700"), "7700")
        prepaid_acct = get_acct(prepaid.get("prepaidAccount", "1710"), "1710")

        if exp_acct and prepaid_acct:
            v_body = {
                "date": voucher_date,
                "description": f"Periodisering {prepaid.get('description', '')}".strip(),
                "postings": [
                    {"account": {"id": exp_acct}, "amountGross": monthly_amt, "amountGrossCurrency": monthly_amt, "date": voucher_date, "row": 1, "vatType": {"id": 0}},
                    {"account": {"id": prepaid_acct}, "amountGross": -monthly_amt, "amountGrossCurrency": -monthly_amt, "date": voucher_date, "row": 2, "vatType": {"id": 0}},
                ],
            }
            _api_post(session, base_url, "/ledger/voucher", v_body, trace, params={"sendToLedger": "true"})

    # Step 3: Salary accruals
    for sal in data.get("salaryAccruals", []):
        amount = _parse_amount(sal.get("amount", 0))
        if amount <= 0:
            continue

        exp_acct = get_acct(sal.get("expenseAccount", "5000"), "5000")
        liab_acct = get_acct(sal.get("liabilityAccount", "2900"), "2900")

        if exp_acct and liab_acct:
            v_body = {
                "date": voucher_date,
                "description": f"Lønnsavsetning {sal.get('description', '')}".strip(),
                "postings": [
                    {"account": {"id": exp_acct}, "amountGross": amount, "amountGrossCurrency": amount, "date": voucher_date, "row": 1, "vatType": {"id": 0}},
                    {"account": {"id": liab_acct}, "amountGross": -amount, "amountGrossCurrency": -amount, "date": voucher_date, "row": 2, "vatType": {"id": 0}},
                ],
            }
            _api_post(session, base_url, "/ledger/voucher", v_body, trace, params={"sendToLedger": "true"})

    # Step 4: Other accruals/provisions
    for other in data.get("otherAccruals", []):
        amount = _parse_amount(other.get("amount", 0))
        if amount <= 0:
            continue

        debit_acct = get_acct(other.get("debitAccount"), "7700")
        credit_acct = get_acct(other.get("creditAccount"), "2900")

        if debit_acct and credit_acct:
            v_body = {
                "date": voucher_date,
                "description": f"Avsetning {other.get('description', '')}".strip(),
                "postings": [
                    {"account": {"id": debit_acct}, "amountGross": amount, "amountGrossCurrency": amount, "date": voucher_date, "row": 1, "vatType": {"id": 0}},
                    {"account": {"id": credit_acct}, "amountGross": -amount, "amountGrossCurrency": -amount, "date": voucher_date, "row": 2, "vatType": {"id": 0}},
                ],
            }
            _api_post(session, base_url, "/ledger/voucher", v_body, trace, params={"sendToLedger": "true"})

    return trace


# ═══════════════════════════════════════════════════════════
# HANDLER: EMPLOYEE FROM PDF
# ═══════════════════════════════════════════════════════════

def handle_employee_pdf(data, session, base_url, ctx):
    trace = []
    today = _today()

    email = data.get("email", "")
    # Validate and fallback: generate email if missing or invalid format
    import re as _re
    if not email or not _re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', email.strip()):
        if email:
            log.warning(f"Invalid email from PDF extraction: '{email}', generating fallback")
        first = data.get("firstName", "ukjent").strip().lower().replace(" ", "")
        last = data.get("lastName", "ukjent").strip().lower().replace(" ", "")
        # Normalize accented chars for email
        for old, new in [("é","e"),("è","e"),("ê","e"),("ë","e"),("à","a"),("â","a"),("ô","o"),("ù","u"),("û","u"),("ü","u"),("ç","c"),("ñ","n"),("ø","o"),("å","a"),("æ","ae")]:
            first = first.replace(old, new)
            last = last.replace(old, new)
        email = f"{first}.{last}@example.org"
    existing = _find_employee_by_email(ctx, email)

    # Resolve department from PDF (departmentName) or fallback to default
    dept_id = None
    dept_name = data.get("departmentName", "")
    if dept_name:
        for d in ctx.get("departments", []):
            if d.get("name", "").lower() == dept_name.lower():
                dept_id = d["id"]
                break
        if not dept_id:
            try:
                r_dept, _ = _api_get(session, base_url, "/department",
                                     {"name": dept_name, "fields": "id,name"}, trace)
                vals = r_dept.get("values", [])
                if vals:
                    dept_id = vals[0]["id"]
                else:
                    dept_result, dept_status = _api_post(session, base_url, "/department",
                                                         {"name": dept_name, "departmentNumber": "1"}, trace)
                    if dept_status == 201:
                        dept_id = dept_result.get("value", {}).get("id")
            except Exception as e:
                log.warning(f"Department lookup/create failed: {e}")
    if not dept_id:
        dept_id = ctx.get("default_department_id")

    # Step 1: Create or update employee
    if existing:
        emp_id = existing["id"]
        user_type = _val(data, "userType", "STANDARD")
        if user_type.upper() == "ADMINISTRATOR":
            user_type = "EXTENDED"
        put_body = {
            "id": emp_id,
            "firstName": data.get("firstName", existing.get("firstName", "")),
            "lastName": data.get("lastName", existing.get("lastName", "")),
            "email": email,
            "dateOfBirth": _val(data, "dateOfBirth", existing.get("dateOfBirth", "1990-05-15")),
            "userType": user_type,
            "version": existing.get("version"),
        }
        if dept_id:
            put_body["department"] = {"id": dept_id}
        nid = _val(data, "nationalIdentityNumber")
        if nid:
            put_body["nationalIdentityNumber"] = nid
        emp_num = _val(data, "employeeNumber")
        if emp_num:
            put_body["employeeNumber"] = str(emp_num)
        _api_put(session, base_url, f"/employee/{emp_id}", trace, body=put_body)
    else:
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
        nid = _val(data, "nationalIdentityNumber")
        if nid:
            post_body["nationalIdentityNumber"] = nid
        emp_num = _val(data, "employeeNumber")
        if emp_num:
            post_body["employeeNumber"] = str(emp_num)

        result, status = _api_post(session, base_url, "/employee", post_body, trace)
        if status != 201:
            return trace
        emp_id = result.get("value", {}).get("id")
        if not emp_id:
            return trace

    # Step 2: Ensure division + employment + details
    div_id = _ensure_division(session, base_url, ctx, trace)
    start_date = _val(data, "startDate", today)
    pct = data.get("percentageOfFullTime", 100.0)

    if div_id:
        has_employment = existing and bool(existing.get("_employments"))
        if not has_employment:
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
                    salary = _parse_amount(data.get("salary", 0))
                    det_body = {
                        "employment": {"id": empl_id},
                        "date": start_date,
                        "employmentType": "ORDINARY",
                        "employmentForm": "PERMANENT",
                        "remunerationType": "MONTHLY_WAGE",
                        "workingHoursScheme": "NOT_SHIFT",
                        "percentageOfFullTimeEquivalent": float(pct),
                    }
                    if salary > 0:
                        det_body["annualSalary"] = round(salary, 2)
                    _api_post(session, base_url, "/employee/employment/details", det_body, trace)

                    # Step 3: Occupation code
                    occ_code = _val(data, "occupationCode")
                    if occ_code and empl_id:
                        r, s = _api_get(session, base_url, "/employee/employment/occupationCode",
                                        {"code": str(occ_code), "fields": "id,code"}, trace)
                        occ_codes = r.get("values", [])
                        if occ_codes:
                            occ_id = occ_codes[0]["id"]
                            r2, _ = _api_get(session, base_url, f"/employee/employment/{empl_id}",
                                             {"fields": "id,version"}, trace)
                            empl_version = r2.get("value", {}).get("version")
                            if empl_version:
                                empl_put = {
                                    "id": empl_id,
                                    "employee": {"id": emp_id},
                                    "startDate": start_date,
                                    "division": {"id": div_id},
                                    "isMainEmployer": True,
                                    "taxDeductionCode": "loennFraHovedarbeidsgiver",
                                    "occupationCode": {"id": occ_id},
                                    "version": empl_version,
                                }
                                _api_put(session, base_url, f"/employee/employment/{empl_id}", trace, body=empl_put)

    # Step 4: Standard time
    std_body = {
        "employee": {"id": emp_id},
        "fromDate": start_date,
        "hoursPerDay": round(7.5 * float(pct) / 100, 2),
    }
    _api_post(session, base_url, "/employee/standardTime", std_body, trace)

    return trace


# ═══════════════════════════════════════════════════════════
# HANDLER: SUPPLIER INVOICE FROM PDF
# ═══════════════════════════════════════════════════════════

def handle_supplier_invoice_pdf(data, session, base_url, ctx):
    """Same as handle_supplier_invoice but data comes from PDF extraction."""
    return handle_supplier_invoice(data, session, base_url, ctx)


# ═══════════════════════════════════════════════════════════
# HANDLER REGISTRY + SOLVE FUNCTION
# ═══════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════
# HANDLER: LEDGER AUDIT
# ═══════════════════════════════════════════════════════════

def handle_ledger_audit(data, session, base_url, ctx):
    """Fix ledger errors described in the prompt. Each error gets a corrective voucher."""
    trace = []
    today = _today()
    errors = data.get("errors", [])

    for i, error in enumerate(errors):
        error_type = error.get("type", "").lower()
        desc = error.get("description", f"Error {i+1}")
        amount = _parse_amount(error.get("amount", 0))
        wrong_account = str(error.get("wrongAccount", ""))
        correct_account = str(error.get("correctAccount", ""))
        voucher_date = _val(error, "date", today)

        if not amount:
            continue

        postings = []

        if any(x in error_type for x in ("wrong_account", "feil konto", "incorrect account", "wrong account", "cuenta incorrecta", "mauvais compte", "falsches konto", "conta errada")):
            # Wrong account: debit correct, credit wrong (reverse the error)
            correct_id = _get_acct_id(ctx, correct_account)
            wrong_id = _get_acct_id(ctx, wrong_account)
            if not correct_id:
                r, _ = _api_get(session, base_url, "/ledger/account", {"number": correct_account, "fields": "id"}, trace)
                vals = r.get("values", [])
                if vals:
                    correct_id = vals[0]["id"]
            if not wrong_id:
                r, _ = _api_get(session, base_url, "/ledger/account", {"number": wrong_account, "fields": "id"}, trace)
                vals = r.get("values", [])
                if vals:
                    wrong_id = vals[0]["id"]
            if correct_id and wrong_id:
                postings = [
                    {"account": {"id": correct_id}, "amountGross": amount, "amountGrossCurrency": amount, "date": voucher_date, "row": 1, "vatType": {"id": 0}},
                    {"account": {"id": wrong_id}, "amountGross": -amount, "amountGrossCurrency": -amount, "date": voucher_date, "row": 2, "vatType": {"id": 0}},
                ]

        elif any(x in error_type for x in ("duplicate", "duplikat", "duplicado", "dupliziert", "doublé", "dobbel")):
            # Duplicate: reverse it
            acct_id = _get_acct_id(ctx, wrong_account) or _get_acct_id(ctx, correct_account)
            if not acct_id:
                acct_num = wrong_account or correct_account
                if acct_num:
                    r, _ = _api_get(session, base_url, "/ledger/account", {"number": acct_num, "fields": "id"}, trace)
                    vals = r.get("values", [])
                    if vals:
                        acct_id = vals[0]["id"]
            bank_id = _get_acct_id(ctx, "1920")
            if acct_id and bank_id:
                postings = [
                    {"account": {"id": acct_id}, "amountGross": -amount, "amountGrossCurrency": -amount, "date": voucher_date, "row": 1, "vatType": {"id": 0}},
                    {"account": {"id": bank_id}, "amountGross": amount, "amountGrossCurrency": amount, "date": voucher_date, "row": 2, "vatType": {"id": 0}},
                ]

        elif any(x in error_type for x in ("vat", "mva", "missing_vat", "missing vat", "iva", "mwst", "tva")):
            # Missing VAT: The expense (e.g. 6300) was posted correctly for the excl. amount,
            # but the VAT posting to 2710 was never created.
            # Correction: debit 2710 (inngående MVA) for VAT amount, credit 1920 for same.
            vat_rate = float(error.get("vatRate", 25)) / 100.0
            vat_amount = round(amount * vat_rate, 2)
            vat_account = "2710"
            vat_acct_id = _get_acct_id(ctx, vat_account)
            if not vat_acct_id:
                r, _ = _api_get(session, base_url, "/ledger/account", {"number": vat_account, "fields": "id"}, trace)
                vals = r.get("values", [])
                if vals:
                    vat_acct_id = vals[0]["id"]
            bank_id = _get_acct_id(ctx, "1920")
            if vat_acct_id and bank_id:
                postings = [
                    {"account": {"id": vat_acct_id}, "amountGross": vat_amount, "amountGrossCurrency": vat_amount, "date": voucher_date, "row": 1, "vatType": {"id": 0}},
                    {"account": {"id": bank_id}, "amountGross": -vat_amount, "amountGrossCurrency": -vat_amount, "date": voucher_date, "row": 2, "vatType": {"id": 0}},
                ]

        elif any(x in error_type for x in ("wrong_amount", "feil belop", "feil beløp", "incorrect amount", "monto incorrecto", "montant incorrect", "falscher betrag", "valor errado")):
            # Wrong amount: reverse wrong, post correct
            correct_amount = _parse_amount(error.get("correctAmount", 0))
            acct_id = _get_acct_id(ctx, correct_account or wrong_account)
            if not acct_id:
                acct_num = correct_account or wrong_account
                if acct_num:
                    r, _ = _api_get(session, base_url, "/ledger/account", {"number": acct_num, "fields": "id"}, trace)
                    vals = r.get("values", [])
                    if vals:
                        acct_id = vals[0]["id"]
            bank_id = _get_acct_id(ctx, "1920")
            diff = correct_amount - amount
            if acct_id and bank_id and diff:
                postings = [
                    {"account": {"id": acct_id}, "amountGross": diff, "amountGrossCurrency": diff, "date": voucher_date, "row": 1, "vatType": {"id": 0}},
                    {"account": {"id": bank_id}, "amountGross": -diff, "amountGrossCurrency": -diff, "date": voucher_date, "row": 2, "vatType": {"id": 0}},
                ]

        if postings:
            voucher_body = {
                "date": voucher_date,
                "description": f"Korreksjon: {desc}",
                "postings": postings,
            }
            _api_post(session, base_url, "/ledger/voucher", voucher_body, trace, params={"sendToLedger": "true"})

    return trace


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
    "travel_expense": handle_travel_expense,
    "delete_travel": handle_delete_travel,
    "bank_recon": handle_bank_recon,
    "project_lifecycle": handle_project_lifecycle,
    "fx_invoice": handle_fx_invoice,
    "reminder_fee": handle_reminder_fee,
    "month_end": handle_month_end,
    "employee_pdf": handle_employee_pdf,
    "supplier_invoice_pdf": handle_supplier_invoice_pdf,
    "ledger_audit": handle_ledger_audit,
}


def solve_deterministic(task_type, prompt, files, session, base_url, ctx,
                         api_key, model, start_time, image_blocks=None):
    """Main entry point: extract data with Gemini, then run deterministic handler."""
    log.info(f"solve_deterministic: task_type={task_type}")

    # Extract structured data
    data = extract_data(api_key, model, prompt, files, task_type, image_blocks)
    if not data:
        log.error(f"Extraction returned empty data for {task_type}")
        raise RuntimeError(f"Extraction failed for {task_type} — falling back to LLM agent")
    log.info(f"Extracted data for {task_type}: {json.dumps(data, ensure_ascii=False)[:1000]}")

    # Run handler
    handler = DETERMINISTIC_HANDLERS.get(task_type)
    if not handler:
        log.error(f"No deterministic handler for task_type={task_type}")
        raise RuntimeError(f"No deterministic handler for {task_type} — falling back to LLM agent")

    trace = handler(data, session, base_url, ctx)
    log.info(f"Deterministic handler {task_type} completed: {len(trace)} API calls")
    return trace
