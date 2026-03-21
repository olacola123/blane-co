# Tripletex API v2 - Comprehensive Field Reference

Base URL: `https://tripletex.no/v2`
OpenAPI spec: `https://tripletex.no/v2/openapi.json`
Swagger UI: `https://tripletex.no/v2-docs/`
Total endpoints: 548

## Authentication

- **Basic Auth**: username=`0`, password=`<session_token>`
- Session token created via `PUT /token/session/:create` with consumerToken + employeeToken
- For NM i AI: auth goes through proxy, token provided in task

## General API Patterns

- Pagination: `from` (offset) + `count` (limit) on all GET list endpoints
- Field selection: `fields=*` for all fields, or `fields=id,name,number` for specific
- Sorting: `sorting=field,-field` (prefix `-` for descending)
- Date format: `yyyy-MM-dd`
- References to other objects: use `{"id": <int>}` object, e.g. `"customer": {"id": 123}`
- All amounts should be rounded to 2 decimals
- `?fields=*` is very useful to see all available data

---

## 1. EMPLOYEE

### POST /employee - Create employee

```json
{
  "firstName": "string",
  "lastName": "string",
  "email": "string",
  "employeeNumber": "string",
  "dateOfBirth": "yyyy-MM-dd",
  "nationalIdentityNumber": "string",
  "dnumber": "string",
  "bankAccountNumber": "string",
  "phoneNumberMobile": "string",
  "phoneNumberHome": "string",
  "phoneNumberWork": "string",
  "isContact": false,
  "userType": "STANDARD|EXTENDED|NO_ACCESS",
  "address": { "addressLine1": "", "city": "", "postalCode": "", "country": {"id": 0} },
  "department": {"id": 0},
  "employeeCategory": {"id": 0},
  "comments": "string",
  "employments": [
    {
      "startDate": "yyyy-MM-dd",
      "endDate": "yyyy-MM-dd",
      "isMainEmployer": true,
      "taxDeductionCode": "loennFraHovedarbeidsgiver|loennFraBiarbeidsgiver|pensjon",
      "employmentDetails": [
        {
          "date": "yyyy-MM-dd",
          "employmentType": "ORDINARY|MARITIME|FREELANCE|NOT_CHOSEN",
          "employmentForm": "PERMANENT|TEMPORARY|...",
          "remunerationType": "MONTHLY_WAGE|HOURLY_WAGE|COMMISION_PERCENTAGE|FEE|PIECEWORK_WAGE|NOT_CHOSEN",
          "annualSalary": 0.0,
          "hourlyWage": 0.0,
          "percentageOfFullTimeEquivalent": 100.0,
          "workingHoursScheme": "NOT_SHIFT|ROUND_THE_CLOCK|...",
          "occupationCode": {"id": 0},
          "payrollTaxMunicipalityId": {"id": 0}
        }
      ]
    }
  ]
}
```

**Key notes:**
- `firstName` + `lastName` are the practical minimum to create
- `userType`: STANDARD = normal login, EXTENDED = full access, NO_ACCESS = no system login
- `isContact: true` = external contact, not an employee
- Employments can be nested at creation
- `taxDeductionCode` defaults based on `isMainEmployer`
- `employeeNumber` auto-generated if not provided

### GET /employee - Search
- Filters: `id`, `firstName`, `lastName`, `employeeNumber`, `email`, `departmentId`, `hasSystemAccess`, `includeContacts`, `onlyProjectManagers`

### Entitlements (roles/permissions)
- `GET /employee/entitlement?employeeId=X` - list all entitlements
- `PUT /employee/entitlement/:grantEntitlementsByTemplate` - assign by template

---

## 2. CUSTOMER

### POST /customer - Create customer

```json
{
  "name": "string",
  "customerNumber": 0,
  "organizationNumber": "string",
  "email": "string",
  "invoiceEmail": "string",
  "phoneNumber": "string",
  "phoneNumberMobile": "string",
  "isPrivateIndividual": false,
  "isInactive": false,
  "invoiceSendMethod": "EMAIL|EHF|EFAKTURA|AVTALEGIRO|VIPPS|PAPER|MANUAL",
  "emailAttachmentType": "LINK|ATTACHMENT",
  "invoicesDueIn": 14,
  "invoicesDueInType": "DAYS|MONTHS|RECURRING_DAY_OF_MONTH",
  "language": "NO|EN",
  "currency": {"id": 1},
  "accountManager": {"id": 0},
  "department": {"id": 0},
  "category1": {"id": 0},
  "physicalAddress": {
    "addressLine1": "",
    "addressLine2": "",
    "postalCode": "",
    "city": "",
    "country": {"id": 0}
  },
  "postalAddress": { ... },
  "deliveryAddress": { "name": "", "addressLine1": "", ... },
  "isSupplier": false,
  "supplierNumber": 0,
  "singleCustomerInvoice": false,
  "bankAccounts": ["string"],
  "discountPercentage": 0.0
}
```

**Key notes:**
- `name` is the practical minimum
- `customerNumber` auto-generated if 0 or omitted
- `isPrivateIndividual` affects display/validation
- `organizationNumber` is the Norwegian org number (9 digits)
- Addresses are optional but important for invoicing

### GET /customer - Search
- Filters: `customerAccountNumber`, `organizationNumber`, `email`, `invoiceEmail`, `customerName`, `phoneNumberMobile`, `isInactive`, `accountManagerId`, `changedSince`

---

## 3. SUPPLIER

### POST /supplier - Create supplier

```json
{
  "name": "string",
  "supplierNumber": 0,
  "organizationNumber": "string",
  "email": "string",
  "invoiceEmail": "string",
  "phoneNumber": "string",
  "isPrivateIndividual": false,
  "isCustomer": false,
  "language": "NO|EN",
  "currency": {"id": 1},
  "bankAccounts": ["string"],
  "physicalAddress": { ... },
  "postalAddress": { ... }
}
```

---

## 4. INVOICE (Outgoing / Customer Invoice)

### Flow: Order -> Invoice -> Payment

**Step 1: Create Order + OrderLines**

### POST /order

```json
{
  "customer": {"id": 0},
  "orderDate": "yyyy-MM-dd",
  "deliveryDate": "yyyy-MM-dd",
  "receiverEmail": "string",
  "invoiceComment": "string",
  "deliveryComment": "string",
  "internalComment": "string",
  "reference": "string",
  "invoicesDueIn": 14,
  "invoicesDueInType": "DAYS|MONTHS|RECURRING_DAY_OF_MONTH",
  "currency": {"id": 1},
  "project": {"id": 0},
  "department": {"id": 0},
  "ourContactEmployee": {"id": 0},
  "discountPercentage": 0.0,
  "markUpOrderLines": 0.0,
  "orderLineSorting": "ID|PRODUCT|CUSTOM",
  "orderLines": [
    {
      "product": {"id": 0},
      "description": "string",
      "count": 1.0,
      "unitPriceExcludingVatCurrency": 100.0,
      "unitPriceIncludingVatCurrency": 125.0,
      "vatType": {"id": 0},
      "discount": 0.0,
      "unitCostCurrency": 0.0
    }
  ]
}
```

**Step 2: Create Invoice from Order**

### POST /invoice - Create invoice

```json
{
  "invoiceDate": "yyyy-MM-dd",
  "invoiceDueDate": "yyyy-MM-dd",
  "invoiceNumber": 0,
  "kid": "string",
  "comment": "string",
  "orders": [{"id": 0}],
  "currency": {"id": 1},
  "customer": {"id": 0},
  "paidAmount": 0.0,
  "paymentTypeId": 0
}
```

Query params:
- `sendToCustomer=true/false`
- `paymentTypeId=X` + `paidAmount=X` for prepayment

**Alternative: Invoice from Order action**

### PUT /order/{id}/:invoice
- `invoiceDate` (REQUIRED)
- `sendToCustomer` (optional)
- `sendType` (optional)
- `paymentTypeId` + `paidAmount` (optional, for prepayment)
- `createBackorder` (optional)
- `createOnAccount` + `amountOnAccount` (optional)

**Key notes:**
- Order and OrderLines can be created inline with the Invoice POST
- `invoiceNumber: 0` = auto-generate
- Only one order per invoice currently supported
- OrderLine: provide either `unitPriceExcludingVatCurrency` OR `unitPriceIncludingVatCurrency` (other calculated)
- Most amount fields on Invoice are READ-ONLY (calculated from order lines)

### Step 3: Register Payment

### PUT /invoice/{id}/:payment
- `paymentDate` (REQUIRED, yyyy-MM-dd)
- `paymentTypeId` (REQUIRED, int) - get from GET /invoice/paymentType
- `paidAmount` (REQUIRED, number) - in currency of paymentType account
- `paidAmountCurrency` (optional) - amount in invoice currency (required for foreign currency)

### Step 4: Send Invoice

### PUT /invoice/{id}/:send
- `sendType` (REQUIRED): EMAIL, EHF, EFAKTURA, etc.
- `overrideEmailAddress` (optional)

### Create Credit Note

### PUT /invoice/{id}/:createCreditNote
- `date` (REQUIRED, yyyy-MM-dd)
- `comment` (optional)
- `creditNoteEmail` (optional) - empty = won't send if email type
- `sendToCustomer` (optional, boolean)
- `sendType` (optional)

### GET /invoice - Search
- `invoiceDateFrom` (REQUIRED), `invoiceDateTo` (REQUIRED)
- Optional: `invoiceNumber`, `kid`, `voucherId`, `customerId`

### GET /invoice/paymentType - List payment types
- Useful to find payment type IDs for `:payment` action

---

## 5. VOUCHER / POSTING (General Ledger)

### POST /ledger/voucher - Create voucher with postings

```json
{
  "date": "yyyy-MM-dd",
  "description": "string",
  "voucherType": {"id": 0},
  "postings": [
    {
      "account": {"id": 0},
      "amount": 0.0,
      "amountGross": 0.0,
      "amountCurrency": 0.0,
      "amountGrossCurrency": 0.0,
      "currency": {"id": 1},
      "description": "string",
      "date": "yyyy-MM-dd",
      "customer": {"id": 0},
      "supplier": {"id": 0},
      "employee": {"id": 0},
      "project": {"id": 0},
      "department": {"id": 0},
      "product": {"id": 0},
      "vatType": {"id": 0},
      "row": 1
    }
  ]
}
```

Query params:
- `sendToLedger=true/false` - requires "Advanced Voucher" permission

**Key notes:**
- IMPORTANT: Only gross amounts are used by the system
- Amounts should be rounded to 2 decimals
- Postings must balance (sum of debit = sum of credit)
- Each posting needs an `account` reference
- `voucherType` can be found via GET /ledger/voucherType
- A posting can reference customer, supplier, employee, project, department
- `row` field controls display ordering

### Supporting endpoints:
- `GET /ledger/account` - find accounts by number, type, etc.
- `GET /ledger/voucherType` - find voucher types
- `PUT /ledger/voucher/{id}/:reverse` - reverse a voucher
- `PUT /ledger/voucher/{id}/:sendToLedger` - send to ledger
- `PUT /ledger/voucher/{id}/:sendToInbox` - send to inbox
- `POST /ledger/voucher/{voucherId}/attachment` - upload attachment

### Historical Vouchers (special)
- `POST /ledger/voucher/historical/historical` - create historical vouchers (outside normal flow)
- `POST /ledger/voucher/historical/employee` - create historical employee based on import

---

## 6. PROJECT

### POST /project - Create project

```json
{
  "name": "string",
  "number": "string",
  "description": "string",
  "projectManager": {"id": 0},
  "customer": {"id": 0},
  "department": {"id": 0},
  "startDate": "yyyy-MM-dd",
  "endDate": "yyyy-MM-dd",
  "projectCategory": {"id": 0},
  "isInternal": false,
  "isOffer": false,
  "isClosed": false,
  "isFixedPrice": false,
  "fixedprice": 0.0,
  "isPriceCeiling": false,
  "priceCeilingAmount": 0.0,
  "currency": {"id": 1},
  "vatType": {"id": 0},
  "reference": "string",
  "invoiceComment": "string",
  "invoiceDueDate": 14,
  "invoiceDueDateType": "DAYS|MONTHS|RECURRING_DAY_OF_MONTH",
  "invoiceReceiverEmail": "string",
  "accessType": "NONE|READ|WRITE",
  "displayNameFormat": "NAME_STANDARD|NAME_INCL_CUSTOMER_NAME|...",
  "forParticipantsOnly": false,
  "mainProject": {"id": 0},
  "participants": [
    {
      "employee": {"id": 0},
      "adminAccess": false
    }
  ],
  "projectActivities": [
    {
      "activity": {"id": 0},
      "startDate": "yyyy-MM-dd",
      "endDate": "yyyy-MM-dd",
      "isClosed": false,
      "budgetHours": 0.0,
      "budgetFeeCurrency": 0.0,
      "budgetHourlyRateCurrency": 0.0
    }
  ],
  "projectHourlyRates": [ ... ]
}
```

**Key notes:**
- `name` is the practical minimum
- `number` auto-generated if null
- `mainProject` for creating sub-projects
- `isFixedPrice: true` = fixed price, `false` = hourly rate
- `isOffer: true` = project offer, `false` = actual project
- Participants and activities can be nested at creation
- `projectManager` should be an employee with project manager rights

### GET /project - Search
- Filters: `name`, `number`, `isOffer`, `projectManagerId`, `departmentId`, `customerId`, `isClosed`, `isFixedPrice`, `startDateFrom/To`, `endDateFrom/To`

---

## 7. BANK RECONCILIATION

### Flow: Import Statement -> Create Reconciliation -> Match -> Close

### POST /bank/statement/import - Upload bank statement
- `bankId` (REQUIRED)
- `accountId` (REQUIRED)
- `fromDate` (REQUIRED, yyyy-MM-dd)
- `toDate` (REQUIRED, yyyy-MM-dd)
- `fileFormat` (REQUIRED)
- File uploaded as multipart

### POST /bank/reconciliation - Create reconciliation

```json
{
  "account": {"id": 0},
  "accountingPeriod": {"id": 0},
  "type": "MANUAL|AUTOMATIC",
  "bankAccountClosingBalanceCurrency": 0.0,
  "isClosed": false
}
```

### POST /bank/reconciliation/match - Create match

```json
{
  "bankReconciliation": {"id": 0},
  "type": "MANUAL|PENDING_SUGGESTION|APPROVED_SUGGESTION|ADJUSTMENT|AUTO_MATCHED",
  "postings": [{"id": 0}],
  "transactions": [{"id": 0}]
}
```

### PUT /bank/reconciliation/match/:suggest - Auto-suggest matches
- Suggests matches for a reconciliation

### PUT /bank/reconciliation/{id}/:adjustment - Add adjustment

### GET /bank/reconciliation/>last - Get last reconciliation
- `accountId` (REQUIRED)

### GET /bank/reconciliation/>lastClosed - Get last closed
- `accountId` (REQUIRED)

---

## 8. SUPPLIER INVOICE (Incoming)

### GET /supplierInvoice - Search
### GET /supplierInvoice/{id} - Get by ID

### POST /supplierInvoice/{invoiceId}/:addPayment - Register payment
- `paymentType` (REQUIRED) - 0 = last used for vendor
- `amount` (REQUIRED)
- `kid` (optional)
- `paymentDate` (REQUIRED)
- `bban` + `paymentTypeName` (optional)

### PUT /supplierInvoice/{invoiceId}/:approve - Approve
### PUT /supplierInvoice/{invoiceId}/:reject - Reject
### PUT /supplierInvoice/{invoiceId}/:changeDimension - Change dimension
### PUT /supplierInvoice/voucher/{id}/postings - Update debit postings

---

## 9. PRODUCT

### POST /product - Create product

```json
{
  "name": "string",
  "number": "string",
  "description": "string",
  "priceExcludingVatCurrency": 0.0,
  "priceIncludingVatCurrency": 0.0,
  "costExcludingVatCurrency": 0.0,
  "vatType": {"id": 0},
  "account": {"id": 0},
  "currency": {"id": 1},
  "department": {"id": 0},
  "supplier": {"id": 0},
  "productUnit": {"id": 0},
  "isInactive": false,
  "isStockItem": false,
  "ean": "string",
  "weight": 0.0,
  "weightUnit": "kg|g|hg",
  "volume": 0.0,
  "volumeUnit": "cm3|dm3|m3"
}
```

---

## 10. DEPARTMENT

### POST /department

```json
{
  "name": "string",
  "departmentNumber": "string",
  "departmentManager": {"id": 0},
  "isInactive": false
}
```

---

## 11. ACTIVITY (for timesheets/projects)

### GET /activity - List activities
### GET /activity/>forTimeSheet - Activities for timesheet

---

## KEY SUPPORTING SCHEMAS

### Address
```json
{
  "addressLine1": "string",
  "addressLine2": "string",
  "postalCode": "string",
  "city": "string",
  "country": {"id": 0}
}
```

### VatType (reference by id)
- `id`, `name`, `number`, `percentage`, `deductionPercentage`
- Get all: `GET /ledger/vatType`

### Account (reference by id)
- `id`, `number`, `name`, `description`
- Key fields: `isBankAccount`, `ledgerType` (GENERAL/CUSTOMER/VENDOR/EMPLOYEE/ASSET)
- Get all: `GET /ledger/account`

### Currency (reference by id)
- NOK is usually id=1
- Get all: `GET /currency`

### Country (reference by id)
- Norway is usually id=161 (check via GET /country)

---

## COMMON PATTERNS & TIPS

1. **Object references**: Always use `{"id": X}` format:
   ```json
   "customer": {"id": 123}
   "account": {"id": 456}
   ```

2. **Sandbox starts empty**: You may need to create prerequisites (customer, product) before creating invoices

3. **Enable modules first**: Some tasks require enabling modules (e.g., department accounting)

4. **Field selection**: Use `?fields=*` to see ALL fields on any entity

5. **Dates**: Always `yyyy-MM-dd` format

6. **Amounts**: Round to 2 decimals. Only gross amounts used for voucher postings.

7. **Invoice flow**: Customer -> Order (with OrderLines) -> Invoice -> Payment
   - Or: Create invoice directly with inline orders/orderlines

8. **Voucher postings must balance**: Sum of all posting amounts must equal zero (debits = credits)

9. **Credit notes**: Created via action endpoint on existing invoice, not as standalone

10. **Auto-generated fields**: invoiceNumber, employeeNumber, project number - set to 0 or null for auto-generation

---

## ENDPOINT COUNTS BY AREA

| Area | Count |
|------|-------|
| employee | 42 |
| project | 47 |
| ledger | 55 |
| salary | 38 |
| timesheet | 37 |
| travelExpense | 43 |
| product | 30 |
| purchaseOrder | 27 |
| bank | 25 |
| order | 21 |
| supplierInvoice | 13 |
| invoice | 12 |
| asset | 11 |
| inventory | 11 |
| documentArchive | 9 |
| yearEnd | 18 |
| resultbudget | 6 |
| company | 6 |
| token | 5 |
| customer | 5 |
| currency | 5 |
| event | 5 |
