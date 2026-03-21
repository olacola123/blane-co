# Tripletex API v2 — Research & Tips for Competition Agent

## 1. Authentication

- **Two tokens needed**: consumer token + employee token -> session token
- **Create session**: `PUT /v2/token/session/:create` (no auth required)
  - Params: `consumerToken`, `employeeToken`, `expirationDate`
  - Expiration must be later than today, expires at midnight on that date
- **Use session token**: username = `0`, password = session token (Basic Auth)
- **Test env**: `api-test.tripletex.tech` — credentials do NOT work in production and vice versa
- **Postman quirk**: `:create` is interpreted as parameter placeholder. Workaround: add manual param `create` with value `:create`

## 2. General API Patterns

### Field Selection (saves tokens & time)
```
GET /v2/project?fields=name,projectManager(email)
GET /v2/order?fields=*,orderLines(*)
```
- Works recursively for sub-objects
- Combine multiple calls into one with nested field selection

### Version Control (Optimistic Locking)
- Every object has a `version` field
- PUT requests MUST include correct `version` — otherwise HTTP 409 (RevisionException, error code 8000)
- Always GET the object first, then PUT with same version

### Object References
- Always reference objects by their **internal Tripletex ID** (not name, not external ref)
- POST requests: NEVER set `id` or `version` (system assigns these)
- Reference existing objects: `{"id": 12345}`

### Error Handling
- HTTP 422: JSON parse error (not 500)
- HTTP 409: Version mismatch (RevisionException)
- HTTP 429: Rate limit — NOTE: `X-Rate-Limit-Reset` header may be MISSING (known bug)
- All responses include `x-tlx-request-id` header for debugging

### Checksum for Sync
- Use `If-None-Match` header with checksum value
- Returns HTTP 304 if data unchanged

## 3. Invoice & Order Creation

### Create Invoice with Embedded Order (POST /invoice)
```json
[
  {
    "invoiceDate": "2020-09-02",
    "invoiceDueDate": "2020-09-12",
    "orders": [
      {
        "deliveryDate": "2020-09-03",
        "orderDate": "2020-09-01",
        "customer": {"id": 3001},
        "orderLines": [
          {
            "count": 2,
            "description": "Example description"
          }
        ]
      }
    ]
  }
]
```

### VAT Price Gotcha
- Set `isPrioritizeAmountsIncludingVat` on the order
- If `true` -> use `unitPriceIncludingVatCurrency` on orderlines
- If `false` -> use exclusive VAT prices
- **Mismatch causes validation error**: "The unit price must be exclusive VAT since the unit price on the order is exclusive VAT"

### Invoice Features
- `sendToCustomer=FALSE` (default TRUE) to create without sending
- `POST /invoice/list` for bulk creation with embedded orders
- `amountOutstanding` field: 0 = fully paid/credited
- `creditedInvoice` and `isCredited` fields for credit status
- `invoiceComment`, `invoiceRemarks` fields available

### Multiple Order Invoicing
- `POST order/:invoiceMultipleOrders` with `sendToCustomer` field

## 4. Invoice Payment Registration

### Customer Invoice Payment
- `PUT /invoice/{id}/:payment`
- Params: payment date, payment type ID, paid amount, paid amount currency (for foreign currency)

### Supplier Invoice Payment
- `POST /supplierInvoice/{invoiceId}/:addPayment`
- **Critical**: `partialPayment` must be `true` for multiple payments
- Without `partialPayment=true`, second payment will fail

### Payment Tracking Workaround (no dedicated endpoint)
1. Get `Invoice.customerId` and `invoice.number`
2. GET all postings where `customerId` matches
3. Filter by `invoiceNumber`
4. Check `closegroupId` (indicates full payment/credit)
5. Sum postings for partial payment status

## 5. Credit Note Creation

- `PUT /invoice/{id}/:createCreditNote`
- Params:
  - `date` (required) — credit note date
  - `comment` (optional)
  - `sendType` (new, replaces deprecated `sendToCustomer`)
  - `credit_note_email` (optional — sends electronically if filled)

## 6. Voucher / Ledger Posting

### Create Voucher (POST /ledger/voucher)
```json
{
  "date": "2022-10-06",
  "description": "Example voucher",
  "postings": [
    {
      "row": 1,
      "date": "2022-10-06",
      "amountGross": 500,
      "amountGrossCurrency": 500,
      "account": {"id": 53120749},
      "vatType": {"id": 3}
    },
    {
      "row": 2,
      "date": "2022-10-06",
      "amountGross": -500,
      "amountGrossCurrency": -500,
      "account": {"id": 53120750},
      "vatType": {"id": 0}
    }
  ]
}
```

### Posting Rules
- **Only gross amounts** are used when creating vouchers
- Postings MUST balance (sum of all amountGross = 0)
- **Ledger accounts**: If account has `ledgerType` != GENERAL, posting MUST include matching object ID:
  - Customer ledger -> include `customer: {"id": X}`
  - Supplier ledger -> include `supplier: {"id": X}`
  - Employee ledger -> include `employee: {"id": X}`
- `externalRef` field for KID numbers / payment references
- Free dimensions: `freeAccountingDimension1/2/3` (writeable, Pro+ packages)

### VAT Type Lookup
- `GET /ledger/vatType` with `typeOfVat=incoming|outgoing` and `vatDate`
- Only returns active VAT codes for the account
- `GET /ledger/account` returns `legalVatTypes` — which VAT types are valid for each account

### Account Lookup
- `GET /ledger/account` — chart of accounts
- Fields: `requiresDepartment`, `requiresProject` — know if you need to add these
- `isApplicableForSupplierInvoice` parameter for filtering

### Voucher Operations
- Reverse: `PUT /ledger/voucher/{id}/:reverse`
- Historical vouchers: `POST /ledger/voucher/historical/*` (for closed years)
- Attachment: `POST /ledger/voucher/importDocument` (PDF, PNG, JPEG, TIFF, EHF)
- External reference lookup: `GET /ledger/voucher/externalVoucherNumber`
- Opening balance: `GET/POST/DELETE /ledger/voucher/openingBalance`

## 7. Customer & Supplier Management

### Create Customer (POST /customer)
- Required: name (at minimum)
- Do NOT set id or version
- Non-Norwegian customers: org number validation may fail via API even though UI works (known bug)

### Create Supplier (POST /supplier)
- Similar pattern to customer

### Key Fields
- `bankAccountPresentation` (replaces deprecated `bankAccounts`)
- `isInactive` status field
- `discountPercentage` on CustomerDTO
- `website` field available
- `language` setting for suppliers

## 8. Project Management

### Key Project Fields
- `name`, `projectManager`, `startDate`, `endDate`
- `displayName` field
- `isReadyForInvoicing` flag
- `invoiceComment` field
- `contact`, `attention`, `reference` fields (settable)

### Project Endpoints
- `POST/GET/PUT/DELETE /project/participant` — manage participants
- `GET /project/projectActivity` and `POST /activity`
- `GET/PUT /project/settings`
- Period reports: `/project/{id}/period/hourlistReport`, `/period/invoiced`, etc.

### Gotcha
- `GET /project` does NOT include closed projects by default

## 9. Employee & Timesheet

### Timesheet Entry
- `POST /timesheet/entry` — create hours
- Can add multiple entries for several users in same request
- **Warning**: Fields present but set to 0 will be NULLED
- `chargeableHours` read-only field

### Employee Data
- `/employee/employment` for employment details
- `taxDeductionCode`, `isMainEmployer` fields
- Entitlements: `GET /employee/entitlement` — check what permissions exist

## 10. Supplier Invoice Management

### Key Endpoints
- `PUT /supplierInvoice/{invoiceId}/:changeDimension` — change project, department, employee, product
- `PUT /supplierInvoice/voucher/{id}/postings` — update postings (with optional `voucherDate`)
- `PUT /supplierinvoice/:approve`, `/{id}/:reject`
- `PUT /supplierinvoice/{id}/:addRecipient`
- `invoiceNumber` field (renamed from `number`)
- `outstandingAmount` field

## 11. Bank Reconciliation

### Endpoints (BETA)
- `POST /bank/statement/import` — required: `bankId`, `accountId`, `fromDate`, `toDate`
- `bank/reconciliation/` — reconciliation with or without bank statement
- `bank/reconciliation/match` — map postings to transactions
- `GET /bank` — supported banks, register numbers, file formats

## 12. Payment Types

### Lookup Endpoints
- `GET /ledger/paymentTypeOut` — outgoing payment types (supplier invoices, VAT, salary, tax)
- Payment types are configured per Tripletex account

## 13. Known Bugs & Quirks

1. **Pagination**: `fullResultSize` behavior changed, may break pagination
2. **changedSince**: Does NOT work on many endpoints (known issue)
3. **Rate limit header missing**: `X-Rate-Limit-Reset` not in 429 responses
4. **Discount ignored**: `discountPercentage` in orders doesn't work when generating invoices
5. **Posting type ignored**: Type field on postings may be disregarded
6. **Invoice number collision**: After API-created invoice, UI may try to reuse same number
7. **Order auto-created**: Creating an invoice auto-creates an order
8. **Null inventory**: `GET /v2/order` returns null for inventory even when inventoryLocation is set
9. **Closed projects hidden**: `/project` endpoint excludes closed projects by default
10. **Non-Norwegian org numbers**: Validation fails via API but works in UI
11. **Webhook protocol mismatch**: Creating webhooks needs HTTP POST, deleting needs HTTPS DELETE

## 14. Currency

- Rates fetched at 06:10 and 18:10 UTC from Norges Bank
- `GET /currency/{id}/rate` for exchange rates
- `factor` field on currency
- `isDisabled` field for inactive currencies

## 15. Useful Performance Endpoints

- `GET /ledger/postingByDate` — much faster than `/ledger/posting` for date-range queries
- Use `fields` parameter everywhere to reduce payload
- `GET /voucherInbox/inboxCount` — quick inbox check

## 16. Document Handling

- `POST /ledger/voucher/importDocument` — PDF, PNG, JPEG, TIFF, EHF
- `POST /documentArchive/{objectType}/{id}` — attach documents to objects
- `mimetype` field in DocumentDTO

## 17. API Action Convention

- Actions use `:` prefix in URL path: `:create`, `:payment`, `:approve`, `:reject`, `:reverse`, `:createCreditNote`, `:addPayment`, `:deliver`
- These are PUT or POST operations on existing resources
