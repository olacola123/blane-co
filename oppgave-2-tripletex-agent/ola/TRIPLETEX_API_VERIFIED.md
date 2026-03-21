# Tripletex API — Verified Field Schemas
# Source: https://kkpqfuj-amager.tripletex.dev/v2/openapi.json
# Extracted: 2026-03-21
# IMPORTANT: All paths are under /v2/ base path

---
## POST /customer

**Summary**: Create customer. Related customer addresses may also be created.

### Request Body: `Customer`

**Required fields**: none explicitly marked (check validation)

  - `accountManager`: object -> Employee (use {id: <int>}) (optional)
  - `bankAccountPresentation`: array of CompanyBankAccountPresentation (optional)
  - `bankAccounts`: array of string (optional)
  - `category1`: object -> CustomerCategory (use {id: <int>}) (optional)
  - `category2`: object -> CustomerCategory (use {id: <int>}) (optional)
  - `category3`: object -> CustomerCategory (use {id: <int>}) (optional)
  - `changes`: array of Change (optional) [READ-ONLY]
  - `currency`: object -> Currency (use {id: <int>}) (optional)
  - `customerNumber`: integer (int32) (optional) [min=0]
  - `deliveryAddress`: object -> DeliveryAddress (use {id: <int>}) (optional)
  - `department`: object -> Department (use {id: <int>}) (optional)
  - `description`: string (optional)
  - `discountPercentage`: number (optional) — Default discount percentage for this customer.
  - `displayName`: string (optional)
  - `email`: string (optional)
  - `emailAttachmentType`: string (optional) [enum=['LINK', 'ATTACHMENT']] — Define the invoice attachment type for emailing to the customer.<br>LINK: Send invoice as link in email.<br>ATTACHMENT: Send invoice as attachment in email.<br>
  - `globalLocationNumber`: integer (int64) (optional) [min=0]
  - `id`: integer (int64) (optional)
  - `invoiceEmail`: string (optional)
  - `invoiceSMSNotificationNumber`: string (optional) — Send SMS-notification to this number. Must be a norwegian phone number
  - `invoiceSendMethod`: string (optional) [enum=['EMAIL', 'EHF', 'EFAKTURA', 'AVTALEGIRO', 'VIPPS', 'PAPER', 'MANUAL']] — Define the invoicing method for the customer.<br>EMAIL: Send invoices as email.<br>EHF: Send invoices as EHF.<br>EFAKTURA: Send invoices as EFAKTURA.<br>AVTALEGIRO: Send invoices as AVTALEGIRO.<br>VIP
  - `invoiceSendSMSNotification`: boolean (optional) — Is sms-notification on/off
  - `invoicesDueIn`: integer (int32) (optional) [min=0, max=10000] — Number of days/months in which invoices created from this customer is due
  - `invoicesDueInType`: string (optional) [enum=['DAYS', 'MONTHS', 'RECURRING_DAY_OF_MONTH']] — Set the time unit of invoicesDueIn. The special case RECURRING_DAY_OF_MONTH enables the due date to be fixed to a specific day of the month, in this case the fixed due date will automatically be set a
  - `isAutomaticNoticeOfDebtCollectionEnabled`: boolean (optional) — Has automatic notice of debt collection enabled for this customer.
  - `isAutomaticReminderEnabled`: boolean (optional) — Has automatic reminders enabled for this customer.
  - `isAutomaticSoftReminderEnabled`: boolean (optional) — Has automatic soft reminders enabled for this customer.
  - `isCustomer`: boolean (optional) [READ-ONLY]
  - `isFactoring`: boolean (optional) — If true; send this customers invoices to factoring (if factoring is turned on in account).
  - `isInactive`: boolean (optional)
  - `isPrivateIndividual`: boolean (optional)
  - `isSupplier`: boolean (optional) — Defines if the customer is also a supplier.
  - `language`: string (optional) [enum=['NO', 'EN']]
  - `ledgerAccount`: object -> Account (use {id: <int>}) (optional)
  - `name`: string (optional)
  - `organizationNumber`: string (optional)
  - `overdueNoticeEmail`: string (optional) — The email address of the customer where the noticing emails are sent in case of an overdue
  - `phoneNumber`: string (optional)
  - `phoneNumberMobile`: string (optional)
  - `physicalAddress`: object -> Address (use {id: <int>}) (optional)
  - `postalAddress`: object -> Address (use {id: <int>}) (optional)
  - `singleCustomerInvoice`: boolean (optional) — Enables various orders on one customer invoice.
  - `supplierNumber`: integer (int32) (optional) [min=0]
  - `url`: string (optional) [READ-ONLY]
  - `version`: integer (int32) (optional)
  - `website`: string (optional)

---
## POST /supplier

**Summary**: Create supplier. Related supplier addresses may also be created.

### Request Body: `Supplier`

**Required fields**: none explicitly marked (check validation)

  - `accountManager`: object -> Employee (use {id: <int>}) (optional)
  - `bankAccountPresentation`: array of CompanyBankAccountPresentation (optional) — List of bankAccount for this supplier
  - `bankAccounts`: array of string (optional) — [DEPRECATED] List of the bank account numbers for this supplier. Norwegian bank account numbers only.
  - `category1`: object -> CustomerCategory (use {id: <int>}) (optional)
  - `category2`: object -> CustomerCategory (use {id: <int>}) (optional)
  - `category3`: object -> CustomerCategory (use {id: <int>}) (optional)
  - `changes`: array of Change (optional) [READ-ONLY]
  - `currency`: object -> Currency (use {id: <int>}) (optional)
  - `customerNumber`: integer (int32) (optional)
  - `deliveryAddress`: object -> DeliveryAddress (use {id: <int>}) (optional)
  - `description`: string (optional)
  - `displayName`: string (optional) [READ-ONLY]
  - `email`: string (optional)
  - `id`: integer (int64) (optional)
  - `invoiceEmail`: string (optional)
  - `isCustomer`: boolean (optional) — Determine if the supplier is also a customer
  - `isInactive`: boolean (optional)
  - `isPrivateIndividual`: boolean (optional)
  - `isSupplier`: boolean (optional) [READ-ONLY]
  - `isWholesaler`: boolean (optional) [READ-ONLY]
  - `language`: string (optional) [enum=['NO', 'EN']]
  - `ledgerAccount`: object -> Account (use {id: <int>}) (optional)
  - `locale`: string (optional) [READ-ONLY]
  - `name`: string (optional)
  - `organizationNumber`: string (optional)
  - `overdueNoticeEmail`: string (optional) — The email address of the customer where the noticing emails are sent in case of an overdue
  - `phoneNumber`: string (optional)
  - `phoneNumberMobile`: string (optional)
  - `physicalAddress`: object -> Address (use {id: <int>}) (optional)
  - `postalAddress`: object -> Address (use {id: <int>}) (optional)
  - `showProducts`: boolean (optional)
  - `supplierNumber`: integer (int32) (optional)
  - `url`: string (optional) [READ-ONLY]
  - `version`: integer (int32) (optional)
  - `website`: string (optional)

---
## POST /employee

**Summary**: Create one employee.

### Request Body: `Employee`

**Required fields**: none explicitly marked (check validation)

  - `address`: object -> Address (use {id: <int>}) (optional)
  - `allowInformationRegistration`: boolean (optional) [READ-ONLY] — Determines if salary information can be registered on the user including hours, travel expenses and employee expenses. The user may also be selected as a project member on projects.
  - `bankAccountNumber`: string (optional)
  - `bic`: string (optional) — Bic (swift) field
  - `changes`: array of Change (optional) [READ-ONLY]
  - `comments`: string (optional)
  - `companyId`: integer (int32) (optional) [READ-ONLY]
  - `creditorBankCountryId`: integer (int32) (optional) — Country of creditor bank field
  - `dateOfBirth`: string (optional)
  - `department`: object -> Department (use {id: <int>}) (optional)
  - `displayName`: string (optional) [READ-ONLY]
  - `dnumber`: string (optional)
  - `email`: string (optional)
  - `employeeCategory`: object -> EmployeeCategory (use {id: <int>}) (optional)
  - `employeeNumber`: string (optional)
  - `employments`: array of Employment (optional)
  - `firstName`: string (optional)
  - `holidayAllowanceEarned`: object -> HolidayAllowanceEarned (use {id: <int>}) (optional)
  - `iban`: string (optional) — IBAN field
  - `id`: integer (int64) (optional)
  - `internationalId`: object -> InternationalId (use {id: <int>}) (optional)
  - `isAuthProjectOverviewURL`: boolean (optional) [READ-ONLY]
  - `isContact`: boolean (optional) — Determines if the employee is a contact (external) in the company.
  - `isProxy`: boolean (optional) [READ-ONLY] — True if this Employee object represents an accounting or auditor office
  - `lastName`: string (optional)
  - `nationalIdentityNumber`: string (optional)
  - `phoneNumberHome`: string (optional)
  - `phoneNumberMobile`: string (optional)
  - `phoneNumberMobileCountry`: object -> Country (use {id: <int>}) (optional)
  - `phoneNumberWork`: string (optional)
  - `pictureId`: integer (int32) (optional) [READ-ONLY]
  - `url`: string (optional) [READ-ONLY]
  - `userType`: string (optional) [enum=['STANDARD', 'EXTENDED', 'NO_ACCESS']] — Define the employee's user type.<br>STANDARD: Reduced access. Users with limited system entitlements.<br>EXTENDED: Users can be given all system entitlements.<br>NO_ACCESS: User with no log on access.
  - `usesAbroadPayment`: boolean (optional) — UsesAbroadPayment field. Determines if we should use domestic or abroad remittance. To be able to use abroad remittance, one has to: 1: have Autopay 2: have valid combination of the fields Iban, Bic (
  - `version`: integer (int32) (optional)
  - `vismaConnect2FAactive`: boolean (optional) [READ-ONLY]

---
## POST /employee/employment

**Summary**: Create employment.

### Request Body: `Employment`

**Required fields**: none explicitly marked (check validation)

  - `changes`: array of Change (optional) [READ-ONLY]
  - `division`: object -> Division (use {id: <int>}) (optional)
  - `employee`: object -> Employee (use {id: <int>}) (optional)
  - `employmentDetails`: array of EmploymentDetails (optional)
  - `employmentEndReason`: string (optional) [enum=['EMPLOYMENT_END_EXPIRED', 'EMPLOYMENT_END_EMPLOYEE', 'EMPLOYMENT_END_EMPLOYER', 'EMPLOYMENT_END_WRONGLY_REPORTED', 'EMPLOYMENT_END_SYSTEM_OR_ACCOUNTANT_CHANGE', 'EMPLOYMENT_END_INTERNAL_CHANGE']] — Define the employment end reason.
  - `employmentId`: string (optional) — Existing employment ID used by the current accounting system
  - `endDate`: string (optional)
  - `id`: integer (int64) (optional)
  - `isMainEmployer`: boolean (optional) — Determines if company is main employer for the employee. Default value is true.<br />Some values will be default set if not sent upon creation of employment: <br/> If isMainEmployer is NOT sent and ta
  - `isRemoveAccessAtEmploymentEnded`: boolean (optional) — If true, access to the employee will be removed when the employment ends. <br />This field is part of the Employee object, therefore changing it for one Employment affects all Employments.
  - `lastSalaryChangeDate`: string (optional)
  - `latestSalary`: object -> EmploymentDetails (use {id: <int>}) (optional)
  - `noEmploymentRelationship`: boolean (optional) — Activate pensions and other benefits with no employment relationship.
  - `startDate`: string (optional)
  - `taxDeductionCode`: string (optional) [enum=['loennFraHovedarbeidsgiver', 'loennFraBiarbeidsgiver', 'pensjon', 'loennTilUtenrikstjenestemann', 'loennKunTrygdeavgiftTilUtenlandskBorger', 'loennKunTrygdeavgiftTilUtenlandskBorgerSomGrensegjenger', 'introduksjonsstoenad', 'ufoereytelserFraAndre', '']] — EMPTY - represents that a tax deduction code is not set on the employment. It is illegal to set the field to this value.  <br /> Default value of this field is loennFraHovedarbeidsgiver or loennFraBia
  - `url`: string (optional) [READ-ONLY]
  - `version`: integer (int32) (optional)

---
## POST /employee/employment/details

**Summary**: Create employment details.

### Request Body: `EmploymentDetails`

**Required fields**: none explicitly marked (check validation)

  - `annualSalary`: number (optional)
  - `changes`: array of Change (optional) [READ-ONLY]
  - `date`: string (optional)
  - `employment`: object -> Employment (use {id: <int>}) (optional)
  - `employmentForm`: string (optional) [enum=['PERMANENT', 'TEMPORARY', 'PERMANENT_AND_HIRED_OUT', 'TEMPORARY_AND_HIRED_OUT', 'TEMPORARY_ON_CALL', 'NOT_CHOSEN']] — Define the employment form.
  - `employmentType`: string (optional) [enum=['ORDINARY', 'MARITIME', 'FREELANCE', 'NOT_CHOSEN']] — Define the employment type.
  - `hourlyWage`: number (optional)
  - `id`: integer (int64) (optional)
  - `maritimeEmployment`: object -> MaritimeEmployment (use {id: <int>}) (optional)
  - `monthlySalary`: number (optional) [READ-ONLY]
  - `occupationCode`: object -> OccupationCode (use {id: <int>}) (optional)
  - `payrollTaxMunicipalityId`: object -> Municipality (use {id: <int>}) (optional)
  - `percentageOfFullTimeEquivalent`: number (optional)
  - `remunerationType`: string (optional) [enum=['MONTHLY_WAGE', 'HOURLY_WAGE', 'COMMISION_PERCENTAGE', 'FEE', 'NOT_CHOSEN', 'PIECEWORK_WAGE']] — Define the remuneration type.
  - `shiftDurationHours`: number (optional)
  - `url`: string (optional) [READ-ONLY]
  - `version`: integer (int32) (optional)
  - `workingHoursScheme`: string (optional) [enum=['NOT_SHIFT', 'ROUND_THE_CLOCK', 'SHIFT_365', 'OFFSHORE_336', 'CONTINUOUS', 'OTHER_SHIFT', 'NOT_CHOSEN']] — Define the working hours scheme type. If you enter a value for SHIFT WORK, you must also enter value for shiftDurationHours

---
## POST /employee/entitlement

**ENDPOINT NOT FOUND** — path `/employee/entitlement` with method `post` does not exist in spec

---
## PUT /employee/entitlement/:grantEntitlementsByTemplate

**Summary**: [BETA] Update employee entitlements.

### Query/Path Parameters
  - `employeeId` (query): integer (int64) (REQUIRED) — Employee ID
  - `template` (query): string (REQUIRED) — Template

*No request body — uses query parameters only*

---
## PUT /employee/entitlement/:grantClientEntitlementsByTemplate

**Summary**: [BETA] Update employee entitlements in client account.

### Query/Path Parameters
  - `employeeId` (query): integer (int64) (REQUIRED) — Employee ID
  - `customerId` (query): integer (int64) (REQUIRED) — Client ID
  - `template` (query): string (REQUIRED) — Template
  - `addToExisting` (query): boolean (optional) [default=False] — Add template to existing entitlements

*No request body — uses query parameters only*

---
## POST /employee/standardTime

**Summary**: Create standard time.

### Request Body: `StandardTime`

**Required fields**: none explicitly marked (check validation)

  - `changes`: array of Change (optional) [READ-ONLY]
  - `employee`: object -> Employee (use {id: <int>}) (optional)
  - `fromDate`: string (optional)
  - `hoursPerDay`: number (optional)
  - `id`: integer (int64) (optional)
  - `url`: string (optional) [READ-ONLY]
  - `version`: integer (int32) (optional)

---
## POST /product

**Summary**: Create new product.

### Request Body: `Product`

**Required fields**: none explicitly marked (check validation)

  - `account`: object -> Account (use {id: <int>}) (optional)
  - `availableStock`: number (optional) [READ-ONLY] — Available only on demand
  - `changes`: array of Change (optional) [READ-ONLY]
  - `costExcludingVatCurrency`: number (optional) — Price purchase (cost) excluding VAT in the product's currency
  - `costPrice`: number (optional) [READ-ONLY] — Cost price of purchase
  - `currency`: object -> Currency (use {id: <int>}) (optional)
  - `department`: object -> Department (use {id: <int>}) (optional)
  - `description`: string (optional)
  - `discountGroup`: object -> DiscountGroup (use {id: <int>}) (optional)
  - `discountPrice`: number (optional) [READ-ONLY]
  - `displayName`: string (optional) [READ-ONLY]
  - `displayNumber`: string (optional) [READ-ONLY]
  - `ean`: string (optional)
  - `elNumber`: string (optional) [READ-ONLY]
  - `expenses`: number (optional)
  - `expensesInPercent`: number (optional) [READ-ONLY]
  - `hasSupplierProductConnected`: boolean (optional)
  - `hsnCode`: string (optional)
  - `id`: integer (int64) (optional)
  - `image`: object -> Document (use {id: <int>}) (optional)
  - `incomingStock`: number (optional) [READ-ONLY] — Available only on demand
  - `isDeletable`: boolean (optional) — For performance reasons, field is deprecated and it will always return false.
  - `isInactive`: boolean (optional)
  - `isRoundPriceIncVat`: boolean (optional) [READ-ONLY] — [BETA] Indicates whether the price incl. VAT is rounded off or not
  - `isStockItem`: boolean (optional)
  - `mainSupplierProduct`: object -> SupplierProduct (use {id: <int>}) (optional)
  - `markupListPercentage`: number (optional) [READ-ONLY]
  - `markupNetPercentage`: number (optional) [READ-ONLY]
  - `minStockLevel`: number (optional) — Minimum available stock level for the product. Applicable only to stock items in the Logistics Basics module.
  - `name`: string (optional)
  - `nrfNumber`: string (optional) [READ-ONLY]
  - `number`: string (optional)
  - `orderLineDescription`: string (optional)
  - `outgoingStock`: number (optional) [READ-ONLY] — Available only on demand
  - `priceExcludingVatCurrency`: number (optional) — Price of purchase excluding VAT in the product's currency
  - `priceInTargetCurrency`: number (optional) [READ-ONLY] — Purchase Price converted in specific currency.
  - `priceIncludingVatCurrency`: number (optional) — Price of purchase including VAT in the product's currency
  - `productUnit`: object -> ProductUnit (use {id: <int>}) (optional)
  - `profit`: number (optional) [READ-ONLY]
  - `profitInPercent`: number (optional) [READ-ONLY]
  - `purchasePriceCurrency`: number (optional) [READ-ONLY] — Purchase Price in product currency. This affects only Supplier Products.
  - `resaleProduct`: object -> Product (use {id: <int>}) (optional)
  - `stockOfGoods`: number (optional) [READ-ONLY] — Available only on demand
  - `supplier`: object -> Supplier (use {id: <int>}) (optional)
  - `url`: string (optional) [READ-ONLY]
  - `vatType`: object -> VatType (use {id: <int>}) (optional)
  - `version`: integer (int32) (optional)
  - `volume`: number (optional)
  - `volumeUnit`: string (optional) [enum=['cm3', 'dm3', 'm3']]
  - `weight`: number (optional)
  - `weightUnit`: string (optional) [enum=['kg', 'g', 'hg']]

---
## POST /order

**Summary**: Create order.

### Request Body: `Order`

**Required fields**: none explicitly marked (check validation)

  - `accountingDimensionValues`: array of AccountingDimensionValue (optional) [READ-ONLY] — Free dimensions for the project connected to the order.
  - `attachment`: array of Document (optional) [READ-ONLY] — [BETA] Attachments belonging to this order
  - `attn`: object -> Contact (use {id: <int>}) (optional)
  - `canCreateBackorder`: boolean (optional) [READ-ONLY]
  - `changes`: array of Change (optional) [READ-ONLY]
  - `contact`: object -> Contact (use {id: <int>}) (optional)
  - `currency`: object -> Currency (use {id: <int>}) (optional)
  - `customer`: object -> Customer (use {id: <int>}) (optional)
  - `customerName`: string (optional) [READ-ONLY]
  - `deliveryAddress`: object -> DeliveryAddress (use {id: <int>}) (optional)
  - `deliveryComment`: string (optional)
  - `deliveryDate`: string (optional)
  - `department`: object -> Department (use {id: <int>}) (optional)
  - `discountPercentage`: number (optional) — Default discount percentage for order lines.
  - `displayName`: string (optional) [READ-ONLY]
  - `id`: integer (int64) (optional)
  - `invoiceComment`: string (optional) — Comment to be displayed in the invoice based on this order. Can be also found in Invoice.invoiceComment on Invoice objects.
  - `invoiceOnAccountVatHigh`: boolean (optional) — Is the on account(a konto) amounts including vat 
  - `invoiceSMSNotificationNumber`: string (optional) — The phone number of the receiver of sms notifications. Must be a norwegian phone number
  - `invoiceSendSMSNotification`: boolean (optional) [READ-ONLY] — Is sms-notification on/off
  - `invoicesDueIn`: integer (int32) (optional) [min=0, max=10000] — Number of days/months in which invoices created from this order is due
  - `invoicesDueInType`: string (optional) [enum=['DAYS', 'MONTHS', 'RECURRING_DAY_OF_MONTH']] — Set the time unit of invoicesDueIn. The special case RECURRING_DAY_OF_MONTH enables the due date to be fixed to a specific day of the month, in this case the fixed due date will automatically be set a
  - `isClosed`: boolean (optional) — Denotes if this order is closed. A closed order can no longer be invoiced unless it is opened again.
  - `isPrioritizeAmountsIncludingVat`: boolean (optional)
  - `isShowOpenPostsOnInvoices`: boolean (optional) — Show account statement - open posts on invoices created from this order
  - `isSubscription`: boolean (optional) — If true, the order is a subscription, which enables periodical invoicing of order lines. First, create an order with isSubscription=true, then approve it for subscription invoicing with the :approveSu
  - `isSubscriptionAutoInvoicing`: boolean (optional) — Automatic invoicing. Starts when the subscription is approved
  - `markUpOrderLines`: number (optional) — Set mark-up (%) for order lines.
  - `number`: string (optional)
  - `orderDate`: string (optional)
  - `orderGroups`: array of OrderGroup (optional) — Order line groups
  - `orderLineSorting`: string (optional) [enum=['ID', 'PRODUCT', 'PRODUCT_DESCENDING', 'CUSTOM']]
  - `orderLines`: array of OrderLine (optional) — Order lines tied to the order. New OrderLines may be embedded here, in some endpoints.
  - `ourContact`: object -> Contact (use {id: <int>}) (optional)
  - `ourContactEmployee`: object -> Employee (use {id: <int>}) (optional)
  - `overdueNoticeEmail`: string (optional)
  - `preliminaryInvoice`: object -> Invoice (use {id: <int>}) (optional)
  - `project`: object -> Project (use {id: <int>}) (optional)
  - `projectManagerNameAndNumber`: string (optional) [READ-ONLY]
  - `receiverEmail`: string (optional)
  - `reference`: string (optional)
  - `sendMethodDescription`: string (optional) — Description of how this invoice will be sent
  - `status`: string (optional) [enum=['NOT_CHOSEN', 'NEW', 'CONFIRMATION_SENT', 'READY_FOR_PICKING', 'PICKED', 'PACKED', 'READY_FOR_SHIPPING', 'READY_FOR_INVOICING', 'INVOICED', 'CANCELLED']] — Logistics only
  - `subscriptionDuration`: integer (int32) (optional) [min=0] — Number of months/years the subscription shall run
  - `subscriptionDurationType`: string (optional) [enum=['MONTHS', 'YEAR']] — The time unit of subscriptionDuration
  - `subscriptionInvoicingTime`: integer (int32) (optional) [min=0] — Number of days/months invoicing in advance/in arrears
  - `subscriptionInvoicingTimeInAdvanceOrArrears`: string (optional) [enum=['ADVANCE', 'ARREARS']] — Invoicing in advance/in arrears
  - `subscriptionInvoicingTimeType`: string (optional) [enum=['DAYS', 'MONTHS']] — The time unit of subscriptionInvoicingTime
  - `subscriptionPeriodsOnInvoice`: integer (int32) (optional) [min=0] — Number of periods on each invoice
  - `subscriptionPeriodsOnInvoiceType`: string (optional) [READ-ONLY] [enum=['MONTHS']] — The time unit of subscriptionPeriodsOnInvoice
  - `totalInvoicedOnAccountAmountAbsoluteCurrency`: number (optional) [READ-ONLY] — Amount paid on account(a konto)
  - `travelReports`: array of TravelExpense (optional) [READ-ONLY] — Travel reports connected to the order.
  - `url`: string (optional) [READ-ONLY]
  - `version`: integer (int32) (optional)

---
## POST /order/orderline

**Summary**: Create order line. When creating several order lines, use /list for better performance.

### Request Body: `OrderLine`

**Required fields**: none explicitly marked (check validation)

  - `amountExcludingVatCurrency`: number (optional) [READ-ONLY] — Total amount on order line excluding VAT in the order's currency
  - `amountIncludingVatCurrency`: number (optional) [READ-ONLY] — Total amount on order line including VAT in the order's currency
  - `changes`: array of Change (optional) [READ-ONLY]
  - `count`: number (optional)
  - `currency`: object -> Currency (use {id: <int>}) (optional)
  - `description`: string (optional)
  - `discount`: number (optional) — Discount given as a percentage (%)
  - `displayName`: string (optional) [READ-ONLY] — Display name of order line
  - `id`: integer (int64) (optional)
  - `inventory`: object -> Inventory (use {id: <int>}) (optional)
  - `inventoryLocation`: object -> InventoryLocation (use {id: <int>}) (optional)
  - `isCharged`: boolean (optional) — Flag indicating whether the order line is charged or not.
  - `isPicked`: boolean (optional) — Only used for Logistics customers who activated the available inventory functionality. Represents whether the line has been picked up or not.
  - `isSubscription`: boolean (optional)
  - `markup`: number (optional) — Markup given as a percentage (%)
  - `order`: object -> Order (use {id: <int>}) (optional)
  - `orderGroup`: object -> OrderGroup (use {id: <int>}) (optional)
  - `orderedQuantity`: number (optional) — Only used for Logistics customers who activated the Backorder functionality. Represents the quantity that was ordered. If nothing is specified, the ordered quantity will be the same as the delivered q
  - `pickedDate`: string (optional) — Only used for Logistics customers who activated the available inventory functionality. Represents the pick date for an order line or null if the line was not picked.
  - `product`: object -> Product (use {id: <int>}) (optional)
  - `sortIndex`: integer (int32) (optional) [min=0] — Defines the presentation order of the lines. Does not need to be, and is often not continuous. Only applicable if parent order has orderLineSorting as CUSTOM.
  - `subscriptionPeriodEnd`: string (optional)
  - `subscriptionPeriodStart`: string (optional)
  - `unitCostCurrency`: number (optional) — Unit price purchase (cost) excluding VAT in the order's currency
  - `unitPriceExcludingVatCurrency`: number (optional) — Unit price of purchase excluding VAT in the order's currency. If only unit price Excl. VAT or unit price Inc. VAT is supplied, we will calculate and update the missing field.
  - `unitPriceIncludingVatCurrency`: number (optional) — Unit price of purchase including VAT in the order's currency. If only unit price Excl. VAT or unit price Inc. VAT is supplied, we will calculate and update the missing field.
  - `url`: string (optional) [READ-ONLY]
  - `vatType`: object -> VatType (use {id: <int>}) (optional)
  - `vendor`: object -> Company (use {id: <int>}) (optional)
  - `version`: integer (int32) (optional)

---
## POST /invoice

**Summary**: Create invoice. Related Order and OrderLines can be created first, or included as new objects inside the Invoice.

### Query/Path Parameters
  - `sendToCustomer` (query): boolean (optional) [default=True] — Equals
  - `paymentTypeId` (query): integer (int32) (optional) — Payment type to register prepayment of the invoice. paymentTypeId and paidAmount are optional, but both must be provided if the invoice has already been paid.
  - `paidAmount` (query): number (optional) — Paid amount to register prepayment of the invoice, in invoice currency. paymentTypeId and paidAmount are optional, but both must be provided if the invoice has already been paid.

### Request Body: `Invoice`

**Required fields**: none explicitly marked (check validation)

  - `amount`: number (optional) [READ-ONLY] — In the company’s currency, typically NOK.
  - `amountCurrency`: number (optional) [READ-ONLY] — In the specified currency.
  - `amountCurrencyOutstanding`: number (optional) [READ-ONLY] — The amountCurrency outstanding based on the history collection, excluding reminders and any existing remits, in the invoice currency.
  - `amountCurrencyOutstandingTotal`: number (optional) [READ-ONLY] — The amountCurrency outstanding based on the history collection and including the last reminder and any existing remits. This is the total invoice balance including reminders and remittances, in the in
  - `amountExcludingVat`: number (optional) [READ-ONLY] — Amount excluding VAT (NOK).
  - `amountExcludingVatCurrency`: number (optional) [READ-ONLY] — Amount excluding VAT in the specified currency.
  - `amountOutstanding`: number (optional) [READ-ONLY] — The amount outstanding based on the history collection, excluding reminders and any existing remits, in the invoice currency.
  - `amountOutstandingTotal`: number (optional) [READ-ONLY] — The amount outstanding based on the history collection and including the last reminder and any existing remits. This is the total invoice balance including reminders and remittances, in the invoice cu
  - `amountRoundoff`: number (optional) [READ-ONLY] — Amount of round off to nearest integer.
  - `amountRoundoffCurrency`: number (optional) [READ-ONLY] — Amount of round off to nearest integer in the specified currency.
  - `changes`: array of Change (optional) [READ-ONLY]
  - `comment`: string (optional) — Comment text for the specific invoice.
  - `creditedInvoice`: integer (int64) (optional) [READ-ONLY] — The id of the original invoice if this is a credit note.
  - `currency`: object -> Currency (use {id: <int>}) (optional)
  - `customer`: object -> Customer (use {id: <int>}) (optional)
  - `deliveryDate`: string (optional) [READ-ONLY] — The delivery date.
  - `documentId`: integer (int32) (optional) [READ-ONLY]
  - `ehfSendStatus`: string (optional) [enum=['DO_NOT_SEND', 'SEND', 'SENT', 'SEND_FAILURE_RECIPIENT_NOT_FOUND']] — [Deprecated] EHF (Peppol) send status. This only shows status for historic EHFs.
  - `id`: integer (int64) (optional)
  - `invoiceComment`: string (optional) [READ-ONLY] — Comment text for the invoice. This was specified on the order as invoiceComment.
  - `invoiceDate`: string (optional)
  - `invoiceDueDate`: string (optional)
  - `invoiceNumber`: integer (int32) (optional) [min=0] — If value is set to 0, the invoice number will be generated.
  - `invoiceRemark`: object -> InvoiceRemark (use {id: <int>}) (optional)
  - `invoiceRemarks`: string (optional) — Deprecated Invoice remarks - please use the 'invoiceRemark' instead.
  - `isApproved`: boolean (optional) [READ-ONLY]
  - `isCharged`: boolean (optional) [READ-ONLY]
  - `isCreditNote`: boolean (optional) [READ-ONLY]
  - `isCredited`: boolean (optional) [READ-ONLY]
  - `isPeriodizationPossible`: boolean (optional) [READ-ONLY]
  - `kid`: string (optional) — KID - Kundeidentifikasjonsnummer.
  - `orderLines`: array of OrderLine (optional) [READ-ONLY] — Orderlines connected to the invoice.
  - `orders`: array of Order (optional) — Related orders. Only one order per invoice is supported at the moment.
  - `paidAmount`: number (optional) — [BETA] Optional. Used to specify the prepaid amount of the invoice. The paid amount can be specified here, or as a parameter to the /invoice API endpoint.
  - `paymentTypeId`: integer (int32) (optional) [min=0] — [BETA] Optional. Used to specify payment type for prepaid invoices. Payment type can be specified here, or as a parameter to the /invoice API endpoint.
  - `postings`: array of Posting (optional) [READ-ONLY] — The invoice postings, which includes a posting for the invoice with a positive amount, and one or more posting for the payments with negative amounts.
  - `projectInvoiceDetails`: array of ProjectInvoiceDetails (optional) [READ-ONLY] — ProjectInvoiceDetails contains additional information about the invoice, in particular invoices for projects. It contains information about the charged project, the fee amount, extra percent and amoun
  - `reminders`: array of Reminder (optional) [READ-ONLY] — Invoice debt collection and reminders.
  - `sumRemits`: number (optional) [READ-ONLY] — The sum of all open remittances of the invoice. Remittances are reimbursement payments back to the customer and are therefore relevant to the bookkeeping of the invoice in the accounts.
  - `travelReports`: array of TravelExpense (optional) [READ-ONLY] — Travel reports connected to the invoice.
  - `url`: string (optional) [READ-ONLY]
  - `version`: integer (int32) (optional)
  - `voucher`: object -> Voucher (use {id: <int>}) (optional)

---
## POST /department

**Summary**: Add new department.

### Request Body: `Department`

**Required fields**: none explicitly marked (check validation)

  - `businessActivityTypeId`: integer (int32) (optional) [READ-ONLY] — The business activity type for this department. Business activity types can be used to separate between different tax categories, and between general and primary VAT reports.  A posting done with a gi
  - `changes`: array of Change (optional) [READ-ONLY]
  - `departmentManager`: object -> Employee (use {id: <int>}) (optional)
  - `departmentNumber`: string (optional)
  - `displayName`: string (optional) [READ-ONLY]
  - `id`: integer (int64) (optional)
  - `isInactive`: boolean (optional)
  - `name`: string (optional)
  - `url`: string (optional) [READ-ONLY]
  - `version`: integer (int32) (optional)

---
## POST /division

**Summary**: Create division.

### Request Body: `Division`

**Required fields**: none explicitly marked (check validation)

  - `changes`: array of Change (optional) [READ-ONLY]
  - `displayName`: string (optional)
  - `endDate`: string (optional)
  - `id`: integer (int64) (optional)
  - `municipality`: object -> Municipality (use {id: <int>}) (optional)
  - `municipalityDate`: string (optional)
  - `name`: string (optional)
  - `organizationNumber`: string (optional)
  - `startDate`: string (optional)
  - `url`: string (optional) [READ-ONLY]
  - `version`: integer (int32) (optional)

---
## POST /project

**Summary**: Add new project.

### Request Body: `Project`

**Required fields**: none explicitly marked (check validation)

  - `accessType`: string (optional) [enum=['NONE', 'READ', 'WRITE']] — READ/WRITE access on project
  - `accountingDimensionValues`: array of AccountingDimensionValue (optional) — [BETA - Requires pilot feature] Free dimensions for the project.
  - `attention`: object -> Contact (use {id: <int>}) (optional)
  - `boligmappaAddress`: object -> Address (use {id: <int>}) (optional)
  - `changes`: array of Change (optional) [READ-ONLY]
  - `contact`: object -> Contact (use {id: <int>}) (optional)
  - `contributionMarginPercent`: number (optional) [READ-ONLY]
  - `currency`: object -> Currency (use {id: <int>}) (optional)
  - `customer`: object -> Customer (use {id: <int>}) (optional)
  - `customerName`: string (optional) [READ-ONLY]
  - `deliveryAddress`: object -> Address (use {id: <int>}) (optional)
  - `department`: object -> Department (use {id: <int>}) (optional)
  - `description`: string (optional)
  - `discountPercentage`: number (optional) [READ-ONLY] — Project discount percentage.
  - `displayName`: string (optional) [READ-ONLY]
  - `displayNameFormat`: string (optional) [enum=['NAME_STANDARD', 'NAME_INCL_CUSTOMER_NAME', 'NAME_INCL_PARENT_NAME', 'NAME_INCL_PARENT_NUMBER', 'NAME_INCL_PARENT_NAME_AND_NUMBER']] — Defines project name presentation in overviews.
  - `endDate`: string (optional)
  - `externalAccountsNumber`: string (optional)
  - `fixedprice`: number (optional) — Fixed price amount, in the project's currency.
  - `forParticipantsOnly`: boolean (optional) — Set to true if only project participants can register information on the project
  - `generalProjectActivitiesPerProjectOnly`: boolean (optional) — Set to true if a general project activity must be linked to project to allow time tracking.
  - `hierarchyLevel`: integer (int32) (optional) [READ-ONLY]
  - `hierarchyNameAndNumber`: string (optional) [READ-ONLY]
  - `id`: integer (int64) (optional)
  - `ignoreCompanyProductDiscountAgreement`: boolean (optional)
  - `invoiceComment`: string (optional) — Comment for project invoices
  - `invoiceDueDate`: integer (int32) (optional) — invoice due date
  - `invoiceDueDateType`: string (optional) [enum=['DAYS', 'MONTHS', 'RECURRING_DAY_OF_MONTH']] — Set the time unit of invoiceDueDate. The special case RECURRING_DAY_OF_MONTH enables the due date to be fixed to a specific day of the month, in this case the fixed due date will automatically be set 
  - `invoiceOnAccountVatHigh`: boolean (optional) — The on account(a konto) amounts including VAT
  - `invoiceReceiverEmail`: string (optional) — Set the project's invoice receiver email. Will override the default invoice receiver email of any customer that may also be set in the request body.
  - `invoiceReserveTotalAmountCurrency`: number (optional) [READ-ONLY] — Total invoice reserve
  - `invoicingPlan`: array of Invoice (optional) [READ-ONLY] — Invoicing plans tied to the project
  - `isClosed`: boolean (optional)
  - `isFixedPrice`: boolean (optional) — Project is fixed price if set to true, hourly rate if set to false.
  - `isInternal`: boolean (optional)
  - `isOffer`: boolean (optional) — If is Project Offer set to true, if is Project set to false. The default value is false.
  - `isPriceCeiling`: boolean (optional) — Set to true if an hourly rate project has a price ceiling.
  - `isReadyForInvoicing`: boolean (optional)
  - `mainProject`: object -> Project (use {id: <int>}) (optional)
  - `markUpFeesEarned`: number (optional) — Set mark-up (%) for fees earned.
  - `markUpOrderLines`: number (optional) — Set mark-up (%) for order lines.
  - `name`: string (optional)
  - `number`: string (optional) — If NULL, a number is generated automatically.
  - `numberOfProjectParticipants`: integer (int32) (optional) [READ-ONLY]
  - `numberOfSubProjects`: integer (int32) (optional) [READ-ONLY]
  - `orderLines`: array of ProjectOrderLine (optional) [READ-ONLY] — Order lines tied to the order
  - `overdueNoticeEmail`: string (optional) — Set the project's overdue notice email. Will override the default overdue notice email of any customer that may also be set in the request body.
  - `participants`: array of ProjectParticipant (optional) — Link to individual project participants.
  - `preliminaryInvoice`: object -> Invoice (use {id: <int>}) (optional)
  - `priceCeilingAmount`: number (optional) — Price ceiling amount, in the project's currency.
  - `projectActivities`: array of ProjectActivity (optional) — Project Activities
  - `projectCategory`: object -> ProjectCategory (use {id: <int>}) (optional)
  - `projectHourlyRates`: array of ProjectHourlyRate (optional) — Project Rate Types tied to the project.
  - `projectManager`: object -> Employee (use {id: <int>}) (optional)
  - `projectManagerNameAndNumber`: string (optional) [READ-ONLY]
  - `reference`: string (optional)
  - `startDate`: string (optional)
  - `totalInvoicedOnAccountAmountAbsoluteCurrency`: number (optional) [READ-ONLY] — Amount paid on account(a konto)
  - `url`: string (optional) [READ-ONLY]
  - `useProductNetPrice`: boolean (optional)
  - `vatType`: object -> VatType (use {id: <int>}) (optional)
  - `version`: integer (int32) (optional)

---
## POST /activity

**Summary**: Add activity.

### Request Body: `Activity`

**Required fields**: none explicitly marked (check validation)

  - `activityType`: string (optional) [enum=['GENERAL_ACTIVITY', 'PROJECT_GENERAL_ACTIVITY', 'PROJECT_SPECIFIC_ACTIVITY', 'TASK']] — PROJECT_SPECIFIC_ACTIVITY are made via project/projectActivity, as they must be part of a project.
  - `changes`: array of Change (optional) [READ-ONLY]
  - `costPercentage`: number (optional)
  - `deletable`: boolean (optional) [READ-ONLY]
  - `description`: string (optional)
  - `displayName`: string (optional)
  - `id`: integer (int64) (optional)
  - `isChargeable`: boolean (optional)
  - `isDisabled`: boolean (optional) [READ-ONLY]
  - `isGeneral`: boolean (optional) [READ-ONLY] — Manipulate these with ActivityType
  - `isProjectActivity`: boolean (optional) [READ-ONLY] — Manipulate these with ActivityType
  - `isTask`: boolean (optional) [READ-ONLY] — Manipulate these with ActivityType
  - `name`: string (optional)
  - `number`: string (optional)
  - `rate`: number (optional)
  - `url`: string (optional) [READ-ONLY]
  - `version`: integer (int32) (optional)

---
## POST /contact

**Summary**: Create contact.

### Request Body: `Contact`

**Required fields**: none explicitly marked (check validation)

  - `changes`: array of Change (optional) [READ-ONLY]
  - `customer`: object -> Customer (use {id: <int>}) (optional)
  - `department`: object -> Department (use {id: <int>}) (optional)
  - `displayName`: string (optional)
  - `email`: string (optional)
  - `firstName`: string (optional)
  - `id`: integer (int64) (optional)
  - `isInactive`: boolean (optional)
  - `lastName`: string (optional)
  - `phoneNumberMobile`: string (optional)
  - `phoneNumberMobileCountry`: object -> Country (use {id: <int>}) (optional)
  - `phoneNumberWork`: string (optional)
  - `url`: string (optional) [READ-ONLY]
  - `version`: integer (int32) (optional)

---
## POST /ledger/voucher

**Summary**: Add new voucher. IMPORTANT: Also creates postings. Only the gross amounts will be used. Amounts should be rounded to 2 decimals.

### Query/Path Parameters
  - `sendToLedger` (query): boolean (optional) [default=True] — Should the voucher be sent to ledger? Requires the "Advanced Voucher" permission.

### Request Body: `Voucher`

**Required fields**: none explicitly marked (check validation)

  - `attachment`: object -> Document (use {id: <int>}) (optional)
  - `changes`: array of Change (optional) [READ-ONLY]
  - `date`: string (optional)
  - `description`: string (optional)
  - `displayName`: string (optional) [READ-ONLY]
  - `document`: object -> Document (use {id: <int>}) (optional)
  - `ediDocument`: object -> Document (use {id: <int>}) (optional)
  - `externalVoucherNumber`: string (optional) — External voucher number. Maximum 70 characters.
  - `id`: integer (int64) (optional)
  - `number`: integer (int32) (optional) [READ-ONLY] [min=0] — System generated number that cannot be changed.
  - `numberAsString`: string (optional) [READ-ONLY]
  - `postings`: array of Posting (optional)
  - `reverseVoucher`: object -> Voucher (use {id: <int>}) (optional)
  - `supplierVoucherType`: string (optional) [READ-ONLY] [enum=['TYPE_SUPPLIER_INVOICE_SIMPLE', 'TYPE_SUPPLIER_INVOICE_DETAILED']] — Supplier voucher type - simple and detailed.
  - `tempNumber`: integer (int32) (optional) [READ-ONLY] [min=0] — Temporary voucher number.
  - `url`: string (optional) [READ-ONLY]
  - `vendorInvoiceNumber`: string (optional) — Vendor invoice number.
  - `version`: integer (int32) (optional)
  - `voucherType`: object -> VoucherType (use {id: <int>}) (optional)
  - `wasAutoMatched`: boolean (optional) [READ-ONLY] — Voucher was auto matched
  - `year`: integer (int32) (optional) [READ-ONLY] [min=0] — System generated number that cannot be changed.

---
## POST /ledger/accountingDimensionName

**Summary**: Create a new free (aka 'user defined') accounting dimension

### Request Body: `AccountingDimensionName`

**Required fields**: none explicitly marked (check validation)

  - `active`: boolean (optional) — Indicates if the dimension is active.
  - `changes`: array of Change (optional) [READ-ONLY]
  - `description`: string (optional) — The description of the dimension.
  - `dimensionIndex`: integer (int32) (optional) [READ-ONLY] — The index of the dimension. Possible vales are 1, 2 and 3
  - `dimensionName`: string (optional) — The name of the dimension.
  - `id`: integer (int64) (optional)
  - `url`: string (optional) [READ-ONLY]
  - `version`: integer (int32) (optional)

---
## POST /ledger/accountingDimensionValue

**Summary**: Create a new value for one of the free (aka 'user defined') accounting dimensions

### Request Body: `AccountingDimensionValue`

**Required fields**: none explicitly marked (check validation)

  - `active`: boolean (optional) — Indicates if the value is active.
  - `changes`: array of Change (optional) [READ-ONLY]
  - `dimensionIndex`: integer (int32) (optional) [min=0] — The index of the dimension this value belongs to.
  - `displayName`: string (optional) — The name of the value.
  - `id`: integer (int64) (optional)
  - `nameAndNumber`: string (optional) [READ-ONLY] — The name and number of the value.
  - `number`: string (optional) — The number of the value, which can consist of letters and numbers.
  - `position`: integer (int32) (optional) [min=0] — The position of the value in the list of values for the dimension.
  - `showInVoucherRegistration`: boolean (optional) — Indicates if the value should be shown in voucher registration.
  - `url`: string (optional) [READ-ONLY]
  - `version`: integer (int32) (optional)

---
## POST /incomingInvoice

**Summary**: [BETA] create an invoice

### Query/Path Parameters
  - `sendTo` (query): string (optional) — 'inbox' | 'nonPosted' | 'ledger' | null. When null: defaults to 'inbox'.

### Request Body: `IncomingInvoiceAggregateExternalWrite`

**Required fields**: none explicitly marked (check validation)

  - `invoiceHeader`: object -> IncomingInvoiceHeaderExternalWrite (use {id: <int>}) (optional)
  - `orderLines`: array of IncomingOrderLineExternalWrite (optional)
  - `version`: integer (int32) (optional) — Voucher version

---
## POST /salary/transaction

**Summary**: Create a new salary transaction.

### Query/Path Parameters
  - `generateTaxDeduction` (query): boolean (optional) [default=False] — Generate tax deduction

### Request Body: `SalaryTransaction`

**Required fields**: none explicitly marked (check validation)

  - `changes`: array of Change (optional) [READ-ONLY]
  - `date`: string (optional) — Voucher date.
  - `id`: integer (int64) (optional)
  - `isHistorical`: boolean (optional) — With historical wage vouchers you can update the wage system with information dated before the opening balance.
  - `month`: integer (int32) (optional)
  - `paySlipsAvailableDate`: string (optional) — The date payslips are made available to the employee. Defaults to voucherDate.
  - `payslips`: array of Payslip (optional) — Link to individual payslip objects.
  - `url`: string (optional) [READ-ONLY]
  - `version`: integer (int32) (optional)
  - `year`: integer (int32) (optional)

---
## POST /travelExpense

**Summary**: Create travel expense.

### Request Body: `TravelExpense`

**Required fields**: none explicitly marked (check validation)

  - `accommodationAllowances`: array of AccommodationAllowance (optional) [READ-ONLY] — Link to individual accommodation allowances.
  - `accountingPeriodClosed`: boolean (optional) [READ-ONLY]
  - `accountingPeriodVATClosed`: boolean (optional) [READ-ONLY]
  - `actions`: array of Link (optional) [READ-ONLY]
  - `amount`: number (optional) [READ-ONLY]
  - `approvedBy`: object -> Employee (use {id: <int>}) (optional)
  - `approvedDate`: string (optional) [READ-ONLY]
  - `attachment`: object -> Document (use {id: <int>}) (optional)
  - `attachmentCount`: integer (int32) (optional) [READ-ONLY]
  - `attestation`: object -> Attestation (use {id: <int>}) (optional)
  - `attestationSteps`: array of AttestationStep (optional)
  - `changes`: array of Change (optional) [READ-ONLY]
  - `chargeableAmount`: number (optional) [READ-ONLY]
  - `chargeableAmountCurrency`: number (optional) [READ-ONLY]
  - `completedBy`: object -> Employee (use {id: <int>}) (optional)
  - `completedDate`: string (optional) [READ-ONLY]
  - `costs`: array of Cost (optional) — Link to individual costs.
  - `date`: string (optional) [READ-ONLY]
  - `department`: object -> Department (use {id: <int>}) (optional)
  - `displayName`: string (optional) [READ-ONLY]
  - `displayNameWithoutNumber`: string (optional) [READ-ONLY]
  - `employee`: object -> Employee (use {id: <int>}) (optional)
  - `fixedInvoicedAmount`: number (optional)
  - `freeDimension1`: object -> AccountingDimensionValue (use {id: <int>}) (optional)
  - `freeDimension2`: object -> AccountingDimensionValue (use {id: <int>}) (optional)
  - `freeDimension3`: object -> AccountingDimensionValue (use {id: <int>}) (optional)
  - `highRateVAT`: number (optional) [READ-ONLY]
  - `id`: integer (int64) (optional)
  - `invoice`: object -> Invoice (use {id: <int>}) (optional)
  - `isApproved`: boolean (optional) [READ-ONLY]
  - `isChargeable`: boolean (optional)
  - `isCompleted`: boolean (optional) [READ-ONLY]
  - `isFixedInvoicedAmount`: boolean (optional)
  - `isIncludeAttachedReceiptsWhenReinvoicing`: boolean (optional)
  - `isMarkupInvoicedPercent`: boolean (optional)
  - `isSalaryAdmin`: boolean (optional) [READ-ONLY]
  - `lowRateVAT`: number (optional) [READ-ONLY]
  - `markupInvoicedPercent`: number (optional)
  - `mediumRateVAT`: number (optional) [READ-ONLY]
  - `mileageAllowances`: array of MileageAllowance (optional) [READ-ONLY] — Link to individual mileage allowances.
  - `number`: integer (int32) (optional) [READ-ONLY]
  - `numberAsString`: string (optional) [READ-ONLY]
  - `paymentAmount`: number (optional) [READ-ONLY]
  - `paymentAmountCurrency`: number (optional) [READ-ONLY]
  - `paymentCurrency`: object -> Currency (use {id: <int>}) (optional)
  - `payslip`: object -> Payslip (use {id: <int>}) (optional)
  - `perDiemCompensations`: array of PerDiemCompensation (optional) — Link to individual per diem compensations.
  - `project`: object -> Project (use {id: <int>}) (optional)
  - `rejectedBy`: object -> Employee (use {id: <int>}) (optional)
  - `rejectedComment`: string (optional) [READ-ONLY]
  - `showPayslip`: boolean (optional) [READ-ONLY]
  - `state`: string (optional) [READ-ONLY] [enum=['ALL', 'REJECTED', 'OPEN', 'APPROVED', 'SALARY_PAID', 'DELIVERED']]
  - `stateName`: string (optional) [READ-ONLY]
  - `title`: string (optional)
  - `travelAdvance`: number (optional)
  - `travelDetails`: object -> TravelDetails (use {id: <int>}) (optional)
  - `type`: integer (int32) (optional) [READ-ONLY] [min=0]
  - `url`: string (optional) [READ-ONLY]
  - `vatType`: object -> VatType (use {id: <int>}) (optional)
  - `version`: integer (int32) (optional)
  - `voucher`: object -> Voucher (use {id: <int>}) (optional)

---
## POST /travelExpense/perDiemCompensation

**Summary**: Create per diem compensation.

### Request Body: `PerDiemCompensation`

**Required fields**: none explicitly marked (check validation)

  - `address`: string (optional)
  - `amount`: number (optional)
  - `changes`: array of Change (optional) [READ-ONLY]
  - `count`: integer (int32) (optional)
  - `countryCode`: string (optional)
  - `id`: integer (int64) (optional)
  - `isDeductionForBreakfast`: boolean (optional)
  - `isDeductionForDinner`: boolean (optional)
  - `isDeductionForLunch`: boolean (optional)
  - `location`: string (optional)
  - `overnightAccommodation`: string (optional) [enum=['NONE', 'HOTEL', 'BOARDING_HOUSE_WITHOUT_COOKING', 'BOARDING_HOUSE_WITH_COOKING']] — Set what sort of accommodation was had overnight.
  - `rate`: number (optional)
  - `rateCategory`: object -> TravelExpenseRateCategory (use {id: <int>}) (optional)
  - `rateType`: object -> TravelExpenseRate (use {id: <int>}) (optional)
  - `travelExpense`: object -> TravelExpense (use {id: <int>}) (optional)
  - `travelExpenseZoneId`: integer (int32) (optional) — Optional travel expense zone id. If not specified, the value from field zone will be used.
  - `url`: string (optional) [READ-ONLY]
  - `version`: integer (int32) (optional)

---
## POST /travelExpense/cost

**Summary**: Create cost.

### Request Body: `Cost`

**Required fields**: none explicitly marked (check validation)

  - `amountCurrencyIncVat`: number (optional)
  - `amountNOKInclVAT`: number (optional)
  - `amountNOKInclVATHigh`: number (optional) [READ-ONLY]
  - `amountNOKInclVATLow`: number (optional) [READ-ONLY]
  - `amountNOKInclVATMedium`: number (optional) [READ-ONLY]
  - `category`: string (optional)
  - `changes`: array of Change (optional) [READ-ONLY]
  - `comments`: string (optional)
  - `costCategory`: object -> TravelCostCategory (use {id: <int>}) (optional)
  - `currency`: object -> Currency (use {id: <int>}) (optional)
  - `date`: string (optional)
  - `id`: integer (int64) (optional)
  - `isChargeable`: boolean (optional)
  - `isPaidByEmployee`: boolean (optional) [READ-ONLY]
  - `participants`: array of CostParticipant (optional) — Link to individual expense participant.
  - `paymentType`: object -> TravelPaymentType (use {id: <int>}) (optional)
  - `predictions`: object (optional)
  - `rate`: number (optional)
  - `travelExpense`: object -> TravelExpense (use {id: <int>}) (optional)
  - `url`: string (optional) [READ-ONLY]
  - `vatType`: object -> VatType (use {id: <int>}) (optional)
  - `version`: integer (int32) (optional)

---
## POST /timesheet/entry

**Summary**: Add new timesheet entry. Only one entry per employee/date/activity/project combination is supported.

### Request Body: `TimesheetEntry`

**Required fields**: none explicitly marked (check validation)

  - `activity`: object -> Activity (use {id: <int>}) (optional)
  - `changes`: array of Change (optional) [READ-ONLY]
  - `chargeable`: boolean (optional) [READ-ONLY]
  - `chargeableHours`: number (optional) [READ-ONLY]
  - `comment`: string (optional)
  - `date`: string (optional)
  - `employee`: object -> Employee (use {id: <int>}) (optional)
  - `hourlyCost`: number (optional) [READ-ONLY]
  - `hourlyCostPercentage`: number (optional) [READ-ONLY]
  - `hourlyRate`: number (optional) [READ-ONLY]
  - `hours`: number (optional)
  - `id`: integer (int64) (optional)
  - `invoice`: object -> Invoice (use {id: <int>}) (optional)
  - `locked`: boolean (optional) [READ-ONLY] — Indicates if the hour can be changed.
  - `project`: object -> Project (use {id: <int>}) (optional)
  - `projectChargeableHours`: number (optional) [min=0, max=24] — Number of chargeable hours on an activity connected to a project.
  - `timeClocks`: array of TimeClock (optional) [READ-ONLY] — Link to stop watches on this hour.
  - `url`: string (optional) [READ-ONLY]
  - `version`: integer (int32) (optional)

---
## PUT /employee/{id}

**Summary**: Update employee.

### Query/Path Parameters
  - `id` (path): integer (int64) (REQUIRED) — Element ID

### Request Body: `Employee`

**Required fields**: none explicitly marked (check validation)

  - `address`: object -> Address (use {id: <int>}) (optional)
  - `allowInformationRegistration`: boolean (optional) [READ-ONLY] — Determines if salary information can be registered on the user including hours, travel expenses and employee expenses. The user may also be selected as a project member on projects.
  - `bankAccountNumber`: string (optional)
  - `bic`: string (optional) — Bic (swift) field
  - `changes`: array of Change (optional) [READ-ONLY]
  - `comments`: string (optional)
  - `companyId`: integer (int32) (optional) [READ-ONLY]
  - `creditorBankCountryId`: integer (int32) (optional) — Country of creditor bank field
  - `dateOfBirth`: string (optional)
  - `department`: object -> Department (use {id: <int>}) (optional)
  - `displayName`: string (optional) [READ-ONLY]
  - `dnumber`: string (optional)
  - `email`: string (optional)
  - `employeeCategory`: object -> EmployeeCategory (use {id: <int>}) (optional)
  - `employeeNumber`: string (optional)
  - `employments`: array of Employment (optional)
  - `firstName`: string (optional)
  - `holidayAllowanceEarned`: object -> HolidayAllowanceEarned (use {id: <int>}) (optional)
  - `iban`: string (optional) — IBAN field
  - `id`: integer (int64) (optional)
  - `internationalId`: object -> InternationalId (use {id: <int>}) (optional)
  - `isAuthProjectOverviewURL`: boolean (optional) [READ-ONLY]
  - `isContact`: boolean (optional) — Determines if the employee is a contact (external) in the company.
  - `isProxy`: boolean (optional) [READ-ONLY] — True if this Employee object represents an accounting or auditor office
  - `lastName`: string (optional)
  - `nationalIdentityNumber`: string (optional)
  - `phoneNumberHome`: string (optional)
  - `phoneNumberMobile`: string (optional)
  - `phoneNumberMobileCountry`: object -> Country (use {id: <int>}) (optional)
  - `phoneNumberWork`: string (optional)
  - `pictureId`: integer (int32) (optional) [READ-ONLY]
  - `url`: string (optional) [READ-ONLY]
  - `userType`: string (optional) [enum=['STANDARD', 'EXTENDED', 'NO_ACCESS']] — Define the employee's user type.<br>STANDARD: Reduced access. Users with limited system entitlements.<br>EXTENDED: Users can be given all system entitlements.<br>NO_ACCESS: User with no log on access.
  - `usesAbroadPayment`: boolean (optional) — UsesAbroadPayment field. Determines if we should use domestic or abroad remittance. To be able to use abroad remittance, one has to: 1: have Autopay 2: have valid combination of the fields Iban, Bic (
  - `version`: integer (int32) (optional)
  - `vismaConnect2FAactive`: boolean (optional) [READ-ONLY]

---
## PUT /employee/employment/{id}

**Summary**: Update employemnt. 

### Query/Path Parameters
  - `id` (path): integer (int64) (REQUIRED) — Element ID

### Request Body: `Employment`

**Required fields**: none explicitly marked (check validation)

  - `changes`: array of Change (optional) [READ-ONLY]
  - `division`: object -> Division (use {id: <int>}) (optional)
  - `employee`: object -> Employee (use {id: <int>}) (optional)
  - `employmentDetails`: array of EmploymentDetails (optional)
  - `employmentEndReason`: string (optional) [enum=['EMPLOYMENT_END_EXPIRED', 'EMPLOYMENT_END_EMPLOYEE', 'EMPLOYMENT_END_EMPLOYER', 'EMPLOYMENT_END_WRONGLY_REPORTED', 'EMPLOYMENT_END_SYSTEM_OR_ACCOUNTANT_CHANGE', 'EMPLOYMENT_END_INTERNAL_CHANGE']] — Define the employment end reason.
  - `employmentId`: string (optional) — Existing employment ID used by the current accounting system
  - `endDate`: string (optional)
  - `id`: integer (int64) (optional)
  - `isMainEmployer`: boolean (optional) — Determines if company is main employer for the employee. Default value is true.<br />Some values will be default set if not sent upon creation of employment: <br/> If isMainEmployer is NOT sent and ta
  - `isRemoveAccessAtEmploymentEnded`: boolean (optional) — If true, access to the employee will be removed when the employment ends. <br />This field is part of the Employee object, therefore changing it for one Employment affects all Employments.
  - `lastSalaryChangeDate`: string (optional)
  - `latestSalary`: object -> EmploymentDetails (use {id: <int>}) (optional)
  - `noEmploymentRelationship`: boolean (optional) — Activate pensions and other benefits with no employment relationship.
  - `startDate`: string (optional)
  - `taxDeductionCode`: string (optional) [enum=['loennFraHovedarbeidsgiver', 'loennFraBiarbeidsgiver', 'pensjon', 'loennTilUtenrikstjenestemann', 'loennKunTrygdeavgiftTilUtenlandskBorger', 'loennKunTrygdeavgiftTilUtenlandskBorgerSomGrensegjenger', 'introduksjonsstoenad', 'ufoereytelserFraAndre', '']] — EMPTY - represents that a tax deduction code is not set on the employment. It is illegal to set the field to this value.  <br /> Default value of this field is loennFraHovedarbeidsgiver or loennFraBia
  - `url`: string (optional) [READ-ONLY]
  - `version`: integer (int32) (optional)

---
## PUT /employee/employment/details/{id}

**Summary**: Update employment details. 

### Query/Path Parameters
  - `id` (path): integer (int64) (REQUIRED) — Element ID

### Request Body: `EmploymentDetails`

**Required fields**: none explicitly marked (check validation)

  - `annualSalary`: number (optional)
  - `changes`: array of Change (optional) [READ-ONLY]
  - `date`: string (optional)
  - `employment`: object -> Employment (use {id: <int>}) (optional)
  - `employmentForm`: string (optional) [enum=['PERMANENT', 'TEMPORARY', 'PERMANENT_AND_HIRED_OUT', 'TEMPORARY_AND_HIRED_OUT', 'TEMPORARY_ON_CALL', 'NOT_CHOSEN']] — Define the employment form.
  - `employmentType`: string (optional) [enum=['ORDINARY', 'MARITIME', 'FREELANCE', 'NOT_CHOSEN']] — Define the employment type.
  - `hourlyWage`: number (optional)
  - `id`: integer (int64) (optional)
  - `maritimeEmployment`: object -> MaritimeEmployment (use {id: <int>}) (optional)
  - `monthlySalary`: number (optional) [READ-ONLY]
  - `occupationCode`: object -> OccupationCode (use {id: <int>}) (optional)
  - `payrollTaxMunicipalityId`: object -> Municipality (use {id: <int>}) (optional)
  - `percentageOfFullTimeEquivalent`: number (optional)
  - `remunerationType`: string (optional) [enum=['MONTHLY_WAGE', 'HOURLY_WAGE', 'COMMISION_PERCENTAGE', 'FEE', 'NOT_CHOSEN', 'PIECEWORK_WAGE']] — Define the remuneration type.
  - `shiftDurationHours`: number (optional)
  - `url`: string (optional) [READ-ONLY]
  - `version`: integer (int32) (optional)
  - `workingHoursScheme`: string (optional) [enum=['NOT_SHIFT', 'ROUND_THE_CLOCK', 'SHIFT_365', 'OFFSHORE_336', 'CONTINUOUS', 'OTHER_SHIFT', 'NOT_CHOSEN']] — Define the working hours scheme type. If you enter a value for SHIFT WORK, you must also enter value for shiftDurationHours

---
## PUT /project/{id}

**Summary**: [BETA] Update project.

### Query/Path Parameters
  - `id` (path): integer (int64) (REQUIRED) — Element ID

### Request Body: `Project`

**Required fields**: none explicitly marked (check validation)

  - `accessType`: string (optional) [enum=['NONE', 'READ', 'WRITE']] — READ/WRITE access on project
  - `accountingDimensionValues`: array of AccountingDimensionValue (optional) — [BETA - Requires pilot feature] Free dimensions for the project.
  - `attention`: object -> Contact (use {id: <int>}) (optional)
  - `boligmappaAddress`: object -> Address (use {id: <int>}) (optional)
  - `changes`: array of Change (optional) [READ-ONLY]
  - `contact`: object -> Contact (use {id: <int>}) (optional)
  - `contributionMarginPercent`: number (optional) [READ-ONLY]
  - `currency`: object -> Currency (use {id: <int>}) (optional)
  - `customer`: object -> Customer (use {id: <int>}) (optional)
  - `customerName`: string (optional) [READ-ONLY]
  - `deliveryAddress`: object -> Address (use {id: <int>}) (optional)
  - `department`: object -> Department (use {id: <int>}) (optional)
  - `description`: string (optional)
  - `discountPercentage`: number (optional) [READ-ONLY] — Project discount percentage.
  - `displayName`: string (optional) [READ-ONLY]
  - `displayNameFormat`: string (optional) [enum=['NAME_STANDARD', 'NAME_INCL_CUSTOMER_NAME', 'NAME_INCL_PARENT_NAME', 'NAME_INCL_PARENT_NUMBER', 'NAME_INCL_PARENT_NAME_AND_NUMBER']] — Defines project name presentation in overviews.
  - `endDate`: string (optional)
  - `externalAccountsNumber`: string (optional)
  - `fixedprice`: number (optional) — Fixed price amount, in the project's currency.
  - `forParticipantsOnly`: boolean (optional) — Set to true if only project participants can register information on the project
  - `generalProjectActivitiesPerProjectOnly`: boolean (optional) — Set to true if a general project activity must be linked to project to allow time tracking.
  - `hierarchyLevel`: integer (int32) (optional) [READ-ONLY]
  - `hierarchyNameAndNumber`: string (optional) [READ-ONLY]
  - `id`: integer (int64) (optional)
  - `ignoreCompanyProductDiscountAgreement`: boolean (optional)
  - `invoiceComment`: string (optional) — Comment for project invoices
  - `invoiceDueDate`: integer (int32) (optional) — invoice due date
  - `invoiceDueDateType`: string (optional) [enum=['DAYS', 'MONTHS', 'RECURRING_DAY_OF_MONTH']] — Set the time unit of invoiceDueDate. The special case RECURRING_DAY_OF_MONTH enables the due date to be fixed to a specific day of the month, in this case the fixed due date will automatically be set 
  - `invoiceOnAccountVatHigh`: boolean (optional) — The on account(a konto) amounts including VAT
  - `invoiceReceiverEmail`: string (optional) — Set the project's invoice receiver email. Will override the default invoice receiver email of any customer that may also be set in the request body.
  - `invoiceReserveTotalAmountCurrency`: number (optional) [READ-ONLY] — Total invoice reserve
  - `invoicingPlan`: array of Invoice (optional) [READ-ONLY] — Invoicing plans tied to the project
  - `isClosed`: boolean (optional)
  - `isFixedPrice`: boolean (optional) — Project is fixed price if set to true, hourly rate if set to false.
  - `isInternal`: boolean (optional)
  - `isOffer`: boolean (optional) — If is Project Offer set to true, if is Project set to false. The default value is false.
  - `isPriceCeiling`: boolean (optional) — Set to true if an hourly rate project has a price ceiling.
  - `isReadyForInvoicing`: boolean (optional)
  - `mainProject`: object -> Project (use {id: <int>}) (optional)
  - `markUpFeesEarned`: number (optional) — Set mark-up (%) for fees earned.
  - `markUpOrderLines`: number (optional) — Set mark-up (%) for order lines.
  - `name`: string (optional)
  - `number`: string (optional) — If NULL, a number is generated automatically.
  - `numberOfProjectParticipants`: integer (int32) (optional) [READ-ONLY]
  - `numberOfSubProjects`: integer (int32) (optional) [READ-ONLY]
  - `orderLines`: array of ProjectOrderLine (optional) [READ-ONLY] — Order lines tied to the order
  - `overdueNoticeEmail`: string (optional) — Set the project's overdue notice email. Will override the default overdue notice email of any customer that may also be set in the request body.
  - `participants`: array of ProjectParticipant (optional) — Link to individual project participants.
  - `preliminaryInvoice`: object -> Invoice (use {id: <int>}) (optional)
  - `priceCeilingAmount`: number (optional) — Price ceiling amount, in the project's currency.
  - `projectActivities`: array of ProjectActivity (optional) — Project Activities
  - `projectCategory`: object -> ProjectCategory (use {id: <int>}) (optional)
  - `projectHourlyRates`: array of ProjectHourlyRate (optional) — Project Rate Types tied to the project.
  - `projectManager`: object -> Employee (use {id: <int>}) (optional)
  - `projectManagerNameAndNumber`: string (optional) [READ-ONLY]
  - `reference`: string (optional)
  - `startDate`: string (optional)
  - `totalInvoicedOnAccountAmountAbsoluteCurrency`: number (optional) [READ-ONLY] — Amount paid on account(a konto)
  - `url`: string (optional) [READ-ONLY]
  - `useProductNetPrice`: boolean (optional)
  - `vatType`: object -> VatType (use {id: <int>}) (optional)
  - `version`: integer (int32) (optional)

---
## PUT /invoice/{id}/:payment

**Summary**: Update invoice. The invoice is updated with payment information. The amount is in the invoice’s currency.

### Query/Path Parameters
  - `id` (path): integer (int64) (REQUIRED) — Invoice id
  - `paymentDate` (query): string (REQUIRED) — Payment date
  - `paymentTypeId` (query): integer (int64) (REQUIRED) — PaymentType id
  - `paidAmount` (query): number (REQUIRED) — Amount paid by the customer in the currency determined by the account of the paymentType
  - `paidAmountCurrency` (query): number (optional) — Amount paid by customer in the invoice currency. Optional, but required for invoices in alternate currencies.

*No request body — uses query parameters only*

---
## PUT /invoice/{id}/:createCreditNote

**Summary**: Creates a new Invoice representing a credit memo that nullifies the given invoice. Updates this invoice and any pre-existing inverse invoice.

### Query/Path Parameters
  - `id` (path): integer (int64) (REQUIRED) — Invoice id
  - `date` (query): string (REQUIRED) — Credit note date
  - `comment` (query): string (optional) — Comment
  - `creditNoteEmail` (query): string (optional) — The credit note will not be sent if the customer send type is email and this field is empty
  - `sendToCustomer` (query): boolean (optional) [default=True] — Equals
  - `sendType` (query): string (optional) — Equals

*No request body — uses query parameters only*

---
## PUT /order/{id}/:invoice

**Summary**: Create new invoice or subscription invoice from order.

### Query/Path Parameters
  - `id` (path): integer (int64) (REQUIRED) — ID of order to invoice.
  - `invoiceDate` (query): string (REQUIRED) — The invoice date
  - `sendToCustomer` (query): boolean (optional) [default=True] — Send invoice to customer
  - `sendType` (query): string (optional) — Send type used for sending the invoice
  - `paymentTypeId` (query): integer (int64) (optional) — Payment type to register prepayment of the invoice. paymentTypeId and paidAmount are optional, but both must be provided if the invoice has already been paid. The payment type must be related to an ac
  - `paidAmount` (query): number (optional) — Paid amount to register prepayment of the invoice, in invoice currency. paymentTypeId and paidAmount are optional, but both must be provided if the invoice has already been paid. This amount is in the
  - `paidAmountAccountCurrency` (query): number (optional) — Amount paid in payment type currency
  - `paymentTypeIdRestAmount` (query): integer (int64) (optional) — Payment type of rest amount. It is possible to have two prepaid payments when invoicing. If paymentTypeIdRestAmount > 0, this second payment will be calculated as invoice amount - paidAmount
  - `paidAmountAccountCurrencyRest` (query): number (optional) — Amount rest in payment type currency
  - `createOnAccount` (query): string (optional) — Create on account(a konto)
  - `amountOnAccount` (query): number (optional) [default=0] — Amount on account
  - `onAccountComment` (query): string (optional) [default=] — On account comment
  - `createBackorder` (query): boolean (optional) [default=False] — Create a backorder for this order, available only for pilot users
  - `invoiceIdIfIsCreditNote` (query): integer (int64) (optional) [default=0] — Id of the invoice a credit note refers to
  - `overrideEmailAddress` (query): string (optional) — Will override email address if sendType = EMAIL

*No request body — uses query parameters only*

---
## GET /invoice

**Summary**: Find invoices corresponding with sent data. Includes charged outgoing invoices only.

### Query/Path Parameters
  - `id` (query): string (optional) — List of IDs
  - `invoiceDateFrom` (query): string (REQUIRED) — From and including
  - `invoiceDateTo` (query): string (REQUIRED) — To and excluding
  - `invoiceNumber` (query): string (optional) — Equals
  - `kid` (query): string (optional) — Equals
  - `voucherId` (query): string (optional) — List of IDs
  - `customerId` (query): string (optional) — Equals
  - `from` (query): integer (optional) [default=0] — From index
  - `count` (query): integer (optional) [default=1000] — Number of elements to return
  - `sorting` (query): string (optional) [default=] — Sorting pattern
  - `fields` (query): string (optional) [default=] — Fields filter pattern

---
## GET /ledger/posting

**Summary**: Find postings corresponding with sent data.

### Query/Path Parameters
  - `dateFrom` (query): string (REQUIRED) — Format is yyyy-MM-dd (from and incl.).
  - `dateTo` (query): string (REQUIRED) — Format is yyyy-MM-dd (to and excl.).
  - `openPostings` (query): string (optional) — Deprecated
  - `accountId` (query): integer (int64) (optional) — Element ID for filtering
  - `supplierId` (query): integer (int64) (optional) — Element ID for filtering
  - `customerId` (query): integer (int64) (optional) — Element ID for filtering
  - `employeeId` (query): integer (int64) (optional) — Element ID for filtering
  - `departmentId` (query): integer (int64) (optional) — Element ID for filtering
  - `projectId` (query): integer (int64) (optional) — Element ID for filtering
  - `productId` (query): integer (int64) (optional) — Element ID for filtering
  - `accountNumberFrom` (query): integer (int32) (optional) — Element ID for filtering
  - `accountNumberTo` (query): integer (int32) (optional) — Element ID for filtering
  - `type` (query): string (optional) — Element ID for filtering
  - `accountingDimensionValue1Id` (query): integer (int64) (optional) — Id of first free accounting dimension.
  - `accountingDimensionValue2Id` (query): integer (int64) (optional) — Id of second free accounting dimension.
  - `accountingDimensionValue3Id` (query): integer (int64) (optional) — Id of third free accounting dimension.
  - `from` (query): integer (optional) [default=0] — From index
  - `count` (query): integer (optional) [default=1000] — Number of elements to return
  - `sorting` (query): string (optional) [default=] — Sorting pattern
  - `fields` (query): string (optional) [default=] — Fields filter pattern

---
## GET /employee/employment/occupationCode

**Summary**: Find all profession codes.

### Query/Path Parameters
  - `id` (query): string (optional) — Element ID
  - `nameNO` (query): string (optional) — Containing
  - `code` (query): string (optional) — Containing
  - `from` (query): integer (optional) [default=0] — From index
  - `count` (query): integer (optional) [default=1000] — Number of elements to return
  - `sorting` (query): string (optional) [default=] — Sorting pattern
  - `fields` (query): string (optional) [default=] — Fields filter pattern

---
## KEY NESTED SCHEMAS (commonly referenced)

### Address
**Required**: none marked
  - `addressAsString`: string (optional) [READ-ONLY]
  - `addressLine1`: string (optional)
  - `addressLine2`: string (optional)
  - `bnr`: integer (int32) (optional) [min=0]
  - `changes`: array of Change (optional) [READ-ONLY]
  - `city`: string (optional)
  - `country`: object -> Country (use {id: <int>}) (optional)
  - `displayName`: string (optional) [READ-ONLY]
  - `displayNameInklMatrikkel`: string (optional) [READ-ONLY]
  - `fnr`: integer (int32) (optional) [min=0]
  - `gnr`: integer (int32) (optional) [min=0]
  - `id`: integer (int64) (optional)
  - `knr`: integer (int32) (optional) [min=0]
  - `postalCode`: string (optional)
  - `snr`: integer (int32) (optional) [min=0]
  - `unitNumber`: string (optional)
  - `url`: string (optional) [READ-ONLY]
  - `version`: integer (int32) (optional)

### DeliveryAddress
**Required**: none marked
  - `addressAsString`: string (optional) [READ-ONLY]
  - `addressLine1`: string (optional)
  - `addressLine2`: string (optional)
  - `bnr`: integer (int32) (optional) [min=0]
  - `changes`: array of Change (optional) [READ-ONLY]
  - `city`: string (optional)
  - `country`: object -> Country (use {id: <int>}) (optional)
  - `customerVendor`: object -> Company (use {id: <int>}) (optional)
  - `displayName`: string (optional) [READ-ONLY]
  - `displayNameInklMatrikkel`: string (optional) [READ-ONLY]
  - `fnr`: integer (int32) (optional) [min=0]
  - `gnr`: integer (int32) (optional) [min=0]
  - `id`: integer (int64) (optional)
  - `knr`: integer (int32) (optional) [min=0]
  - `name`: string (optional)
  - `postalCode`: string (optional)
  - `snr`: integer (int32) (optional) [min=0]
  - `unitNumber`: string (optional)
  - `url`: string (optional) [READ-ONLY]
  - `version`: integer (int32) (optional)

### VoucherType
**Required**: none marked
  - `changes`: array of Change (optional) [READ-ONLY]
  - `displayName`: string (optional)
  - `id`: integer (int64) (optional)
  - `name`: string (optional)
  - `url`: string (optional) [READ-ONLY]
  - `version`: integer (int32) (optional)

### Posting
**Required**: none marked
  - `account`: object -> Account (use {id: <int>}) (optional)
  - `amortizationAccount`: object -> Account (use {id: <int>}) (optional)
  - `amortizationEndDate`: string (optional)
  - `amortizationStartDate`: string (optional) — Amortization start date. AmortizationAccountId, amortizationStartDate and amortizationEndDate should be provided.
  - `amount`: number (optional)
  - `amountCurrency`: number (optional)
  - `amountGross`: number (optional)
  - `amountGrossCurrency`: number (optional)
  - `asset`: object -> Asset (use {id: <int>}) (optional)
  - `changes`: array of Change (optional) [READ-ONLY]
  - `closeGroup`: object -> CloseGroup (use {id: <int>}) (optional)
  - `currency`: object -> Currency (use {id: <int>}) (optional)
  - `customer`: object -> Customer (use {id: <int>}) (optional)
  - `date`: string (optional)
  - `department`: object -> Department (use {id: <int>}) (optional)
  - `description`: string (optional)
  - `employee`: object -> Employee (use {id: <int>}) (optional)
  - `externalRef`: string (optional) [READ-ONLY] — External reference for identifying payment basis of the posting, e.g., KID, customer identification or credit note number.
  - `freeAccountingDimension1`: object -> AccountingDimensionValue (use {id: <int>}) (optional)
  - `freeAccountingDimension2`: object -> AccountingDimensionValue (use {id: <int>}) (optional)
  - `freeAccountingDimension3`: object -> AccountingDimensionValue (use {id: <int>}) (optional)
  - `id`: integer (int64) (optional)
  - `invoiceNumber`: string (optional)
  - `isAmountVatClosed`: boolean (optional) [READ-ONLY] — Is amount of this posting (for VAT purposes) changeable
  - `isVatReadonly`: boolean (optional) [READ-ONLY] — Is vat code readonly?
  - `matched`: boolean (optional) [READ-ONLY]
  - `postingRuleId`: integer (int32) (optional) — The payment type id associated with the posting. This ID will only be set if the payment types used is an internal payment type like 'Nettbank' - it is not set if the payment is a bank payment like Au
  - `product`: object -> Product (use {id: <int>}) (optional)
  - `project`: object -> Project (use {id: <int>}) (optional)
  - `quantityAmount1`: number (optional) — The quantity amount associated with the posting
  - `quantityAmount2`: number (optional) — The quantity amount associated with the posting
  - `quantityType1`: object -> ProductUnit (use {id: <int>}) (optional)
  - `quantityType2`: object -> ProductUnit (use {id: <int>}) (optional)
  - `row`: integer (int32) (optional) [min=0]
  - `supplier`: object -> Supplier (use {id: <int>}) (optional)
  - `systemGenerated`: boolean (optional) [READ-ONLY]
  - `taxTransactionType`: string (optional) [READ-ONLY]
  - `taxTransactionTypeId`: integer (int32) (optional) [READ-ONLY]
  - `termOfPayment`: string (optional)
  - `type`: string (optional) [READ-ONLY] [enum=['INCOMING_PAYMENT', 'INCOMING_PAYMENT_OPPOSITE', 'INCOMING_INVOICE_CUSTOMER_POSTING', 'INVOICE_EXPENSE', 'OUTGOING_INVOICE_CUSTOMER_POSTING', 'WAGE']]
  - `url`: string (optional) [READ-ONLY]
  - `vatType`: object -> VatType (use {id: <int>}) (optional)
  - `version`: integer (int32) (optional)
  - `voucher`: object -> Voucher (use {id: <int>}) (optional)

### OrderLine
**Required**: none marked
  - `amountExcludingVatCurrency`: number (optional) [READ-ONLY] — Total amount on order line excluding VAT in the order's currency
  - `amountIncludingVatCurrency`: number (optional) [READ-ONLY] — Total amount on order line including VAT in the order's currency
  - `changes`: array of Change (optional) [READ-ONLY]
  - `count`: number (optional)
  - `currency`: object -> Currency (use {id: <int>}) (optional)
  - `description`: string (optional)
  - `discount`: number (optional) — Discount given as a percentage (%)
  - `displayName`: string (optional) [READ-ONLY] — Display name of order line
  - `id`: integer (int64) (optional)
  - `inventory`: object -> Inventory (use {id: <int>}) (optional)
  - `inventoryLocation`: object -> InventoryLocation (use {id: <int>}) (optional)
  - `isCharged`: boolean (optional) — Flag indicating whether the order line is charged or not.
  - `isPicked`: boolean (optional) — Only used for Logistics customers who activated the available inventory functionality. Represents whether the line has been picked up or not.
  - `isSubscription`: boolean (optional)
  - `markup`: number (optional) — Markup given as a percentage (%)
  - `order`: object -> Order (use {id: <int>}) (optional)
  - `orderGroup`: object -> OrderGroup (use {id: <int>}) (optional)
  - `orderedQuantity`: number (optional) — Only used for Logistics customers who activated the Backorder functionality. Represents the quantity that was ordered. If nothing is specified, the ordered quantity will be the same as the delivered q
  - `pickedDate`: string (optional) — Only used for Logistics customers who activated the available inventory functionality. Represents the pick date for an order line or null if the line was not picked.
  - `product`: object -> Product (use {id: <int>}) (optional)
  - `sortIndex`: integer (int32) (optional) [min=0] — Defines the presentation order of the lines. Does not need to be, and is often not continuous. Only applicable if parent order has orderLineSorting as CUSTOM.
  - `subscriptionPeriodEnd`: string (optional)
  - `subscriptionPeriodStart`: string (optional)
  - `unitCostCurrency`: number (optional) — Unit price purchase (cost) excluding VAT in the order's currency
  - `unitPriceExcludingVatCurrency`: number (optional) — Unit price of purchase excluding VAT in the order's currency. If only unit price Excl. VAT or unit price Inc. VAT is supplied, we will calculate and update the missing field.
  - `unitPriceIncludingVatCurrency`: number (optional) — Unit price of purchase including VAT in the order's currency. If only unit price Excl. VAT or unit price Inc. VAT is supplied, we will calculate and update the missing field.
  - `url`: string (optional) [READ-ONLY]
  - `vatType`: object -> VatType (use {id: <int>}) (optional)
  - `vendor`: object -> Company (use {id: <int>}) (optional)
  - `version`: integer (int32) (optional)

### Employment
**Required**: none marked
  - `changes`: array of Change (optional) [READ-ONLY]
  - `division`: object -> Division (use {id: <int>}) (optional)
  - `employee`: object -> Employee (use {id: <int>}) (optional)
  - `employmentDetails`: array of EmploymentDetails (optional)
  - `employmentEndReason`: string (optional) [enum=['EMPLOYMENT_END_EXPIRED', 'EMPLOYMENT_END_EMPLOYEE', 'EMPLOYMENT_END_EMPLOYER', 'EMPLOYMENT_END_WRONGLY_REPORTED', 'EMPLOYMENT_END_SYSTEM_OR_ACCOUNTANT_CHANGE', 'EMPLOYMENT_END_INTERNAL_CHANGE']] — Define the employment end reason.
  - `employmentId`: string (optional) — Existing employment ID used by the current accounting system
  - `endDate`: string (optional)
  - `id`: integer (int64) (optional)
  - `isMainEmployer`: boolean (optional) — Determines if company is main employer for the employee. Default value is true.<br />Some values will be default set if not sent upon creation of employment: <br/> If isMainEmployer is NOT sent and ta
  - `isRemoveAccessAtEmploymentEnded`: boolean (optional) — If true, access to the employee will be removed when the employment ends. <br />This field is part of the Employee object, therefore changing it for one Employment affects all Employments.
  - `lastSalaryChangeDate`: string (optional)
  - `latestSalary`: object -> EmploymentDetails (use {id: <int>}) (optional)
  - `noEmploymentRelationship`: boolean (optional) — Activate pensions and other benefits with no employment relationship.
  - `startDate`: string (optional)
  - `taxDeductionCode`: string (optional) [enum=['loennFraHovedarbeidsgiver', 'loennFraBiarbeidsgiver', 'pensjon', 'loennTilUtenrikstjenestemann', 'loennKunTrygdeavgiftTilUtenlandskBorger', 'loennKunTrygdeavgiftTilUtenlandskBorgerSomGrensegjenger', 'introduksjonsstoenad', 'ufoereytelserFraAndre', '']] — EMPTY - represents that a tax deduction code is not set on the employment. It is illegal to set the field to this value.  <br /> Default value of this field is loennFraHovedarbeidsgiver or loennFraBia
  - `url`: string (optional) [READ-ONLY]
  - `version`: integer (int32) (optional)

### EmploymentDetails
**Required**: none marked
  - `annualSalary`: number (optional)
  - `changes`: array of Change (optional) [READ-ONLY]
  - `date`: string (optional)
  - `employment`: object -> Employment (use {id: <int>}) (optional)
  - `employmentForm`: string (optional) [enum=['PERMANENT', 'TEMPORARY', 'PERMANENT_AND_HIRED_OUT', 'TEMPORARY_AND_HIRED_OUT', 'TEMPORARY_ON_CALL', 'NOT_CHOSEN']] — Define the employment form.
  - `employmentType`: string (optional) [enum=['ORDINARY', 'MARITIME', 'FREELANCE', 'NOT_CHOSEN']] — Define the employment type.
  - `hourlyWage`: number (optional)
  - `id`: integer (int64) (optional)
  - `maritimeEmployment`: object -> MaritimeEmployment (use {id: <int>}) (optional)
  - `monthlySalary`: number (optional) [READ-ONLY]
  - `occupationCode`: object -> OccupationCode (use {id: <int>}) (optional)
  - `payrollTaxMunicipalityId`: object -> Municipality (use {id: <int>}) (optional)
  - `percentageOfFullTimeEquivalent`: number (optional)
  - `remunerationType`: string (optional) [enum=['MONTHLY_WAGE', 'HOURLY_WAGE', 'COMMISION_PERCENTAGE', 'FEE', 'NOT_CHOSEN', 'PIECEWORK_WAGE']] — Define the remuneration type.
  - `shiftDurationHours`: number (optional)
  - `url`: string (optional) [READ-ONLY]
  - `version`: integer (int32) (optional)
  - `workingHoursScheme`: string (optional) [enum=['NOT_SHIFT', 'ROUND_THE_CLOCK', 'SHIFT_365', 'OFFSHORE_336', 'CONTINUOUS', 'OTHER_SHIFT', 'NOT_CHOSEN']] — Define the working hours scheme type. If you enter a value for SHIFT WORK, you must also enter value for shiftDurationHours

### SalaryTransaction
**Required**: none marked
  - `changes`: array of Change (optional) [READ-ONLY]
  - `date`: string (optional) — Voucher date.
  - `id`: integer (int64) (optional)
  - `isHistorical`: boolean (optional) — With historical wage vouchers you can update the wage system with information dated before the opening balance.
  - `month`: integer (int32) (optional)
  - `paySlipsAvailableDate`: string (optional) — The date payslips are made available to the employee. Defaults to voucherDate.
  - `payslips`: array of Payslip (optional) — Link to individual payslip objects.
  - `url`: string (optional) [READ-ONLY]
  - `version`: integer (int32) (optional)
  - `year`: integer (int32) (optional)

### TimesheetEntry
**Required**: none marked
  - `activity`: object -> Activity (use {id: <int>}) (optional)
  - `changes`: array of Change (optional) [READ-ONLY]
  - `chargeable`: boolean (optional) [READ-ONLY]
  - `chargeableHours`: number (optional) [READ-ONLY]
  - `comment`: string (optional)
  - `date`: string (optional)
  - `employee`: object -> Employee (use {id: <int>}) (optional)
  - `hourlyCost`: number (optional) [READ-ONLY]
  - `hourlyCostPercentage`: number (optional) [READ-ONLY]
  - `hourlyRate`: number (optional) [READ-ONLY]
  - `hours`: number (optional)
  - `id`: integer (int64) (optional)
  - `invoice`: object -> Invoice (use {id: <int>}) (optional)
  - `locked`: boolean (optional) [READ-ONLY] — Indicates if the hour can be changed.
  - `project`: object -> Project (use {id: <int>}) (optional)
  - `projectChargeableHours`: number (optional) [min=0, max=24] — Number of chargeable hours on an activity connected to a project.
  - `timeClocks`: array of TimeClock (optional) [READ-ONLY] — Link to stop watches on this hour.
  - `url`: string (optional) [READ-ONLY]
  - `version`: integer (int32) (optional)

### TravelExpense
**Required**: none marked
  - `accommodationAllowances`: array of AccommodationAllowance (optional) [READ-ONLY] — Link to individual accommodation allowances.
  - `accountingPeriodClosed`: boolean (optional) [READ-ONLY]
  - `accountingPeriodVATClosed`: boolean (optional) [READ-ONLY]
  - `actions`: array of Link (optional) [READ-ONLY]
  - `amount`: number (optional) [READ-ONLY]
  - `approvedBy`: object -> Employee (use {id: <int>}) (optional)
  - `approvedDate`: string (optional) [READ-ONLY]
  - `attachment`: object -> Document (use {id: <int>}) (optional)
  - `attachmentCount`: integer (int32) (optional) [READ-ONLY]
  - `attestation`: object -> Attestation (use {id: <int>}) (optional)
  - `attestationSteps`: array of AttestationStep (optional)
  - `changes`: array of Change (optional) [READ-ONLY]
  - `chargeableAmount`: number (optional) [READ-ONLY]
  - `chargeableAmountCurrency`: number (optional) [READ-ONLY]
  - `completedBy`: object -> Employee (use {id: <int>}) (optional)
  - `completedDate`: string (optional) [READ-ONLY]
  - `costs`: array of Cost (optional) — Link to individual costs.
  - `date`: string (optional) [READ-ONLY]
  - `department`: object -> Department (use {id: <int>}) (optional)
  - `displayName`: string (optional) [READ-ONLY]
  - `displayNameWithoutNumber`: string (optional) [READ-ONLY]
  - `employee`: object -> Employee (use {id: <int>}) (optional)
  - `fixedInvoicedAmount`: number (optional)
  - `freeDimension1`: object -> AccountingDimensionValue (use {id: <int>}) (optional)
  - `freeDimension2`: object -> AccountingDimensionValue (use {id: <int>}) (optional)
  - `freeDimension3`: object -> AccountingDimensionValue (use {id: <int>}) (optional)
  - `highRateVAT`: number (optional) [READ-ONLY]
  - `id`: integer (int64) (optional)
  - `invoice`: object -> Invoice (use {id: <int>}) (optional)
  - `isApproved`: boolean (optional) [READ-ONLY]
  - `isChargeable`: boolean (optional)
  - `isCompleted`: boolean (optional) [READ-ONLY]
  - `isFixedInvoicedAmount`: boolean (optional)
  - `isIncludeAttachedReceiptsWhenReinvoicing`: boolean (optional)
  - `isMarkupInvoicedPercent`: boolean (optional)
  - `isSalaryAdmin`: boolean (optional) [READ-ONLY]
  - `lowRateVAT`: number (optional) [READ-ONLY]
  - `markupInvoicedPercent`: number (optional)
  - `mediumRateVAT`: number (optional) [READ-ONLY]
  - `mileageAllowances`: array of MileageAllowance (optional) [READ-ONLY] — Link to individual mileage allowances.
  - `number`: integer (int32) (optional) [READ-ONLY]
  - `numberAsString`: string (optional) [READ-ONLY]
  - `paymentAmount`: number (optional) [READ-ONLY]
  - `paymentAmountCurrency`: number (optional) [READ-ONLY]
  - `paymentCurrency`: object -> Currency (use {id: <int>}) (optional)
  - `payslip`: object -> Payslip (use {id: <int>}) (optional)
  - `perDiemCompensations`: array of PerDiemCompensation (optional) — Link to individual per diem compensations.
  - `project`: object -> Project (use {id: <int>}) (optional)
  - `rejectedBy`: object -> Employee (use {id: <int>}) (optional)
  - `rejectedComment`: string (optional) [READ-ONLY]
  - `showPayslip`: boolean (optional) [READ-ONLY]
  - `state`: string (optional) [READ-ONLY] [enum=['ALL', 'REJECTED', 'OPEN', 'APPROVED', 'SALARY_PAID', 'DELIVERED']]
  - `stateName`: string (optional) [READ-ONLY]
  - `title`: string (optional)
  - `travelAdvance`: number (optional)
  - `travelDetails`: object -> TravelDetails (use {id: <int>}) (optional)
  - `type`: integer (int32) (optional) [READ-ONLY] [min=0]
  - `url`: string (optional) [READ-ONLY]
  - `vatType`: object -> VatType (use {id: <int>}) (optional)
  - `version`: integer (int32) (optional)
  - `voucher`: object -> Voucher (use {id: <int>}) (optional)

### PerDiemCompensation
**Required**: none marked
  - `address`: string (optional)
  - `amount`: number (optional)
  - `changes`: array of Change (optional) [READ-ONLY]
  - `count`: integer (int32) (optional)
  - `countryCode`: string (optional)
  - `id`: integer (int64) (optional)
  - `isDeductionForBreakfast`: boolean (optional)
  - `isDeductionForDinner`: boolean (optional)
  - `isDeductionForLunch`: boolean (optional)
  - `location`: string (optional)
  - `overnightAccommodation`: string (optional) [enum=['NONE', 'HOTEL', 'BOARDING_HOUSE_WITHOUT_COOKING', 'BOARDING_HOUSE_WITH_COOKING']] — Set what sort of accommodation was had overnight.
  - `rate`: number (optional)
  - `rateCategory`: object -> TravelExpenseRateCategory (use {id: <int>}) (optional)
  - `rateType`: object -> TravelExpenseRate (use {id: <int>}) (optional)
  - `travelExpense`: object -> TravelExpense (use {id: <int>}) (optional)
  - `travelExpenseZoneId`: integer (int32) (optional) — Optional travel expense zone id. If not specified, the value from field zone will be used.
  - `url`: string (optional) [READ-ONLY]
  - `version`: integer (int32) (optional)

### Cost
**Required**: none marked
  - `amountCurrencyIncVat`: number (optional)
  - `amountNOKInclVAT`: number (optional)
  - `amountNOKInclVATHigh`: number (optional) [READ-ONLY]
  - `amountNOKInclVATLow`: number (optional) [READ-ONLY]
  - `amountNOKInclVATMedium`: number (optional) [READ-ONLY]
  - `category`: string (optional)
  - `changes`: array of Change (optional) [READ-ONLY]
  - `comments`: string (optional)
  - `costCategory`: object -> TravelCostCategory (use {id: <int>}) (optional)
  - `currency`: object -> Currency (use {id: <int>}) (optional)
  - `date`: string (optional)
  - `id`: integer (int64) (optional)
  - `isChargeable`: boolean (optional)
  - `isPaidByEmployee`: boolean (optional) [READ-ONLY]
  - `participants`: array of CostParticipant (optional) — Link to individual expense participant.
  - `paymentType`: object -> TravelPaymentType (use {id: <int>}) (optional)
  - `predictions`: object (optional)
  - `rate`: number (optional)
  - `travelExpense`: object -> TravelExpense (use {id: <int>}) (optional)
  - `url`: string (optional) [READ-ONLY]
  - `vatType`: object -> VatType (use {id: <int>}) (optional)
  - `version`: integer (int32) (optional)

### Contact
**Required**: none marked
  - `changes`: array of Change (optional) [READ-ONLY]
  - `customer`: object -> Customer (use {id: <int>}) (optional)
  - `department`: object -> Department (use {id: <int>}) (optional)
  - `displayName`: string (optional)
  - `email`: string (optional)
  - `firstName`: string (optional)
  - `id`: integer (int64) (optional)
  - `isInactive`: boolean (optional)
  - `lastName`: string (optional)
  - `phoneNumberMobile`: string (optional)
  - `phoneNumberMobileCountry`: object -> Country (use {id: <int>}) (optional)
  - `phoneNumberWork`: string (optional)
  - `url`: string (optional) [READ-ONLY]
  - `version`: integer (int32) (optional)

### IncomingInvoiceHeaderExternalWrite
**Required**: none marked
  - `currencyId`: integer (int64) (optional) — Currency used in the invoice.
  - `description`: string (optional) — Description of the invoice.
  - `dueDate`: string (optional) — The date the invoice is due.
  - `invoiceAmount`: number (optional) — Amount including VAT.
  - `invoiceDate`: string (optional) — The date the invoice was issued.
  - `invoiceNumber`: string (optional) — Number of the invoice
  - `purchaseOrderId`: integer (int64) (optional) — The purchase order that the invoice belongs to.
  - `vendorId`: integer (int64) (optional) — The Incoming invoice vendor
  - `voucherTypeId`: integer (int64) (optional) — Voucher type of the invoice.

### IncomingOrderLineExternalWrite
**Required**: none marked
  - `accountId`: integer (int64) (optional) — ID of the account.
  - `amountInclVat`: number (optional) — Amount including VAT. Max 2 decimals
  - `assetId`: integer (int64) (optional) — ID of the asset.
  - `budgetOrderLineId`: integer (int64) (optional) — ID of order line used as budget item
  - `count`: number (optional) — Count of the order line. Max 10 decimals
  - `customerId`: integer (int64) (optional) — ID of the customer.
  - `departmentId`: integer (int64) (optional) — ID of the department.
  - `description`: string (optional) — Description of the order line.
  - `employeeId`: integer (int64) (optional) — ID of the employee.
  - `externalId`: string (optional) — Unique Id for this invoice. Will be used in validation messages.
  - `freeDimension1Id`: integer (int64) (optional) — Free dimension 1 value id.
  - `freeDimension2Id`: integer (int64) (optional) — Free dimension 2 value id.
  - `freeDimension3Id`: integer (int64) (optional) — Free dimension 3 value id.
  - `periodizationAccountId`: integer (int64) (optional) — Periodization Account ID.
  - `periodizationPeriodsCount`: integer (int32) (optional) [min=1] — Number of Periodization Periods.
  - `periodizationStartMonth`: integer (int32) (optional) [min=1, max=12] — Periodization Start Month.
  - `periodizationStartYear`: integer (int32) (optional) [min=1990] — Periodization Start Year.
  - `productId`: integer (int64) (optional) — ID of the product.
  - `productUnitId1`: integer (int64) (optional) — ID of the product unit 1.
  - `productUnitId2`: integer (int64) (optional) — ID of the product unit 2.
  - `projectId`: integer (int64) (optional) — ID of the project.
  - `quantityAmount1`: number (optional) — Quantity amount 1.
  - `quantityAmount2`: number (optional) — Quantity amount 2.
  - `reInvoiceOnProject`: boolean (optional) — Re-invoice on project flag.
  - `row`: integer (int32) (optional) [min=1] — Row number of the order line. Starts at 1
  - `taxTransactionTypeId`: integer (int64) (optional) — ID of tax transaction.
  - `vatTypeId`: integer (int64) (optional) — ID of the VAT type.
  - `vendorId`: integer (int64) (optional) — ID of the vendor.

### CompanyBankAccountPresentation
**Required**: none marked
  - `bban`: string (optional) — Bban-number
  - `bic`: string (optional) — BIC/SWIFT for this bankaccount
  - `country`: object -> Country (use {id: <int>}) (optional)
  - `iban`: string (optional) — Iban-number
  - `provider`: string (optional) [READ-ONLY] [enum=['NETS', 'AUTOPAY']]

---
## ENDPOINTS NOT FOUND IN SPEC
  - POST /employee/entitlement

Note: `/employee/entitlement` only has GET (list) — to grant entitlements use PUT /:grantEntitlementsByTemplate