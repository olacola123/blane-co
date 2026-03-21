# Tripletex Task Tracker — Oppdateres live

## Scoreoversikt (sist oppdatert: 21. mars 22:00)

| Task | Type | Checks | Vår score | #1 Ninjas | #2 sf | #3 ACR | Status | Problem/Fix |
|------|------|--------|-----------|-----------|-------|--------|--------|-------------|
| 01 | Customer m/adresse | 7-8 | 1.50 | 1.50 | 1.50 | **2.00** | PARTIAL | Mangler felt? ACR fikset med 12 tries. Lagt til language+invoiceEmail i v14 |
| 02 | Product | 7 | 2.00 | 2.00 | 2.00 | 2.00 | OK | Perfekt. Kan forbedre efficiency for T3 |
| 03 | Invoice (send) | 7 | 2.00 | 2.00 | 2.00 | 2.00 | OK | Perfekt |
| 04 | Departments (3 stk) | 7 | 2.00 | 2.00 | 2.00 | 2.00 | OK | Perfekt |
| 05 | Employee m/fødselsdato+start | 8 | 1.33 | **2.00** | 1.33 | **2.00** | PARTIAL | Mangler employment details? Lagt til percentageOfFullTimeEquivalent i v14 |
| 06 | Supplier | 7 | 1.20 | 1.27 | 1.25 | 1.25 | **ALLE SLITER** | Ingen over 1.33! Lagt til language+invoiceEmail i v14 |
| 07 | Customer enkel | 7-8 | 2.00 | 2.00 | 2.00 | 2.00 | OK | Perfekt |
| 08 | Order m/produkter | 8 | 1.20 | 2.00 | 2.00 | 2.00 | **VI SLITER** | Alle andre har 2.00. Pris-fix i v12.8 bør ha fikset det |
| 09 | Invoice multiline (3 prod) | 8 | 2.33 | 2.53 | 2.67 | **4.00** | OK | Efficiency gap til ACR. Trenger færre kall |
| 10 | Project fixed price+delbetaling | 8 | 2.40 | 2.67 | 2.67 | **4.00** | OK | PUT /project er BETA men funker. ACR har perfekt |
| 11 | **SALARY** | 7-8 | **0** | **0** | **1.50** | 1.00 | **FEIL** | salary-modul deaktivert→403. Voucher 5000→2780 (v12.9). sf fikk 1.50! |
| 12 | **SUPPLIER INVOICE** | 8 | **0** | 1.00 | 1.00 | **4.00** | **FEIL** | /incomingInvoice er BETA→403. Voucher-tilnærming korrekt. ACR: 4.00! |
| 13 | Project enkel | 7 | 2.40 | 1.13 | 1.38 | 2.40 | OK | Vi er blant beste! |
| 14 | Accounting dimensions | 13 | **4.00** | **4.00** | **4.00** | **4.00** | PERFEKT | Alle team perfekt |
| 15 | Timesheet+prosjektfaktura | 8 | 2.44 | 2.80 | 3.00 | **3.33** | OK | Beløp-fix i v12.9 (TOTAL=hours×rate) |
| 16 | Reverse payment | 8 | 2.40 | 2.53 | 3.00 | 3.00 | OK | Funker, efficiency kan forbedres |
| 17 | Travel expense | 8 | 3.20 | **3.50** | **3.50** | **3.50** | OK | Litt bak. Per diem fix i v12 |
| 18 | Payment (registrer betaling) | 7-8 | **4.00** | **4.00** | **4.00** | **4.00** | PERFEKT | Alle team perfekt |

### T3 Tasks (nye oppgavetyper, Tier 3 ×3)

| Task | Sannsynlig type | Vår score | Ninjas | sf | påske | Status | Problem/Fix |
|------|----------------|-----------|--------|-----|-------|--------|-------------|
| 19 | Slett reiseregning | — | **0** (1t) | — | — | IKKE TRUFFET | State-maskin fix i v14: undeliver→unapprove→delete |
| 20 | ? | — | — | — | — | UKJENT | Ingen har truffet |
| 21 | Oppdater ansatt? | — | 1.93 (1t) | 2.14 (1t) | — | IKKE TRUFFET | Delvis løst av andre |
| 22 | Kontaktperson? | — | **0** (1t) | **0** (1t) | — | IKKE TRUFFET | INGEN har løst! Contact person fix i v14 |
| 23 | Standalone voucher? | — | 0.60 (1t) | — | — | IKKE TRUFFET | Voucher date-per-posting fix i v14 |
| 24 | Oppdater kunde? | — | — | — | 2.25 (1t) | IKKE TRUFFET | Delvis løst av påske |
| 25 | Credit note? (LETT!) | — | **6.00** (1t) | 4.20 (1t) | — | IKKE TRUFFET | Ninjas PERFEKT på 1. forsøk! GRATIS POENG |
| 26 | Ansatt som admin? | — | 0.60 (1t) | 2.10 (1t) | 0.60 (1t) | IKKE TRUFFET | Vanskelig, sf best |
| 27 | Product spesial-MVA? | — | 1.50 (1t) | — | — | IKKE TRUFFET | 0% avis, 15% mat? |
| 28 | Enkel avdeling? | — | 1.50 (2t) | — | — | IKKE TRUFFET | Partial |
| 29 | ? | — | — | — | — | UKJENT | Ingen har truffet |
| 30 | ? | — | — | — | — | UKJENT | Ingen har truffet |

---

## Submission Log (fyll inn etter hver submission)

| Tid (CET) | Task # | Score | Checks | Oppgavetype | Kall | Errors | Notater |
|-----------|--------|-------|--------|-------------|------|--------|---------|
| | | | | | | | |

---

## Kjente problemer og fikser per versjon

### v14 (live nå)
- Supplier/customer: `language:"NO"` + `invoiceEmail` = email
- Employee: `percentageOfFullTimeEquivalent:100.0` ALWAYS
- Delete travel: state-maskin (undeliver→unapprove→delete)
- Contact person: tydelig instruksjon med customer-ref
- Voucher: `date` på HVER posting
- BETA-endpoints varslet i prompt
- Salary: employment REQUIRED

### v13 (tidligere)
- Supplier invoice: "ALDRI bruk /incomingInvoice" + steg-for-steg voucher
- FORBIDDEN GET-liste for efficiency

### v12.9 (tidligere)
- Salary voucher: 5000→2780 (ikke 1920!)
- Timesheet beløp: TOTAL = hours × rate

### v12.8 (tidligere)
- Pris-tolkning: "28950 kr" = ekskl. MVA
- Product/customer pre-fetch
- Product duplikat-håndtering

---

## Potensiell poenggevinst

| Kilde | Poeng nå | Potensielt | Gevinst |
|-------|----------|-----------|---------|
| Task 06 (supplier language) | 1.20 | 2.00 | +0.80 |
| Task 05 (employee details) | 1.33 | 2.00 | +0.67 |
| Task 08 (order) | 1.20 | 2.00 | +0.80 |
| Task 11 (salary voucher) | 0 | 1.50 | +1.50 |
| Task 12 (supplier inv) | 0 | 4.00 | +4.00 |
| Task 25 (T3 lett) | 0 | 6.00 | +6.00 |
| Task 21 (T3 update) | 0 | 2.00 | +2.00 |
| Task 19 (T3 delete travel) | 0 | 3.00 | +3.00 |
| Andre T3 tasks | 0 | ~10 | +10 |
| Efficiency forbedring T3 | — | — | +5-10 |
| **TOTALT** | **36.41** | **~65-75** | **+30-40** |

---

## Hva scoring sjekker (bekreftet/gjetning)

### Customer (Task 01, 07)
- name, organizationNumber, email, phoneNumber, isCustomer
- postalAddress: addressLine1, postalCode, city (kun Task 01)
- **language?** (ukjent — testet i v14)
- **invoiceEmail?** (ukjent — testet i v14)

### Product (Task 02)
- name, number, priceExcludingVatCurrency, vatType (id:3 for 25%)
- priceIncludingVatCurrency

### Supplier (Task 06)
- name, organizationNumber, email, phoneNumber, isSupplier, isCustomer:false
- **7. sjekk ukjent!** Trolig: language, invoiceEmail, eller postalAddress

### Employee (Task 05)
- firstName, lastName, email, dateOfBirth
- employment: startDate
- employment details: employmentType, employmentForm, percentageOfFullTimeEquivalent?
- **3 av 8 sjekker feiler** — trolig employment details

### Salary (Task 11)
- Scoring sjekker trolig voucher-posteringer (salary-modul deaktivert)
- Konto 5000 (debit) + 2780 (kredit)
- Grunnlønn og bonus som separate posteringer

### Supplier Invoice (Task 12)
- Supplier opprettet med org.nr
- Voucher med: expense-konto (debit), 2710/inngående MVA (debit), 2400/leverandørgjeld (kredit+supplier ref)
- Beløp: excl = incl/1.25, MVA = incl - excl
