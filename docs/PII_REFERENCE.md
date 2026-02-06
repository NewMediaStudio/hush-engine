# PII Classification Reference

This document provides a comprehensive reference of Personally Identifiable Information (PII) types, their risk classifications, detection patterns, and regulatory context. Use this reference for testing the Hush Engine's detection capabilities.

## Regulatory Sources

This classification draws from authoritative sources:

| Source | Description |
|--------|-------------|
| **[NIST SP 800-122](https://csrc.nist.gov/pubs/sp/800/122/final)** | Guide to Protecting the Confidentiality of PII |
| **[GDPR](https://gdpr-info.eu/art-9-gdpr/)** | EU General Data Protection Regulation (Art. 9 Special Categories) |
| **[HIPAA](https://cphs.berkeley.edu/hipaa/hipaa18.html)** | 18 PHI Identifiers under the Privacy Rule |
| **[CCPA/CPRA](https://oag.ca.gov/privacy/ccpa)** | California Consumer Privacy Act & Privacy Rights Act |
| **[FIPS 199](https://csrc.nist.gov/publications/detail/fips/199/final)** | Security Categorization of Federal Information |

## Risk Level Definitions

| Level | NIST Impact | Description | Examples |
|-------|-------------|-------------|----------|
| **Critical** | High | Immediate identity theft, financial fraud, or physical harm risk. Regulatory violations carry severe penalties. | SSN, Biometrics, Passport, Financial credentials |
| **High** | High | Significant harm potential. Can enable identity theft when combined with other data. | Medical records, Government IDs, Credit cards |
| **Medium** | Moderate | Moderate harm if disclosed. Often requires combination with other data for significant impact. | Full name, DOB, Address, Phone, Email |
| **Low** | Low | Minimal individual harm. May contribute to profiling when aggregated. | Age range, Gender, Zip code, Job title |

---

## Complete PII Reference Table

### Critical Risk (NIST High Impact)

| Identifier | Standard Format / Regex Pattern | Key Regulations | Hush Engine Support |
|------------|--------------------------------|-----------------|---------------------|
| **Social Security Number (SSN)** | `\b\d{3}-\d{2}-\d{4}\b` or `\b\d{9}\b` | HIPAA, CCPA, NIST | ✅ `NATIONAL_ID` |
| **Social Insurance Number (SIN)** | `\b\d{3}[-\s]?\d{3}[-\s]?\d{3}\b` | PIPEDA (Canada) | ✅ `NATIONAL_ID` |
| **National ID Number** | Varies by country | GDPR, Local laws | ✅ `NATIONAL_ID` (35+ countries via python-stdnum) |
| **Passport Number** | `\b[A-Z]{1,2}\d{6,9}\b` | HIPAA, GDPR, CCPA | ✅ `NATIONAL_ID` |
| **Driver's License Number** | Varies by state/province | HIPAA, CCPA | ✅ `NATIONAL_ID` |
| **Biometric Data (Fingerprints)** | Binary/encoded data, `BIO-\d{8,12}` | GDPR Art.9, CCPA, BIPA | ✅ `BIOMETRIC` |
| **Biometric Data (Voice Print)** | Binary/encoded data | GDPR Art.9, CCPA, BIPA | ✅ `BIOMETRIC` |
| **Biometric Data (Facial Recognition)** | Binary/encoded data | GDPR Art.9, CCPA, BIPA | ✅ `BIOMETRIC` |
| **Genetic Data** | Various sequence formats | GDPR Art.9, GINA, HIPAA | ❌ Planned |
| **Bank Account + Routing Number** | `\b\d{9,17}\b` with context | CCPA, PCI-DSS | ⚠️ Partial |
| **Credit Card + CVV** | `\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b` | PCI-DSS, CCPA | ✅ `CREDIT_CARD` (Luhn validated) |
| **Login Credentials (Username + Password)** | Context-dependent | CCPA, GDPR | ✅ `CREDENTIAL` |
| **AWS Access Key** | `(AKIA\|ASIA)[A-Z0-9]{16}` | Best Practice | ✅ `CREDENTIAL` |
| **Stripe Secret Key** | `sk_live_[0-9a-zA-Z]{24,}` | Best Practice | ✅ `CREDENTIAL` |
| **API Keys / Tokens** | Various patterns | Best Practice | ✅ `CREDENTIAL` |

### High Risk (NIST High/Moderate Impact)

| Identifier | Standard Format / Regex Pattern | Key Regulations | Hush Engine Support |
|------------|--------------------------------|-----------------|---------------------|
| **Medical Record Number (MRN)** | Varies by institution | HIPAA | ⚠️ Context-based |
| **Health Plan ID** | Varies by plan | HIPAA | ⚠️ Context-based |
| **Blood Type** | `\b(AB?\|B?\|O)[+-]\b` | HIPAA, GDPR Art.9 | ✅ `MEDICAL` |
| **Medical Diagnosis / Condition** | Named conditions + ICD codes | HIPAA, GDPR Art.9 | ✅ `MEDICAL` |
| **Medication / Prescription** | Drug names, dosages | HIPAA, GDPR Art.9 | ✅ `MEDICAL` |
| **Lab Results / Vital Signs** | `BP:\s*\d{2,3}/\d{2,3}` etc. | HIPAA | ✅ `MEDICAL` |
| **ICD-10 Code** | `\b[A-TV-Z]\d{2}(\.\d{1,4})?\b` | HIPAA | ✅ `MEDICAL` |
| **Mental Health Information** | Named conditions | HIPAA, GDPR Art.9 | ✅ `MEDICAL` |
| **Substance Use Information** | Named substances, treatment | HIPAA (42 CFR Part 2) | ✅ `MEDICAL` |
| **Sexual Orientation** | Various terms | GDPR Art.9, CCPA | ⚠️ Partial |
| **HIV/AIDS Status** | Contextual | HIPAA, GDPR Art.9 | ✅ `MEDICAL` |
| **Racial/Ethnic Origin** | Named categories | GDPR Art.9, CCPA | ❌ Planned |
| **Religious Beliefs** | Named religions/beliefs | GDPR Art.9 | ❌ Planned |
| **Political Opinions** | Party affiliation, views | GDPR Art.9 | ❌ Planned |
| **Trade Union Membership** | Union names | GDPR Art.9 | ❌ Planned |
| **Criminal History** | Contextual | GDPR Art.10, FCRA | ❌ Planned |
| **IBAN** | `\b[A-Z]{2}\d{2}[A-Z0-9]{4,}\b` | GDPR, PSD2 | ✅ `IBAN_CODE` (116 countries, Mod-97 validated) |
| **SWIFT/BIC Code** | `\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?\b` | GDPR, PSD2 | ✅ `FINANCIAL` |
| **Full Face Photo** | Image detection | HIPAA, GDPR | ✅ `FACE` (OpenCV Haar cascade) |
| **Vehicle ID (VIN)** | `\b[A-HJ-NPR-Z0-9]{17}\b` | HIPAA, CCPA | ✅ `VEHICLE` |
| **License Plate** | Varies by jurisdiction | HIPAA, CCPA | ✅ `VEHICLE` |
| **Device Identifier (IMEI, MAC)** | `\b\d{15}\b` / `([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}` | HIPAA, CCPA | ✅ `NETWORK` |
| **HTTP Cookie / Session ID** | Various formats | GDPR, CCPA | ✅ `NETWORK` |
| **Immigration/Citizenship Status** | Contextual | CCPA (as of 2024) | ❌ Planned |

### Medium Risk (NIST Moderate Impact)

| Identifier | Standard Format / Regex Pattern | Key Regulations | Hush Engine Support |
|------------|--------------------------------|-----------------|---------------------|
| **Full Name** | `\b[A-Z][a-z]+\s+[A-Z][a-z]+\b` | HIPAA, GDPR, CCPA | ✅ `PERSON` (NLP-based) |
| **Email Address** | `\b[\w.-]+@[\w.-]+\.\w+\b` | GDPR, CCPA | ✅ `EMAIL_ADDRESS` |
| **Phone Number** | `\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b` | HIPAA, GDPR, CCPA | ✅ `PHONE_NUMBER` (150+ countries via phonenumbers) |
| **Physical Address** | Street + City + State/Province + Postal | HIPAA, GDPR, CCPA | ✅ `LOCATION` |
| **Street Address** | `\b\d+\s+[A-Z][a-z]+\s+(St\|Ave\|Rd\|Blvd)\.?\b` | HIPAA, GDPR | ✅ `LOCATION` |
| **City, State/Province** | `\b[A-Z][a-z]+,?\s+[A-Z]{2}\b` | HIPAA, GDPR | ✅ `LOCATION` |
| **Postal/ZIP Code** | `\b\d{5}(-\d{4})?\b` or `\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b` | HIPAA, GDPR | ✅ `LOCATION` |
| **Date of Birth** | `\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b` | HIPAA, GDPR, CCPA | ✅ `DATE_TIME` |
| **Age (exact)** | `\b\d{1,3}\s*(years?\|yrs?)\s*old\b` or `Age:\s*\d+` | HIPAA | ✅ `AGE` |
| **IP Address** | `\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b` | GDPR, CCPA | ✅ `IP_ADDRESS` |
| **GPS Coordinates** | `\b-?\d{1,3}\.\d{3,8},\s*-?\d{1,3}\.\d{3,8}\b` | CCPA (precise geolocation) | ✅ `COORDINATES` |
| **Latitude/Longitude** | DMS format: `\d+°\d+'\d+"[NS]` | CCPA | ✅ `COORDINATES` |
| **Account Number** | Context-dependent numeric | HIPAA, CCPA | ⚠️ Context-based |
| **Certificate/License Number** | Alphanumeric patterns | HIPAA | ⚠️ Context-based |
| **Gender Identity** | Various terms (male, female, non-binary, etc.) | GDPR Art.9 | ✅ `GENDER` |
| **Fax Number** | Same as phone | HIPAA | ✅ `PHONE_NUMBER` |
| **URL (Personal)** | `https?://[\w.-]+/[\w/-]*` | HIPAA | ✅ `URL` |
| **Company Name** | `\b[A-Z][\w&'.-]+\s+(Ltd\|Inc\|LLC)\b` | Various | ✅ `COMPANY` |
| **Employer Information** | Context-dependent | CCPA | ⚠️ Context-based |
| **Education Records** | Context-dependent | FERPA | ❌ Planned |
| **UK NHS Number** | `\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b` | UK Data Protection Act | ✅ `UK_NHS` |
| **Currency Amount** | `[\$£€¥]\s?\d{1,3}(,\d{3})*(\.\d{2})?` | Context-dependent | ✅ `FINANCIAL` |

### Low Risk (NIST Low Impact)

| Identifier | Standard Format / Regex Pattern | Key Regulations | Hush Engine Support |
|------------|--------------------------------|-----------------|---------------------|
| **Age Range** | `\b(18-24\|25-34\|35-44\|etc.)\b` | Minimal | ❌ Not targeted |
| **ZIP Code (3-digit)** | `\b\d{3}\b` | HIPAA Safe Harbor | ❌ Not targeted |
| **Year of Birth** | `\b(19\|20)\d{2}\b` | Minimal (unless >89) | ⚠️ Context-based |
| **Public Job Title** | Free text | Minimal | ❌ Not targeted |
| **Public Business Address** | Standard address format | Minimal | ✅ `LOCATION` |
| **Cookie ID** | Various formats | GDPR, CCPA | ✅ `NETWORK` |
| **Advertising ID** | UUID format | CCPA | ✅ `NETWORK` |

---

## HIPAA 18 PHI Identifiers Mapping

Per [HIPAA Privacy Rule](https://cphs.berkeley.edu/hipaa/hipaa18.html), these identifiers must be removed for de-identification:

| # | HIPAA Identifier | Hush Engine Entity | Status | Notes |
|---|------------------|-------------------|--------|-------|
| 1 | Names | `PERSON` | ✅ Full | Multi-NER cascade with ensemble scoring |
| 2 | Geographic data smaller than state | `LOCATION` | ✅ Full | libpostal + cities/countries databases |
| 3 | Dates (except year) + Ages >89 | `DATE_TIME`, `AGE` | ✅ Full | dateparser + regex patterns |
| 4 | Phone numbers | `PHONE_NUMBER` | ✅ Full | libphonenumber (150+ countries) |
| 5 | Fax numbers | `PHONE_NUMBER` | ✅ Full | Same as phone detection |
| 6 | Email addresses | `EMAIL_ADDRESS` | ✅ Full | RFC-compliant regex |
| 7 | Social Security numbers | `NATIONAL_ID` | ✅ Full | SSN + 35 international ID formats |
| 8 | Medical record numbers | `ID`, `MEDICAL` | ⚠️ Partial | Context-dependent detection |
| 9 | Health plan beneficiary numbers | `ID`, `MEDICAL` | ⚠️ Partial | Context-dependent detection |
| 10 | Account numbers | `ID`, `FINANCIAL` | ⚠️ Partial | Labeled patterns + context |
| 11 | Certificate/license numbers | `NATIONAL_ID`, `ID` | ✅ Full | Driver's license, passport patterns |
| 12 | Vehicle identifiers | `VEHICLE` | ✅ Full | VIN validation, license plates |
| 13 | Device identifiers | `NETWORK` | ✅ Full | MAC, IMEI, UUID patterns |
| 14 | Web URLs | `URL` | ✅ Full | urlextract library |
| 15 | IP addresses | `IP_ADDRESS` | ✅ Full | IPv4 + IPv6, version string filtering |
| 16 | Biometric identifiers | `BIOMETRIC` | ✅ Full | BIO- prefixed IDs, labeled patterns |
| 17 | Full-face photographs | `FACE` | ✅ Full | OpenCV Haar cascade (images only) |
| 18 | Any unique identifying characteristic | `ID` | ⚠️ Partial | Generic ID patterns (CUST-, EMP-, etc.) |

---

## GDPR Special Category Data (Article 9)

| Category | Hush Engine Entity | Status | Notes |
|----------|-------------------|--------|-------|
| Racial or ethnic origin | — | ❌ Planned | Requires sensitive context analysis |
| Political opinions | — | ❌ Planned | Requires sensitive context analysis |
| Religious or philosophical beliefs | — | ❌ Planned | Requires sensitive context analysis |
| Trade union membership | — | ❌ Planned | Organization name matching |
| Genetic data | — | ❌ Planned | DNA sequence formats |
| Biometric data | `BIOMETRIC` | ✅ Full | BIO- IDs, fingerprint/facial recognition labels |
| Health data | `MEDICAL` | ✅ Full | ICD-10, conditions, medications, blood types |
| Sex life or sexual orientation | `GENDER` | ⚠️ Partial | Gender identity terms only |

### Additional GDPR-Relevant Entities (v1.4.0)

| Category | Hush Engine Entity | Status | Notes |
|----------|-------------------|--------|-------|
| Online identifiers | `IP_ADDRESS`, `NETWORK` | ✅ Full | IP, MAC, cookies, device IDs |
| Location data | `LOCATION`, `COORDINATES` | ✅ Full | Addresses, GPS coordinates |
| Economic data | `FINANCIAL`, `CREDIT_CARD` | ✅ Full | IBAN, SWIFT, card numbers |
| Government identifiers | `NATIONAL_ID` | ✅ Full | SSN, passport, driver's license (35+ countries) |
| Authentication data | `CREDENTIAL` | ✅ Full | Passwords, API keys, tokens |

---

## Testing Guidelines

### Sample Test Values

| Entity Type | Test Value | Expected Detection |
|-------------|------------|-------------------|
| NATIONAL_ID | `123-45-6789` (SSN) | ✅ |
| EMAIL_ADDRESS | `john.doe@example.com` | ✅ |
| PHONE_NUMBER | `(416) 555-0123` | ✅ |
| PHONE_NUMBER | `+1-555-123-4567` | ✅ |
| CREDIT_CARD | `4111 1111 1111 1111` | ✅ |
| LOCATION | `123 Main St, Toronto, ON M5V 2H1` | ✅ |
| LOCATION | `Portland, OR 97201` | ✅ |
| COORDINATES | `40.7128, -74.0060` | ✅ |
| COORDINATES | `40°42'46"N 74°0'22"W` | ✅ |
| AGE | `25 years old` | ✅ |
| AGE | `Age: 45` | ✅ |
| AGE | `aged 30` | ✅ |
| MEDICAL | `Blood Type: A+` | ✅ |
| MEDICAL | `Diagnosed with Type 2 Diabetes` | ✅ |
| MEDICAL | `Prescribed Metformin 500mg` | ✅ |
| MEDICAL | `ICD-10: E11.9` | ✅ |
| GENDER | `Gender: Non-binary` | ✅ |
| FINANCIAL | `SWIFT: DEUTDEFF` | ✅ |
| FINANCIAL | `$1,234.56` | ✅ |
| CREDENTIAL | `AKIAIOSFODNN7EXAMPLE` (AWS Key) | ✅ |
| IBAN_CODE | `DE89370400440532013000` | ✅ |
| IP_ADDRESS | `192.168.1.1` | ✅ |
| UK_NHS | `123 456 7890` | ✅ (context-dependent) |
| NATIONAL_ID | `AB123456` (Canada Passport) | ✅ |
| NATIONAL_ID | `123456789` (US Passport) | ✅ (context-dependent) |
| NATIONAL_ID | `CFABC1234` (Germany Passport) | ✅ |
| NATIONAL_ID | `D1234567` (California DL) | ✅ (context-dependent) |
| NATIONAL_ID | `A1234-12345-12345` (Ontario DL) | ✅ |
| NATIONAL_ID | `SMITH906152AB1AB` (UK DL) | ✅ |
| VEHICLE | `1HGBH41JXMN109186` (VIN) | ✅ |
| VEHICLE | `JH4KA8260MC000000` (VIN) | ✅ |
| NETWORK | `00:1A:2B:3C:4D:5E` (MAC) | ✅ |
| NETWORK | `00-1A-2B-3C-4D-5E` (MAC) | ✅ |
| NETWORK | `353456789012345` (IMEI) | ✅ |
| NETWORK | `550e8400-e29b-41d4-a716-446655440000` (UUID) | ✅ |
| CREDENTIAL | `sk_live_abc123...` (Stripe Key) | ✅ |
| CREDENTIAL | `AKIAIOSFODNN7EXAMPLE` (AWS Key) | ✅ |
| BIOMETRIC | `BIO-7459126830` | ✅ |
| ID | `CUST-12345` (Customer ID) | ✅ |

### False Positive Watch List

These patterns may cause false positives and should be validated:

| Pattern | Risk | Mitigation |
|---------|------|------------|
| 10-digit numbers | NHS vs Phone | Context analysis, area code validation |
| 5-digit numbers | ZIP vs random | Context required |
| Dates | DOB vs document dates | Context required |
| Common drug names | Medication vs casual mention | Medical context required |
| State abbreviations | Location vs text | Context required |
| 9-digit numbers | SSN vs Passport vs random | Context required |
| 7-8 digit alphanumeric | Driver's license vs random ID | Context required |
| 17-character alphanumeric | VIN vs serial numbers | Check for I, O, Q exclusion |
| 15-digit numbers | IMEI vs other numeric IDs | Context required |
| 12-character hex | MAC address vs hash fragments | Format (colons/dashes) helps |

---

## References

- [NIST SP 800-122: Guide to Protecting PII](https://csrc.nist.gov/pubs/sp/800/122/final)
- [GDPR Article 9: Special Categories](https://gdpr-info.eu/art-9-gdpr/)
- [HIPAA PHI Identifiers](https://cphs.berkeley.edu/hipaa/hipaa18.html)
- [CCPA Official Text](https://oag.ca.gov/privacy/ccpa)
- [ICO: What is Special Category Data?](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/lawful-basis/special-category-data/)
- [UC Berkeley HIPAA Privacy](https://cphs.berkeley.edu/hipaa/hipaa18.html)
- [California Privacy Protection Agency](https://cppa.ca.gov/)

---

*Last updated: 2026-02-06*
*Document version: 1.6 - Aligns with hush-engine v1.4.0. New entity types: BIOMETRIC, CREDENTIAL, ID, NATIONAL_ID (consolidates SSN), NETWORK, VEHICLE. LightGBM NER for faster inference.*
