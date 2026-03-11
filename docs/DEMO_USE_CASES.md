# AI-Force Image Guardrails - Demo Use Cases

## Overview

Enterprise use cases demonstrating Image Guardrails on the AI-Force Platform.

---

## Use Case 1: Banking - Mobile Check Deposit

### Client Profile
- **Industry:** Banking / FinTech
- **Challenge:** Validate check images for mobile deposit while detecting fraud and PII

### Business Scenario
A bank's mobile app allows customers to deposit checks by taking photos. Images must be validated for quality and sensitive data must be handled securely.

### Flow

```mermaid
sequenceDiagram
    participant Customer
    participant App as Mobile Banking App
    participant Platform as AI-Force Platform
    participant Guard as Image Guardrails
    participant OCR as Tesseract OCR
    participant PII as /scan/prompt API

    Customer->>App: Take photo of check (front & back)
    App->>Platform: Validate check image
    Platform->>Guard: Run guardrails

    Guard->>Guard: 1. File Validation (quality, resolution)
    Guard->>OCR: Extract check details
    OCR->>PII: "Pay to: John Smith, Account: 123456789, Amount: $1,500"
    PII->>Guard: PII detected: Name, Account Number, Routing Number

    Guard->>Platform: ALLOW with secure PII handling
    Platform->>App: Check validated - Deposit processing
    App->>Customer: Deposit of $1,500 pending
```

### Guardrails Applied

| Check | Detects | Compliance |
|-------|---------|------------|
| File Validation | Image quality, blur detection | Deposit accuracy |
| PII Detection | Account numbers, routing numbers | PCI-DSS |
| OCR Extraction | Check amount, payee, date | Fraud prevention |

### Demo Scenario
1. Upload clear check image → PII extracted securely → **ALLOW**
2. Upload blurry check → **REJECT** - "Image quality insufficient"
3. Upload check with mismatched amounts → **FLAG** for review

### Business Value
- Secure mobile deposit processing
- PCI-DSS compliance automated
- Fraud detection via validation

---

## Use Case 2: Real Estate Platform - Property Listing Validation

### Client Profile
- **Industry:** Real Estate / PropTech
- **Challenge:** Ensure property listing images are professional and compliant

### Business Scenario
A real estate marketplace validates property images uploaded by agents before listing goes live.

### Flow

```mermaid
sequenceDiagram
    participant Agent as Real Estate Agent
    participant Portal as Listing Portal
    participant Platform as AI-Force Platform
    participant Guard as Image Guardrails

    Agent->>Portal: Upload property photos
    Portal->>Platform: Validate images
    Platform->>Guard: Run guardrails

    Guard->>Guard: 1. File Validation (quality check)
    Guard->>Guard: 2. Face Detection (privacy)
    Guard->>Guard: 3. PII Detection (address signs, documents)

    alt Clean Images
        Guard->>Portal: ALLOW
        Portal->>Agent: Listing published
    else Issues Found
        Guard->>Portal: REJECT - "2 faces detected"
        Portal->>Agent: Please remove people from photos
    end
```

### Guardrails Applied

| Check | Detects | Reason |
|-------|---------|--------|
| Faces | People in photos | Privacy - tenants/owners visible |
| PII (OCR) | Address, documents | Sensitive info visible |
| File Validation | Low quality images | Professional standards |

### Demo Scenario
1. Upload empty room photo → **ALLOW**
2. Upload room with family visible → **REJECT** - "3 faces detected"
3. Upload image showing mail with address → **REJECT** - "PII detected: ADDRESS"

### Business Value
- Protect tenant/owner privacy
- Ensure professional listing quality
- Avoid legal issues with exposed PII

---

## Use Case 3: HR Platform - Resume & Document Processing

### Client Profile
- **Industry:** HR Tech / Recruitment
- **Challenge:** Process candidate documents while ensuring compliance

### Business Scenario
An HR platform processes resumes, ID documents, and certificates uploaded by job applicants.

### Flow

```mermaid
sequenceDiagram
    participant Candidate
    participant Portal as HR Portal
    participant Platform as AI-Force Platform
    participant Guard as Image Guardrails
    participant OCR as Tesseract OCR
    participant PII as /scan/prompt API

    Candidate->>Portal: Upload ID + Resume + Certificates
    Portal->>Platform: Process documents
    Platform->>Guard: Validate images

    Guard->>Guard: 1. File Validation
    Guard->>OCR: Extract text
    OCR->>PII: Send extracted text
    PII->>Guard: PII detected: SSN, DOB, Address

    Guard->>Platform: ALLOW with PII metadata
    Platform->>Portal: Documents processed (PII logged securely)
```

### Guardrails Applied

| Check | Detects | Compliance |
|-------|---------|------------|
| PII Detection | SSN, DOB, Address, Phone | GDPR, CCPA |
| Face Detection | Photo on ID/Resume | Bias prevention |
| File Validation | Valid document formats | - |

### Demo Scenario
1. Upload resume PDF as image → OCR extracts text → PII detected → **ALLOW** (logged)
2. Upload ID card → Face detected, PII extracted → **ALLOW** (secure handling)

### Business Value
- GDPR/CCPA compliance
- Reduce bias in hiring (face detection awareness)
- Secure PII handling with audit trail

---

## Use Case 4: E-commerce - Product Review Images

### Client Profile
- **Industry:** E-commerce / Marketplace
- **Challenge:** Moderate user-submitted product review images

### Business Scenario
An e-commerce platform allows customers to upload images with their product reviews. These need moderation.

### Flow

```mermaid
sequenceDiagram
    participant Customer
    participant Shop as E-commerce Platform
    participant Platform as AI-Force Platform
    participant Guard as Image Guardrails

    Customer->>Shop: Submit review with image
    Shop->>Platform: Validate review image
    Platform->>Guard: Run guardrails

    Guard->>Guard: 1. NSFW Check
    Guard->>Guard: 2. Violence Check
    Guard->>Guard: 3. PII Detection (receipts, cards)

    alt Safe Image
        Guard->>Shop: ALLOW
        Shop->>Customer: Review published
    else Unsafe Image
        Guard->>Shop: REJECT - "Credit card visible"
        Shop->>Customer: Please remove sensitive info
    end
```

### Guardrails Applied

| Check | Detects | Reason |
|-------|---------|--------|
| NSFW | Inappropriate images | Community standards |
| PII (OCR) | Credit cards, receipts | Customer data protection |
| Violence | Damaged/unsafe products | Platform safety |

### Demo Scenario
1. Upload product photo → **ALLOW**
2. Upload receipt with credit card visible → **REJECT** - "Credit card detected"
3. Upload inappropriate image → **REJECT** - "NSFW score: 0.91"

### Business Value
- Protect customer PII
- Maintain platform reputation
- Automated moderation at scale

---

## Use Case 5: Insurance - Claims Processing

### Client Profile
- **Industry:** Insurance
- **Challenge:** Process claim images (accidents, damage) while detecting fraud

### Business Scenario
An insurance company processes claim images submitted by policyholders.

### Flow

```mermaid
sequenceDiagram
    participant Claimant
    participant Portal as Claims Portal
    participant Platform as AI-Force Platform
    participant Guard as Image Guardrails
    participant OCR as Tesseract OCR
    participant LLM as Damage Assessment LLM

    Claimant->>Portal: Upload accident/damage photos
    Portal->>Platform: Process claim images
    Platform->>Guard: Validate images

    Guard->>Guard: 1. File Validation (metadata check)
    Guard->>Guard: 2. Violence/Gore (accident images - allowed)
    Guard->>OCR: Extract visible text (license plates, documents)

    Guard->>Platform: ALLOW with extracted metadata
    Platform->>LLM: Assess damage
    LLM->>Portal: Damage estimate: $2,500
```

### Guardrails Applied

| Check | Detects | Action |
|-------|---------|--------|
| File Validation | Image metadata, timestamps | Fraud detection |
| PII (OCR) | License plates, policy numbers | Secure logging |
| Violence | Accident imagery | ALLOW (expected for claims) |

### Demo Scenario
1. Upload car damage photo → **ALLOW** → Damage assessed
2. Upload photo with license plate → PII detected, logged securely → **ALLOW**
3. Upload manipulated image → Metadata mismatch → **FLAG for review**

### Business Value
- Faster claims processing
- Fraud detection via metadata
- Secure PII handling

---

## Summary

| Use Case | Industry | Key Guardrails | Primary Value |
|----------|----------|----------------|---------------|
| Mobile Check Deposit | Banking | PII, File Validation | PCI-DSS Compliance |
| Property Listings | Real Estate | Faces, PII | Privacy protection |
| Document Processing | HR/Recruitment | PII, Faces | Compliance (GDPR) |
| Review Images | E-commerce | NSFW, PII | Customer protection |
| Claims Processing | Insurance | File Validation, PII | Fraud detection |

### Platform Capabilities Demonstrated

| Capability | Description |
|------------|-------------|
| **Early Exit** | Stops on first failed check - efficient processing |
| **Explainability** | Clear rejection reasons for users |
| **PII via API** | External /scan/prompt for sensitive data |
| **Configurable** | Thresholds adjustable per use case |
| **Compliance Ready** | GDPR, HIPAA, PCI-DSS support |
