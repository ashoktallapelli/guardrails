# AI-Force Image Guardrails - Demo Use Cases

## Overview

Enterprise use cases demonstrating Image Guardrails on the AI-Force Platform.

---

## Use Case 1: Education Platform - Student Assignment Submission

### Client Profile
- **Industry:** EdTech / Online Learning
- **Challenge:** Validate student-uploaded assignment images for inappropriate content

### Business Scenario
An online education platform allows students to upload handwritten assignments, diagrams, or project photos. Content must be validated before submission to instructors.

### Flow

```mermaid
sequenceDiagram
    participant Student
    participant Portal as Education Portal
    participant Platform as AI-Force Platform
    participant Guard as Image Guardrails

    Student->>Portal: Upload assignment image
    Portal->>Platform: Validate image
    Platform->>Guard: Run guardrails

    Guard->>Guard: 1. File Validation
    Guard->>Guard: 2. NSFW Detection
    Guard->>Guard: 3. Violence Detection
    Guard->>Guard: 4. Hate Symbols Check

    alt Safe Content
        Guard->>Portal: ALLOW
        Portal->>Student: Assignment submitted
    else Unsafe Content
        Guard->>Portal: REJECT - "Inappropriate content detected"
        Portal->>Student: Submission rejected - Policy violation
    end
```

### Guardrails Applied

| Check | Detects | Action |
|-------|---------|--------|
| File Validation | Valid image format | REJECT if invalid |
| NSFW | Inappropriate images | REJECT |
| Violence | Violent imagery | REJECT |
| Hate Symbols | Offensive symbols | REJECT |

### Demo Scenario
1. Upload math assignment photo → **ALLOW**
2. Upload image with inappropriate drawing → **REJECT** - "NSFW score: 0.85"
3. Upload image with hate symbol → **REJECT** - "Hate symbols detected"

### Business Value
- Safe learning environment
- Protect instructors from inappropriate content
- Automated policy enforcement at scale

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

## Use Case 3: Custom Merchandise Platform - Design Uploads

### Client Profile
- **Industry:** Print-on-Demand / Custom Merchandise
- **Challenge:** Validate user-uploaded designs for t-shirts, mugs, posters before printing

### Business Scenario
A custom merchandise platform allows users to upload designs for printing on products. Designs must be screened for inappropriate content before production.

### Flow

```mermaid
sequenceDiagram
    participant User
    participant Shop as Merchandise Platform
    participant Platform as AI-Force Platform
    participant Guard as Image Guardrails

    User->>Shop: Upload custom t-shirt design
    Shop->>Platform: Validate design image
    Platform->>Guard: Run guardrails

    Guard->>Guard: 1. File Validation (resolution, format)
    Guard->>Guard: 2. NSFW Detection
    Guard->>Guard: 3. Violence Detection
    Guard->>Guard: 4. Hate Symbols Check

    alt Safe Design
        Guard->>Shop: ALLOW
        Shop->>User: Design approved - Ready to print
    else Unsafe Design
        Guard->>Shop: REJECT - "Hate symbols detected"
        Shop->>User: Design rejected - Policy violation
    end
```

### Guardrails Applied

| Check | Detects | Action |
|-------|---------|--------|
| NSFW | Adult/explicit imagery | REJECT |
| Violence | Weapons, gore | REJECT |
| Hate Symbols | Nazi, extremist symbols | REJECT |
| File Validation | Print quality requirements | REJECT if low quality |

### Demo Scenario
1. Upload artistic design → **ALLOW** → Proceed to print
2. Upload design with swastika → **REJECT** - "Hate symbols detected"
3. Upload design with weapon imagery → **REJECT** - "Violence: guns=0.91"

### Business Value
- Prevent printing offensive merchandise
- Protect brand reputation
- Avoid legal liability
- Automated screening at scale

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

## Use Case 5: Gaming Platform - User Avatar Uploads

### Client Profile
- **Industry:** Gaming / Entertainment
- **Challenge:** Moderate user-uploaded profile avatars and in-game images

### Business Scenario
A gaming platform allows users to upload custom profile avatars and share screenshots. All uploads must be screened for inappropriate content to maintain community standards.

### Flow

```mermaid
sequenceDiagram
    participant Gamer
    participant Portal as Gaming Platform
    participant Platform as AI-Force Platform
    participant Guard as Image Guardrails

    Gamer->>Portal: Upload profile avatar
    Portal->>Platform: Validate avatar image
    Platform->>Guard: Run guardrails

    Guard->>Guard: 1. File Validation
    Guard->>Guard: 2. NSFW Detection
    Guard->>Guard: 3. Violence Detection
    Guard->>Guard: 4. Hate Symbols Check

    alt Safe Avatar
        Guard->>Portal: ALLOW
        Portal->>Gamer: Avatar updated successfully
    else Unsafe Avatar
        Guard->>Portal: REJECT - "Inappropriate content"
        Portal->>Gamer: Avatar rejected - Community guidelines violation
    end
```

### Guardrails Applied

| Check | Detects | Action |
|-------|---------|--------|
| NSFW | Explicit/adult imagery | REJECT |
| Violence | Gore, weapons, blood | REJECT |
| Hate Symbols | Extremist symbols | REJECT |
| File Validation | Valid image format | REJECT if invalid |

### Demo Scenario
1. Upload gaming character avatar → **ALLOW**
2. Upload avatar with explicit content → **REJECT** - "NSFW score: 0.92"
3. Upload avatar with hate symbol → **REJECT** - "Hate symbols detected"

### Business Value
- Safe gaming environment for all ages
- Protect community from toxic content
- Automated moderation at scale
- Maintain platform reputation

---

## Summary

| Use Case | Industry | Key Guardrails | Primary Value |
|----------|----------|----------------|---------------|
| Assignment Submission | Education | NSFW, Violence, Hate | Safe learning environment |
| Property Listings | Real Estate | Faces, PII | Privacy protection |
| Design Uploads | Custom Merchandise | NSFW, Violence, Hate | Brand protection |
| Review Images | E-commerce | NSFW, PII | Customer protection |
| Avatar Uploads | Gaming | NSFW, Violence, Hate | Community safety |

### Platform Capabilities Demonstrated

| Capability | Description |
|------------|-------------|
| **Early Exit** | Stops on first failed check - efficient processing |
| **Explainability** | Clear rejection reasons for users |
| **PII via API** | External /scan/prompt for sensitive data |
| **Configurable** | Thresholds adjustable per use case |
| **Compliance Ready** | GDPR, HIPAA, PCI-DSS support |
