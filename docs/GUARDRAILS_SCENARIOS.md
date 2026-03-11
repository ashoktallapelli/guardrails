# Guardrails Scenarios for AI Applications

## Overview

This document outlines guardrail requirements for different AI modalities and data types.

---

## 1. AI Modality Scenarios

### 1.1 Image to Text (Vision AI)

**Use Cases:** OCR, Image Captioning, Visual Q&A, Document Processing

```
[Input Image] → [AI Model] → [Output Text]
```

| Stage | Guardrails | Purpose |
|-------|------------|---------|
| **Input** | NSFW Detection | Block explicit images |
| **Input** | Violence Detection | Block violent/gory images |
| **Input** | Face Detection | Privacy - block images with faces |
| **Input** | PII Detection (OCR) | Detect PII in document images |
| **Output** | Text PII Detection | Ensure generated text doesn't leak PII |
| **Output** | Prompt Injection Detection | Prevent malicious instructions in output |

**Example Flow:**
```
User uploads ID card image
  → Face detected → REJECT
  → OR PII detected (SSN visible) → REJECT
  → Safe image → Process → Check output text for PII leakage
```

---

### 1.2 Text to Image (Generative AI)

**Use Cases:** DALL-E, Stable Diffusion, Midjourney, Marketing Content Generation

```
[Input Text Prompt] → [AI Model] → [Output Image]
```

| Stage | Guardrails | Purpose |
|-------|------------|---------|
| **Input** | Prompt Injection Detection | Block malicious prompts |
| **Input** | Harmful Content Detection | Block requests for violence, NSFW |
| **Input** | Brand/Celebrity Name Detection | Prevent trademark/likeness issues |
| **Output** | NSFW Detection | Block explicit generated images |
| **Output** | Violence Detection | Block violent generated content |
| **Output** | Watermark Detection | Ensure AI watermark present OR detect existing watermarks |
| **Output** | Logo Detection | Detect accidental brand logo generation |
| **Output** | Face Detection | Detect if realistic faces generated |

**Example Flow:**
```
User prompt: "Generate product advertisement"
  → Check prompt for harmful content
  → Generate image
  → Check output for NSFW/Violence → REJECT if found
  → Check for brand logos → REJECT if found
  → Add AI watermark → ALLOW
```

---

### 1.3 Image to Image (Editing/Transformation)

**Use Cases:** Style Transfer, Image Editing, Inpainting, Super Resolution, Background Removal

```
[Input Image] → [AI Model] → [Output Image]
```

| Stage | Guardrails | Purpose |
|-------|------------|---------|
| **Input** | NSFW Detection | Block explicit source images |
| **Input** | Violence Detection | Block violent source images |
| **Input** | Face Detection | Privacy protection |
| **Input** | Copyright Detection | Detect copyrighted images |
| **Input** | Watermark Detection | Detect watermarked images |
| **Output** | NSFW Detection | Ensure output isn't explicit |
| **Output** | Violence Detection | Ensure output isn't violent |
| **Output** | Deepfake Detection | Detect manipulated faces |
| **Output** | Watermark Preservation | Ensure watermarks not removed |

**Example Flow:**
```
User uploads image for style transfer
  → Check for NSFW/Violence → REJECT if found
  → Check for faces → REJECT or blur
  → Process image
  → Verify output doesn't contain NSFW/Violence
  → ALLOW
```

---

## 2. PII Detection by Domain

### 2.1 Financial Data (PCI-DSS, SOX Compliance)

| PII Type | Pattern/Detection | Regulation |
|----------|-------------------|------------|
| Credit Card Number | 16 digits, Luhn check | PCI-DSS |
| Bank Account Number | 8-17 digits | SOX |
| Routing Number | 9 digits (ABA) | SOX |
| SSN | XXX-XX-XXXX | GLBA |
| Tax ID / EIN | XX-XXXXXXX | IRS |
| Income/Salary | Currency patterns | Internal |
| Investment Account | Brokerage patterns | SEC |

**Example Entities:**
```
Credit Card: 4111-1111-1111-1111
SSN: 123-45-6789
Bank Account: 1234567890
Routing: 021000021
```

---

### 2.2 Medical Data (HIPAA Compliance)

| PII Type | Pattern/Detection | HIPAA Category |
|----------|-------------------|----------------|
| Patient Name | NER detection | PHI |
| Medical Record Number (MRN) | Alphanumeric ID | PHI |
| Date of Birth | Date patterns | PHI |
| Diagnosis/Condition | Medical NER | PHI |
| Prescription/Medication | Drug names | PHI |
| Insurance ID | Policy numbers | PHI |
| Provider Name | NER detection | PHI |
| Lab Results | Numeric + units | PHI |
| Biometric Data | Fingerprint, DNA | PHI |

**HIPAA 18 Identifiers:**
1. Names
2. Geographic data (smaller than state)
3. Dates (except year)
4. Phone numbers
5. Fax numbers
6. Email addresses
7. SSN
8. Medical record numbers
9. Health plan beneficiary numbers
10. Account numbers
11. Certificate/license numbers
12. Vehicle identifiers
13. Device identifiers
14. Web URLs
15. IP addresses
16. Biometric identifiers
17. Full-face photographs
18. Any other unique identifier

---

## 3. Output Guardrails

### 3.1 Watermark Detection

**Purpose:** Detect watermarks in images (input) or ensure watermarks are added (output)

| Use Case | Action |
|----------|--------|
| Detect copyrighted watermarks | REJECT input with watermarks |
| Detect AI-generated markers | Identify AI-generated content |
| Ensure output has watermark | ADD watermark to AI outputs |
| Prevent watermark removal | REJECT if watermark stripped |

**Detection Methods:**
- Frequency domain analysis (DCT/DWT)
- Visible watermark OCR
- Invisible watermark extraction
- C2PA/Content Credentials verification

---

### 3.2 Logo Detection

**Purpose:** Detect brand logos to prevent trademark infringement

| Use Case | Action |
|----------|--------|
| Input contains brand logos | WARN or REJECT |
| AI generated brand logos | REJECT output |
| Competitor logo detection | Flag for review |
| Copyright protection | Block unauthorized use |

**Detection Methods:**
- Object detection models (trained on logo datasets)
- Template matching
- Feature matching (SIFT/ORB)
- Brand-specific classifiers

**Common Logo Datasets:**
- FlickrLogos-32
- LogoDet-3K
- QMUL-OpenLogo

---

## 4. Implementation Roadmap

### Currently Implemented
| Check | Model | Status |
|-------|-------|--------|
| NSFW Detection | AdamCodd ViT | ✅ Done |
| Violence Detection | CLIP ViT-H/14 | ✅ Done |
| Hate Symbols | CLIP ViT-H/14 | ✅ Done |
| Face Detection | OpenCV Haar Cascade | ✅ Done |
| PII Detection (General) | Presidio + Tesseract | ✅ Done |

### To Be Implemented
| Check | Suggested Model | Priority |
|-------|-----------------|----------|
| Financial PII | Presidio + Custom Recognizers | High |
| Medical PII (HIPAA) | Presidio + Medical NER | High |
| Watermark Detection | Frequency Analysis / CNN | Medium |
| Logo Detection | CLIP / Custom Object Detection | Medium |
| Deepfake Detection | CNN-based classifiers | Medium |
| Prompt Injection | Text classification | High |

---

## 5. Architecture Overview

```
                    ┌─────────────────────────────────────┐
                    │           INPUT GUARDRAILS          │
                    ├─────────────────────────────────────┤
                    │  • File Validation                  │
                    │  • NSFW Detection                   │
                    │  • Violence Detection               │
                    │  • Hate Symbol Detection            │
                    │  • PII Detection (Image OCR)        │
                    │  • Face Detection                   │
                    │  • Watermark Detection              │
                    │  • Logo Detection                   │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │           AI MODEL                  │
                    │  (Image→Text / Text→Image / etc.)   │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │          OUTPUT GUARDRAILS          │
                    ├─────────────────────────────────────┤
                    │  • NSFW Detection                   │
                    │  • Violence Detection               │
                    │  • PII Leakage Detection            │
                    │  • Watermark Addition/Verification  │
                    │  • Logo Detection                   │
                    │  • Deepfake Detection               │
                    └─────────────────────────────────────┘
```

---

## 6. Compliance Mapping

| Regulation | Guardrails Required |
|------------|---------------------|
| **GDPR** | PII Detection, Face Detection, Right to erasure |
| **HIPAA** | Medical PII, PHI Detection, Access logging |
| **PCI-DSS** | Credit Card Detection, Encryption |
| **SOX** | Financial Data Protection, Audit trails |
| **CCPA** | PII Detection, Consumer data protection |
| **AI Act (EU)** | Transparency, Watermarking, Bias detection |

---

## 7. Summary

| Scenario | Key Input Guardrails | Key Output Guardrails |
|----------|---------------------|----------------------|
| **Image → Text** | NSFW, Violence, Faces, PII | Text PII Leakage |
| **Text → Image** | Prompt Injection, Harmful Content | NSFW, Violence, Watermark, Logo |
| **Image → Image** | NSFW, Violence, Copyright, Faces | NSFW, Violence, Deepfake |
| **Financial** | Credit Card, SSN, Bank Account | Data Masking |
| **Medical** | HIPAA 18 Identifiers | PHI Anonymization |
