# Enterprise AI Image Guardrails Reference

## Overview

This document provides a comprehensive reference for implementing production-grade image guardrails for enterprise AI systems. It covers safety checks, available models, accuracy benchmarks, and regulatory compliance requirements.

---

## Table of Contents

1. [Guardrail Categories](#guardrail-categories)
2. [Safety Guardrails](#1-safety-guardrails)
3. [Privacy Guardrails](#2-privacy-guardrails)
4. [Trust & Authenticity Guardrails](#3-trust--authenticity-guardrails)
5. [Compliance Guardrails](#4-compliance-guardrails)
6. [Quality Guardrails](#5-quality-guardrails)
7. [Regulatory Requirements](#regulatory-requirements)
8. [Implementation Status](#implementation-status)
9. [Model Comparison Matrix](#model-comparison-matrix)

---

## Guardrail Categories

| Category | Purpose | Priority |
|----------|---------|----------|
| **Safety** | Prevent harmful content | Critical |
| **Privacy** | Protect personal information | Critical |
| **Trust** | Ensure content authenticity | High |
| **Compliance** | Meet legal/brand requirements | High |
| **Quality** | Ensure technical quality | Medium |

---

## 1. Safety Guardrails

### 1.1 NSFW/Adult Content Detection

**Purpose:** Detect sexually explicit, pornographic, or adult content.

| Model | Type | Accuracy | Size | License |
|-------|------|----------|------|---------|
| **AdamCodd/vit-base-nsfw-detector** | ViT | 96.5% | 330MB | Apache 2.0 |
| **Falconsai/nsfw_image_detection** | ViT | 95% | 330MB | Apache 2.0 |
| **opennsfw2** | ResNet-50 | 90% | 24MB | MIT |
| **yahoo/open_nsfw** | CaffeNet | 85% | 80MB | BSD |

**Recommended:** AdamCodd/vit-base-nsfw-detector (best accuracy)

**HuggingFace Links:**
- https://huggingface.co/AdamCodd/vit-base-nsfw-detector
- https://huggingface.co/Falconsai/nsfw_image_detection

**Regulations:** COPPA, Platform Terms of Service, Broadcasting Standards

---

### 1.2 Violence Detection

**Purpose:** Detect violent imagery including blood, gore, injury, fighting.

| Model | Type | Accuracy | Size | License |
|-------|------|----------|------|---------|
| **CLIP ViT-H/14 (LAION)** | Zero-shot | 78% | 2.5GB | MIT |
| **CLIP ViT-L/14 (OpenAI)** | Zero-shot | 75% | 1.7GB | MIT |
| **CLIP ViT-B/32 (OpenAI)** | Zero-shot | 63% | 605MB | MIT |

**Detection Labels:**
```
"violence", "blood", "gore", "injury", "fighting", "assault", "murder"
```

**HuggingFace Links:**
- https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K
- https://huggingface.co/openai/clip-vit-large-patch14

**Regulations:** Platform Content Policies, Broadcasting Standards, App Store Guidelines

---

### 1.3 Weapons Detection

**Purpose:** Detect firearms, knives, and other dangerous weapons.

| Model | Type | Accuracy | Classes | Size |
|-------|------|----------|---------|------|
| **YOLOv8x-weapons** | Object Detection | 95%+ | Guns, Rifles, Pistols, Knives | 130MB |
| **YOLOv8n-weapons** | Object Detection | 90% | Guns, Knives | 6MB |
| **CLIP ViT-H/14** | Zero-shot | 78% | Any weapon term | 2.5GB |

**YOLOv8 Weapon Classes:**
- Handgun, Pistol, Revolver
- Rifle, Shotgun, Assault Rifle
- Knife, Machete, Sword

**HuggingFace/GitHub Links:**
- https://huggingface.co/mshamrai/yolov8x-coco-weapons
- https://github.com/ultralytics/ultralytics

**Regulations:** Platform Policies, Advertising Standards, Regional Laws

---

### 1.4 Hate Symbols Detection

**Purpose:** Detect extremist imagery, hate symbols, racist iconography.

| Model | Type | Accuracy | Size | License |
|-------|------|----------|------|---------|
| **CLIP ViT-H/14 (LAION)** | Zero-shot | 78% | 2.5GB | MIT |
| **CLIP ViT-L/14 (OpenAI)** | Zero-shot | 75% | 1.7GB | MIT |

**Detection Labels:**
```
"nazi symbol", "swastika", "kkk imagery", "white supremacist symbol",
"confederate flag", "hate symbol", "extremist imagery", "racist symbol"
```

**Limitations:** Zero-shot may miss obscure or coded symbols. Consider maintaining a custom symbol database.

**Regulations:**
- **Germany:** NetzDG (Network Enforcement Act)
- **EU:** Digital Services Act (DSA)
- **US:** Platform Terms of Service

---

### 1.5 Self-Harm/Suicide Content

**Purpose:** Detect imagery depicting self-harm, cutting, suicide.

| Model | Type | Accuracy | Availability |
|-------|------|----------|--------------|
| **CLIP ViT-H/14** | Zero-shot | ~70% | Open Source |
| **Custom CNN** | Classification | 85%+ | Requires training |
| **Commercial APIs** | Various | 90%+ | Paid (AWS, Google, Azure) |

**Detection Labels (CLIP):**
```
"self-harm", "cutting", "suicide", "self-injury", "wrist cutting"
```

**Note:** Limited open-source options. Consider commercial APIs for production.

**Regulations:**
- **EU:** Digital Services Act (DSA) - illegal content removal
- Platform suicide prevention policies

---

### 1.6 Child Safety (CSAM)

**Purpose:** Detect Child Sexual Abuse Material.

| Solution | Type | Availability | Notes |
|----------|------|--------------|-------|
| **Microsoft PhotoDNA** | Hash matching | Enterprise license | Industry standard |
| **Google CSAI Match** | Hash matching | Enterprise license | Used by major platforms |
| **NCMEC Hash Database** | Hash matching | Law enforcement | Requires partnership |
| **Thorn Safer** | AI + Hash | Non-profit license | For qualifying organizations |

**CRITICAL:**
- No open-source models available (ethical/legal reasons)
- Must use certified hash-matching services
- Legal obligation to report to NCMEC (US) or equivalent

**Regulations:**
- **US:** PROTECT Act, CPPA, 18 U.S.C. § 2256
- **EU:** Directive 2011/93/EU
- **UK:** Protection of Children Act 1978
- **Global:** UN Convention on the Rights of the Child

---

### 1.7 Drug/Substance Detection

**Purpose:** Detect illegal drugs, paraphernalia, substance abuse imagery.

| Model | Type | Accuracy | Availability |
|-------|------|----------|--------------|
| **CLIP ViT-H/14** | Zero-shot | ~70% | Open Source |
| **Custom YOLOv8** | Object Detection | 90%+ | Requires training |
| **Commercial APIs** | Various | 85%+ | Paid |

**Detection Labels (CLIP):**
```
"drugs", "cocaine", "marijuana", "pills", "syringe", "drug paraphernalia",
"smoking drugs", "drug use"
```

**Regulations:** Platform Policies, Advertising Standards, Regional Drug Laws

---

## 2. Privacy Guardrails

### 2.1 Face Detection

**Purpose:** Detect human faces for privacy protection.

| Model | Type | Accuracy | Speed | Size |
|-------|------|----------|-------|------|
| **YOLOv8n-face** | Object Detection | 98%+ | 5ms | 6MB |
| **YOLOv8x-face** | Object Detection | 99%+ | 20ms | 130MB |
| **RetinaFace** | Detection | 99.4% | 15ms | 100MB |
| **MTCNN** | Detection | 95% | 50ms | 2MB |
| **Haar Cascade** | Classical CV | 50-70% | 10ms | 1MB |
| **dlib HOG** | Classical CV | 80% | 30ms | 100MB |

**Recommended:** YOLOv8-face (best accuracy/speed balance)

**GitHub/HuggingFace Links:**
- https://github.com/derronqi/yolov8-face
- https://github.com/serengil/retinaface
- https://github.com/ipazc/mtcnn

**Regulations:**
- **GDPR** (EU): Biometric data is sensitive personal data
- **CCPA** (California): Right to opt-out of facial recognition
- **BIPA** (Illinois): Explicit consent required
- **LGPD** (Brazil): Similar to GDPR

---

### 2.2 License Plate Detection

**Purpose:** Detect and redact vehicle license plates.

| Model | Type | Accuracy | Size |
|-------|------|----------|------|
| **YOLOv8n-license-plate** | Object Detection | 95%+ | 6MB |
| **YOLOv11-license-plate** | Object Detection | 97%+ | 10MB |
| **OpenALPR** | OCR + Detection | 95% | 50MB |

**HuggingFace/GitHub Links:**
- https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8
- https://github.com/openalpr/openalpr

**Regulations:**
- **GDPR** (EU): License plates are personal data
- **CCPA** (California): Personal information protection
- Street View policies (Google, Apple)

---

### 2.3 PII in Images (OCR-based)

**Purpose:** Detect text-based PII in images (emails, phone numbers, SSN, etc.)

| Component | Model | Purpose |
|-----------|-------|---------|
| **OCR** | Tesseract, EasyOCR, PaddleOCR | Text extraction |
| **NER** | Presidio, spaCy | Entity recognition |

**Entities Detected:**
- Email addresses
- Phone numbers
- Social Security Numbers
- Credit card numbers
- Addresses
- Names
- IP addresses

**Libraries:**
- https://github.com/microsoft/presidio
- https://github.com/JaidedAI/EasyOCR

**Regulations:**
- **GDPR** (EU): All personal data
- **CCPA** (California): Personal information
- **HIPAA** (US): Protected health information
- **PCI-DSS**: Credit card data

---

### 2.4 Document Detection

**Purpose:** Detect identity documents, passports, IDs, credit cards.

| Model | Type | Accuracy | Classes |
|-------|------|----------|---------|
| **YOLOv8-documents** | Object Detection | 95%+ | ID, Passport, License, Card |
| **CLIP ViT-H/14** | Zero-shot | 80% | Any document type |
| **DocTR** | Document AI | 90%+ | Various documents |

**Detection Labels (CLIP):**
```
"identity document", "passport", "driver license", "credit card",
"social security card", "id card", "government document"
```

**Regulations:** GDPR, KYC/AML requirements, Identity Theft Prevention

---

## 3. Trust & Authenticity Guardrails

### 3.1 AI-Generated Image Detection (Deepfake)

**Purpose:** Detect synthetic/AI-generated images.

| Model | Type | Accuracy | Size |
|-------|------|----------|------|
| **umm-maybe/AI-image-detector** | ViT | 90%+ | 350MB |
| **Hive AI Detector** | Commercial | 95%+ | API |
| **Illuminarty** | Commercial | 92%+ | API |

**HuggingFace Links:**
- https://huggingface.co/umm-maybe/AI-image-detector

**Detection Capabilities:**
- GAN-generated faces
- Stable Diffusion images
- DALL-E images
- Midjourney images
- Face swaps

**Regulations:**
- **EU AI Act**: Synthetic media disclosure requirements
- **China**: Deep synthesis regulations
- Platform authenticity policies

---

### 3.2 Watermark Detection

**Purpose:** Detect copyrighted watermarks and logos.

| Model | Type | Accuracy |
|-------|------|----------|
| **Custom CNN** | Classification | 85%+ |
| **CLIP ViT-H/14** | Zero-shot | 70% |
| **Template Matching** | Classical CV | 90%+ (known watermarks) |

**Detection Labels (CLIP):**
```
"watermarked image", "stock photo watermark", "copyright watermark"
```

**Regulations:** Copyright Law, Digital Millennium Copyright Act (DMCA)

---

### 3.3 Screenshot/Screen Capture Detection

**Purpose:** Detect screenshots of apps, websites, or UI elements.

| Model | Type | Accuracy |
|-------|------|----------|
| **CLIP ViT-H/14** | Zero-shot | 75% |
| **Custom CNN** | Classification | 90%+ |

**Detection Labels (CLIP):**
```
"screenshot", "screen capture", "mobile app screenshot", "website screenshot"
```

---

## 4. Compliance Guardrails

### 4.1 Logo/Trademark Detection

**Purpose:** Detect brand logos and trademarks.

| Model | Type | Accuracy | Classes |
|-------|------|----------|---------|
| **CLIP ViT-H/14** | Zero-shot | 75% | Any brand |
| **LogoDet-3K** | Object Detection | 90%+ | 3000 logos |
| **Custom YOLOv8** | Object Detection | 95%+ | Specific brands |

**GitHub Links:**
- https://github.com/Wangjing1551/LogoDet-3K-Dataset

**Regulations:** Trademark Law, Brand Guidelines, Advertising Standards

---

### 4.2 Alcohol/Tobacco Detection

**Purpose:** Detect alcohol and tobacco products for age-restricted content.

| Model | Type | Accuracy |
|-------|------|----------|
| **CLIP ViT-H/14** | Zero-shot | 80% |
| **Custom YOLOv8** | Object Detection | 95%+ |

**Detection Labels (CLIP):**
```
"alcohol bottle", "beer", "wine", "cigarette", "smoking", "vaping",
"tobacco product", "liquor"
```

**Regulations:**
- Alcohol advertising laws (vary by region)
- Tobacco advertising bans
- Age verification requirements

---

### 4.3 Gambling Content Detection

**Purpose:** Detect gambling-related imagery.

| Model | Type | Accuracy |
|-------|------|----------|
| **CLIP ViT-H/14** | Zero-shot | 75% |

**Detection Labels (CLIP):**
```
"casino", "gambling", "slot machine", "poker chips", "roulette",
"sports betting", "lottery"
```

**Regulations:** Gambling advertising laws, Age restrictions

---

## 5. Quality Guardrails

### 5.1 Image Quality Assessment

**Purpose:** Assess technical quality of images.

| Model | Type | Metrics |
|-------|------|---------|
| **BRISQUE** | No-reference | Quality score 0-100 |
| **NIQE** | No-reference | Naturalness score |
| **CLIP-IQA** | Learned | Quality/aesthetic score |

**GitHub Links:**
- https://github.com/chaofengc/IQA-PyTorch

---

### 5.2 Resolution/Format Validation

**Checks:**
- Minimum/maximum resolution
- Aspect ratio limits
- File format validation (magic bytes)
- File size limits
- Color space validation

---

## Regulatory Requirements

### Global Regulations Summary

| Regulation | Region | Key Requirements | Penalties |
|------------|--------|------------------|-----------|
| **GDPR** | EU | Data protection, consent, right to erasure | Up to €20M or 4% global revenue |
| **CCPA/CPRA** | California | Consumer privacy rights, opt-out | $2,500-$7,500 per violation |
| **COPPA** | US | Children's privacy, parental consent | $50,000+ per violation |
| **BIPA** | Illinois | Biometric data consent | $1,000-$5,000 per violation |
| **Digital Services Act** | EU | Illegal content removal, transparency | Up to 6% global revenue |
| **NetzDG** | Germany | Hate speech removal within 24h | Up to €50M |
| **AI Act** | EU | AI system transparency, risk assessment | Up to €35M or 7% revenue |
| **LGPD** | Brazil | Data protection similar to GDPR | Up to 2% revenue |
| **PIPL** | China | Data localization, consent | Up to ¥50M |
| **POPIA** | South Africa | Data protection, consent | Up to R10M or imprisonment |

### Content-Specific Requirements

| Content Type | Regulation | Requirement |
|--------------|------------|-------------|
| CSAM | US 18 U.S.C. § 2258A | Mandatory reporting to NCMEC |
| Biometric Data | GDPR Art. 9 | Explicit consent required |
| Children's Data | COPPA | Parental consent under 13 |
| Hate Speech | NetzDG | 24h removal for obvious cases |
| Terrorist Content | EU TCO Regulation | 1h removal after notice |
| Deepfakes | EU AI Act | Disclosure requirements |

---

## Implementation Status

### Current Implementation

| Check | Status | Model | Accuracy |
|-------|--------|-------|----------|
| File Validation | ✅ Implemented | Magic bytes | 100% |
| NSFW | ✅ Implemented | AdamCodd ViT | 96.5% |
| Violence | ✅ Implemented | CLIP ViT-H/14 | 78% |
| Hate Symbols | ✅ Implemented | CLIP ViT-H/14 | 78% |
| PII (OCR) | ✅ Implemented | Presidio | 90%+ |
| Faces | ✅ Implemented | Haar Cascade | 50-70% |

### Recommended Upgrades

| Check | Current | Recommended | Accuracy Gain |
|-------|---------|-------------|---------------|
| Faces | Haar Cascade (50-70%) | YOLOv8-Face | +28-48% |
| Weapons | CLIP (78%) | YOLOv8-weapons | +17% |
| License Plates | None | YOLOv8-LP | New |
| AI Detection | None | AI-image-detector | New |
| Self-Harm | None | CLIP + Rules | New |

### Priority Implementation Roadmap

**Phase 1 - Critical (Week 1-2):**
1. Upgrade Face Detection → YOLOv8-Face
2. Add License Plate Detection
3. Add Dedicated Weapons Detection

**Phase 2 - High Priority (Week 3-4):**
4. Add AI-Generated Detection
5. Add Document Detection
6. Integrate CSAM hash-matching service

**Phase 3 - Medium Priority (Week 5-6):**
7. Add Self-Harm Detection
8. Add Drug Detection
9. Add Logo/Trademark Detection

---

## Model Comparison Matrix

### Accuracy vs Speed vs Size

| Model | Accuracy | Inference (ms) | Size | GPU Required |
|-------|----------|----------------|------|--------------|
| CLIP ViT-B/32 | 63% | 20ms | 605MB | No |
| CLIP ViT-L/14 | 75% | 50ms | 1.7GB | Recommended |
| CLIP ViT-H/14 | 78% | 100ms | 2.5GB | Recommended |
| YOLOv8n | 90% | 5ms | 6MB | No |
| YOLOv8x | 95%+ | 20ms | 130MB | Recommended |
| AdamCodd NSFW | 96.5% | 30ms | 330MB | No |

### Model Selection Guide

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
| Low latency (<50ms) | YOLOv8n + CLIP-B/32 | Fast inference |
| High accuracy | YOLOv8x + CLIP-H/14 | Best detection |
| Edge deployment | YOLOv8n only | Small footprint |
| Cost-sensitive | CLIP-B/32 | Single model, multiple tasks |
| Enterprise production | Full stack (all models) | Maximum coverage |

---

## Hardware Requirements

### Minimum (CPU-only)
- 8GB RAM
- 4 CPU cores
- 10GB storage
- Inference: 500ms-2s per image

### Recommended (GPU)
- 16GB RAM
- 8GB VRAM (NVIDIA GPU)
- 50GB storage
- Inference: 100-200ms per image

### Production (High throughput)
- 32GB RAM
- 24GB VRAM (NVIDIA A10/A100)
- 100GB storage
- Inference: 50-100ms per image
- Throughput: 20-50 images/second

---

## References

### Model Repositories
- HuggingFace: https://huggingface.co/models
- Ultralytics: https://github.com/ultralytics/ultralytics
- OpenCV: https://github.com/opencv/opencv

### Regulatory Resources
- GDPR: https://gdpr.eu/
- CCPA: https://oag.ca.gov/privacy/ccpa
- EU AI Act: https://artificialintelligenceact.eu/
- NCMEC: https://www.missingkids.org/

### Industry Standards
- Trust & Safety Professional Association
- Tech Coalition (child safety)
- GIFCT (terrorism content)

---

*Document Version: 1.0*
*Last Updated: 2026-03-10*
