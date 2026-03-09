# Demo Guide: Image Guardrails Pipeline

## Overview

**What is Image Guardrails?**

A local-first security pipeline that validates and sanitizes images/text BEFORE sending to AI models. It acts as a gatekeeper to:
- Block unsafe content (NSFW, violence, hate symbols)
- Protect sensitive data (PII, faces)
- Ensure compliance (GDPR, HIPAA)

**Why do we need it?**
- AI models can be misused with harmful inputs
- Sensitive data (PII) can leak into AI training/logs
- Compliance requirements demand data protection
- No dependency on external APIs (runs 100% locally)

---

## Two Decision States

| Decision | Meaning | What Happens |
|----------|---------|--------------|
| **ALLOW** | Safe, no issues found | Original image returned (EXIF stripped) |
| **REJECT** | Unsafe content, PII, or faces detected | Image blocked, not processed |

Pipeline stops immediately on first failed check (early rejection).

**Reason Examples:**
- `REJECT`: "Unsafe content detected: violence=0.01, weapons=0.87"
- `REJECT`: "PII detected: 4 entities (PERSON, EMAIL_ADDRESS)"
- `REJECT`: "Faces detected: 2 face(s) found"
- `ALLOW`: "All checks passed"

---

## Pipeline Architecture

```
Input Image/Text
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│                    VALIDATION LAYER                       │
├──────────────────────────────────────────────────────────┤
│  1. File Type Check (magic bytes, not extension)          │
│  2. File Size Check (max 10MB)                            │
│  3. Resolution Check (max 4096x4096)                      │
└──────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│                    SAFETY LAYER                           │
├──────────────────────────────────────────────────────────┤
│  4. NSFW Detection (OpenNSFW2/AdamCodd)  → REJECT if unsafe│
│  5. Violence/Weapons Detection (CLIP)    → REJECT if unsafe│
│  6. Hate Symbol Detection (CLIP)         → REJECT if unsafe│
└──────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│                    PRIVACY LAYER                          │
├──────────────────────────────────────────────────────────┤
│  7. PII Detection (Tesseract + Presidio)  → REJECT if found│
│  8. Face Detection (OpenCV)               → REJECT if found│
│  9. EXIF Metadata Stripping (remove GPS, device info)     │
└──────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│                    OUTPUT                                 │
├──────────────────────────────────────────────────────────┤
│  Decision: ALLOW / REJECT                                 │
│  Reason: Human-readable explanation                       │
│  Sanitized Image (if ALLOW only)                          │
│  Audit Log Entry                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Models Used & Why

### 1. NSFW Detection

We support two NSFW detection models:

#### Option A: OpenNSFW2 (Default)

| Property | Value |
|----------|-------|
| **Architecture** | ResNet-50 (CNN) |
| **Size** | ~24MB |
| **Accuracy** | ~90% |
| **Framework** | TensorFlow/Keras |
| **Speed** | Fast |
| **Source** | Yahoo Research (open-source) |

**Why OpenNSFW2?**
- Lightweight and fast
- Works offline (no API calls)
- Industry standard (Yahoo's open-source model)
- Good balance of accuracy vs. speed
- Low memory footprint

#### Option B: AdamCodd ViT (Higher Accuracy)

| Property | Value |
|----------|-------|
| **Architecture** | ViT-base-patch16-384 (Vision Transformer) |
| **Size** | ~330MB |
| **Accuracy** | **96.54%** |
| **Framework** | PyTorch/Transformers |
| **Speed** | Slower |
| **Source** | HuggingFace (AdamCodd/vit-base-nsfw-detector) |

**Why AdamCodd?**
- **Higher accuracy** (96.54% vs 90%)
- Modern transformer architecture
- Better on edge cases
- Trained on larger dataset

**When to use which?**

| Use Case | Recommended Model |
|----------|-------------------|
| General use, speed priority | OpenNSFW2 |
| High accuracy needed | AdamCodd |
| Low memory environment | OpenNSFW2 |
| AI-generated images | AdamCodd |

**Configuration:**
```yaml
# config.yaml
nsfw_model: "opennsfw2"  # Default - fast, lightweight
# OR
nsfw_model: "adamcodd"   # Higher accuracy, larger model
```

---

### 2. Violence/Weapons/Hate Detection: CLIP (OpenAI)

| Property | Value |
|----------|-------|
| **Model** | ViT-B/32 (Vision Transformer) |
| **Size** | ~605MB |
| **Type** | Zero-shot classification |
| **Framework** | PyTorch/Transformers |
| **Source** | OpenAI (open-source) |

**Why CLIP?**
- **Zero-shot learning** - No training needed, just describe what to detect
- **Flexible** - Add new categories by adding text labels
- **Multi-purpose** - One model for violence, weapons, AND hate symbols
- **Accurate** - Trained on 400M image-text pairs
- **Extensible** - Easy to add new detection categories

**How it works:**
```python
# CLIP compares image against text labels using cosine similarity
labels = [
    "a safe, normal photograph",
    "a document, text, screenshot, form",  # Prevents false positives
    "violence, gore, blood, injury",
    "weapons, guns, knives",
    "disturbing, graphic content"
]
# Returns probability scores for each label (softmax, sum = 1.0)
```

**Document Label Fix:**
We added a "document" label to prevent false positives on text images (receipts, forms, PII documents). Without it, CLIP would incorrectly flag text documents as "weapons" or "violence".

| Image Type | document score | weapons score | Result |
|------------|----------------|---------------|--------|
| PII document | **1.00** | 0.00 | ALLOW |
| Gun photo | 0.00 | **0.87** | REJECT |
| Normal photo | 0.11 | 0.26 | ALLOW |

---

### 3. PII Detection: Presidio + Tesseract OCR

| Component | Purpose | Size |
|-----------|---------|------|
| **Tesseract OCR** | Extract text from images | ~22MB |
| **Presidio Analyzer** | Detect PII entities | Part of Presidio |
| **Presidio Anonymizer** | Replace/mask PII | Part of Presidio |
| **spaCy (en_core_web_lg)** | Named Entity Recognition | ~740MB |

**Why Presidio?**
- **Microsoft open-source** - Enterprise-grade, well maintained
- **Comprehensive** - Detects 15+ PII types
- **Configurable** - Adjustable confidence thresholds
- **Multiple operators** - Replace, mask, redact, hash
- **Production ready** - Used in enterprise environments

**PII Types Detected:**

| Type | Example | Confidence |
|------|---------|------------|
| PERSON | John Smith | 0.85 |
| EMAIL_ADDRESS | john@example.com | 1.0 |
| PHONE_NUMBER | 555-123-4567 | 0.4 |
| CREDIT_CARD | 4111-1111-1111-1111 | 1.0 |
| US_SSN | 123-45-6789 | 1.0 |
| IP_ADDRESS | 192.168.1.1 | 0.6 |
| LOCATION | New York, NY | 0.85 |
| URL | https://example.com | 0.5 |

**Anonymization Operators:**

| Operator | Input | Output |
|----------|-------|--------|
| replace | John Smith | `<PERSON>` |
| redact | John Smith | (empty) |
| mask | John Smith | `****` |
| hash | John Smith | `a1b2c3d4` |

---

### 4. Face Detection: OpenCV Haar Cascade

| Property | Value |
|----------|-------|
| **Model** | Haar Cascade Classifier |
| **Size** | <1MB |
| **Speed** | Very fast (real-time) |
| **Framework** | OpenCV |

**Why Haar Cascade?**
- **Extremely lightweight** (<1MB vs 500MB+ for deep learning)
- **Fast** - Real-time detection
- **No GPU required**
- **Good enough** for privacy blur (not face recognition)
- **Battle-tested** - Used for 20+ years

---

### 5. File Validation: python-magic (libmagic)

**Why magic bytes instead of file extension?**

| Method | Security |
|--------|----------|
| File extension | Can be spoofed (rename .exe to .jpg) |
| Magic bytes | Checks actual file content |

**Example:**
```
File: malware.jpg (actually an executable)
Extension check: "image/jpeg" ← WRONG
Magic bytes check: "application/x-executable" ← CORRECT, REJECTED
```

---

## Model Sizes & Downloads

### Summary Table

| Model | Disk Size | Runtime Memory | Purpose |
|-------|-----------|----------------|---------|
| OpenNSFW2 | 24MB | ~200MB | NSFW detection (default) |
| AdamCodd ViT | 330MB | ~500MB | NSFW detection (high accuracy) |
| CLIP (ViT-B/32) | 605MB | ~1GB | Violence/weapons/hate |
| spaCy en_core_web_lg | 740MB | ~800MB | NER for PII detection |
| Presidio | 5MB | ~100MB | PII detection engine |
| Tesseract | 22MB | ~100MB | OCR |
| OpenCV Haar | <1MB | ~50MB | Face detection |

### Detailed Model Breakdown

#### 1. OpenNSFW2 (NSFW Detection - Default)

| Property | Details |
|----------|---------|
| **File** | `open_nsfw_weights.h5` |
| **Size** | 24 MB |
| **Location** | `~/.opennsfw2/weights/` |
| **Download** | https://github.com/bhky/opennsfw2/releases |
| **Architecture** | ResNet-50 (CNN) |
| **Framework** | TensorFlow/Keras |

#### 2. AdamCodd ViT (NSFW Detection - High Accuracy)

| Property | Details |
|----------|---------|
| **Files** | `pytorch_model.bin`, `config.json`, etc. |
| **Size** | 330 MB |
| **Location** | `~/.cache/huggingface/hub/models--AdamCodd--vit-base-nsfw-detector/` |
| **Download** | HuggingFace (auto-cached) |
| **Architecture** | ViT-base-patch16-384 (Transformer) |
| **Framework** | PyTorch |

#### 3. CLIP (Violence/Weapons/Hate Detection)

| Property | Details |
|----------|---------|
| **Files** | `pytorch_model.bin` (605MB), `config.json`, `vocab.json`, etc. |
| **Total Size** | 605 MB |
| **Location** | `~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/` |
| **Download** | HuggingFace (auto-cached) |
| **Architecture** | ViT-B/32 (Vision Transformer) |
| **Framework** | PyTorch |

**CLIP Files Breakdown:**
```
models--openai--clip-vit-base-patch32/
└── snapshots/
    └── <hash>/
        ├── pytorch_model.bin    605 MB  (main model weights)
        ├── config.json          4 KB
        ├── vocab.json           862 KB
        ├── merges.txt           525 KB
        ├── preprocessor_config.json  316 B
        ├── tokenizer_config.json     592 B
        └── special_tokens_map.json   389 B
```

#### 4. spaCy en_core_web_lg (NER for PII)

| Property | Details |
|----------|---------|
| **Size** | 740 MB |
| **Location** | Python site-packages |
| **Download** | `python -m spacy download en_core_web_lg` |
| **Purpose** | Named Entity Recognition |
| **Framework** | spaCy |

#### 5. Tesseract OCR

| Property | Details |
|----------|---------|
| **Size** | 22 MB (English data) |
| **Location** | System install |
| **Download** | OS package manager or installer |
| **Purpose** | Extract text from images |

### Total Disk Space Required

| Configuration | Disk Space | RAM Required |
|---------------|------------|--------------|
| **Minimal** (OpenNSFW2 only) | ~800 MB | ~1.5 GB |
| **Default** (OpenNSFW2 + CLIP) | ~1.4 GB | ~2 GB |
| **Full** (AdamCodd + CLIP) | ~1.7 GB | ~2.5 GB |

### Cache Locations

| OS | HuggingFace Cache Path |
|----|------------------------|
| **Windows** | `C:\Users\<username>\.cache\huggingface\hub\` |
| **macOS** | `~/.cache/huggingface/hub/` |
| **Linux** | `~/.cache/huggingface/hub/` |

### First Run vs Subsequent Runs

| Metric | First Run | Subsequent Runs |
|--------|-----------|-----------------|
| **Model Loading** | 5-10 seconds | 2-3 seconds |
| **Image Processing** | ~2 seconds | ~1-2 seconds |
| **Memory Peak** | ~2.5 GB | ~2 GB |

Models are cached after first download, so subsequent runs are faster.

---

## Demo Commands

### Setup

**Terminal 1: Start API Server**
```bash
cd /Users/ashoktallapelli/Workspace/AI_ML/guardrails
uv run uvicorn api:app --reload --port 8000
```

**Terminal 2: Run Demos**
```bash
cd /Users/ashoktallapelli/Workspace/AI_ML/guardrails
```

---

### Demo 1: Safe Image → ALLOW

```bash
uv run python image_guard.py test_images/sample.jpg
```

**Expected Output:**
```
==================================================
  DECISION: ALLOW
==================================================
  Reason: All checks passed

  Safety Scores:
    NSFW:         0.0009 (SAFE)
    Violence:     0.02
    Weapons:      0.26
    Safe:         0.49 (SAFE)
```

**Talking Point:** Image passed all safety checks. No PII or faces found. Original image returned with EXIF metadata stripped.

---

### Demo 2: Weapons Image → REJECT

```bash
uv run python image_guard.py test_images/guns.jpg
```

**Expected Output:**
```
==================================================
  DECISION: REJECT
==================================================
  Reason: Unsafe content detected: violence=0.01, weapons=0.87

  Safety Scores:
    NSFW:         0.0006 (SAFE)
    Weapons:      0.87 (UNSAFE)
```

**Talking Point:** CLIP detected weapons with 87% confidence. Image blocked. No output file created.

---

### Demo 3: PII Document → REJECT

```bash
uv run python -m guardrails test_images/pii_test.png
```

**Expected Output:**
```
==================================================
  DECISION: REJECT
==================================================
  Reason: PII detected: 4 entities (PERSON, EMAIL_ADDRESS, US_SSN, CREDIT_CARD)
  Rejected by: pii
  Skipped checks: faces
```

**Talking Point:** Document contains PII (name, email, SSN, credit card). Image rejected to protect sensitive data. No output file created.

---

### Demo 4: Text PII Anonymization

```bash
uv run python image_guard.py --text "Contact John Smith at john@example.com, SSN: 123-45-6789, Card: 4111-1111-1111-1111" --anonymize
```

**Expected Output:**
```
Original: Contact John Smith at john@example.com, SSN: 123-45-6789, Card: 4111-1111-1111-1111
Anonymized: Contact <PERSON> at <EMAIL_ADDRESS>, SSN: <US_SSN>, Card: <CREDIT_CARD>
```

**Talking Point:** All PII replaced with type labels. Original data never reaches AI model. Useful for sanitizing prompts before sending to LLMs.

---

### Demo 5: API - Image Scan

```bash
curl -s -X POST "http://localhost:8000/scan/image" \
  -F "file=@test_images/pii_test.png" | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(json.dumps({
    'decision': d['decision'],
    'reason': d['reason'],
    'is_safe': d['is_safe'],
    'is_redacted': d['is_redacted'],
    'pii_count': d['results']['pii']['score']
}, indent=2))"
```

**Expected Output:**
```json
{
  "decision": "REJECT",
  "rejected_by": "pii",
  "reason": "PII detected: 4 entities (PERSON, EMAIL_ADDRESS, US_SSN, CREDIT_CARD)",
  "is_safe": false,
  "not_run": ["faces"]
}
```

---

### Demo 6: API - Text Scan

```bash
curl -s -X POST "http://localhost:8000/scan/text" \
  -H "Content-Type: application/json" \
  -d '{"input_text": "Call John at 555-123-4567 or email john@company.com"}' | python3 -m json.tool
```

**Expected Output:**
```json
{
  "decision": "REJECT",
  "reason": "PII detected: 3 entities (PERSON, PHONE_NUMBER, EMAIL_ADDRESS)",
  "is_safe": false
}
```

---

### Demo 7: Swagger UI

Open browser: **http://localhost:8000/docs**

**Talking Point:** Interactive API documentation. Can test endpoints directly from browser.

---

### Demo 8: Audit Log

```bash
tail -3 guardrails_audit.log | python3 -m json.tool
```

**Talking Point:** Every decision logged with timestamp, input path, scores, and reasons. Essential for compliance and debugging.

---

## Configuration Highlights

```yaml
# NSFW Model Selection
nsfw_model: "opennsfw2"  # Fast, lightweight (default)
# nsfw_model: "adamcodd" # Higher accuracy (96.54%)

# Thresholds (lower = stricter)
nsfw_threshold: 0.80      # REJECT if NSFW score >= 0.80
violence_threshold: 0.70  # REJECT if violence+weapons >= 0.70
hate_symbol_threshold: 0.75

# Features can be toggled
enable_violence_check: true
enable_hate_symbol_check: true
enable_pii_redaction: true
enable_face_blur: true

# PII Settings
pii_score_threshold: 0.35  # Lower to catch phone numbers
pii_operator: "replace"    # replace, redact, mask, hash

# Limits
max_file_size_mb: 10
max_resolution:
  width: 4096
  height: 4096
```

---

## Key Talking Points for Managers

| Point | Details |
|-------|---------|
| **Security** | Blocks NSFW, violence, weapons, hate symbols before AI processing |
| **Privacy** | Auto-redacts PII and faces (GDPR/HIPAA friendly) |
| **Local Processing** | 100% on-premise, no data sent to external services |
| **Auditable** | Every decision logged with timestamp and scores |
| **Configurable** | Thresholds and features adjustable per use case |
| **API Ready** | REST API for easy integration with existing systems |
| **Extensible** | Add new detection categories via CLIP labels |
| **Model Choice** | Can switch between speed (OpenNSFW2) and accuracy (AdamCodd) |

---

## Integration Options

### 1. CLI (Batch Processing)
```bash
for img in images/*.jpg; do
  uv run python image_guard.py "$img" --json >> results.jsonl
done
```

### 2. REST API (Real-time)
```python
import requests

response = requests.post(
    "http://localhost:8000/scan/image",
    files={"file": open("image.jpg", "rb")}
)
result = response.json()
if result["decision"] == "ALLOW":
    # Send to AI model
    pass
```

### 3. Python Import (Library)
```python
from image_guard import run_guardrails, load_config

config = load_config()
result = run_guardrails("image.jpg", config)
if result["decision"] == "ALLOW":
    # Process image
    pass
```

---

## Summary

```
Image Guardrails = Safety + Privacy + Compliance

✓ Blocks NSFW, violence, weapons, hate symbols
✓ Rejects images with PII (names, emails, SSN, credit cards)
✓ Rejects images with faces for privacy
✓ Strips EXIF metadata (GPS, device info)
✓ Early rejection - stops on first failed check
✓ 100% local processing
✓ Full audit trail
✓ REST API ready
✓ Configurable thresholds
✓ Two NSFW models (speed vs accuracy)
```

---

## Q&A Preparation

**Q: Why not use cloud APIs like AWS Rekognition?**
A: Local processing = no data leaves network, no API costs, no latency, works offline.

**Q: How accurate is it?**
A: NSFW: 90-96.5% (depending on model), Violence/Weapons: ~85-90% with CLIP.

**Q: Can it detect new types of harmful content?**
A: Yes, CLIP allows zero-shot detection. Just add new text labels.

**Q: What about performance?**
A: First image takes ~5-10s (model loading). Subsequent images ~1-2s.

**Q: Does it work on Windows?**
A: Yes, fully tested on Windows with PowerShell examples in documentation.

---

*Last updated: March 2026*
