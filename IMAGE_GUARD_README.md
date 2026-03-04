# image_guard.py - Image Guardrails Pipeline

A local-first pre-inference pipeline that validates and sanitizes images before AI processing.

## Overview

`image_guard.py` provides three modes of operation:

| Mode | Description | Modifies Image |
|------|-------------|----------------|
| **Normal Mode** | Validates, sanitizes, and saves cleaned image | Yes |
| **Analyze-Only Mode** | Returns JSON analysis without modifying image | No |
| **Text PII Mode** | Detect/anonymize PII in text input | N/A |

---

## Features

| Feature | Model/Framework | Purpose |
|---------|-----------------|---------|
| File Type Validation | libmagic | Validates by magic bytes, not extension |
| Resolution Check | Pillow | Enforces max resolution limits |
| EXIF Stripping | Pillow | Removes GPS, device IDs, metadata |
| NSFW Detection | OpenNSFW2 or AdamCodd | Detects explicit content |
| Violence Detection | CLIP | Detects violence, gore, blood |
| Weapons Detection | CLIP | Detects guns, knives, weapons |
| Hate Symbol Detection | CLIP | Detects extremist/racist imagery |
| Image PII Detection/Redaction | Tesseract + Presidio | OCR + mask sensitive text in images |
| Text PII Detection/Anonymization | Presidio | Detect/anonymize PII in plain text |
| Face Detection/Blur | OpenCV | Detect and anonymize faces |
| Perceptual Hashing | imagehash | For known-bad content matching |

---

## Installation

### System Dependencies

**macOS:**
```bash
brew install libmagic tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libmagic1 tesseract-ocr
```

**Windows:**
1. Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
2. libmagic: Handled by `python-magic-bin` package

### Python Dependencies

```bash
uv sync
```

---

## Usage

### Normal Mode (Sanitize Image)

Process image and save sanitized output:

```bash
# Basic usage
uv run python image_guard.py image.jpg

# With custom config
uv run python image_guard.py image.jpg --config config.yaml

# With custom output directory
uv run python image_guard.py image.jpg --output-dir ./output

# JSON output
uv run python image_guard.py image.jpg --json
```

### Analyze-Only Mode (No Modification)

Get JSON analysis without modifying the image:

```bash
uv run python image_guard.py image.jpg --analyze-only
```

### Text PII Mode

Detect and optionally anonymize PII in text:

```bash
# Detect PII in text (direct input)
uv run python image_guard.py --text "My email is john@example.com"

# Detect PII from file
uv run python image_guard.py --text-file document.txt

# Anonymize PII in text
uv run python image_guard.py --text "Contact: John Smith, john@example.com" --anonymize

# Anonymize PII from file
uv run python image_guard.py --text-file sensitive.txt --anonymize
```

---

## CLI Arguments

### Image Mode

| Argument | Short | Description |
|----------|-------|-------------|
| `image` | - | Path to input image |
| `--config` | `-c` | Path to YAML config file |
| `--output-dir` | `-o` | Directory for sanitized output |
| `--json` | - | Output result as JSON |
| `--analyze-only` | `-a` | Analyze only, don't modify image |

### Text PII Mode

| Argument | Short | Description |
|----------|-------|-------------|
| `--text` | `-t` | Text to analyze for PII (direct input) |
| `--text-file` | - | Path to text file for PII analysis |
| `--anonymize` | - | Anonymize detected PII (use with --text/--text-file) |

---

## Output Formats

### Normal Mode (Text)

```
============================================================
DECISION: ALLOW
============================================================
Input:    test_images/sample.jpg
SHA256:   a1b2c3d4e5f6...
MIME:     image/jpeg
Size:     800x600
NSFW:     0.0234
Violence: 0.0200
Weapons:  0.0500
Safe:     0.9200
Output:   test_images/sample_sanitized.jpg
Reasons:  All checks passed
============================================================
```

### Analyze-Only Mode (JSON)

```json
{
  "input_path": "test_images/sample.jpg",
  "timestamp": "2026-03-03T10:30:00.000000+00:00",
  "decision": "ALLOW",
  "is_safe": true,
  "results": {
    "file_validation": {
      "valid": true,
      "mime_type": "image/jpeg",
      "error": null,
      "max_size_mb": 10
    },
    "resolution": {
      "valid": true,
      "width": 800,
      "height": 600,
      "max_width": 4096,
      "max_height": 4096
    },
    "nsfw": {
      "safe": true,
      "score": 0.0234,
      "threshold": 0.8
    },
    "violence": {
      "safe": true,
      "scores": {
        "safe": 0.9200,
        "violence": 0.0200,
        "weapons": 0.0500,
        "disturbing": 0.0100
      },
      "threshold": 0.7
    },
    "hate_symbols": {
      "safe": true,
      "scores": {
        "safe": 0.9700,
        "hate_symbols": 0.0100,
        "nazi_symbols": 0.0100,
        "racist_symbols": 0.0100,
        "combined_hate_score": 0.0300
      },
      "threshold": 0.75
    },
    "pii": {
      "enabled": true,
      "text_found": false,
      "entities": [],
      "entity_count": 0
    },
    "faces": {
      "enabled": true,
      "face_count": 1,
      "faces": [
        {"x": 100, "y": 100, "width": 50, "height": 50}
      ]
    }
  },
  "meta": {
    "sha256": "a1b2c3d4e5f6...",
    "mime_type": "image/jpeg",
    "file_size_bytes": 102400,
    "perceptual_hash": "8fc7ce862cfeefe7",
    "processing_ms": 1234
  }
}
```

### Text PII Mode (JSON)

```json
{
  "input_type": "text",
  "timestamp": "2026-03-04T06:27:21.598666+00:00",
  "original_length": 86,
  "pii": {
    "enabled": true,
    "original_length": 86,
    "anonymized_text": "My name is <PERSON> and my email is <EMAIL_ADDRESS>.",
    "anonymized_length": 52,
    "entities": [
      {
        "type": "EMAIL_ADDRESS",
        "original_text": "john@example.com",
        "score": 1.0,
        "start": 38,
        "end": 54
      },
      {
        "type": "PERSON",
        "original_text": "John Smith",
        "score": 0.85,
        "start": 11,
        "end": 21
      }
    ],
    "entity_count": 2
  },
  "anonymized": true,
  "output_text": "My name is <PERSON> and my email is <EMAIL_ADDRESS>."
}
```

---

## NSFW Model Options

Two NSFW detection models are supported:

| Feature | OpenNSFW2 (default) | AdamCodd |
|---------|---------------------|----------|
| **Architecture** | ResNet-50 (CNN) | ViT (Transformer) |
| **Framework** | TensorFlow | PyTorch |
| **Accuracy** | ~90% | **96.54%** |
| **Model Size** | ~24MB | ~330MB |
| **Speed** | Faster | Slower |

```yaml
# config.yaml
nsfw_model: "opennsfw2"  # Default, smaller, faster
nsfw_model: "adamcodd"   # Higher accuracy, larger
```

---

## Configuration

All settings are configured in `config.yaml`:

```yaml
# File Validation
allowed_mime_types:
  - "image/jpeg"
  - "image/png"
  - "image/webp"
  - "image/gif"

max_file_size_mb: 10

max_resolution:
  width: 4096
  height: 4096

# NSFW Model: "opennsfw2" (default) or "adamcodd" (higher accuracy)
nsfw_model: "opennsfw2"

# Detection Thresholds (lower = stricter)
nsfw_threshold: 0.80
violence_threshold: 0.70
hate_symbol_threshold: 0.75

# Enable/Disable Features
enable_violence_check: true
enable_hate_symbol_check: true
enable_pii_redaction: true
enable_face_blur: true

# Output Settings
face_blur_kernel_size: 51
output_quality: 95
```

### Threshold Reference

| Check | Config Key | Default | REJECT When |
|-------|------------|---------|-------------|
| NSFW | `nsfw_threshold` | 0.80 | score >= 0.80 |
| Violence | `violence_threshold` | 0.70 | (violence + weapons + disturbing) >= 0.70 |
| Hate Symbols | `hate_symbol_threshold` | 0.75 | (hate + nazi + racist) >= 0.75 |
| File Size | `max_file_size_mb` | 10 | size > 10MB |
| Resolution | `max_resolution` | 4096x4096 | width or height > 4096 |

---

## Pipeline Flow

### Normal Mode

```
Input Image
    │
    ├─► 1. SHA256 Hash (audit trail)
    │
    ├─► 2. File Type Check (magic bytes) ──► REJECT if invalid
    │
    ├─► 3. Resolution Check ──► REJECT if too large
    │
    ├─► 4. Strip EXIF Metadata
    │
    ├─► 5. NSFW Detection (OpenNSFW2) ──► REJECT if >= 0.80
    │
    ├─► 6. Violence/Weapons (CLIP) ──► REJECT if >= 0.70
    │
    ├─► 7. Hate Symbols (CLIP) ──► REJECT if >= 0.75
    │
    ├─► 8. PII Redaction (Presidio + OCR)
    │
    ├─► 9. Face Blur (OpenCV)
    │
    ├─► 10. Perceptual Hash
    │
    └─► 11. Save Sanitized Image
```

### Analyze-Only Mode

```
Input Image
    │
    ├─► 1. SHA256 Hash
    │
    ├─► 2. File Type Check ──► Record result
    │
    ├─► 3. Resolution Check ──► Record result
    │
    ├─► 4. NSFW Detection ──► Record score
    │
    ├─► 5. Violence/Weapons ──► Record scores
    │
    ├─► 6. Hate Symbols ──► Record scores
    │
    ├─► 7. PII Detection ──► Record entities (no redaction)
    │
    ├─► 8. Face Detection ──► Record faces (no blur)
    │
    ├─► 9. Perceptual Hash
    │
    └─► Output JSON (no image modification)
```

---

## CLIP Detection Labels

### Violence/Safety Labels

```python
safety_labels = [
    "a safe, normal photograph",                    # → safe
    "violence, gore, blood, injury, fighting",      # → violence
    "weapons, guns, knives, dangerous objects",     # → weapons
    "disturbing, graphic, shocking content",        # → disturbing
]
```

### Hate Symbol Labels

```python
hate_labels = [
    "a normal, safe photograph without any symbols",           # → safe
    "hate symbols, extremist imagery, or offensive symbols",   # → hate_symbols
    "nazi symbols, swastika, or white supremacist imagery",    # → nazi_symbols
    "confederate flag or racist symbols",                      # → racist_symbols
]
```

---

## Functions Reference

### Core Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `run_guardrails()` | Full image pipeline with sanitization | Dict with decision, output_path |
| `analyze_image()` | Image analysis only, no modification | Dict with all scores |
| `analyze_text()` | Text PII analysis/anonymization | Dict with PII results |
| `load_config()` | Load YAML config | Config dict |

### Validation Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `validate_file_type()` | Check MIME type via magic bytes | (bool, str) |
| `validate_resolution()` | Check image dimensions | (bool, str) |
| `sha256_file()` | Compute file hash | str |

### Detection Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `check_nsfw()` | NSFW detection (OpenNSFW2/AdamCodd) | (bool, float) |
| `check_violence_safety()` | Violence/weapons (CLIP) | (bool, Dict) |
| `check_hate_symbols()` | Hate symbols (CLIP) | (bool, Dict) |
| `detect_pii()` | PII detection in images (OCR) | Dict |
| `detect_text_pii()` | PII detection in plain text | Dict |
| `detect_faces()` | Face detection only | Dict |

### Sanitization Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `strip_exif()` | Remove EXIF metadata | bytes |
| `redact_pii()` | Mask PII with black boxes (images) | PIL Image |
| `anonymize_text_pii()` | Anonymize PII in text | Dict |
| `blur_faces()` | Blur detected faces | PIL Image |

### Utility Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `compute_perceptual_hash()` | Image fingerprint | str (hex) |

---

## Environment Variables

Set automatically to suppress SSL errors and enable offline mode:

```python
os.environ["HF_HUB_OFFLINE"] = "1"          # HuggingFace offline mode
os.environ["TRANSFORMERS_OFFLINE"] = "1"    # Transformers offline mode
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"  # Disable token checks
```

### Windows SSL Fix

Automatically applied on Windows:

```python
if sys.platform == 'win32':
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
```

---

## Models & Memory

| Model | Size | Runtime Memory | Used For |
|-------|------|----------------|----------|
| OpenNSFW2 | ~24MB | ~200MB | NSFW detection (default) |
| AdamCodd ViT | ~330MB | ~500MB | NSFW detection (higher accuracy) |
| CLIP (ViT-B/32) | ~605MB | ~1GB | Violence, weapons, hate symbols |
| Tesseract | ~22MB | ~100MB | OCR for PII |
| spaCy (en_core_web_lg) | ~740MB | ~1GB | PII entity detection |
| OpenCV Haar Cascade | <1MB | ~50MB | Face detection |

---

## Example Test Results

| Image | Decision | NSFW | Weapons | Violence | Disturbing |
|-------|----------|------|---------|----------|------------|
| guns.jpg | **REJECT** | 0.0006 | **0.8729** | 0.0080 | 0.0517 |
| toy_gun.jpg | **REJECT** | 0.0005 | **0.9759** | 0.0034 | 0.0085 |
| hunting.jpg | **REJECT** | 0.0009 | **0.8875** | 0.0031 | 0.0999 |
| knife.jpg | ALLOW | 0.2009 | 0.3097 | 0.0702 | 0.2034 |
| portrait.jpg | ALLOW | 0.0007 | 0.0495 | 0.0085 | 0.0104 |
| nsfw1.jpeg | **REJECT** | 0.3228 | 0.0463 | 0.0528 | **0.7157** |

---

## Troubleshooting

### CLIP Model Not Loading

**Error:** `Skipping violence check - model not available offline`

**Solution:** Download model to local cache first:
```bash
# With internet access
python -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"
```

### NSFW Weights Not Found

**Error:** `NSFW weights not found at ~/.opennsfw2/weights/open_nsfw_weights.h5`

**Solution:** Download weights manually:
```bash
mkdir -p ~/.opennsfw2/weights
curl -L https://github.com/bhky/opennsfw2/releases/download/v0.1.0/open_nsfw_weights.h5 \
  -o ~/.opennsfw2/weights/open_nsfw_weights.h5
```

### Resolution Too Large

**Error:** `resolution: valid: false`

**Solution:** Increase limit in config.yaml:
```yaml
max_resolution:
  width: 6000
  height: 6000
```

### Knife Not Detected

**Issue:** Kitchen knives score low on weapons detection

**Solution:** Lower threshold in config.yaml:
```yaml
violence_threshold: 0.40  # More strict
```

---

## Configuration Requirement

**`config.yaml` is required** - the script will fail if not found.

```bash
# Without config.yaml → Error
$ uv run python image_guard.py image.jpg
Error: Config file not found: config.yaml

# With config.yaml → Works
$ uv run python image_guard.py image.jpg
Loaded config from config.yaml
```

This ensures **single source of truth** - no hidden defaults, consistent behavior.

---

## Decision Logic Summary

| Check | Condition | Result |
|-------|-----------|--------|
| File Type | MIME not in allowed list | REJECT |
| File Size | > max_file_size_mb | REJECT |
| Resolution | > max_resolution | REJECT |
| NSFW | score >= 0.80 | REJECT |
| Violence | (violence + weapons + disturbing) >= 0.70 | REJECT |
| Hate Symbols | (hate + nazi + racist) >= 0.75 | REJECT |
| PII | Detected | INFO (no reject) |
| Faces | Detected | INFO (no reject) |

---

## Supported PII Entity Types

The following PII entities can be detected and anonymized:

| Entity Type | Description | Example |
|-------------|-------------|---------|
| `PERSON` | Person names | John Smith |
| `EMAIL_ADDRESS` | Email addresses | john@example.com |
| `PHONE_NUMBER` | Phone numbers | +1-555-123-4567 |
| `CREDIT_CARD` | Credit card numbers | 4111-1111-1111-1111 |
| `US_SSN` | US Social Security Numbers | 123-45-6789 |
| `IP_ADDRESS` | IP addresses | 192.168.1.1 |
| `IBAN_CODE` | Bank account numbers | DE89370400440532013000 |
| `LOCATION` | Locations and addresses | New York, NY |
| `DATE_TIME` | Dates and times | March 4, 2026 |
| `NRP` | Nationalities, religions, political groups | American |
| `MEDICAL_LICENSE` | Medical license numbers | MD12345 |
| `URL` | URLs and web addresses | https://example.com |

### PII Anonymization Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `replace` | Replace with type label | `<EMAIL_ADDRESS>` |
| `redact` | Remove entirely | (empty) |
| `mask` | Partial masking | `john****@****.com` |
| `hash` | SHA-256 hash | `a1b2c3d4...` |

---

## FastAPI Server (api.py)

A REST API that exposes the same functionality as the CLI.

### Start Server

```bash
# Development mode with auto-reload
uvicorn api:app --reload --port 8000

# Production mode
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze/image` | POST | Analyze image without modification |
| `/process/image` | POST | Full pipeline - sanitize and return |
| `/analyze/text` | POST | Detect PII in text |
| `/anonymize/text` | POST | Anonymize PII in text |
| `/health` | GET | Health check and model status |
| `/config` | GET | View current configuration |

### Example Requests

```bash
# Analyze image
curl -X POST -F "file=@image.jpg" http://localhost:8000/analyze/image

# Process image (returns sanitized JPEG)
curl -X POST -F "file=@image.jpg" http://localhost:8000/process/image -o sanitized.jpg

# Detect PII in text
curl -X POST http://localhost:8000/analyze/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Email: john@example.com"}'

# Anonymize text
curl -X POST http://localhost:8000/anonymize/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Contact John Smith at john@example.com"}'

# Health check
curl http://localhost:8000/health
```

### Interactive Docs

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

*Last updated: March 2026*
