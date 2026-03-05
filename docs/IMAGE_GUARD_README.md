# Image Guardrails

A local-first pipeline that validates and sanitizes images before AI processing.

## Features

- **NSFW Detection** - OpenNSFW2 or AdamCodd ViT
- **Violence/Weapons Detection** - CLIP zero-shot classification
- **Hate Symbol Detection** - CLIP zero-shot classification
- **PII Detection & Redaction** - Tesseract OCR + Presidio
- **Face Detection & Blur** - OpenCV
- **Text PII Anonymization** - Presidio

## Installation

### System Dependencies

**Windows:**
1. Tesseract: Download from https://github.com/UB-Mannheim/tesseract/wiki
2. libmagic: Handled by `python-magic-bin` package (auto-installed)

**macOS:**
```bash
brew install libmagic tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libmagic1 tesseract-ocr
```

### Python Dependencies

```bash
uv sync
```

## Quick Start

### CLI - Image Processing

```bash
# Process image (sanitize and save)
uv run python image_guard.py image.jpg

# Analyze only (no modification)
uv run python image_guard.py image.jpg --analyze-only

# JSON output
uv run python image_guard.py image.jpg --json
```

### CLI - Text PII

```bash
# Detect PII
uv run python image_guard.py --text "Email: john@example.com"

# Anonymize PII
uv run python image_guard.py --text "Contact John at john@example.com" --anonymize
```

### API Server

**Start server:**
```bash
uv run uvicorn api:app --reload --port 8000
```

**Scan image (curl):**
```bash
curl -X POST "http://localhost:8000/scan/image" -F "file=@image.jpg"
```

**Scan text (curl):**
```bash
curl -X POST "http://localhost:8000/scan/text" -H "Content-Type: application/json" -d "{\"input_text\": \"Contact John at john@example.com\"}"
```

**Windows PowerShell alternative:**
```powershell
# Scan image
Invoke-RestMethod -Uri "http://localhost:8000/scan/image" -Method POST -Form @{file = Get-Item "image.jpg"}

# Scan text
Invoke-RestMethod -Uri "http://localhost:8000/scan/text" -Method POST -ContentType "application/json" -Body '{"input_text": "Contact John at john@example.com"}'

# Health check
Invoke-RestMethod -Uri "http://localhost:8000/check_health"
```

## Decision States

| Decision | Meaning | Output |
|----------|---------|--------|
| **ALLOW** | Safe, no redaction needed | Original image (EXIF stripped) |
| **REDACT** | Safe, PII/faces found | Sanitized image returned |
| **REJECT** | Unsafe content detected | No image returned |

### Reason Examples

| Decision | Reason |
|----------|--------|
| REJECT | `Unsafe content detected: violence=0.01, weapons=0.87` |
| REDACT | `PII redacted: 4, Faces blurred: 1` |
| ALLOW | `All checks passed, no redaction needed` |

## Configuration

All settings in `config.yaml`:

```yaml
# Thresholds (lower = stricter)
nsfw_threshold: 0.80
violence_threshold: 0.70
hate_symbol_threshold: 0.75

# Enable/Disable
enable_violence_check: true
enable_hate_symbol_check: true
enable_pii_redaction: true
enable_face_blur: true

# Limits
max_file_size_mb: 10
max_resolution:
  width: 4096
  height: 4096
```

## API Response Format

```json
{
  "decision": "REDACT",
  "reason": "PII redacted: 4, Faces blurred: 0",
  "is_safe": true,
  "is_redacted": true,
  "results": {
    "nsfw": {"score": 0.001, "threshold": 0.8, "is_pass": true},
    "violence": {"score": 0.001, "threshold": 0.7, "is_pass": true},
    "pii": {"score": 4.0, "is_pass": true},
    "faces": {"score": 0.0, "is_pass": true}
  },
  "sanitized_image_base64": "..."
}
```

## Output Folders

```
output/
├── allow/    # Safe images (EXIF stripped only)
├── redact/   # Sanitized images (PII/faces redacted)
└── (rejected images not saved)
```

## Troubleshooting

**NSFW weights not found:**

macOS/Linux:
```bash
mkdir -p ~/.opennsfw2/weights
curl -L https://github.com/bhky/opennsfw2/releases/download/v0.1.0/open_nsfw_weights.h5 \
  -o ~/.opennsfw2/weights/open_nsfw_weights.h5
```

Windows PowerShell:
```powershell
mkdir "$env:USERPROFILE\.opennsfw2\weights" -Force
Invoke-WebRequest -Uri "https://github.com/bhky/opennsfw2/releases/download/v0.1.0/open_nsfw_weights.h5" -OutFile "$env:USERPROFILE\.opennsfw2\weights\open_nsfw_weights.h5"
```

**CLIP model not loading:**
```bash
python -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"
```

**Windows SSL certificate errors:**
The script automatically handles SSL certificates on Windows using `certifi`. If issues persist, ensure `certifi` is installed:
```bash
pip install certifi
```

**False positives on text/document images:**
The CLIP classification includes a "document" label to prevent false positives on text images like receipts, forms, or PII documents.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/scan/image` | Scan and sanitize image |
| POST | `/scan/text` | Scan and anonymize text |
| GET | `/check_health` | Health check |
| GET | `/config` | View configuration |

## Interactive Docs

Once API is running:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

*See `tools_reference.md` for detailed tool documentation.*
