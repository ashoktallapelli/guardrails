# Image Guardrails

A local-first pipeline that validates and sanitizes images before AI processing.

## Features

- **NSFW Detection** - OpenNSFW2 or AdamCodd ViT
- **Violence/Weapons Detection** - CLIP zero-shot classification
- **Hate Symbol Detection** - CLIP zero-shot classification
- **PII Detection** - Tesseract OCR + Presidio (rejects if found)
- **Face Detection** - OpenCV (rejects if found)
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
| **ALLOW** | Safe, no issues | Original image (EXIF stripped) |
| **REJECT** | Unsafe content, PII, or faces detected | No image returned |

Pipeline stops immediately on first failed check (early rejection).

### Reason Examples

| Decision | Reason |
|----------|--------|
| REJECT | `Unsafe content detected: violence=0.01, weapons=0.87` |
| REJECT | `PII detected: 4 entities (PERSON, EMAIL_ADDRESS)` |
| REJECT | `Faces detected: 2 face(s) found` |
| ALLOW | `All checks passed` |

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

**On REJECT (early exit):**
```json
{
  "decision": "REJECT",
  "rejected_by": "nsfw",
  "reason": "NSFW score 0.92 >= threshold 0.8",
  "is_safe": false,
  "checks": {
    "file_validation": {"safe": true, "action": "allow"},
    "nsfw": {"safe": false, "score": 0.92, "action": "reject"}
  },
  "not_run": ["violence", "hate_symbols", "pii", "faces"]
}
```

**On ALLOW:**
```json
{
  "decision": "ALLOW",
  "reason": "All checks passed",
  "is_safe": true,
  "checks": {
    "nsfw": {"safe": true, "score": 0.001},
    "violence": {"safe": true, "score": 0.02},
    "pii": {"safe": true, "score": 0},
    "faces": {"safe": true, "score": 0}
  },
  "image_base64": "..."
}
```

## Output Folders

```
output/
├── allow/    # Safe images (EXIF stripped only)
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
