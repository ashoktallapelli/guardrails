# Guardrails

A pluggable pipeline for validating and sanitizing images/text before AI processing.

## Features

- **NSFW Detection** - OpenNSFW2 or AdamCodd ViT
- **Violence/Weapons Detection** - CLIP zero-shot
- **Hate Symbol Detection** - CLIP zero-shot
- **PII Detection & Redaction** - Tesseract OCR + Presidio
- **Face Detection & Blur** - OpenCV
- **Pluggable Architecture** - Add new checks without code changes

## Project Structure

```
guardrails/
├── src/guardrails/          # Main package
│   ├── base.py              # BaseCheck class
│   ├── pipeline.py          # Orchestrator
│   ├── config.py            # Config loading
│   ├── cli.py               # CLI entry point
│   ├── api.py               # FastAPI endpoints
│   └── checks/              # Pluggable checks
│       ├── nsfw.py
│       ├── violence.py
│       ├── hate_symbols.py
│       ├── pii.py
│       └── faces.py
├── tests/
├── docs/
├── scripts/                 # Utility scripts
├── config.yaml              # Configuration
└── pyproject.toml
```

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
- Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
- libmagic: Handled by `python-magic-bin` (auto-installed)

### Python Dependencies

```bash
uv sync
```

## Usage

### CLI

```bash
# Process image
uv run python -m guardrails test_images/sample.jpg

# Analyze only (no modification)
uv run python -m guardrails test_images/sample.jpg --analyze-only

# JSON output
uv run python -m guardrails test_images/sample.jpg --json

# Text PII
uv run python -m guardrails --text "Contact john@example.com" --anonymize
```

### API

```bash
# Start server
uv run uvicorn guardrails.api:app --reload --port 8000

# Scan image
curl -X POST "http://localhost:8000/scan/image" -F "file=@image.jpg"

# Scan text
curl -X POST "http://localhost:8000/scan/text" \
  -H "Content-Type: application/json" \
  -d '{"input_text": "Contact john@example.com"}'

# Health check
curl http://localhost:8000/check_health
```

### Python

```python
from guardrails import Pipeline, load_config

config = load_config()
pipeline = Pipeline(config)

# Image
result = pipeline.run_image("image.jpg")
if result["decision"] == "ALLOW":
    # Process image
    pass

# Text
result = pipeline.run_text("Contact john@example.com", anonymize=True)
print(result["anonymized_text"])
```

## Decision States

| Decision | Meaning | Output |
|----------|---------|--------|
| **ALLOW** | Safe, no issues | Original image (EXIF stripped) |
| **REDACT** | Safe, PII/faces found | Sanitized image |
| **REJECT** | Unsafe content | No image returned |

## Configuration

Edit `config.yaml`:

```yaml
# Checks to run (in order)
guardrails:
  - file_validation
  - nsfw
  - violence
  - hate_symbols
  - pii
  - faces

# Thresholds (lower = stricter)
nsfw_threshold: 0.80
violence_threshold: 0.70
hate_symbol_threshold: 0.75

# NSFW model: "opennsfw2" (fast) or "adamcodd" (accurate)
nsfw_model: "adamcodd"

# Enable/disable
enable_violence: true
enable_pii: true
enable_faces: true
```

## Adding a New Check

**Step 1:** Create `src/guardrails/checks/watermark.py`

```python
from guardrails.base import BaseCheck, CheckResult

class WatermarkCheck(BaseCheck):
    name = "watermark"
    can_redact = True

    def check(self, image, config):
        # Detection logic
        return CheckResult(safe=True, score=0.0, action="allow")

    def redact(self, image, config):
        # Redaction logic
        return image
```

**Step 2:** Add to `checks/__init__.py`

```python
from .watermark import WatermarkCheck
```

**Step 3:** Add to `pipeline.py` AVAILABLE_CHECKS

```python
AVAILABLE_CHECKS = {
    ...
    "watermark": WatermarkCheck,
}
```

**Step 4:** Enable in `config.yaml`

```yaml
guardrails:
  - nsfw
  - watermark
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/scan/image` | Scan and sanitize image |
| POST | `/scan/text` | Scan and anonymize text |
| GET | `/check_health` | Health check |
| GET | `/config` | View configuration |

**Swagger UI:** http://localhost:8000/docs

## Troubleshooting

**NSFW weights not found:**
```bash
# macOS/Linux
mkdir -p ~/.opennsfw2/weights
curl -L https://github.com/bhky/opennsfw2/releases/download/v0.1.0/open_nsfw_weights.h5 \
  -o ~/.opennsfw2/weights/open_nsfw_weights.h5
```

**Windows SSL errors:**
```bash
uv sync  # certifi handles SSL certificates
```

---

*See `docs/` for detailed documentation.*
