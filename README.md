# Guardrails

A pluggable pipeline for validating and sanitizing images/text before AI processing.

## Features

- **NSFW Detection** - AdamCodd ViT (96.5% accuracy)
- **Violence Detection** - CLIP ViT-H/14 zero-shot (78% accuracy)
- **Weapons Detection** - CLIP ViT-H/14 zero-shot
- **Hate Symbol Detection** - CLIP ViT-H/14 (configurable labels)
- **Self-Harm Detection** - CLIP ViT-H/14 zero-shot
- **PII Detection** - Tesseract OCR + Presidio
- **Face Detection** - OpenCV Haar Cascade
- **Pluggable Architecture** - Add new checks via config
- **Configurable Labels** - Customize detection labels per region

## Models Used

| Check | Model | Accuracy | Size |
|-------|-------|----------|------|
| NSFW | AdamCodd/vit-base-nsfw-detector | 96.5% | 330MB |
| Violence/Hate | laion/CLIP-ViT-H-14-laion2B-s32B-b79K | 78% | 3.7GB |
| PII | Tesseract + Presidio | 90%+ | - |
| Faces | OpenCV Haar Cascade | 50-70% | 1MB |

**Model Downloads:**
- CLIP: https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K
- NSFW: https://huggingface.co/AdamCodd/vit-base-nsfw-detector

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

### Model Downloads

Models are downloaded automatically on first run. For offline/air-gapped environments:

```bash
# Download models manually
huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K
huggingface-cli download AdamCodd/vit-base-nsfw-detector
```

**Cache locations:**
- macOS/Linux: `~/.cache/huggingface/hub/`
- Windows: `C:\Users\<username>\.cache\huggingface\hub\`

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
| **REJECT** | Unsafe content, PII, or faces detected | No image returned |

Pipeline stops immediately on first failed check (early rejection).

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

# Model paths
# Use HuggingFace model ID or local path
model_paths:
  clip: "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

# Model precision (FP16 = faster, less memory)
use_fp16: true

# For Windows (use full local path):
# model_paths:
#   clip: "C:/Users/username/.cache/huggingface/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/main"

# NSFW model: "opennsfw2" (fast) or "adamcodd" (accurate)
nsfw_model: "adamcodd"

# Thresholds (lower = stricter)
nsfw_threshold: 0.80
violence_threshold: 0.70
hate_symbol_threshold: 0.75

# Configurable detection labels
# First 2 = safe categories, rest = unsafe categories
violence_labels:
  - "a safe, normal photograph"
  - "a document, text, screenshot"
  - "violence, gore, blood, injury, fighting"
  - "guns, weapons, knives, dangerous objects"
  - "self-harm, suicide, cutting"

hate_labels:
  - "a safe, normal photograph without symbols"
  - "a document, text, form, screenshot"
  - "nazi swastika tilted on red background, SS bolts, hitler salute"
  - "KKK imagery, white hood, burning cross, racist symbols"
  - "ISIS flag, terrorist symbols, extremist imagery"

# Regional customization (e.g., India - exclude religious swastika):
# hate_labels:
#   - "a safe, normal photograph without symbols"
#   - "a document, text, form, screenshot"
#   - "nazi swastika tilted 45 degrees on red background, SS bolts"
#   - "KKK imagery, burning cross, racist symbols"
#   - "ISIS flag, terrorist symbols"

# Offline mode (after models downloaded)
environment:
  hf_hub_offline: true
  transformers_offline: true
```

## Adding a New Check

**Step 1:** Create `src/guardrails/checks/watermark.py`

```python
from guardrails.base import BaseCheck, CheckResult

class WatermarkCheck(BaseCheck):
    name = "watermark"
    can_reject = True

    def check(self, image, config):
        # Detection logic - return "reject" if watermark found
        return CheckResult(safe=True, score=0.0, action="allow")
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

## API Response Example

```json
{
  "decision": "REJECT",
  "reason": "Unsafe content detected: guns=0.99, total=0.99",
  "is_safe": false,
  "results": {
    "nsfw": {"score": 0.02, "threshold": 0.8, "is_pass": true},
    "violence": {"score": 0.99, "threshold": 0.7, "is_pass": false}
  },
  "image_base64": null,
  "meta": {
    "sha256": "abc123...",
    "processing_ms": 245
  }
}
```

## Troubleshooting

**HuggingFace model not loading (Windows):**

Use full local path in config.yaml:
```yaml
model_paths:
  clip: "C:/Users/username/.cache/huggingface/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/main"
```

**NSFW weights not found:**
```bash
# macOS/Linux
mkdir -p ~/.opennsfw2/weights
curl -L https://github.com/bhky/opennsfw2/releases/download/v0.1.0/open_nsfw_weights.h5 \
  -o ~/.opennsfw2/weights/open_nsfw_weights.h5
```

**SSL certificate errors:**
```bash
uv sync  # certifi handles SSL certificates
```

**Model download taking too long:**

Download manually and set offline mode:
```bash
# Download
huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K

# Then set in config.yaml
environment:
  hf_hub_offline: true
```

## Hardware Requirements

| Setup | RAM | Storage | Inference |
|-------|-----|---------|-----------|
| CPU only | 8GB | 10GB | 500ms-2s |
| GPU (recommended) | 16GB + 8GB VRAM | 20GB | 100-200ms |

---

*See `docs/` for detailed documentation.*
