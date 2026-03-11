# Image Guardrails

A pluggable pipeline for validating and sanitizing images before AI processing.

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

## Pipeline Flow

```mermaid
flowchart TB
    subgraph INPUT[" "]
        A[/"📷 Image Upload"/]
    end

    subgraph PIPELINE["⚡ Guardrails Pipeline"]
        direction TB
        B["1️⃣ File Validation<br/>Size & Type Check"]
        C["2️⃣ NSFW Detection<br/>Threshold: 0.80"]
        D["3️⃣ Violence Detection<br/>Threshold: 0.70"]
        E["4️⃣ Hate Symbols<br/>Threshold: 0.75"]
        F["5️⃣ PII Detection<br/>OCR → AI-Force"]
        G["6️⃣ Face Detection<br/>Privacy Check"]

        B --> C --> D --> E --> F --> G
    end

    subgraph OUTPUT[" "]
        H["✅ ALLOW<br/>Return sanitized image"]
        I["❌ REJECT<br/>Return reason"]
    end

    A --> B
    G -->|All Pass| H
    B -.->|Fail| I
    C -.->|Fail| I
    D -.->|Fail| I
    E -.->|Fail| I
    F -.->|Fail| I
    G -.->|Fail| I

    style A fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style B fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style C fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style D fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style E fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style F fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style G fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style H fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style I fill:#ffebee,stroke:#d32f2f,stroke-width:2px
    style PIPELINE fill:#fafafa,stroke:#616161,stroke-width:1px
```

### Key Behaviors

| Behavior | Description |
|----------|-------------|
| **Early Exit** | Pipeline stops immediately on first failed check |
| **Explainability** | Rejection includes specific reason (e.g., "guns=0.89") |
| **Skipped Checks** | Response includes `not_run` list of checks not executed |
| **No Redaction** | Only ALLOW or REJECT decisions |

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
```

### API

```bash
# Start server
uv run uvicorn guardrails.api:app --reload --port 8000

# Scan image
curl -X POST "http://localhost:8000/scan/image" -F "file=@image.jpg"

# Health check
curl http://localhost:8000/check_health
```

### Python

```python
from guardrails import Pipeline, load_config

config = load_config()
pipeline = Pipeline(config)

result = pipeline.run_image("image.jpg")
if result["decision"] == "ALLOW":
    # Process image
    pass
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
| GET | `/check_health` | Health check |
| GET | `/config` | View configuration |

**Swagger UI:** http://localhost:8000/docs

### API Request Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as /scan/image
    participant Pipeline
    participant FileCheck as File Validation
    participant NSFW as NSFW Check
    participant Violence as Violence Check
    participant Hate as Hate Symbols
    participant OCR as Tesseract OCR
    participant AIForce as AI-Force /scan/prompt
    participant Faces as Face Detection

    Client->>API: POST image file
    API->>API: Validate file type & size
    API->>Pipeline: Run image guardrails

    Pipeline->>FileCheck: Validate file
    FileCheck-->>Pipeline: PASS

    Pipeline->>NSFW: Check image
    NSFW-->>Pipeline: score=0.02 (PASS)

    Pipeline->>Violence: Check image
    Violence-->>Pipeline: score=0.15 (PASS)

    Pipeline->>Hate: Check image
    Hate-->>Pipeline: score=0.05 (PASS)

    Pipeline->>OCR: Extract text from image
    OCR-->>Pipeline: extracted_text
    Pipeline->>AIForce: POST /scan/prompt (extracted_text)
    AIForce-->>Pipeline: PII result (PASS/REJECT)

    Pipeline->>Faces: Detect faces
    Faces-->>Pipeline: face_count=0 (PASS)

    Pipeline-->>API: ALLOW (all checks passed)
    API->>API: Strip EXIF metadata
    API-->>Client: {"decision": "ALLOW", "image_base64": "..."}
```

### API Rejection Flow (Early Exit)

```mermaid
sequenceDiagram
    participant Client
    participant API as /scan/image
    participant Pipeline
    participant FileCheck as File Validation
    participant NSFW as NSFW Check
    participant Violence as Violence Check
    participant Hate as Hate Symbols
    participant OCR as Tesseract OCR
    participant AIForce as AI-Force /scan/prompt
    participant Faces as Face Detection

    Client->>API: POST image file
    API->>API: Validate file type & size
    API->>Pipeline: Run image guardrails

    Pipeline->>FileCheck: Validate file
    FileCheck-->>Pipeline: PASS

    Pipeline->>NSFW: Check image
    NSFW-->>Pipeline: score=0.02 (PASS)

    Pipeline->>Violence: Check image
    Violence-->>Pipeline: score=0.89, guns detected (FAIL)

    Note over Pipeline,Faces: Early Exit - Remaining checks skipped

    Pipeline-->>API: REJECT
    API-->>Client: {"decision": "REJECT", "reason": "guns=0.89", "not_run": ["hate_symbols", "pii", "faces"]}
```

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
