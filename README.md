# Image Guardrails Demo

A local-first image guardrails pipeline for validating and sanitizing images before AI processing.

## Features

- **File Type Validation**: Validates by magic bytes (libmagic), not extension
- **EXIF Stripping**: Removes GPS, device IDs, and other metadata
- **NSFW Detection**: Uses OpenNSFW2 (CNN-based, runs locally)
- **Violence/Safety Detection**: CLIP-based classifier for violence, weapons, disturbing content
- **Hate Symbol Detection**: CLIP zero-shot classification for extremist/hate imagery
- **PII Redaction**: OCR + Microsoft Presidio to mask sensitive text in images
- **Face Blur**: OpenCV-based face detection with Gaussian blur
- **Perceptual Hashing**: For known-bad content matching
- **Analyze-Only Mode**: Get JSON analysis without modifying the image

## Setup

### 1. Install System Dependencies

**macOS:**
```bash
brew install libmagic tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libmagic1 tesseract-ocr
```

**Windows:**
1. **Tesseract OCR**: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
   - Add to PATH: `C:\Program Files\Tesseract-OCR`
2. **libmagic**: Automatically handled by `python-magic-bin` package (no manual install needed)

### 2. Install Python Dependencies

```bash
cd guardrails
uv sync
```

### 3. Download Test Images

```bash
# Download a sample image
curl -L "https://picsum.photos/800/600" -o test_images/sample.jpg
```

## Usage

### Process a Single Image (with sanitization)

```bash
uv run python image_guard.py test_images/sample.jpg
```

With JSON output:
```bash
uv run python image_guard.py test_images/sample.jpg --json
```

### Analyze-Only Mode (no image modification)

Get comprehensive JSON analysis without modifying the image:
```bash
uv run python image_guard.py test_images/sample.jpg --analyze-only
```

Example output:
```json
{
  "input_path": "test_images/sample.jpg",
  "decision": "ALLOW",
  "is_safe": true,
  "results": {
    "file_validation": {"valid": true, "mime_type": "image/jpeg"},
    "resolution": {"valid": true, "width": 800, "height": 600},
    "nsfw": {"safe": true, "score": 0.0234, "threshold": 0.8},
    "violence": {
      "safe": true,
      "scores": {"safe": 0.92, "violence": 0.02, "weapons": 0.05, "disturbing": 0.01},
      "threshold": 0.7
    },
    "hate_symbols": {
      "safe": true,
      "scores": {"safe": 0.97, "hate_symbols": 0.01, "nazi_symbols": 0.01, "racist_symbols": 0.01},
      "threshold": 0.75
    },
    "pii": {"enabled": true, "text_found": false, "entities": []},
    "faces": {"enabled": true, "face_count": 1, "faces": [{"x": 100, "y": 100, "width": 50, "height": 50}]}
  },
  "meta": {
    "sha256": "a1b2c3d4...",
    "perceptual_hash": "0f1e2d3c",
    "processing_ms": 1234
  }
}
```

### Demo App

Interactive mode:
```bash
uv run python demo_app.py interactive
```

Process single image:
```bash
uv run python demo_app.py single test_images/sample.jpg
```

Batch process a directory:
```bash
uv run python demo_app.py batch test_images/
```

## Configuration

Edit `config.yaml` to adjust thresholds:

```yaml
nsfw_threshold: 0.80              # Lower = stricter
violence_threshold: 0.70          # Combined unsafe score threshold
enable_hate_symbol_check: true    # CLIP-based hate symbol detection
hate_symbol_threshold: 0.75       # Hate symbol score threshold
enable_pii_redaction: true
enable_face_blur: true
max_file_size_mb: 10
max_resolution:
  width: 4096
  height: 4096
```

## CLIP Detection Labels

### Violence/Safety Detection Labels

```python
violence_labels = [
    "a safe, normal photograph",                    # → safe
    "violence, gore, blood, or graphic injury",     # → violence
    "weapons, guns, knives, or firearms",           # → weapons
    "disturbing, shocking, or distressing content", # → disturbing
]
```

**Decision Logic:** If `violence + weapons + disturbing >= 0.70` → REJECT

### Hate Symbol Detection Labels

```python
hate_labels = [
    "a normal, safe photograph without any symbols",           # → safe
    "hate symbols, extremist imagery, or offensive symbols",   # → hate_symbols
    "nazi symbols, swastika, or white supremacist imagery",    # → nazi_symbols
    "confederate flag or racist symbols",                      # → racist_symbols
]
```

**Decision Logic:** If `hate_symbols + nazi_symbols + racist_symbols >= 0.75` → REJECT

## Pipeline Flow

```
Input Image
    │
    ├─► File Type Check (magic bytes) ──► REJECT if invalid
    │
    ├─► Size/Resolution Check ──► REJECT if too large (>4096x4096)
    │
    ├─► Strip EXIF Metadata
    │
    ├─► NSFW Detection (OpenNSFW2) ──► REJECT if >= 0.80
    │
    ├─► Violence/Safety (CLIP) ──► REJECT if unsafe >= 0.70
    │       ├── violence/gore
    │       ├── weapons (guns, knives, firearms)
    │       └── disturbing content
    │
    ├─► Hate Symbol Detection (CLIP) ──► REJECT if >= 0.75
    │       ├── extremist imagery
    │       ├── nazi symbols (swastika, SS)
    │       └── racist symbols (confederate flag)
    │
    ├─► PII Redaction (Presidio + OCR)
    │
    ├─► Face Blur (OpenCV)
    │
    ├─► Compute Perceptual Hash
    │
    └─► Output Sanitized Image + Decision Log
```

## Example Output

### Standard Mode
```
============================================================
DECISION: ALLOW
============================================================
Input:    test_images/sample.jpg
SHA256:   a1b2c3d4e5f6...
MIME:     image/jpeg
Size:     800x600
NSFW:     0.0234 (safe)
Violence: 0.0200
Weapons:  0.0500
Safe:     0.9200
Output:   test_images/sample_sanitized.jpg
Reasons:  All checks passed
============================================================
```

### Test Results Example

| Image | Decision | NSFW | Weapons | Violence | Disturbing |
|-------|----------|------|---------|----------|------------|
| toy_gun.jpg | REJECT | 0.0005 | **0.9759** | 0.0034 | 0.0085 |
| hunting.jpg | REJECT | 0.0009 | **0.8875** | 0.0031 | 0.0999 |
| knife.jpg | ALLOW | 0.2009 | 0.3097 | 0.0702 | 0.2034 |
| portrait.jpg | ALLOW | 0.0007 | 0.0495 | 0.0085 | 0.0104 |

## Image Q&A (Visual Question Answering)

After an image passes guardrails, you can ask questions about it:

```bash
# Describe the image
uv run python image_qa.py test_images/sample.jpg --describe

# Ask a specific question
uv run python image_qa.py test_images/sample.jpg --question "What colors are in this image?"

# Interactive Q&A mode
uv run python image_qa.py test_images/sample.jpg --interactive
```

**Flow:**
```
Image → Guardrails Check → If PASS → Load BLIP Model → Answer Questions
                         → If REJECT → "Cannot proceed with rejected image"
```

## Troubleshooting

### Windows SSL Certificate Error

If you see `SSL: CERTIFICATE_VERIFY_FAILED` error on Windows:

```powershell
# Re-sync dependencies (certifi should fix it):
uv sync

# Or manually download model weights:
# 1. Download: https://github.com/bhky/opennsfw2/releases/download/v0.1.0/open_nsfw_weights.h5
# 2. Place in: C:\Users\<username>\.opennsfw2\weights\open_nsfw_weights.h5
```

### Image Resolution Too Large

If you see `resolution: valid: false`, the image exceeds the max resolution (default 4096x4096).

Options:
1. Increase limit in `config.yaml`:
   ```yaml
   max_resolution:
     width: 6000
     height: 6000
   ```
2. Resize the image before processing

### Low Detection Scores for Knives

CLIP may score kitchen knives lower than firearms. To catch knives:
1. Lower `violence_threshold` to `0.4` in `config.yaml`
2. Or add knife-specific labels to the detection code
