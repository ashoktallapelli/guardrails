# Image Guardrails Demo

A local-first image guardrails pipeline for validating and sanitizing images before AI processing.

## Features

- **File Type Validation**: Validates by magic bytes (libmagic), not extension
- **EXIF Stripping**: Removes GPS, device IDs, and other metadata
- **NSFW Detection**: Uses OpenNSFW2 (CNN-based, runs locally)
- **Violence/Safety Detection**: CLIP-based classifier for violence, weapons, disturbing content
- **PII Redaction**: OCR + Microsoft Presidio to mask sensitive text in images
- **Face Blur**: OpenCV-based face detection with Gaussian blur
- **Perceptual Hashing**: For known-bad content matching

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

### Process a Single Image

```bash
uv run python guard_image.py test_images/sample.jpg
```

With JSON output:
```bash
uv run python guard_image.py test_images/sample.jpg --json
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
nsfw_threshold: 0.80       # Lower = stricter
enable_pii_redaction: true
enable_face_blur: true
max_file_size_mb: 10
```

## Pipeline Flow

```
Input Image
    │
    ├─► File Type Check (magic bytes) ──► REJECT if invalid
    │
    ├─► Size/Resolution Check ──► REJECT if too large
    │
    ├─► Strip EXIF Metadata
    │
    ├─► NSFW Detection (OpenNSFW2) ──► REJECT if >= 0.80
    │
    ├─► Violence/Safety (CLIP) ──► REJECT if unsafe >= 0.70
    │       ├── violence/gore
    │       ├── weapons
    │       └── disturbing content
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
