# Image Guardrails Demo

A local-first image guardrails pipeline for validating and sanitizing images before AI processing.

## Features

- **File Type Validation**: Validates by magic bytes, not extension
- **EXIF Stripping**: Removes GPS, device IDs, and other metadata
- **NSFW Detection**: Uses OpenNSFW2 (CNN-based, runs locally)
- **PII Redaction**: OCR + Presidio to mask sensitive text in images
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
    ├─► NSFW Detection ──► REJECT if score >= threshold
    │
    ├─► PII Redaction (OCR + mask)
    │
    ├─► Face Blur (anonymize)
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
Output:   test_images/sample_sanitized.jpg
Reasons:  All checks passed
============================================================
```
