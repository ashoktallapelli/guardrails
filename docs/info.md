Demo Guide: Image Guardrails Pipeline

  Overview

  What is Image Guardrails?

  A local-first security pipeline that validates and sanitizes images/text BEFORE sending to AI models. It acts as a gatekeeper to:
  - Block unsafe content (NSFW, violence, hate symbols)
  - Protect sensitive data (PII, faces)
  - Ensure compliance (GDPR, HIPAA)

  Why do we need it?
  - AI models can be misused with harmful inputs
  - Sensitive data (PII) can leak into AI training/logs
  - Compliance requirements demand data protection
  - No dependency on external APIs (runs 100% locally)

  ---
  Three Decision States
  ┌──────────┬───────────────────────────┬──────────────────────────────────────────┐
  │ Decision │          Meaning          │               What Happens               │
  ├──────────┼───────────────────────────┼──────────────────────────────────────────┤
  │ ALLOW    │ Safe, no issues found     │ Original image returned (EXIF stripped)  │
  ├──────────┼───────────────────────────┼──────────────────────────────────────────┤
  │ REDACT   │ Safe, but PII/faces found │ Sanitized image with redactions returned │
  ├──────────┼───────────────────────────┼──────────────────────────────────────────┤
  │ REJECT   │ Unsafe content detected   │ Image blocked, not processed             │
  └──────────┴───────────────────────────┴──────────────────────────────────────────┘
  Reason Examples:
  - REJECT: "Unsafe content detected: violence=0.01, weapons=0.87"
  - REDACT: "PII redacted: 4, Faces blurred: 1"
  - ALLOW: "All checks passed, no redaction needed"

  ---
  Pipeline Architecture

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
  │  4. NSFW Detection (OpenNSFW2)         → REJECT if unsafe │
  │  5. Violence/Weapons Detection (CLIP)  → REJECT if unsafe │
  │  6. Hate Symbol Detection (CLIP)       → REJECT if unsafe │
  └──────────────────────────────────────────────────────────┘
         │
         ▼
  ┌──────────────────────────────────────────────────────────┐
  │                    PRIVACY LAYER                          │
  ├──────────────────────────────────────────────────────────┤
  │  7. PII Detection & Redaction (Tesseract + Presidio)      │
  │  8. Face Detection & Blur (OpenCV)                        │
  │  9. EXIF Metadata Stripping (remove GPS, device info)     │
  └──────────────────────────────────────────────────────────┘
         │
         ▼
  ┌──────────────────────────────────────────────────────────┐
  │                    OUTPUT                                 │
  ├──────────────────────────────────────────────────────────┤
  │  Decision: ALLOW / REDACT / REJECT                        │
  │  Reason: Human-readable explanation                       │
  │  Sanitized Image (if ALLOW or REDACT)                     │
  │  Audit Log Entry                                          │
  └──────────────────────────────────────────────────────────┘

  ---
  Models Used & Why

  1. NSFW Detection: OpenNSFW2
  ┌───────────┬─────────────────┐
  │ Property  │      Value      │
  ├───────────┼─────────────────┤
  │ Model     │ ResNet-50 (CNN) │
  ├───────────┼─────────────────┤
  │ Size      │ ~24MB           │
  ├───────────┼─────────────────┤
  │ Accuracy  │ ~90%            │
  ├───────────┼─────────────────┤
  │ Framework │ TensorFlow      │
  └───────────┴─────────────────┘
  Why OpenNSFW2?
  - Lightweight and fast
  - Works offline (no API calls)
  - Industry standard (Yahoo's open-source model)
  - Good balance of accuracy vs. speed

  Alternative Available: AdamCodd ViT (96.5% accuracy, but 330MB)

  ---
  2. Violence/Weapons/Hate Detection: CLIP (OpenAI)
  ┌───────────┬───────────────────────────────┐
  │ Property  │             Value             │
  ├───────────┼───────────────────────────────┤
  │ Model     │ ViT-B/32 (Vision Transformer) │
  ├───────────┼───────────────────────────────┤
  │ Size      │ ~605MB                        │
  ├───────────┼───────────────────────────────┤
  │ Type      │ Zero-shot classification      │
  ├───────────┼───────────────────────────────┤
  │ Framework │ PyTorch                       │
  └───────────┴───────────────────────────────┘
  Why CLIP?
  - Zero-shot learning - No training needed, just describe what to detect
  - Flexible - Add new categories by adding text labels
  - Multi-purpose - One model for violence, weapons, AND hate symbols
  - Accurate - Trained on 400M image-text pairs

  How it works:
  labels = [
      "a safe, normal photograph",
      "a document, text, screenshot, form",  # Prevents false positives
      "violence, gore, blood, injury",
      "weapons, guns, knives",
      "disturbing, graphic content"
  ]
  # CLIP compares image against all labels and returns probability scores

  Document Label Fix:
  We added a "document" label to prevent false positives on text images (receipts, forms, PII documents). Without it, CLIP would incorrectly flag text documents as "weapons" or "violence".

  ---
  3. PII Detection: Presidio + Tesseract OCR
  ┌────────────────────────┬─────────────────────────────┐
  │       Component        │           Purpose           │
  ├────────────────────────┼─────────────────────────────┤
  │ Tesseract OCR          │ Extract text from images    │
  ├────────────────────────┼─────────────────────────────┤
  │ Presidio Analyzer      │ Detect PII entities in text │
  ├────────────────────────┼─────────────────────────────┤
  │ Presidio Anonymizer    │ Replace/mask PII            │
  ├────────────────────────┼─────────────────────────────┤
  │ spaCy (en_core_web_lg) │ Named Entity Recognition    │
  └────────────────────────┴─────────────────────────────┘
  Why Presidio?
  - Microsoft open-source - Enterprise-grade, well maintained
  - Comprehensive - Detects 15+ PII types
  - Configurable - Adjustable confidence thresholds
  - Multiple operators - Replace, mask, redact, hash

  PII Types Detected:
  ┌───────────────┬─────────────────────┐
  │     Type      │       Example       │
  ├───────────────┼─────────────────────┤
  │ PERSON        │ John Smith          │
  ├───────────────┼─────────────────────┤
  │ EMAIL_ADDRESS │ john@example.com    │
  ├───────────────┼─────────────────────┤
  │ PHONE_NUMBER  │ 555-123-4567        │
  ├───────────────┼─────────────────────┤
  │ CREDIT_CARD   │ 4111-1111-1111-1111 │
  ├───────────────┼─────────────────────┤
  │ US_SSN        │ 123-45-6789         │
  ├───────────────┼─────────────────────┤
  │ IP_ADDRESS    │ 192.168.1.1         │
  └───────────────┴─────────────────────┘
  ---
  4. Face Detection: OpenCV Haar Cascade
  ┌───────────┬─────────────────────────┐
  │ Property  │          Value          │
  ├───────────┼─────────────────────────┤
  │ Model     │ Haar Cascade Classifier │
  ├───────────┼─────────────────────────┤
  │ Size      │ <1MB                    │
  ├───────────┼─────────────────────────┤
  │ Speed     │ Very fast               │
  ├───────────┼─────────────────────────┤
  │ Framework │ OpenCV                  │
  └───────────┴─────────────────────────┘
  Why Haar Cascade?
  - Extremely lightweight (<1MB vs 500MB+ for deep learning)
  - Fast - Real-time detection
  - No GPU required
  - Good enough for privacy blur (not recognition)

  ---
  5. File Validation: python-magic (libmagic)

  Why magic bytes instead of file extension?
  - Extensions can be spoofed (rename .exe to .jpg)
  - Magic bytes check actual file content
  - Security best practice

  ---
  Demo Commands

  Setup: Start API Server (Terminal 1)
  cd /Users/ashoktallapelli/Workspace/AI_ML/guardrails
  uv run uvicorn api:app --reload --port 8000

  Demo Terminal (Terminal 2)
  cd /Users/ashoktallapelli/Workspace/AI_ML/guardrails

  ---
  Demo 1: Safe Image → ALLOW

  uv run python image_guard.py test_images/sample.jpg
  Expected Output:
  DECISION: ALLOW
  Reason: All checks passed, no redaction needed
  Explain: Image passed all safety checks, no PII or faces found.

  ---
  Demo 2: Weapons Image → REJECT

  uv run python image_guard.py test_images/guns.jpg
  Expected Output:
  DECISION: REJECT
  Reason: Unsafe content detected: violence=0.01, weapons=0.87
  Explain: CLIP detected weapons with 87% confidence, image blocked.

  ---
  Demo 3: PII Document → REDACT

  uv run python image_guard.py test_images/pii_test.png
  Expected Output:
  DECISION: REDACT
  Reason: PII redacted: 4, Faces blurred: 0
  Explain: Document contains PII (name, email, SSN, credit card). PII is redacted, safe image returned.

  ---
  Demo 4: Text PII Anonymization

  uv run python image_guard.py --text "Contact John Smith at john@example.com, SSN: 123-45-6789, Card: 4111-1111-1111-1111" --anonymize
  Expected Output:
  Contact <PERSON> at <EMAIL_ADDRESS>, SSN: <US_SSN>, Card: <CREDIT_CARD>
  Explain: All PII replaced with type labels. Original data never reaches AI model.

  ---
  Demo 5: API - Image Scan

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

  ---
  Demo 6: API - Text Scan

  curl -s -X POST "http://localhost:8000/scan/text" \
    -H "Content-Type: application/json" \
    -d '{"input_text": "Call John at 555-123-4567 or email john@company.com"}' | python3 -m json.tool

  ---
  Demo 7: Show Swagger Docs

  Open browser: http://localhost:8000/docs

  ---
  Demo 8: Show Audit Log

  tail -5 guardrails_audit.log | python3 -m json.tool
  Explain: Every decision is logged for compliance and audit trail.

  ---
  Configuration Highlights

  # Thresholds (lower = stricter)
  nsfw_threshold: 0.80      # REJECT if NSFW score >= 0.80
  violence_threshold: 0.70  # REJECT if violence+weapons >= 0.70
  hate_symbol_threshold: 0.75

  # Features can be toggled
  enable_violence_check: true
  enable_pii_redaction: true
  enable_face_blur: true

  # Limits
  max_file_size_mb: 10
  max_resolution: 4096x4096

  ---
  Key Talking Points for Managers
  ┌──────────────┬──────────────────────────────────────────────┐
  │    Point     │                    Value                     │
  ├──────────────┼──────────────────────────────────────────────┤
  │ Security     │ Blocks unsafe content before AI processing   │
  ├──────────────┼──────────────────────────────────────────────┤
  │ Privacy      │ Auto-redacts PII (GDPR/HIPAA compliant)      │
  ├──────────────┼──────────────────────────────────────────────┤
  │ Local        │ 100% on-premise, no data leaves the network  │
  ├──────────────┼──────────────────────────────────────────────┤
  │ Auditable    │ Every decision logged with timestamp         │
  ├──────────────┼──────────────────────────────────────────────┤
  │ Configurable │ Thresholds adjustable per use case           │
  ├──────────────┼──────────────────────────────────────────────┤
  │ API Ready    │ REST API for easy integration                │
  ├──────────────┼──────────────────────────────────────────────┤
  │ Extensible   │ Add new detection categories via CLIP labels │
  └──────────────┴──────────────────────────────────────────────┘
  ---
  Memory & Performance
  ┌──────────────────┬───────┬────────┬───────────────────────┐
  │      Model       │ Size  │ Memory │        Purpose        │
  ├──────────────────┼───────┼────────┼───────────────────────┤
  │ OpenNSFW2        │ 24MB  │ ~200MB │ NSFW detection        │
  ├──────────────────┼───────┼────────┼───────────────────────┤
  │ CLIP             │ 605MB │ ~1GB   │ Violence/weapons/hate │
  ├──────────────────┼───────┼────────┼───────────────────────┤
  │ Presidio + spaCy │ 740MB │ ~1GB   │ PII detection         │
  ├──────────────────┼───────┼────────┼───────────────────────┤
  │ OpenCV           │ <1MB  │ ~50MB  │ Face detection        │
  └──────────────────┴───────┴────────┴───────────────────────┘
  Total: ~1.4GB disk, ~2GB RAM at runtime

  ---
  Summary

  Image Guardrails = Safety + Privacy + Compliance

  ✓ Blocks NSFW, violence, weapons, hate symbols
  ✓ Redacts PII (names, emails, SSN, credit cards)
  ✓ Blurs faces for privacy
  ✓ Strips EXIF metadata (GPS, device info)
  ✓ 100% local processing
  ✓ Full audit trail
  ✓ REST API ready

  ---