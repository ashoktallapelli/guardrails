# Image Guardrails - Tools Reference

A comprehensive reference for all tools used in the image guardrails pipeline.

---

## 1. NSFW Detection Models

Two NSFW detection models are supported. Configure in `config.yaml` with `nsfw_model`:

### Comparison

| Feature | OpenNSFW2 | AdamCodd |
|---------|-----------|----------|
| **Architecture** | ResNet-50 (CNN) | ViT (Transformer) |
| **Framework** | TensorFlow/Keras | PyTorch |
| **Accuracy** | ~90% | **96.54%** |
| **Model Size** | ~24MB | ~330MB |
| **Input Size** | 224x224 | 384x384 |
| **Speed** | Faster | Slower |
| **AI-generated** | Limited | Use `vit-nsfw-stable-diffusion` |

### 1a. OpenNSFW2 (Default)

| Property | Value |
|----------|-------|
| **Type** | Python library |
| **Modality** | Image only |
| **Task** | Image Classification |
| **Model** | ResNet-50 (CNN) |
| **Size** | ~24MB weights |
| **Source** | [GitHub - bhky/opennsfw2](https://github.com/bhky/opennsfw2) |

### Input Format

| Input Type | Format | Example |
|------------|--------|---------|
| File path | String | `"photo.jpg"` |
| PIL Image | PIL.Image.Image | `Image.open("photo.jpg")` |
| NumPy array | np.ndarray (224x224x3) | Preprocessed array |

### Output Format

| Output | Type | Range | Meaning |
|--------|------|-------|---------|
| NSFW Score | `float` | 0.0 - 1.0 | Probability of NSFW content |

### Score Interpretation

| Score Range | Meaning | Action |
|-------------|---------|--------|
| 0.00 - 0.30 | Safe | ALLOW |
| 0.30 - 0.80 | Questionable | REVIEW |
| 0.80 - 1.00 | NSFW (explicit) | REJECT |

### Code Example

```python
import opennsfw2 as n2
from PIL import Image

# Option 1: File path
score = n2.predict_image("photo.jpg")

# Option 2: PIL Image
img = Image.open("photo.jpg")
score = n2.predict_image(img)

print(f"NSFW Score: {score}")  # 0.0234

# Decision
if score >= 0.80:
    print("REJECT - NSFW content")
else:
    print("ALLOW - Safe content")
```

### Main Use Cases

| Use Case | Description |
|----------|-------------|
| Content moderation | Block explicit images in user uploads |
| AI safety | Filter images before sending to LLM |
| Platform safety | Prevent NSFW content on websites/apps |
| Compliance | Meet legal/policy requirements |

### 1b. AdamCodd/vit-base-nsfw-detector

| Property | Value |
|----------|-------|
| **Type** | Python library (transformers) |
| **Modality** | Image only |
| **Task** | Image Classification |
| **Model** | ViT-base-patch16-384 (Transformer) |
| **Size** | ~330MB |
| **Accuracy** | 96.54% |
| **Source** | [HuggingFace - AdamCodd/vit-base-nsfw-detector](https://huggingface.co/AdamCodd/vit-base-nsfw-detector) |

### Input/Output Format

| Input | Output |
|-------|--------|
| PIL Image | 2 classes: `normal`, `nsfw` with scores |

### Code Example

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# Load model
processor = AutoImageProcessor.from_pretrained("AdamCodd/vit-base-nsfw-detector")
model = AutoModelForImageClassification.from_pretrained("AdamCodd/vit-base-nsfw-detector")

# Process image
image = Image.open("photo.jpg")
inputs = processor(images=image, return_tensors="pt")

# Get prediction
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)

# Output: [normal, nsfw]
normal_score = probs[0][0].item()
nsfw_score = probs[0][1].item()

print(f"Normal: {normal_score:.4f}, NSFW: {nsfw_score:.4f}")

if nsfw_score >= 0.80:
    print("REJECT - NSFW content")
```

### Configuration

In `config.yaml`:

```yaml
# Use AdamCodd model (higher accuracy)
nsfw_model: "adamcodd"
nsfw_threshold: 0.80
```

### When to Use

| Use Case | Recommended Model |
|----------|-------------------|
| General photos | AdamCodd (higher accuracy) |
| AI-generated images | `AdamCodd/vit-nsfw-stable-diffusion` |
| Low memory / fast inference | OpenNSFW2 |
| Offline / no PyTorch | OpenNSFW2 |

---

## 2. CLIP (OpenAI)

### Overview

| Property | Value |
|----------|-------|
| **Type** | Python library (transformers) |
| **Modality** | Multimodal (Image + Text) |
| **Task** | Zero-Shot Image Classification |
| **Model** | ViT-B/32 Transformer |
| **Size** | ~605MB (pytorch_model.bin) |
| **Source** | [HuggingFace - openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) |

### Input Format

| Input Type | Format | Required |
|------------|--------|----------|
| Image | PIL Image | Yes |
| Text labels | List of strings | Yes |

### Output Format

| Output | Type | Range | Meaning |
|--------|------|-------|---------|
| Similarity scores | List of floats | 0.0 - 1.0 per label | How well image matches each label |

### How Scores Are Calculated (Softmax)

CLIP uses **softmax** - all scores are divided so they **sum to 1.0 (100%)**.

**Step-by-Step Process:**

```
Step 1: CLIP computes raw similarity (logits) for each label
        ┌──────────────┬───────────┐
        │ Label        │ Raw Score │
        ├──────────────┼───────────┤
        │ safe         │ 2.5       │
        │ violence     │ 0.8       │
        │ weapons      │ 0.5       │
        └──────────────┴───────────┘

Step 2: Softmax converts to probabilities (sum = 1.0)

        Formula: e^score / sum(e^all_scores)

        ┌──────────────┬───────────┐
        │ Label        │ Score     │
        ├──────────────┼───────────┤
        │ safe         │ 0.75      │
        │ violence     │ 0.15      │
        │ weapons      │ 0.10      │
        ├──────────────┼───────────┤
        │ TOTAL        │ 1.00      │
        └──────────────┴───────────┘
```

**Visual Example:**

```
Image of a normal landscape:

safe       ████████████████████████░░░░░░  0.80 (80%)
violence   ████░░░░░░░░░░░░░░░░░░░░░░░░░░  0.12 (12%)
weapons    ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.08 (8%)
           ─────────────────────────────
           TOTAL                          1.00 (100%)
```

```
Image with a gun:

safe       ████░░░░░░░░░░░░░░░░░░░░░░░░░░  0.15 (15%)
violence   ██████░░░░░░░░░░░░░░░░░░░░░░░░  0.20 (20%)
weapons    ████████████████████░░░░░░░░░░  0.65 (65%)
           ─────────────────────────────
           TOTAL                          1.00 (100%)
```

**Key Rules:**

| Rule | Description |
|------|-------------|
| **Sum = 1.0** | All scores always add up to 1.0 (100%) |
| **Relative** | Scores are compared against each other |
| **Winner takes more** | Highest match gets highest score |
| **More labels** | Each label gets smaller share |

**Important: Minimum Labels Required**

| Labels Provided | Result |
|-----------------|--------|
| 1 label | Always ~1.0 (useless) |
| 2+ labels | Meaningful comparison |

Always include a **"safe/normal"** baseline label for proper comparison.

### Code Example

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Load model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Input: Image + Text labels
image = Image.open("photo.jpg")
labels = [
    "a safe, normal photograph",
    "a document, text, screenshot, form, receipt",  # Prevents false positives on text images
    "violence, gore, blood, injury, fighting",
    "weapons, guns, knives, dangerous objects",
    "disturbing, graphic, shocking content"
]

# Process
inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

# Get scores
with torch.no_grad():
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).squeeze().tolist()

# Output
scores = {
    "safe": probs[0],       # 0.49
    "document": probs[1],   # 0.11 (high for text/document images)
    "violence": probs[2],   # 0.02
    "weapons": probs[3],    # 0.26
    "disturbing": probs[4]  # 0.12
}

# Decision (document label is treated as safe)
unsafe_score = scores["violence"] + scores["weapons"] + scores["disturbing"]
if unsafe_score >= 0.70:
    print("REJECT - Unsafe content")
else:
    print("ALLOW - Safe content")
```

### Document Label (False Positive Prevention)

The **document label** prevents false positives on text/document images:

| Image Type | document score | weapons score | Result |
|------------|----------------|---------------|--------|
| PII document (text) | **1.00** | 0.00 | ALLOW ✓ |
| Gun photo | 0.00 | **0.87** | REJECT ✓ |
| Normal photo | 0.11 | 0.26 | ALLOW ✓ |

**Why it works:** CLIP distributes probability across all labels (softmax). When the "document" label absorbs most probability for text images, the violence/weapons scores become negligible.

### Main Use Cases

| Use Case | Description |
|----------|-------------|
| Violence detection | Detect violent/graphic content |
| Weapons detection | Detect guns, knives, dangerous objects |
| Hate symbol detection | Detect extremist/racist imagery |
| Content categorization | Classify images by custom labels |
| Flexible classification | Add new categories without retraining |

### Why CLIP is Powerful

| Feature | Benefit |
|---------|---------|
| Zero-shot | No training needed - just describe what to detect |
| Flexible | Add new categories by adding text labels |
| Multi-purpose | One model for many detection tasks |
| Contrastive learning | Learns image-text relationships |

---

## 3. Presidio (PII Detection & Anonymization)

Presidio supports both **image** and **text** PII processing.

### 3a. Presidio Image Redactor

| Property | Value |
|----------|-------|
| **Type** | Python library |
| **Modality** | Image |
| **Task** | PII Detection & Redaction |
| **Size** | Depends on spaCy model (~740MB) |
| **Source** | [GitHub - microsoft/presidio](https://github.com/microsoft/presidio) |

### Input Format

| Input Type | Format |
|------------|--------|
| PIL Image | PIL.Image.Image |

### Output Format

| Output | Type | Description |
|--------|------|-------------|
| Redacted Image | PIL.Image.Image | Image with PII masked (black boxes) |

### Internal Process (How It Works)

Presidio takes image as input and outputs redacted image, but internally does multiple steps:

```
┌─────────────────┐
│  Input Image    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Tesseract OCR  │  ← Step 1: Extract text from image
│  (Image → Text) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Presidio       │  ← Step 2: Detect PII in extracted text
│  Analyzer       │     (uses spaCy NER + regex patterns)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Get bounding   │  ← Step 3: Find location of PII in image
│  boxes for PII  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Draw black     │  ← Step 4: Cover PII with black boxes
│  boxes over PII │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Output:        │
│  Redacted Image │
└─────────────────┘
```

### Visual Example

**Input Image:**
```
┌─────────────────────────┐
│ Name: John Smith        │
│ Email: john@example.com │
│ SSN: 123-45-6789        │
│ Phone: 555-123-4567     │
└─────────────────────────┘
```

**Output Image:**
```
┌─────────────────────────┐
│ Name: ██████████        │
│ Email: ████████████████ │
│ SSN: ███████████        │
│ Phone: ████████████     │
└─────────────────────────┘
```

### Dependencies

```
Presidio Image Redactor
        │
        ├── Tesseract OCR (extract text from image)
        │
        ├── Presidio Analyzer (detect PII in text)
        │       │
        │       └── spaCy en_core_web_lg (NER for names/entities)
        │
        └── PIL/Pillow (draw black boxes on image)
```

### Code Example

```python
from presidio_image_redactor import ImageRedactorEngine
from PIL import Image

# Input: Image
image = Image.open("id_card.jpg")

# Process: Detect & redact PII
engine = ImageRedactorEngine()
redacted = engine.redact(image)

# Output: Redacted image
redacted.save("id_card_redacted.jpg")
```

### PII Types Detected

| PII Type | Example |
|----------|---------|
| Names | John Smith |
| Email | john@example.com |
| Phone | +1-555-123-4567 |
| SSN | 123-45-6789 |
| Credit Card | 4111-1111-1111-1111 |
| Address | 123 Main St |
| Date of Birth | 01/15/1990 |

### Summary

| Question | Answer |
|----------|--------|
| Input | Image (PIL) |
| Output | Redacted Image (PIL) with black boxes |
| Internal steps | OCR → PII Detection → Draw black boxes |
| Dependencies | Tesseract + spaCy (~740MB) |

### 3b. Presidio Text Analyzer & Anonymizer

| Property | Value |
|----------|-------|
| **Type** | Python library |
| **Modality** | Text |
| **Task** | PII Detection & Anonymization |
| **Source** | [GitHub - microsoft/presidio](https://github.com/microsoft/presidio) |

### Input/Output Format

| Mode | Input | Output |
|------|-------|--------|
| Detection | Text string | List of PII entities with scores |
| Anonymization | Text string | Anonymized text with replacements |

### Code Example

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

text = "Contact John at john@example.com or 555-123-4567"

# Detection only
analyzer = AnalyzerEngine()
results = analyzer.analyze(text=text, language="en", score_threshold=0.35)

for r in results:
    print(f"{r.entity_type}: {text[r.start:r.end]} (score: {r.score})")
# PERSON: John (score: 0.85)
# EMAIL_ADDRESS: john@example.com (score: 1.0)
# PHONE_NUMBER: 555-123-4567 (score: 0.4)

# Anonymization
anonymizer = AnonymizerEngine()
result = anonymizer.anonymize(text=text, analyzer_results=results)
print(result.text)
# "Contact <PERSON> at <EMAIL_ADDRESS> or <PHONE_NUMBER>"
```

### Anonymization Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `replace` | Replace with label | `John` → `<PERSON>` |
| `redact` | Remove entirely | `John` → `` |
| `mask` | Mask with characters | `John` → `****` |
| `hash` | Replace with hash | `John` → `a1b2c3d4` |

### Configuration in config.yaml

```yaml
enable_text_pii: true
pii_language: "en"
pii_score_threshold: 0.35  # Lower = catch more (phone numbers score ~0.4)
pii_operator: "replace"    # replace, redact, mask, hash
```

---

## 4. Tesseract OCR

### Overview

| Property | Value |
|----------|-------|
| **Type** | System library + Python wrapper |
| **Modality** | Image |
| **Task** | Image-to-Text |
| **Size** | ~22MB (eng.traineddata) |
| **Source** | [GitHub - tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract) |

### Input Format

| Input Type | Format |
|------------|--------|
| PIL Image | PIL.Image.Image |
| File path | String |

### Output Format

| Output | Type | Description |
|--------|------|-------------|
| Extracted text | String | All text found in image |

### Code Example

```python
import pytesseract
from PIL import Image

# Extract text
image = Image.open("screenshot.png")
text = pytesseract.image_to_string(image)

print(text)
# "Hello World
#  Email: john@example.com
#  Phone: 555-1234"
```

### Main Use Cases

| Use Case | Description |
|----------|-------------|
| Text extraction | Get text from images/screenshots |
| PII detection | Extract text for PII scanning |
| Prompt injection | Extract text to check for injection patterns |
| Document processing | Digitize scanned documents |

---

## 5. OpenCV

### Overview

| Property | Value |
|----------|-------|
| **Type** | Python library |
| **Modality** | Image |
| **Task** | Object Detection (Face Detection) |
| **Size** | <1MB (Haar cascade XML) |
| **Source** | [GitHub - opencv/opencv](https://github.com/opencv/opencv) |

### Input Format

| Input Type | Format |
|------------|--------|
| NumPy array | np.ndarray (BGR format) |
| PIL Image | Convert to NumPy first |

### Output Format

| Output | Type | Description |
|--------|------|-------------|
| Face bounding boxes | List of (x, y, w, h) | Coordinates of detected faces |

### Code Example

```python
import cv2
import numpy as np
from PIL import Image

# Load image
pil_img = Image.open("photo.jpg")
cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# Load face detector
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Detect faces
gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# Blur each face
for (x, y, w, h) in faces:
    roi = cv_img[y:y+h, x:x+w]
    roi = cv2.GaussianBlur(roi, (51, 51), 30)
    cv_img[y:y+h, x:x+w] = roi

print(f"Blurred {len(faces)} face(s)")
```

### Main Use Cases

| Use Case | Description |
|----------|-------------|
| Face blur | Anonymize faces for privacy |
| GDPR compliance | Protect identity in images |
| Face detection | Count/locate faces in images |

---

## 6. python-magic (libmagic)

### Overview

| Property | Value |
|----------|-------|
| **Type** | Python library + System library |
| **Modality** | File bytes |
| **Task** | File Type Validation |
| **Size** | <1MB |
| **Source** | [GitHub - ahupp/python-magic](https://github.com/ahupp/python-magic) |

### Input Format

| Input Type | Format |
|------------|--------|
| File path | String |
| Bytes | bytes |

### Output Format

| Output | Type | Description |
|--------|------|-------------|
| MIME type | String | e.g., "image/jpeg" |

### Code Example

```python
import magic

# From file
mime = magic.from_file("photo.jpg", mime=True)
print(mime)  # "image/jpeg"

# From bytes
with open("photo.jpg", "rb") as f:
    data = f.read()
mime = magic.from_buffer(data, mime=True)
print(mime)  # "image/jpeg"

# Validate
allowed = ["image/jpeg", "image/png", "image/webp"]
if mime not in allowed:
    print("REJECT - Invalid file type")
```

### Why Use Magic Bytes?

| Method | Security |
|--------|----------|
| File extension | ❌ Easily spoofed (rename .exe to .jpg) |
| Magic bytes | ✅ Checks actual file content |

---

## 7. BLIP (Salesforce)

### Overview

| Property | Value |
|----------|-------|
| **Type** | Python library (transformers) |
| **Modality** | Multimodal (Image + Text) |
| **Task** | Visual Question Answering / Image Captioning |
| **Size** | ~990MB (pytorch_model.bin) |
| **Source** | [HuggingFace - Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base) |

### Input Format

| Input Type | Format | Required |
|------------|--------|----------|
| Image | PIL Image | Yes |
| Question | String (optional) | For Q&A mode |

### Output Format

| Output | Type | Description |
|--------|------|-------------|
| Caption/Answer | String | Generated text description |

### Code Example

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

image = Image.open("photo.jpg")

# Image captioning
inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
print(caption)  # "a dog playing in the park"

# Visual Q&A
question = "What color is the dog?"
inputs = processor(image, question, return_tensors="pt")
out = model.generate(**inputs)
answer = processor.decode(out[0], skip_special_tokens=True)
print(answer)  # "brown"
```

### Main Use Cases

| Use Case | Description |
|----------|-------------|
| Image captioning | Generate descriptions of images |
| Visual Q&A | Answer questions about images |
| Accessibility | Describe images for visually impaired |
| Content understanding | Understand image content before processing |

---

## 8. spaCy (en_core_web_lg)

### Overview

| Property | Value |
|----------|-------|
| **Type** | Python library |
| **Modality** | Text |
| **Task** | Token Classification (NER) |
| **Size** | ~740MB |
| **Source** | [GitHub - explosion/spacy-models](https://github.com/explosion/spacy-models) |

### Input Format

| Input Type | Format |
|------------|--------|
| Text | String |

### Output Format

| Output | Type | Description |
|--------|------|-------------|
| Entities | List of (text, label) | Named entities found |

### Code Example

```python
import spacy

nlp = spacy.load("en_core_web_lg")

text = "John Smith works at Microsoft in Seattle. His email is john@microsoft.com"
doc = nlp(text)

for ent in doc.ents:
    print(f"{ent.text} -> {ent.label_}")

# Output:
# John Smith -> PERSON
# Microsoft -> ORG
# Seattle -> GPE
```

### Entity Types

| Label | Description |
|-------|-------------|
| PERSON | People names |
| ORG | Organizations |
| GPE | Countries, cities |
| DATE | Dates |
| MONEY | Monetary values |

---

## 9. imagehash (Perceptual Hashing)

### Overview

| Property | Value |
|----------|-------|
| **Type** | Python library |
| **Modality** | Image |
| **Task** | Image Fingerprinting |
| **Size** | <1MB |
| **Source** | [GitHub - JohannesBuchner/imagehash](https://github.com/JohannesBuchner/imagehash) |

### Input Format

| Input Type | Format |
|------------|--------|
| PIL Image | PIL.Image.Image |

### Output Format

| Output | Type | Description |
|--------|------|-------------|
| Hash | String (hex) | 16-character perceptual hash |

### How It Works

```
Input Image (any size)
        │
        ▼
┌─────────────────┐
│  Resize to 8x8  │  ← Normalize size
└────────┬────────┘
        │
        ▼
┌─────────────────┐
│  Convert to     │  ← Remove color info
│  Grayscale      │
└────────┬────────┘
        │
        ▼
┌─────────────────┐
│  Compute avg    │  ← Calculate average pixel value
│  pixel value    │
└────────┬────────┘
        │
        ▼
┌─────────────────┐
│  Create 64-bit  │  ← Each pixel: 1 if > avg, 0 if < avg
│  binary hash    │
└────────┬────────┘
        │
        ▼
┌─────────────────┐
│  Output:        │  ← Convert to hex string
│  "8fc7ce862cfe" │
└─────────────────┘
```

### Code Example

```python
import imagehash
from PIL import Image

# Compute hash
img = Image.open("photo.jpg")
phash = str(imagehash.average_hash(img))
print(phash)  # "8fc7ce862cfeefe7"

# Compare two images (Hamming distance)
hash1 = imagehash.average_hash(Image.open("photo1.jpg"))
hash2 = imagehash.average_hash(Image.open("photo2.jpg"))

distance = hash1 - hash2
print(f"Difference: {distance}")  # 0 = identical, >10 = different

# Check against blocklist
BANNED_HASHES = {"8fc7ce862cfeefe7", "abcd1234efgh5678"}
if phash in BANNED_HASHES:
    print("BLOCKED: Known bad content!")
```

### Main Use Cases

| Use Case | Description |
|----------|-------------|
| Known-bad matching | Block images matching a blocklist of hashes |
| Duplicate detection | Find similar/duplicate images |
| Content tracking | Track image spread across platforms |
| Audit logging | Record fingerprint for later analysis |

### Hash Comparison

| Distance | Meaning |
|----------|---------|
| 0 | Identical images |
| 1-5 | Minor modifications (resize, crop, compress) |
| 6-10 | Moderate changes |
| >10 | Different images |

---

## Quick Comparison

| Tool | Input | Output | Main Use |
|------|-------|--------|----------|
| **OpenNSFW2** | Image | Float (0-1) | NSFW detection |
| **AdamCodd ViT** | Image | Float (0-1) | NSFW detection (96% accuracy) |
| **CLIP** | Image + Text | Scores per label | Violence/weapons/hate/document detection |
| **Presidio Image** | Image | Redacted image | PII masking in images |
| **Presidio Text** | Text | Entities / Anonymized text | PII detection/anonymization |
| **Tesseract** | Image | Text | OCR extraction |
| **OpenCV** | Image | Bounding boxes | Face detection/blur |
| **python-magic** | File | MIME type | File validation |
| **BLIP** | Image + Text | Text | Image Q&A |
| **spaCy** | Text | Entities | NER/PII detection |
| **imagehash** | Image | Hash string | Known-bad matching |
| **FastAPI** | HTTP requests | JSON (with reason) | REST API (/scan/*) |

---

## Threshold Configuration

All thresholds are configured in **`config.yaml`**.

### Detection Thresholds

| Check | Config Key | Default | Decision Logic |
|-------|------------|---------|----------------|
| **NSFW** | `nsfw_threshold` | `0.80` | REJECT if score ≥ 0.80 |
| **Violence** | `violence_threshold` | `0.70` | REJECT if (violence + weapons + disturbing) ≥ 0.70 |
| **Hate Symbols** | `hate_symbol_threshold` | `0.75` | REJECT if (hate + nazi + racist) ≥ 0.75 |
| **PII** | `pii_score_threshold` | `0.35` | Detect entities with confidence ≥ 0.35 |

### File Validation Limits

| Check | Config Key | Default | Decision Logic |
|-------|------------|---------|----------------|
| **File Size** | `max_file_size_mb` | `10` | REJECT if > 10MB |
| **Width** | `max_resolution.width` | `4096` | REJECT if > 4096px |
| **Height** | `max_resolution.height` | `4096` | REJECT if > 4096px |
| **MIME Types** | `allowed_mime_types` | jpeg, png, webp, gif | REJECT if not in list |

### Enable/Disable Features

| Feature | Config Key | Default |
|---------|------------|---------|
| Violence check | `enable_violence_check` | `true` |
| Hate symbol check | `enable_hate_symbol_check` | `true` |
| PII redaction | `enable_pii_redaction` | `true` |
| Face blur | `enable_face_blur` | `true` |

### Tiered Response Thresholds (demo_app.py)

| Action | Config Key | Default | Logic |
|--------|------------|---------|-------|
| Auto ALLOW | `thresholds.auto_allow_below` | `0.3` | NSFW < 0.3 |
| Manual Review | `thresholds.manual_review_above` | `0.5` | 0.5 ≤ NSFW < 0.8 |
| Auto REJECT | `thresholds.auto_reject_above` | `0.8` | NSFW ≥ 0.8 |

### Full config.yaml Example

```yaml
# File validation
allowed_mime_types:
  - "image/jpeg"
  - "image/png"
  - "image/webp"
  - "image/gif"

max_file_size_mb: 10

max_resolution:
  width: 4096
  height: 4096

# NSFW Detection
nsfw_model: "opennsfw2"  # or "adamcodd" for higher accuracy
nsfw_threshold: 0.80

# Violence/Safety Detection
enable_violence_check: true
violence_threshold: 0.70

# Hate Symbol Detection
enable_hate_symbol_check: true
hate_symbol_threshold: 0.75

# Privacy features - Image
enable_pii_redaction: true
enable_face_blur: true
face_blur_kernel_size: 51

# Privacy features - Text PII
enable_text_pii: true
pii_language: "en"
pii_score_threshold: 0.35  # Lower to catch phone numbers (~0.4)
pii_operator: "replace"    # replace, redact, mask, hash

# Output settings
output_quality: 95

# Logging
log_level: "INFO"
log_decisions_to_file: true
log_file_path: "guardrails_audit.log"
```

### Adjusting Thresholds

**More Strict (catch more unsafe content):**
```yaml
nsfw_threshold: 0.50           # Lower = stricter
violence_threshold: 0.40       # Catches knives better
hate_symbol_threshold: 0.50    # More sensitive
```

**More Lenient (fewer false positives):**
```yaml
nsfw_threshold: 0.90           # Higher = more lenient
violence_threshold: 0.85
hate_symbol_threshold: 0.85
```

### Checks WITHOUT Thresholds (Info Only)

| Check | Output | Purpose |
|-------|--------|---------|
| **PII Detection** | Entity list | Information only - no REJECT |
| **Face Detection** | Face count + bounding boxes | Information only - no REJECT |
| **Perceptual Hash** | Hash string | For blocklist matching |

---

## Memory Requirements

| Tool | Download Size | Runtime Memory |
|------|---------------|----------------|
| OpenNSFW2 | ~24MB | ~200MB |
| CLIP | ~605MB | ~1GB |
| Presidio + spaCy | ~740MB | ~1GB |
| Tesseract | ~22MB | ~100MB |
| OpenCV | <1MB | ~50MB |
| BLIP | ~990MB | ~2GB |
| imagehash | <1MB | ~10MB |
| python-magic | <1MB | ~5MB |

---

## Analyze-Only Mode Summary

### Models USED in Analyze-Only Mode

| Tool | Purpose | Output |
|------|---------|--------|
| **libmagic** | File type validation | MIME type |
| **Pillow** | Image loading | PIL Image |
| **OpenNSFW2** | NSFW detection | Score 0-1 |
| **CLIP** | Violence/weapons/hate detection | Scores per label |
| **Tesseract** | OCR (for PII detection) | Extracted text |
| **spaCy + Presidio** | PII entity detection | Entity list |
| **OpenCV** | Face detection | Face count + boxes |
| **imagehash** | Perceptual hash | Hash string |

### Features NOT USED in Analyze-Only Mode

| Feature | Framework | Reason |
|---------|-----------|--------|
| EXIF stripping | Pillow/piexif | No image modification |
| PII redaction | Presidio ImageRedactor | Detection only |
| Face blur | OpenCV GaussianBlur | Detection only |
| Image saving | Pillow | No output file |
| BLIP Q&A | Salesforce BLIP | Separate script |

---

## 10. FastAPI REST API

### Overview

| Property | Value |
|----------|-------|
| **File** | `api.py` |
| **Framework** | FastAPI |
| **Port** | 8000 (default) |
| **Standard** | Aligned with AIForce SGS |

### Start Server

```bash
uv run uvicorn api:app --reload --port 8000
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/check_health` | Health check and model status |
| `GET` | `/config` | View current configuration |
| `POST` | `/scan/image` | Scan image (analyze + auto sanitize) |
| `POST` | `/scan/text` | Scan text for PII (analyze + auto anonymize) |

### Decision States

| Decision | Meaning | Image Returned |
|----------|---------|----------------|
| **ALLOW** | Safe, no redaction needed | ✅ Original (EXIF stripped) |
| **REDACT** | Safe, PII/faces found and redacted | ✅ Sanitized image |
| **REJECT** | Unsafe content detected | ❌ No image |

### Example Requests

**Health Check:**
```bash
curl http://localhost:8000/check_health
```

**Scan Image:**
```bash
curl -X POST "http://localhost:8000/scan/image" \
  -F "file=@test_images/sample.jpg"
```

**Scan Text:**
```bash
curl -X POST "http://localhost:8000/scan/text" \
  -H "Content-Type: application/json" \
  -d '{"input_text": "Contact John at john@example.com or 555-123-4567"}'
```

### Response Format (scan/image)

```json
{
  "decision": "REDACT",
  "reason": "PII redacted: 4, Faces blurred: 0",
  "is_safe": true,
  "is_redacted": true,
  "results": {
    "nsfw": {"score": 0.001, "threshold": 0.8, "is_pass": true},
    "violence": {"score": 0.001, "threshold": 0.7, "is_pass": true},
    "hate_symbols": {"score": 0.0003, "threshold": 0.75, "is_pass": true},
    "pii": {"score": 4.0, "threshold": 0, "is_pass": true},
    "faces": {"score": 0.0, "threshold": 0, "is_pass": true}
  },
  "sanitized_image_base64": "...",
  "meta": {
    "sha256": "c7deaf...",
    "perceptual_hash": "0000000000000000",
    "processing_ms": 8500,
    "filename": "pii_test.png"
  }
}
```

### Response Format (scan/text)

```json
{
  "decision": "REDACT",
  "reason": "PII anonymized: 4 entities (PHONE_NUMBER, PERSON, EMAIL_ADDRESS, URL)",
  "is_safe": true,
  "is_redacted": true,
  "results": {
    "pii": {"score": 4.0, "threshold": 0.35, "is_pass": true}
  },
  "sanitized_text": "Contact <PERSON> at <EMAIL_ADDRESS> or <PHONE_NUMBER>",
  "entities": [
    {"type": "PERSON", "original_text": "John", "score": 0.85},
    {"type": "EMAIL_ADDRESS", "original_text": "john@example.com", "score": 1.0},
    {"type": "PHONE_NUMBER", "original_text": "555-123-4567", "score": 0.4}
  ]
}
```

### Reason Formats (aligned with CLI)

| Decision | Reason Format |
|----------|---------------|
| REJECT (NSFW) | `NSFW score {score} >= threshold {threshold}` |
| REJECT (Violence) | `Unsafe content detected: violence={v}, weapons={w}` |
| REJECT (Hate) | `Hate symbols detected: combined_score={score}` |
| REDACT | `PII redacted: {count}, Faces blurred: {count}` |
| ALLOW | `All checks passed, no redaction needed` |

---

## 11. Audit Logging

### Overview

All decisions are logged to a JSON-lines file for audit trail.

### Configuration

```yaml
log_decisions_to_file: true
log_file_path: "guardrails_audit.log"
log_level: "INFO"
```

### Log Format (Image)

Each line is a JSON object:

```json
{
  "timestamp": "2026-03-04T10:30:00.000000+00:00",
  "input_type": "image",
  "decision": "ALLOW",
  "is_safe": true,
  "input_path": "test_images/sample.jpg",
  "sha256": "741ba009...",
  "mime_type": "image/jpeg",
  "nsfw_score": 0.0009,
  "violence_scores": {"safe": 0.56, "document": 0.11, "violence": 0.02, "weapons": 0.18},
  "hate_scores": {"safe": 0.35, "document": 0.04, "combined_hate_score": 0.61},
  "reasons": ["All checks passed, no redaction needed"]
}
```

### Decision Values in Logs

| Decision | is_safe | Meaning |
|----------|---------|---------|
| ALLOW | true | Safe, no redaction needed |
| REDACT | true | Safe, PII/faces redacted |
| REJECT | false | Unsafe content blocked |

### Text PII Log Format

```json
{
  "timestamp": "2026-03-04T10:30:00.000000+00:00",
  "input_type": "text",
  "decision": "ALLOW",
  "is_safe": true,
  "text_length": 48,
  "pii_count": 4,
  "anonymized": true
}
```

---

## 12. Output Structure

### Folder Structure (Decision-Based)

```
output/
├── allow/                    # Safe images, no redaction
│   ├── b803f7a859c3.jpg
│   └── ...
├── redact/                   # Safe images with PII/faces redacted
│   ├── c52692b99170.jpg
│   └── ...
└── (rejected images not saved)
```

### Decision-Based Organization

| Decision | Folder | Contents |
|----------|--------|----------|
| ALLOW | `output/allow/` | Original images (EXIF stripped only) |
| REDACT | `output/redact/` | Sanitized images (PII/faces redacted) |
| REJECT | (not saved) | Unsafe content - no output |

### JSON Results (--save-json)

```bash
uv run python image_guard.py test.jpg --save-json /tmp/results
```

Creates:
```
/tmp/results/
└── c52692b99170.json
```

### Why UUID Filenames?

| Reason | Benefit |
|--------|---------|
| Unbiased | No info about content in filename |
| Unique | No collisions |
| Traceable | ID links to JSON metadata |

---

## PII Score Threshold

### Default: 0.35

The `pii_score_threshold` controls minimum confidence for PII detection.

| Entity | Typical Score | Detected at 0.35? | Detected at 0.5? |
|--------|---------------|-------------------|------------------|
| Email | 1.0 | ✅ Yes | ✅ Yes |
| Person | 0.85 | ✅ Yes | ✅ Yes |
| Phone | 0.4 | ✅ Yes | ❌ No |
| URL | 0.5 | ✅ Yes | ✅ Yes |

### Configuration

```yaml
pii_score_threshold: 0.35  # Catches phone numbers (score ~0.4)
```

### Functions Using This Threshold

| Function | Purpose |
|----------|---------|
| `detect_pii()` | Image PII detection |
| `redact_pii()` | Image PII redaction |
| `detect_text_pii()` | Text PII detection |
| `anonymize_text_pii()` | Text PII anonymization |

---

## Changelog

### March 2026 (Latest)

**CLIP Document Label Fix:**
- Added `"a document, text, screenshot, form, receipt"` label to CLIP classification
- Prevents false positives on text/document images (PII documents, receipts, forms)
- Document images now get high "document" score, reducing violence/weapons false positives

**Three-Decision Model:**
- Added REDACT decision alongside ALLOW and REJECT
- ALLOW: Safe, no redaction needed
- REDACT: Safe, but PII/faces found and automatically redacted
- REJECT: Unsafe content (NSFW, violence, hate symbols)

**API Alignment (SGS Standard):**
- Renamed endpoints: `/scan/image`, `/scan/text`, `/check_health`
- Added `reason` field to all responses (aligned with CLI output)
- Response includes `decision`, `reason`, `is_safe`, `is_redacted`, `results`

**Output Organization:**
- Images now saved to decision-based folders: `output/allow/`, `output/redact/`
- REJECT images are not saved (unsafe content)

**Reason Formats:**
- NSFW: `NSFW score {score} >= threshold {threshold}`
- Violence: `Unsafe content detected: violence={v}, weapons={w}`
- Hate: `Hate symbols detected: combined_score={score}`
- REDACT: `PII redacted: {count}, Faces blurred: {count}`
- ALLOW: `All checks passed, no redaction needed`

---

*Last updated: March 2026*
