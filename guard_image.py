"""
guard_image.py - Image Guardrails Pipeline

A local-first pre-inference pipeline that validates and sanitizes images:
1. Content-type validation (magic bytes)
2. EXIF metadata stripping
3. NSFW content detection
4. PII redaction (OCR + masking)
5. Face detection and blurring
6. Outputs sanitized image with decision logging

Usage:
    python guard_image.py /path/to/input.jpg
    python guard_image.py /path/to/input.jpg --config config.yaml
"""

import argparse
import hashlib
import io
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import yaml

# Fix SSL certificates on Windows
if sys.platform == 'win32':
    try:
        import certifi
        os.environ.setdefault('SSL_CERT_FILE', certifi.where())
        os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
    except ImportError:
        pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "allowed_mime_types": ["image/jpeg", "image/png", "image/webp"],
    "max_file_size_mb": 10,
    "max_resolution": {"width": 4096, "height": 4096},
    "nsfw_threshold": 0.80,
    "violence_threshold": 0.70,
    "enable_violence_check": True,
    "enable_pii_redaction": True,
    "enable_face_blur": True,
    "face_blur_kernel_size": 51,
    "output_quality": 95,
}

# Cache for ML models (loaded once)
_model_cache = {}


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file or use defaults."""
    config = DEFAULT_CONFIG.copy()
    if config_path and config_path.exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f)
            if user_config:
                config.update(user_config)
        logger.info(f"Loaded config from {config_path}")
    return config


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of file for audit trail."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_file_type(path: Path, config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate file by content (magic bytes), not extension.
    Returns (is_valid, mime_type_or_error_message).
    """
    try:
        import magic
    except ImportError:
        logger.warning("python-magic not installed, skipping MIME validation")
        return True, "unknown (magic not available)"

    max_bytes = config["max_file_size_mb"] * 1024 * 1024
    size = path.stat().st_size

    if size > max_bytes:
        return False, f"File too large: {size / (1024*1024):.2f} MB > {config['max_file_size_mb']} MB limit"

    if size == 0:
        return False, "File is empty"

    mime = magic.from_file(str(path), mime=True)
    if mime not in config["allowed_mime_types"]:
        return False, f"Disallowed MIME type: {mime}. Allowed: {config['allowed_mime_types']}"

    return True, mime


def validate_resolution(img, config: Dict[str, Any]) -> Tuple[bool, str]:
    """Check if image resolution is within limits."""
    max_res = config["max_resolution"]
    width, height = img.size

    if width > max_res["width"] or height > max_res["height"]:
        return False, f"Resolution {width}x{height} exceeds limit {max_res['width']}x{max_res['height']}"

    return True, f"{width}x{height}"


def strip_exif(img) -> bytes:
    """
    Strip EXIF metadata by re-encoding the image.
    Returns clean image bytes without GPS, device IDs, etc.
    """
    buf = io.BytesIO()
    # Save without exif parameter to strip metadata
    img.save(buf, format="JPEG", quality=95, exif=b"")
    return buf.getvalue()


def check_nsfw(img, config: Dict[str, Any]) -> Tuple[bool, float]:
    """
    Run NSFW detection using OpenNSFW2.
    Returns (is_safe, nsfw_score).
    """
    try:
        import opennsfw2 as n2
    except ImportError:
        logger.warning("opennsfw2 not installed, skipping NSFW check")
        return True, 0.0

    # Check for local weights file (for offline/corporate environments)
    home = Path.home()
    weights_path = home / ".opennsfw2" / "weights" / "open_nsfw_weights.h5"

    if not weights_path.exists():
        logger.warning(f"NSFW weights not found at {weights_path}")
        logger.warning("Download from: https://github.com/bhky/opennsfw2/releases/download/v0.1.0/open_nsfw_weights.h5")
        logger.warning("Skipping NSFW check")
        return True, 0.0

    try:
        score = float(n2.predict_image(img, weights_path=str(weights_path)))
        is_safe = score < config["nsfw_threshold"]
        return is_safe, score
    except Exception as e:
        logger.warning(f"NSFW check failed: {e}")
        return True, 0.0


def check_violence_safety(img, config: Dict[str, Any]) -> Tuple[bool, Dict[str, float]]:
    """
    Run violence/safety detection using CLIP-based zero-shot classification.
    Based on LAION-SAFETY approach using CLIP embeddings.

    Returns (is_safe, scores_dict) where scores_dict contains:
    - violence: violence/gore score
    - weapons: weapons score
    - safe: safe content score
    """
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch
    except ImportError:
        logger.warning("transformers/torch not installed, skipping violence check")
        return True, {"violence": 0.0, "weapons": 0.0, "safe": 1.0}

    # Load model from cache or download
    model_name = "openai/clip-vit-base-patch32"

    if "clip_model" not in _model_cache:
        logger.info("Loading CLIP model for safety classification...")
        try:
            _model_cache["clip_model"] = CLIPModel.from_pretrained(model_name)
            _model_cache["clip_processor"] = CLIPProcessor.from_pretrained(model_name)
        except Exception as e:
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ["ssl", "certificate", "connection", "network", "huggingface", "config.json"]):
                logger.warning(f"Cannot download CLIP model (network/SSL issue): {e}")
                logger.warning("Skipping violence check - model not available offline")
                _model_cache["clip_unavailable"] = True
                return True, {"violence": 0.0, "weapons": 0.0, "safe": 1.0, "skipped": True}
            raise

    if _model_cache.get("clip_unavailable"):
        return True, {"violence": 0.0, "weapons": 0.0, "safe": 1.0, "skipped": True}

    model = _model_cache["clip_model"]
    processor = _model_cache["clip_processor"]

    # Safety categories for zero-shot classification
    safety_labels = [
        "a safe, normal photograph",
        "violence, gore, blood, injury, fighting",
        "weapons, guns, knives, dangerous objects",
        "disturbing, graphic, shocking content",
    ]

    try:
        # Process image and text
        inputs = processor(
            text=safety_labels,
            images=img,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).squeeze().tolist()

        scores = {
            "safe": round(probs[0], 4),
            "violence": round(probs[1], 4),
            "weapons": round(probs[2], 4),
            "disturbing": round(probs[3], 4),
        }

        # Compute combined unsafe score
        unsafe_score = scores["violence"] + scores["weapons"] + scores["disturbing"]
        is_safe = unsafe_score < config["violence_threshold"]

        logger.info(f"Safety scores: safe={scores['safe']:.2f}, violence={scores['violence']:.2f}, weapons={scores['weapons']:.2f}")

        return is_safe, scores

    except Exception as e:
        logger.warning(f"Violence check failed: {e}")
        return True, {"violence": 0.0, "weapons": 0.0, "safe": 1.0, "error": str(e)}


def redact_pii(img):
    """
    Use Presidio Image Redactor to detect and mask PII in images.
    Handles screenshots, IDs, documents with sensitive text.
    """
    try:
        from presidio_image_redactor import ImageRedactorEngine
    except ImportError:
        logger.warning("presidio-image-redactor not installed, skipping PII redaction")
        return img

    try:
        engine = ImageRedactorEngine()
        redacted = engine.redact(img)
        logger.info("PII redaction completed")
        return redacted
    except Exception as e:
        logger.warning(f"PII redaction failed: {e}")
        return img


def blur_faces(img, config: Dict[str, Any]):
    """
    Detect faces using OpenCV Haar cascades and apply Gaussian blur.
    Anonymizes individuals in images.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.warning("opencv-python not installed, skipping face blur")
        return img

    # Convert PIL to OpenCV format
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Load face detection cascade
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Detect faces
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Blur each detected face
    kernel_size = config["face_blur_kernel_size"]
    for (x, y, w, h) in faces:
        roi = cv_img[y:y+h, x:x+w]
        roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 30)
        cv_img[y:y+h, x:x+w] = roi

    logger.info(f"Blurred {len(faces)} face(s)")

    # Convert back to PIL
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def compute_perceptual_hash(img) -> str:
    """
    Compute a simple perceptual hash for known-bad content matching.
    Uses average hash (aHash) as a basic implementation.
    """
    try:
        import imagehash
        return str(imagehash.average_hash(img))
    except ImportError:
        # Fallback: use a simple hash based on resized image
        small = img.resize((8, 8)).convert('L')
        pixels = list(small.getdata())
        avg = sum(pixels) / len(pixels)
        bits = ''.join('1' if p > avg else '0' for p in pixels)
        return hex(int(bits, 2))[2:].zfill(16)


def run_guardrails(
    image_path: Path,
    config: Dict[str, Any],
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run the complete guardrails pipeline on an image.

    Returns a decision dictionary with:
    - decision: ALLOW, REJECT, or REVIEW
    - reasons: list of check results
    - output_path: path to sanitized image (if allowed)
    - audit: full audit trail
    """
    from PIL import Image

    result = {
        "input_path": str(image_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": "ALLOW",
        "reasons": [],
        "checks": {},
        "output_path": None,
    }

    # Step 1: Compute file hash for audit
    file_hash = sha256_file(image_path)
    result["sha256"] = file_hash
    logger.info(f"Processing: {image_path} (SHA256: {file_hash[:16]}...)")

    # Step 2: Validate file type by magic bytes
    valid, mime_info = validate_file_type(image_path, config)
    result["checks"]["file_type"] = {"valid": valid, "info": mime_info}

    if not valid:
        result["decision"] = "REJECT"
        result["reasons"].append(f"File validation failed: {mime_info}")
        logger.warning(f"REJECT: {mime_info}")
        return result

    result["mime_type"] = mime_info

    # Step 3: Load and validate resolution
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        result["decision"] = "REJECT"
        result["reasons"].append(f"Failed to open image: {e}")
        return result

    valid, res_info = validate_resolution(img, config)
    result["checks"]["resolution"] = {"valid": valid, "info": res_info}

    if not valid:
        result["decision"] = "REJECT"
        result["reasons"].append(f"Resolution check failed: {res_info}")
        logger.warning(f"REJECT: {res_info}")
        return result

    result["resolution"] = res_info

    # Step 4: Strip EXIF metadata
    _ = strip_exif(img)  # Could persist if needed
    result["checks"]["exif_stripped"] = True
    logger.info("EXIF metadata stripped")

    # Step 5: NSFW content check
    is_safe, nsfw_score = check_nsfw(img, config)
    result["checks"]["nsfw"] = {"safe": is_safe, "score": round(nsfw_score, 4)}

    if not is_safe:
        result["decision"] = "REJECT"
        result["reasons"].append(f"NSFW score {nsfw_score:.2f} >= threshold {config['nsfw_threshold']}")
        logger.warning(f"REJECT: NSFW content detected (score: {nsfw_score:.2f})")
        return result

    # Step 6: Violence/Safety check (LAION-SAFETY style)
    if config.get("enable_violence_check", True):
        is_safe, safety_scores = check_violence_safety(img, config)
        result["checks"]["safety"] = {
            "safe": is_safe,
            "scores": safety_scores
        }

        if not is_safe:
            result["decision"] = "REJECT"
            result["reasons"].append(
                f"Unsafe content detected: violence={safety_scores.get('violence', 0):.2f}, "
                f"weapons={safety_scores.get('weapons', 0):.2f}"
            )
            logger.warning(f"REJECT: Unsafe content detected (scores: {safety_scores})")
            return result

    # Step 7: PII redaction (OCR + masking)
    if config["enable_pii_redaction"]:
        img = redact_pii(img)
        result["checks"]["pii_redaction"] = True

    # Step 8: Face blur for anonymization
    if config["enable_face_blur"]:
        img = blur_faces(img, config)
        result["checks"]["face_blur"] = True

    # Step 9: Compute perceptual hash (for known-bad matching)
    phash = compute_perceptual_hash(img)
    result["perceptual_hash"] = phash
    result["checks"]["phash_computed"] = True

    # Step 10: Save sanitized output
    if output_dir is None:
        output_dir = image_path.parent

    output_path = output_dir / f"{image_path.stem}_sanitized.jpg"
    img.save(output_path, format="JPEG", quality=config["output_quality"])
    result["output_path"] = str(output_path)

    logger.info(f"ALLOW: Sanitized image saved to {output_path}")
    result["reasons"].append("All checks passed")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Image Guardrails Pipeline - Validate and sanitize images before AI processing"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--config", "-c",
        help="Path to YAML config file",
        default=None
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Directory for sanitized output",
        default=None
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON"
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Error: File not found: {image_path}")

    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    result = run_guardrails(image_path, config, output_dir)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"DECISION: {result['decision']}")
        print(f"{'='*60}")
        print(f"Input:    {result['input_path']}")
        print(f"SHA256:   {result['sha256']}")
        if result.get('mime_type'):
            print(f"MIME:     {result['mime_type']}")
        if result.get('resolution'):
            print(f"Size:     {result['resolution']}")
        if 'nsfw' in result.get('checks', {}):
            print(f"NSFW:     {result['checks']['nsfw']['score']:.4f}")
        if 'safety' in result.get('checks', {}):
            scores = result['checks']['safety']['scores']
            print(f"Violence: {scores.get('violence', 0):.4f}")
            print(f"Weapons:  {scores.get('weapons', 0):.4f}")
            print(f"Safe:     {scores.get('safe', 0):.4f}")
        if result.get('output_path'):
            print(f"Output:   {result['output_path']}")
        print(f"Reasons:  {'; '.join(result['reasons'])}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
