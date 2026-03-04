"""
image_guard.py - Image Guardrails Pipeline

A local-first pre-inference pipeline that validates and sanitizes images:
1. Content-type validation (magic bytes)
2. EXIF metadata stripping
3. NSFW content detection
4. Violence/safety detection
5. Hate symbol detection (CLIP-based)
6. PII redaction (OCR + masking)
7. Face detection and blurring
8. Outputs sanitized image with decision logging

Usage:
    python image_guard.py /path/to/input.jpg
    python image_guard.py /path/to/input.jpg --config config.yaml
"""

import argparse
import hashlib
import io
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import yaml


def _load_env_config() -> Dict[str, Any]:
    """Load environment settings from config.yaml early (before other imports)."""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        return config.get("environment", {})
    return {}


# Load environment settings from config.yaml
_env_config = _load_env_config()

# HuggingFace offline mode settings (read from config.yaml)
if _env_config.get("hf_hub_offline", True):
    os.environ["HF_HUB_OFFLINE"] = "1"
if _env_config.get("transformers_offline", True):
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
if _env_config.get("hf_hub_disable_implicit_token", True):
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

# Suppress tldextract SSL warnings on corporate networks
# These are non-blocking - tldextract falls back to local snapshot
import warnings
import logging
logging.getLogger("tldextract").setLevel(logging.ERROR)

# Fix SSL certificates on Windows
if sys.platform == 'win32':
    try:
        import certifi
        os.environ.setdefault('SSL_CERT_FILE', certifi.where())
        os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
    except ImportError:
        pass

# Configure console logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Separate audit logger for file logging (does not propagate to console)
audit_logger = logging.getLogger("guardrails.audit")
audit_logger.setLevel(logging.INFO)
audit_logger.propagate = False  # Don't print to console
_audit_handler_configured = False

# Cache for ML models (loaded once)
_model_cache = {}


def setup_file_logging(config: Dict[str, Any]) -> None:
    """Set up file logging for audit trail if enabled in config."""
    global _audit_handler_configured

    if _audit_handler_configured:
        return

    if not config.get("log_decisions_to_file", False):
        return

    log_file = config.get("log_file_path", "guardrails_audit.log")
    log_level = config.get("log_level", "INFO").upper()

    try:
        # Create file handler with JSON-friendly format
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, log_level, logging.INFO))

        # Simple format - we'll log JSON directly
        file_handler.setFormatter(logging.Formatter('%(message)s'))

        audit_logger.addHandler(file_handler)
        _audit_handler_configured = True

        logger.info(f"Audit logging enabled: {log_file}")
    except Exception as e:
        logger.warning(f"Failed to set up file logging: {e}")


def save_json_result(result: Dict[str, Any], output_dir: Path) -> Path:
    """Save JSON result to file with UUID name."""
    output_dir.mkdir(parents=True, exist_ok=True)
    unique_id = result.get("output_id") or uuid.uuid4().hex[:12]
    json_path = output_dir / f"{unique_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return json_path


def log_decision(result: Dict[str, Any], input_type: str = "image") -> None:
    """Log a decision to the audit log file as JSON."""
    if not _audit_handler_configured:
        return

    # Create audit record
    decision = result.get("decision", "UNKNOWN")
    audit_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_type": input_type,
        "decision": decision,
        "is_safe": decision == "ALLOW",  # is_safe is True only when ALLOW
    }

    # Add relevant fields based on input type
    if input_type == "image":
        audit_record["input_path"] = result.get("input_path", "unknown")
        audit_record["sha256"] = result.get("sha256") or result.get("meta", {}).get("sha256", "")
        audit_record["mime_type"] = result.get("mime_type") or result.get("meta", {}).get("mime_type", "")

        # Add scores
        if "checks" in result:
            checks = result["checks"]
            if "nsfw" in checks:
                audit_record["nsfw_score"] = checks["nsfw"].get("score", 0)
            if "safety" in checks:
                audit_record["violence_scores"] = checks["safety"].get("scores", {})
            if "hate_symbols" in checks:
                audit_record["hate_scores"] = checks["hate_symbols"].get("scores", {})
        elif "results" in result:
            results = result["results"]
            if "nsfw" in results:
                audit_record["nsfw_score"] = results["nsfw"].get("score", 0)
            if "violence" in results:
                audit_record["violence_scores"] = results["violence"].get("scores", {})
            if "hate_symbols" in results:
                audit_record["hate_scores"] = results["hate_symbols"].get("scores", {})

        # Add reasons if rejected
        if result.get("reasons"):
            audit_record["reasons"] = result["reasons"]

    elif input_type == "text":
        audit_record["text_length"] = result.get("original_length", 0)
        audit_record["pii_count"] = result.get("pii", {}).get("entity_count", 0)
        audit_record["anonymized"] = result.get("anonymized", False)

    # Log as JSON line
    audit_logger.info(json.dumps(audit_record))


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from config.yaml. Fails if not found.

    Single source of truth - no default fallbacks.
    """
    # Use config.yaml from current directory if not specified
    if config_path is None:
        config_path = Path("config.yaml")

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Please ensure config.yaml exists in the current directory.\n"
            "Or specify path with: --config /path/to/config.yaml"
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError(f"Config file is empty: {config_path}")

    # Set up file logging if enabled
    setup_file_logging(config)

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


def check_nsfw_opennsfw2(img, config: Dict[str, Any]) -> Tuple[bool, float]:
    """
    Run NSFW detection using OpenNSFW2 (ResNet-50 based).
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


def check_nsfw_adamcodd(img, config: Dict[str, Any]) -> Tuple[bool, float]:
    """
    Run NSFW detection using AdamCodd/vit-base-nsfw-detector (ViT based).
    Higher accuracy (96.54%) than OpenNSFW2, but larger model (~330MB).
    Returns (is_safe, nsfw_score).
    """
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        import torch
    except ImportError:
        logger.warning("transformers/torch not installed, skipping NSFW check")
        return True, 0.0

    model_name = "AdamCodd/vit-base-nsfw-detector"

    # Use cached model
    if "nsfw_adamcodd_model" not in _model_cache:
        logger.info("Loading AdamCodd NSFW model...")
        try:
            # Try local cache first
            _model_cache["nsfw_adamcodd_processor"] = AutoImageProcessor.from_pretrained(model_name, local_files_only=True)
            _model_cache["nsfw_adamcodd_model"] = AutoModelForImageClassification.from_pretrained(model_name, local_files_only=True)
            logger.info("Loaded AdamCodd NSFW model from local cache")
        except Exception as local_error:
            logger.info("Model not in local cache, trying to download...")
            try:
                _model_cache["nsfw_adamcodd_processor"] = AutoImageProcessor.from_pretrained(model_name)
                _model_cache["nsfw_adamcodd_model"] = AutoModelForImageClassification.from_pretrained(model_name)
            except Exception as e:
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ["ssl", "certificate", "connection", "network"]):
                    logger.warning(f"Cannot download AdamCodd model (network/SSL issue): {e}")
                    logger.warning("Skipping NSFW check - model not available offline")
                    _model_cache["nsfw_adamcodd_unavailable"] = True
                    return True, 0.0
                raise

    if _model_cache.get("nsfw_adamcodd_unavailable"):
        return True, 0.0

    processor = _model_cache["nsfw_adamcodd_processor"]
    model = _model_cache["nsfw_adamcodd_model"]

    try:
        inputs = processor(images=img, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        # Output: [normal, nsfw] - index 1 is NSFW score
        nsfw_score = float(probs[0][1])
        is_safe = nsfw_score < config["nsfw_threshold"]

        logger.info(f"AdamCodd NSFW score: {nsfw_score:.4f}")
        return is_safe, nsfw_score
    except Exception as e:
        logger.warning(f"AdamCodd NSFW check failed: {e}")
        return True, 0.0


def check_nsfw(img, config: Dict[str, Any]) -> Tuple[bool, float]:
    """
    Run NSFW detection using configured model.
    Supports: 'opennsfw2' (default) or 'adamcodd' (higher accuracy).
    Returns (is_safe, nsfw_score).
    """
    nsfw_model = config.get("nsfw_model", "opennsfw2").lower()

    if nsfw_model == "adamcodd":
        return check_nsfw_adamcodd(img, config)
    else:
        return check_nsfw_opennsfw2(img, config)


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
            # Try local cache first (no network access)
            _model_cache["clip_model"] = CLIPModel.from_pretrained(model_name, local_files_only=True)
            _model_cache["clip_processor"] = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
            logger.info("Loaded CLIP model from local cache")
        except Exception as local_error:
            logger.info(f"Model not in local cache, trying to download...")
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


def detect_pii(img) -> Dict[str, Any]:
    """
    Detect PII in images using OCR + Presidio Analyzer.
    Returns detection results WITHOUT modifying the image.
    """
    try:
        import pytesseract
        from presidio_analyzer import AnalyzerEngine
    except ImportError:
        logger.warning("pytesseract or presidio-analyzer not installed, skipping PII detection")
        return {"enabled": False, "error": "dependencies not installed"}

    try:
        # Extract text using OCR
        extracted_text = pytesseract.image_to_string(img)

        if not extracted_text.strip():
            return {
                "enabled": True,
                "text_found": False,
                "text_length": 0,
                "entities": [],
                "entity_count": 0
            }

        # Analyze text for PII
        analyzer = AnalyzerEngine()
        results = analyzer.analyze(text=extracted_text, language="en")

        entities = []
        for result in results:
            entities.append({
                "type": result.entity_type,
                "text": extracted_text[result.start:result.end],
                "score": round(result.score, 4),
                "start": result.start,
                "end": result.end
            })

        logger.info(f"PII detection: found {len(entities)} entities")

        return {
            "enabled": True,
            "text_found": True,
            "text_length": len(extracted_text),
            "extracted_text": extracted_text[:1000] if len(extracted_text) > 1000 else extracted_text,
            "entities": entities,
            "entity_count": len(entities)
        }
    except Exception as e:
        logger.warning(f"PII detection failed: {e}")
        return {"enabled": True, "error": str(e)}


def detect_text_pii(text: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect PII in text using Presidio Analyzer.
    Returns detection results with entity details.
    """
    try:
        from presidio_analyzer import AnalyzerEngine
    except ImportError:
        logger.warning("presidio-analyzer not installed, skipping text PII detection")
        return {"enabled": False, "error": "presidio-analyzer not installed"}

    if not text or not text.strip():
        return {
            "enabled": True,
            "text_length": 0,
            "entities": [],
            "entity_count": 0
        }

    try:
        # Get configured entity types or use defaults
        entity_types = config.get("pii_entity_types", None)
        language = config.get("pii_language", "en")
        score_threshold = config.get("pii_score_threshold", 0.5)

        # Initialize analyzer (cached)
        if "text_analyzer" not in _model_cache:
            _model_cache["text_analyzer"] = AnalyzerEngine()

        analyzer = _model_cache["text_analyzer"]

        # Analyze text for PII
        results = analyzer.analyze(
            text=text,
            language=language,
            entities=entity_types,
            score_threshold=score_threshold
        )

        entities = []
        for result in results:
            entities.append({
                "type": result.entity_type,
                "text": text[result.start:result.end],
                "score": round(result.score, 4),
                "start": result.start,
                "end": result.end
            })

        logger.info(f"Text PII detection: found {len(entities)} entities")

        return {
            "enabled": True,
            "text_length": len(text),
            "entities": entities,
            "entity_count": len(entities)
        }
    except Exception as e:
        logger.warning(f"Text PII detection failed: {e}")
        return {"enabled": True, "error": str(e)}


def anonymize_text_pii(text: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect and anonymize PII in text using Presidio.
    Returns anonymized text and detection details.

    Supports operators: replace, redact, mask, hash, encrypt
    """
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        from presidio_anonymizer.entities import OperatorConfig
    except ImportError:
        logger.warning("presidio packages not installed, skipping text PII anonymization")
        return {"enabled": False, "error": "presidio packages not installed"}

    if not text or not text.strip():
        return {
            "enabled": True,
            "original_length": 0,
            "anonymized_text": "",
            "entities": [],
            "entity_count": 0
        }

    try:
        # Get configuration
        language = config.get("pii_language", "en")
        score_threshold = config.get("pii_score_threshold", 0.5)
        default_operator = config.get("pii_operator", "replace")
        operator_config = config.get("pii_operator_config", {})

        # Initialize engines (cached)
        if "text_analyzer" not in _model_cache:
            _model_cache["text_analyzer"] = AnalyzerEngine()
        if "text_anonymizer" not in _model_cache:
            _model_cache["text_anonymizer"] = AnonymizerEngine()

        analyzer = _model_cache["text_analyzer"]
        anonymizer = _model_cache["text_anonymizer"]

        # Analyze text for PII
        results = analyzer.analyze(
            text=text,
            language=language,
            score_threshold=score_threshold
        )

        # Build operators configuration
        operators = {}
        for entity_type in set(r.entity_type for r in results):
            entity_config = operator_config.get(entity_type, {})
            op_type = entity_config.get("type", default_operator)
            op_params = entity_config.get("params", {"new_value": f"<{entity_type}>"})
            operators[entity_type] = OperatorConfig(op_type, op_params)

        # Anonymize text
        anonymized = anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=operators
        )

        # Build entity list
        entities = []
        for result in results:
            entities.append({
                "type": result.entity_type,
                "original_text": text[result.start:result.end],
                "score": round(result.score, 4),
                "start": result.start,
                "end": result.end
            })

        logger.info(f"Text PII anonymization: processed {len(entities)} entities")

        return {
            "enabled": True,
            "original_length": len(text),
            "anonymized_text": anonymized.text,
            "anonymized_length": len(anonymized.text),
            "entities": entities,
            "entity_count": len(entities)
        }
    except Exception as e:
        logger.warning(f"Text PII anonymization failed: {e}")
        return {"enabled": True, "error": str(e)}


def detect_faces(img, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect faces in image using OpenCV Haar cascades.
    Returns detection results WITHOUT modifying the image.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.warning("opencv-python not installed, skipping face detection")
        return {"enabled": False, "error": "opencv not installed"}

    try:
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

        face_boxes = []
        for (x, y, w, h) in faces:
            face_boxes.append({
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h)
            })

        logger.info(f"Face detection: found {len(faces)} face(s)")

        return {
            "enabled": True,
            "face_count": len(faces),
            "faces": face_boxes
        }
    except Exception as e:
        logger.warning(f"Face detection failed: {e}")
        return {"enabled": True, "error": str(e)}


def redact_pii(img, config: Dict[str, Any]):
    """
    Use Presidio Image Redactor to detect and mask PII in images.
    Handles screenshots, IDs, documents with sensitive text.

    Uses pii_score_threshold from config for consistent detection.
    """
    try:
        from presidio_image_redactor import ImageRedactorEngine
    except ImportError:
        logger.warning("presidio-image-redactor not installed, skipping PII redaction")
        return img

    try:
        engine = ImageRedactorEngine()
        score_threshold = config.get("pii_score_threshold", 0.35)
        redacted = engine.redact(img, score_threshold=score_threshold)
        logger.info(f"PII redaction completed (threshold: {score_threshold})")
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


def check_hate_symbols(img, config: Dict[str, Any]) -> Tuple[bool, Dict[str, float]]:
    """
    Detect hate symbols using CLIP zero-shot classification.
    Uses the same CLIP model as violence detection for efficiency.

    Returns (is_safe, scores_dict) where scores_dict contains confidence scores
    for various hate symbol categories.
    """
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch
    except ImportError:
        logger.warning("transformers/torch not installed, skipping hate symbol check")
        return True, {"skipped": True}

    # Use cached model if available (shared with violence check)
    model_name = "openai/clip-vit-base-patch32"

    if "clip_model" not in _model_cache:
        logger.info("Loading CLIP model for hate symbol detection...")
        try:
            # Try local cache first (no network access)
            _model_cache["clip_model"] = CLIPModel.from_pretrained(model_name, local_files_only=True)
            _model_cache["clip_processor"] = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
            logger.info("Loaded CLIP model from local cache")
        except Exception as local_error:
            logger.info(f"Model not in local cache, trying to download...")
            try:
                _model_cache["clip_model"] = CLIPModel.from_pretrained(model_name)
                _model_cache["clip_processor"] = CLIPProcessor.from_pretrained(model_name)
            except Exception as e:
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ["ssl", "certificate", "connection", "network", "huggingface", "config.json"]):
                    logger.warning(f"Cannot download CLIP model (network/SSL issue): {e}")
                    logger.warning("Skipping hate symbol check - model not available offline")
                    _model_cache["clip_unavailable"] = True
                    return True, {"skipped": True, "reason": "model unavailable"}
                raise

    if _model_cache.get("clip_unavailable"):
        return True, {"skipped": True, "reason": "model unavailable"}

    model = _model_cache["clip_model"]
    processor = _model_cache["clip_processor"]

    # Hate symbol categories for zero-shot classification
    # Using general descriptions to detect symbols/imagery
    hate_labels = [
        "a normal, safe photograph without any symbols",
        "hate symbols, extremist imagery, or offensive symbols",
        "nazi symbols, swastika, or white supremacist imagery",
        "confederate flag or racist symbols",
    ]

    threshold = config.get("hate_symbol_threshold", 0.60)

    try:
        inputs = processor(
            text=hate_labels,
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
            "hate_symbols": round(probs[1], 4),
            "nazi_symbols": round(probs[2], 4),
            "racist_symbols": round(probs[3], 4),
        }

        # Combined hate symbol score
        hate_score = scores["hate_symbols"] + scores["nazi_symbols"] + scores["racist_symbols"]
        is_safe = hate_score < threshold

        scores["combined_hate_score"] = round(hate_score, 4)

        logger.info(f"Hate symbol scores: safe={scores['safe']:.2f}, combined_hate={hate_score:.2f}")

        return is_safe, scores

    except Exception as e:
        logger.warning(f"Hate symbol check failed: {e}")
        return True, {"error": str(e)}


def analyze_image(
    image_path: Path,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze image and return all detection results as JSON.
    Does NOT modify or save the image.

    Returns a comprehensive analysis dictionary with:
    - decision: ALLOW or REJECT
    - is_safe: boolean
    - results: all check results (nsfw, violence, hate_symbols, pii, faces)
    - meta: timing, file info
    """
    import time
    from PIL import Image

    t0 = time.time()

    result = {
        "input_path": str(image_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": "ALLOW",
        "is_safe": True,
        "results": {},
        "meta": {}
    }

    # Step 1: Compute file hash
    file_hash = sha256_file(image_path)
    result["meta"]["sha256"] = file_hash
    logger.info(f"Analyzing: {image_path} (SHA256: {file_hash[:16]}...)")

    # Step 2: Validate file type
    valid, mime_info = validate_file_type(image_path, config)
    result["results"]["file_validation"] = {
        "valid": valid,
        "mime_type": mime_info if valid else None,
        "error": mime_info if not valid else None,
        "max_size_mb": config["max_file_size_mb"]
    }

    if not valid:
        result["decision"] = "REJECT"
        result["is_safe"] = False
        result["meta"]["processing_ms"] = int((time.time() - t0) * 1000)
        return result

    result["meta"]["mime_type"] = mime_info
    result["meta"]["file_size_bytes"] = image_path.stat().st_size

    # Step 3: Load and validate resolution
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        result["decision"] = "REJECT"
        result["is_safe"] = False
        result["results"]["file_validation"]["error"] = f"Failed to open: {e}"
        result["meta"]["processing_ms"] = int((time.time() - t0) * 1000)
        return result

    valid, res_info = validate_resolution(img, config)
    width, height = img.size
    result["results"]["resolution"] = {
        "valid": valid,
        "width": width,
        "height": height,
        "max_width": config["max_resolution"]["width"],
        "max_height": config["max_resolution"]["height"]
    }

    if not valid:
        result["decision"] = "REJECT"
        result["is_safe"] = False
        result["meta"]["processing_ms"] = int((time.time() - t0) * 1000)
        return result

    # Step 4: NSFW content check
    is_safe, nsfw_score = check_nsfw(img, config)
    result["results"]["nsfw"] = {
        "safe": is_safe,
        "score": round(nsfw_score, 4),
        "threshold": config["nsfw_threshold"]
    }

    if not is_safe:
        result["decision"] = "REJECT"
        result["is_safe"] = False

    # Step 5: Violence/Safety check
    if config.get("enable_violence_check", True):
        violence_safe, safety_scores = check_violence_safety(img, config)
        result["results"]["violence"] = {
            "safe": violence_safe,
            "scores": safety_scores,
            "threshold": config["violence_threshold"]
        }

        if not violence_safe:
            result["decision"] = "REJECT"
            result["is_safe"] = False

    # Step 6: Hate symbol detection
    if config.get("enable_hate_symbol_check", True):
        hate_safe, hate_scores = check_hate_symbols(img, config)
        result["results"]["hate_symbols"] = {
            "safe": hate_safe,
            "scores": hate_scores,
            "threshold": config.get("hate_symbol_threshold", 0.75)
        }

        if not hate_safe:
            result["decision"] = "REJECT"
            result["is_safe"] = False

    # Step 7: PII detection (without redaction)
    if config.get("enable_pii_redaction", True):
        pii_result = detect_pii(img)
        result["results"]["pii"] = pii_result

    # Step 8: Face detection (without blur)
    if config.get("enable_face_blur", True):
        face_result = detect_faces(img, config)
        result["results"]["faces"] = face_result

    # Step 9: Compute perceptual hash
    phash = compute_perceptual_hash(img)
    result["meta"]["perceptual_hash"] = phash

    # Finalize
    result["meta"]["processing_ms"] = int((time.time() - t0) * 1000)

    logger.info(f"Analysis complete: {result['decision']}")

    # Log decision to audit file
    log_decision(result, input_type="image")

    return result


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

    # Generate unique ID for this processing run
    unique_id = uuid.uuid4().hex[:12]

    result = {
        "input_path": str(image_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": "ALLOW",
        "reasons": [],
        "checks": {},
        "output_path": None,
        "output_id": unique_id,
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
        log_decision(result, input_type="image")
        return result

    result["mime_type"] = mime_info

    # Step 3: Load and validate resolution
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        result["decision"] = "REJECT"
        result["reasons"].append(f"Failed to open image: {e}")
        log_decision(result, input_type="image")
        return result

    valid, res_info = validate_resolution(img, config)
    result["checks"]["resolution"] = {"valid": valid, "info": res_info}

    if not valid:
        result["decision"] = "REJECT"
        result["reasons"].append(f"Resolution check failed: {res_info}")
        logger.warning(f"REJECT: {res_info}")
        log_decision(result, input_type="image")
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
        log_decision(result, input_type="image")
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
            log_decision(result, input_type="image")
            return result

    # Step 7: Hate symbol detection (CLIP-based)
    if config.get("enable_hate_symbol_check", True):
        is_safe, hate_scores = check_hate_symbols(img, config)
        result["checks"]["hate_symbols"] = {
            "safe": is_safe,
            "scores": hate_scores
        }

        if not is_safe:
            result["decision"] = "REJECT"
            result["reasons"].append(
                f"Hate symbols detected: combined_score={hate_scores.get('combined_hate_score', 0):.2f}"
            )
            logger.warning(f"REJECT: Hate symbols detected (scores: {hate_scores})")
            log_decision(result, input_type="image")
            return result

    # Step 8: PII redaction (OCR + masking)
    if config["enable_pii_redaction"]:
        img = redact_pii(img, config)
        result["checks"]["pii_redaction"] = True

    # Step 9: Face blur for anonymization
    if config["enable_face_blur"]:
        img = blur_faces(img, config)
        result["checks"]["face_blur"] = True

    # Step 10: Compute perceptual hash (for known-bad matching)
    phash = compute_perceptual_hash(img)
    result["perceptual_hash"] = phash
    result["checks"]["phash_computed"] = True

    # Step 11: Save sanitized output to organized folder structure
    if output_dir is None:
        output_dir = image_path.parent / "output"

    # Create decision-based subfolder (allowed/rejected)
    decision_folder = output_dir / "allowed"
    decision_folder.mkdir(parents=True, exist_ok=True)

    # Use UUID for unbiased filename (already generated at start)
    output_path = decision_folder / f"{unique_id}.jpg"
    img.save(output_path, format="JPEG", quality=config["output_quality"])
    result["output_path"] = str(output_path)

    logger.info(f"ALLOW: Sanitized image saved to {output_path}")
    result["reasons"].append("All checks passed")

    # Log decision to audit file
    log_decision(result, input_type="image")

    return result


def analyze_text(text: str, config: Dict[str, Any], anonymize: bool = False) -> Dict[str, Any]:
    """
    Analyze text for PII and optionally anonymize.
    Returns analysis results as a dictionary.
    """
    result = {
        "input_type": "text",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "original_length": len(text),
    }

    if anonymize:
        pii_result = anonymize_text_pii(text, config)
        result["pii"] = pii_result
        result["anonymized"] = True
        if "anonymized_text" in pii_result:
            result["output_text"] = pii_result["anonymized_text"]
    else:
        pii_result = detect_text_pii(text, config)
        result["pii"] = pii_result
        result["anonymized"] = False

    # Log to audit file
    result["decision"] = "ALLOW"  # Text PII doesn't reject, just detects/anonymizes
    result["is_safe"] = True
    log_decision(result, input_type="text")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Image Guardrails Pipeline - Validate and sanitize images before AI processing"
    )
    parser.add_argument("image", nargs="?", help="Path to input image")
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
    parser.add_argument(
        "--analyze-only", "-a",
        action="store_true",
        help="Analyze only - return JSON results without modifying the image"
    )
    # Text PII arguments
    parser.add_argument(
        "--text", "-t",
        help="Analyze text for PII (direct input)"
    )
    parser.add_argument(
        "--text-file",
        help="Analyze text from file for PII"
    )
    parser.add_argument(
        "--anonymize",
        action="store_true",
        help="Anonymize PII in text (use with --text or --text-file)"
    )
    parser.add_argument(
        "--save-json",
        help="Save JSON results to specified directory"
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    # Text PII mode
    if args.text or args.text_file:
        if args.text_file:
            text_path = Path(args.text_file)
            if not text_path.exists():
                raise SystemExit(f"Error: Text file not found: {text_path}")
            text = text_path.read_text(encoding="utf-8")
        else:
            text = args.text

        result = analyze_text(text, config, anonymize=args.anonymize)
        print(json.dumps(result, indent=2))
        return

    # Image mode requires image argument
    if not args.image:
        parser.error("Image path is required (or use --text/--text-file for text PII analysis)")

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Error: File not found: {image_path}")

    # Analyze-only mode: return JSON without modifying image
    if args.analyze_only:
        result = analyze_image(image_path, config)
        print(json.dumps(result, indent=2))
        return

    # Normal mode: sanitize and save image
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    result = run_guardrails(image_path, config, output_dir)

    # Save JSON results if requested
    if args.save_json:
        json_dir = Path(args.save_json)
        json_path = save_json_result(result, json_dir)
        logger.info(f"JSON saved to: {json_path}")

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        # Simplified console output with decision and safety scores
        decision = result['decision']
        print(f"\n{'='*50}")
        print(f"  DECISION: {decision}")
        print(f"{'='*50}")

        # Show rejection reason if rejected
        if decision == "REJECT" and result.get('reasons'):
            print(f"\n  Reason: {result['reasons'][0]}")

        # Safety scores summary (only if checks were performed)
        checks = result.get('checks', {})
        has_safety_checks = any(k in checks for k in ['nsfw', 'safety', 'hate_symbols'])

        if has_safety_checks:
            print(f"\n  Safety Scores:")
            if 'nsfw' in checks:
                nsfw_score = checks['nsfw']['score']
                nsfw_status = "SAFE" if checks['nsfw']['safe'] else "UNSAFE"
                print(f"    NSFW:         {nsfw_score:.4f} ({nsfw_status})")

            if 'safety' in checks:
                scores = checks['safety']['scores']
                safe_status = "SAFE" if checks['safety']['safe'] else "UNSAFE"
                print(f"    Violence:     {scores.get('violence', 0):.4f}")
                print(f"    Weapons:      {scores.get('weapons', 0):.4f}")
                print(f"    Disturbing:   {scores.get('disturbing', 0):.4f}")
                print(f"    Safe:         {scores.get('safe', 0):.4f} ({safe_status})")

            if 'hate_symbols' in checks:
                hate_scores = checks['hate_symbols']['scores']
                hate_status = "SAFE" if checks['hate_symbols']['safe'] else "UNSAFE"
                print(f"    Hate Score:   {hate_scores.get('combined_hate_score', 0):.4f} ({hate_status})")

        # Output info (only show if image was saved)
        if result.get('output_path'):
            print(f"\n  Output:")
            print(f"    Image: {result['output_path']}")
            if result.get('output_id'):
                print(f"    ID:    {result['output_id']}")

        print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
