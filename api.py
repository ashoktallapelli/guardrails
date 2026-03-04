"""
api.py - FastAPI REST API for Image Guardrails

Provides REST endpoints for image validation, sanitization, and text PII detection.
Reuses all functions from image_guard.py for consistency with CLI.

Usage:
    uvicorn api:app --reload --port 8000

Endpoints:
    POST /analyze/image     - Analyze image without modification
    POST /process/image     - Full pipeline with sanitization
    POST /analyze/text      - Detect PII in text
    POST /anonymize/text    - Anonymize PII in text
    GET  /health            - Health check and model status
    GET  /config            - View current configuration
"""

import io
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from PIL import Image

# Import functions from image_guard.py
from image_guard import (
    load_config,
    sha256_file,
    validate_resolution,
    strip_exif,
    check_nsfw,
    check_violence_safety,
    check_hate_symbols,
    detect_pii,
    detect_text_pii,
    anonymize_text_pii,
    detect_faces,
    redact_pii,
    blur_faces,
    compute_perceptual_hash,
    _model_cache,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global config
_config: Dict[str, Any] = {}


# ============================================================
# Pydantic Models
# ============================================================

class TextRequest(BaseModel):
    """Request body for text PII analysis."""
    text: str = Field(..., min_length=1, description="Text to analyze for PII")


class TextResponse(BaseModel):
    """Response for text PII analysis."""
    input_type: str = "text"
    timestamp: str
    original_length: int
    pii: Dict[str, Any]
    anonymized: bool
    output_text: Optional[str] = None


class ImageAnalysisResponse(BaseModel):
    """Response for image analysis."""
    input_filename: str
    timestamp: str
    decision: str
    is_safe: bool
    results: Dict[str, Any]
    meta: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    models_loaded: Dict[str, bool]
    config_loaded: bool


class ConfigResponse(BaseModel):
    """Configuration response."""
    config: Dict[str, Any]


# ============================================================
# Startup/Shutdown
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup: Load config and optionally preload models
    global _config
    try:
        _config = load_config()
        logger.info("Configuration loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}")
        raise

    # Optionally preload models (uncomment to enable)
    # await preload_models()

    yield

    # Shutdown: Cleanup
    logger.info("Shutting down API server")


async def preload_models():
    """Preload ML models for faster first request."""
    logger.info("Preloading models...")

    # Create a small test image
    test_img = Image.new('RGB', (100, 100), color='white')

    # Trigger model loading
    if _config.get("enable_violence_check", True):
        try:
            check_violence_safety(test_img, _config)
            logger.info("CLIP model preloaded")
        except Exception as e:
            logger.warning(f"Could not preload CLIP: {e}")

    if _config.get("nsfw_model") == "adamcodd":
        try:
            check_nsfw(test_img, _config)
            logger.info("AdamCodd NSFW model preloaded")
        except Exception as e:
            logger.warning(f"Could not preload AdamCodd: {e}")

    logger.info("Model preloading complete")


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Image Guardrails API",
    description="REST API for image validation, sanitization, and PII detection",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# Helper Functions
# ============================================================

def validate_file_type_bytes(file_bytes: bytes, filename: str) -> tuple[bool, str]:
    """Validate file type from bytes using magic."""
    try:
        import magic
    except ImportError:
        return True, "unknown (magic not available)"

    max_bytes = _config.get("max_file_size_mb", 10) * 1024 * 1024
    size = len(file_bytes)

    if size > max_bytes:
        return False, f"File too large: {size / (1024*1024):.2f} MB > {_config['max_file_size_mb']} MB limit"

    if size == 0:
        return False, "File is empty"

    mime = magic.from_buffer(file_bytes, mime=True)
    allowed = _config.get("allowed_mime_types", ["image/jpeg", "image/png", "image/webp", "image/gif"])

    if mime not in allowed:
        return False, f"Disallowed MIME type: {mime}. Allowed: {allowed}"

    return True, mime


def compute_hash_bytes(file_bytes: bytes) -> str:
    """Compute SHA-256 hash of bytes."""
    import hashlib
    return hashlib.sha256(file_bytes).hexdigest()


async def process_uploaded_image(file: UploadFile) -> tuple[bytes, Image.Image]:
    """Read and validate uploaded image file."""
    # Read file bytes
    file_bytes = await file.read()

    # Validate file type
    valid, mime_info = validate_file_type_bytes(file_bytes, file.filename or "unknown")
    if not valid:
        raise HTTPException(status_code=400, detail=mime_info)

    # Open image
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to open image: {e}")

    # Validate resolution
    valid, res_info = validate_resolution(img, _config)
    if not valid:
        raise HTTPException(status_code=400, detail=res_info)

    return file_bytes, img


# ============================================================
# Endpoints
# ============================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Returns status and loaded model information.
    """
    models_loaded = {
        "clip": "clip_model" in _model_cache,
        "nsfw_opennsfw2": "nsfw_opennsfw2_model" in _model_cache,
        "nsfw_adamcodd": "nsfw_adamcodd_model" in _model_cache,
        "text_analyzer": "text_analyzer" in _model_cache,
        "text_anonymizer": "text_anonymizer" in _model_cache,
    }

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        models_loaded=models_loaded,
        config_loaded=bool(_config),
    )


@app.get("/config", response_model=ConfigResponse, tags=["Configuration"])
async def get_config():
    """
    Get current configuration.
    Returns sanitized config (excluding sensitive values).
    """
    # Return config without sensitive values
    safe_config = {k: v for k, v in _config.items() if k != "environment"}
    return ConfigResponse(config=safe_config)


@app.post("/analyze/image", response_model=ImageAnalysisResponse, tags=["Image"])
async def analyze_image_endpoint(file: UploadFile = File(..., description="Image file to analyze")):
    """
    Analyze image without modification.

    Returns comprehensive analysis including:
    - File validation results
    - NSFW detection score
    - Violence/weapons detection scores
    - Hate symbol detection scores
    - PII detection (text in image)
    - Face detection count
    """
    t0 = time.time()

    # Process uploaded file
    file_bytes, img = await process_uploaded_image(file)

    # Compute hash
    file_hash = compute_hash_bytes(file_bytes)

    # Initialize result
    result = {
        "input_filename": file.filename or "unknown",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": "ALLOW",
        "is_safe": True,
        "results": {},
        "meta": {
            "sha256": file_hash,
            "file_size_bytes": len(file_bytes),
        }
    }

    width, height = img.size
    result["results"]["resolution"] = {
        "valid": True,
        "width": width,
        "height": height,
    }

    # NSFW check
    is_safe, nsfw_score = check_nsfw(img, _config)
    result["results"]["nsfw"] = {
        "safe": is_safe,
        "score": round(nsfw_score, 4),
        "threshold": _config.get("nsfw_threshold", 0.8),
    }
    if not is_safe:
        result["decision"] = "REJECT"
        result["is_safe"] = False

    # Violence check
    if _config.get("enable_violence_check", True):
        violence_safe, safety_scores = check_violence_safety(img, _config)
        result["results"]["violence"] = {
            "safe": violence_safe,
            "scores": safety_scores,
            "threshold": _config.get("violence_threshold", 0.7),
        }
        if not violence_safe:
            result["decision"] = "REJECT"
            result["is_safe"] = False

    # Hate symbol check
    if _config.get("enable_hate_symbol_check", True):
        hate_safe, hate_scores = check_hate_symbols(img, _config)
        result["results"]["hate_symbols"] = {
            "safe": hate_safe,
            "scores": hate_scores,
            "threshold": _config.get("hate_symbol_threshold", 0.75),
        }
        if not hate_safe:
            result["decision"] = "REJECT"
            result["is_safe"] = False

    # PII detection
    if _config.get("enable_pii_redaction", True):
        pii_result = detect_pii(img, _config)
        result["results"]["pii"] = pii_result

    # Face detection
    if _config.get("enable_face_blur", True):
        face_result = detect_faces(img, _config)
        result["results"]["faces"] = face_result

    # Perceptual hash
    phash = compute_perceptual_hash(img)
    result["meta"]["perceptual_hash"] = phash
    result["meta"]["processing_ms"] = int((time.time() - t0) * 1000)

    return ImageAnalysisResponse(**result)


@app.post("/process/image", tags=["Image"])
async def process_image_endpoint(
    file: UploadFile = File(..., description="Image file to process"),
    return_image: bool = True,
):
    """
    Full guardrails pipeline with sanitization.

    Applies:
    - EXIF stripping
    - PII redaction
    - Face blurring

    Returns sanitized image as JPEG or JSON with base64 image.
    """
    t0 = time.time()

    # Process uploaded file
    file_bytes, img = await process_uploaded_image(file)
    file_hash = compute_hash_bytes(file_bytes)

    result = {
        "input_filename": file.filename or "unknown",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": "ALLOW",
        "is_safe": True,
        "sha256": file_hash,
        "checks": {},
    }

    # NSFW check
    is_safe, nsfw_score = check_nsfw(img, _config)
    result["checks"]["nsfw"] = {"safe": is_safe, "score": round(nsfw_score, 4)}
    if not is_safe:
        raise HTTPException(
            status_code=422,
            detail=f"NSFW content detected (score: {nsfw_score:.2f})"
        )

    # Violence check
    if _config.get("enable_violence_check", True):
        violence_safe, safety_scores = check_violence_safety(img, _config)
        result["checks"]["violence"] = {"safe": violence_safe, "scores": safety_scores}
        if not violence_safe:
            raise HTTPException(
                status_code=422,
                detail=f"Unsafe content detected: {safety_scores}"
            )

    # Hate symbol check
    if _config.get("enable_hate_symbol_check", True):
        hate_safe, hate_scores = check_hate_symbols(img, _config)
        result["checks"]["hate_symbols"] = {"safe": hate_safe, "scores": hate_scores}
        if not hate_safe:
            raise HTTPException(
                status_code=422,
                detail=f"Hate symbols detected: {hate_scores}"
            )

    # Strip EXIF
    img = strip_exif(img)
    result["checks"]["exif_stripped"] = True

    # PII redaction
    if _config.get("enable_pii_redaction", True):
        img = redact_pii(img, _config)
        result["checks"]["pii_redacted"] = True

    # Face blur
    if _config.get("enable_face_blur", True):
        img = blur_faces(img, _config)
        result["checks"]["faces_blurred"] = True

    # Compute perceptual hash
    phash = compute_perceptual_hash(img)
    result["perceptual_hash"] = phash
    result["processing_ms"] = int((time.time() - t0) * 1000)

    # Return sanitized image
    if return_image:
        buf = io.BytesIO()
        quality = _config.get("output_quality", 95)
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="image/jpeg",
            headers={
                "X-Guardrails-Decision": result["decision"],
                "X-Guardrails-SHA256": file_hash,
                "X-Guardrails-Perceptual-Hash": phash,
                "X-Guardrails-Processing-Ms": str(result["processing_ms"]),
            }
        )

    # Return JSON with base64 image
    import base64
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=_config.get("output_quality", 95))
    result["sanitized_image_base64"] = base64.b64encode(buf.getvalue()).decode()

    return JSONResponse(content=result)


@app.post("/analyze/text", response_model=TextResponse, tags=["Text PII"])
async def analyze_text_endpoint(request: TextRequest):
    """
    Detect PII in text without modification.

    Returns list of detected entities with:
    - Entity type (PERSON, EMAIL_ADDRESS, etc.)
    - Original text
    - Confidence score
    - Character positions
    """
    pii_result = detect_text_pii(request.text, _config)

    return TextResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        original_length=len(request.text),
        pii=pii_result,
        anonymized=False,
    )


@app.post("/anonymize/text", response_model=TextResponse, tags=["Text PII"])
async def anonymize_text_endpoint(request: TextRequest):
    """
    Detect and anonymize PII in text.

    Replaces detected PII with type labels (e.g., <PERSON>, <EMAIL_ADDRESS>).
    Operator can be configured in config.yaml (replace, redact, mask, hash).
    """
    pii_result = anonymize_text_pii(request.text, _config)

    return TextResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        original_length=len(request.text),
        pii=pii_result,
        anonymized=True,
        output_text=pii_result.get("anonymized_text"),
    )


# ============================================================
# Error Handlers
# ============================================================

@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request, exc):
    """Handle missing config file."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Configuration error: {exc}"}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected errors."""
    logger.exception("Unexpected error")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}"}
    )


# ============================================================
# Run with: uvicorn api:app --reload --port 8000
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
