"""
api.py - FastAPI REST API for Image Guardrails

Aligned with AIForce Security Guardrails Service standard.

Usage:
    uvicorn api:app --reload --port 8000

Endpoints:
    POST /scan/image        - Scan image for safety (analyze + optional sanitization)
    POST /scan/text         - Scan text for PII (analyze + optional anonymization)
    GET  /check_health      - Health check
    GET  /config            - View current configuration
"""

import base64
import io
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from PIL import Image

# Import functions from image_guard.py
from image_guard import (
    load_config,
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
# Pydantic Models - Aligned with SGS Standard
# ============================================================

class MetricResponseOutput(BaseModel):
    """Standard scanner result format."""
    score: float
    threshold: float
    is_pass: bool
    is_error: bool = False


class ScanImageResponse(BaseModel):
    """Response for /scan/image endpoint."""
    decision: str  # ALLOW, REDACT, or REJECT
    is_safe: bool
    results: Dict[str, MetricResponseOutput]
    is_redacted: bool = False
    sanitized_image_base64: Optional[str] = None
    meta: Dict[str, Any] = {}


class TextScanRequest(BaseModel):
    """Request body for /scan/text endpoint."""
    input_text: str = Field(..., min_length=1, description="Text to scan for PII")


class TextScanResponse(BaseModel):
    """Response for /scan/text endpoint."""
    decision: str  # ALLOW or REDACT
    is_safe: bool
    results: Dict[str, MetricResponseOutput]
    is_redacted: bool = False
    sanitized_text: Optional[str] = None
    entities: list = []


class HealthCheckResponse(BaseModel):
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
    global _config
    try:
        _config = load_config()
        logger.info("Configuration loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}")
        raise

    yield

    logger.info("Shutting down API server")


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Image Guardrails API",
    description="REST API for image and text safety scanning - Aligned with SGS standard",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# Helper Functions
# ============================================================

def validate_file_type_bytes(file_bytes: bytes) -> tuple[bool, str]:
    """Validate file type from bytes using magic."""
    try:
        import magic
    except ImportError:
        return True, "unknown"

    max_bytes = _config.get("max_file_size_mb", 10) * 1024 * 1024
    size = len(file_bytes)

    if size > max_bytes:
        return False, f"File too large: {size / (1024*1024):.2f} MB"

    if size == 0:
        return False, "File is empty"

    mime = magic.from_buffer(file_bytes, mime=True)
    allowed = _config.get("allowed_mime_types", ["image/jpeg", "image/png", "image/webp", "image/gif"])

    if mime not in allowed:
        return False, f"Disallowed MIME type: {mime}"

    return True, mime


def compute_hash_bytes(file_bytes: bytes) -> str:
    """Compute SHA-256 hash of bytes."""
    import hashlib
    return hashlib.sha256(file_bytes).hexdigest()


async def process_uploaded_image(file: UploadFile) -> tuple[bytes, Image.Image]:
    """Read and validate uploaded image file."""
    file_bytes = await file.read()

    valid, mime_info = validate_file_type_bytes(file_bytes)
    if not valid:
        raise HTTPException(status_code=400, detail=mime_info)

    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to open image: {e}")

    valid, res_info = validate_resolution(img, _config)
    if not valid:
        raise HTTPException(status_code=400, detail=res_info)

    return file_bytes, img


# ============================================================
# Endpoints - Aligned with /scan/* pattern
# ============================================================

@app.get("/check_health", response_model=HealthCheckResponse, tags=["Health"])
async def check_health():
    """
    Health check endpoint.
    Returns status and loaded model information.
    """
    models_loaded = {
        "clip": "clip_model" in _model_cache,
        "nsfw_adamcodd": "nsfw_adamcodd_model" in _model_cache,
        "text_analyzer": "text_analyzer" in _model_cache,
    }

    return HealthCheckResponse(
        status="OK",
        timestamp=datetime.now(timezone.utc).isoformat(),
        models_loaded=models_loaded,
        config_loaded=bool(_config),
    )


@app.get("/config", response_model=ConfigResponse, tags=["Configuration"])
async def get_config():
    """Get current configuration (excluding sensitive values)."""
    safe_config = {k: v for k, v in _config.items() if k != "environment"}
    return ConfigResponse(config=safe_config)


@app.post("/scan/image", response_model=ScanImageResponse, tags=["Scan"])
async def scan_image(
    file: UploadFile = File(..., description="Image file to scan"),
):
    """
    Scan image for safety violations.

    Decisions:
    - REJECT: Unsafe content (NSFW, violence, hate symbols) - no image returned
    - REDACT: Safe but PII/faces found - sanitized image returned
    - ALLOW: Safe and clean - original image returned (EXIF stripped only)

    Checks:
    - NSFW content
    - Violence/weapons
    - Hate symbols
    - PII in image (OCR)
    - Faces detected
    """
    t0 = time.time()

    # Process uploaded file
    file_bytes, img = await process_uploaded_image(file)
    file_hash = compute_hash_bytes(file_bytes)

    results: Dict[str, MetricResponseOutput] = {}
    is_safe = True

    # NSFW check
    nsfw_safe, nsfw_score = check_nsfw(img, _config)
    nsfw_threshold = _config.get("nsfw_threshold", 0.8)
    results["nsfw"] = MetricResponseOutput(
        score=round(nsfw_score, 4),
        threshold=nsfw_threshold,
        is_pass=nsfw_safe,
        is_error=False
    )
    if not nsfw_safe:
        is_safe = False

    # Violence check
    if _config.get("enable_violence_check", True):
        violence_safe, safety_scores = check_violence_safety(img, _config)
        violence_threshold = _config.get("violence_threshold", 0.7)
        combined_score = (
            safety_scores.get("violence", 0) +
            safety_scores.get("weapons", 0) +
            safety_scores.get("disturbing", 0)
        )
        results["violence"] = MetricResponseOutput(
            score=round(combined_score, 4),
            threshold=violence_threshold,
            is_pass=violence_safe,
            is_error=False
        )
        if not violence_safe:
            is_safe = False

    # Hate symbols check
    if _config.get("enable_hate_symbol_check", True):
        hate_safe, hate_scores = check_hate_symbols(img, _config)
        hate_threshold = _config.get("hate_symbol_threshold", 0.75)
        combined_hate = hate_scores.get("combined_hate_score", 0)
        results["hate_symbols"] = MetricResponseOutput(
            score=round(combined_hate, 4),
            threshold=hate_threshold,
            is_pass=hate_safe,
            is_error=False
        )
        if not hate_safe:
            is_safe = False

    # PII detection
    pii_count = 0
    if _config.get("enable_pii_redaction", True):
        pii_result = detect_pii(img, _config)
        pii_count = pii_result.get("entity_count", 0)
        results["pii"] = MetricResponseOutput(
            score=float(pii_count),
            threshold=0,
            is_pass=True,
            is_error=False
        )

    # Face detection
    face_count = 0
    if _config.get("enable_face_blur", True):
        face_result = detect_faces(img, _config)
        face_count = face_result.get("face_count", 0)
        results["faces"] = MetricResponseOutput(
            score=float(face_count),
            threshold=0,
            is_pass=True,
            is_error=False
        )

    # Determine decision
    sanitized_image_base64 = None
    is_redacted = False

    if not is_safe:
        # REJECT - unsafe content, no image returned
        decision = "REJECT"
    else:
        # Safe - check if redaction needed
        needs_redaction = (pii_count > 0) or (face_count > 0)

        # Always strip EXIF
        img = strip_exif(img)

        if needs_redaction:
            # REDACT - apply PII redaction and face blur
            decision = "REDACT"
            is_redacted = True

            if _config.get("enable_pii_redaction", True) and pii_count > 0:
                img = redact_pii(img, _config)

            if _config.get("enable_face_blur", True) and face_count > 0:
                img = blur_faces(img, _config)
        else:
            # ALLOW - clean image, just EXIF stripped
            decision = "ALLOW"

        # Return image as base64
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_config.get("output_quality", 95))
        sanitized_image_base64 = base64.b64encode(buf.getvalue()).decode()

    # Build response
    phash = compute_perceptual_hash(img)
    processing_ms = int((time.time() - t0) * 1000)

    return ScanImageResponse(
        decision=decision,
        is_safe=is_safe,
        results=results,
        is_redacted=is_redacted,
        sanitized_image_base64=sanitized_image_base64,
        meta={
            "sha256": file_hash,
            "perceptual_hash": phash,
            "processing_ms": processing_ms,
            "filename": file.filename or "unknown",
        }
    )


@app.post("/scan/text", response_model=TextScanResponse, tags=["Scan"])
async def scan_text(request: TextScanRequest):
    """
    Scan text for PII and automatically anonymize.

    Decisions:
    - REDACT: PII found and anonymized
    - ALLOW: No PII found, text unchanged

    Detects and anonymizes:
    - Names (PERSON)
    - Email addresses
    - Phone numbers
    - Credit cards
    - SSN
    - Other PII entities
    """
    # Always run anonymization (will return original if no PII)
    pii_result = anonymize_text_pii(request.input_text, _config)
    sanitized_text = pii_result.get("anonymized_text")
    entities = pii_result.get("entities", [])

    pii_count = len(entities)
    pii_threshold = _config.get("pii_score_threshold", 0.35)

    # Determine decision based on PII found
    if pii_count > 0:
        decision = "REDACT"
        is_redacted = True
    else:
        decision = "ALLOW"
        is_redacted = False

    results: Dict[str, MetricResponseOutput] = {}
    results["pii"] = MetricResponseOutput(
        score=float(pii_count),
        threshold=pii_threshold,
        is_pass=True,
        is_error=False
    )

    return TextScanResponse(
        decision=decision,
        is_safe=True,
        results=results,
        is_redacted=is_redacted,
        sanitized_text=sanitized_text,
        entities=entities,
    )


# ============================================================
# Legacy endpoints (for backward compatibility)
# ============================================================

@app.get("/health", include_in_schema=False)
async def health_legacy():
    """Legacy health endpoint - redirects to /check_health."""
    return await check_health()


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
