"""
api.py - FastAPI REST API for guardrails.

Usage:
    uvicorn guardrails.api:app --reload --port 8000

Endpoints:
    POST /scan/image    - Scan image for safety
    POST /scan/text     - Scan text for PII
    GET  /check_health  - Health check
    GET  /config        - View configuration
"""

import base64
import hashlib
import io
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

from guardrails.config import load_config
from guardrails.pipeline import Pipeline
from guardrails import model_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global config and pipeline
_config: Dict[str, Any] = {}
_pipeline: Optional[Pipeline] = None


# ============================================================
# Pydantic Models
# ============================================================

class MetricResponseOutput(BaseModel):
    """Standard scanner result format."""
    score: float
    threshold: float
    is_pass: bool
    is_error: bool = False


class ScanImageResponse(BaseModel):
    """Response for /scan/image endpoint."""
    decision: str  # ALLOW or REJECT
    reason: str  # Human-readable reason
    is_safe: bool
    results: Dict[str, MetricResponseOutput]
    image_base64: Optional[str] = None
    meta: Dict[str, Any] = {}


class TextScanRequest(BaseModel):
    """Request body for /scan/text endpoint."""
    input_text: str = Field(..., min_length=1, description="Text to scan for PII")


class TextScanResponse(BaseModel):
    """Response for /scan/text endpoint."""
    decision: str  # ALLOW or REJECT
    reason: str
    is_safe: bool
    results: Dict[str, MetricResponseOutput]
    entities: list = []


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
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
    global _config, _pipeline

    try:
        _config = load_config()
        _pipeline = Pipeline(_config)
        logger.info("Configuration and pipeline loaded successfully")

        # Preload models by running a dummy check
        logger.info("Preloading models...")
        dummy_img = Image.new('RGB', (100, 100), color='white')
        _pipeline.run(dummy_img, input_type="image")
        logger.info("Models preloaded successfully")

    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}")
        raise

    yield

    # Cleanup
    model_cache.clear()
    logger.info("Shutting down API server")


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Guardrails API",
    description="REST API for image and text safety scanning",
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

    # Validate resolution
    max_res = _config.get("max_resolution", {"width": 4096, "height": 4096})
    width, height = img.size
    if width > max_res["width"] or height > max_res["height"]:
        raise HTTPException(
            status_code=400,
            detail=f"Resolution {width}x{height} exceeds limit {max_res['width']}x{max_res['height']}"
        )

    return file_bytes, img


# ============================================================
# Endpoints
# ============================================================

@app.get("/check_health", response_model=HealthCheckResponse, tags=["Health"])
async def check_health():
    """Health check endpoint."""
    return HealthCheckResponse(
        status="OK",
        timestamp=datetime.now(timezone.utc).isoformat(),
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
    ## Overview

    Validates an image before AI processing.

    This endpoint analyzes the uploaded image using configured guardrail scanners
    to detect potential safety, security, and compliance risks.

    **Validation includes checks for:**
    - NSFW/explicit content detection
    - Violence and weapons detection
    - Hate symbols and extremist imagery
    - Personally Identifiable Information (PII) in text
    - Face detection for privacy protection

    ---

    ## Request

    **Content-Type:** `multipart/form-data`

    **file** *(required)*
    - The image file to scan
    - **Supported formats:** JPEG, PNG, WebP, GIF
    - **Max file size:** 10 MB (configurable)
    - **Max resolution:** 4096x4096 (configurable)

    ---

    ## Response

    **decision** - Final verdict for the image
    - `ALLOW`: Image is safe, no issues detected. Image returned (EXIF stripped).
    - `REJECT`: Unsafe content detected. No image returned.

    **reason** - Human-readable explanation of the decision

    **is_safe** - Boolean indicating if content passed safety checks

    **results** - Individual scanner results with scores and thresholds

    **image_base64** - Base64-encoded output image (null if rejected)

    **meta** - Metadata including SHA-256 hash, processing time, filename

    ---

    ## Operational Behavior

    - Image is validated for file type using magic bytes (not extension)
    - All configured checks run in sequence
    - REJECT decision stops processing immediately
    - EXIF metadata is stripped from all returned images
    - Original image is never stored

    ---

    ## Example Response (ALLOW)

    ```json
    {
      "decision": "ALLOW",
      "reason": "All checks passed",
      "is_safe": true,
      "results": {
        "nsfw": {"score": 0.02, "threshold": 0.8, "is_pass": true},
        "violence": {"score": 0.05, "threshold": 0.7, "is_pass": true}
      },
      "image_base64": "/9j/4AAQSkZJRg...",
      "meta": {
        "sha256": "abc123...",
        "processing_ms": 245,
        "filename": "photo.jpg"
      }
    }
    ```

    ## Example Response (REJECT)

    ```json
    {
      "decision": "REJECT",
      "reason": "Faces detected: 2 face(s) found",
      "is_safe": false,
      "results": {
        "nsfw": {"score": 0.02, "threshold": 0.8, "is_pass": true},
        "faces": {"score": 2.0, "threshold": 0.0, "is_pass": false}
      },
      "image_base64": null,
      "meta": {
        "sha256": "abc123...",
        "processing_ms": 312,
        "filename": "photo.jpg"
      }
    }
    ```
    """
    t0 = time.time()

    # Process uploaded file
    file_bytes, img = await process_uploaded_image(file)
    file_hash = compute_hash_bytes(file_bytes)

    # Run pipeline
    pipeline_result = _pipeline.run(img, input_type="image")

    # Convert to API response format
    results: Dict[str, MetricResponseOutput] = {}

    for check_name, check_data in pipeline_result.get("checks", {}).items():
        if isinstance(check_data, dict) and "score" in check_data:
            results[check_name] = MetricResponseOutput(
                score=round(check_data.get("score", 0), 4),
                threshold=check_data.get("threshold", 0),
                is_pass=check_data.get("safe", True),
                is_error=False
            )

    # Get decision and reason
    decision = pipeline_result.get("decision", "ALLOW")
    reasons = pipeline_result.get("reasons", [])
    reason = reasons[0] if reasons else "All checks passed"

    # Encode image if not rejected
    image_base64 = None
    if decision != "REJECT" and "output" in pipeline_result:
        buf = io.BytesIO()
        pipeline_result["output"].save(buf, format="JPEG", quality=_config.get("output_quality", 95))
        image_base64 = base64.b64encode(buf.getvalue()).decode()

    processing_ms = int((time.time() - t0) * 1000)

    return ScanImageResponse(
        decision=decision,
        reason=reason,
        is_safe=pipeline_result.get("is_safe", True),
        results=results,
        image_base64=image_base64,
        meta={
            "sha256": file_hash,
            "processing_ms": processing_ms,
            "filename": file.filename or "unknown",
        }
    )


@app.post("/scan/text", response_model=TextScanResponse, tags=["Scan"])
async def scan_text(request: TextScanRequest):
    """
    ## Overview

    Validates text for PII before AI processing.

    This endpoint analyzes the input text using Presidio to detect
    Personally Identifiable Information (PII) to ensure compliance and privacy.

    **Detection includes:**
    - Email addresses
    - Phone numbers
    - Person names
    - Credit card numbers
    - Social Security Numbers (SSN)
    - IP addresses
    - Locations

    ---

    ## Request

    **Content-Type:** `application/json`

    **input_text** *(required)*
    - The text content to scan for PII
    - Must not be empty
    - Must contain at least 1 character

    ---

    ## Response

    **decision** - Final verdict for the text
    - `ALLOW`: No PII detected. Text is safe.
    - `REJECT`: PII found. Text rejected with details.

    **reason** - Human-readable explanation with entity counts

    **is_safe** - Boolean indicating if text passed checks

    **results** - Scanner results with entity count

    **entities** - List of detected PII entities with type, value, and confidence

    ---

    ## Example Request

    ```json
    {
      "input_text": "Contact John Doe at john@example.com or 555-123-4567"
    }
    ```

    ## Example Response (REJECT)

    ```json
    {
      "decision": "REJECT",
      "reason": "PII detected: 3 entities (PERSON, EMAIL_ADDRESS, PHONE_NUMBER)",
      "is_safe": false,
      "results": {
        "pii": {"score": 3, "threshold": 0.35, "is_pass": false}
      },
      "entities": [
        {"type": "PERSON", "text": "John Doe", "score": 0.85},
        {"type": "EMAIL_ADDRESS", "text": "john@example.com", "score": 0.95},
        {"type": "PHONE_NUMBER", "text": "555-123-4567", "score": 0.75}
      ]
    }
    ```

    ## Example Response (ALLOW)

    ```json
    {
      "decision": "ALLOW",
      "reason": "All checks passed",
      "is_safe": true,
      "results": {
        "pii": {"score": 0, "threshold": 0.35, "is_pass": true}
      },
      "entities": []
    }
    ```
    """
    # Run pipeline
    pipeline_result = _pipeline.run_text(request.input_text, anonymize=True)

    # Convert to API response format
    results: Dict[str, MetricResponseOutput] = {}
    entities = []

    pii_check = pipeline_result.get("checks", {}).get("pii", {})
    if pii_check:
        pii_details = pii_check.get("details", {})
        results["pii"] = MetricResponseOutput(
            score=float(pii_details.get("entity_count", 0)),
            threshold=_config.get("pii_score_threshold", 0.35),
            is_pass=True,
            is_error=False
        )
        entities = pii_details.get("entities", [])

    # Get decision and reason
    decision = pipeline_result.get("decision", "ALLOW")
    reasons = pipeline_result.get("reasons", [])
    reason = reasons[0] if reasons else "All checks passed"

    return TextScanResponse(
        decision=decision,
        reason=reason,
        is_safe=pipeline_result.get("is_safe", True),
        results=results,
        entities=entities,
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
# Run with: uvicorn guardrails.api:app --reload --port 8000
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("guardrails.api:app", host="0.0.0.0", port=8000, reload=True)
