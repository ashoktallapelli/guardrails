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
    decision: str  # ALLOW, REDACT, or REJECT
    reason: str  # Human-readable reason
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
    reason: str
    is_safe: bool
    results: Dict[str, MetricResponseOutput]
    is_redacted: bool = False
    sanitized_text: Optional[str] = None
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
    Scan image for safety violations.

    Decisions:
    - REJECT: Unsafe content - no image returned
    - REDACT: Safe but PII/faces found - sanitized image returned
    - ALLOW: Safe and clean - original image returned (EXIF stripped)
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
    sanitized_image_base64 = None
    if decision != "REJECT" and "output" in pipeline_result:
        buf = io.BytesIO()
        pipeline_result["output"].save(buf, format="JPEG", quality=_config.get("output_quality", 95))
        sanitized_image_base64 = base64.b64encode(buf.getvalue()).decode()

    processing_ms = int((time.time() - t0) * 1000)

    return ScanImageResponse(
        decision=decision,
        reason=reason,
        is_safe=pipeline_result.get("is_safe", True),
        results=results,
        is_redacted=pipeline_result.get("is_redacted", False),
        sanitized_image_base64=sanitized_image_base64,
        meta={
            "sha256": file_hash,
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
    - ALLOW: No PII found
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

    if decision == "REDACT" and entities:
        entity_types = list(set(e.get("type", "UNKNOWN") for e in entities))
        reason = f"PII anonymized: {len(entities)} entities ({', '.join(entity_types)})"
    else:
        reason = reasons[0] if reasons else "No PII detected"

    return TextScanResponse(
        decision=decision,
        reason=reason,
        is_safe=True,
        results=results,
        is_redacted=pipeline_result.get("is_redacted", False),
        sanitized_text=pipeline_result.get("anonymized_text"),
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
