"""
guardrails - Image and text safety guardrails pipeline.

A pluggable pipeline that validates and sanitizes content before AI processing:
- NSFW detection
- Violence/weapons detection
- Hate symbol detection
- PII detection and redaction
- Face detection and blur

Usage:
    from guardrails import Pipeline, load_config

    config = load_config()
    pipeline = Pipeline(config)
    result = pipeline.run_image("image.jpg")

    if result["decision"] == "ALLOW":
        # Process image
        pass
"""

__version__ = "1.0.0"

from guardrails.config import load_config
from guardrails.pipeline import Pipeline, AVAILABLE_CHECKS
from guardrails.base import BaseCheck, CheckResult

__all__ = [
    "Pipeline",
    "load_config",
    "BaseCheck",
    "CheckResult",
    "AVAILABLE_CHECKS",
    "__version__",
]
