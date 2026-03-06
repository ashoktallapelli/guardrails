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

# Set HuggingFace env vars BEFORE any imports to prevent SSL errors
import os
import sys
from pathlib import Path

def _early_setup():
    """Set environment variables before HuggingFace imports."""
    config_path = Path("config.yaml")
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        env = config.get("environment", {})
        if env.get("hf_hub_offline"):
            os.environ["HF_HUB_OFFLINE"] = "1"
        if env.get("transformers_offline"):
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if env.get("hf_hub_disable_implicit_token"):
            os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

    # Fix SSL on Windows
    if sys.platform == 'win32':
        try:
            import certifi
            os.environ.setdefault('SSL_CERT_FILE', certifi.where())
            os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
        except ImportError:
            pass

_early_setup()

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
