"""
config.py - Configuration loading and management.

Loads configuration from config.yaml and sets up environment.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


def _setup_environment(env_config: Dict[str, Any]) -> None:
    """Set up environment variables from config."""
    # HuggingFace offline mode settings
    if env_config.get("hf_hub_offline", True):
        os.environ["HF_HUB_OFFLINE"] = "1"
    if env_config.get("transformers_offline", True):
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if env_config.get("hf_hub_disable_implicit_token", True):
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

    # Fix SSL certificates on Windows
    if sys.platform == 'win32':
        try:
            import certifi
            os.environ.setdefault('SSL_CERT_FILE', certifi.where())
            os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
        except ImportError:
            pass


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. Defaults to config.yaml in current directory.

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file not found
        ValueError: If config file is empty
    """
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

    # Set up environment from config
    env_config = config.get("environment", {})
    _setup_environment(env_config)

    logger.info(f"Loaded config from {config_path}")
    return config


def get_default_config() -> Dict[str, Any]:
    """
    Get configuration from config.yaml.

    Always loads from config.yaml - single source of truth.

    Returns:
        Configuration dictionary
    """
    return load_config()
