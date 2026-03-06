"""
config.py - Configuration loading.

Loads configuration from config.yaml.
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
    if env_config.get("hf_hub_offline"):
        os.environ["HF_HUB_OFFLINE"] = "1"
    if env_config.get("transformers_offline"):
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if env_config.get("hf_hub_disable_implicit_token"):
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

    # Fix SSL certificates on Windows
    if sys.platform == 'win32':
        try:
            import certifi
            os.environ.setdefault('SSL_CERT_FILE', certifi.where())
            os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
        except ImportError:
            pass


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. Defaults to ./config.yaml

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file not found
    """
    path = Path(config_path) if config_path else Path("config.yaml")

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError(f"Config file is empty: {path}")

    # Set up environment from config
    env_config = config.get("environment", {})
    _setup_environment(env_config)

    logger.info(f"Loaded config from {path}")
    return config
