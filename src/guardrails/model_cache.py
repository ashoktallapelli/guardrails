"""
model_cache.py - Thread-safe shared model cache.

All checks share this cache to avoid loading models multiple times.
"""

import threading
from typing import Any, Dict

# Thread-safe model cache
_cache: Dict[str, Any] = {}
_lock = threading.Lock()


def get(key: str) -> Any:
    """Get a model from cache."""
    with _lock:
        return _cache.get(key)


def set(key: str, model: Any) -> None:
    """Store a model in cache."""
    with _lock:
        _cache[key] = model


def has(key: str) -> bool:
    """Check if model exists in cache."""
    with _lock:
        return key in _cache


def clear() -> None:
    """Clear all cached models."""
    with _lock:
        _cache.clear()
