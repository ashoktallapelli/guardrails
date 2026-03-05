"""
nsfw.py - NSFW content detection check.

Supports two models:
- OpenNSFW2 (default): ResNet-50 based, ~24MB, ~90% accuracy
- AdamCodd: ViT based, ~330MB, ~96.5% accuracy
"""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

from guardrails.base import BaseCheck, CheckResult

logger = logging.getLogger(__name__)

# Model cache (shared across instances)
_model_cache = {}


class NSFWCheck(BaseCheck):
    """Detects NSFW/explicit content in images."""

    name = "nsfw"
    input_type = "image"
    can_reject = True
    can_redact = False

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.threshold = config.get("nsfw_threshold", 0.8)
        self.model_type = config.get("nsfw_model", "opennsfw2").lower()

    def check(self, input_data, config: Dict[str, Any]) -> CheckResult:
        """
        Run NSFW detection on image.

        Args:
            input_data: PIL Image
            config: Configuration with nsfw_threshold, nsfw_model

        Returns:
            CheckResult with NSFW score
        """
        if not self.enabled:
            return CheckResult(safe=True, score=0.0, action="allow", details={"skipped": True})

        if self.model_type == "adamcodd":
            is_safe, score = self._check_adamcodd(input_data, config)
        else:
            is_safe, score = self._check_opennsfw2(input_data, config)

        action = "allow" if is_safe else "reject"
        reason = None if is_safe else f"NSFW score {score:.2f} >= threshold {self.threshold}"

        return CheckResult(
            safe=is_safe,
            score=round(score, 4),
            action=action,
            threshold=self.threshold,
            reason=reason,
            details={"model": self.model_type}
        )

    def _check_opennsfw2(self, img, config: Dict[str, Any]) -> Tuple[bool, float]:
        """Run NSFW detection using OpenNSFW2."""
        try:
            import opennsfw2 as n2
        except ImportError:
            logger.warning("opennsfw2 not installed, skipping NSFW check")
            return True, 0.0

        # Check for local weights file
        home = Path.home()
        weights_path = home / ".opennsfw2" / "weights" / "open_nsfw_weights.h5"

        if not weights_path.exists():
            logger.warning(f"NSFW weights not found at {weights_path}")
            logger.warning("Download from: https://github.com/bhky/opennsfw2/releases/download/v0.1.0/open_nsfw_weights.h5")
            return True, 0.0

        try:
            score = float(n2.predict_image(img, weights_path=str(weights_path)))
            is_safe = score < self.threshold
            logger.info(f"NSFW score: {score:.4f} (threshold: {self.threshold})")
            return is_safe, score
        except Exception as e:
            logger.warning(f"NSFW check failed: {e}")
            return True, 0.0

    def _check_adamcodd(self, img, config: Dict[str, Any]) -> Tuple[bool, float]:
        """Run NSFW detection using AdamCodd/vit-base-nsfw-detector."""
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            import torch
        except ImportError:
            logger.warning("transformers/torch not installed, skipping NSFW check")
            return True, 0.0

        model_name = "AdamCodd/vit-base-nsfw-detector"

        if "nsfw_adamcodd_model" not in _model_cache:
            logger.info("Loading AdamCodd NSFW model...")
            try:
                _model_cache["nsfw_adamcodd_processor"] = AutoImageProcessor.from_pretrained(
                    model_name, local_files_only=True
                )
                _model_cache["nsfw_adamcodd_model"] = AutoModelForImageClassification.from_pretrained(
                    model_name, local_files_only=True
                )
                logger.info("Loaded AdamCodd NSFW model from local cache")
            except Exception:
                logger.info("Model not in local cache, trying to download...")
                try:
                    _model_cache["nsfw_adamcodd_processor"] = AutoImageProcessor.from_pretrained(model_name)
                    _model_cache["nsfw_adamcodd_model"] = AutoModelForImageClassification.from_pretrained(model_name)
                except Exception as e:
                    logger.warning(f"Cannot load AdamCodd model: {e}")
                    _model_cache["nsfw_adamcodd_unavailable"] = True
                    return True, 0.0

        if _model_cache.get("nsfw_adamcodd_unavailable"):
            return True, 0.0

        processor = _model_cache["nsfw_adamcodd_processor"]
        model = _model_cache["nsfw_adamcodd_model"]

        try:
            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)

            nsfw_score = float(probs[0][1])
            is_safe = nsfw_score < self.threshold
            logger.info(f"AdamCodd NSFW score: {nsfw_score:.4f} (threshold: {self.threshold})")
            return is_safe, nsfw_score
        except Exception as e:
            logger.warning(f"AdamCodd NSFW check failed: {e}")
            return True, 0.0
