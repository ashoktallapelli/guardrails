"""
violence.py - Violence and weapons detection check.

Uses CLIP zero-shot classification to detect:
- Violence, gore, blood, injury
- Weapons, guns, knives
- Disturbing content
"""

import logging
from typing import Any, Dict

from guardrails.base import BaseCheck, CheckResult

logger = logging.getLogger(__name__)

# Model cache (shared across instances)
_model_cache = {}


class ViolenceCheck(BaseCheck):
    """Detects violence, weapons, and disturbing content using CLIP."""

    name = "violence"
    input_type = "image"
    can_reject = True
    can_redact = False

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.threshold = config.get("violence_threshold", 0.7)

    def check(self, input_data, config: Dict[str, Any]) -> CheckResult:
        """
        Run violence/weapons detection using CLIP zero-shot classification.

        Args:
            input_data: PIL Image
            config: Configuration with violence_threshold

        Returns:
            CheckResult with violence scores
        """
        if not self.enabled:
            return CheckResult(safe=True, score=0.0, action="allow", details={"skipped": True})

        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
        except ImportError:
            logger.warning("transformers/torch not installed, skipping violence check")
            return CheckResult(
                safe=True,
                score=0.0,
                action="allow",
                details={"skipped": True, "reason": "dependencies not installed"}
            )

        # Load CLIP model
        model_name = "openai/clip-vit-base-patch32"

        if "clip_model" not in _model_cache:
            logger.info("Loading CLIP model for safety classification...")
            try:
                _model_cache["clip_model"] = CLIPModel.from_pretrained(model_name, local_files_only=True)
                _model_cache["clip_processor"] = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
                logger.info("Loaded CLIP model from local cache")
            except Exception:
                logger.info("Model not in local cache, trying to download...")
                try:
                    _model_cache["clip_model"] = CLIPModel.from_pretrained(model_name)
                    _model_cache["clip_processor"] = CLIPProcessor.from_pretrained(model_name)
                except Exception as e:
                    logger.warning(f"Cannot load CLIP model: {e}")
                    _model_cache["clip_unavailable"] = True
                    return CheckResult(
                        safe=True,
                        score=0.0,
                        action="allow",
                        details={"skipped": True, "reason": "model unavailable"}
                    )

        if _model_cache.get("clip_unavailable"):
            return CheckResult(safe=True, score=0.0, action="allow", details={"skipped": True})

        model = _model_cache["clip_model"]
        processor = _model_cache["clip_processor"]

        # Safety labels for zero-shot classification
        safety_labels = [
            "a safe, normal photograph",
            "a document, text, screenshot, form, receipt",
            "violence, gore, blood, injury, fighting",
            "weapons, guns, knives, dangerous objects",
            "disturbing, graphic, shocking content",
        ]

        try:
            inputs = processor(
                text=safety_labels,
                images=input_data,
                return_tensors="pt",
                padding=True
            )

            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1).squeeze().tolist()

            scores = {
                "safe": round(probs[0], 4),
                "document": round(probs[1], 4),
                "violence": round(probs[2], 4),
                "weapons": round(probs[3], 4),
                "disturbing": round(probs[4], 4),
            }

            # Combined unsafe score (document is treated as safe)
            unsafe_score = scores["violence"] + scores["weapons"] + scores["disturbing"]
            is_safe = unsafe_score < self.threshold

            logger.info(
                f"Safety scores: safe={scores['safe']:.2f}, document={scores['document']:.2f}, "
                f"violence={scores['violence']:.2f}, weapons={scores['weapons']:.2f}"
            )

            action = "allow" if is_safe else "reject"
            reason = None
            if not is_safe:
                reason = f"Unsafe content detected: violence={scores['violence']:.2f}, weapons={scores['weapons']:.2f}"

            return CheckResult(
                safe=is_safe,
                score=round(unsafe_score, 4),
                action=action,
                threshold=self.threshold,
                reason=reason,
                details=scores
            )

        except Exception as e:
            logger.warning(f"Violence check failed: {e}")
            return CheckResult(
                safe=True,
                score=0.0,
                action="allow",
                details={"error": str(e)}
            )
