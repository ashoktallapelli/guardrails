"""
violence.py - Violence and weapons detection check.

Uses CLIP zero-shot classification to detect:
- Violence, gore, blood, injury
- Weapons, guns, knives
- Disturbing content
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

from guardrails.base import BaseCheck, CheckResult, fail_result
from guardrails import model_cache

logger = logging.getLogger(__name__)


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

        fail_closed = config.get("fail_closed", False)

        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
        except ImportError:
            return fail_result(self.name, "transformers/torch not installed", fail_closed)

        # Load CLIP model - use config path if provided
        model_paths = config.get("model_paths", {})
        model_name = model_paths.get("clip") or "openai/clip-vit-base-patch32"

        # Expand user path and env vars if local path provided (Windows: %USERPROFILE%)
        if model_paths.get("clip"):
            model_name = os.path.expandvars(model_name)
            model_name = str(Path(model_name).expanduser())

        if not model_cache.has("clip_model"):
            logger.info(f"Loading CLIP model from: {model_name}")
            try:
                model_cache.set("clip_model", CLIPModel.from_pretrained(model_name, local_files_only=True))
                model_cache.set("clip_processor", CLIPProcessor.from_pretrained(model_name, local_files_only=True))
                logger.info("Loaded CLIP model from local cache")
            except Exception:
                logger.info("Model not in local cache, trying to download...")
                try:
                    model_cache.set("clip_model", CLIPModel.from_pretrained(model_name))
                    model_cache.set("clip_processor", CLIPProcessor.from_pretrained(model_name))
                except Exception as e:
                    logger.warning(f"Cannot load CLIP model: {e}")
                    logger.warning("Set model_paths.clip in config.yaml to specify custom path")
                    model_cache.set("clip_unavailable", True)
                    return fail_result(self.name, f"model unavailable: {e}", fail_closed)

        if model_cache.get("clip_unavailable"):
            return fail_result(self.name, "model unavailable", fail_closed)

        model = model_cache.get("clip_model")
        processor = model_cache.get("clip_processor")

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
            return fail_result(self.name, str(e), fail_closed)
