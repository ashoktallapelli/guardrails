"""
hate_symbols.py - Hate symbol detection check.

Uses CLIP zero-shot classification to detect:
- Hate symbols and extremist imagery
- Nazi symbols, swastika
- Racist symbols
"""

import logging
from typing import Any, Dict

from guardrails.base import BaseCheck, CheckResult

logger = logging.getLogger(__name__)

# Model cache (shared with violence check)
_model_cache = {}


class HateSymbolsCheck(BaseCheck):
    """Detects hate symbols and extremist imagery using CLIP."""

    name = "hate_symbols"
    input_type = "image"
    can_reject = True
    can_redact = False

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.threshold = config.get("hate_symbol_threshold", 0.75)

    def check(self, input_data, config: Dict[str, Any]) -> CheckResult:
        """
        Run hate symbol detection using CLIP zero-shot classification.

        Args:
            input_data: PIL Image
            config: Configuration with hate_symbol_threshold

        Returns:
            CheckResult with hate symbol scores
        """
        if not self.enabled:
            return CheckResult(safe=True, score=0.0, action="allow", details={"skipped": True})

        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
        except ImportError:
            logger.warning("transformers/torch not installed, skipping hate symbol check")
            return CheckResult(
                safe=True,
                score=0.0,
                action="allow",
                details={"skipped": True, "reason": "dependencies not installed"}
            )

        # Load CLIP model (shared with violence check)
        model_name = "openai/clip-vit-base-patch32"

        if "clip_model" not in _model_cache:
            logger.info("Loading CLIP model for hate symbol detection...")
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

        # Hate symbol labels for zero-shot classification
        hate_labels = [
            "a normal, safe photograph without any symbols",
            "a document, text, screenshot, form, receipt",
            "hate symbols, extremist imagery, or offensive symbols",
            "nazi symbols, swastika, or white supremacist imagery",
            "confederate flag or racist symbols",
        ]

        try:
            inputs = processor(
                text=hate_labels,
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
                "hate_symbols": round(probs[2], 4),
                "nazi_symbols": round(probs[3], 4),
                "racist_symbols": round(probs[4], 4),
            }

            # Combined hate score (document is treated as safe)
            hate_score = scores["hate_symbols"] + scores["nazi_symbols"] + scores["racist_symbols"]
            scores["combined_hate_score"] = round(hate_score, 4)
            is_safe = hate_score < self.threshold

            logger.info(
                f"Hate symbol scores: safe={scores['safe']:.2f}, document={scores['document']:.2f}, "
                f"combined_hate={hate_score:.2f}"
            )

            action = "allow" if is_safe else "reject"
            reason = None
            if not is_safe:
                reason = f"Hate symbols detected: combined_score={hate_score:.2f}"

            return CheckResult(
                safe=is_safe,
                score=round(hate_score, 4),
                action=action,
                threshold=self.threshold,
                reason=reason,
                details=scores
            )

        except Exception as e:
            logger.warning(f"Hate symbol check failed: {e}")
            return CheckResult(
                safe=True,
                score=0.0,
                action="allow",
                details={"error": str(e)}
            )
