"""
hate_symbols.py - Hate symbol detection check.

Uses CLIP zero-shot classification to detect:
- Hate symbols and extremist imagery
- Nazi symbols, swastika
- Racist symbols
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

from guardrails.base import BaseCheck, CheckResult, fail_result
from guardrails import model_cache

logger = logging.getLogger(__name__)


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

        fail_closed = config.get("fail_closed", False)

        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
        except ImportError:
            return fail_result(self.name, "transformers/torch not installed", fail_closed)

        # Load CLIP model (shared with violence check) - use config path if provided
        model_paths = config.get("model_paths", {})
        model_name = model_paths.get("clip") or "openai/clip-vit-base-patch32"

        # Expand user path and env vars if local path provided (Windows: %USERPROFILE%)
        if model_paths.get("clip"):
            model_name = os.path.expandvars(model_name)
            model_name = str(Path(model_name).expanduser())

        if not model_cache.has("clip_model"):
            logger.info(f"Loading CLIP model from: {model_name}")

            # Check if FP16 is enabled (reduces memory by 50%, faster inference)
            use_fp16 = config.get("use_fp16", False)
            dtype_kwargs = {}
            if use_fp16:
                dtype_kwargs["torch_dtype"] = torch.float16
                logger.info("Using FP16 precision (half memory, faster inference)")

            try:
                model_cache.set("clip_model", CLIPModel.from_pretrained(model_name, local_files_only=True, **dtype_kwargs))
                model_cache.set("clip_processor", CLIPProcessor.from_pretrained(model_name, local_files_only=True))
                logger.info("Loaded CLIP model from local cache")
            except Exception:
                logger.info("Model not in local cache, trying to download...")
                try:
                    model_cache.set("clip_model", CLIPModel.from_pretrained(model_name, **dtype_kwargs))
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

        # Hate symbol labels for zero-shot classification
        # Load from config or use defaults
        default_labels = [
            "a safe, normal photograph without symbols",
            "a document, text, form, screenshot",
            "nazi swastika tilted on red background, SS bolts, hitler salute",
            "KKK imagery, white hood, burning cross, racist symbols",
            "ISIS flag, terrorist symbols, extremist imagery",
        ]
        hate_labels = config.get("hate_labels", default_labels)

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

            # Build scores dict dynamically from labels
            # First 2 labels are safe categories, rest are unsafe
            scores = {}
            for i, label in enumerate(hate_labels):
                # Create key from first word of label
                key = label.split(",")[0].split()[0].lower().replace("-", "_")
                if i == 0:
                    key = "safe"
                elif i == 1:
                    key = "document"
                scores[key] = round(probs[i], 4)

            # Combined hate score (skip first 2 safe categories)
            hate_score = sum(probs[2:])
            scores["combined_hate_score"] = round(hate_score, 4)
            is_safe = hate_score < self.threshold

            # Log scores (exclude safe categories)
            unsafe_categories = {k: v for k, v in scores.items() if k not in ["safe", "document"]}
            logger.info(f"Hate symbol scores: safe={scores['safe']:.2f}, unsafe={unsafe_categories}")

            action = "allow" if is_safe else "reject"
            reason = None
            if not is_safe:
                # Find the highest scoring category
                unsafe_categories = {k: v for k, v in scores.items() if k not in ["safe", "document"]}
                top_category = max(unsafe_categories, key=unsafe_categories.get)
                reason = f"Hate symbols detected: {top_category}={scores[top_category]:.2f}, total={hate_score:.2f}"

            return CheckResult(
                safe=is_safe,
                score=round(hate_score, 4),
                action=action,
                threshold=self.threshold,
                reason=reason,
                details=scores
            )

        except Exception as e:
            return fail_result(self.name, str(e), fail_closed)
