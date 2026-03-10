"""
nsfw.py - NSFW content detection check.

Supports two models:
- OpenNSFW2 (default): ResNet-50 based, ~24MB, ~90% accuracy
- AdamCodd: ViT based, ~330MB, ~96.5% accuracy
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from guardrails.base import BaseCheck, CheckResult, fail_result
from guardrails import model_cache

logger = logging.getLogger(__name__)


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

        fail_closed = config.get("fail_closed", False)

        if self.model_type == "adamcodd":
            result = self._check_adamcodd(input_data, config, fail_closed)
        else:
            result = self._check_opennsfw2(input_data, config, fail_closed)

        # If helper returned a CheckResult (error case), return it directly
        if isinstance(result, CheckResult):
            return result

        is_safe, score = result
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

    def _check_opennsfw2(self, img, config: Dict[str, Any], fail_closed: bool) -> Union[Tuple[bool, float], CheckResult]:
        """Run NSFW detection using OpenNSFW2."""
        try:
            import opennsfw2 as n2
        except ImportError:
            return fail_result(self.name, "opennsfw2 not installed", fail_closed)

        # Check for local weights file - use config path if provided
        model_paths = config.get("model_paths", {})
        if model_paths.get("opennsfw2"):
            # Support ~ and environment variables (Windows: %USERPROFILE%)
            path_str = os.path.expandvars(model_paths["opennsfw2"])
            weights_path = Path(path_str).expanduser()
        else:
            home = Path.home()
            weights_path = home / ".opennsfw2" / "weights" / "open_nsfw_weights.h5"

        if not weights_path.exists():
            logger.warning(f"NSFW weights not found at {weights_path}")
            logger.warning("Download from: https://github.com/bhky/opennsfw2/releases/download/v0.1.0/open_nsfw_weights.h5")
            return fail_result(self.name, f"weights not found at {weights_path}", fail_closed)

        try:
            score = float(n2.predict_image(img, weights_path=str(weights_path)))
            is_safe = score < self.threshold
            logger.info(f"NSFW score: {score:.4f} (threshold: {self.threshold})")
            return is_safe, score
        except Exception as e:
            return fail_result(self.name, str(e), fail_closed)

    def _check_adamcodd(self, img, config: Dict[str, Any], fail_closed: bool) -> Union[Tuple[bool, float], CheckResult]:
        """Run NSFW detection using AdamCodd/vit-base-nsfw-detector."""
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            import torch
        except ImportError:
            return fail_result(self.name, "transformers/torch not installed", fail_closed)

        # Use config path if provided, otherwise use HuggingFace model name
        model_paths = config.get("model_paths", {})
        model_name = model_paths.get("adamcodd") or "AdamCodd/vit-base-nsfw-detector"

        # Expand user path and env vars if local path provided
        if model_paths.get("adamcodd"):
            model_name = os.path.expandvars(model_name)
            model_name = str(Path(model_name).expanduser())

        if not model_cache.has("nsfw_adamcodd_model"):
            logger.info(f"Loading AdamCodd NSFW model from: {model_name}")

            # Check if FP16 is enabled (reduces memory by 50%, faster inference)
            use_fp16 = config.get("use_fp16", False)
            dtype_kwargs = {}
            if use_fp16:
                dtype_kwargs["torch_dtype"] = torch.float16
                logger.info("Using FP16 precision (half memory, faster inference)")

            try:
                model_cache.set("nsfw_adamcodd_processor", AutoImageProcessor.from_pretrained(
                    model_name, local_files_only=True
                ))
                model_cache.set("nsfw_adamcodd_model", AutoModelForImageClassification.from_pretrained(
                    model_name, local_files_only=True, **dtype_kwargs
                ))
                logger.info("Loaded AdamCodd NSFW model from local cache")
            except Exception:
                logger.info("Model not in local cache, trying to download...")
                try:
                    model_cache.set("nsfw_adamcodd_processor", AutoImageProcessor.from_pretrained(model_name))
                    model_cache.set("nsfw_adamcodd_model", AutoModelForImageClassification.from_pretrained(model_name, **dtype_kwargs))
                except Exception as e:
                    logger.warning(f"Cannot load AdamCodd model: {e}")
                    model_cache.set("nsfw_adamcodd_unavailable", True)
                    return fail_result(self.name, f"model unavailable: {e}", fail_closed)

        if model_cache.get("nsfw_adamcodd_unavailable"):
            return fail_result(self.name, "model unavailable", fail_closed)

        processor = model_cache.get("nsfw_adamcodd_processor")
        model = model_cache.get("nsfw_adamcodd_model")

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
            return fail_result(self.name, str(e), fail_closed)
