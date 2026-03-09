"""
pipeline.py - Orchestrates guardrail checks execution.

Runs checks in configured order and makes final decision:
- REJECT: Unsafe content detected (stops immediately)
- ALLOW: Safe and clean
"""

import io
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from guardrails.base import BaseCheck, CheckResult
from guardrails.checks import (
    FileValidationCheck,
    NSFWCheck,
    ViolenceCheck,
    HateSymbolsCheck,
    PIICheck,
    FacesCheck,
)

logger = logging.getLogger(__name__)

# Available checks mapping
AVAILABLE_CHECKS = {
    "file_validation": FileValidationCheck,
    "nsfw": NSFWCheck,
    "violence": ViolenceCheck,
    "hate_symbols": HateSymbolsCheck,
    "pii": PIICheck,
    "faces": FacesCheck,
}


class Pipeline:
    """
    Orchestrates execution of guardrail checks.

    Usage:
        config = load_config()
        pipeline = Pipeline(config)
        result = pipeline.run(image)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline with configuration.

        Args:
            config: Configuration dictionary with 'guardrails' list
        """
        self.config = config
        self.checks: List[BaseCheck] = []
        self._load_checks()

    def _load_checks(self):
        """Load enabled checks from configuration."""
        # Default check order if not specified
        default_checks = [
            "file_validation",
            "nsfw",
            "violence",
            "hate_symbols",
            "pii",
            "faces",
        ]

        enabled = self.config.get("guardrails", default_checks)

        for name in enabled:
            if name in AVAILABLE_CHECKS:
                check_class = AVAILABLE_CHECKS[name]
                self.checks.append(check_class(self.config))
                logger.debug(f"Loaded check: {name}")
            else:
                logger.warning(f"Unknown check: {name}")

        logger.info(f"Pipeline loaded {len(self.checks)} checks: {[c.name for c in self.checks]}")

    def run(self, input_data, input_type: str = "image") -> Dict[str, Any]:
        """
        Run all checks on input data.

        Args:
            input_data: Image (PIL) or text (str)
            input_type: "image" or "text"

        Returns:
            Result dictionary with:
            - decision: ALLOW or REJECT
            - reasons: List of human-readable reasons
            - checks: Individual check results
            - is_safe: Boolean
            - output: Processed data (if not rejected)
        """
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision": "ALLOW",
            "reasons": [],
            "checks": {},
            "is_safe": True,
        }

        for check in self.checks:
            # Skip if wrong input type
            if check.input_type != input_type and check.input_type != "both":
                continue

            # Skip if check is disabled
            if not check.enabled:
                result["checks"][check.name] = {"skipped": True}
                continue

            # Run the check
            check_result = check.check(input_data, self.config)
            result["checks"][check.name] = {
                "safe": check_result.safe,
                "score": check_result.score,
                "action": check_result.action,
                "threshold": check_result.threshold,
                "details": check_result.details,
            }

            # Handle rejection (stop immediately)
            if check_result.action == "reject":
                result["decision"] = "REJECT"
                result["is_safe"] = False
                result["reasons"].append(check.get_reason(check_result))
                logger.warning(f"REJECT: {check.get_reason(check_result)}")
                return result

        # All checks passed
        result["reasons"].append("All checks passed")
        result["output"] = input_data
        logger.info(f"Pipeline result: {result['decision']}")
        return result

    def run_image(self, image_path_or_pil, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Convenience method to run pipeline on image file.

        Args:
            image_path_or_pil: Path to image or PIL Image
            output_path: Optional path to save output

        Returns:
            Result dictionary
        """
        from pathlib import Path
        from PIL import Image

        # Load image if path provided
        if isinstance(image_path_or_pil, (str, Path)):
            path = Path(image_path_or_pil)

            # Run file validation first
            file_check = FileValidationCheck(self.config)
            file_result = file_check.check(path, self.config)

            if file_result.action == "reject":
                return {
                    "decision": "REJECT",
                    "reasons": [file_check.get_reason(file_result)],
                    "checks": {"file_validation": file_result.details},
                    "is_safe": False,
                }

            img = Image.open(path).convert("RGB")
            input_path = str(path)
        else:
            img = image_path_or_pil
            input_path = "memory"

        # Run pipeline
        result = self.run(img, input_type="image")
        result["input_path"] = input_path

        # Strip EXIF if not rejected
        if result["decision"] != "REJECT":
            result["output"] = self._strip_exif(result["output"])

        # Save output if path provided
        if output_path and result["decision"] != "REJECT":
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            result["output"].save(output, format="JPEG", quality=self.config.get("output_quality", 95))
            result["output_path"] = str(output)

        return result

    def run_text(self, text: str, anonymize: bool = True) -> Dict[str, Any]:
        """
        Convenience method to run pipeline on text.

        Args:
            text: Text to check
            anonymize: Whether to anonymize PII

        Returns:
            Result dictionary
        """
        result = self.run(text, input_type="text")
        result["input_type"] = "text"
        result["original_length"] = len(text)

        if anonymize and result.get("output"):
            result["anonymized_text"] = result["output"]

        return result

    def _strip_exif(self, img):
        """Strip EXIF metadata from image."""
        from PIL import Image

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95, exif=b"")
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    @classmethod
    def list_available_checks(cls) -> List[str]:
        """List all available check names."""
        return list(AVAILABLE_CHECKS.keys())
