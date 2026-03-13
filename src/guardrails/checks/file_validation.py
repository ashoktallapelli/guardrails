"""
file_validation.py - File type and size validation check.

Validates files using magic bytes (not extension) for security.
"""

import logging
from pathlib import Path
from typing import Any, Dict

from guardrails.base import BaseCheck, CheckResult, fail_result

logger = logging.getLogger(__name__)


class FileValidationCheck(BaseCheck):
    """Validates file type (magic bytes) and size."""

    name = "file_validation"
    input_type = "image"
    can_reject = True
    can_redact = False

    def check(self, input_data, config: Dict[str, Any]) -> CheckResult:
        """
        Validate file by content (magic bytes), not extension.

        Args:
            input_data: Path to file (str or Path). If PIL Image, skip validation.
            config: Configuration with max_file_size_mb, allowed_mime_types

        Returns:
            CheckResult with validation status
        """
        if not self.enabled:
            return CheckResult(safe=True, score=0.0, action="allow", details={"skipped": True})

        # Skip if input is already a PIL Image (already loaded)
        if not isinstance(input_data, (str, Path)):
            return CheckResult(
                safe=True,
                score=0.0,
                action="allow",
                details={"skipped": True, "reason": "already loaded image"}
            )

        path = Path(input_data) if isinstance(input_data, str) else input_data

        try:
            import magic
        except ImportError:
            return fail_result(self.name, "python-magic not installed")

        try:
            # Check file size
            max_bytes = config.get("max_file_size_mb", 10) * 1024 * 1024
            size = path.stat().st_size

            if size > max_bytes:
                reason = f"File too large: {size / (1024*1024):.2f} MB > {config.get('max_file_size_mb', 10)} MB"
                logger.warning(f"REJECT: {reason}")
                return CheckResult(
                    safe=False,
                    score=1.0,
                    action="reject",
                    reason=reason,
                    details={"size_bytes": size, "max_bytes": max_bytes}
                )

            if size == 0:
                reason = "File is empty"
                logger.warning(f"REJECT: {reason}")
                return CheckResult(
                    safe=False,
                    score=1.0,
                    action="reject",
                    reason=reason
                )

            # Check MIME type
            mime = magic.from_file(str(path), mime=True)
            allowed = config.get("allowed_mime_types", ["image/jpeg", "image/png", "image/webp", "image/gif"])

            if mime not in allowed:
                reason = f"Disallowed MIME type: {mime}"
                logger.warning(f"REJECT: {reason}")
                return CheckResult(
                    safe=False,
                    score=1.0,
                    action="reject",
                    reason=reason,
                    details={"mime_type": mime, "allowed": allowed}
                )

            logger.info(f"File validation passed: {mime}, {size} bytes")
            return CheckResult(
                safe=True,
                score=0.0,
                action="allow",
                details={"mime_type": mime, "size_bytes": size}
            )

        except Exception as e:
            return fail_result(self.name, str(e))
