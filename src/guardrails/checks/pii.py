"""
pii.py - PII detection check using AI-Force Platform.

Handles image PII detection:
- Extracts text from images using Tesseract OCR
- Sends text to AI-Force /scan/prompt endpoint for PII detection
- Returns REJECT if PII is detected
"""

import logging
import requests
import urllib3
from typing import Any, Dict

from guardrails.base import BaseCheck, CheckResult, fail_result

# Disable SSL warnings (for corporate environments)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class PIICheck(BaseCheck):
    """Detects PII in images using OCR + AI-Force API."""

    name = "pii"
    input_type = "image"
    can_reject = True
    can_redact = False

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # AI-Force API configuration
        pii_config = config.get("pii_api", {})
        self.api_url = pii_config.get("url", "https://aiforce.hcltech.com/sgs")
        self.api_token = pii_config.get("token", "")
        self.security_group = pii_config.get("security_group", "security_group_0")
        self.verify_ssl = pii_config.get("verify_ssl", False)
        self.timeout = pii_config.get("timeout", 30)

    def check(self, input_data, config: Dict[str, Any]) -> CheckResult:
        """
        Detect PII in image using OCR + AI-Force API.

        Args:
            input_data: PIL Image
            config: Configuration dictionary

        Returns:
            CheckResult with PII detection results
        """
        if not self.enabled:
            return CheckResult(safe=True, score=0.0, action="allow", details={"skipped": True})

        # Validate API token
        if not self.api_token:
            return fail_result(self.name, "Missing pii_api.token in config.yaml")

        return self._check_image(input_data, config)

    def _check_image(self, img, config: Dict[str, Any]) -> CheckResult:
        """Detect PII in image using OCR + AI-Force API."""
        try:
            import pytesseract
        except ImportError:
            return fail_result(self.name, "pytesseract not installed")

        try:
            # Extract text using OCR
            extracted_text = pytesseract.image_to_string(img)

            if not extracted_text.strip():
                return CheckResult(
                    safe=True,
                    score=0.0,
                    action="allow",
                    details={"text_found": False, "pii_detected": False}
                )

            # Send to AI-Force API for PII detection
            return self._call_aiforce_api(extracted_text)

        except Exception as e:
            return fail_result(self.name, f"OCR error: {e}")

    def _call_aiforce_api(self, text: str) -> CheckResult:
        """Call AI-Force /scan/prompt endpoint for PII detection."""
        url = f"{self.api_url}/scan/prompt"

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "prompt_name": "pii_scan",
            "input_prompt": text,
            "variables": {},
            "security_group": self.security_group
        }

        # Log request (no sensitive data)
        logger.info(f"AI-Force API request: url={url}, security_group={self.security_group}, text_length={len(text)}")

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                verify=self.verify_ssl,
                timeout=self.timeout
            )

            # Log response status
            logger.info(f"AI-Force API response: status_code={response.status_code}")

            response.raise_for_status()
            data = response.json()

            # Parse AI-Force response
            pii_result = data.get("results", {}).get("Detect PII", {})
            is_pii_pass = pii_result.get("is_pass", True)
            pii_score = pii_result.get("score", 0.0)

            is_safe = is_pii_pass
            action = "allow" if is_safe else "reject"

            # Build detailed results
            failed_checks = []
            for check_name, check_data in data.get("results", {}).items():
                if not check_data.get("is_pass", True):
                    failed_checks.append(check_name)

            reason = None
            if not is_safe:
                reason = f"PII detected by AI-Force: {', '.join(failed_checks)}"

            logger.info(f"AI-Force PII result: pii_detected={not is_pii_pass}, score={pii_score}, failed_checks={failed_checks}")

            return CheckResult(
                safe=is_safe,
                score=pii_score,
                action=action,
                threshold=pii_result.get("threshold", 0.8),
                reason=reason,
                details={
                    "text_found": True,
                    "pii_detected": not is_pii_pass,
                    "is_redacted": data.get("is_redacted", False),
                    "sanitized_text": data.get("sanitized_text", ""),
                    "failed_checks": failed_checks,
                    "api_response": data
                }
            )

        except requests.exceptions.Timeout:
            logger.error(f"AI-Force API timeout after {self.timeout}s")
            return fail_result(self.name, "AI-Force API timeout")

        except requests.exceptions.RequestException as e:
            logger.error(f"AI-Force API request failed: {type(e).__name__}")
            return fail_result(self.name, f"AI-Force API error: {e}")

        except Exception as e:
            logger.error(f"AI-Force API unexpected error: {type(e).__name__}")
            return fail_result(self.name, f"Unexpected error: {e}")

    def get_reason(self, result: CheckResult) -> str:
        """Generate human-readable reason."""
        if result.reason:
            return result.reason
        if result.details.get("pii_detected"):
            return "PII detected by AI-Force"
        return "No PII detected"
