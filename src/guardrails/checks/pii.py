"""
pii.py - PII detection check.

Handles both image PII (OCR + Presidio) and text PII:
- Extracts text from images using Tesseract OCR
- Detects PII entities using Presidio Analyzer
- Returns REJECT if PII is detected
"""

import logging
from typing import Any, Dict

from guardrails.base import BaseCheck, CheckResult, fail_result
from guardrails import model_cache

logger = logging.getLogger(__name__)


class PIICheck(BaseCheck):
    """Detects PII in images and text."""

    name = "pii"
    input_type = "both"  # Works on both images and text
    can_reject = True    # PII triggers rejection
    can_redact = False

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.threshold = config.get("pii_score_threshold", 0.35)

    def check(self, input_data, config: Dict[str, Any]) -> CheckResult:
        """
        Detect PII in image (using OCR) or text.

        Args:
            input_data: PIL Image or text string
            config: Configuration with pii_score_threshold, pii_language

        Returns:
            CheckResult with PII entity count
        """
        if not self.enabled:
            return CheckResult(safe=True, score=0.0, action="allow", details={"skipped": True})

        self._fail_closed = config.get("fail_closed", False)

        # Determine if input is image or text
        if isinstance(input_data, str):
            return self._check_text(input_data, config)
        else:
            return self._check_image(input_data, config)

    def _check_image(self, img, config: Dict[str, Any]) -> CheckResult:
        """Detect PII in image using OCR + Presidio."""
        try:
            import pytesseract
            from presidio_analyzer import AnalyzerEngine
        except ImportError:
            return fail_result(self.name, "pytesseract or presidio-analyzer not installed", self._fail_closed)

        try:
            # Extract text using OCR
            extracted_text = pytesseract.image_to_string(img)

            if not extracted_text.strip():
                return CheckResult(
                    safe=True,
                    score=0.0,
                    action="allow",
                    details={"text_found": False, "entity_count": 0}
                )

            # Analyze text for PII
            if not model_cache.has("text_analyzer"):
                model_cache.set("text_analyzer", AnalyzerEngine())

            analyzer = model_cache.get("text_analyzer")
            language = config.get("pii_language", "en")
            results = analyzer.analyze(
                text=extracted_text,
                language=language,
                score_threshold=self.threshold
            )

            entities = []
            for result in results:
                entities.append({
                    "type": result.entity_type,
                    "text": extracted_text[result.start:result.end],
                    "score": round(result.score, 4),
                    "start": result.start,
                    "end": result.end
                })

            entity_count = len(entities)
            logger.info(f"PII detection: found {entity_count} entities")

            is_safe = entity_count == 0
            action = "allow" if is_safe else "reject"
            entity_types = list(set(e["type"] for e in entities))
            reason = None if is_safe else f"PII detected: {entity_count} entities ({', '.join(entity_types)})"

            return CheckResult(
                safe=is_safe,
                score=float(entity_count),
                action=action,
                reason=reason,
                details={
                    "text_found": True,
                    "entity_count": entity_count,
                    "entities": entities,
                    "text_length": len(extracted_text)
                }
            )

        except Exception as e:
            return fail_result(self.name, str(e), self._fail_closed)

    def _check_text(self, text: str, config: Dict[str, Any]) -> CheckResult:
        """Detect PII in text using Presidio."""
        try:
            from presidio_analyzer import AnalyzerEngine
        except ImportError:
            return fail_result(self.name, "presidio-analyzer not installed", self._fail_closed)

        if not text or not text.strip():
            return CheckResult(
                safe=True,
                score=0.0,
                action="allow",
                details={"entity_count": 0}
            )

        try:
            if not model_cache.has("text_analyzer"):
                model_cache.set("text_analyzer", AnalyzerEngine())

            analyzer = model_cache.get("text_analyzer")
            language = config.get("pii_language", "en")
            results = analyzer.analyze(
                text=text,
                language=language,
                score_threshold=self.threshold
            )

            entities = []
            for result in results:
                entities.append({
                    "type": result.entity_type,
                    "text": text[result.start:result.end],
                    "score": round(result.score, 4),
                    "start": result.start,
                    "end": result.end
                })

            entity_count = len(entities)
            logger.info(f"Text PII detection: found {entity_count} entities")

            is_safe = entity_count == 0
            action = "allow" if is_safe else "reject"
            entity_types = list(set(e["type"] for e in entities))
            reason = None if is_safe else f"PII detected: {entity_count} entities ({', '.join(entity_types)})"

            return CheckResult(
                safe=is_safe,
                score=float(entity_count),
                action=action,
                reason=reason,
                details={
                    "entity_count": entity_count,
                    "entities": entities,
                    "text_length": len(text)
                }
            )

        except Exception as e:
            return fail_result(self.name, str(e), self._fail_closed)

    def redact(self, input_data, config: Dict[str, Any]):
        """
        Redact PII from image or text.

        Args:
            input_data: PIL Image or text string
            config: Configuration

        Returns:
            Redacted image or anonymized text
        """
        if isinstance(input_data, str):
            return self._redact_text(input_data, config)
        else:
            return self._redact_image(input_data, config)

    def _redact_image(self, img, config: Dict[str, Any]):
        """Redact PII from image using Presidio Image Redactor."""
        try:
            from presidio_image_redactor import ImageRedactorEngine
        except ImportError:
            logger.warning("presidio-image-redactor not installed, skipping PII redaction")
            return img

        try:
            engine = ImageRedactorEngine()
            redacted = engine.redact(img, score_threshold=self.threshold)
            logger.info(f"PII redaction completed (threshold: {self.threshold})")
            return redacted
        except Exception as e:
            logger.warning(f"PII redaction failed: {e}")
            return img

    def _redact_text(self, text: str, config: Dict[str, Any]) -> str:
        """Anonymize PII in text using Presidio Anonymizer."""
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            from presidio_anonymizer.entities import OperatorConfig
        except ImportError:
            logger.warning("presidio packages not installed, skipping text anonymization")
            return text

        if not text or not text.strip():
            return text

        try:
            if not model_cache.has("text_analyzer"):
                model_cache.set("text_analyzer", AnalyzerEngine())
            if not model_cache.has("text_anonymizer"):
                model_cache.set("text_anonymizer", AnonymizerEngine())

            analyzer = model_cache.get("text_analyzer")
            anonymizer = model_cache.get("text_anonymizer")

            language = config.get("pii_language", "en")
            default_operator = config.get("pii_operator", "replace")

            results = analyzer.analyze(
                text=text,
                language=language,
                score_threshold=self.threshold
            )

            # Build operators
            operators = {}
            for entity_type in set(r.entity_type for r in results):
                operators[entity_type] = OperatorConfig(
                    default_operator,
                    {"new_value": f"<{entity_type}>"}
                )

            anonymized = anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators=operators
            )

            logger.info(f"Text PII anonymization: processed {len(results)} entities")
            return anonymized.text

        except Exception as e:
            logger.warning(f"Text PII anonymization failed: {e}")
            return text

    def get_reason(self, result: CheckResult) -> str:
        """Generate human-readable reason."""
        if result.reason:
            return result.reason
        entity_count = result.details.get("entity_count", 0)
        if entity_count > 0:
            return f"PII detected: {entity_count} entities"
        return "No PII detected"
