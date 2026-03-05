"""
base.py - Base class for all guardrail checks.

All checks must inherit from BaseCheck and implement the check() method.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CheckResult:
    """Result from a guardrail check."""
    safe: bool
    score: float
    action: str  # "allow", "redact", "reject"
    threshold: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None


class BaseCheck(ABC):
    """
    Base class for all guardrail checks.

    Attributes:
        name: Unique identifier for the check
        input_type: "image", "text", or "both"
        can_reject: Whether this check can reject content
        can_redact: Whether this check can redact content
    """

    name: str = "base"
    input_type: str = "image"  # "image", "text", "both"
    can_reject: bool = True
    can_redact: bool = False

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get(f"enable_{self.name}", True)
        self.threshold = config.get(f"{self.name}_threshold", 0.5)

    @abstractmethod
    def check(self, input_data, config: Dict[str, Any]) -> CheckResult:
        """
        Run the guardrail check.

        Args:
            input_data: Image (PIL) or text (str)
            config: Configuration dictionary

        Returns:
            CheckResult with safe, score, action, and details
        """
        pass

    def redact(self, input_data, config: Dict[str, Any]):
        """
        Apply redaction to input data.
        Override this method if can_redact=True.

        Args:
            input_data: Image (PIL) or text (str)
            config: Configuration dictionary

        Returns:
            Redacted input data
        """
        return input_data

    def get_reason(self, result: CheckResult) -> str:
        """Generate human-readable reason for the decision."""
        if result.reason:
            return result.reason
        if not result.safe:
            return f"{self.name}: score {result.score:.2f} >= threshold {result.threshold}"
        return f"{self.name}: passed"
