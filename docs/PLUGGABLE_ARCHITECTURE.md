# Pluggable Guardrails Architecture

## Feedback
> "We should make code pluggable - if any new guardrail needs to be added we should be able to add without much code changes"

---

## Current Architecture (Monolithic)

All guardrails are hardcoded in `image_guard.py`:

```
image_guard.py
└── run_guardrails()
    ├── check_nsfw()         # Hardcoded
    ├── check_violence()     # Hardcoded
    ├── check_hate_symbols() # Hardcoded
    ├── detect_pii()         # Hardcoded
    └── detect_faces()       # Hardcoded
```

### Problems
- Adding new guardrail requires modifying core `run_guardrails()` function
- Can't easily enable/disable guardrails without code changes
- Hard to test guardrails in isolation
- Difficult to share/reuse guardrails across projects

---

## Proposed Architecture (Pluggable)

### Folder Structure

```
guardrails/
├── __init__.py
├── base.py              # Base class for all guardrails
├── registry.py          # Plugin registry & discovery
├── pipeline.py          # Orchestrates guardrail execution
├── plugins/
│   ├── __init__.py
│   ├── nsfw.py          # NSFW detection guardrail
│   ├── violence.py      # Violence/weapons guardrail
│   ├── hate_symbols.py  # Hate symbols guardrail
│   ├── pii_image.py     # PII detection in images
│   ├── pii_text.py      # PII detection in text
│   ├── faces.py         # Face detection & blur
│   └── file_validation.py  # File type/size validation
├── config.yaml          # Enable/configure guardrails
└── image_guard.py       # Entry point (simplified)
```

---

## Implementation Details

### 1. Base Guardrail Class (`base.py`)

```python
from abc import ABC, abstractmethod
from typing import Dict, Any
from PIL import Image

class BaseGuardrail(ABC):
    """Base class for all guardrails."""

    name: str = "base"           # Unique identifier
    description: str = ""        # Human-readable description
    version: str = "1.0.0"       # Version for tracking

    # Guardrail type
    INPUT_TYPE = "image"         # "image", "text", or "both"

    # Decision behavior
    CAN_REJECT = True            # Can this guardrail reject content?
    CAN_REDACT = False           # Can this guardrail redact content?

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get(f"enable_{self.name}", True)
        self.threshold = config.get(f"{self.name}_threshold", 0.5)

    @abstractmethod
    def check(self, input_data, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the guardrail check.

        Returns:
            {
                "safe": bool,
                "score": float,
                "details": dict,
                "action": "allow" | "redact" | "reject"
            }
        """
        pass

    def redact(self, input_data, config: Dict[str, Any]):
        """Optional: Apply redaction if guardrail supports it."""
        return input_data

    def get_reason(self, result: Dict[str, Any]) -> str:
        """Generate human-readable reason for the decision."""
        return f"{self.name}: score={result.get('score', 0):.2f}"
```

### 2. Example Plugin: NSFW Guardrail (`plugins/nsfw.py`)

```python
from guardrails.base import BaseGuardrail
from typing import Dict, Any

class NSFWGuardrail(BaseGuardrail):
    name = "nsfw"
    description = "Detects NSFW/explicit content in images"
    version = "1.0.0"
    INPUT_TYPE = "image"
    CAN_REJECT = True
    CAN_REDACT = False

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_type = config.get("nsfw_model", "opennsfw2")
        self._model = None

    def _load_model(self):
        if self._model is None:
            if self.model_type == "opennsfw2":
                # Load OpenNSFW2
                pass
            elif self.model_type == "adamcodd":
                # Load AdamCodd
                pass
        return self._model

    def check(self, image, config: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return {"safe": True, "score": 0.0, "skipped": True}

        score = self._get_nsfw_score(image)
        is_safe = score < self.threshold

        return {
            "safe": is_safe,
            "score": round(score, 4),
            "threshold": self.threshold,
            "action": "reject" if not is_safe else "allow",
            "details": {"model": self.model_type}
        }

    def get_reason(self, result: Dict[str, Any]) -> str:
        if not result["safe"]:
            return f"NSFW score {result['score']:.2f} >= threshold {result['threshold']}"
        return "NSFW check passed"
```

### 3. Example Plugin: Violence Guardrail (`plugins/violence.py`)

```python
from guardrails.base import BaseGuardrail
from typing import Dict, Any

class ViolenceGuardrail(BaseGuardrail):
    name = "violence"
    description = "Detects violence, weapons, and disturbing content"
    version = "1.0.0"
    INPUT_TYPE = "image"
    CAN_REJECT = True
    CAN_REDACT = False

    def check(self, image, config: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return {"safe": True, "score": 0.0, "skipped": True}

        # CLIP-based detection
        scores = self._get_safety_scores(image)

        unsafe_score = (
            scores["violence"] +
            scores["weapons"] +
            scores["disturbing"]
        )
        is_safe = unsafe_score < self.threshold

        return {
            "safe": is_safe,
            "score": round(unsafe_score, 4),
            "threshold": self.threshold,
            "action": "reject" if not is_safe else "allow",
            "details": scores
        }

    def get_reason(self, result: Dict[str, Any]) -> str:
        if not result["safe"]:
            details = result.get("details", {})
            return f"Unsafe content detected: violence={details.get('violence', 0):.2f}, weapons={details.get('weapons', 0):.2f}"
        return "Violence check passed"
```

### 4. Example Plugin: PII Guardrail with Redaction (`plugins/pii_image.py`)

```python
from guardrails.base import BaseGuardrail
from typing import Dict, Any

class PIIImageGuardrail(BaseGuardrail):
    name = "pii_image"
    description = "Detects and redacts PII in images using OCR"
    version = "1.0.0"
    INPUT_TYPE = "image"
    CAN_REJECT = False      # PII doesn't reject, only redacts
    CAN_REDACT = True

    def check(self, image, config: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return {"safe": True, "score": 0, "skipped": True}

        entities = self._detect_pii(image)
        entity_count = len(entities)

        return {
            "safe": True,  # PII doesn't make image "unsafe"
            "score": entity_count,
            "action": "redact" if entity_count > 0 else "allow",
            "details": {
                "entity_count": entity_count,
                "entities": entities
            }
        }

    def redact(self, image, config: Dict[str, Any]):
        """Apply black boxes over detected PII."""
        # Redaction logic here
        return redacted_image

    def get_reason(self, result: Dict[str, Any]) -> str:
        count = result.get("details", {}).get("entity_count", 0)
        return f"PII redacted: {count}"
```

### 5. Plugin Registry (`registry.py`)

```python
from typing import Dict, Type, List
from guardrails.base import BaseGuardrail
import importlib
import os

class GuardrailRegistry:
    """Discovers and manages guardrail plugins."""

    _plugins: Dict[str, Type[BaseGuardrail]] = {}

    @classmethod
    def register(cls, guardrail_class: Type[BaseGuardrail]):
        """Register a guardrail plugin."""
        cls._plugins[guardrail_class.name] = guardrail_class
        return guardrail_class

    @classmethod
    def get(cls, name: str) -> Type[BaseGuardrail]:
        """Get a guardrail class by name."""
        return cls._plugins.get(name)

    @classmethod
    def list_all(cls) -> List[str]:
        """List all registered guardrails."""
        return list(cls._plugins.keys())

    @classmethod
    def discover_plugins(cls, plugin_dir: str = "plugins"):
        """Auto-discover plugins from directory."""
        for filename in os.listdir(plugin_dir):
            if filename.endswith(".py") and not filename.startswith("_"):
                module_name = filename[:-3]
                importlib.import_module(f"guardrails.plugins.{module_name}")

# Decorator for easy registration
def guardrail(cls):
    GuardrailRegistry.register(cls)
    return cls
```

### 6. Pipeline Orchestrator (`pipeline.py`)

```python
from typing import List, Dict, Any
from guardrails.registry import GuardrailRegistry
from guardrails.base import BaseGuardrail

class GuardrailPipeline:
    """Orchestrates execution of guardrails."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.guardrails: List[BaseGuardrail] = []
        self._load_guardrails()

    def _load_guardrails(self):
        """Load enabled guardrails from config."""
        enabled = self.config.get("guardrails", [
            "file_validation",
            "nsfw",
            "violence",
            "hate_symbols",
            "pii_image",
            "faces"
        ])

        for name in enabled:
            guardrail_class = GuardrailRegistry.get(name)
            if guardrail_class:
                self.guardrails.append(guardrail_class(self.config))

    def run(self, input_data, input_type: str = "image") -> Dict[str, Any]:
        """Run all guardrails on input."""
        results = {
            "decision": "ALLOW",
            "reasons": [],
            "checks": {},
            "is_safe": True,
            "is_redacted": False
        }

        processed_data = input_data

        for guardrail in self.guardrails:
            # Skip if wrong input type
            if guardrail.INPUT_TYPE != input_type and guardrail.INPUT_TYPE != "both":
                continue

            # Run check
            check_result = guardrail.check(processed_data, self.config)
            results["checks"][guardrail.name] = check_result

            # Handle rejection
            if check_result.get("action") == "reject":
                results["decision"] = "REJECT"
                results["is_safe"] = False
                results["reasons"].append(guardrail.get_reason(check_result))
                return results  # Stop on first rejection

            # Handle redaction
            if check_result.get("action") == "redact" and guardrail.CAN_REDACT:
                processed_data = guardrail.redact(processed_data, self.config)
                results["is_redacted"] = True
                results["reasons"].append(guardrail.get_reason(check_result))

        # Final decision
        if results["is_redacted"]:
            results["decision"] = "REDACT"

        if not results["reasons"]:
            results["reasons"].append("All checks passed, no redaction needed")

        results["output"] = processed_data
        return results
```

---

## Configuration (`config.yaml`)

```yaml
# Enable/disable guardrails (order matters for execution)
guardrails:
  - file_validation
  - nsfw
  - violence
  - hate_symbols
  - pii_image
  - faces

# Individual guardrail settings
nsfw_model: "opennsfw2"
nsfw_threshold: 0.80

violence_threshold: 0.70
enable_violence: true

hate_symbol_threshold: 0.75
enable_hate_symbols: true

pii_score_threshold: 0.35
enable_pii_image: true

enable_faces: true
face_blur_kernel_size: 51
```

---

## Adding a New Guardrail

### Step 1: Create Plugin File

```python
# guardrails/plugins/watermark.py
from guardrails.base import BaseGuardrail
from guardrails.registry import guardrail

@guardrail  # Auto-registers with registry
class WatermarkGuardrail(BaseGuardrail):
    name = "watermark"
    description = "Detects watermarks in images"
    INPUT_TYPE = "image"
    CAN_REJECT = False
    CAN_REDACT = True

    def check(self, image, config):
        has_watermark = self._detect_watermark(image)
        return {
            "safe": True,
            "score": 1.0 if has_watermark else 0.0,
            "action": "redact" if has_watermark else "allow"
        }

    def redact(self, image, config):
        return self._remove_watermark(image)
```

### Step 2: Enable in Config

```yaml
guardrails:
  - nsfw
  - violence
  - watermark  # Just add here!

watermark_threshold: 0.5
enable_watermark: true
```

### Step 3: Done!

No changes to core pipeline code required.

---

## Benefits

| Benefit | Description |
|---------|-------------|
| **Easy to Add** | Create file + add to config |
| **Easy to Remove** | Just remove from config |
| **Independent** | Each guardrail is self-contained |
| **Testable** | Test each guardrail in isolation |
| **Configurable** | All settings in config.yaml |
| **Reusable** | Share plugins across projects |
| **Versioned** | Track guardrail versions |

---

## Migration Path

1. **Phase 1**: Create base classes and registry (no breaking changes)
2. **Phase 2**: Migrate existing guardrails to plugins one by one
3. **Phase 3**: Update `run_guardrails()` to use pipeline
4. **Phase 4**: Deprecate old hardcoded functions

---

## Example: Custom Corporate Guardrail

```python
# guardrails/plugins/corporate_logo.py
from guardrails.base import BaseGuardrail
from guardrails.registry import guardrail

@guardrail
class CorporateLogoGuardrail(BaseGuardrail):
    """Detects if corporate logo is present (for compliance)."""

    name = "corporate_logo"
    description = "Ensures corporate logo is present in images"
    INPUT_TYPE = "image"
    CAN_REJECT = True

    def check(self, image, config):
        has_logo = self._detect_logo(image)
        return {
            "safe": has_logo,  # Reject if no logo
            "score": 1.0 if has_logo else 0.0,
            "action": "allow" if has_logo else "reject"
        }

    def get_reason(self, result):
        if not result["safe"]:
            return "Corporate logo not found in image"
        return "Corporate logo detected"
```

---

## Timeline Estimate

| Phase | Description | Effort |
|-------|-------------|--------|
| Phase 1 | Base classes + registry | 1-2 days |
| Phase 2 | Migrate existing guardrails | 2-3 days |
| Phase 3 | Update pipeline | 1 day |
| Phase 4 | Testing + documentation | 1-2 days |
| **Total** | | **5-8 days** |

---

*Created: March 2026*
