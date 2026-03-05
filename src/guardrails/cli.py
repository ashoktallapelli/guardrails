"""
cli.py - Command-line interface for guardrails.

Usage:
    python -m guardrails image.jpg
    python -m guardrails image.jpg --analyze-only
    python -m guardrails --text "Contact john@example.com" --anonymize
"""

import argparse
import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from guardrails.config import load_config
from guardrails.pipeline import Pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def run_image(
    image_path: Path,
    config: Dict[str, Any],
    output_dir: Optional[Path] = None,
    analyze_only: bool = False
) -> Dict[str, Any]:
    """
    Run guardrails on an image.

    Args:
        image_path: Path to input image
        config: Configuration dictionary
        output_dir: Optional output directory
        analyze_only: If True, only analyze without saving

    Returns:
        Result dictionary
    """
    from PIL import Image

    pipeline = Pipeline(config)
    unique_id = uuid.uuid4().hex[:12]

    # Compute file hash
    file_hash = sha256_file(image_path)
    logger.info(f"Processing: {image_path} (SHA256: {file_hash[:16]}...)")

    # Load image
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {
            "decision": "REJECT",
            "reasons": [f"Failed to open image: {e}"],
            "is_safe": False,
        }

    # Run pipeline
    result = pipeline.run(img, input_type="image")
    result["input_path"] = str(image_path)
    result["sha256"] = file_hash
    result["output_id"] = unique_id

    if analyze_only:
        # Remove output image from result for analyze-only mode
        if "output" in result:
            del result["output"]
        return result

    # Save output if not rejected
    if result["decision"] != "REJECT":
        if output_dir is None:
            output_dir = image_path.parent / "output"

        decision_folder = output_dir / result["decision"].lower()
        decision_folder.mkdir(parents=True, exist_ok=True)

        output_path = decision_folder / f"{unique_id}.jpg"
        result["output"].save(
            output_path,
            format="JPEG",
            quality=config.get("output_quality", 95)
        )
        result["output_path"] = str(output_path)
        logger.info(f"{result['decision']}: Image saved to {output_path}")

    # Remove PIL image from result (not JSON serializable)
    if "output" in result:
        del result["output"]

    return result


def run_text(
    text: str,
    config: Dict[str, Any],
    anonymize: bool = False
) -> Dict[str, Any]:
    """
    Run guardrails on text.

    Args:
        text: Input text
        config: Configuration dictionary
        anonymize: Whether to anonymize PII

    Returns:
        Result dictionary
    """
    pipeline = Pipeline(config)
    result = pipeline.run_text(text, anonymize=anonymize)
    return result


def print_result(result: Dict[str, Any]) -> None:
    """Print formatted result to console."""
    decision = result.get('decision', 'UNKNOWN')

    print(f"\n{'='*50}")
    print(f"  DECISION: {decision}")
    print(f"{'='*50}")

    # Show reasons
    reasons = result.get('reasons', [])
    if reasons:
        print(f"\n  Reason: {reasons[0]}")

    # Safety scores
    checks = result.get('checks', {})
    has_safety_checks = any(k in checks for k in ['nsfw', 'violence', 'hate_symbols'])

    if has_safety_checks:
        print(f"\n  Safety Scores:")

        if 'nsfw' in checks:
            nsfw = checks['nsfw']
            status = "SAFE" if nsfw.get('safe', True) else "UNSAFE"
            print(f"    NSFW:         {nsfw.get('score', 0):.4f} ({status})")

        if 'violence' in checks:
            violence = checks['violence']
            details = violence.get('details', {})
            status = "SAFE" if violence.get('safe', True) else "UNSAFE"
            print(f"    Violence:     {details.get('violence', 0):.4f}")
            print(f"    Weapons:      {details.get('weapons', 0):.4f}")
            print(f"    Safe:         {details.get('safe', 0):.4f} ({status})")

        if 'hate_symbols' in checks:
            hate = checks['hate_symbols']
            details = hate.get('details', {})
            status = "SAFE" if hate.get('safe', True) else "UNSAFE"
            print(f"    Hate Score:   {details.get('combined_hate_score', 0):.4f} ({status})")

    # Redaction info
    if decision in ["ALLOW", "REDACT"]:
        print(f"\n  Redaction:")
        pii_count = checks.get('pii', {}).get('details', {}).get('entity_count', 0)
        face_count = checks.get('faces', {}).get('details', {}).get('face_count', 0)
        print(f"    PII Found:    {pii_count}")
        print(f"    Faces Found:  {face_count}")
        print(f"    Is Redacted:  {result.get('is_redacted', False)}")

    # Output info
    if result.get('output_path'):
        print(f"\n  Output:")
        print(f"    Folder: {decision.lower()}/")
        print(f"    Image:  {result['output_path']}")
        if result.get('output_id'):
            print(f"    ID:     {result['output_id']}")

    print(f"{'='*50}\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Guardrails - Validate and sanitize content before AI processing"
    )
    parser.add_argument("image", nargs="?", help="Path to input image")
    parser.add_argument(
        "--config", "-c",
        help="Path to YAML config file",
        default=None
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Directory for sanitized output",
        default=None
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON"
    )
    parser.add_argument(
        "--analyze-only", "-a",
        action="store_true",
        help="Analyze only - return results without modifying the image"
    )
    # Text PII arguments
    parser.add_argument(
        "--text", "-t",
        help="Analyze text for PII (direct input)"
    )
    parser.add_argument(
        "--text-file",
        help="Analyze text from file for PII"
    )
    parser.add_argument(
        "--anonymize",
        action="store_true",
        help="Anonymize PII in text (use with --text or --text-file)"
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    # Text mode
    if args.text or args.text_file:
        if args.text_file:
            text_path = Path(args.text_file)
            if not text_path.exists():
                raise SystemExit(f"Error: Text file not found: {text_path}")
            text = text_path.read_text(encoding="utf-8")
        else:
            text = args.text

        result = run_text(text, config, anonymize=args.anonymize)
        print(json.dumps(result, indent=2, default=str))
        return

    # Image mode requires image argument
    if not args.image:
        parser.error("Image path is required (or use --text/--text-file for text PII analysis)")

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Error: File not found: {image_path}")

    output_dir = Path(args.output_dir) if args.output_dir else None

    result = run_image(
        image_path,
        config,
        output_dir=output_dir,
        analyze_only=args.analyze_only
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print_result(result)


if __name__ == "__main__":
    main()
