"""
demo_app.py - Interactive Demo for Image Guardrails

This script provides multiple ways to test the guardrails:
1. CLI mode: Process single images
2. Batch mode: Process a directory of images
3. Interactive mode: Step-by-step demonstration

Usage:
    uv run python demo_app.py --help
    uv run python demo_app.py single test_images/sample.jpg
    uv run python demo_app.py batch test_images/
    uv run python demo_app.py interactive
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from guard_image import run_guardrails, load_config, DEFAULT_CONFIG


def create_test_images_dir():
    """Create test_images directory with instructions."""
    test_dir = Path(__file__).parent / "test_images"
    test_dir.mkdir(exist_ok=True)

    readme = test_dir / "README.txt"
    if not readme.exists():
        readme.write_text("""Test Images Directory
=====================

Place your test images here to test the guardrails pipeline.

Suggested test cases:
1. Normal photo (JPEG/PNG) - should ALLOW
2. Photo with faces - should ALLOW with faces blurred
3. Screenshot with text/PII - should ALLOW with PII redacted
4. Large file (>10MB) - should REJECT
5. Non-image file renamed to .jpg - should REJECT
6. Image with EXIF/GPS data - should ALLOW with metadata stripped

You can download sample images from:
- https://unsplash.com (free photos)
- https://picsum.photos (random placeholder images)

Example to download a test image:
    curl -L "https://picsum.photos/800/600" -o test_images/sample.jpg
""")
    return test_dir


def process_single(image_path: str, config_path: str = None, output_json: bool = False):
    """Process a single image through the guardrails."""
    path = Path(image_path)
    if not path.exists():
        print(f"Error: File not found: {path}")
        return None

    config = load_config(Path(config_path) if config_path else None)
    result = run_guardrails(path, config)

    if output_json:
        print(json.dumps(result, indent=2))
    else:
        print_result(result)

    return result


def process_batch(directory: str, config_path: str = None):
    """Process all images in a directory."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        print(f"Error: Not a directory: {dir_path}")
        return []

    config = load_config(Path(config_path) if config_path else None)

    # Find all image files
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
    images = [f for f in dir_path.iterdir() if f.suffix.lower() in extensions]

    if not images:
        print(f"No images found in {dir_path}")
        return []

    print(f"\nProcessing {len(images)} images from {dir_path}\n")
    print("=" * 70)

    results = []
    stats = {"ALLOW": 0, "REJECT": 0, "REVIEW": 0}

    for img_path in sorted(images):
        result = run_guardrails(img_path, config)
        results.append(result)
        stats[result["decision"]] = stats.get(result["decision"], 0) + 1

        status_icon = "✓" if result["decision"] == "ALLOW" else "✗"
        print(f"{status_icon} {img_path.name}: {result['decision']}")

    print("=" * 70)
    print(f"\nSummary: {stats['ALLOW']} allowed, {stats['REJECT']} rejected, {stats.get('REVIEW', 0)} for review")

    return results


def interactive_demo():
    """Run an interactive demonstration of each guardrail step."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           Image Guardrails - Interactive Demo                    ║
╚══════════════════════════════════════════════════════════════════╝

This demo will walk you through each step of the guardrails pipeline.

The pipeline includes:
  1. File type validation (magic bytes, not extension)
  2. File size and resolution checks
  3. EXIF metadata stripping (removes GPS, device info)
  4. NSFW content detection
  5. PII redaction (OCR + masking)
  6. Face detection and blurring
  7. Perceptual hash computation

""")

    # Create test directory
    test_dir = create_test_images_dir()
    print(f"Test images directory: {test_dir}")

    # Check for test images
    extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    images = [f for f in test_dir.iterdir() if f.suffix.lower() in extensions]

    if not images:
        print(f"""
No test images found. Please add some images to: {test_dir}

Quick way to get a test image:
    curl -L "https://picsum.photos/800/600" -o {test_dir}/sample.jpg

Then run this demo again.
""")
        return

    print(f"\nFound {len(images)} test image(s):")
    for i, img in enumerate(images, 1):
        print(f"  {i}. {img.name}")

    # Select image
    print("\nEnter image number to process (or 'q' to quit):")
    try:
        choice = input("> ").strip()
        if choice.lower() == 'q':
            return

        idx = int(choice) - 1
        if 0 <= idx < len(images):
            selected = images[idx]
        else:
            print("Invalid selection")
            return
    except (ValueError, EOFError):
        print("Invalid input")
        return

    # Process with verbose output
    print(f"\n{'='*60}")
    print(f"Processing: {selected.name}")
    print(f"{'='*60}\n")

    config = load_config()
    result = run_guardrails(selected, config)

    print("\n" + "="*60)
    print("DETAILED RESULTS")
    print("="*60)
    print(json.dumps(result, indent=2))


def print_result(result: Dict[str, Any]):
    """Pretty print a guardrail result."""
    decision = result["decision"]
    color_start = "\033[92m" if decision == "ALLOW" else "\033[91m"
    color_end = "\033[0m"

    print(f"\n{'='*60}")
    print(f"DECISION: {color_start}{decision}{color_end}")
    print(f"{'='*60}")
    print(f"Input:    {result['input_path']}")
    print(f"SHA256:   {result['sha256'][:32]}...")

    if result.get('mime_type'):
        print(f"MIME:     {result['mime_type']}")
    if result.get('resolution'):
        print(f"Size:     {result['resolution']}")

    checks = result.get('checks', {})
    if 'nsfw' in checks:
        score = checks['nsfw']['score']
        print(f"NSFW:     {score:.4f} {'(safe)' if checks['nsfw']['safe'] else '(unsafe)'}")

    if checks.get('pii_redaction'):
        print(f"PII:      Redaction applied")
    if checks.get('face_blur'):
        print(f"Faces:    Blur applied")

    if result.get('perceptual_hash'):
        print(f"pHash:    {result['perceptual_hash']}")

    if result.get('output_path'):
        print(f"Output:   {result['output_path']}")

    print(f"Reasons:  {'; '.join(result['reasons'])}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Image Guardrails Demo Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python demo_app.py single image.jpg
  uv run python demo_app.py single image.jpg --json
  uv run python demo_app.py batch ./test_images/
  uv run python demo_app.py interactive
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Single image processing
    single_parser = subparsers.add_parser("single", help="Process a single image")
    single_parser.add_argument("image", help="Path to image file")
    single_parser.add_argument("--config", "-c", help="Path to config YAML")
    single_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Batch processing
    batch_parser = subparsers.add_parser("batch", help="Process all images in a directory")
    batch_parser.add_argument("directory", help="Path to directory containing images")
    batch_parser.add_argument("--config", "-c", help="Path to config YAML")

    # Interactive demo
    subparsers.add_parser("interactive", help="Run interactive demonstration")

    # Setup command
    subparsers.add_parser("setup", help="Create test_images directory with instructions")

    args = parser.parse_args()

    if args.command == "single":
        process_single(args.image, args.config, args.json)
    elif args.command == "batch":
        process_batch(args.directory, args.config)
    elif args.command == "interactive":
        interactive_demo()
    elif args.command == "setup":
        test_dir = create_test_images_dir()
        print(f"Created test directory: {test_dir}")
        print("See test_images/README.txt for instructions")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
