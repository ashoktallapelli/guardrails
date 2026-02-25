"""
image_qa.py - Visual Question Answering for Images

After an image passes guardrails, users can ask questions about it.
Uses BLIP (Salesforce) model for image understanding.

Usage:
    python image_qa.py test_images/sample.jpg
    python image_qa.py test_images/sample.jpg --question "What is in this image?"
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

# Cache for model
_qa_model_cache = {}


def load_qa_model():
    """Load BLIP model for visual question answering."""
    if "model" in _qa_model_cache:
        return _qa_model_cache["model"], _qa_model_cache["processor"]

    try:
        from transformers import BlipProcessor, BlipForQuestionAnswering
    except ImportError:
        print("Error: transformers not installed. Run: uv pip install transformers")
        return None, None

    model_name = "Salesforce/blip-vqa-base"

    try:
        print("Loading BLIP VQA model...")
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForQuestionAnswering.from_pretrained(model_name)
        _qa_model_cache["model"] = model
        _qa_model_cache["processor"] = processor
        print("Model loaded successfully.")
        return model, processor
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nIf you have network/SSL issues, you can:")
        print("1. Download model manually from HuggingFace")
        print("2. Or use a local model path")
        return None, None


def answer_question(image: Image.Image, question: str) -> str:
    """
    Answer a question about the image.

    Args:
        image: PIL Image object
        question: Question to ask about the image

    Returns:
        Answer string
    """
    model, processor = load_qa_model()
    if model is None:
        return "Error: Model not available"

    try:
        import torch

        # Process inputs
        inputs = processor(image, question, return_tensors="pt")

        # Generate answer
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50)

        # Decode answer
        answer = processor.decode(outputs[0], skip_special_tokens=True)
        return answer

    except Exception as e:
        return f"Error generating answer: {e}"


def describe_image(image: Image.Image) -> str:
    """
    Generate a description/caption for the image.
    Uses BLIP captioning model.
    """
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch
    except ImportError:
        return "Error: transformers not installed"

    model_name = "Salesforce/blip-image-captioning-base"

    try:
        if "caption_model" not in _qa_model_cache:
            print("Loading BLIP captioning model...")
            processor = BlipProcessor.from_pretrained(model_name)
            model = BlipForConditionalGeneration.from_pretrained(model_name)
            _qa_model_cache["caption_model"] = model
            _qa_model_cache["caption_processor"] = processor

        model = _qa_model_cache["caption_model"]
        processor = _qa_model_cache["caption_processor"]

        inputs = processor(image, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50)

        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return caption

    except Exception as e:
        return f"Error generating caption: {e}"


def run_guardrails_and_qa(
    image_path: Path,
    question: Optional[str] = None,
    skip_guardrails: bool = False
) -> Tuple[bool, str]:
    """
    Run guardrails on image, then answer question if passed.

    Args:
        image_path: Path to image file
        question: Question to ask (optional, will prompt if not provided)
        skip_guardrails: Skip guardrails check (for testing)

    Returns:
        (passed_guardrails, answer_or_error)
    """
    from guard_image import run_guardrails, load_config

    # Step 1: Run guardrails (unless skipped)
    if not skip_guardrails:
        config = load_config()
        result = run_guardrails(image_path, config)

        if result["decision"] != "ALLOW":
            return False, f"Image rejected: {'; '.join(result['reasons'])}"

        print(f"\n✓ Image passed guardrails (NSFW: {result['checks']['nsfw']['score']:.2f})")

    # Step 2: Load image
    image = Image.open(image_path).convert("RGB")

    # Step 3: If no question, generate description
    if not question:
        print("\nGenerating image description...")
        caption = describe_image(image)
        return True, f"Description: {caption}"

    # Step 4: Answer the question
    print(f"\nQuestion: {question}")
    answer = answer_question(image, question)
    return True, f"Answer: {answer}"


def interactive_mode(image_path: Path):
    """Interactive Q&A session with an image."""
    from guard_image import run_guardrails, load_config

    print(f"\n{'='*60}")
    print("Image Q&A - Interactive Mode")
    print(f"{'='*60}")

    # Run guardrails first
    print(f"\nChecking image: {image_path}")
    config = load_config()
    result = run_guardrails(image_path, config)

    if result["decision"] != "ALLOW":
        print(f"\n✗ Image REJECTED: {'; '.join(result['reasons'])}")
        print("Cannot proceed with Q&A on rejected images.")
        return

    print(f"✓ Image passed guardrails")
    print(f"  NSFW: {result['checks']['nsfw']['score']:.4f}")
    if 'safety' in result['checks']:
        scores = result['checks']['safety']['scores']
        print(f"  Safe: {scores.get('safe', 'N/A')}")

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Generate initial description
    print("\nGenerating image description...")
    caption = describe_image(image)
    print(f"\nImage Description: {caption}")

    # Interactive Q&A loop
    print("\n" + "-"*60)
    print("Ask questions about the image (type 'quit' to exit):")
    print("-"*60)

    while True:
        try:
            question = input("\nYour question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            if not question:
                continue

            answer = answer_question(image, question)
            print(f"Answer: {answer}")

        except (KeyboardInterrupt, EOFError):
            break

    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(
        description="Visual Question Answering with Guardrails"
    )
    parser.add_argument("image", help="Path to image file")
    parser.add_argument(
        "--question", "-q",
        help="Question to ask about the image"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive Q&A mode"
    )
    parser.add_argument(
        "--skip-guardrails",
        action="store_true",
        help="Skip guardrails check (for testing)"
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Just describe the image"
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    if args.interactive:
        interactive_mode(image_path)
    else:
        passed, response = run_guardrails_and_qa(
            image_path,
            question=args.question if not args.describe else None,
            skip_guardrails=args.skip_guardrails
        )

        if not passed:
            print(f"\n✗ {response}")
            sys.exit(1)
        else:
            print(f"\n{response}")


if __name__ == "__main__":
    main()
