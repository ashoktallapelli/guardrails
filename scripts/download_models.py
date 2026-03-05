"""
download_models.py - Download and cache models for offline use

Usage:
    python download_models.py
    python download_models.py --model adamcodd
    python download_models.py --model clip
    python download_models.py --all
"""

import argparse
import os
import ssl
import sys

# Fix SSL certificate issues on Windows/corporate networks
if sys.platform == 'win32':
    try:
        import certifi
        os.environ['SSL_CERT_FILE'] = certifi.where()
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    except ImportError:
        pass

# Disable SSL verification if still failing (corporate proxy)
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
os.environ['CURL_CA_BUNDLE'] = ''

# Disable offline mode for downloads
os.environ.pop('HF_HUB_OFFLINE', None)
os.environ.pop('TRANSFORMERS_OFFLINE', None)


def download_adamcodd():
    """Download AdamCodd NSFW detector model."""
    print("Downloading AdamCodd/vit-base-nsfw-detector...")
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    AutoImageProcessor.from_pretrained("AdamCodd/vit-base-nsfw-detector")
    AutoModelForImageClassification.from_pretrained("AdamCodd/vit-base-nsfw-detector")
    print("AdamCodd model downloaded successfully!")


def download_clip():
    """Download CLIP model for violence/hate detection."""
    print("Downloading openai/clip-vit-base-patch32...")
    from transformers import CLIPProcessor, CLIPModel

    CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    print("CLIP model downloaded successfully!")


def download_opennsfw2():
    """Download OpenNSFW2 weights."""
    print("Downloading OpenNSFW2 weights...")
    import os
    from pathlib import Path
    import urllib.request

    home = Path.home()
    weights_dir = home / ".opennsfw2" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / "open_nsfw_weights.h5"

    if weights_path.exists():
        print(f"OpenNSFW2 weights already exist at {weights_path}")
        return

    url = "https://github.com/bhky/opennsfw2/releases/download/v0.1.0/open_nsfw_weights.h5"
    print(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, weights_path)
    print(f"OpenNSFW2 weights downloaded to {weights_path}")


def main():
    parser = argparse.ArgumentParser(description="Download models for image_guard.py")
    parser.add_argument(
        "--model", "-m",
        choices=["adamcodd", "clip", "opennsfw2"],
        help="Specific model to download"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Download all models"
    )
    args = parser.parse_args()

    if args.all:
        download_opennsfw2()
        download_adamcodd()
        download_clip()
        print("\nAll models downloaded!")
    elif args.model == "adamcodd":
        download_adamcodd()
    elif args.model == "clip":
        download_clip()
    elif args.model == "opennsfw2":
        download_opennsfw2()
    else:
        print("Available models:")
        print("  --model adamcodd   : AdamCodd NSFW detector (ViT, ~330MB)")
        print("  --model clip       : CLIP for violence/hate detection (~605MB)")
        print("  --model opennsfw2  : OpenNSFW2 weights (~24MB)")
        print("  --all              : Download all models")
        print("\nExample:")
        print("  python download_models.py --model adamcodd")
        print("  python download_models.py --all")


if __name__ == "__main__":
    main()
