"""Download or verify Face Liveness ONNX models."""

import argparse
from pathlib import Path
import sys


def print_download_instructions() -> None:
    """Print instructions for downloading models."""
    print(
        """
=============================================================
           Face Liveness ONNX Model Download Guide           
=============================================================

This project needs pre-trained ONNX models for face detection
and liveness (anti-spoofing). You can either copy them from the
reference assets or download them automatically.

Option 1: Copy from reference (if /ref is available)
    cd face-antispoof-onnx
    mkdir -p models
    cp ../ref/models/*.onnx models/
    # Or on Windows PowerShell:
    mkdir models -Force
    Copy-Item ../ref/models/*.onnx models/

Required files:
  - best_model_quantized.onnx (600 KB)  # liveness
  - detector_quantized.onnx (~2 MB)     # face detector

Optional files:
  - best_model.onnx (1.82 MB)           # FP32 liveness
  - detector.onnx (~6 MB)               # FP32 detector

Option 2: Download from release/URL
  1. Create models directory:
     mkdir -p models
  2. Download files to models/ directory:
     - best_model_quantized.onnx
     - detector_quantized.onnx
  3. Verify downloads:
     python scripts/download_models.py --verify

Option 3: Train your own model
  See docs/TRAINING.md for instructions and requirements.

Verification
After downloading/copying, verify models:
    python -c "from pathlib import Path; \\
        print('Models ready!' if \\
        Path('models/best_model_quantized.onnx').exists() and \\
        Path('models/detector_quantized.onnx').exists() \\
        else 'Models missing')"

Expected directory structure:
  face-antispoof-onnx/
    models/
      best_model_quantized.onnx    (required)
      detector_quantized.onnx      (required)
      best_model.onnx              (optional)
      detector.onnx                (optional)
    src/
    scripts/
    demo.py

For more information, see:
  - README.md
  - QUICKSTART.md
  - docs/DEPLOYMENT.md
=============================================================
"""
    )


def verify_models(models_dir: Path = Path("models")) -> bool:
    """Verify that required models exist."""
    required_models = [
        "best_model_quantized.onnx",
        "detector_quantized.onnx",
    ]

    optional_models = [
        "best_model.onnx",
        "detector.onnx",
    ]

    print("\nVerifying models...\n")
    all_found = True

    for model_name in required_models:
        model_path = models_dir / model_name
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"✔ {model_name} ({size_mb:.2f} MB)")
        else:
            print(f"✖ {model_name} - MISSING (required)")
            all_found = False

    print()

    for model_name in optional_models:
        model_path = models_dir / model_name
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"  {model_name} ({size_mb:.2f} MB) [optional]")
        else:
            print(f"  {model_name} - not present [optional]")

    print()

    if all_found:
        print("All required models found!")
        return True

    print("Some required models are missing")
    print("\nRun: python scripts/download_models.py")
    print("for download instructions.")
    return False


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download or verify face liveness models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify that required models exist",
    )

    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Models directory (default: models)",
    )

    args = parser.parse_args()
    models_dir = Path(args.models_dir)

    if args.verify:
        sys.exit(0 if verify_models(models_dir) else 1)

    print_download_instructions()


if __name__ == "__main__":
    main()
