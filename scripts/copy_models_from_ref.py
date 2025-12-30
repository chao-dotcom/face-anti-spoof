"""
Copy models from reference implementation to this project.

This is a one-time setup script to copy pre-trained models from
the reference implementation directory to make this project independent.
"""

import argparse
import shutil
import sys
from pathlib import Path


def copy_models_from_ref(
    ref_dir: str = "../ref",
    output_dir: str = "models",
    verbose: bool = True,
) -> bool:
    """
    Copy ONNX models from reference directory.

    Args:
        ref_dir: Path to reference implementation directory
        output_dir: Output directory for models
        verbose: Print progress messages

    Returns:
        True if successful, False otherwise
    """
    ref_path = Path(ref_dir)
    output_path = Path(output_dir)

    # Validate ref directory exists
    if not ref_path.exists():
        print(f"✗ Error: Reference directory not found: {ref_dir}")
        print("\nPlease provide the correct path to the reference implementation.")
        print("Example: python scripts/copy_models_from_ref.py --ref-dir /path/to/ref")
        return False

    ref_models_dir = ref_path / "models"
    if not ref_models_dir.exists():
        print(f"✗ Error: Models directory not found: {ref_models_dir}")
        return False

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Models to copy
    models_to_copy = [
        "best_model_quantized.onnx",  # Required
        "detector_quantized.onnx",    # Required
        "best_model.onnx",             # Optional
        "detector.onnx",               # Optional
    ]

    required_models = [
        "best_model_quantized.onnx",
        "detector_quantized.onnx",
    ]

    # Copy each model
    copied_count = 0
    failed_count = 0

    print(f"\n📦 Copying models from {ref_dir}\n")

    for model_name in models_to_copy:
        src = ref_models_dir / model_name
        dst = output_path / model_name

        is_required = model_name in required_models

        if not src.exists():
            if is_required:
                print(f"✗ {model_name} - NOT FOUND (required)")
                failed_count += 1
            else:
                if verbose:
                    print(f"  {model_name} - not found [optional]")
            continue

        try:
            # Copy file
            shutil.copy2(src, dst)

            # Get file size
            size_mb = dst.stat().st_size / (1024 * 1024)

            print(f"✓ {model_name} ({size_mb:.2f} MB)")
            copied_count += 1

        except Exception as e:
            print(f"✗ {model_name} - FAILED: {e}")
            failed_count += 1

    # Summary
    print(f"\n{'─' * 60}")
    print(f"Copied {copied_count} model(s) to {output_dir}/")

    if failed_count > 0:
        print(f"⚠ Warning: {failed_count} model(s) failed to copy")
        return False

    # Check if all required models are present
    all_required_present = all(
        (output_path / model).exists() for model in required_models
    )

    if all_required_present:
        print("\n✅ All required models copied successfully!")
        print("\nYou can now run:")
        print("  python demo.py --mode webcam")
        return True
    else:
        print("\n⚠ Warning: Some required models are missing")
        return False


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Copy models from reference implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default (assumes ref is in ../ref)
  python scripts/copy_models_from_ref.py

  # Custom ref directory
  python scripts/copy_models_from_ref.py --ref-dir /path/to/ref

  # Custom output directory
  python scripts/copy_models_from_ref.py --output-dir my_models
        """,
    )

    parser.add_argument(
        "--ref-dir",
        type=str,
        default="../ref",
        help="Path to reference implementation directory (default: ../ref)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for models (default: models)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    args = parser.parse_args()

    # Copy models
    success = copy_models_from_ref(
        ref_dir=args.ref_dir,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
