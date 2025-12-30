"""
Quantize ONNX model to INT8 for smaller size and faster inference.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import onnx
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType
from onnxruntime.quantization import CalibrationDataReader
import cv2


class ImageDataReader(CalibrationDataReader):
    """Data reader for calibration during static quantization."""

    def __init__(
        self,
        calibration_images: List[str],
        input_name: str = "input",
        input_size: int = 128,
    ):
        """
        Initialize calibration data reader.

        Args:
            calibration_images: List of image paths for calibration
            input_name: Name of model input
            input_size: Input image size
        """
        self.calibration_images = calibration_images
        self.input_name = input_name
        self.input_size = input_size
        self.current_index = 0

    def get_next(self) -> Optional[dict]:
        """Get next calibration sample."""
        if self.current_index >= len(self.calibration_images):
            return None

        # Load and preprocess image
        image_path = self.calibration_images[self.current_index]
        image = cv2.imread(image_path)
        
        if image is None:
            self.current_index += 1
            return self.get_next()

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        image = cv2.resize(
            image,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_CUBIC,
        )

        # Normalize
        image = image.astype(np.float32) / 255.0

        # Transpose to (C, H, W)
        image = np.transpose(image, (2, 0, 1))

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        self.current_index += 1

        return {self.input_name: image}


def quantize_model_dynamic(
    model_path: str,
    output_path: str,
) -> str:
    """
    Dynamically quantize ONNX model (no calibration needed).

    Args:
        model_path: Path to input ONNX model
        output_path: Path to save quantized model

    Returns:
        Path to quantized model
    """
    print("Quantizing model (dynamic)...")
    
    quantize_dynamic(
        model_path,
        output_path,
        weight_type=QuantType.QUInt8,
    )
    
    print("✓ Dynamic quantization complete")
    return output_path


def quantize_model_static(
    model_path: str,
    output_path: str,
    calibration_dir: str,
    num_samples: int = 100,
    input_size: int = 128,
) -> str:
    """
    Statically quantize ONNX model with calibration data.

    Args:
        model_path: Path to input ONNX model
        output_path: Path to save quantized model
        calibration_dir: Directory with calibration images
        num_samples: Number of calibration samples
        input_size: Input image size

    Returns:
        Path to quantized model
    """
    print(f"Quantizing model (static) with {num_samples} calibration samples...")
    
    # Get calibration images
    calibration_path = Path(calibration_dir)
    image_files = []
    
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_files.extend(calibration_path.glob(f"**/{ext}"))
    
    # Limit to num_samples
    image_files = [str(f) for f in image_files[:num_samples]]
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {calibration_dir}")
    
    print(f"Found {len(image_files)} calibration images")
    
    # Create data reader
    data_reader = ImageDataReader(image_files, "input", input_size)
    
    # Quantize
    quantize_static(
        model_path,
        output_path,
        data_reader,
        quant_format=QuantType.QUInt8,
    )
    
    print("✓ Static quantization complete")
    return output_path


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Quantize ONNX model to INT8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "model",
        type=str,
        help="Path to ONNX model",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: add '_quantized' suffix)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["dynamic", "static"],
        default="dynamic",
        help="Quantization mode (default: dynamic)",
    )

    parser.add_argument(
        "--calibration-dir",
        type=str,
        default=None,
        help="Directory with calibration images (required for static mode)",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of calibration samples (default: 100)",
    )

    parser.add_argument(
        "--input-size",
        type=int,
        default=128,
        help="Input image size (default: 128)",
    )

    args = parser.parse_args()

    # Validate model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    # Determine output path
    output_path = args.output
    if output_path is None:
        model_path = Path(args.model)
        output_path = str(model_path.parent / f"{model_path.stem}_quantized.onnx")

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        # Get original size
        original_size = Path(args.model).stat().st_size / (1024 * 1024)
        print(f"\nOriginal model size: {original_size:.2f} MB")

        # Quantize
        if args.mode == "dynamic":
            quantize_model_dynamic(args.model, output_path)
        else:
            if args.calibration_dir is None:
                print("Error: --calibration-dir required for static quantization")
                sys.exit(1)
            
            quantize_model_static(
                args.model,
                output_path,
                args.calibration_dir,
                args.num_samples,
                args.input_size,
            )

        # Get quantized size
        quantized_size = Path(output_path).stat().st_size / (1024 * 1024)
        compression_ratio = (1 - quantized_size / original_size) * 100

        print(f"\n✓ Quantization complete!")
        print(f"  Original: {original_size:.2f} MB")
        print(f"  Quantized: {quantized_size:.2f} MB")
        print(f"  Reduction: {compression_ratio:.1f}%")
        print(f"  Output: {output_path}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
