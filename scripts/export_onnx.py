"""
Export PyTorch model to ONNX format.

This script converts a trained PyTorch checkpoint to ONNX format
for cross-platform deployment.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
from collections import OrderedDict

import torch
import onnx
import onnxsim

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.minifas import create_model


def load_checkpoint(
    checkpoint_path: str,
    device: str,
    input_size: int = 128,
    num_classes: int = 2,
) -> torch.nn.Module:
    """
    Load model from PyTorch checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file
        device: Device to load model on ('cpu' or 'cuda')
        input_size: Input image size
        num_classes: Number of output classes

    Returns:
        Loaded model in evaluation mode
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Extract state dict (handle different checkpoint formats)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Create model
    model = create_model(
        input_size=input_size,
        num_classes=num_classes,
        num_channels=3,
    ).to(device)

    # Clean up state dict keys (handle different naming conventions)
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        
        # Remove 'module.' prefix (from DataParallel)
        if new_key.startswith("module."):
            new_key = new_key[7:]
        
        # Handle legacy naming
        new_key = new_key.replace("model.prob", "model.logits")
        new_key = new_key.replace(".prob", ".logits")
        new_key = new_key.replace("model.drop", "model.dropout")
        new_key = new_key.replace(".drop", ".dropout")
        new_key = new_key.replace("FTGenerator.ft.", "FTGenerator.fourier_transform.")
        new_key = new_key.replace("FTGenerator.ft", "FTGenerator.fourier_transform")
        
        new_state_dict[new_key] = value

    # Load weights
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    print("✓ Checkpoint loaded successfully")
    return model


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_size: int = 128,
    simplify: bool = True,
    opset_version: int = 13,
) -> str:
    """
    Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model in eval mode
        output_path: Path to save ONNX model
        input_size: Input image size
        simplify: Whether to simplify ONNX graph
        opset_version: ONNX opset version

    Returns:
        Path to saved ONNX model
    """
    print("\nExporting to ONNX...")
    print(f"Output: {output_path}")
    print(f"Input size: {input_size}x{input_size}")
    print(f"Opset version: {opset_version}")

    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
    )

    print("✓ ONNX model exported")

    # Simplify ONNX model
    if simplify:
        print("\nSimplifying ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx_model, check = onnxsim.simplify(onnx_model)
        
        if not check:
            print("⚠ Warning: Simplified model validation failed")
        else:
            onnx.save(onnx_model, output_path)
            print("✓ ONNX model simplified")

    # Verify ONNX model
    print("\nVerifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verified")

    # Print model info
    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"\n✓ Export complete!")
    print(f"  File: {output_path}")
    print(f"  Size: {file_size:.2f} MB")

    return output_path


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export PyTorch model to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to PyTorch checkpoint (.pth file)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for ONNX model (default: replace .pth with .onnx)",
    )

    parser.add_argument(
        "--input-size",
        type=int,
        default=128,
        help="Input image size (default: 128)",
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of classes (default: 2)",
    )

    parser.add_argument(
        "--opset-version",
        type=int,
        default=13,
        help="ONNX opset version (default: 13)",
    )

    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip ONNX simplification",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use (default: cpu)",
    )

    args = parser.parse_args()

    # Validate checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Determine output path
    output_path = args.output
    if output_path is None:
        output_path = str(Path(args.checkpoint).with_suffix(".onnx"))

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        # Load model
        model = load_checkpoint(
            args.checkpoint,
            args.device,
            args.input_size,
            args.num_classes,
        )

        # Export to ONNX
        export_to_onnx(
            model,
            output_path,
            args.input_size,
            simplify=not args.no_simplify,
            opset_version=args.opset_version,
        )

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
