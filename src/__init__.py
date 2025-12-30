"""
Face Liveness ONNX Package

Production-ready face liveness system with ONNX runtime.
"""

__version__ = "1.0.0"
__author__ = "Face Liveness Contributors"
__license__ = "Apache-2.0"

from src.inference.inference import FaceAntiSpoof, VideoAntiSpoof
from src.detection.detector import FaceDetector
from src.core.exceptions import (
    AntiSpoofError,
    ModelLoadError,
    InferenceError,
    InputValidationError,
)

__all__ = [
    "FaceAntiSpoof",
    "VideoAntiSpoof",
    "FaceDetector",
    "AntiSpoofError",
    "ModelLoadError",
    "InferenceError",
    "InputValidationError",
    "__version__",
]
