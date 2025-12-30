"""Face anti-spoofing inference module."""

from typing import Dict, Any, List, Optional
from pathlib import Path
from collections import deque
import logging

import numpy as np
import onnxruntime as ort

from src.core.exceptions import ModelLoadError, InferenceError, InputValidationError
from src.core.preprocessing import preprocess_face, preprocess_batch
from src.core.validators import FaceQualityChecker

logger = logging.getLogger(__name__)


class FaceAntiSpoof:
    """
    Face anti-spoofing inference engine.

    Uses ONNX runtime for cross-platform compatibility.
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        input_size: int = 128,
        providers: Optional[List[str]] = None,
        enable_quality_check: bool = True,
    ):
        """
        Initialize anti-spoofing model.

        Args:
            model_path: Path to ONNX model file
            threshold: Decision threshold for logit difference
            input_size: Model input size (width/height)
            providers: ONNX execution providers (None for auto)
            enable_quality_check: Whether to validate input quality

        Raises:
            ModelLoadError: If model cannot be loaded
        """
        self.model_path = model_path
        self.threshold = threshold
        self.input_size = input_size

        # Load model
        self.session = self._load_model(providers)
        self.input_name = self.session.get_inputs()[0].name

        # Quality checker
        self.quality_checker = (
            FaceQualityChecker() if enable_quality_check else None
        )

        # Metrics
        self.total_predictions = 0
        self.real_predictions = 0
        self.spoof_predictions = 0

        logger.info(f"Loaded anti-spoof model from {model_path}")

    def _load_model(
        self, providers: Optional[List[str]] = None
    ) -> ort.InferenceSession:
        """
        Load ONNX model with optimizations.

        Args:
            providers: Execution providers (None for auto)

        Returns:
            ONNX inference session

        Raises:
            ModelLoadError: If model loading fails
        """
        if not Path(self.model_path).exists():
            raise ModelLoadError(f"Model not found: {self.model_path}")

        try:
            # Session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            sess_options.intra_op_num_threads = 4
            sess_options.inter_op_num_threads = 2

            # Providers
            if providers is None:
                providers = ["CPUExecutionProvider"]

            session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers,
            )

            return session

        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}") from e

    def _process_logits(
        self, raw_logits: np.ndarray, threshold: float
    ) -> Dict[str, Any]:
        """
        Process raw logits into decision.

        Args:
            raw_logits: (2,) array with [real_logit, spoof_logit]
            threshold: Decision threshold

        Returns:
            Dictionary with prediction results
        """
        real_logit = float(raw_logits[0])
        spoof_logit = float(raw_logits[1])
        logit_diff = real_logit - spoof_logit

        is_real = logit_diff >= threshold
        confidence = abs(logit_diff)

        return {
            "is_real": bool(is_real),
            "status": "real" if is_real else "spoof",
            "logit_diff": float(logit_diff),
            "real_logit": float(real_logit),
            "spoof_logit": float(spoof_logit),
            "confidence": float(confidence),
        }

    def predict(
        self,
        face_crop: np.ndarray,
        threshold: Optional[float] = None,
        check_quality: bool = True,
    ) -> Dict[str, Any]:
        """
        Predict if face crop is real or spoof.

        Args:
            face_crop: RGB face image (H, W, 3)
            threshold: Decision threshold (None for default)
            check_quality: Whether to validate input quality

        Returns:
            Dictionary with prediction results:
                - is_real: bool
                - status: 'real' or 'spoof'
                - confidence: float
                - logit_diff: float
                - real_logit: float
                - spoof_logit: float
                - quality: Optional[Dict] if check_quality=True

        Raises:
            InputValidationError: If input is invalid
            InferenceError: If inference fails
        """
        if threshold is None:
            threshold = self.threshold

        # Validate input
        if face_crop is None or face_crop.size == 0:
            raise InputValidationError("Empty face crop provided")

        if face_crop.ndim != 3 or face_crop.shape[2] != 3:
            raise InputValidationError(
                f"Expected RGB image (H, W, 3), got shape {face_crop.shape}"
            )

        # Quality check
        quality_result = None
        if check_quality and self.quality_checker:
            bbox = {"width": face_crop.shape[1], "height": face_crop.shape[0]}
            quality_result = self.quality_checker.check(
                face_crop=face_crop,
                bbox=bbox,
                confidence=1.0,  # Not from detector
            )

            if not quality_result["acceptable"]:
                logger.warning(
                    f"Quality check failed: {quality_result['recommendation']}"
                )

        try:
            # Preprocess
            input_tensor = preprocess_face(face_crop, self.input_size)

            # Inference
            logits = self.session.run([], {self.input_name: input_tensor})[0]

            if logits.shape != (1, 2):
                raise InferenceError(
                    f"Unexpected output shape: {logits.shape}, expected (1, 2)"
                )

            # Process results
            result = self._process_logits(logits[0], threshold)

            # Update metrics
            self.total_predictions += 1
            if result["is_real"]:
                self.real_predictions += 1
            else:
                self.spoof_predictions += 1

            # Add quality info
            if quality_result:
                result["quality"] = quality_result

            return result

        except Exception as e:
            raise InferenceError(f"Inference failed: {e}") from e

    def predict_batch(
        self,
        face_crops: List[np.ndarray],
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Predict batch of face crops.

        Args:
            face_crops: List of RGB face images
            threshold: Decision threshold (None for default)

        Returns:
            List of prediction dictionaries

        Raises:
            InferenceError: If batch inference fails
        """
        if threshold is None:
            threshold = self.threshold

        if not face_crops:
            return []

        try:
            # Preprocess batch
            batch_input = preprocess_batch(face_crops, self.input_size)

            # Batch inference
            logits = self.session.run([], {self.input_name: batch_input})[0]

            if logits.shape != (len(face_crops), 2):
                raise InferenceError(
                    f"Unexpected output shape: {logits.shape}, "
                    f"expected ({len(face_crops)}, 2)"
                )

            # Process results
            results = []
            for i, logit in enumerate(logits):
                result = self._process_logits(logit, threshold)
                results.append(result)

                # Update metrics
                self.total_predictions += 1
                if result["is_real"]:
                    self.real_predictions += 1
                else:
                    self.spoof_predictions += 1

            return results

        except Exception as e:
            raise InferenceError(f"Batch inference failed: {e}") from e

    def get_stats(self) -> Dict[str, Any]:
        """
        Get inference statistics.

        Returns:
            Dictionary with prediction statistics
        """
        spoof_rate = (
            self.spoof_predictions / self.total_predictions
            if self.total_predictions > 0
            else 0.0
        )

        return {
            "total_predictions": self.total_predictions,
            "real_predictions": self.real_predictions,
            "spoof_predictions": self.spoof_predictions,
            "spoof_rate": spoof_rate,
        }


class VideoAntiSpoof(FaceAntiSpoof):
    """
    Video stream anti-spoofing with temporal filtering.

    Extends FaceAntiSpoof with multi-frame consistency checking.
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        input_size: int = 128,
        providers: Optional[List[str]] = None,
        enable_quality_check: bool = True,
        temporal_window: int = 5,
        consistency_threshold: float = 0.8,
    ):
        """
        Initialize video anti-spoofing model.

        Args:
            model_path: Path to ONNX model
            threshold: Decision threshold
            input_size: Model input size
            providers: ONNX execution providers
            enable_quality_check: Whether to validate input quality
            temporal_window: Number of frames for consistency check
            consistency_threshold: Minimum agreement ratio
        """
        super().__init__(
            model_path=model_path,
            threshold=threshold,
            input_size=input_size,
            providers=providers,
            enable_quality_check=enable_quality_check,
        )

        self.temporal_window = temporal_window
        self.consistency_threshold = consistency_threshold

        # Temporal history
        self.decision_history = deque(maxlen=temporal_window)
        self.confidence_history = deque(maxlen=temporal_window)

    def predict_temporal(
        self,
        face_crop: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Predict with temporal filtering.

        Args:
            face_crop: RGB face image
            threshold: Decision threshold

        Returns:
            Dictionary with:
                - ready: bool - whether decision is ready
                - decision: Optional[str] - 'real' or 'spoof' if ready
                - is_real: Optional[bool] - if ready
                - consistency: float - agreement ratio
                - avg_confidence: float - average confidence
                - frame_breakdown: Dict - frame counts
        """
        # Get single-frame prediction
        result = self.predict(face_crop, threshold=threshold, check_quality=False)

        # Update history
        self.decision_history.append(result["is_real"])
        self.confidence_history.append(result["confidence"])

        # Need full window
        if len(self.decision_history) < self.temporal_window:
            return {
                "ready": False,
                "frames_needed": self.temporal_window - len(self.decision_history),
                "current_prediction": result,
            }

        # Check consistency
        real_count = sum(self.decision_history)
        spoof_count = self.temporal_window - real_count
        max_count = max(real_count, spoof_count)

        consistency_ratio = max_count / self.temporal_window
        avg_confidence = float(np.mean(self.confidence_history))

        # Decision ready?
        if consistency_ratio >= self.consistency_threshold:
            final_decision = real_count > spoof_count
            status = "real" if final_decision else "spoof"

            return {
                "ready": True,
                "decision": status,
                "status": status,  # For drawing code
                "is_real": final_decision,
                "consistency": consistency_ratio,
                "confidence": avg_confidence,  # Use averaged confidence for display
                "avg_confidence": avg_confidence,
                "frame_breakdown": {
                    "real": real_count,
                    "spoof": spoof_count,
                    "total": self.temporal_window,
                },
            }
        else:
            return {
                "ready": False,
                "consistency": consistency_ratio,
                "reason": "Inconsistent predictions across frames",
                "current_prediction": result,
            }

    def reset(self) -> None:
        """Reset temporal filter state."""
        self.decision_history.clear()
        self.confidence_history.clear()
