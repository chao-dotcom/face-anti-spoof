"""Face detection module with YuNet detector."""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
import cv2

from src.core.exceptions import DetectionError, ModelLoadError
from src.core.preprocessing import expand_bbox, crop_face


class FaceDetector:
    """
    Face detector using OpenCV's YuNet model.

    Provides face detection with confidence scoring and quality filtering.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.8,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        min_face_size: int = 60,
        min_margin: int = 5,
    ):
        """
        Initialize face detector.

        Args:
            model_path: Path to YuNet ONNX model
            confidence_threshold: Minimum confidence for detections
            nms_threshold: NMS IoU threshold
            top_k: Maximum number of detections before NMS
            min_face_size: Minimum face size in pixels
            min_margin: Minimum margin from image edge

        Raises:
            ModelLoadError: If model cannot be loaded
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.min_face_size = min_face_size
        self.min_margin = min_margin

        self.detector = self._load_model()

    def _load_model(self) -> cv2.FaceDetectorYN:
        """
        Load YuNet face detection model.

        Returns:
            Initialized face detector

        Raises:
            ModelLoadError: If model cannot be loaded
        """
        if not Path(self.model_path).exists():
            raise ModelLoadError(f"Model not found: {self.model_path}")

        try:
            # Initial size will be updated per image
            initial_size = (320, 320)

            detector = cv2.FaceDetectorYN.create(
                str(self.model_path),
                "",  # Config file (empty for ONNX)
                initial_size,
                self.confidence_threshold,
                self.nms_threshold,
                self.top_k,
            )

            return detector

        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}") from e

    def detect(
        self,
        image: np.ndarray,
        return_landmarks: bool = False,
    ) -> List[Dict]:
        """
        Detect faces in image.

        Args:
            image: RGB image (H, W, 3)
            return_landmarks: Whether to return facial landmarks

        Returns:
            List of detection dictionaries with:
                - bbox: Dict[str, float] with 'x', 'y', 'width', 'height'
                - confidence: float
                - landmarks: Optional[np.ndarray] (5, 2) if return_landmarks=True

        Raises:
            DetectionError: If detection fails
        """
        if image is None or image.size == 0:
            raise DetectionError("Empty image provided")

        if image.ndim != 3 or image.shape[2] != 3:
            raise DetectionError(
                f"Expected RGB image (H, W, 3), got shape {image.shape}"
            )

        try:
            img_h, img_w = image.shape[:2]

            # Update detector input size
            self.detector.setInputSize((img_w, img_h))

            # Detect faces
            _, faces = self.detector.detect(image)

            if faces is None or len(faces) == 0:
                return []

            # Process detections
            detections = []
            for face in faces:
                # Parse detection
                x, y, w, h = face[:4].astype(int)
                conf = float(face[14])
                landmarks = face[4:14].reshape((5, 2)) if return_landmarks else None

                # Boundary checks
                if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
                    continue

                # Margin check
                dist_left = x
                dist_right = img_w - (x + w)
                dist_top = y
                dist_bottom = img_h - (y + h)

                if min(dist_left, dist_right, dist_top, dist_bottom) < self.min_margin:
                    continue

                # Size check
                if w < self.min_face_size or h < self.min_face_size:
                    continue

                # Build detection dict
                detection = {
                    "bbox": {
                        "x": float(x),
                        "y": float(y),
                        "width": float(w),
                        "height": float(h),
                    },
                    "confidence": conf,
                }

                if return_landmarks and landmarks is not None:
                    detection["landmarks"] = landmarks

                detections.append(detection)

            return detections

        except Exception as e:
            raise DetectionError(f"Detection failed: {e}") from e

    def crop_face(
        self,
        image: np.ndarray,
        detection: Dict,
        padding: float = 1.5,
    ) -> np.ndarray:
        """
        Crop face from image with padding.

        Args:
            image: RGB image (H, W, 3)
            detection: Detection dict from detect()
            padding: Bbox expansion factor (1.5 recommended)

        Returns:
            Cropped face image

        Raises:
            DetectionError: If cropping fails
        """
        try:
            img_h, img_w = image.shape[:2]
            bbox = detection["bbox"]

            # Expand bbox
            expanded_bbox = expand_bbox(bbox, img_w, img_h, expansion=padding)

            # Crop
            face_crop = crop_face(image, expanded_bbox)

            if face_crop.size == 0:
                raise DetectionError("Crop resulted in empty image")

            return face_crop

        except Exception as e:
            raise DetectionError(f"Face cropping failed: {e}") from e

    def detect_largest(
        self,
        image: np.ndarray,
        return_landmarks: bool = False,
    ) -> Optional[Dict]:
        """
        Detect largest face in image.

        Args:
            image: RGB image
            return_landmarks: Whether to return landmarks

        Returns:
            Detection dict or None if no faces found
        """
        detections = self.detect(image, return_landmarks=return_landmarks)

        if not detections:
            return None

        # Find largest face by area
        largest = max(
            detections, key=lambda d: d["bbox"]["width"] * d["bbox"]["height"]
        )

        return largest
