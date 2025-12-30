"""Unit tests for face detection."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path

from src.detection import FaceDetector
from src.core.exceptions import DetectionError, ModelLoadError


class TestFaceDetector:
    """Tests for FaceDetector class."""

    @pytest.fixture
    def mock_cv_detector(self):
        """Create mock OpenCV detector."""
        detector = Mock()
        detector.detect.return_value = (
            None,
            np.array([
                [100, 100, 80, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95]
            ])
        )
        return detector

    @pytest.fixture
    def detector(self, mock_cv_detector, tmp_path):
        """Create FaceDetector instance."""
        model_path = tmp_path / "detector.onnx"
        model_path.touch()

        with patch("src.detection.detector.cv2.FaceDetectorYN.create") as mock_create:
            mock_create.return_value = mock_cv_detector
            detector = FaceDetector(model_path=str(model_path))

        return detector

    def test_init_success(self, tmp_path):
        """Test successful initialization."""
        model_path = tmp_path / "detector.onnx"
        model_path.touch()

        with patch("src.detection.detector.cv2.FaceDetectorYN.create"):
            detector = FaceDetector(
                model_path=str(model_path),
                confidence_threshold=0.8,
                min_face_size=60,
            )

            assert detector.confidence_threshold == 0.8
            assert detector.min_face_size == 60

    def test_init_model_not_found(self):
        """Test initialization with missing model."""
        with pytest.raises(ModelLoadError, match="Model not found"):
            FaceDetector(model_path="nonexistent.onnx")

    def test_detect_success(self, detector):
        """Test successful face detection."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        detections = detector.detect(image)

        assert len(detections) == 1
        assert "bbox" in detections[0]
        assert "confidence" in detections[0]
        assert detections[0]["bbox"]["x"] == 100
        assert detections[0]["bbox"]["width"] == 80
        assert detections[0]["confidence"] == pytest.approx(0.95)

    def test_detect_empty_image(self, detector):
        """Test detection with empty image."""
        with pytest.raises(DetectionError, match="Empty image"):
            detector.detect(np.array([]))

    def test_detect_wrong_shape(self, detector):
        """Test detection with wrong image shape."""
        image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)  # Grayscale

        with pytest.raises(DetectionError, match="Expected RGB image"):
            detector.detect(image)

    def test_detect_no_faces(self, detector):
        """Test detection with no faces found."""
        detector.detector.detect.return_value = (None, None)

        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = detector.detect(image)

        assert len(detections) == 0

    def test_detect_face_too_small(self, detector):
        """Test detection filtering small faces."""
        # Face smaller than min_face_size
        detector.detector.detect.return_value = (
            None,
            np.array([[100, 100, 40, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95]])
        )

        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = detector.detect(image)

        assert len(detections) == 0  # Filtered out

    def test_crop_face(self, detector):
        """Test face cropping."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        detection = {
            "bbox": {"x": 100.0, "y": 100.0, "width": 80.0, "height": 80.0},
            "confidence": 0.95,
        }

        face_crop = detector.crop_face(image, detection, padding=1.5)

        # With 1.5x padding, 80x80 becomes 120x120
        # Check that crop is not empty
        assert face_crop.size > 0
        assert face_crop.shape[2] == 3  # RGB

    def test_detect_largest(self, detector):
        """Test detecting largest face."""
        # Return multiple faces
        detector.detector.detect.return_value = (
            None,
            np.array([
                [100, 100, 60, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95],  # 3600 area
                [300, 100, 80, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.90],  # 6400 area (largest)
            ])
        )

        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        largest = detector.detect_largest(image)

        assert largest is not None
        assert largest["bbox"]["width"] == 80  # Largest face

    def test_detect_largest_no_faces(self, detector):
        """Test detect_largest with no faces."""
        detector.detector.detect.return_value = (None, None)

        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        largest = detector.detect_largest(image)

        assert largest is None
