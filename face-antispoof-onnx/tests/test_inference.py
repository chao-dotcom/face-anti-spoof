"""Unit tests for face anti-spoofing inference."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.inference import FaceAntiSpoof, VideoAntiSpoof
from src.core.exceptions import ModelLoadError, InferenceError, InputValidationError


class TestFaceAntiSpoof:
    """Tests for FaceAntiSpoof class."""

    @pytest.fixture
    def mock_session(self):
        """Create mock ONNX session."""
        session = Mock()
        session.get_inputs.return_value = [Mock(name="input")]
        session.run.return_value = [np.array([[2.5, 1.0]])]  # real_logit, spoof_logit
        return session

    @pytest.fixture
    def model(self, mock_session, tmp_path):
        """Create FaceAntiSpoof instance with mocked session."""
        model_path = tmp_path / "test_model.onnx"
        model_path.touch()  # Create empty file

        with patch("src.inference.inference.ort.InferenceSession") as mock_ort:
            mock_ort.return_value = mock_session
            model = FaceAntiSpoof(
                model_path=str(model_path),
                threshold=0.5,
                enable_quality_check=False,
            )

        return model

    def test_init_success(self, tmp_path):
        """Test successful initialization."""
        model_path = tmp_path / "model.onnx"
        model_path.touch()

        with patch("src.inference.inference.ort.InferenceSession"):
            model = FaceAntiSpoof(
                model_path=str(model_path),
                threshold=0.5,
            )

            assert model.threshold == 0.5
            assert model.input_size == 128

    def test_init_model_not_found(self):
        """Test initialization with missing model."""
        with pytest.raises(ModelLoadError, match="Model not found"):
            FaceAntiSpoof(model_path="nonexistent.onnx")

    def test_predict_success(self, model):
        """Test successful prediction."""
        # Create dummy face crop
        face_crop = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

        result = model.predict(face_crop)

        assert "is_real" in result
        assert "status" in result
        assert "confidence" in result
        assert "logit_diff" in result
        assert result["is_real"] is True  # 2.5 - 1.0 = 1.5 > 0.5
        assert result["status"] == "real"

    def test_predict_empty_input(self, model):
        """Test prediction with empty input."""
        with pytest.raises(InputValidationError, match="Empty face crop"):
            model.predict(np.array([]))

    def test_predict_wrong_shape(self, model):
        """Test prediction with wrong input shape."""
        # Grayscale instead of RGB
        face_crop = np.random.randint(0, 255, (128, 128), dtype=np.uint8)

        with pytest.raises(InputValidationError, match="Expected RGB image"):
            model.predict(face_crop)

    def test_predict_batch(self, model):
        """Test batch prediction."""
        face_crops = [
            np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            for _ in range(3)
        ]

        # Mock batch output
        model.session.run.return_value = [
            np.array([[2.0, 1.0], [1.0, 2.0], [3.0, 0.5]])
        ]

        results = model.predict_batch(face_crops)

        assert len(results) == 3
        assert results[0]["is_real"] is True  # 2.0 - 1.0 = 1.0 > 0.5
        assert results[1]["is_real"] is False  # 1.0 - 2.0 = -1.0 < 0.5
        assert results[2]["is_real"] is True  # 3.0 - 0.5 = 2.5 > 0.5

    def test_get_stats(self, model):
        """Test statistics tracking."""
        face_crop = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

        # Make predictions
        model.predict(face_crop)  # Real
        model.session.run.return_value = [np.array([[0.5, 2.0]])]  # Spoof
        model.predict(face_crop)

        stats = model.get_stats()

        assert stats["total_predictions"] == 2
        assert stats["real_predictions"] == 1
        assert stats["spoof_predictions"] == 1
        assert stats["spoof_rate"] == 0.5


class TestVideoAntiSpoof:
    """Tests for VideoAntiSpoof class."""

    @pytest.fixture
    def video_model(self, tmp_path):
        """Create VideoAntiSpoof instance."""
        model_path = tmp_path / "model.onnx"
        model_path.touch()

        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input")]
        mock_session.run.return_value = [np.array([[2.0, 1.0]])]

        with patch("src.inference.inference.ort.InferenceSession") as mock_ort:
            mock_ort.return_value = mock_session
            model = VideoAntiSpoof(
                model_path=str(model_path),
                temporal_window=3,
                consistency_threshold=0.8,
                enable_quality_check=False,
            )

        return model

    def test_predict_temporal_not_ready(self, video_model):
        """Test temporal prediction with insufficient frames."""
        face_crop = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

        # First frame
        result = video_model.predict_temporal(face_crop)

        assert result["ready"] is False
        assert "frames_needed" in result
        assert result["frames_needed"] == 2

    def test_predict_temporal_ready_consistent(self, video_model):
        """Test temporal prediction with consistent frames."""
        face_crop = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

        # Add 3 frames (all real)
        for _ in range(3):
            result = video_model.predict_temporal(face_crop)

        assert result["ready"] is True
        assert result["is_real"] is True
        assert result["decision"] == "real"
        assert result["consistency"] == 1.0  # 3/3 agree
        assert "avg_confidence" in result

    def test_predict_temporal_ready_mixed(self, video_model):
        """Test temporal prediction with mixed frames."""
        face_crop = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

        # Frame 1: Real
        video_model.predict_temporal(face_crop)

        # Frame 2: Spoof
        video_model.session.run.return_value = [np.array([[0.5, 2.0]])]
        video_model.predict_temporal(face_crop)

        # Frame 3: Real
        video_model.session.run.return_value = [np.array([[2.0, 1.0]])]
        result = video_model.predict_temporal(face_crop)

        assert result["ready"] is False  # 2/3 = 0.67 < 0.8 threshold

    def test_reset(self, video_model):
        """Test reset functionality."""
        face_crop = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

        # Add frames
        video_model.predict_temporal(face_crop)
        video_model.predict_temporal(face_crop)

        # Reset
        video_model.reset()

        # Should need full window again
        result = video_model.predict_temporal(face_crop)
        assert result["ready"] is False
        assert result["frames_needed"] == 2
