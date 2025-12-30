"""Test configuration and fixtures."""

import pytest
import numpy as np


@pytest.fixture
def sample_face_crop():
    """Generate sample face crop for testing."""
    return np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)


@pytest.fixture
def sample_image():
    """Generate sample image for testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_bbox():
    """Sample bounding box."""
    return {
        "x": 100.0,
        "y": 100.0,
        "width": 80.0,
        "height": 80.0,
    }


@pytest.fixture
def sample_detection(sample_bbox):
    """Sample detection result."""
    return {
        "bbox": sample_bbox,
        "confidence": 0.95,
    }
