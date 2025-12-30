"""Input validation and quality checking utilities."""

from typing import Dict, Any, Tuple
import numpy as np
import cv2


class LightingQualityChecker:
    """Pre-inference lighting quality validation."""

    def __init__(
        self,
        min_intensity: float = 30,
        max_intensity: float = 250,
        min_std: float = 15,
        max_saturation_ratio: float = 0.05,
    ):
        """
        Initialize lighting quality checker.

        Args:
            min_intensity: Minimum acceptable mean intensity
            max_intensity: Maximum acceptable mean intensity
            min_std: Minimum acceptable standard deviation
            max_saturation_ratio: Maximum acceptable ratio of saturated pixels
        """
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.min_std = min_std
        self.max_saturation_ratio = max_saturation_ratio

    def check(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze lighting quality of input image.

        Args:
            image: RGB image (H, W, 3)

        Returns:
            Dictionary with quality metrics and recommendations
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Basic statistics
        mean_intensity = float(np.mean(gray))
        std_intensity = float(np.std(gray))

        # Check for saturation
        saturated_pixels = np.sum((gray < 5) | (gray > 250))
        saturation_ratio = saturated_pixels / gray.size

        # Evaluate conditions
        too_dark = mean_intensity < self.min_intensity
        too_bright = mean_intensity > self.max_intensity
        low_contrast = std_intensity < self.min_std
        over_saturated = saturation_ratio > self.max_saturation_ratio

        # Overall quality
        is_acceptable = not (too_dark or too_bright or low_contrast or over_saturated)

        # Generate recommendation
        issues = []
        if too_dark:
            issues.append("Insufficient lighting - increase ambient light")
        if too_bright:
            issues.append("Overexposed - reduce direct lighting or backlight")
        if low_contrast:
            issues.append("Low contrast - improve lighting diffusion")
        if over_saturated:
            issues.append("Too many saturated pixels - adjust exposure")

        return {
            "acceptable": is_acceptable,
            "mean_intensity": mean_intensity,
            "std_intensity": std_intensity,
            "saturation_ratio": saturation_ratio,
            "issues": issues,
            "recommendation": "OK" if is_acceptable else "; ".join(issues),
        }


class BlurDetector:
    """Motion blur detection using Laplacian variance."""

    def __init__(self, threshold: float = 100.0):
        """
        Initialize blur detector.

        Args:
            threshold: Minimum Laplacian variance for acceptable sharpness
        """
        self.threshold = threshold

    def estimate_blur(self, image: np.ndarray) -> float:
        """
        Estimate blur level using Laplacian variance.

        Args:
            image: RGB or grayscale image

        Returns:
            Blur score (higher = sharper, lower = blurrier)
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(laplacian_var)

    def is_acceptable(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Check if image blur is acceptable.

        Args:
            image: RGB or grayscale image

        Returns:
            Tuple of (acceptable, blur_score)
        """
        blur_score = self.estimate_blur(image)
        acceptable = blur_score >= self.threshold
        return acceptable, blur_score


class FaceQualityChecker:
    """Comprehensive face quality checking."""

    def __init__(
        self,
        min_face_size: int = 64,
        min_confidence: float = 0.8,
        blur_threshold: float = 100.0,
        lighting_check: bool = True,
    ):
        """
        Initialize face quality checker.

        Args:
            min_face_size: Minimum face width/height in pixels
            min_confidence: Minimum detection confidence
            blur_threshold: Minimum blur score for acceptable sharpness
            lighting_check: Whether to perform lighting quality check
        """
        self.min_face_size = min_face_size
        self.min_confidence = min_confidence

        self.blur_detector = BlurDetector(threshold=blur_threshold)
        self.lighting_checker = (
            LightingQualityChecker() if lighting_check else None
        )

    def check(
        self,
        face_crop: np.ndarray,
        bbox: Dict[str, float],
        confidence: float,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive quality check on face.

        Args:
            face_crop: Cropped face image (RGB)
            bbox: Bounding box dict with 'width' and 'height'
            confidence: Detection confidence score

        Returns:
            Dictionary with quality assessment
        """
        issues = []

        # Size check
        if bbox["width"] < self.min_face_size or bbox["height"] < self.min_face_size:
            issues.append(
                f"Face too small ({bbox['width']:.0f}x{bbox['height']:.0f} px)"
            )

        # Confidence check
        if confidence < self.min_confidence:
            issues.append(f"Low detection confidence ({confidence:.2f})")

        # Blur check
        acceptable_blur, blur_score = self.blur_detector.is_acceptable(face_crop)
        if not acceptable_blur:
            issues.append(f"Image too blurry (score: {blur_score:.1f})")

        # Lighting check
        lighting_result = None
        if self.lighting_checker:
            lighting_result = self.lighting_checker.check(face_crop)
            if not lighting_result["acceptable"]:
                issues.extend(lighting_result["issues"])

        is_acceptable = len(issues) == 0

        return {
            "acceptable": is_acceptable,
            "issues": issues,
            "blur_score": blur_score,
            "lighting": lighting_result,
            "recommendation": "OK" if is_acceptable else "; ".join(issues),
        }
