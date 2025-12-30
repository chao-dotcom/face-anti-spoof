"""Core exceptions for face anti-spoofing."""


class AntiSpoofError(Exception):
    """Base exception for anti-spoofing errors."""

    pass


class ModelLoadError(AntiSpoofError):
    """Failed to load model."""

    pass


class InferenceError(AntiSpoofError):
    """Inference failed."""

    pass


class InputValidationError(AntiSpoofError):
    """Input validation failed."""

    pass


class ConfigurationError(AntiSpoofError):
    """Configuration error."""

    pass


class DetectionError(AntiSpoofError):
    """Face detection error."""

    pass
