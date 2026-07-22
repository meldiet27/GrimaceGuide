"""Domain-level exceptions used across the core package."""


class GrimaceGuideError(Exception):
    """Base exception for all GrimaceGuide-specific errors."""


class ImageLoadError(GrimaceGuideError):
    """Raised when an image cannot be read or decoded."""


class LandmarkDetectionError(GrimaceGuideError):
    """Raised when landmark detection fails or returns no results."""


class LandmarkAPIError(GrimaceGuideError):
    """Raised when the remote landmark inference API fails."""


class ScoringError(GrimaceGuideError):
    """Raised when the FGS scoring cannot be computed."""
