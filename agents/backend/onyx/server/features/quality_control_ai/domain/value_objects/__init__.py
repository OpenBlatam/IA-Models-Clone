"""
Value Objects

Immutable value objects that represent domain concepts without identity.
"""

from .image_metadata import ImageMetadata
from .detection_result import DetectionResult
from .quality_metrics import QualityMetrics

__all__ = [
    "ImageMetadata",
    "DetectionResult",
    "QualityMetrics",
]



