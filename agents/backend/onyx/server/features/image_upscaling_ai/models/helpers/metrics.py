"""
Metrics for Image Upscaling
===========================

Data classes for upscaling metrics and quality assessment.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List


@dataclass
class UpscalingMetrics:
    """Metrics for upscaling operation."""
    original_size: Tuple[int, int]
    upscaled_size: Tuple[int, int]
    scale_factor: float
    processing_time: float
    quality_score: Optional[float] = None
    sharpness_score: Optional[float] = None
    artifact_score: Optional[float] = None
    method_used: str = ""
    success: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class QualityMetrics:
    """Quality assessment metrics."""
    sharpness: float
    contrast: float
    brightness: float
    noise_level: float
    artifact_count: float
    overall_quality: float


@dataclass
class ImageQualityMetrics:
    """Metrics for image quality assessment."""
    brightness: float
    contrast: float
    sharpness: float
    resolution: Tuple[int, int]
    is_valid: bool
    warnings: List[str]
    errors: List[str]


