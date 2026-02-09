"""
Quality Metrics
===============

Data classes for quality metrics.
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional


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


@dataclass
class ProcessingMetrics:
    """Metrics for clothing change processing."""
    processing_time: float
    mask_quality: float
    prompt_quality: float
    result_quality: Optional[float] = None
    success: bool = True
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


