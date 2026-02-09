"""
Metrics Utilities
=================

Utilities for processing metrics and quality assessment.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ProcessingMetrics:
    """Metrics for clothing change processing."""
    processing_time: float
    mask_quality: float
    prompt_quality: float
    result_quality: Optional[float] = None
    success: bool = True
    errors: List[str] = field(default_factory=list)


