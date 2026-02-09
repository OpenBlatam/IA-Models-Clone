"""
Frequency Analysis Module
=========================

This module provides comprehensive frequency analysis for acceleration data
from sensors and encoder readings.
"""

from .frequency_analyzer import (
    FrequencyAnalyzer,
    FrequencyAnalysisMethod,
    FrequencyComponent,
    FrequencyAnalysisResult,
    MotionFrequencyBands
)

__all__ = [
    "FrequencyAnalyzer",
    "FrequencyAnalysisMethod",
    "FrequencyComponent",
    "FrequencyAnalysisResult",
    "MotionFrequencyBands",
]

