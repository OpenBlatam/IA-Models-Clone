"""
ML Audio Dataclasses Module

Data structures for ML audio analysis.
"""

from typing import Optional, List
from dataclasses import dataclass
import numpy as np


@dataclass
class AudioFeatures:
    """Extracted audio features."""
    mfcc: np.ndarray
    chroma: np.ndarray
    spectral_contrast: np.ndarray
    tonnetz: np.ndarray
    tempo: float
    beats: np.ndarray
    duration: float


@dataclass
class MLPrediction:
    """ML model prediction result."""
    genre: Optional[str] = None
    mood: Optional[str] = None
    energy_level: float = 0.0
    complexity_score: float = 0.0
    instrument_detection: List[str] = None
    confidence: float = 0.0



