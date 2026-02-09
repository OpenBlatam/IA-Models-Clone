"""
ML Audio Submodule
Aggregates ML audio analysis components.
"""

from .dataclasses import AudioFeatures, MLPrediction
from .feature_extractor import AudioFeatureExtractor
from .classifier import GenreClassifier
from .analyzer import MLMusicAnalyzer, get_ml_analyzer

__all__ = [
    "AudioFeatures",
    "MLPrediction",
    "AudioFeatureExtractor",
    "GenreClassifier",
    "MLMusicAnalyzer",
    "get_ml_analyzer",
]



