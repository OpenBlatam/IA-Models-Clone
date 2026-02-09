"""
Core Models Submodule
Aggregates various deep learning model components.
"""

from .genre_classifier import DeepGenreClassifier
from .mood_detector import DeepMoodDetector
from .multitask import MultiTaskMusicModel
from .transformer_encoder import TransformerMusicEncoder
from .analyzer import DeepMusicAnalyzer, get_deep_analyzer

__all__ = [
    "DeepGenreClassifier",
    "DeepMoodDetector",
    "MultiTaskMusicModel",
    "TransformerMusicEncoder",
    "DeepMusicAnalyzer",
    "get_deep_analyzer",
]



