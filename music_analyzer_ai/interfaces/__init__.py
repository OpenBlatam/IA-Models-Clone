"""
Interfaces and Abstract Base Classes for Music Analyzer AI
Defines contracts for all components
"""

from .model_interface import IMusicModel, IMusicClassifier, IMusicEncoder
from .trainer_interface import ITrainer, ITrainingCallback
from .analyzer_interface import IMusicAnalyzer, IFeatureExtractor
from .inference_interface import IInferenceEngine, IBatchProcessor

__all__ = [
    "IMusicModel",
    "IMusicClassifier", 
    "IMusicEncoder",
    "ITrainer",
    "ITrainingCallback",
    "IMusicAnalyzer",
    "IFeatureExtractor",
    "IInferenceEngine",
    "IBatchProcessor"
]













