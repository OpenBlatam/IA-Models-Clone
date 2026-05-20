"""
Deep Learning Service - Modular Architecture
============================================

Modular deep learning service following best practices:
- Object-oriented model architectures
- Functional data processing pipelines
- Comprehensive training and evaluation
- Experiment tracking and checkpointing
"""

__version__ = "2.0.0"
__author__ = "Blatam Academy"

from .service import DeepLearningService
from .config.config_loader import ConfigLoader, TrainingConfig

__all__ = [
    "DeepLearningService",
    "ConfigLoader",
    "TrainingConfig",
]



