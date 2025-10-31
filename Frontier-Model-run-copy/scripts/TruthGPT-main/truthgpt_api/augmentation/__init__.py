"""
TruthGPT Data Augmentation Module
================================

TensorFlow-like data augmentation implementations for TruthGPT.
"""

from .image_augmentation import ImageAugmentation
from .text_augmentation import TextAugmentation
from .audio_augmentation import AudioAugmentation
from .base import BaseAugmentation

__all__ = [
    'BaseAugmentation', 'ImageAugmentation', 
    'TextAugmentation', 'AudioAugmentation'
]


