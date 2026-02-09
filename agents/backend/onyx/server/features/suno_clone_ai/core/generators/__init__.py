"""
Music Generators Module

Provides unified interface for different music generation models
following deep learning best practices.
"""

from .base_generator import BaseMusicGenerator
from .transformers_generator import TransformersMusicGenerator
from .refactored_generator import RefactoredMusicGenerator, get_refactored_music_generator

__all__ = [
    "BaseMusicGenerator",
    "TransformersMusicGenerator",
    "RefactoredMusicGenerator",
    "get_refactored_music_generator"
]

