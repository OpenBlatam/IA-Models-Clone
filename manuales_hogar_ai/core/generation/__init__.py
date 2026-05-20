"""
Generation Module
================

Módulo especializado para generación de manuales.
"""

from .text_generator import TextManualGenerator
from .image_generator import ImageManualGenerator
from .combined_generator import CombinedManualGenerator

__all__ = [
    "TextManualGenerator",
    "ImageManualGenerator",
    "CombinedManualGenerator",
]

