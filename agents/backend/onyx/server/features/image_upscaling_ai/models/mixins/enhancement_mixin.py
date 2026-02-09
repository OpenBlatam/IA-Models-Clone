"""
Enhancement Mixin

Contains image enhancement and post-processing methods.
"""

import logging
from typing import Union, Tuple, Optional
from pathlib import Path
from PIL import Image

from ..helpers import (
    UpscalingMetrics,
    QualityCalculator,
    PostprocessingMethods,
)

logger = logging.getLogger(__name__)


class EnhancementMixin:
    """
    Mixin providing image enhancement functionality.
    
    This mixin contains:
    - Edge enhancement
    - Texture enhancement
    - Color enhancement
    - Anti-aliasing
    - Artifact reduction
    - Adaptive contrast
    - Frequency analysis
    """
    
    def enhance_edges(
        self,
        image: Image.Image,
        strength: float = 1.2
    ) -> Image.Image:
        """Enhance edges in image."""
        return PostprocessingMethods.enhance_edges(image, strength=strength)
    
    def apply_anti_aliasing(
        self,
        image: Image.Image,
        strength: float = 0.3
    ) -> Image.Image:
        """Apply anti-aliasing to image."""
        return PostprocessingMethods.apply_anti_aliasing(image, strength=strength)
    
    def reduce_artifacts(
        self,
        image: Image.Image,
        method: str = "bilateral",
        strength: float = 0.5
    ) -> Image.Image:
        """Reduce artifacts in image."""
        return PostprocessingMethods.reduce_artifacts(image, method=method, strength=strength)
    
    def texture_enhancement(
        self,
        image: Image.Image,
        strength: float = 0.3
    ) -> Image.Image:
        """Enhance textures in image."""
        return PostprocessingMethods.texture_enhancement(image, strength=strength)
    
    def color_enhancement(
        self,
        image: Image.Image,
        saturation: float = 1.1,
        vibrance: float = 1.05
    ) -> Image.Image:
        """Enhance colors in image."""
        return PostprocessingMethods.color_enhancement(image, saturation=saturation, vibrance=vibrance)
    
    def adaptive_contrast_enhancement(
        self,
        image: Image.Image
    ) -> Image.Image:
        """Apply adaptive contrast enhancement."""
        return PostprocessingMethods.adaptive_contrast_enhancement(image)
    
    def enhance_with_frequency_analysis(
        self,
        image: Image.Image,
        strength: float = 0.5
    ) -> Image.Image:
        """Enhance image using frequency analysis."""
        return PostprocessingMethods.enhance_with_frequency_analysis(image, strength=strength)


