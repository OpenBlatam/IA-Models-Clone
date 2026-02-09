"""
ML/AI Mixin

Contains machine learning and AI-powered upscaling methods.
"""

import logging
from typing import Union, Tuple, Optional, List
from pathlib import Path
from PIL import Image

from ..helpers import (
    UpscalingMetrics,
    QualityCalculator,
    PostprocessingMethods,
)

logger = logging.getLogger(__name__)


class MLAIMixin:
    """
    Mixin providing ML/AI upscaling functionality.
    
    This mixin contains:
    - ML enhancement methods
    - Ensemble learning
    - Meta-learning
    - Intelligent fusion
    - Attention mechanisms
    - Gradient boosting
    - Neural style transfer
    """
    
    def upscale_with_ml_enhancement(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        ml_model: str = "auto",
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """
        Upscale with machine learning-based enhancement.
        
        This method should be implemented in the main class
        as it requires access to upscale() and other methods.
        """
        raise NotImplementedError("This method should be implemented in the main class")
    
    def upscale_with_ensemble_learning(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        ensemble_size: int = 5,
        diversity_weight: float = 0.3,
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """
        Upscale using ensemble learning with diverse methods.
        
        This method should be implemented in the main class.
        """
        raise NotImplementedError("This method should be implemented in the main class")
    
    def upscale_with_meta_learning(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        meta_strategy: str = "adaptive",
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """
        Upscale using meta-learning approach.
        
        This method should be implemented in the main class.
        """
        raise NotImplementedError("This method should be implemented in the main class")
    
    def upscale_with_intelligent_fusion(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        fusion_methods: List[str] = None,
        fusion_weights: List[float] = None,
        return_metrics: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """
        Upscale with intelligent fusion of multiple methods.
        
        This method should be implemented in the main class.
        """
        raise NotImplementedError("This method should be implemented in the main class")


