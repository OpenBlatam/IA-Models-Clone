"""
Preprocessing Orchestrator
===========================

Orchestrates image preprocessing operations.
"""

import logging
from typing import Union, Optional
from pathlib import Path
from PIL import Image
import numpy as np
import torch

from ..processing import ImagePreprocessor

logger = logging.getLogger(__name__)


class PreprocessingOrchestrator:
    """Orchestrates image preprocessing operations."""
    
    def __init__(self, preprocessor: ImagePreprocessor):
        """
        Initialize preprocessing orchestrator.
        
        Args:
            preprocessor: Image preprocessor instance
        """
        self.preprocessor = preprocessor
    
    def preprocess_for_model(
        self,
        image: Union[Image.Image, str, Path, np.ndarray],
        auto_enhance: bool = False,
    ) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image
            auto_enhance: Automatically enhance image
            
        Returns:
            Preprocessed tensor
        """
        if auto_enhance:
            return self.preprocessor.preprocess_with_enhancement(image, auto_enhance=True)
        else:
            return self.preprocessor.preprocess(image)
    
    def preprocess_for_inpainting(
        self,
        image: Union[Image.Image, str, Path, np.ndarray],
    ) -> Image.Image:
        """
        Preprocess image for inpainting pipeline.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed PIL image
        """
        return self.preprocessor.preprocess_for_inpainting(image)
    
    def batch_preprocess(
        self,
        images: list[Union[Image.Image, str, Path, np.ndarray]],
    ) -> torch.Tensor:
        """
        Preprocess multiple images in batch.
        
        Args:
            images: List of input images
            
        Returns:
            Batch tensor
        """
        return self.preprocessor.batch_preprocess(images)
    
    def analyze(
        self,
        image: Union[Image.Image, str, Path, np.ndarray],
    ) -> dict:
        """
        Analyze image properties.
        
        Args:
            image: Input image
            
        Returns:
            Analysis results
        """
        return self.preprocessor.analyze_image(image)
    
    def optimize(
        self,
        image: Union[Image.Image, str, Path, np.ndarray],
        mode: str = "memory",
    ) -> Image.Image:
        """
        Optimize image for processing.
        
        Args:
            image: Input image
            mode: Optimization mode ('memory' or 'quality')
            
        Returns:
            Optimized image
        """
        return self.preprocessor.optimize_image(image, mode=mode)


