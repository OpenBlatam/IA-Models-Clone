"""
Image Preprocessor
==================

Handles image preprocessing with validation and optimization for Flux2 clothing changer.

Refactored version with modular helpers for better maintainability.

This module provides ImagePreprocessor (backward compatible) and ImagePreprocessorV2 (enhanced).
"""

import logging
from typing import Union
from pathlib import Path
from PIL import Image
import numpy as np
import torch

try:
    from transformers import CLIPImageProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..constants import MAX_IMAGE_SIZE, MIN_IMAGE_SIZE
from .helpers import (
    ImageConverter,
    ImageValidator,
    ImageResizer,
    ImageEnhancer,
    ImageAnalyzer,
    ImageOptimizer,
)

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Handles image preprocessing with validation and optimization.
    
    Refactored to use modular helpers:
    - ImageConverter: Converts various formats to PIL
    - ImageValidator: Validates image properties
    - ImageResizer: Handles resizing operations
    """
    
    def __init__(
        self,
        clip_processor: CLIPImageProcessor,
        device: torch.device,
        max_size: int = MAX_IMAGE_SIZE,
        min_size: int = MIN_IMAGE_SIZE
    ):
        """
        Initialize image preprocessor.
        
        Args:
            clip_processor: CLIP image processor
            device: Torch device
            max_size: Maximum image size
            min_size: Minimum image size
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required")
        
        self.clip_processor = clip_processor
        self.device = device
        self.max_size = max_size
        self.min_size = min_size
        
        # Initialize helpers
        self.converter = ImageConverter
        self.validator = ImageValidator
        self.resizer = ImageResizer
        self.enhancer = ImageEnhancer
        self.analyzer = ImageAnalyzer
        self.optimizer = ImageOptimizer
    
    def preprocess(
        self,
        image: Union[Image.Image, str, Path, np.ndarray]
    ) -> torch.Tensor:
        """
        Preprocess image for model input with validation.
        
        Args:
            image: Input image (PIL, path, or numpy array)
            
        Returns:
            Preprocessed tensor on device
        """
        # Convert to PIL
        pil_image = self.converter.to_pil_image(image)
        
        # Validate and resize
        pil_image = self._validate_and_resize(pil_image)
        
        # Process with CLIP
        inputs = self.clip_processor(images=pil_image, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)
    
    def preprocess_for_inpainting(
        self,
        image: Union[Image.Image, str, Path, np.ndarray]
    ) -> Image.Image:
        """
        Preprocess image for inpainting pipeline.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed PIL image
        """
        # Convert to PIL
        pil_image = self.converter.to_pil_image(image)
        
        # Validate and resize
        return self._validate_and_resize(pil_image)
    
    def _validate_and_resize(self, image: Image.Image) -> Image.Image:
        """
        Validate and resize image if needed.
        
        Args:
            image: Input image
            
        Returns:
            Validated and resized image
        """
        # Validate dimensions
        self.validator.validate_dimensions(image)
        
        # Validate and resize using helper
        return self.resizer.validate_and_resize(
            image,
            max_size=self.max_size,
            min_size=self.min_size
        )
    
    def batch_preprocess(
        self,
        images: list[Union[Image.Image, str, Path, np.ndarray]]
    ) -> torch.Tensor:
        """
        Preprocess multiple images in batch.
        
        Args:
            images: List of input images
            
        Returns:
            Batch tensor on device
        """
        processed_images = []
        
        for image in images:
            pil_image = self.converter.to_pil_image(image)
            pil_image = self._validate_and_resize(pil_image)
            processed_images.append(pil_image)
        
        # Process batch with CLIP
        inputs = self.clip_processor(images=processed_images, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)
    
    def preprocess_with_enhancement(
        self,
        image: Union[Image.Image, str, Path, np.ndarray],
        auto_enhance: bool = True
    ) -> torch.Tensor:
        """
        Preprocess image with optional automatic enhancement.
        
        Args:
            image: Input image
            auto_enhance: Automatically enhance image quality
            
        Returns:
            Preprocessed tensor on device
        """
        pil_image = self.converter.to_pil_image(image)
        
        # Analyze and enhance if needed
        if auto_enhance:
            enhancement_needs = self.analyzer.needs_enhancement(pil_image)
            if any(enhancement_needs["recommendations"].values()):
                pil_image = self.enhancer.auto_enhance(
                    pil_image,
                    enhance_contrast=enhancement_needs["recommendations"]["enhance_contrast"],
                    enhance_sharpness=enhancement_needs["recommendations"]["enhance_sharpness"],
                    enhance_brightness=enhancement_needs["recommendations"]["enhance_brightness"],
                )
        
        # Validate and resize
        pil_image = self._validate_and_resize(pil_image)
        
        # Process with CLIP
        inputs = self.clip_processor(images=pil_image, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)
    
    def analyze_image(
        self,
        image: Union[Image.Image, str, Path, np.ndarray]
    ) -> dict:
        """
        Analyze image properties.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with analysis results
        """
        pil_image = self.converter.to_pil_image(image)
        return self.analyzer.analyze_quality(pil_image)
    
    def optimize_image(
        self,
        image: Union[Image.Image, str, Path, np.ndarray],
        mode: str = "memory"
    ) -> Image.Image:
        """
        Optimize image for processing.
        
        Args:
            image: Input image
            mode: Optimization mode ('memory' or 'quality')
            
        Returns:
            Optimized PIL image
        """
        pil_image = self.converter.to_pil_image(image)
        
        if mode == "memory":
            return self.optimizer.optimize_memory(pil_image, max_dimension=self.max_size)
        elif mode == "quality":
            return self.optimizer.optimize_for_quality(pil_image, min_dimension=self.min_size)
        else:
            return pil_image
