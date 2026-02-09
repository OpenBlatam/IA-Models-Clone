"""
Image Preprocessor V2
=====================

Advanced image preprocessing with modular architecture and enhanced features.

Refactored version with:
- Modular helper system
- Enhanced preprocessing pipeline
- Quality analysis and auto-enhancement
- Batch processing optimization
- Caching support
"""

import logging
from typing import Union, Optional, Dict, Any, List, Callable
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


class ImagePreprocessorV2:
    """
    Advanced image preprocessor with modular architecture.
    
    Features:
    - Multiple preprocessing modes
    - Automatic quality enhancement
    - Batch processing
    - Image analysis
    - Optimization strategies
    - Caching support
    """
    
    def __init__(
        self,
        clip_processor: CLIPImageProcessor,
        device: torch.device,
        max_size: int = MAX_IMAGE_SIZE,
        min_size: int = MIN_IMAGE_SIZE,
        auto_enhance: bool = False,
        optimization_mode: str = "balanced",
        enable_cache: bool = False,
    ):
        """
        Initialize image preprocessor V2.
        
        Args:
            clip_processor: CLIP image processor
            device: Torch device
            max_size: Maximum image size
            min_size: Minimum image size
            auto_enhance: Automatically enhance images
            optimization_mode: Optimization mode ('memory', 'quality', 'balanced')
            enable_cache: Enable preprocessing cache
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required")
        
        self.clip_processor = clip_processor
        self.device = device
        self.max_size = max_size
        self.min_size = min_size
        self.auto_enhance = auto_enhance
        self.optimization_mode = optimization_mode
        self.enable_cache = enable_cache
        
        # Initialize helpers
        self.converter = ImageConverter
        self.validator = ImageValidator
        self.resizer = ImageResizer
        self.enhancer = ImageEnhancer
        self.analyzer = ImageAnalyzer
        self.optimizer = ImageOptimizer
        
        # Cache for processed images
        self._cache: Dict[str, torch.Tensor] = {} if enable_cache else None
    
    def preprocess(
        self,
        image: Union[Image.Image, str, Path, np.ndarray],
        mode: str = "standard",
        return_analysis: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image
            mode: Preprocessing mode ('standard', 'enhanced', 'optimized')
            return_analysis: Return quality analysis
            
        Returns:
            Preprocessed tensor (and analysis if requested)
        """
        # Check cache
        cache_key = self._get_cache_key(image, mode)
        if self.enable_cache and cache_key in self._cache:
            logger.debug("Using cached preprocessed image")
            return self._cache[cache_key]
        
        # Convert to PIL
        pil_image = self.converter.to_pil_image(image)
        
        # Analyze if needed
        analysis = None
        if return_analysis or self.auto_enhance:
            analysis = self.analyzer.analyze_quality(pil_image)
        
        # Apply preprocessing based on mode
        if mode == "enhanced" or (self.auto_enhance and analysis):
            pil_image = self._apply_enhancement(pil_image, analysis)
        elif mode == "optimized":
            pil_image = self._apply_optimization(pil_image)
        
        # Validate and resize
        pil_image = self._validate_and_resize(pil_image)
        
        # Process with CLIP
        inputs = self.clip_processor(images=pil_image, return_tensors="pt")
        tensor = inputs["pixel_values"].to(self.device)
        
        # Cache result
        if self.enable_cache:
            self._cache[cache_key] = tensor
        
        if return_analysis:
            return tensor, analysis or self.analyzer.analyze_quality(pil_image)
        
        return tensor
    
    def preprocess_for_inpainting(
        self,
        image: Union[Image.Image, str, Path, np.ndarray],
        enhance: bool = False
    ) -> Image.Image:
        """
        Preprocess image for inpainting pipeline.
        
        Args:
            image: Input image
            enhance: Apply enhancement
            
        Returns:
            Preprocessed PIL image
        """
        pil_image = self.converter.to_pil_image(image)
        
        if enhance:
            analysis = self.analyzer.needs_enhancement(pil_image)
            pil_image = self._apply_enhancement(pil_image, analysis)
        
        return self._validate_and_resize(pil_image)
    
    def batch_preprocess(
        self,
        images: List[Union[Image.Image, str, Path, np.ndarray]],
        mode: str = "standard",
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> torch.Tensor:
        """
        Preprocess multiple images in batch.
        
        Args:
            images: List of input images
            mode: Preprocessing mode
            progress_callback: Optional progress callback
            
        Returns:
            Batch tensor on device
        """
        processed_images = []
        total = len(images)
        
        for i, image in enumerate(images):
            if progress_callback:
                progress_callback(i, total)
            
            pil_image = self.converter.to_pil_image(image)
            
            if mode == "enhanced":
                analysis = self.analyzer.needs_enhancement(pil_image)
                pil_image = self._apply_enhancement(pil_image, analysis)
            elif mode == "optimized":
                pil_image = self._apply_optimization(pil_image)
            
            pil_image = self._validate_and_resize(pil_image)
            processed_images.append(pil_image)
        
        if progress_callback:
            progress_callback(total, total)
        
        # Process batch with CLIP
        inputs = self.clip_processor(images=processed_images, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)
    
    def analyze_image(
        self,
        image: Union[Image.Image, str, Path, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Analyze image properties and quality.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with analysis results
        """
        pil_image = self.converter.to_pil_image(image)
        quality = self.analyzer.analyze_quality(pil_image)
        needs = self.analyzer.needs_enhancement(pil_image)
        
        return {
            **quality,
            "enhancement_recommendations": needs["recommendations"],
        }
    
    def optimize_image(
        self,
        image: Union[Image.Image, str, Path, np.ndarray],
        mode: Optional[str] = None
    ) -> Image.Image:
        """
        Optimize image for processing.
        
        Args:
            image: Input image
            mode: Optimization mode (uses instance default if None)
            
        Returns:
            Optimized PIL image
        """
        pil_image = self.converter.to_pil_image(image)
        mode = mode or self.optimization_mode
        
        if mode == "memory":
            return self.optimizer.optimize_memory(pil_image, max_dimension=self.max_size)
        elif mode == "quality":
            return self.optimizer.optimize_for_quality(pil_image, min_dimension=self.min_size)
        else:  # balanced
            # Apply both optimizations
            optimized = self.optimizer.optimize_memory(pil_image, max_dimension=self.max_size)
            return self.optimizer.optimize_for_quality(optimized, min_dimension=self.min_size)
    
    def _validate_and_resize(self, image: Image.Image) -> Image.Image:
        """Validate and resize image."""
        self.validator.validate_dimensions(image)
        return self.resizer.validate_and_resize(
            image,
            max_size=self.max_size,
            min_size=self.min_size
        )
    
    def _apply_enhancement(
        self,
        image: Image.Image,
        analysis: Optional[Dict[str, Any]] = None
    ) -> Image.Image:
        """Apply automatic enhancement based on analysis."""
        if analysis is None:
            analysis = self.analyzer.needs_enhancement(image)
        
        recommendations = analysis.get("recommendations", {})
        
        return self.enhancer.auto_enhance(
            image,
            enhance_contrast=recommendations.get("enhance_contrast", False),
            enhance_sharpness=recommendations.get("enhance_sharpness", True),
            enhance_brightness=recommendations.get("enhance_brightness", False),
            reduce_noise=recommendations.get("reduce_noise", False),
        )
    
    def _apply_optimization(self, image: Image.Image) -> Image.Image:
        """Apply optimization based on mode."""
        return self.optimize_image(image, mode=self.optimization_mode)
    
    def _get_cache_key(
        self,
        image: Union[Image.Image, str, Path, np.ndarray],
        mode: str
    ) -> str:
        """Generate cache key for image."""
        if isinstance(image, (str, Path)):
            return f"{str(image)}_{mode}"
        elif isinstance(image, np.ndarray):
            return f"numpy_{hash(image.tobytes())}_{mode}"
        else:
            return f"pil_{id(image)}_{mode}"
    
    def clear_cache(self) -> None:
        """Clear preprocessing cache."""
        if self._cache:
            self._cache.clear()
            logger.info("Preprocessing cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._cache:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "size": len(self._cache),
            "keys": list(self._cache.keys()),
        }

