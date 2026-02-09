"""
Batch Processing Mixin

Contains batch processing and async methods.
"""

import logging
import asyncio
from typing import Union, List, Optional, Callable, Tuple
from pathlib import Path
from PIL import Image

from ..helpers import (
    UpscalingMetrics,
    AsyncProcessingUtils,
    BatchProcessingUtils,
)

logger = logging.getLogger(__name__)


class BatchProcessingMixin:
    """
    Mixin providing batch processing functionality.
    
    This mixin contains:
    - Batch upscaling
    - Async upscaling
    - Batch async upscaling
    - Progress tracking
    - Parallel processing
    """
    
    async def upscale_async(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        method: str = "lanczos",
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ) -> Union[Image.Image, Tuple[Image.Image, UpscalingMetrics]]:
        """Asynchronously upscale image."""
        return await AsyncProcessingUtils.run_async(
            self.executor,
            self.upscale,
            image,
            scale_factor,
            method,
            progress_callback=progress_callback,
            **kwargs
        )
    
    def batch_upscale(
        self,
        images: List[Union[Image.Image, str, Path]],
        scale_factor: float,
        method: str = "lanczos",
        batch_size: int = 4,
        max_workers: int = 2,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ) -> List[Image.Image]:
        """Upscale multiple images in batch."""
        def process_image(img):
            return self.upscale(img, scale_factor, method, **kwargs)
        
        return BatchProcessingUtils.process_batch_sync(
            images,
            process_image,
            batch_size=batch_size,
            max_workers=max_workers,
            progress_callback=progress_callback
        )
    
    async def batch_upscale_async(
        self,
        images: List[Union[Image.Image, str, Path]],
        scale_factor: float,
        method: str = "lanczos",
        max_concurrent: int = 4,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ) -> List[Image.Image]:
        """Asynchronously upscale multiple images."""
        async def process_image(img):
            return await self.upscale_async(img, scale_factor, method, **kwargs)
        
        return await BatchProcessingUtils.process_batch_async(
            images,
            process_image,
            max_concurrent=max_concurrent,
            progress_callback=progress_callback
        )
    
    def batch_upscale_with_analysis(
        self,
        images: List[Union[Image.Image, str, Path]],
        scale_factor: float,
        batch_size: int = 4,
        max_workers: int = 2,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ) -> List[Tuple[Image.Image, Dict[str, Any]]]:
        """
        Batch upscale with per-image method optimization.
        
        Returns:
            List of tuples (upscaled_image, analysis_dict)
        """
        results = []
        
        for i, image in enumerate(images):
            if progress_callback:
                progress_callback(i, len(images))
            
            # Analyze and select best method
            if isinstance(image, (str, Path)):
                pil_image = Image.open(image).convert("RGB")
            else:
                pil_image = image.convert("RGB")
            
            from ..helpers import MethodSelector, ImageAnalysisUtils
            
            analysis = ImageAnalysisUtils.analyze_image_characteristics(pil_image)
            best_method = MethodSelector.select_best_method(pil_image, scale_factor)
            
            # Upscale with best method
            result = self.upscale(pil_image, scale_factor, best_method, **kwargs)
            
            results.append((result, {
                "method_used": best_method,
                "analysis": analysis,
            }))
        
        if progress_callback:
            progress_callback(len(images), len(images))
        
        return results


