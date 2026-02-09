"""
Image Transformer for Flux2 Clothing Changer
============================================

Advanced image transformation utilities.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from PIL import Image, ImageEnhance, ImageFilter
import logging

logger = logging.getLogger(__name__)


@dataclass
class TransformResult:
    """Transform result."""
    image: Image.Image
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ImageTransformer:
    """Advanced image transformation system."""
    
    def __init__(self):
        """Initialize image transformer."""
        self.transform_history: List[Dict[str, Any]] = []
    
    def resize(
        self,
        image: Image.Image,
        size: Tuple[int, int],
        resample: int = Image.LANCZOS,
        maintain_aspect: bool = True,
    ) -> TransformResult:
        """
        Resize image.
        
        Args:
            image: Input image
            size: Target size (width, height)
            resample: Resampling method
            maintain_aspect: Maintain aspect ratio
            
        Returns:
            Transform result
        """
        if maintain_aspect:
            image.thumbnail(size, resample)
            result_image = image
        else:
            result_image = image.resize(size, resample)
        
        return TransformResult(
            image=result_image,
            metadata={
                "transform": "resize",
                "original_size": image.size,
                "new_size": result_image.size,
            },
        )
    
    def adjust_brightness(
        self,
        image: Image.Image,
        factor: float,
    ) -> TransformResult:
        """
        Adjust image brightness.
        
        Args:
            image: Input image
            factor: Brightness factor (1.0 = no change)
            
        Returns:
            Transform result
        """
        enhancer = ImageEnhance.Brightness(image)
        result_image = enhancer.enhance(factor)
        
        return TransformResult(
            image=result_image,
            metadata={
                "transform": "brightness",
                "factor": factor,
            },
        )
    
    def adjust_contrast(
        self,
        image: Image.Image,
        factor: float,
    ) -> TransformResult:
        """
        Adjust image contrast.
        
        Args:
            image: Input image
            factor: Contrast factor (1.0 = no change)
            
        Returns:
            Transform result
        """
        enhancer = ImageEnhance.Contrast(image)
        result_image = enhancer.enhance(factor)
        
        return TransformResult(
            image=result_image,
            metadata={
                "transform": "contrast",
                "factor": factor,
            },
        )
    
    def adjust_saturation(
        self,
        image: Image.Image,
        factor: float,
    ) -> TransformResult:
        """
        Adjust image saturation.
        
        Args:
            image: Input image
            factor: Saturation factor (1.0 = no change)
            
        Returns:
            Transform result
        """
        enhancer = ImageEnhance.Color(image)
        result_image = enhancer.enhance(factor)
        
        return TransformResult(
            image=result_image,
            metadata={
                "transform": "saturation",
                "factor": factor,
            },
        )
    
    def apply_filter(
        self,
        image: Image.Image,
        filter_type: str,
    ) -> TransformResult:
        """
        Apply image filter.
        
        Args:
            image: Input image
            filter_type: Filter type (blur, sharpen, edge_enhance, etc.)
            
        Returns:
            Transform result
        """
        filter_map = {
            "blur": ImageFilter.BLUR,
            "sharpen": ImageFilter.SHARPEN,
            "edge_enhance": ImageFilter.EDGE_ENHANCE,
            "smooth": ImageFilter.SMOOTH,
        }
        
        if filter_type not in filter_map:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        result_image = image.filter(filter_map[filter_type])
        
        return TransformResult(
            image=result_image,
            metadata={
                "transform": "filter",
                "filter_type": filter_type,
            },
        )
    
    def convert_format(
        self,
        image: Image.Image,
        mode: str,
    ) -> TransformResult:
        """
        Convert image format/mode.
        
        Args:
            image: Input image
            mode: Target mode (RGB, RGBA, L, etc.)
            
        Returns:
            Transform result
        """
        result_image = image.convert(mode)
        
        return TransformResult(
            image=result_image,
            metadata={
                "transform": "convert",
                "original_mode": image.mode,
                "new_mode": mode,
            },
        )
    
    def apply_transforms(
        self,
        image: Image.Image,
        transforms: List[Dict[str, Any]],
    ) -> TransformResult:
        """
        Apply multiple transforms.
        
        Args:
            image: Input image
            transforms: List of transform definitions
            
        Returns:
            Transform result
        """
        result_image = image
        metadata = {"transforms": []}
        
        for transform_def in transforms:
            transform_type = transform_def["type"]
            params = transform_def.get("params", {})
            
            if transform_type == "resize":
                result = self.resize(result_image, **params)
            elif transform_type == "brightness":
                result = self.adjust_brightness(result_image, **params)
            elif transform_type == "contrast":
                result = self.adjust_contrast(result_image, **params)
            elif transform_type == "saturation":
                result = self.adjust_saturation(result_image, **params)
            elif transform_type == "filter":
                result = self.apply_filter(result_image, **params)
            elif transform_type == "convert":
                result = self.convert_format(result_image, **params)
            else:
                logger.warning(f"Unknown transform type: {transform_type}")
                continue
            
            result_image = result.image
            metadata["transforms"].append(result.metadata)
        
        return TransformResult(
            image=result_image,
            metadata=metadata,
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get transformer statistics."""
        return {
            "total_transforms": len(self.transform_history),
        }


