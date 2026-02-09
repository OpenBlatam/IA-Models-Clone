"""
Image Optimizer for Piel Mejorador AI SAM3
==========================================

Advanced image processing optimizations.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

logger = logging.getLogger(__name__)


class ImageOptimizer:
    """
    Optimizes images before processing.
    
    Features:
    - Resize optimization
    - Quality adjustment
    - Format conversion
    - Pre-processing enhancements
    """
    
    @staticmethod
    def optimize_for_processing(
        image_path: Path,
        max_dimension: int = 2048,
        quality: int = 85,
        target_format: str = "JPEG"
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        Optimize image for AI processing.
        
        Args:
            image_path: Path to image
            max_dimension: Maximum dimension (width or height)
            quality: JPEG quality (1-100)
            target_format: Target format (JPEG, PNG, WEBP)
            
        Returns:
            Tuple of (optimized_path, metadata)
        """
        try:
            with Image.open(image_path) as img:
                original_size = img.size
                original_format = img.format
                
                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # Resize if too large
                if max(img.size) > max_dimension:
                    img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                
                # Save optimized image
                output_path = image_path.parent / f"optimized_{image_path.name}"
                
                if target_format == "JPEG":
                    img.save(
                        output_path,
                        "JPEG",
                        quality=quality,
                        optimize=True
                    )
                elif target_format == "PNG":
                    img.save(output_path, "PNG", optimize=True)
                elif target_format == "WEBP":
                    img.save(output_path, "WEBP", quality=quality)
                else:
                    img.save(output_path)
                
                metadata = {
                    "original_size": original_size,
                    "optimized_size": img.size,
                    "original_format": original_format,
                    "optimized_format": target_format,
                    "size_reduction": 1 - (img.size[0] * img.size[1]) / (original_size[0] * original_size[1]),
                }
                
                logger.info(
                    f"Image optimized: {original_size} -> {img.size} "
                    f"({metadata['size_reduction']:.1%} reduction)"
                )
                
                return output_path, metadata
                
        except Exception as e:
            logger.error(f"Error optimizing image: {e}")
            return image_path, {}
    
    @staticmethod
    def preprocess_for_enhancement(
        image_path: Path,
        brightness: float = 1.0,
        contrast: float = 1.0,
        sharpness: float = 1.0
    ) -> Path:
        """
        Preprocess image for enhancement.
        
        Args:
            image_path: Path to image
            brightness: Brightness factor (0.0-2.0)
            contrast: Contrast factor (0.0-2.0)
            sharpness: Sharpness factor (0.0-2.0)
            
        Returns:
            Path to preprocessed image
        """
        try:
            with Image.open(image_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # Apply enhancements
                if brightness != 1.0:
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(brightness)
                
                if contrast != 1.0:
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(contrast)
                
                if sharpness != 1.0:
                    enhancer = ImageEnhance.Sharpness(img)
                    img = enhancer.enhance(sharpness)
                
                # Save preprocessed image
                output_path = image_path.parent / f"preprocessed_{image_path.name}"
                img.save(output_path, "JPEG", quality=95)
                
                return output_path
                
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return image_path
    
    @staticmethod
    def get_image_info(image_path: Path) -> Dict[str, Any]:
        """
        Get image information.
        
        Args:
            image_path: Path to image
            
        Returns:
            Image metadata
        """
        try:
            with Image.open(image_path) as img:
                return {
                    "size": img.size,
                    "format": img.format,
                    "mode": img.mode,
                    "file_size": image_path.stat().st_size,
                    "has_transparency": img.mode in ("RGBA", "LA", "P"),
                }
        except Exception as e:
            logger.error(f"Error getting image info: {e}")
            return {}




