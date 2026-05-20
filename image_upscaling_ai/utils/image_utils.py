"""
Image Utilities
================

Utility functions for image processing.
"""

import logging
from typing import Tuple, Optional, List
from PIL import Image, ImageOps, ImageEnhance
import numpy as np

logger = logging.getLogger(__name__)


class ImageUtils:
    """Utility functions for image operations."""
    
    @staticmethod
    def ensure_rgb(image: Image.Image) -> Image.Image:
        """Ensure image is in RGB mode."""
        if image.mode != "RGB":
            return image.convert("RGB")
        return image
    
    @staticmethod
    def get_image_info(image: Image.Image) -> dict:
        """Get comprehensive image information."""
        return {
            "size": image.size,
            "mode": image.mode,
            "format": image.format,
            "width": image.width,
            "height": image.height,
            "aspect_ratio": image.width / image.height if image.height > 0 else 0,
            "total_pixels": image.width * image.height,
        }
    
    @staticmethod
    def resize_maintain_aspect(
        image: Image.Image,
        max_size: Tuple[int, int],
        resample: Image.Resampling = Image.Resampling.LANCZOS
    ) -> Image.Image:
        """Resize image maintaining aspect ratio."""
        width, height = image.size
        max_width, max_height = max_size
        
        # Calculate scaling factor
        scale = min(max_width / width, max_height / height)
        
        if scale >= 1.0:
            return image  # No need to upscale
        
        new_size = (int(width * scale), int(height * scale))
        return image.resize(new_size, resample)
    
    @staticmethod
    def pad_to_size(
        image: Image.Image,
        target_size: Tuple[int, int],
        fill_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> Image.Image:
        """Pad image to target size."""
        width, height = image.size
        target_width, target_height = target_size
        
        # Calculate padding
        pad_left = (target_width - width) // 2
        pad_top = (target_height - height) // 2
        pad_right = target_width - width - pad_left
        pad_bottom = target_height - height - pad_top
        
        return ImageOps.expand(
            image,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=fill_color
        )
    
    @staticmethod
    def crop_to_aspect(
        image: Image.Image,
        aspect_ratio: float
    ) -> Image.Image:
        """Crop image to specific aspect ratio."""
        width, height = image.size
        current_aspect = width / height
        
        if abs(current_aspect - aspect_ratio) < 0.01:
            return image  # Already correct aspect
        
        if current_aspect > aspect_ratio:
            # Image is wider, crop width
            new_width = int(height * aspect_ratio)
            left = (width - new_width) // 2
            return image.crop((left, 0, left + new_width, height))
        else:
            # Image is taller, crop height
            new_height = int(width / aspect_ratio)
            top = (height - new_height) // 2
            return image.crop((0, top, width, top + new_height))
    
    @staticmethod
    def enhance_image(
        image: Image.Image,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        sharpness: float = 1.0
    ) -> Image.Image:
        """Apply multiple enhancements to image."""
        enhanced = image
        
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(brightness)
        
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(contrast)
        
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(saturation)
        
        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(sharpness)
        
        return enhanced
    
    @staticmethod
    def compare_images(
        image1: Image.Image,
        image2: Image.Image
    ) -> dict:
        """Compare two images."""
        # Ensure same size
        if image1.size != image2.size:
            image2 = image2.resize(image1.size, Image.Resampling.LANCZOS)
        
        # Convert to arrays
        arr1 = np.array(image1.convert("RGB"))
        arr2 = np.array(image2.convert("RGB"))
        
        # Calculate differences
        diff = np.abs(arr1.astype(float) - arr2.astype(float))
        mean_diff = np.mean(diff) / 255.0
        max_diff = np.max(diff) / 255.0
        
        # Calculate similarity
        similarity = 1.0 - mean_diff
        
        return {
            "similarity": similarity,
            "mean_difference": mean_diff,
            "max_difference": max_diff,
            "size_match": image1.size == image2.size
        }
    
    @staticmethod
    def create_thumbnail(
        image: Image.Image,
        size: Tuple[int, int],
        quality: int = 85
    ) -> Image.Image:
        """Create thumbnail of image."""
        image.thumbnail(size, Image.Resampling.LANCZOS)
        return image
    
    @staticmethod
    def split_into_tiles(
        image: Image.Image,
        tile_size: Tuple[int, int],
        overlap: int = 0
    ) -> List[Tuple[Image.Image, Tuple[int, int]]]:
        """Split image into tiles."""
        width, height = image.size
        tile_width, tile_height = tile_size
        
        tiles = []
        for y in range(0, height, tile_height - overlap):
            for x in range(0, width, tile_width - overlap):
                right = min(x + tile_width, width)
                bottom = min(y + tile_height, height)
                
                tile = image.crop((x, y, right, bottom))
                tiles.append((tile, (x, y)))
        
        return tiles
    
    @staticmethod
    def combine_tiles(
        tiles: List[Tuple[Image.Image, Tuple[int, int]]],
        output_size: Tuple[int, int]
    ) -> Image.Image:
        """Combine tiles into single image."""
        result = Image.new("RGB", output_size)
        
        for tile, (x, y) in tiles:
            result.paste(tile, (x, y))
        
        return result


