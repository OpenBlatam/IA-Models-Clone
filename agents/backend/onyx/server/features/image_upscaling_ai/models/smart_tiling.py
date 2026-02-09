"""
Smart Tiling System
==================

Intelligent tiling system for processing large images efficiently.
"""

import logging
import numpy as np
from typing import Tuple, List, Optional, Callable
from PIL import Image
import math

logger = logging.getLogger(__name__)


class SmartTiling:
    """
    Smart tiling system for large image processing.
    
    Features:
    - Automatic tile size calculation
    - Overlap handling for seamless results
    - Memory-efficient processing
    - GPU/CPU optimization
    - Quality preservation at tile boundaries
    """
    
    def __init__(
        self,
        max_tile_size: int = 512,
        overlap: int = 32,
        min_tile_size: int = 256,
        memory_limit_mb: Optional[int] = None,
    ):
        """
        Initialize smart tiling system.
        
        Args:
            max_tile_size: Maximum tile size (will auto-calculate if memory_limit_mb is set)
            overlap: Overlap between tiles in pixels
            min_tile_size: Minimum tile size
            memory_limit_mb: Memory limit in MB (auto-calculates tile size)
        """
        self.max_tile_size = max_tile_size
        self.overlap = overlap
        self.min_tile_size = min_tile_size
        self.memory_limit_mb = memory_limit_mb
        
        if memory_limit_mb:
            # Calculate optimal tile size based on memory
            # Estimate: 4 bytes per pixel * scale_factor^2 * channels
            # For 4x upscaling: 4 * 16 * 3 = 192 bytes per pixel
            # For 512x512 tile at 4x: 512^2 * 192 / 1024^2 ≈ 50 MB
            self.max_tile_size = min(
                max_tile_size,
                int(math.sqrt(memory_limit_mb * 1024 * 1024 / (192 * 4)))  # Conservative estimate
            )
            logger.info(f"Auto-calculated tile size: {self.max_tile_size} (memory limit: {memory_limit_mb}MB)")
    
    def calculate_tiles(
        self,
        image_size: Tuple[int, int],
        scale_factor: float
    ) -> List[Tuple[int, int, int, int]]:
        """
        Calculate tile positions for image.
        
        Args:
            image_size: (width, height) of original image
            scale_factor: Scale factor
            
        Returns:
            List of (x, y, width, height) tile coordinates
        """
        width, height = image_size
        
        # Calculate effective tile size (accounting for overlap)
        effective_tile = self.max_tile_size - 2 * self.overlap
        
        # Calculate number of tiles
        tiles_x = math.ceil(width / effective_tile)
        tiles_y = math.ceil(height / effective_tile)
        
        tiles = []
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                # Calculate tile position
                x = tx * effective_tile
                y = ty * effective_tile
                
                # Calculate tile size (with overlap)
                tile_width = min(self.max_tile_size, width - x + self.overlap)
                tile_height = min(self.max_tile_size, height - y + self.overlap)
                
                # Adjust for boundaries
                if x + tile_width > width:
                    tile_width = width - x
                if y + tile_height > height:
                    tile_height = height - y
                
                tiles.append((x, y, tile_width, tile_height))
        
        logger.debug(f"Calculated {len(tiles)} tiles for {image_size} image")
        return tiles
    
    def process_tiled(
        self,
        image: Image.Image,
        scale_factor: float,
        upscale_func: Callable[[Image.Image, float], Image.Image],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Image.Image:
        """
        Process image in tiles and combine results.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            upscale_func: Function to upscale a tile
            progress_callback: Progress callback (current, total)
            
        Returns:
            Upscaled image
        """
        width, height = image.size
        
        # Calculate tiles
        tiles = self.calculate_tiles((width, height), scale_factor)
        
        if len(tiles) == 1:
            # Single tile, no need for tiling
            logger.debug("Image fits in single tile, processing without tiling")
            return upscale_func(image, scale_factor)
        
        logger.info(f"Processing {len(tiles)} tiles for {width}x{height} image")
        
        # Process each tile
        upscaled_tiles = []
        for idx, (x, y, tile_w, tile_h) in enumerate(tiles):
            if progress_callback:
                progress_callback(idx + 1, len(tiles))
            
            # Extract tile
            tile = image.crop((x, y, x + tile_w, y + tile_h))
            
            # Upscale tile
            upscaled_tile = upscale_func(tile, scale_factor)
            
            # Calculate position in final image
            upscaled_x = int(x * scale_factor)
            upscaled_y = int(y * scale_factor)
            upscaled_w, upscaled_h = upscaled_tile.size
            
            # Remove overlap from edges
            if idx > 0:  # Not first tile
                # Remove left overlap
                overlap_x = int(self.overlap * scale_factor)
                upscaled_tile = upscaled_tile.crop((overlap_x, 0, upscaled_w, upscaled_h))
                upscaled_x += overlap_x
                upscaled_w -= overlap_x
            
            if idx >= len(tiles) - (len(tiles) // math.ceil(width / (self.max_tile_size - 2 * self.overlap))):  # Last row
                # Remove bottom overlap
                overlap_y = int(self.overlap * scale_factor)
                upscaled_tile = upscaled_tile.crop((0, 0, upscaled_w, upscaled_h - overlap_y))
                upscaled_h -= overlap_y
            
            upscaled_tiles.append({
                "image": upscaled_tile,
                "x": upscaled_x,
                "y": upscaled_y,
                "width": upscaled_w,
                "height": upscaled_h
            })
        
        # Combine tiles
        final_width = int(width * scale_factor)
        final_height = int(height * scale_factor)
        final_image = Image.new("RGB", (final_width, final_height))
        
        for tile_info in upscaled_tiles:
            final_image.paste(
                tile_info["image"],
                (tile_info["x"], tile_info["y"])
            )
        
        logger.info(f"Combined {len(tiles)} tiles into {final_width}x{final_height} image")
        return final_image
    
    def should_tile(
        self,
        image_size: Tuple[int, int],
        scale_factor: float
    ) -> bool:
        """
        Determine if image should be tiled.
        
        Args:
            image_size: (width, height) of image
            scale_factor: Scale factor
            
        Returns:
            True if tiling is recommended
        """
        width, height = image_size
        
        # Calculate output size
        out_width = int(width * scale_factor)
        out_height = int(height * scale_factor)
        
        # Estimate memory usage (rough estimate)
        # 4 bytes per pixel * channels
        estimated_memory_mb = (out_width * out_height * 3 * 4) / (1024 * 1024)
        
        # Tile if image is large or memory limit is set
        if self.memory_limit_mb and estimated_memory_mb > self.memory_limit_mb * 0.8:
            return True
        
        # Tile if image is very large
        if width > self.max_tile_size * 2 or height > self.max_tile_size * 2:
            return True
        
        return False
    
    @staticmethod
    def auto_tile_size(
        image_size: Tuple[int, int],
        scale_factor: float,
        available_memory_mb: int = 2048
    ) -> int:
        """
        Automatically calculate optimal tile size.
        
        Args:
            image_size: (width, height) of image
            scale_factor: Scale factor
            available_memory_mb: Available memory in MB
            
        Returns:
            Optimal tile size
        """
        width, height = image_size
        out_width = int(width * scale_factor)
        out_height = int(height * scale_factor)
        
        # Estimate memory per tile (conservative)
        # 4 bytes per pixel * 3 channels * scale_factor^2
        bytes_per_pixel = 4 * 3 * (scale_factor ** 2)
        
        # Use 50% of available memory for safety
        safe_memory_bytes = (available_memory_mb * 1024 * 1024) * 0.5
        
        # Calculate max tile size
        max_pixels_per_tile = safe_memory_bytes / bytes_per_pixel
        max_tile_size = int(math.sqrt(max_pixels_per_tile))
        
        # Clamp to reasonable values
        max_tile_size = max(256, min(1024, max_tile_size))
        
        return max_tile_size


