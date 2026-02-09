"""
Resolution Handler for Flux2 Clothing Changer
==============================================

Handles different image resolutions and aspect ratios for optimal results.
"""

import torch
from typing import Tuple, Optional, Union
from PIL import Image
import numpy as np
import logging
import math

logger = logging.getLogger(__name__)


class ResolutionHandler:
    """Handles image resolution and aspect ratio processing."""
    
    # Common resolutions for Flux2
    SUPPORTED_RESOLUTIONS = [
        (512, 512),
        (768, 768),
        (1024, 1024),
        (512, 768),
        (768, 512),
        (1024, 768),
        (768, 1024),
        (1024, 1280),
        (1280, 1024),
    ]
    
    # Optimal resolution for clothing changes
    OPTIMAL_RESOLUTION = (1024, 1024)
    
    def __init__(
        self,
        target_resolution: Optional[Tuple[int, int]] = None,
        maintain_aspect_ratio: bool = True,
        padding_mode: str = "edge",
    ):
        """
        Initialize resolution handler.
        
        Args:
            target_resolution: Target resolution (width, height). None for auto.
            maintain_aspect_ratio: Whether to maintain aspect ratio
            padding_mode: Padding mode for resizing ('edge', 'constant', 'reflect')
        """
        self.target_resolution = target_resolution or self.OPTIMAL_RESOLUTION
        self.maintain_aspect_ratio = maintain_aspect_ratio
        self.padding_mode = padding_mode
    
    def prepare_image(
        self,
        image: Image.Image,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Prepare image for processing with optimal resolution.
        
        Args:
            image: Input image
            target_size: Optional target size (width, height)
            
        Returns:
            Tuple of (processed_image, metadata)
        """
        target_size = target_size or self.target_resolution
        original_size = image.size
        original_aspect = original_size[0] / original_size[1]
        target_aspect = target_size[0] / target_size[1]
        
        metadata = {
            "original_size": original_size,
            "target_size": target_size,
            "resized": False,
            "padded": False,
        }
        
        if self.maintain_aspect_ratio and abs(original_aspect - target_aspect) > 0.01:
            # Resize maintaining aspect ratio
            if original_aspect > target_aspect:
                # Image is wider
                new_width = target_size[0]
                new_height = int(target_size[0] / original_aspect)
            else:
                # Image is taller
                new_height = target_size[1]
                new_width = int(target_size[1] * original_aspect)
            
            # Resize
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            metadata["resized"] = True
            metadata["resized_size"] = (new_width, new_height)
            
            # Pad to target size
            if new_width != target_size[0] or new_height != target_size[1]:
                image = self._pad_image(image, target_size)
                metadata["padded"] = True
        else:
            # Direct resize to target
            if original_size != target_size:
                image = image.resize(target_size, Image.Resampling.LANCZOS)
                metadata["resized"] = True
        
        return image, metadata
    
    def _pad_image(
        self,
        image: Image.Image,
        target_size: Tuple[int, int],
    ) -> Image.Image:
        """
        Pad image to target size.
        
        Args:
            image: Image to pad
            target_size: Target size (width, height)
            
        Returns:
            Padded image
        """
        width, height = image.size
        target_width, target_height = target_size
        
        # Calculate padding
        pad_left = (target_width - width) // 2
        pad_right = target_width - width - pad_left
        pad_top = (target_height - height) // 2
        pad_bottom = target_height - height - pad_top
        
        # Create padded image
        if self.padding_mode == "edge":
            # Use edge padding
            padded = Image.new(image.mode, target_size)
            padded.paste(image, (pad_left, pad_top))
            # Extend edges
            if pad_left > 0:
                left_edge = image.crop((0, 0, 1, height))
                for x in range(pad_left):
                    padded.paste(left_edge, (x, pad_top))
            if pad_right > 0:
                right_edge = image.crop((width - 1, 0, width, height))
                for x in range(target_width - pad_right, target_width):
                    padded.paste(right_edge, (x, pad_top))
            if pad_top > 0:
                top_edge = image.crop((0, 0, width, 1))
                for y in range(pad_top):
                    padded.paste(top_edge, (pad_left, y))
            if pad_bottom > 0:
                bottom_edge = image.crop((0, height - 1, width, height))
                for y in range(target_height - pad_bottom, target_height):
                    padded.paste(bottom_edge, (pad_left, y))
        else:
            # Use constant or reflect padding
            import numpy as np
            img_array = np.array(image)
            if self.padding_mode == "constant":
                padded_array = np.pad(
                    img_array,
                    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
            else:  # reflect
                padded_array = np.pad(
                    img_array,
                    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode="reflect",
                )
            padded = Image.fromarray(padded_array)
        
        return padded
    
    def restore_image(
        self,
        processed_image: Image.Image,
        metadata: Dict[str, Any],
    ) -> Image.Image:
        """
        Restore image to original size if it was resized/padded.
        
        Args:
            processed_image: Processed image
            metadata: Metadata from prepare_image
            
        Returns:
            Restored image
        """
        if not metadata.get("resized", False):
            return processed_image
        
        original_size = metadata["original_size"]
        
        if metadata.get("padded", False):
            # Crop padding first
            resized_size = metadata.get("resized_size", processed_image.size)
            pad_left = (processed_image.size[0] - resized_size[0]) // 2
            pad_top = (processed_image.size[1] - resized_size[1]) // 2
            processed_image = processed_image.crop(
                (pad_left, pad_top, pad_left + resized_size[0], pad_top + resized_size[1])
            )
        
        # Resize to original
        if processed_image.size != original_size:
            processed_image = processed_image.resize(
                original_size,
                Image.Resampling.LANCZOS
            )
        
        return processed_image
    
    def get_optimal_resolution(
        self,
        image_size: Tuple[int, int],
    ) -> Tuple[int, int]:
        """
        Get optimal resolution for given image size.
        
        Args:
            image_size: Current image size (width, height)
            
        Returns:
            Optimal resolution (width, height)
        """
        width, height = image_size
        aspect = width / height
        
        # Find closest supported resolution
        best_res = self.OPTIMAL_RESOLUTION
        best_diff = float('inf')
        
        for res_w, res_h in self.SUPPORTED_RESOLUTIONS:
            res_aspect = res_w / res_h
            diff = abs(aspect - res_aspect)
            if diff < best_diff:
                best_diff = diff
                best_res = (res_w, res_h)
        
        return best_res
    
    def calculate_tile_size(
        self,
        image_size: Tuple[int, int],
        max_tile_size: int = 1024,
        overlap: int = 128,
    ) -> Tuple[Tuple[int, int], int, int]:
        """
        Calculate tile size for processing large images.
        
        Args:
            image_size: Image size (width, height)
            max_tile_size: Maximum tile size
            overlap: Overlap between tiles
            
        Returns:
            Tuple of (tile_size, num_tiles_x, num_tiles_y)
        """
        width, height = image_size
        
        # Calculate number of tiles needed
        num_tiles_x = math.ceil(width / (max_tile_size - overlap))
        num_tiles_y = math.ceil(height / (max_tile_size - overlap))
        
        # Calculate actual tile size
        tile_width = math.ceil(width / num_tiles_x) + overlap
        tile_height = math.ceil(height / num_tiles_y) + overlap
        
        return (tile_width, tile_height), num_tiles_x, num_tiles_y


