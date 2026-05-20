"""
Image Processor for Color Grading AI
====================================

Handles image processing and color grading operations.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import asyncio
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Image processor for color grading operations.
    
    Features:
    - Load and save images
    - Apply color transformations
    - Extract color information
    - Apply LUTs
    - Color space conversions
    """
    
    def __init__(self):
        """Initialize image processor."""
        pass
    
    async def load_image(self, image_path: str) -> Image.Image:
        """
        Load image from file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image object
        """
        def _load():
            return Image.open(image_path)
        
        return await asyncio.to_thread(_load)
    
    async def save_image(
        self,
        image: Image.Image,
        output_path: str,
        quality: int = 95
    ) -> str:
        """
        Save image to file.
        
        Args:
            image: PIL Image object
            output_path: Output file path
            quality: JPEG quality (1-100)
            
        Returns:
            Path to saved image
        """
        def _save():
            image.save(output_path, quality=quality)
            return output_path
        
        return await asyncio.to_thread(_save)
    
    async def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """
        Get image information.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with image metadata
        """
        image = await self.load_image(image_path)
        
        return {
            "width": image.width,
            "height": image.height,
            "format": image.format,
            "mode": image.mode,
            "size": image.size,
        }
    
    async def apply_color_grading(
        self,
        image_path: str,
        output_path: str,
        color_params: Dict[str, Any]
    ) -> str:
        """
        Apply color grading to image.
        
        Args:
            image_path: Input image path
            output_path: Output image path
            color_params: Color grading parameters
            
        Returns:
            Path to processed image
        """
        image = await self.load_image(image_path)
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Apply color transformations
        if "brightness" in color_params:
            img_array = self._adjust_brightness(img_array, color_params["brightness"])
        
        if "contrast" in color_params:
            img_array = self._adjust_contrast(img_array, color_params["contrast"])
        
        if "saturation" in color_params:
            img_array = self._adjust_saturation(img_array, color_params["saturation"])
        
        if "color_balance" in color_params:
            img_array = self._adjust_color_balance(img_array, color_params["color_balance"])
        
        if "curves" in color_params:
            img_array = self._apply_curves(img_array, color_params["curves"])
        
        # Convert back to PIL Image
        result_image = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
        
        # Save result
        await self.save_image(result_image, output_path)
        
        return output_path
    
    def _adjust_brightness(self, img_array: np.ndarray, value: float) -> np.ndarray:
        """Adjust image brightness."""
        return np.clip(img_array + value * 255, 0, 255)
    
    def _adjust_contrast(self, img_array: np.ndarray, value: float) -> np.ndarray:
        """Adjust image contrast."""
        mean = img_array.mean()
        return np.clip((img_array - mean) * value + mean, 0, 255)
    
    def _adjust_saturation(self, img_array: np.ndarray, value: float) -> np.ndarray:
        """Adjust image saturation."""
        # Convert to HSV
        hsv = self._rgb_to_hsv(img_array)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * value, 0, 1)
        return self._hsv_to_rgb(hsv)
    
    def _adjust_color_balance(
        self,
        img_array: np.ndarray,
        balance: Dict[str, float]
    ) -> np.ndarray:
        """Adjust color balance."""
        r_adj = balance.get("r", 0.0)
        g_adj = balance.get("g", 0.0)
        b_adj = balance.get("b", 0.0)
        
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] + r_adj * 255, 0, 255)
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] + g_adj * 255, 0, 255)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] + b_adj * 255, 0, 255)
        
        return img_array
    
    def _apply_curves(self, img_array: np.ndarray, curves: Dict[str, Any]) -> np.ndarray:
        """Apply color curves (simplified implementation)."""
        # This is a simplified version - full implementation would use proper curve interpolation
        result = img_array.copy()
        
        if "rgb" in curves:
            # Apply RGB curve
            curve = curves["rgb"]
            for i in range(3):  # R, G, B channels
                result[:, :, i] = np.interp(
                    result[:, :, i],
                    np.linspace(0, 255, len(curve)),
                    np.array(curve) * 255
                )
        
        return np.clip(result, 0, 255)
    
    def _rgb_to_hsv(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to HSV."""
        rgb_norm = rgb / 255.0
        hsv = np.zeros_like(rgb_norm)
        
        max_val = rgb_norm.max(axis=2)
        min_val = rgb_norm.min(axis=2)
        delta = max_val - min_val
        
        # Hue
        hsv[:, :, 0] = np.where(
            delta == 0, 0,
            np.where(
                max_val == rgb_norm[:, :, 0],
                ((rgb_norm[:, :, 1] - rgb_norm[:, :, 2]) / delta) % 6,
                np.where(
                    max_val == rgb_norm[:, :, 1],
                    (rgb_norm[:, :, 2] - rgb_norm[:, :, 0]) / delta + 2,
                    (rgb_norm[:, :, 0] - rgb_norm[:, :, 1]) / delta + 4
                )
            )
        ) / 6.0
        
        # Saturation
        hsv[:, :, 1] = np.where(max_val == 0, 0, delta / max_val)
        
        # Value
        hsv[:, :, 2] = max_val
        
        return hsv
    
    def _hsv_to_rgb(self, hsv: np.ndarray) -> np.ndarray:
        """Convert HSV to RGB."""
        h = hsv[:, :, 0] * 6
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        
        c = v * s
        x = c * (1 - np.abs((h % 2) - 1))
        m = v - c
        
        rgb = np.zeros_like(hsv)
        
        conditions = [
            (h >= 0) & (h < 1),
            (h >= 1) & (h < 2),
            (h >= 2) & (h < 3),
            (h >= 3) & (h < 4),
            (h >= 4) & (h < 5),
            (h >= 5) & (h < 6),
        ]
        
        choices = [
            [c, x, 0],
            [x, c, 0],
            [0, c, x],
            [0, x, c],
            [x, 0, c],
            [c, 0, x],
        ]
        
        for i, (cond, choice) in enumerate(zip(conditions, choices)):
            rgb[:, :, i % 3] = np.where(cond, choice[0] + m, rgb[:, :, i % 3])
            rgb[:, :, (i + 1) % 3] = np.where(cond, choice[1] + m, rgb[:, :, (i + 1) % 3])
            rgb[:, :, (i + 2) % 3] = np.where(cond, choice[2] + m, rgb[:, :, (i + 2) % 3])
        
        return (rgb * 255).astype(np.uint8)
    
    async def extract_dominant_colors(
        self,
        image_path: str,
        num_colors: int = 5
    ) -> List[Tuple[int, int, int]]:
        """
        Extract dominant colors from image.
        
        Args:
            image_path: Path to image file
            num_colors: Number of dominant colors to extract
            
        Returns:
            List of RGB color tuples
        """
        image = await self.load_image(image_path)
        
        # Resize for faster processing
        image.thumbnail((200, 200))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Reshape to 2D array
        pixels = img_array.reshape(-1, 3)
        
        # Use simple color quantization (K-means requires sklearn)
        # For production, would use proper clustering
        # Simplified: sample evenly distributed colors
        step = len(pixels) // num_colors
        sample_indices = [i * step for i in range(num_colors)]
        colors = pixels[sample_indices].astype(int)
        
        # Ensure we have exactly num_colors
        if len(colors) < num_colors:
            # Pad with last color
            last_color = colors[-1] if len(colors) > 0 else [128, 128, 128]
            while len(colors) < num_colors:
                colors = np.vstack([colors, last_color])
        
        return [tuple(color) for color in colors[:num_colors]]

