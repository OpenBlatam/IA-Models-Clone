"""
Compression Mixin

Contains image compression and optimization functionality.
"""

import logging
from typing import Union, Dict, Any, Optional
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


class CompressionMixin:
    """
    Mixin providing compression and optimization functionality.
    
    This mixin contains:
    - Image compression
    - Size optimization
    - Format conversion
    - Quality optimization
    - Compression analysis
    """
    
    def compress_image(
        self,
        image: Image.Image,
        quality: int = 85,
        optimize: bool = True,
        format: str = "JPEG"
    ) -> Image.Image:
        """
        Compress image while maintaining quality.
        
        Args:
            image: Input image
            quality: Compression quality (1-100)
            optimize: Optimize compression
            format: Output format (JPEG, PNG, WEBP)
            
        Returns:
            Compressed image
        """
        # Convert to RGB if needed for JPEG
        if format.upper() == "JPEG" and image.mode in ["RGBA", "LA", "P"]:
            # Create white background for transparency
            rgb_image = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode == "RGBA":
                rgb_image.paste(image, mask=image.split()[3])
            else:
                rgb_image.paste(image)
            image = rgb_image
        
        return image
    
    def optimize_image_size(
        self,
        image: Image.Image,
        target_size_kb: Optional[float] = None,
        max_dimension: Optional[int] = None,
        quality: int = 85
    ) -> Dict[str, Any]:
        """
        Optimize image size.
        
        Args:
            image: Input image
            target_size_kb: Target size in KB (None = no target)
            max_dimension: Maximum dimension (None = no limit)
            quality: Compression quality
            
        Returns:
            Dictionary with optimization results
        """
        from io import BytesIO
        
        original_size = image.size
        original_bytes = len(image.tobytes())
        
        # Resize if needed
        if max_dimension:
            width, height = image.size
            if width > max_dimension or height > max_dimension:
                if width > height:
                    ratio = max_dimension / width
                else:
                    ratio = max_dimension / height
                new_size = (int(width * ratio), int(height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Compress
        compressed = self.compress_image(image, quality=quality, format="JPEG")
        
        # Check size
        buffer = BytesIO()
        compressed.save(buffer, format="JPEG", quality=quality, optimize=True)
        compressed_bytes = len(buffer.getvalue())
        compressed_size_kb = compressed_bytes / 1024
        
        # If target size specified and not met, adjust quality
        if target_size_kb and compressed_size_kb > target_size_kb:
            for q in range(quality, 10, -5):
                buffer = BytesIO()
                compressed.save(buffer, format="JPEG", quality=q, optimize=True)
                test_size = len(buffer.getvalue()) / 1024
                if test_size <= target_size_kb:
                    compressed = self.compress_image(image, quality=q, format="JPEG")
                    compressed_bytes = len(buffer.getvalue())
                    compressed_size_kb = test_size
                    quality = q
                    break
        
        compression_ratio = (1 - compressed_bytes / original_bytes) * 100 if original_bytes > 0 else 0
        
        return {
            "original_size": original_size,
            "compressed_size": compressed.size,
            "original_bytes": original_bytes,
            "compressed_bytes": compressed_bytes,
            "compressed_size_kb": compressed_size_kb,
            "compression_ratio": compression_ratio,
            "quality_used": quality,
            "compressed_image": compressed,
        }
    
    def analyze_compression(
        self,
        original: Image.Image,
        compressed: Image.Image
    ) -> Dict[str, Any]:
        """
        Analyze compression results.
        
        Args:
            original: Original image
            compressed: Compressed image
            
        Returns:
            Dictionary with analysis results
        """
        from io import BytesIO
        from ..helpers import QualityCalculator
        
        # Calculate sizes
        orig_buffer = BytesIO()
        original.save(orig_buffer, format="PNG")
        orig_bytes = len(orig_buffer.getvalue())
        
        comp_buffer = BytesIO()
        compressed.save(comp_buffer, format="JPEG", quality=85)
        comp_bytes = len(comp_buffer.getvalue())
        
        compression_ratio = (1 - comp_bytes / orig_bytes) * 100 if orig_bytes > 0 else 0
        
        # Calculate quality
        orig_quality = QualityCalculator.calculate_quality_metrics(original)
        comp_quality = QualityCalculator.calculate_quality_metrics(compressed)
        quality_loss = orig_quality.overall_quality - comp_quality.overall_quality
        
        return {
            "original_size_bytes": orig_bytes,
            "compressed_size_bytes": comp_bytes,
            "compression_ratio": compression_ratio,
            "size_reduction_kb": (orig_bytes - comp_bytes) / 1024,
            "original_quality": orig_quality.overall_quality,
            "compressed_quality": comp_quality.overall_quality,
            "quality_loss": quality_loss,
            "efficiency": compression_ratio / abs(quality_loss) if quality_loss != 0 else compression_ratio,
        }
    
    def smart_compress(
        self,
        image: Image.Image,
        target_quality: float = 0.8,
        max_size_kb: Optional[float] = None
    ) -> Image.Image:
        """
        Smart compression maintaining target quality.
        
        Args:
            image: Input image
            target_quality: Target quality level (0.0-1.0)
            max_size_kb: Maximum size in KB
            
        Returns:
            Compressed image
        """
        from ..helpers import QualityCalculator
        
        # Try different quality levels
        for quality in range(95, 40, -5):
            compressed = self.compress_image(image, quality=quality, format="JPEG")
            comp_quality = QualityCalculator.calculate_quality_metrics(compressed)
            
            if comp_quality.overall_quality >= target_quality:
                # Check size if specified
                if max_size_kb:
                    from io import BytesIO
                    buffer = BytesIO()
                    compressed.save(buffer, format="JPEG", quality=quality)
                    size_kb = len(buffer.getvalue()) / 1024
                    if size_kb <= max_size_kb:
                        return compressed
                else:
                    return compressed
        
        # Return best available
        return self.compress_image(image, quality=40, format="JPEG")


