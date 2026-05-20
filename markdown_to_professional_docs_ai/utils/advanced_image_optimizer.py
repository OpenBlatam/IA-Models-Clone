"""Advanced image optimization with intelligent compression"""
from typing import Dict, Any, Optional, List
from pathlib import Path
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)


class AdvancedImageOptimizer:
    """Advanced image optimization with intelligent compression"""
    
    def __init__(self):
        self.supported_formats = ['JPEG', 'PNG', 'WEBP', 'AVIF']
        self.quality_levels = {
            'low': 60,
            'medium': 80,
            'high': 95,
            'lossless': 100
        }
    
    def optimize_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        quality: str = 'medium',
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
        format: Optional[str] = None,
        progressive: bool = True
    ) -> Dict[str, Any]:
        """
        Optimize image
        
        Args:
            image_path: Path to input image
            output_path: Optional output path
            quality: Quality level (low, medium, high, lossless)
            max_width: Optional maximum width
            max_height: Optional maximum height
            format: Optional output format
            progressive: Use progressive encoding for JPEG
            
        Returns:
            Optimization result
        """
        try:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Open image
            img = Image.open(path)
            original_size = path.stat().st_size
            original_format = img.format
            original_dimensions = img.size
            
            # Resize if needed
            if max_width or max_height:
                img = self._resize_image(img, max_width, max_height)
            
            # Determine output format
            output_format = format or original_format or 'JPEG'
            
            # Determine output path
            if not output_path:
                output_path = str(path.parent / f"{path.stem}_optimized.{output_format.lower()}")
            
            # Optimize based on format
            if output_format.upper() == 'JPEG':
                img = img.convert('RGB')
                img.save(
                    output_path,
                    'JPEG',
                    quality=self.quality_levels.get(quality, 80),
                    optimize=True,
                    progressive=progressive
                )
            elif output_format.upper() == 'PNG':
                img.save(
                    output_path,
                    'PNG',
                    optimize=True,
                    compress_level=9
                )
            elif output_format.upper() == 'WEBP':
                img.save(
                    output_path,
                    'WEBP',
                    quality=self.quality_levels.get(quality, 80),
                    method=6
                )
            else:
                img.save(output_path, output_format)
            
            # Get optimized size
            optimized_size = Path(output_path).stat().st_size
            compression_ratio = (1 - (optimized_size / original_size)) * 100
            
            return {
                "success": True,
                "original_path": str(path),
                "optimized_path": output_path,
                "original_size": original_size,
                "optimized_size": optimized_size,
                "compression_ratio": round(compression_ratio, 2),
                "original_dimensions": original_dimensions,
                "optimized_dimensions": img.size,
                "format": output_format
            }
        except Exception as e:
            logger.error(f"Error optimizing image: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _resize_image(
        self,
        img: Image.Image,
        max_width: Optional[int],
        max_height: Optional[int]
    ) -> Image.Image:
        """Resize image maintaining aspect ratio"""
        width, height = img.size
        
        if max_width and width > max_width:
            ratio = max_width / width
            height = int(height * ratio)
            width = max_width
        
        if max_height and height > max_height:
            ratio = max_height / height
            width = int(width * ratio)
            height = max_height
        
        return img.resize((width, height), Image.Resampling.LANCZOS)
    
    def batch_optimize(
        self,
        image_paths: List[str],
        quality: str = 'medium',
        max_width: Optional[int] = None,
        max_height: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Batch optimize images
        
        Args:
            image_paths: List of image paths
            quality: Quality level
            max_width: Optional maximum width
            max_height: Optional maximum height
            
        Returns:
            List of optimization results
        """
        results = []
        
        for image_path in image_paths:
            result = self.optimize_image(
                image_path,
                quality=quality,
                max_width=max_width,
                max_height=max_height
            )
            results.append(result)
        
        return results
    
    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """
        Get image information
        
        Args:
            image_path: Path to image
            
        Returns:
            Image information
        """
        try:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            img = Image.open(path)
            
            return {
                "path": str(path),
                "format": img.format,
                "mode": img.mode,
                "size": img.size,
                "width": img.width,
                "height": img.height,
                "file_size": path.stat().st_size,
                "has_transparency": img.mode in ('RGBA', 'LA', 'P')
            }
        except Exception as e:
            logger.error(f"Error getting image info: {e}")
            return {
                "error": str(e)
            }


# Global optimizer
_image_optimizer: Optional[AdvancedImageOptimizer] = None


def get_image_optimizer() -> AdvancedImageOptimizer:
    """Get global image optimizer"""
    global _image_optimizer
    if _image_optimizer is None:
        _image_optimizer = AdvancedImageOptimizer()
    return _image_optimizer

