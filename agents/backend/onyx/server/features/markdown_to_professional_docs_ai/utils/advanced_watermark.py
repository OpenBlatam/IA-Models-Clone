"""Advanced watermarking system"""
from typing import Dict, Any, Optional
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import logging

logger = logging.getLogger(__name__)


class AdvancedWatermarker:
    """Advanced watermarking system"""
    
    def __init__(self):
        self.watermark_types = ['text', 'image', 'logo']
    
    def add_watermark(
        self,
        document_path: str,
        watermark_config: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add watermark to document
        
        Args:
            document_path: Path to document
            watermark_config: Watermark configuration
            output_path: Optional output path
            
        Returns:
            Watermark result
        """
        try:
            path = Path(document_path)
            if not path.exists():
                raise FileNotFoundError(f"Document not found: {document_path}")
            
            extension = path.suffix.lower()
            
            if extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                return self._watermark_image(path, watermark_config, output_path)
            elif extension == '.pdf':
                return self._watermark_pdf(path, watermark_config, output_path)
            else:
                return {
                    "success": False,
                    "error": f"Watermarking not supported for format: {extension}"
                }
        except Exception as e:
            logger.error(f"Error adding watermark: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _watermark_image(
        self,
        path: Path,
        config: Dict[str, Any],
        output_path: Optional[str]
    ) -> Dict[str, Any]:
        """Add watermark to image"""
        try:
            if not output_path:
                output_path = str(path.parent / f"{path.stem}_watermarked{path.suffix}")
            
            img = Image.open(path)
            
            # Convert to RGBA if needed
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            watermark_type = config.get('type', 'text')
            
            if watermark_type == 'text':
                img = self._add_text_watermark(img, config)
            elif watermark_type == 'image':
                img = self._add_image_watermark(img, config)
            elif watermark_type == 'logo':
                img = self._add_logo_watermark(img, config)
            
            # Save
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            img.save(output_path, quality=95)
            
            return {
                "success": True,
                "output_path": output_path,
                "watermark_type": watermark_type
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _add_text_watermark(
        self,
        img: Image.Image,
        config: Dict[str, Any]
    ) -> Image.Image:
        """Add text watermark"""
        text = config.get('text', 'WATERMARK')
        position = config.get('position', 'center')  # center, top-left, top-right, bottom-left, bottom-right
        opacity = config.get('opacity', 0.5)
        font_size = config.get('font_size', 50)
        color = config.get('color', (255, 255, 255))
        angle = config.get('angle', 0)
        
        # Create watermark layer
        watermark = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)
        
        # Try to load font
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Get text dimensions
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate position
        if position == 'center':
            x = (img.width - text_width) // 2
            y = (img.height - text_height) // 2
        elif position == 'top-left':
            x = 10
            y = 10
        elif position == 'top-right':
            x = img.width - text_width - 10
            y = 10
        elif position == 'bottom-left':
            x = 10
            y = img.height - text_height - 10
        elif position == 'bottom-right':
            x = img.width - text_width - 10
            y = img.height - text_height - 10
        else:
            x = (img.width - text_width) // 2
            y = (img.height - text_height) // 2
        
        # Draw text
        draw.text((x, y), text, font=font, fill=(*color, int(255 * opacity)))
        
        # Rotate if needed
        if angle != 0:
            watermark = watermark.rotate(angle, expand=False)
        
        # Composite
        img = Image.alpha_composite(img, watermark)
        
        return img
    
    def _add_image_watermark(
        self,
        img: Image.Image,
        config: Dict[str, Any]
    ) -> Image.Image:
        """Add image watermark"""
        watermark_path = config.get('watermark_image')
        if not watermark_path:
            return img
        
        try:
            watermark = Image.open(watermark_path)
            if watermark.mode != 'RGBA':
                watermark = watermark.convert('RGBA')
            
            # Resize watermark
            max_size = config.get('max_size', 200)
            watermark.thumbnail((max_size, max_size))
            
            # Position
            position = config.get('position', 'bottom-right')
            if position == 'bottom-right':
                x = img.width - watermark.width - 10
                y = img.height - watermark.height - 10
            else:
                x = (img.width - watermark.width) // 2
                y = (img.height - watermark.height) // 2
            
            # Opacity
            opacity = config.get('opacity', 0.5)
            alpha = watermark.split()[3]
            alpha = alpha.point(lambda p: int(p * opacity))
            watermark.putalpha(alpha)
            
            # Paste
            img.paste(watermark, (x, y), watermark)
            
            return img
        except Exception as e:
            logger.error(f"Error adding image watermark: {e}")
            return img
    
    def _add_logo_watermark(
        self,
        img: Image.Image,
        config: Dict[str, Any]
    ) -> Image.Image:
        """Add logo watermark (similar to image)"""
        return self._add_image_watermark(img, config)
    
    def _watermark_pdf(
        self,
        path: Path,
        config: Dict[str, Any],
        output_path: Optional[str]
    ) -> Dict[str, Any]:
        """Add watermark to PDF"""
        # Placeholder - would use PyPDF2 or reportlab
        return {
            "success": False,
            "error": "PDF watermarking not yet implemented"
        }


# Global watermarker
_watermarker: Optional[AdvancedWatermarker] = None


def get_watermarker() -> AdvancedWatermarker:
    """Get global watermarker"""
    global _watermarker
    if _watermarker is None:
        _watermarker = AdvancedWatermarker()
    return _watermarker

