"""Watermark utilities for documents"""
from typing import Optional, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import io
import os


class WatermarkGenerator:
    """Generate watermarks for documents"""
    
    def __init__(self):
        self.default_font_size = 60
        self.default_opacity = 0.3
    
    def create_text_watermark(
        self,
        text: str,
        width: int = 800,
        height: int = 600,
        font_size: Optional[int] = None,
        color: str = "#CCCCCC",
        opacity: float = 0.3,
        angle: int = -45
    ) -> bytes:
        """
        Create a text watermark image
        
        Args:
            text: Watermark text
            width: Image width
            height: Image height
            font_size: Font size
            color: Text color
            opacity: Opacity (0.0 to 1.0)
            angle: Rotation angle in degrees
            
        Returns:
            Watermark image as bytes
        """
        # Create transparent image
        img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        
        # Try to load font
        try:
            font = ImageFont.truetype("arial.ttf", font_size or self.default_font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 
                                         font_size or self.default_font_size)
            except:
                font = ImageFont.load_default()
        
        # Get text dimensions
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate position (center)
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        # Convert color to RGBA
        rgba = self._hex_to_rgba(color, opacity)
        
        # Draw text
        draw.text((x, y), text, fill=rgba, font=font)
        
        # Rotate if needed
        if angle != 0:
            img = img.rotate(angle, expand=True)
        
        # Convert to bytes
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return buf.read()
    
    def create_image_watermark(
        self,
        watermark_path: str,
        opacity: float = 0.3
    ) -> Optional[bytes]:
        """
        Create watermark from image file
        
        Args:
            watermark_path: Path to watermark image
            opacity: Opacity (0.0 to 1.0)
            
        Returns:
            Watermark image as bytes or None
        """
        try:
            watermark = Image.open(watermark_path)
            
            # Convert to RGBA if needed
            if watermark.mode != 'RGBA':
                watermark = watermark.convert('RGBA')
            
            # Apply opacity
            alpha = watermark.split()[3]
            alpha = alpha.point(lambda p: int(p * opacity))
            watermark.putalpha(alpha)
            
            # Convert to bytes
            buf = io.BytesIO()
            watermark.save(buf, format='PNG')
            buf.seek(0)
            return buf.read()
        except Exception as e:
            print(f"Error creating image watermark: {e}")
            return None
    
    def _hex_to_rgba(self, hex_color: str, opacity: float) -> tuple:
        """Convert hex color to RGBA tuple"""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        a = int(255 * opacity)
        return (r, g, b, a)


# Global watermark generator
_watermark_generator: Optional[WatermarkGenerator] = None


def get_watermark_generator() -> WatermarkGenerator:
    """Get global watermark generator"""
    global _watermark_generator
    if _watermark_generator is None:
        _watermark_generator = WatermarkGenerator()
    return _watermark_generator

