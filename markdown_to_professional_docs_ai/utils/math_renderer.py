"""Math Renderer - Render LaTeX/MathJax formulas"""
import re
from typing import List, Dict, Any, Optional
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib import mathtext
import logging

logger = logging.getLogger(__name__)


class MathRenderer:
    """Render mathematical formulas to images"""
    
    def __init__(self):
        self.font_size = 14
        self.dpi = 150
    
    def extract_math_blocks(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract math blocks from content
        
        Args:
            content: Markdown content
            
        Returns:
            List of math blocks
        """
        math_blocks = []
        
        # Inline math: $...$
        inline_pattern = r'\$([^$]+)\$'
        for match in re.finditer(inline_pattern, content):
            math_blocks.append({
                "type": "inline",
                "formula": match.group(1),
                "position": match.start()
            })
        
        # Block math: $$...$$
        block_pattern = r'\$\$([^$]+)\$\$'
        for match in re.finditer(block_pattern, content):
            math_blocks.append({
                "type": "block",
                "formula": match.group(1),
                "position": match.start()
            })
        
        # LaTeX environments
        latex_pattern = r'\\begin\{(\w+)\}(.*?)\\end\{\1\}'
        for match in re.finditer(latex_pattern, content, re.DOTALL):
            math_blocks.append({
                "type": "environment",
                "environment": match.group(1),
                "formula": match.group(2),
                "position": match.start()
            })
        
        return math_blocks
    
    def render_formula(self, formula: str, inline: bool = True) -> Optional[bytes]:
        """
        Render LaTeX formula to image
        
        Args:
            formula: LaTeX formula
            inline: Whether formula is inline
            
        Returns:
            Image bytes or None
        """
        try:
            # Create figure
            fig = plt.figure(figsize=(10, 2) if inline else (10, 4))
            ax = fig.add_subplot(111)
            ax.axis('off')
            
            # Render formula
            parser = mathtext.MathTextParser("Bitmap")
            try:
                parser.to_png(
                    f"output.png",
                    formula,
                    color='black',
                    dpi=self.dpi
                )
            except Exception as e:
                logger.warning(f"Could not render formula with mathtext: {e}")
                # Fallback: simple text rendering
                ax.text(0.5, 0.5, f"${formula}$", 
                       ha='center', va='center',
                       fontsize=self.font_size,
                       transform=ax.transAxes)
            
            # Save to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight', 
                       transparent=True, pad_inches=0.1)
            buf.seek(0)
            image_bytes = buf.read()
            buf.close()
            plt.close(fig)
            
            return image_bytes
        except Exception as e:
            logger.error(f"Error rendering formula: {e}")
            return None
    
    def formula_to_base64(self, formula: str, inline: bool = True) -> Optional[str]:
        """
        Convert formula to base64 string
        
        Args:
            formula: LaTeX formula
            inline: Whether formula is inline
            
        Returns:
            Base64 string or None
        """
        image_bytes = self.render_formula(formula, inline)
        if image_bytes:
            return base64.b64encode(image_bytes).decode('utf-8')
        return None
    
    def create_mathjax_html(self, formula: str, inline: bool = True) -> str:
        """
        Create MathJax HTML for formula
        
        Args:
            formula: LaTeX formula
            inline: Whether formula is inline
            
        Returns:
            HTML string with MathJax
        """
        if inline:
            return f'<span class="math">\\({formula}\\)</span>'
        else:
            return f'<div class="math">\\[{formula}\\]</div>'


# Global math renderer
_math_renderer: Optional[MathRenderer] = None


def get_math_renderer() -> MathRenderer:
    """Get global math renderer"""
    global _math_renderer
    if _math_renderer is None:
        _math_renderer = MathRenderer()
    return _math_renderer

