"""
DeepSeek Image Generator
=========================

Uses external image generation APIs with DeepSeek prompt enhancement.
Supports multiple image generation backends.
"""

import logging
import requests
from typing import Optional, Dict, Any
from PIL import Image
import io
import base64

logger = logging.getLogger(__name__)


class DeepSeekImageGenerator:
    """
    Image generator using DeepSeek for prompts and external APIs for generation.
    """
    
    def __init__(self, api_key: str = "sk-753365753f074509bb52496e038691f6"):
        """Initialize generator."""
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1"
    
    def generate_with_stability_ai(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        stability_api_key: Optional[str] = None,
    ) -> Optional[Image.Image]:
        """
        Generate image using Stability AI API.
        
        Args:
            prompt: Generation prompt
            width: Image width
            height: Image height
            stability_api_key: Stability AI API key (optional)
            
        Returns:
            Generated image or None
        """
        if not stability_api_key:
            logger.warning("Stability AI API key not provided")
            return None
        
        try:
            response = requests.post(
                "https://api.stability.ai/v2beta/stable-image/generate/core",
                headers={
                    "Authorization": f"Bearer {stability_api_key}",
                    "Accept": "image/*"
                },
                files={
                    "none": ""
                },
                data={
                    "prompt": prompt,
                    "output_format": "png",
                    "width": width,
                    "height": height,
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return Image.open(io.BytesIO(response.content))
            else:
                logger.error(f"Stability AI error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error with Stability AI: {e}")
            return None
    
    def generate_with_replicate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        replicate_api_token: Optional[str] = None,
    ) -> Optional[Image.Image]:
        """
        Generate image using Replicate API.
        
        Args:
            prompt: Generation prompt
            width: Image width
            height: Image height
            replicate_api_token: Replicate API token (optional)
            
        Returns:
            Generated image or None
        """
        if not replicate_api_token:
            logger.warning("Replicate API token not provided")
            return None
        
        try:
            import replicate
            
            output = replicate.run(
                "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
                input={
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                }
            )
            
            if output:
                image_url = output[0] if isinstance(output, list) else output
                img_response = requests.get(image_url, timeout=30)
                if img_response.status_code == 200:
                    return Image.open(io.BytesIO(img_response.content))
                    
        except ImportError:
            logger.warning("Replicate library not installed")
        except Exception as e:
            logger.error(f"Error with Replicate: {e}")
        
        return None
    
    def generate_simple(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
    ) -> Image.Image:
        """
        Generate a simple placeholder image with enhanced prompt.
        
        This is a fallback when no image generation API is available.
        In production, replace this with actual image generation.
        
        Args:
            prompt: Generation prompt
            width: Image width
            height: Image height
            
        Returns:
            Placeholder image
        """
        from PIL import ImageDraw, ImageFont
        
        # Create image
        img = Image.new('RGB', (width, height), color=(240, 240, 250))
        draw = ImageDraw.Draw(img)
        
        # Try to load font
        try:
            font_large = ImageFont.truetype("arial.ttf", 32)
            font_small = ImageFont.truetype("arial.ttf", 20)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Draw text
        title = "DeepSeek Generated Image"
        prompt_text = f"Prompt: {prompt[:80]}..." if len(prompt) > 80 else f"Prompt: {prompt}"
        
        # Center text
        bbox_title = draw.textbbox((0, 0), title, font=font_large)
        bbox_prompt = draw.textbbox((0, 0), prompt_text, font=font_small)
        
        title_x = (width - (bbox_title[2] - bbox_title[0])) // 2
        title_y = height // 2 - 40
        prompt_x = (width - (bbox_prompt[2] - bbox_prompt[0])) // 2
        prompt_y = height // 2 + 20
        
        draw.text((title_x, title_y), title, fill=(100, 100, 150), font=font_large)
        draw.text((prompt_x, prompt_y), prompt_text, fill=(150, 150, 150), font=font_small)
        
        # Draw border
        draw.rectangle([10, 10, width-10, height-10], outline=(200, 200, 200), width=3)
        
        return img


