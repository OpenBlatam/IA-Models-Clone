"""
Image Generation Providers
Supports multiple AI image generation services
"""

import asyncio
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pathlib import Path
import logging
from PIL import Image, ImageDraw
import httpx

logger = logging.getLogger(__name__)


class ImageProvider(ABC):
    """Base class for image generation providers"""
    
    @abstractmethod
    async def generate_image(
        self,
        prompt: str,
        width: int,
        height: int,
        style: str = "realistic"
    ) -> Path:
        """Generate an image from a prompt"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available"""
        pass


class OpenAIProvider(ImageProvider):
    """OpenAI DALL-E image generation provider"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1"
    
    def is_available(self) -> bool:
        return self.api_key is not None
    
    async def generate_image(
        self,
        prompt: str,
        width: int,
        height: int,
        style: str = "realistic"
    ) -> Path:
        """Generate image using OpenAI DALL-E"""
        if not self.is_available():
            raise ValueError("OpenAI API key not configured")
        
        # Map resolution to DALL-E supported sizes
        size_map = {
            (1024, 1024): "1024x1024",
            (1792, 1024): "1792x1024",
            (1024, 1792): "1024x1792",
        }
        
        # Find closest supported size
        dall_e_size = "1024x1024"  # Default
        for (w, h), size in size_map.items():
            if abs(w - width) < 100 and abs(h - height) < 100:
                dall_e_size = size
                break
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/images/generations",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "dall-e-3",
                        "prompt": prompt,
                        "size": dall_e_size,
                        "quality": "standard",
                        "n": 1,
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                # Download image
                image_url = data["data"][0]["url"]
                image_response = await client.get(image_url)
                image_response.raise_for_status()
                
                # Save image
                output_dir = Path("/tmp/faceless_video/images")
                output_dir.mkdir(parents=True, exist_ok=True)
                image_path = output_dir / f"dalle_{hash(prompt) % 100000}.png"
                
                with open(image_path, "wb") as f:
                    f.write(image_response.content)
                
                # Resize if needed
                if width != 1024 or height != 1024:
                    img = Image.open(image_path)
                    img = img.resize((width, height), Image.Resampling.LANCZOS)
                    img.save(image_path)
                
                logger.info(f"Generated image with DALL-E: {image_path}")
                return image_path
                
        except Exception as e:
            logger.error(f"OpenAI image generation failed: {str(e)}")
            raise


class StabilityAIProvider(ImageProvider):
    """Stability AI image generation provider"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("STABILITY_AI_API_KEY")
        self.base_url = "https://api.stability.ai/v1"
    
    def is_available(self) -> bool:
        return self.api_key is not None
    
    async def generate_image(
        self,
        prompt: str,
        width: int,
        height: int,
        style: str = "realistic"
    ) -> Path:
        """Generate image using Stability AI"""
        if not self.is_available():
            raise ValueError("Stability AI API key not configured")
        
        # Stability AI supports specific aspect ratios
        # Map to closest supported
        aspect_ratio = width / height
        if aspect_ratio > 1.5:
            width, height = 1024, 768
        elif aspect_ratio < 0.67:
            width, height = 768, 1024
        else:
            width, height = 1024, 1024
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "text_prompts": [{"text": prompt}],
                        "cfg_scale": 7,
                        "height": height,
                        "width": width,
                        "samples": 1,
                        "steps": 30,
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                # Extract base64 image
                import base64
                image_data = data["artifacts"][0]["base64"]
                image_bytes = base64.b64decode(image_data)
                
                # Save image
                output_dir = Path("/tmp/faceless_video/images")
                output_dir.mkdir(parents=True, exist_ok=True)
                image_path = output_dir / f"stability_{hash(prompt) % 100000}.png"
                
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                
                logger.info(f"Generated image with Stability AI: {image_path}")
                return image_path
                
        except Exception as e:
            logger.error(f"Stability AI image generation failed: {str(e)}")
            raise


class PlaceholderProvider(ImageProvider):
    """Placeholder image provider (fallback)"""
    
    def is_available(self) -> bool:
        return True  # Always available
    
    async def generate_image(
        self,
        prompt: str,
        width: int,
        height: int,
        style: str = "realistic"
    ) -> Path:
        """Generate placeholder image"""
        output_dir = Path("/tmp/faceless_video/images")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(image)
        
        # Create gradient based on prompt hash
        prompt_hash = hash(prompt)
        for y in range(height):
            ratio = y / height
            r = int(20 + (prompt_hash * 30) % 235)
            g = int(50 + (prompt_hash * 50) % 200)
            b = int(100 + (prompt_hash * 40) % 180)
            color = (r, g, b)
            draw.line([(0, y), (width, y)], fill=color)
        
        # Add text overlay
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("arial.ttf", 48)
        except:
            font = ImageFont.load_default()
        
        text = "AI Generated"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        position = ((width - text_width) // 2, (height - text_height) // 2)
        draw.text(position, text, fill=(255, 255, 255), font=font)
        
        image_path = output_dir / f"placeholder_{hash(prompt) % 100000}.png"
        image.save(image_path)
        
        logger.debug(f"Generated placeholder image: {image_path}")
        return image_path


def get_image_provider() -> ImageProvider:
    """Get the best available image provider"""
    # Try OpenAI first
    openai_provider = OpenAIProvider()
    if openai_provider.is_available():
        return openai_provider
    
    # Try Stability AI
    stability_provider = StabilityAIProvider()
    if stability_provider.is_available():
        return stability_provider
    
    # Fallback to placeholder
    logger.warning("No AI image provider configured, using placeholder")
    return PlaceholderProvider()

