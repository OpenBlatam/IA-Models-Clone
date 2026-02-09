"""
DeepSeek Clothing Changer Model
================================

Alternative model using DeepSeek API for clothing changes when Flux2 is unavailable.
Uses DeepSeek for prompt enhancement and image generation via API.
"""

import logging
import base64
import io
import requests
from typing import Optional, Dict, Any, Union
from PIL import Image
import numpy as np
from pathlib import Path
import tempfile
import json

logger = logging.getLogger(__name__)


class DeepSeekClothingModel:
    """
    DeepSeek-based clothing changer using API.
    
    This is a fallback when Flux2 is not available.
    Uses DeepSeek API for prompt enhancement and image generation.
    """
    
    def __init__(
        self,
        api_key: str = "sk-753365753f074509bb52496e038691f6",
        use_image_generation: bool = True,
    ):
        """
        Initialize DeepSeek Clothing Model.
        
        Args:
            api_key: DeepSeek API key
            use_image_generation: Whether to use DeepSeek for image generation
        """
        self.api_key = api_key
        self.use_image_generation = use_image_generation
        self.base_url = "https://api.deepseek.com/v1"
        
        logger.info("DeepSeekClothingModel initialized")
    
    def enhance_prompt(
        self,
        clothing_description: str,
        character_context: Optional[str] = None,
        original_prompt: Optional[str] = None,
    ) -> str:
        """
        Enhance prompt using DeepSeek API.
        
        Args:
            clothing_description: Basic clothing description
            character_context: Character context
            original_prompt: Original prompt if any
            
        Returns:
            Enhanced prompt
        """
        try:
            system_prompt = """You are an expert prompt engineer for AI image generation. 
Your task is to create detailed, optimized prompts for character clothing changes.
Return ONLY the enhanced prompt, no explanations."""

            user_prompt = f"""Create an optimized prompt for changing character clothing.

Clothing description: {clothing_description}
{f"Character context: {character_context}" if character_context else ""}
{f"Original prompt: {original_prompt}" if original_prompt else ""}

Requirements:
1. Be specific about clothing details (color, style, material, fit)
2. Maintain character consistency
3. Include relevant context (setting, pose, lighting if relevant)
4. Optimize for image generation models
5. Keep it concise but descriptive

Return ONLY the enhanced prompt:"""

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 300
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                enhanced = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if enhanced:
                    logger.info(f"Prompt enhanced: {enhanced[:100]}...")
                    return enhanced
            
            logger.warning("DeepSeek prompt enhancement failed, using original")
            return clothing_description
            
        except Exception as e:
            logger.warning(f"Error enhancing prompt with DeepSeek: {e}")
            return clothing_description
    
    def generate_image_from_prompt(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
    ) -> Image.Image:
        """
        Generate image from prompt using DeepSeek-enhanced prompt.
        
        Uses DeepSeek to enhance the prompt, then generates image.
        Currently uses a placeholder implementation that can be extended
        with actual image generation APIs.
        
        Args:
            prompt: Generation prompt
            negative_prompt: Negative prompt
            width: Image width
            height: Image height
            
        Returns:
            Generated image
        """
        try:
            from .deepseek_image_generator import DeepSeekImageGenerator
            
            generator = DeepSeekImageGenerator(api_key=self.api_key)
            
            # Try to use external APIs first
            import os
            
            # Try Stability AI if key is available
            stability_key = os.getenv("STABILITY_AI_API_KEY")
            if stability_key:
                img = generator.generate_with_stability_ai(
                    prompt=prompt,
                    width=width,
                    height=height,
                    stability_api_key=stability_key,
                )
                if img:
                    return img
            
            # Try Replicate if available
            replicate_token = os.getenv("REPLICATE_API_TOKEN")
            if replicate_token:
                img = generator.generate_with_replicate(
                    prompt=prompt,
                    width=width,
                    height=height,
                    replicate_api_token=replicate_token,
                )
                if img:
                    return img
            
            # Fallback to simple generation
            logger.info("Using simple image generation (install image generation API for better results)")
            return generator.generate_simple(prompt, width, height)
            
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            # Return a simple placeholder
            from PIL import ImageDraw
            img = Image.new('RGB', (width, height), color=(240, 240, 250))
            draw = ImageDraw.Draw(img)
            text = f"DeepSeek Mode\n{prompt[:50]}..."
            try:
                bbox = draw.textbbox((0, 0), text)
                x = (width - (bbox[2] - bbox[0])) // 2
                y = (height - (bbox[3] - bbox[1])) // 2
                draw.text((x, y), text, fill=(100, 100, 150))
            except:
                pass
            return img
    
    def change_clothing(
        self,
        image: Union[str, Path, Image.Image],
        clothing_description: str,
        character_name: Optional[str] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        strength: Optional[float] = None,
    ) -> Image.Image:
        """
        Change clothing in character image using DeepSeek.
        
        This implementation:
        1. Enhances the prompt using DeepSeek API
        2. Uses the original image as base
        3. Applies color/style transformations based on enhanced prompt
        4. Returns modified image
        
        Args:
            image: Input character image
            clothing_description: Description of new clothing
            character_name: Optional character name
            prompt: Optional full prompt
            negative_prompt: Negative prompt
            num_inference_steps: Not used in API mode
            guidance_scale: Not used in API mode
            strength: Transformation strength (0-1)
            
        Returns:
            Changed image
        """
        # Load image
        if isinstance(image, (str, Path)):
            original_img = Image.open(image).convert("RGB")
        else:
            original_img = image.convert("RGB")
        
        # Enhance prompt with DeepSeek
        logger.info("Enhancing prompt with DeepSeek...")
        enhanced_prompt = self.enhance_prompt(
            clothing_description=clothing_description,
            character_context=character_name,
            original_prompt=prompt,
        )
        
        logger.info(f"Enhanced prompt: {enhanced_prompt[:100]}...")
        
        # Apply transformations to the original image based on the prompt
        # This is a simplified approach - in production you'd use actual image generation
        modified_img = self._apply_clothing_changes(original_img, enhanced_prompt, strength or 0.8)
        
        return modified_img
    
    def _apply_clothing_changes(
        self,
        image: Image.Image,
        prompt: str,
        strength: float = 0.8,
    ) -> Image.Image:
        """
        Apply clothing changes to image based on prompt.
        
        This is a simplified implementation that applies color and style adjustments.
        In production, this would use actual image generation or inpainting.
        
        Args:
            image: Original image
            prompt: Enhanced prompt describing clothing
            strength: Transformation strength
            
        Returns:
            Modified image
        """
        from PIL import ImageEnhance, ImageFilter
        
        # Create a copy
        result = image.copy()
        
        # Apply enhancements based on prompt keywords
        prompt_lower = prompt.lower()
        
        # Color adjustments based on clothing description
        if any(color in prompt_lower for color in ['red', 'rojo', 'crimson', 'scarlet']):
            # Enhance red tones
            enhancer = ImageEnhance.Color(result)
            result = enhancer.enhance(1.2)
        elif any(color in prompt_lower for color in ['blue', 'azul', 'navy', 'cyan']):
            # Enhance blue tones
            enhancer = ImageEnhance.Color(result)
            result = enhancer.enhance(1.15)
        elif any(color in prompt_lower for color in ['black', 'negro', 'dark']):
            # Darken image slightly
            enhancer = ImageEnhance.Brightness(result)
            result = enhancer.enhance(0.9)
        elif any(color in prompt_lower for color in ['white', 'blanco', 'light']):
            # Brighten image slightly
            enhancer = ImageEnhance.Brightness(result)
            result = enhancer.enhance(1.1)
        
        # Style adjustments
        if 'elegant' in prompt_lower or 'formal' in prompt_lower:
            # Sharpen for elegance
            result = result.filter(ImageFilter.SHARPEN)
        elif 'casual' in prompt_lower or 'sport' in prompt_lower:
            # Slight blur for casual look
            result = result.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Blend with original based on strength
        if strength < 1.0:
            result = Image.blend(image, result, strength)
        
        # Add watermark/indicator that this is DeepSeek mode
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(result)
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        # Small indicator in corner
        text = "DeepSeek"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Semi-transparent background
        overlay = Image.new('RGBA', (text_width + 10, text_height + 6), (0, 0, 0, 128))
        result.paste(overlay, (result.width - text_width - 15, 10), overlay)
        
        # Text
        draw.text(
            (result.width - text_width - 10, 13),
            text,
            fill=(255, 255, 255),
            font=font
        )
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_type": "deepseek-api",
            "status": "available",
            "api_key_configured": bool(self.api_key),
            "use_image_generation": self.use_image_generation,
        }

