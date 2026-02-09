"""
Prompt Enhancer
===============

Enhances prompts based on style and requirements.
"""

import logging

from shared.enums import AvatarStyle

logger = logging.getLogger(__name__)


class PromptEnhancer:
    """Utility class for prompt enhancement."""
    
    @staticmethod
    def enhance_prompt(prompt: str, style: AvatarStyle) -> str:
        """Enhance prompt based on style.
        
        Args:
            prompt: Base prompt
            style: Avatar style
        
        Returns:
            Enhanced prompt
        """
        
        style_enhancements = {
            AvatarStyle.REALISTIC: (
                "professional headshot, high quality, detailed, "
                "8k, photorealistic, sharp focus"
            ),
            AvatarStyle.CARTOON: (
                "cartoon style, vibrant colors, clean lines, "
                "professional illustration"
            ),
            AvatarStyle.ANIME: (
                "anime style, detailed, professional, high quality, "
                "manga art style"
            ),
            AvatarStyle.ARTISTIC: (
                "artistic portrait, creative, high quality, "
                "detailed, unique style"
            ),
        }
        
        enhancement = style_enhancements.get(style, style_enhancements[AvatarStyle.REALISTIC])
        return f"{prompt}, {enhancement}"

