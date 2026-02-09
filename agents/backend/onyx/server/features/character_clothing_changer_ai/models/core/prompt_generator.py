"""
Prompt Generator
================

Generates prompts for clothing change operations.
"""

import logging
from typing import Optional

from ..constants import DEFAULT_NEGATIVE_PROMPT

logger = logging.getLogger(__name__)


class PromptGenerator:
    """Generates prompts for clothing change operations."""
    
    def __init__(self, default_negative_prompt: str = DEFAULT_NEGATIVE_PROMPT):
        """
        Initialize prompt generator.
        
        Args:
            default_negative_prompt: Default negative prompt
        """
        self.default_negative_prompt = default_negative_prompt
    
    def generate_prompt(
        self,
        clothing_description: str,
        style: Optional[str] = None,
        quality_tags: Optional[list] = None,
    ) -> str:
        """
        Generate a prompt for clothing change.
        
        Args:
            clothing_description: Description of clothing
            style: Style description (optional)
            quality_tags: Quality tags (optional)
            
        Returns:
            Generated prompt
        """
        base_prompt = f"a character wearing {clothing_description}"
        
        if style:
            base_prompt += f", {style}"
        
        if quality_tags:
            base_prompt += ", " + ", ".join(quality_tags)
        else:
            base_prompt += ", high quality, detailed, professional photography"
        
        return base_prompt
    
    def generate_negative_prompt(
        self,
        custom_negative: Optional[str] = None,
    ) -> str:
        """
        Generate negative prompt.
        
        Args:
            custom_negative: Custom negative prompt (optional)
            
        Returns:
            Negative prompt
        """
        if custom_negative:
            return custom_negative
        return self.default_negative_prompt
    
    def enhance_prompt(
        self,
        base_prompt: str,
        character_name: Optional[str] = None,
        additional_details: Optional[list] = None,
    ) -> str:
        """
        Enhance a base prompt with additional details.
        
        Args:
            base_prompt: Base prompt
            character_name: Character name (optional)
            additional_details: Additional details (optional)
            
        Returns:
            Enhanced prompt
        """
        prompt = base_prompt
        
        if character_name:
            prompt = f"{character_name}, {prompt}"
        
        if additional_details:
            prompt += ", " + ", ".join(additional_details)
        
        return prompt


