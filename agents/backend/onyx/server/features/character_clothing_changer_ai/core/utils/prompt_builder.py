"""
Prompt Builder Utility
======================

Utility for building and enhancing prompts consistently across the application.
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path

from ...models.prompt_enhancer import PromptEnhancer
from ...models.core.prompt_generator import PromptGenerator

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Builds and enhances prompts for clothing changes."""
    
    def __init__(self):
        """Initialize Prompt Builder."""
        self.prompt_enhancer = PromptEnhancer()
        self.prompt_generator = PromptGenerator()
    
    def build_prompt(
        self,
        clothing_description: str,
        character_name: Optional[str] = None,
        style: Optional[str] = None,
        quality_level: str = "high",
        enhance: bool = True,
    ) -> str:
        """
        Build a complete prompt for clothing change.
        
        Args:
            clothing_description: Description of clothing
            character_name: Optional character name
            style: Optional style
            quality_level: Quality level (low, medium, high, ultra)
            enhance: Whether to enhance the prompt
            
        Returns:
            Complete prompt
        """
        if enhance:
            # Use prompt enhancer
            prompt = self.prompt_enhancer.enhance_prompt(
                clothing_description=clothing_description,
                style=style,
                quality_level=quality_level,
            )
        else:
            prompt = clothing_description
        
        # Add character name if provided
        if character_name:
            prompt = self.prompt_generator.enhance_prompt(
                base_prompt=prompt,
                character_name=character_name,
            )
        
        return prompt
    
    def build_negative_prompt(
        self,
        custom_negative: Optional[str] = None,
        exclude_clothing: Optional[list] = None,
    ) -> str:
        """
        Build negative prompt.
        
        Args:
            custom_negative: Custom negative prompt
            exclude_clothing: List of clothing items to exclude
            
        Returns:
            Negative prompt
        """
        if custom_negative:
            return custom_negative
        
        # Use prompt enhancer for negative prompt
        return self.prompt_enhancer.create_negative_prompt(
            base_negative=None,
            exclude_clothing=exclude_clothing,
        )
    
    def enhance_existing_prompt(
        self,
        base_prompt: str,
        clothing_description: str,
        character_name: Optional[str] = None,
        style: Optional[str] = None,
        quality_level: str = "high",
    ) -> str:
        """
        Enhance an existing prompt.
        
        Args:
            base_prompt: Existing prompt to enhance
            clothing_description: Clothing description
            character_name: Optional character name
            style: Optional style
            quality_level: Quality level
            
        Returns:
            Enhanced prompt
        """
        # Use prompt generator's enhance method
        enhanced = self.prompt_generator.enhance_prompt(
            base_prompt=base_prompt,
            character_name=character_name,
        )
        
        # Add quality terms if needed
        if quality_level in ["high", "ultra"]:
            enhanced = self.prompt_enhancer.enhance_prompt(
                clothing_description=enhanced,
                style=style,
                quality_level=quality_level,
            )
        
        return enhanced

