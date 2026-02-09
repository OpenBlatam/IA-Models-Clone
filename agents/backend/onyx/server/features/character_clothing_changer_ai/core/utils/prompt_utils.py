"""
Prompt Utilities
===============

Handles prompt validation and style analysis.
"""

import logging
from typing import Dict, Any

from ...models.prompt_enhancer import PromptEnhancer, ClothingStyleAnalyzer

logger = logging.getLogger(__name__)


class PromptUtils:
    """Utilities for prompt validation and style analysis."""
    
    def __init__(self):
        """Initialize Prompt Utilities."""
        self.prompt_enhancer = PromptEnhancer()
        self.style_analyzer = ClothingStyleAnalyzer()
    
    def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Validate a prompt and get suggestions.
        
        Args:
            prompt: Prompt to validate
            
        Returns:
            Validation results
        """
        return self.prompt_enhancer.validate_prompt(prompt)
    
    def analyze_clothing_style(self, description: str) -> Dict[str, Any]:
        """
        Analyze clothing description to extract style information.
        
        Args:
            description: Clothing description
            
        Returns:
            Style analysis
        """
        return self.style_analyzer.analyze(description)


