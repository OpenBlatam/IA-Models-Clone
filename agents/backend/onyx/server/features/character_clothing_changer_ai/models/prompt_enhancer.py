"""
Prompt Enhancer for Clothing Changes
====================================

Utilities for enhancing and validating prompts for better clothing change results.
"""

import re
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PromptEnhancer:
    """Enhances prompts for better clothing change results."""
    
    # Quality boosters
    QUALITY_BOOSTERS = [
        "high quality",
        "detailed",
        "professional photography",
        "sharp focus",
        "8k resolution",
        "ultra detailed",
    ]
    
    # Style descriptors
    STYLE_DESCRIPTORS = {
        "casual": ["casual", "everyday", "relaxed"],
        "formal": ["formal", "elegant", "sophisticated", "professional"],
        "sporty": ["sporty", "athletic", "active", "performance"],
        "vintage": ["vintage", "retro", "classic", "timeless"],
        "modern": ["modern", "contemporary", "trendy", "fashion-forward"],
        "elegant": ["elegant", "refined", "luxurious", "premium"],
    }
    
    # Clothing types
    CLOTHING_TYPES = [
        "dress", "suit", "shirt", "pants", "skirt", "jacket",
        "coat", "blazer", "t-shirt", "jeans", "shorts", "sweater",
        "hoodie", "vest", "tunic", "gown", "outfit", "ensemble"
    ]
    
    def __init__(self):
        """Initialize prompt enhancer."""
        self.quality_boosters_used = set()
    
    def enhance_prompt(
        self,
        clothing_description: str,
        style: Optional[str] = None,
        quality_level: str = "high",
        add_details: bool = True,
    ) -> str:
        """
        Enhance a clothing description prompt.
        
        Args:
            clothing_description: Base clothing description
            style: Optional style (casual, formal, sporty, etc.)
            quality_level: Quality level (low, medium, high, ultra)
            add_details: Whether to add quality boosters
            
        Returns:
            Enhanced prompt
        """
        # Clean and normalize
        prompt = self._clean_description(clothing_description)
        
        # Add style if specified
        if style and style.lower() in self.STYLE_DESCRIPTORS:
            style_words = self.STYLE_DESCRIPTORS[style.lower()]
            prompt = f"{prompt}, {', '.join(style_words[:2])}"
        
        # Add quality boosters
        if add_details:
            quality_terms = self._get_quality_terms(quality_level)
            prompt = f"{prompt}, {', '.join(quality_terms)}"
        
        # Ensure clothing type is mentioned
        if not self._has_clothing_type(prompt):
            prompt = f"wearing {prompt}"
        
        return prompt
    
    def create_full_prompt(
        self,
        clothing_description: str,
        character_context: Optional[str] = None,
        style: Optional[str] = None,
        quality_level: str = "high",
    ) -> str:
        """
        Create a full prompt for image generation.
        
        Args:
            clothing_description: Clothing description
            character_context: Optional character context
            style: Optional style
            quality_level: Quality level
            
        Returns:
            Full prompt
        """
        # Build prompt parts
        parts = []
        
        if character_context:
            parts.append(character_context)
        
        # Enhanced clothing description
        enhanced_clothing = self.enhance_prompt(
            clothing_description,
            style=style,
            quality_level=quality_level,
        )
        parts.append(enhanced_clothing)
        
        # Combine
        prompt = ", ".join(parts)
        
        return prompt
    
    def create_negative_prompt(
        self,
        base_negative: Optional[str] = None,
        exclude_clothing: Optional[List[str]] = None,
    ) -> str:
        """
        Create an enhanced negative prompt.
        
        Args:
            base_negative: Base negative prompt
            exclude_clothing: List of clothing types to exclude
            
        Returns:
            Enhanced negative prompt
        """
        negative_parts = [
            "blurry",
            "low quality",
            "distorted",
            "deformed",
            "bad anatomy",
            "ugly",
            "duplicate",
            "mutation",
            "mutilated",
            "extra limbs",
            "missing limbs",
            "bad proportions",
        ]
        
        if base_negative:
            negative_parts.insert(0, base_negative)
        
        if exclude_clothing:
            for clothing in exclude_clothing:
                negative_parts.append(f"old {clothing}")
                negative_parts.append(f"worn {clothing}")
        
        return ", ".join(negative_parts)
    
    def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Validate a prompt and return suggestions.
        
        Args:
            prompt: Prompt to validate
            
        Returns:
            Dict with validation results and suggestions
        """
        issues = []
        suggestions = []
        
        # Check length
        if len(prompt) < 10:
            issues.append("prompt_too_short")
            suggestions.append("Add more details about the clothing")
        
        if len(prompt) > 500:
            issues.append("prompt_too_long")
            suggestions.append("Consider shortening the prompt")
        
        # Check for clothing type
        if not self._has_clothing_type(prompt):
            issues.append("no_clothing_type")
            suggestions.append("Specify the type of clothing (dress, suit, etc.)")
        
        # Check for quality terms
        has_quality = any(term in prompt.lower() for term in self.QUALITY_BOOSTERS)
        if not has_quality:
            suggestions.append("Add quality descriptors for better results")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "suggestions": suggestions,
            "length": len(prompt),
        }
    
    def _clean_description(self, description: str) -> str:
        """Clean and normalize description."""
        # Remove extra whitespace
        description = re.sub(r'\s+', ' ', description.strip())
        
        # Remove common prefixes
        description = re.sub(r'^(a|an|the)\s+', '', description, flags=re.IGNORECASE)
        
        return description
    
    def _get_quality_terms(self, quality_level: str) -> List[str]:
        """Get quality terms based on level."""
        base_terms = ["high quality", "detailed"]
        
        if quality_level == "ultra":
            return base_terms + ["8k resolution", "ultra detailed", "professional photography"]
        elif quality_level == "high":
            return base_terms + ["professional photography", "sharp focus"]
        elif quality_level == "medium":
            return base_terms
        else:
            return ["good quality"]
    
    def _has_clothing_type(self, prompt: str) -> bool:
        """Check if prompt mentions a clothing type."""
        prompt_lower = prompt.lower()
        return any(clothing_type in prompt_lower for clothing_type in self.CLOTHING_TYPES)


class ClothingStyleAnalyzer:
    """Analyzes clothing descriptions to extract style information."""
    
    def analyze(self, description: str) -> Dict[str, Any]:
        """
        Analyze clothing description to extract style information.
        
        Args:
            description: Clothing description
            
        Returns:
            Dict with style analysis
        """
        description_lower = description.lower()
        
        # Detect style
        detected_style = None
        for style, keywords in PromptEnhancer.STYLE_DESCRIPTORS.items():
            if any(keyword in description_lower for keyword in keywords):
                detected_style = style
                break
        
        # Detect clothing type
        clothing_type = None
        for ctype in PromptEnhancer.CLOTHING_TYPES:
            if ctype in description_lower:
                clothing_type = ctype
                break
        
        # Detect colors
        color_keywords = [
            "red", "blue", "green", "yellow", "black", "white", "gray", "grey",
            "pink", "purple", "orange", "brown", "beige", "navy", "maroon",
            "gold", "silver", "bronze", "ivory", "cream"
        ]
        colors = [color for color in color_keywords if color in description_lower]
        
        return {
            "style": detected_style,
            "clothing_type": clothing_type,
            "colors": colors,
            "description_length": len(description),
        }


