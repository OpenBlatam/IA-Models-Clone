"""
Prompt Optimizer for Flux2 Clothing Changer
============================================

Advanced prompt optimization and enhancement.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class PromptAnalysis:
    """Analysis of a prompt."""
    word_count: int
    quality_score: float
    has_style: bool
    has_quality_terms: bool
    has_details: bool
    suggestions: List[str]


class PromptOptimizer:
    """Advanced prompt optimization system."""
    
    def __init__(self):
        """Initialize prompt optimizer."""
        # Quality terms that improve results
        self.quality_terms = [
            "high quality", "detailed", "professional", "sharp", "crisp",
            "well-lit", "high resolution", "4k", "8k", "photorealistic",
            "masterpiece", "best quality", "ultra detailed",
        ]
        
        # Style terms
        self.style_terms = [
            "photography", "portrait", "fashion", "editorial", "studio",
            "cinematic", "artistic", "vintage", "modern", "contemporary",
        ]
        
        # Detail terms
        self.detail_terms = [
            "texture", "fabric", "material", "pattern", "design",
            "stitching", "embroidery", "accessories", "jewelry",
        ]
        
        # Negative terms to avoid
        self.negative_terms = [
            "blurry", "low quality", "distorted", "deformed", "ugly",
            "bad anatomy", "worst quality", "low resolution",
        ]
    
    def analyze_prompt(self, prompt: str) -> PromptAnalysis:
        """
        Analyze a prompt and provide feedback.
        
        Args:
            prompt: Input prompt
            
        Returns:
            PromptAnalysis with suggestions
        """
        prompt_lower = prompt.lower()
        words = prompt.split()
        word_count = len(words)
        
        # Check for quality terms
        has_quality_terms = any(term in prompt_lower for term in self.quality_terms)
        
        # Check for style terms
        has_style = any(term in prompt_lower for term in self.style_terms)
        
        # Check for detail terms
        has_details = any(term in prompt_lower for term in self.detail_terms)
        
        # Check for negative terms
        has_negative = any(term in prompt_lower for term in self.negative_terms)
        
        # Calculate quality score
        quality_score = 0.5  # Base score
        
        if has_quality_terms:
            quality_score += 0.2
        if has_style:
            quality_score += 0.15
        if has_details:
            quality_score += 0.1
        if word_count >= 10:
            quality_score += 0.05
        
        if has_negative:
            quality_score -= 0.3
        
        quality_score = max(0.0, min(1.0, quality_score))
        
        # Generate suggestions
        suggestions = []
        
        if not has_quality_terms:
            suggestions.append("Add quality terms like 'high quality', 'detailed', or 'professional'")
        
        if not has_style:
            suggestions.append("Consider adding style context like 'fashion photography' or 'editorial'")
        
        if not has_details:
            suggestions.append("Add detail terms like 'texture', 'fabric', or 'pattern' for better results")
        
        if word_count < 10:
            suggestions.append(f"Prompt is short ({word_count} words). Consider adding more descriptive details.")
        
        if has_negative:
            suggestions.append("Remove negative terms that may affect quality")
        
        return PromptAnalysis(
            word_count=word_count,
            quality_score=quality_score,
            has_style=has_style,
            has_quality_terms=has_quality_terms,
            has_details=has_details,
            suggestions=suggestions,
        )
    
    def optimize_prompt(
        self,
        base_prompt: str,
        clothing_description: str,
        style: Optional[str] = None,
        quality_level: str = "high",
    ) -> str:
        """
        Optimize a prompt for better results.
        
        Args:
            base_prompt: Base prompt
            clothing_description: Clothing description
            style: Optional style (photography, fashion, editorial, etc.)
            quality_level: Quality level (low, medium, high, ultra)
            
        Returns:
            Optimized prompt
        """
        # Start with base prompt
        optimized = base_prompt
        
        # Add clothing description if not present
        if clothing_description.lower() not in optimized.lower():
            optimized = f"{optimized}, {clothing_description}"
        
        # Add quality terms based on level
        quality_additions = {
            "low": [],
            "medium": ["high quality"],
            "high": ["high quality", "detailed", "professional"],
            "ultra": ["ultra detailed", "masterpiece", "best quality", "8k", "photorealistic"],
        }
        
        for term in quality_additions.get(quality_level, []):
            if term not in optimized.lower():
                optimized = f"{optimized}, {term}"
        
        # Add style if specified
        if style and style.lower() not in optimized.lower():
            style_mapping = {
                "photography": "professional photography",
                "fashion": "fashion photography",
                "editorial": "editorial style",
                "portrait": "portrait photography",
                "cinematic": "cinematic lighting",
            }
            style_term = style_mapping.get(style.lower(), style)
            optimized = f"{optimized}, {style_term}"
        
        # Add detail terms
        detail_additions = ["texture", "fabric details"]
        for term in detail_additions:
            if term not in optimized.lower():
                optimized = f"{optimized}, {term}"
        
        # Clean up (remove duplicate commas, extra spaces)
        optimized = re.sub(r",\s*,+", ",", optimized)
        optimized = re.sub(r"\s+", " ", optimized)
        optimized = optimized.strip()
        
        return optimized
    
    def generate_negative_prompt(
        self,
        base_negative: Optional[str] = None,
        exclude_terms: Optional[List[str]] = None,
    ) -> str:
        """
        Generate optimized negative prompt.
        
        Args:
            base_negative: Base negative prompt
            exclude_terms: Terms to exclude from negative
            
        Returns:
            Optimized negative prompt
        """
        default_negative = (
            "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
            "worst quality, low resolution, jpeg artifacts, watermark, "
            "text, signature, username, multiple people, extra limbs"
        )
        
        if base_negative:
            negative = base_negative
        else:
            negative = default_negative
        
        # Remove excluded terms
        if exclude_terms:
            for term in exclude_terms:
                negative = negative.replace(term, "")
        
        # Clean up
        negative = re.sub(r",\s*,+", ",", negative)
        negative = re.sub(r"\s+", " ", negative)
        negative = negative.strip()
        
        return negative
    
    def compare_prompts(
        self,
        prompt1: str,
        prompt2: str,
    ) -> Dict[str, Any]:
        """
        Compare two prompts and provide analysis.
        
        Args:
            prompt1: First prompt
            prompt2: Second prompt
            
        Returns:
            Comparison analysis
        """
        analysis1 = self.analyze_prompt(prompt1)
        analysis2 = self.analyze_prompt(prompt2)
        
        return {
            "prompt1": {
                "text": prompt1,
                "analysis": analysis1,
            },
            "prompt2": {
                "text": prompt2,
                "analysis": analysis2,
            },
            "comparison": {
                "quality_difference": analysis2.quality_score - analysis1.quality_score,
                "word_count_difference": analysis2.word_count - analysis1.word_count,
                "better_prompt": "prompt2" if analysis2.quality_score > analysis1.quality_score else "prompt1",
            },
        }
    
    def extract_keywords(self, prompt: str) -> List[str]:
        """Extract keywords from prompt."""
        # Remove common words
        stop_words = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "are", "was", "were",
        }
        
        words = re.findall(r"\b\w+\b", prompt.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Count frequency
        keyword_counts = Counter(keywords)
        
        # Return top keywords
        return [word for word, count in keyword_counts.most_common(10)]


