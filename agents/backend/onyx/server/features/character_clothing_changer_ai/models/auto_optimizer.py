"""
Auto Optimizer for Flux2 Clothing Changer
==========================================

Automatic optimization of processing parameters based on image analysis.
"""

import torch
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AutoOptimizer:
    """Automatically optimizes processing parameters."""
    
    def __init__(
        self,
        enable_auto_tuning: bool = True,
        enable_adaptive_steps: bool = True,
        enable_adaptive_guidance: bool = True,
    ):
        """
        Initialize auto optimizer.
        
        Args:
            enable_auto_tuning: Enable automatic parameter tuning
            enable_adaptive_steps: Adapt inference steps based on complexity
            enable_adaptive_guidance: Adapt guidance scale based on image
        """
        self.enable_auto_tuning = enable_auto_tuning
        self.enable_adaptive_steps = enable_adaptive_steps
        self.enable_adaptive_guidance = enable_adaptive_guidance
    
    def optimize_parameters(
        self,
        image: Image.Image,
        clothing_description: str,
        base_steps: int = 50,
        base_guidance: float = 7.5,
        base_strength: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Optimize parameters based on image analysis.
        
        Args:
            image: Input image
            clothing_description: Clothing description
            base_steps: Base inference steps
            base_guidance: Base guidance scale
            base_strength: Base inpainting strength
            
        Returns:
            Dictionary of optimized parameters
        """
        if not self.enable_auto_tuning:
            return {
                "num_inference_steps": base_steps,
                "guidance_scale": base_guidance,
                "strength": base_strength,
            }
        
        # Analyze image complexity
        complexity = self._analyze_complexity(image)
        
        # Analyze description complexity
        desc_complexity = self._analyze_description_complexity(clothing_description)
        
        # Optimize steps
        if self.enable_adaptive_steps:
            steps = self._optimize_steps(base_steps, complexity, desc_complexity)
        else:
            steps = base_steps
        
        # Optimize guidance
        if self.enable_adaptive_guidance:
            guidance = self._optimize_guidance(base_guidance, complexity, desc_complexity)
        else:
            guidance = base_guidance
        
        # Optimize strength
        strength = self._optimize_strength(base_strength, complexity)
        
        return {
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "strength": strength,
            "complexity_score": complexity,
            "description_complexity": desc_complexity,
        }
    
    def _analyze_complexity(self, image: Image.Image) -> float:
        """Analyze image complexity (0.0 to 1.0)."""
        try:
            import cv2
            
            img_array = np.array(image.convert("RGB"))
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Texture complexity (variance)
            texture_complexity = np.var(gray) / 255.0
            
            # Color complexity
            color_complexity = np.std(img_array, axis=2).mean() / 255.0
            
            # Combine metrics
            complexity = (
                edge_density * 0.4 +
                texture_complexity * 0.3 +
                color_complexity * 0.3
            )
            
            return max(0.0, min(1.0, complexity))
            
        except ImportError:
            # Fallback
            img_array = np.array(image.convert("RGB"))
            complexity = np.std(img_array) / 255.0
            return max(0.0, min(1.0, complexity))
    
    def _analyze_description_complexity(self, description: str) -> float:
        """Analyze description complexity (0.0 to 1.0)."""
        # Simple heuristic based on length and keywords
        words = description.split()
        length_score = min(1.0, len(words) / 20.0)
        
        # Check for complex terms
        complex_terms = [
            "detailed", "intricate", "elaborate", "ornate",
            "patterned", "textured", "layered", "multi-colored"
        ]
        has_complex_terms = any(term in description.lower() for term in complex_terms)
        term_score = 0.3 if has_complex_terms else 0.0
        
        return max(0.0, min(1.0, length_score + term_score))
    
    def _optimize_steps(
        self,
        base_steps: int,
        complexity: float,
        desc_complexity: float,
    ) -> int:
        """Optimize inference steps."""
        # Higher complexity = more steps
        combined_complexity = (complexity + desc_complexity) / 2.0
        
        # Adjust steps: base ± 20%
        adjustment = (combined_complexity - 0.5) * 0.4  # -0.2 to +0.2
        steps = int(base_steps * (1.0 + adjustment))
        
        # Clamp to reasonable range
        return max(20, min(100, steps))
    
    def _optimize_guidance(
        self,
        base_guidance: float,
        complexity: float,
        desc_complexity: float,
    ) -> float:
        """Optimize guidance scale."""
        # Higher complexity = higher guidance
        combined_complexity = (complexity + desc_complexity) / 2.0
        
        # Adjust guidance: base ± 30%
        adjustment = (combined_complexity - 0.5) * 0.6  # -0.3 to +0.3
        guidance = base_guidance * (1.0 + adjustment)
        
        # Clamp to reasonable range
        return max(3.0, min(15.0, guidance))
    
    def _optimize_strength(self, base_strength: float, complexity: float) -> float:
        """Optimize inpainting strength."""
        # Higher complexity = slightly higher strength
        adjustment = (complexity - 0.5) * 0.2  # -0.1 to +0.1
        strength = base_strength + adjustment
        
        # Clamp to reasonable range
        return max(0.5, min(1.0, strength))


