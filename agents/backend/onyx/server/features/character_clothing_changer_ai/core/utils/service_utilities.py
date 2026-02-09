"""
Service Utilities
=================

Utility operations for the clothing changer service, including
prompt validation, style analysis, cache management, and model info.
"""

import logging
from typing import Optional, Dict, Any

from ...models.prompt_enhancer import PromptEnhancer, ClothingStyleAnalyzer
from ...models.embedding_cache import EmbeddingCache
from ...models.quality_metrics import QualityMetrics
from ...models.flux2_clothing_model import Flux2ClothingChangerModel
from ...models.deepseek_clothing_model import DeepSeekClothingModel

logger = logging.getLogger(__name__)


class ServiceUtilities:
    """Utility operations for the service."""
    
    def __init__(
        self,
        prompt_enhancer: PromptEnhancer,
        style_analyzer: ClothingStyleAnalyzer,
        quality_metrics: QualityMetrics,
        cache: Optional[EmbeddingCache] = None,
    ):
        """
        Initialize Service Utilities.
        
        Args:
            prompt_enhancer: Prompt enhancer instance
            style_analyzer: Style analyzer instance
            quality_metrics: Quality metrics instance
            cache: Optional cache instance
        """
        self.prompt_enhancer = prompt_enhancer
        self.style_analyzer = style_analyzer
        self.quality_metrics = quality_metrics
        self.cache = cache
    
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
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get embedding cache statistics.
        
        Returns:
            Cache statistics
        """
        if self.cache is None:
            return {"cache_enabled": False}
        
        stats = self.cache.get_cache_stats()
        stats["cache_enabled"] = True
        return stats
    
    def clear_cache(self) -> None:
        """Clear embedding cache."""
        if self.cache:
            self.cache.clear_cache()
            logger.info("Cache cleared")
    
    def get_model_info(
        self,
        model: Optional[Flux2ClothingChangerModel],
        deepseek_model: Optional[DeepSeekClothingModel],
        use_deepseek_fallback: bool
    ) -> Dict[str, Any]:
        """
        Get model information.
        
        Args:
            model: Flux2 model instance
            deepseek_model: DeepSeek model instance
            use_deepseek_fallback: Whether using DeepSeek fallback
            
        Returns:
            Model information dictionary
        """
        if use_deepseek_fallback and deepseek_model:
            info = deepseek_model.get_model_info()
            info["fallback_mode"] = True
            info["primary_model"] = "DeepSeek"
            info["fallback_reason"] = "Flux2 model not available"
            return info
        elif model is None:
            return {"status": "not_initialized"}
        
        info = model.get_model_info()
        info["fallback_mode"] = False
        info["primary_model"] = "Flux2"
        
        # Add cache info
        if self.cache:
            info["cache_stats"] = self.cache.get_cache_stats()
        
        return info

