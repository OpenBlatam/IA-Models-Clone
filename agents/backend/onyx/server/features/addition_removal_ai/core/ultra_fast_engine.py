"""
Ultra Fast AI Engine with All Optimizations Combined
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import logging

from .models.enhanced_ai_engine import EnhancedAIEngine
from .models.fast_models import create_fast_analyzer
from .models.extreme_optimization import prune_model, optimize_memory_usage
from .models.optimized_inference import compile_model, create_fast_inference_model
from ..utils.precomputation import PrecomputedFeatures, create_embedding_cache
from ..utils.fast_inference import FastInferenceEngine

logger = logging.getLogger(__name__)


class UltraFastAIEngine(EnhancedAIEngine):
    """Ultra-fast AI engine with all optimizations"""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        use_gpu: bool = True,
        use_pruning: bool = False,
        pruning_ratio: float = 0.3,
        use_precomputation: bool = True,
        cache_dir: str = "./cache"
    ):
        """
        Initialize ultra-fast AI engine
        
        Args:
            config: Configuration
            device: PyTorch device
            use_gpu: Use GPU
            use_pruning: Enable model pruning
            pruning_ratio: Pruning ratio
            use_precomputation: Enable precomputation
            cache_dir: Cache directory
        """
        # Initialize parent
        super().__init__(config, device, use_gpu)
        
        # Pruning
        if use_pruning:
            if self.transformer_analyzer and hasattr(self.transformer_analyzer, 'model'):
                self.transformer_analyzer.model = prune_model(
                    self.transformer_analyzer.model,
                    pruning_ratio=pruning_ratio
                )
            logger.info(f"Models pruned with {pruning_ratio*100}% sparsity")
        
        # Compile all models
        if hasattr(torch, 'compile'):
            try:
                if self.transformer_analyzer and hasattr(self.transformer_analyzer, 'model'):
                    self.transformer_analyzer.model = compile_model(
                        self.transformer_analyzer.model,
                        mode="reduce-overhead"
                    )
                logger.info("Models compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Compilation failed: {e}")
        
        # Precomputation
        if use_precomputation:
            self.precomputed = PrecomputedFeatures(cache_dir)
        else:
            self.precomputed = None
        
        # Memory optimization
        optimize_memory_usage(self)
        
        logger.info("UltraFastAIEngine initialized with all optimizations")
    
    def analyze_content_ultra_fast(self, content: str) -> Dict[str, Any]:
        """
        Ultra-fast content analysis with caching
        
        Args:
            content: Content to analyze
            
        Returns:
            Analysis results
        """
        # Use precomputation if available
        if self.precomputed and self.transformer_analyzer:
            def compute_embedding(text):
                features = self.transformer_analyzer.extract_features(text)
                return features.get("pooler_output", torch.zeros(768))
            
            embedding = self.precomputed.get_or_compute(content, compute_embedding)
            
            # Quick analysis from embedding
            return {
                "embedding": embedding.cpu().numpy().tolist(),
                "cached": True
            }
        
        # Fallback to standard analysis
        return self.analyze_content(content)
    
    def analyze_batch_ultra_fast(self, contents: List[str]) -> List[Dict[str, Any]]:
        """
        Ultra-fast batch analysis
        
        Args:
            contents: List of contents
            
        Returns:
            List of analysis results
        """
        if self.precomputed and self.transformer_analyzer:
            results = []
            for content in contents:
                result = self.analyze_content_ultra_fast(content)
                results.append(result)
            return results
        
        return self.analyze_batch(contents)


def create_ultra_fast_engine(
    config: Optional[Dict[str, Any]] = None,
    use_gpu: bool = True,
    use_pruning: bool = True,
    use_precomputation: bool = True
) -> UltraFastAIEngine:
    """Factory function for ultra-fast engine"""
    return UltraFastAIEngine(
        config=config,
        use_gpu=use_gpu,
        use_pruning=use_pruning,
        use_precomputation=use_precomputation
    )

