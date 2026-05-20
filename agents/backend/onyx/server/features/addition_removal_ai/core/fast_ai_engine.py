"""
Fast AI Engine with Optimizations
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
import logging
from functools import lru_cache

from .models.fast_models import create_fast_analyzer, optimize_model_for_inference
from .models.enhanced_ai_engine import EnhancedAIEngine
from ..utils.fast_inference import FastInferenceEngine, BatchProcessor

logger = logging.getLogger(__name__)


class FastAIEngine(EnhancedAIEngine):
    """Fast version of AI Engine with optimizations"""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        use_gpu: bool = True,
        use_fast_models: bool = True,
        batch_size: int = 32
    ):
        """
        Initialize fast AI engine
        
        Args:
            config: Configuration
            device: PyTorch device
            use_gpu: Use GPU
            use_fast_models: Use optimized fast models
            batch_size: Batch size for processing
        """
        self.use_fast_models = use_fast_models
        self.batch_size = batch_size
        
        # Initialize parent
        super().__init__(config, device, use_gpu)
        
        # Fast models
        if use_fast_models:
            self._initialize_fast_models()
        
        # Batch processor
        self.batch_processor = BatchProcessor(self._process_batch, batch_size=batch_size)
        
        logger.info(f"FastAIEngine initialized with batch_size={batch_size}")
    
    def _initialize_fast_models(self):
        """Initialize fast optimized models"""
        try:
            tokenizer, model = create_fast_analyzer(self.device)
            if model:
                # Optimize
                example_input = tokenizer("test", return_tensors="pt").to(self.device)
                self.fast_model = optimize_model_for_inference(model, example_input)
                self.fast_tokenizer = tokenizer
                self.fast_engine = FastInferenceEngine(
                    self.fast_model, self.fast_tokenizer, self.device, self.batch_size
                )
            else:
                self.fast_model = None
                self.fast_tokenizer = None
                self.fast_engine = None
        except Exception as e:
            logger.warning(f"Fast models initialization failed: {e}")
            self.fast_model = None
            self.fast_tokenizer = None
            self.fast_engine = None
    
    def _process_batch(self, texts: List[str]) -> List[Dict]:
        """Process batch of texts"""
        if self.fast_engine:
            return self.fast_engine.process_batch(texts)
        return [self.analyze_content(text) for text in texts]
    
    @lru_cache(maxsize=1000)
    def analyze_content_fast(self, content: str) -> Dict[str, Any]:
        """
        Fast cached content analysis
        
        Args:
            content: Content to analyze
            
        Returns:
            Analysis results
        """
        if self.fast_engine:
            return self.fast_engine.fast_analyze(content)
        return self.analyze_content(content)
    
    def analyze_batch(self, contents: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze batch of contents
        
        Args:
            contents: List of contents
            
        Returns:
            List of analysis results
        """
        return self.batch_processor.process(contents)
    
    def generate_batch(
        self,
        prompts: List[str],
        max_length: int = 100
    ) -> List[str]:
        """
        Generate content for batch of prompts
        
        Args:
            prompts: List of prompts
            max_length: Maximum length
            
        Returns:
            List of generated texts
        """
        results = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i+self.batch_size]
            batch_results = [
                self.generate_content(prompt, max_length)
                for prompt in batch
            ]
            results.extend(batch_results)
        return results
    
    def calculate_similarity_batch(
        self,
        text_pairs: List[tuple]
    ) -> List[float]:
        """
        Calculate similarity for batch of text pairs
        
        Args:
            text_pairs: List of (text1, text2) tuples
            
        Returns:
            List of similarity scores
        """
        if self.transformer_analyzer:
            return [
                self.transformer_analyzer.analyze_similarity(t1, t2)
                for t1, t2 in text_pairs
            ]
        return [0.0] * len(text_pairs)


def create_fast_ai_engine(
    config: Optional[Dict[str, Any]] = None,
    use_gpu: bool = True,
    batch_size: int = 32
) -> FastAIEngine:
    """Factory function to create fast AI engine"""
    return FastAIEngine(
        config=config,
        use_gpu=use_gpu,
        use_fast_models=True,
        batch_size=batch_size
    )

