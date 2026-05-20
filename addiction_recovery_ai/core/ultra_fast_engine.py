"""
Ultra-Fast Recovery Engine combining all optimizations
"""

import torch
from typing import Dict, List, Optional, Any
import logging

from .models.fast_models import (
    create_fast_predictors,
    create_fast_sentiment_analyzer
)
from .models.enhanced_analyzer import EnhancedAddictionAnalyzer
from ..utils.fast_inference import FastInferenceEngine, AsyncInference

logger = logging.getLogger(__name__)


class UltraFastRecoveryEngine(EnhancedAddictionAnalyzer):
    """Ultra-fast engine with all optimizations"""
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        use_gpu: bool = True,
        use_jit: bool = True,
        use_quantization: bool = True,
        use_compile: bool = True,
        cache_size: int = 1000,
        batch_size: int = 32
    ):
        """
        Initialize ultra-fast engine
        
        Args:
            device: PyTorch device
            use_gpu: Use GPU
            use_jit: Use JIT compilation
            use_quantization: Use INT8 quantization
            use_compile: Use torch.compile
            cache_size: Cache size for inference
            batch_size: Batch size for processing
        """
        self.device = device or torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        
        # Initialize fast inference engine
        self.fast_engine = FastInferenceEngine(
            device=self.device,
            cache_size=cache_size,
            batch_size=batch_size,
            use_gpu=use_gpu
        )
        
        # Initialize parent without models (we'll add fast ones)
        super().__init__(
            device=self.device,
            use_gpu=use_gpu,
            use_sentiment=False,  # We'll add fast version
            use_predictors=False,  # We'll add fast versions
            use_llm=False  # Skip LLM for speed
        )
        
        # Replace with fast models
        try:
            self.progress_predictor, self.relapse_predictor = create_fast_predictors(
                device=self.device,
                use_jit=use_jit,
                use_quantization=use_quantization,
                use_compile=use_compile
            )
            logger.info("Fast predictors loaded")
        except Exception as e:
            logger.warning(f"Fast predictors not available: {e}")
            # Fallback to regular models
            from .models.sentiment_analyzer import create_progress_predictor, create_relapse_predictor
            self.progress_predictor = create_progress_predictor(device=self.device)
            self.relapse_predictor = create_relapse_predictor(device=self.device)
        
        # Fast sentiment analyzer
        try:
            self.sentiment_analyzer = create_fast_sentiment_analyzer(
                device=self.device,
                use_quantization=use_quantization
            )
            logger.info("Fast sentiment analyzer loaded")
        except Exception as e:
            logger.warning(f"Fast sentiment analyzer not available: {e}")
            self.sentiment_analyzer = None
        
        # Async inference wrappers
        if self.progress_predictor:
            self.async_progress = AsyncInference(self.progress_predictor, self.device)
        if self.relapse_predictor:
            self.async_relapse = AsyncInference(self.relapse_predictor, self.device)
        
        logger.info(f"UltraFastRecoveryEngine initialized on {self.device}")
    
    def predict_progress(
        self,
        features: Dict[str, float]
    ) -> float:
        """Ultra-fast progress prediction with caching"""
        if self.progress_predictor:
            return self.fast_engine.predict_progress_cached(
                self.progress_predictor,
                features
            )
        return super().predict_progress(features)
    
    def predict_relapse_risk(
        self,
        sequence: List[Dict[str, float]]
    ) -> float:
        """Ultra-fast relapse risk prediction"""
        if self.relapse_predictor:
            # Use batch processing for single item
            results = self.fast_engine.predict_relapse_batch(
                self.relapse_predictor,
                [sequence]
            )
            return results[0] if results else 0.5
        return super().predict_relapse_risk(sequence)
    
    def predict_relapse_batch(
        self,
        sequences: List[List[Dict[str, float]]]
    ) -> List[float]:
        """Batch relapse risk prediction"""
        if self.relapse_predictor:
            return self.fast_engine.predict_relapse_batch(
                self.relapse_predictor,
                sequences
            )
        return [super().predict_relapse_risk(seq) for seq in sequences]
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Fast sentiment analysis"""
        if self.sentiment_analyzer and self.sentiment_analyzer.available:
            try:
                from transformers import AutoTokenizer
                tokenizer = self.sentiment_analyzer.tokenizer
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.sentiment_analyzer.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                
                # Get label
                label_id = probs.argmax().item()
                label = "POSITIVE" if label_id == 1 else "NEGATIVE"
                score = probs[0][label_id].item()
                
                return {"label": label, "score": score}
            except Exception as e:
                logger.error(f"Sentiment analysis failed: {e}")
        
        return super().analyze_sentiment(text)
    
    def analyze_sentiment_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Batch sentiment analysis"""
        if self.sentiment_analyzer and self.sentiment_analyzer.available:
            try:
                from transformers import AutoTokenizer
                tokenizer = self.sentiment_analyzer.tokenizer
                
                # Batch tokenize
                inputs = tokenizer(
                    texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Batch predict
                with torch.no_grad():
                    outputs = self.sentiment_analyzer.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                
                # Process results
                results = []
                for i, prob in enumerate(probs):
                    label_id = prob.argmax().item()
                    label = "POSITIVE" if label_id == 1 else "NEGATIVE"
                    score = prob[label_id].item()
                    results.append({"label": label, "score": score})
                
                return results
            except Exception as e:
                logger.error(f"Batch sentiment analysis failed: {e}")
        
        return [self.analyze_sentiment(text) for text in texts]
    
    def clear_cache(self):
        """Clear inference cache"""
        self.fast_engine.clear_cache()
    
    def benchmark(self, num_runs: int = 100) -> Dict[str, Any]:
        """Benchmark inference speed"""
        from ..utils.fast_inference import benchmark_inference
        
        results = {}
        
        # Benchmark progress predictor
        if self.progress_predictor:
            test_features = torch.randn(1, 10).to(self.device)
            results["progress"] = benchmark_inference(
                self.progress_predictor,
                test_features,
                num_runs=num_runs
            )
        
        # Benchmark relapse predictor
        if self.relapse_predictor:
            test_sequence = torch.randn(1, 30, 5).to(self.device)
            results["relapse"] = benchmark_inference(
                self.relapse_predictor,
                test_sequence,
                num_runs=num_runs
            )
        
        return results


def create_ultra_fast_engine(
    device: Optional[torch.device] = None,
    use_gpu: bool = True,
    use_all_optimizations: bool = True
) -> UltraFastRecoveryEngine:
    """Factory function for ultra-fast engine"""
    return UltraFastRecoveryEngine(
        device=device,
        use_gpu=use_gpu,
        use_jit=use_all_optimizations,
        use_quantization=use_all_optimizations,
        use_compile=use_all_optimizations
    )

