"""
Fast Analyzer with Optimizations
"""

import torch
from typing import Dict, Optional, Any, List
import logging

from .models.enhanced_analyzer import EnhancedAddictionAnalyzer
from .models.fast_models import create_fast_predictors, create_fast_sentiment_analyzer
from ..utils.fast_inference import FastInferenceEngine

logger = logging.getLogger(__name__)


class FastRecoveryAnalyzer(EnhancedAddictionAnalyzer):
    """Fast version with optimizations"""
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        use_gpu: bool = True,
        use_fast_models: bool = True,
        use_jit: bool = True,
        use_quantization: bool = True
    ):
        """Initialize fast analyzer"""
        self.use_fast_models = use_fast_models
        
        # Initialize fast inference engine
        self.fast_engine = FastInferenceEngine(
            device=device,
            use_gpu=use_gpu,
            cache_size=1000,
            batch_size=32
        )
        
        # Initialize parent
        super().__init__(
            device=device,
            use_gpu=use_gpu,
            use_sentiment=False,  # We'll add fast version
            use_predictors=False,  # We'll add fast versions
            use_llm=False  # Skip LLM for speed
        )
        
        # Replace with fast models if enabled
        if use_fast_models:
            try:
                fast_progress, fast_relapse = create_fast_predictors(
                    device=self.device,
                    use_jit=use_jit,
                    use_quantization=use_quantization
                )
                self.progress_predictor = fast_progress
                self.relapse_predictor = fast_relapse
                logger.info("Fast models loaded")
            except Exception as e:
                logger.warning(f"Fast models not available: {e}")
                # Fallback
                from .models.sentiment_analyzer import create_progress_predictor, create_relapse_predictor
                self.progress_predictor = create_progress_predictor(device=self.device)
                self.relapse_predictor = create_relapse_predictor(device=self.device)
        
        # Fast sentiment analyzer
        try:
            self.sentiment_analyzer = create_fast_sentiment_analyzer(
                device=self.device,
                use_quantization=use_quantization
            )
        except Exception as e:
            logger.warning(f"Fast sentiment analyzer not available: {e}")
            self.sentiment_analyzer = None
        
        logger.info("FastRecoveryAnalyzer initialized")
    
    def predict_progress(self, features: Dict[str, float]) -> float:
        """Fast cached progress prediction"""
        if self.progress_predictor:
            return self.fast_engine.predict_progress_cached(
                self.progress_predictor,
                features
            )
        return super().predict_progress(features)
    
    def predict_relapse_risk(self, sequence: List[Dict[str, float]]) -> float:
        """Fast relapse risk prediction"""
        if self.relapse_predictor:
            results = self.fast_engine.predict_relapse_batch(
                self.relapse_predictor,
                [sequence]
            )
            return results[0] if results else 0.5
        return super().predict_relapse_risk(sequence)
    
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
                
                label_id = probs.argmax().item()
                label = "POSITIVE" if label_id == 1 else "NEGATIVE"
                score = probs[0][label_id].item()
                
                return {"label": label, "score": score}
            except Exception as e:
                logger.error(f"Sentiment analysis failed: {e}")
        
        return super().analyze_sentiment(text)
    
    def clear_cache(self):
        """Clear inference cache"""
        self.fast_engine.clear_cache()


def create_fast_analyzer(
    device: Optional[torch.device] = None,
    use_gpu: bool = True
) -> FastRecoveryAnalyzer:
    """Factory function for fast analyzer"""
    return FastRecoveryAnalyzer(
        device=device,
        use_gpu=use_gpu,
        use_fast_models=True,
        use_jit=True,
        use_quantization=True
    )

