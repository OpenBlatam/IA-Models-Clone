"""
Fast Inference Utilities
"""

import torch
from typing import List, Optional, Dict, Any
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class FastInferenceEngine:
    """Fast inference engine with caching and batching"""
    
    def __init__(self, model, tokenizer=None, device=None, batch_size=32):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._cache = {}
    
    @lru_cache(maxsize=1000)
    def encode_cached(self, text: str) -> torch.Tensor:
        """Cached encoding"""
        if self.tokenizer:
            inputs = self.tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
            return inputs.to(self.device)
        return None
    
    def process_batch(self, texts: List[str]) -> List[Any]:
        """Process batch of texts"""
        results = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            
            if self.tokenizer:
                inputs = self.tokenizer(
                    batch, return_tensors="pt", padding=True, 
                    truncation=True, max_length=128
                ).to(self.device)
            else:
                inputs = batch
            
            with torch.no_grad():
                if hasattr(self.model, '__call__'):
                    outputs = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)
                else:
                    outputs = self.model(inputs)
            
            results.extend(outputs)
        
        return results
    
    def fast_analyze(self, text: str) -> Dict[str, Any]:
        """Fast single text analysis"""
        if text in self._cache:
            return self._cache[text]
        
        result = self.process_batch([text])[0]
        self._cache[text] = result
        return result


class BatchProcessor:
    """Batch processor for efficient processing"""
    
    def __init__(self, processor_fn, batch_size=32):
        self.processor_fn = processor_fn
        self.batch_size = batch_size
    
    def process(self, items: List[Any]) -> List[Any]:
        """Process items in batches"""
        results = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i+self.batch_size]
            batch_results = self.processor_fn(batch)
            results.extend(batch_results)
        return results


def create_fast_engine(model, tokenizer=None, batch_size=32):
    """Create fast inference engine"""
    return FastInferenceEngine(model, tokenizer, batch_size=batch_size)

