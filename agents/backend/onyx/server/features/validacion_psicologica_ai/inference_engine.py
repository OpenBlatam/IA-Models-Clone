"""
Inference Engine
================
Optimized inference with batching and caching
"""

from typing import Dict, Any, List, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import structlog
from functools import lru_cache
import hashlib
import json

from .deep_learning_models import get_device
from .data_loader import DataPreprocessor

logger = structlog.get_logger()


class InferenceEngine:
    """
    Optimized inference engine with batching and caching
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        batch_size: int = 32,
        max_length: int = 512,
        use_cache: bool = True,
        cache_size: int = 1000
    ):
        """
        Initialize inference engine
        
        Args:
            model: Model for inference
            tokenizer: Tokenizer
            batch_size: Batch size for inference
            max_length: Maximum sequence length
            use_cache: Enable caching
            cache_size: Cache size
        """
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_cache = use_cache
        self.device = get_device()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Cache
        self.cache = {} if use_cache else None
        self.cache_size = cache_size
        
        # Preprocessor
        self.preprocessor = DataPreprocessor()
        
        logger.info(
            "InferenceEngine initialized",
            device=str(self.device),
            batch_size=batch_size,
            use_cache=use_cache
        )
    
    def predict(
        self,
        texts: Union[str, List[str]],
        return_probs: bool = False
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Predict on texts
        
        Args:
            texts: Single text or list of texts
            return_probs: Return probabilities
            
        Returns:
            Predictions
        """
        # Handle single text
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
        
        # Check cache
        if self.use_cache:
            cached_results = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self.cache:
                    cached_results.append((i, self.cache[cache_key]))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            # If all cached, return
            if len(uncached_texts) == 0:
                results = [r[1] for r in sorted(cached_results, key=lambda x: x[0])]
                return results[0] if is_single else results
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
            cached_results = []
        
        # Process uncached texts
        if len(uncached_texts) > 0:
            new_results = self._batch_predict(uncached_texts, return_probs)
            
            # Update cache
            if self.use_cache:
                for text, result in zip(uncached_texts, new_results):
                    cache_key = self._get_cache_key(text)
                    self._update_cache(cache_key, result)
            
            # Combine results
            all_results = [None] * len(texts)
            for idx, result in cached_results:
                all_results[idx] = result
            for idx, result in zip(uncached_indices, new_results):
                all_results[idx] = result
            
            results = all_results
        else:
            results = [r[1] for r in sorted(cached_results, key=lambda x: x[0])]
        
        return results[0] if is_single else results
    
    def _batch_predict(
        self,
        texts: List[str],
        return_probs: bool = False
    ) -> List[Dict[str, Any]]:
        """Batch prediction"""
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            try:
                # Preprocess
                inputs = self.preprocessor.preprocess_texts(batch_texts, self.max_length)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Inference
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Process outputs
                batch_results = self._process_outputs(outputs, return_probs)
                results.extend(batch_results)
                
            except Exception as e:
                logger.error("Error in batch prediction", error=str(e))
                # Return default results for failed batch
                results.extend([{"error": str(e)}] * len(batch_texts))
        
        return results
    
    def _process_outputs(
        self,
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        return_probs: bool
    ) -> List[Dict[str, Any]]:
        """Process model outputs"""
        if isinstance(outputs, dict):
            logits = outputs.get("logits", outputs.get("predictions"))
        else:
            logits = outputs
        
        probs = torch.softmax(logits, dim=-1) if logits.dim() > 1 else logits
        predictions = torch.argmax(probs, dim=-1) if probs.dim() > 1 else probs
        
        results = []
        for i in range(predictions.shape[0]):
            result = {
                "prediction": int(predictions[i].item())
            }
            
            if return_probs:
                result["probabilities"] = probs[i].cpu().tolist()
                result["confidence"] = float(probs[i].max().item())
            
            results.append(result)
        
        return results
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _update_cache(self, key: str, value: Any) -> None:
        """Update cache with size limit"""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = value
    
    def clear_cache(self) -> None:
        """Clear inference cache"""
        if self.cache is not None:
            self.cache.clear()
            logger.info("Inference cache cleared")


class ModelServer:
    """Model server for production inference"""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        inference_engine: Optional[InferenceEngine] = None
    ):
        """
        Initialize model server
        
        Args:
            model: Model to serve
            tokenizer: Tokenizer
            inference_engine: Inference engine (optional, will create if not provided)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.inference_engine = inference_engine or InferenceEngine(
            model=model,
            tokenizer=tokenizer
        )
        
        logger.info("ModelServer initialized")
    
    def serve(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Serve predictions
        
        Args:
            texts: Input texts
            **kwargs: Additional arguments
            
        Returns:
            Predictions
        """
        return self.inference_engine.predict(texts, **kwargs)




