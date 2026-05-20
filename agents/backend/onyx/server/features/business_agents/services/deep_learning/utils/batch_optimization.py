"""
Batch Optimization Utilities
=============================

Utilities for optimizing batch processing and inference.
"""

import torch
from typing import List, Dict, Any, Optional, Callable
import logging
from functools import lru_cache
import hashlib

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Optimized batch processor for inference.
    
    Features:
    - Dynamic batching
    - Automatic padding
    - Memory-efficient processing
    - Caching
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        batch_size: int = 32,
        max_length: Optional[int] = None,
        use_amp: bool = True
    ):
        """
        Initialize batch processor.
        
        Args:
            model: Model for inference
            device: Target device
            batch_size: Batch size
            max_length: Maximum sequence length (for padding)
            use_amp: Whether to use mixed precision
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_amp = use_amp and device.type == "cuda"
        self.model.eval()
    
    def process_batch(
        self,
        inputs: List[Any],
        collate_fn: Optional[Callable] = None
    ) -> List[Any]:
        """
        Process inputs in optimized batches.
        
        Args:
            inputs: List of inputs
            collate_fn: Function to collate inputs into batches
        
        Returns:
            List of outputs
        """
        outputs = []
        
        # Process in batches
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:i+self.batch_size]
            
            # Collate if function provided
            if collate_fn:
                batch_tensor = collate_fn(batch)
            else:
                # Default: stack tensors
                batch_tensor = torch.stack([torch.tensor(x) for x in batch])
            
            batch_tensor = batch_tensor.to(self.device, non_blocking=True)
            
            # Inference
            with torch.no_grad():
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        batch_output = self.model(batch_tensor)
                else:
                    batch_output = self.model(batch_tensor)
            
            outputs.extend(batch_output.cpu().tolist())
        
        return outputs


class InferenceCache:
    """
    Cache for inference results.
    
    Reduces redundant computations for repeated inputs.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize inference cache.
        
        Args:
            max_size: Maximum cache size
        """
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _hash_input(self, input_data: Any) -> str:
        """Hash input data for cache key."""
        if isinstance(input_data, torch.Tensor):
            data_bytes = input_data.cpu().numpy().tobytes()
        elif isinstance(input_data, (list, tuple)):
            data_bytes = str(input_data).encode()
        else:
            data_bytes = str(input_data).encode()
        
        return hashlib.md5(data_bytes).hexdigest()
    
    def get(self, input_data: Any) -> Optional[Any]:
        """
        Get cached result.
        
        Args:
            input_data: Input data
        
        Returns:
            Cached result or None
        """
        key = self._hash_input(input_data)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, input_data: Any, result: Any) -> None:
        """
        Cache result.
        
        Args:
            input_data: Input data
            result: Result to cache
        """
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        key = self._hash_input(input_data)
        self.cache[key] = result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "max_size": self.max_size
        }
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0


class OptimizedInference:
    """
    Optimized inference wrapper with caching and batching.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        batch_size: int = 32,
        use_cache: bool = True,
        cache_size: int = 1000,
        use_amp: bool = True
    ):
        """
        Initialize optimized inference.
        
        Args:
            model: Model for inference
            device: Target device
            batch_size: Batch size
            use_cache: Whether to use caching
            cache_size: Cache size
            use_amp: Whether to use mixed precision
        """
        self.model = model
        self.device = device
        self.batch_processor = BatchProcessor(
            model, device, batch_size, use_amp=use_amp
        )
        self.cache = InferenceCache(cache_size) if use_cache else None
        self.model.eval()
    
    def __call__(self, inputs: Any, use_cache: Optional[bool] = None) -> Any:
        """
        Perform optimized inference.
        
        Args:
            inputs: Input data
            use_cache: Whether to use cache (overrides default)
        
        Returns:
            Inference results
        """
        use_cache = use_cache if use_cache is not None else (self.cache is not None)
        
        # Check cache
        if use_cache and self.cache:
            cached_result = self.cache.get(inputs)
            if cached_result is not None:
                return cached_result
        
        # Process
        if isinstance(inputs, list):
            result = self.batch_processor.process_batch(inputs)
        else:
            # Single input
            with torch.no_grad():
                input_tensor = inputs.to(self.device) if isinstance(inputs, torch.Tensor) else torch.tensor(inputs).to(self.device)
                
                if self.batch_processor.use_amp:
                    with torch.cuda.amp.autocast():
                        result = self.model(input_tensor)
                else:
                    result = self.model(input_tensor)
                
                result = result.cpu()
        
        # Cache result
        if use_cache and self.cache:
            self.cache.set(inputs, result)
        
        return result
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.cache:
            return self.cache.get_stats()
        return {"cache_enabled": False}



