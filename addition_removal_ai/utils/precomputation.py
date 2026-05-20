"""
Precomputation and Caching for Maximum Speed
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Callable
from functools import lru_cache
import hashlib
import pickle
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache for precomputed embeddings"""
    
    def __init__(self, cache_dir: str = "./cache", max_size: int = 10000):
        """
        Initialize embedding cache
        
        Args:
            cache_dir: Cache directory
            max_size: Maximum cache size
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self._memory_cache = {}
    
    def _get_key(self, text: str) -> str:
        """Generate cache key"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[torch.Tensor]:
        """Get cached embedding"""
        key = self._get_key(text)
        
        # Check memory cache
        if key in self._memory_cache:
            return self._memory_cache[key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{key}.pt"
        if cache_file.exists():
            try:
                embedding = torch.load(cache_file)
                self._memory_cache[key] = embedding
                return embedding
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        return None
    
    def set(self, text: str, embedding: torch.Tensor):
        """Cache embedding"""
        key = self._get_key(text)
        
        # Memory cache
        if len(self._memory_cache) >= self.max_size:
            # Remove oldest (simple FIFO)
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]
        
        self._memory_cache[key] = embedding
        
        # Disk cache
        cache_file = self.cache_dir / f"{key}.pt"
        try:
            torch.save(embedding, cache_file)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")


class PrecomputedFeatures:
    """Precomputed features for fast access"""
    
    def __init__(self, cache_dir: str = "./cache"):
        """
        Initialize precomputed features
        
        Args:
            cache_dir: Cache directory
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_cache = EmbeddingCache(cache_dir)
    
    def get_or_compute(
        self,
        text: str,
        compute_fn: Callable[[str], torch.Tensor]
    ) -> torch.Tensor:
        """
        Get cached or compute embedding
        
        Args:
            text: Input text
            compute_fn: Function to compute embedding
            
        Returns:
            Embedding tensor
        """
        # Try cache
        cached = self.embedding_cache.get(text)
        if cached is not None:
            return cached
        
        # Compute
        embedding = compute_fn(text)
        
        # Cache
        self.embedding_cache.set(text, embedding)
        
        return embedding


@lru_cache(maxsize=10000)
def cached_encode(text: str, model_name: str = "default") -> str:
    """Cached text encoding"""
    return text  # Placeholder, replace with actual encoding


def create_embedding_cache(cache_dir: str = "./cache") -> EmbeddingCache:
    """Factory function for embedding cache"""
    return EmbeddingCache(cache_dir)

