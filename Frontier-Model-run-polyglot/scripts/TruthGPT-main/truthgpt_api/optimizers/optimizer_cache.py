"""
Optimizer Cache
===============

Thread-safe cache for optimizer instances.
Encapsulates caching logic and eliminates global state.
"""

import threading
import hashlib
from typing import Dict, Any, Optional


class OptimizerCache:
    """
    Thread-safe cache for optimizer instances.
    
    Responsibilities:
    - Cache optimizer instances by configuration
    - Generate cache keys from configuration
    - Provide thread-safe access
    """
    
    def __init__(self):
        """Initialize the cache."""
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()
    
    def _generate_key(
        self,
        optimizer_type: str,
        learning_rate: float,
        **kwargs
    ) -> str:
        """
        Generate a cache key from optimizer configuration.
        
        Args:
            optimizer_type: Type of optimizer
            learning_rate: Learning rate
            **kwargs: Additional parameters
        
        Returns:
            Cache key string
        """
        # Create a hashable representation
        config_str = f"{optimizer_type}_{learning_rate}_{hash(frozenset(kwargs.items()))}"
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def get(
        self,
        optimizer_type: str,
        learning_rate: float,
        **kwargs
    ) -> Optional[Any]:
        """
        Get cached optimizer if available.
        
        Args:
            optimizer_type: Type of optimizer
            learning_rate: Learning rate
            **kwargs: Additional parameters
        
        Returns:
            Cached optimizer or None
        """
        key = self._generate_key(optimizer_type, learning_rate, **kwargs)
        with self._lock:
            return self._cache.get(key)
    
    def put(
        self,
        optimizer_type: str,
        learning_rate: float,
        optimizer: Any,
        **kwargs
    ) -> None:
        """
        Store optimizer in cache.
        
        Args:
            optimizer_type: Type of optimizer
            learning_rate: Learning rate
            optimizer: Optimizer instance to cache
            **kwargs: Additional parameters
        """
        key = self._generate_key(optimizer_type, learning_rate, **kwargs)
        with self._lock:
            self._cache[key] = optimizer
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        """Get cache size."""
        with self._lock:
            return len(self._cache)

