"""
Optimizer Statistics
====================

Encapsulates statistics tracking for optimizer creation and usage.
Eliminates global state and provides thread-safe statistics management.
"""

import threading
import time
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class OptimizerStatistics:
    """
    Thread-safe statistics tracker for optimizer operations.
    
    Responsibilities:
    - Track optimizer creation counts
    - Track cache hits
    - Track errors
    - Calculate performance metrics
    """
    
    created: int = 0
    from_core: int = 0
    from_pytorch: int = 0
    cache_hits: int = 0
    errors: int = 0
    creation_times: list[float] = field(default_factory=list)
    last_created: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def record_creation(
        self,
        backend: str,
        creation_time: float = 0.0
    ) -> None:
        """
        Record an optimizer creation.
        
        Args:
            backend: Backend used ('core' or 'pytorch')
            creation_time: Time taken to create (optional)
        """
        with self._lock:
            self.created += 1
            self.last_created = time.time()
            
            if backend == 'core':
                self.from_core += 1
            elif backend == 'pytorch':
                self.from_pytorch += 1
            
            if creation_time > 0:
                self.creation_times.append(creation_time)
    
    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        with self._lock:
            self.cache_hits += 1
    
    def record_error(self) -> None:
        """Record an error."""
        with self._lock:
            self.errors += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current statistics with calculated metrics.
        
        Returns:
            Dictionary with statistics and calculated rates
        """
        with self._lock:
            total = self.created
            cache_hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
            core_usage_rate = (self.from_core / total * 100) if total > 0 else 0
            pytorch_fallback_rate = (self.from_pytorch / total * 100) if total > 0 else 0
            
            return {
                'created': self.created,
                'from_core': self.from_core,
                'from_pytorch': self.from_pytorch,
                'cache_hits': self.cache_hits,
                'errors': self.errors,
                'creation_times': self.creation_times.copy(),
                'last_created': self.last_created,
                'cache_hit_rate': cache_hit_rate,
                'core_usage_rate': core_usage_rate,
                'pytorch_fallback_rate': pytorch_fallback_rate
            }
    
    def reset(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self.created = 0
            self.from_core = 0
            self.from_pytorch = 0
            self.cache_hits = 0
            self.errors = 0
            self.creation_times.clear()
            self.last_created = time.time()

