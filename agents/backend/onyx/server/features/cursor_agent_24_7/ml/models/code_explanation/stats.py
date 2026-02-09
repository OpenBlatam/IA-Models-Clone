"""
Statistics tracking for code explanation model
"""

from typing import Dict, Any


class ModelStats:
    """Tracks model usage statistics"""
    
    def __init__(self):
        """Initialize statistics tracker"""
        self._stats: Dict[str, int] = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0
        }
    
    def increment_request(self):
        """Increment total request count"""
        self._stats["total_requests"] += 1
    
    def increment_cache_hit(self):
        """Increment cache hit count"""
        self._stats["cache_hits"] += 1
    
    def increment_cache_miss(self):
        """Increment cache miss count"""
        self._stats["cache_misses"] += 1
    
    def increment_error(self):
        """Increment error count"""
        self._stats["errors"] += 1
    
    def get_stats(
        self,
        initialized: bool,
        model_name: str,
        device: str,
        cache_enabled: bool = True
    ) -> Dict[str, Any]:
        """
        Get complete statistics.
        
        Args:
            initialized: Whether model is initialized
            model_name: Model name
            device: Device name
            cache_enabled: Whether cache is enabled
            
        Returns:
            Dictionary with statistics
        """
        total = self._stats["total_requests"]
        cache_hit_rate = (
            (self._stats["cache_hits"] / total * 100) if total > 0 else 0.0
        )
        
        return {
            **self._stats,
            "cache_hit_rate": round(cache_hit_rate, 2),
            "initialized": initialized,
            "model_name": model_name,
            "device": device,
            "cache_enabled": cache_enabled
        }
    
    def reset(self):
        """Reset all statistics"""
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0
        }

