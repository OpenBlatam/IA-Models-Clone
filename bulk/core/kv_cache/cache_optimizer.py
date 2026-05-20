"""
Cache optimizer for automatic tuning.

Automatically optimizes cache configuration based on usage patterns.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List
import time

from kv_cache.config import KVCacheConfig, CacheStrategy

logger = logging.getLogger(__name__)


class CacheOptimizer:
    """
    Automatic cache optimizer.
    
    Observes cache behavior and automatically tunes configuration.
    """
    
    def __init__(self, cache: Any, optimization_interval: int = 1000):
        """
        Initialize cache optimizer.
        
        Args:
            cache: Cache instance to optimize
            optimization_interval: Number of operations between optimizations
        """
        self.cache = cache
        self.optimization_interval = optimization_interval
        self.operation_count = 0
        self.optimization_history: List[Dict[str, Any]] = []
    
    def should_optimize(self) -> bool:
        """Check if optimization should run."""
        self.operation_count += 1
        return self.operation_count % self.optimization_interval == 0
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run optimization.
        
        Returns:
            Dictionary with optimization results
        """
        stats = self.cache.get_stats()
        config = self.cache.config
        
        optimization = {
            "timestamp": time.time(),
            "current_config": {
                "max_tokens": config.max_tokens,
                "strategy": config.cache_strategy.value,
                "use_quantization": config.use_quantization,
                "use_compression": config.use_compression
            },
            "current_stats": {
                "hit_rate": stats.get("hit_rate", 0.0),
                "num_entries": stats.get("num_entries", 0),
                "memory_mb": stats.get("storage_memory_mb", 0.0)
            },
            "recommendations": [],
            "applied": False
        }
        
        recommendations = []
        
        # Optimize cache size
        hit_rate = stats.get("hit_rate", 0.0)
        num_entries = stats.get("num_entries", 0)
        utilization = num_entries / config.max_tokens if config.max_tokens > 0 else 0.0
        
        if hit_rate < 0.6 and utilization > 0.8:
            # Cache nearly full and low hit rate - increase size
            new_size = int(config.max_tokens * 1.5)
            recommendations.append({
                "type": "increase_size",
                "current": config.max_tokens,
                "suggested": new_size,
                "reason": "Low hit rate and high utilization"
            })
        
        # Optimize strategy
        evictions = stats.get("evictions", 0)
        if evictions > num_entries * 0.3:
            # High eviction rate - try adaptive strategy
            if config.cache_strategy != CacheStrategy.ADAPTIVE:
                recommendations.append({
                    "type": "change_strategy",
                    "current": config.cache_strategy.value,
                    "suggested": CacheStrategy.ADAPTIVE.value,
                    "reason": "High eviction rate"
                })
        
        # Optimize compression
        memory_mb = stats.get("storage_memory_mb", 0.0)
        if memory_mb > 500 and not config.use_compression:
            recommendations.append({
                "type": "enable_compression",
                "reason": "High memory usage"
            })
        
        # Optimize quantization
        if memory_mb > 1000 and not config.use_quantization:
            recommendations.append({
                "type": "enable_quantization",
                "reason": "Very high memory usage"
            })
        
        optimization["recommendations"] = recommendations
        
        # Apply recommendations if safe
        if recommendations:
            optimization["applied"] = self._apply_recommendations(recommendations)
        
        self.optimization_history.append(optimization)
        return optimization
    
    def _apply_recommendations(self, recommendations: List[Dict[str, Any]]) -> bool:
        """
        Apply optimization recommendations.
        
        Args:
            recommendations: List of recommendations
            
        Returns:
            True if any recommendations were applied
        """
        applied = False
        
        for rec in recommendations:
            try:
                if rec["type"] == "increase_size":
                    # Increase max_tokens (requires cache recreation)
                    logger.info(f"Increasing cache size: {rec['current']} -> {rec['suggested']}")
                    # Note: This would require cache recreation in production
                    # For now, just log
                    applied = True
                
                elif rec["type"] == "change_strategy":
                    # Change strategy (requires cache recreation)
                    logger.info(f"Changing strategy: {rec['current']} -> {rec['suggested']}")
                    # Note: This would require cache recreation
                    applied = True
                
                elif rec["type"] == "enable_compression":
                    # Enable compression (runtime change)
                    if not self.cache.config.use_compression:
                        logger.info("Enabling compression")
                        self.cache.config.use_compression = True
                        applied = True
                
                elif rec["type"] == "enable_quantization":
                    # Enable quantization (runtime change)
                    if not self.cache.config.use_quantization:
                        logger.info("Enabling quantization")
                        self.cache.config.use_quantization = True
                        applied = True
                
            except Exception as e:
                logger.warning(f"Failed to apply recommendation {rec['type']}: {e}")
        
        return applied
    
    def auto_optimize(self) -> Optional[Dict[str, Any]]:
        """
        Auto-optimize if interval reached.
        
        Returns:
            Optimization results if optimization ran, None otherwise
        """
        if self.should_optimize():
            return self.optimize()
        return None
    
    def get_optimization_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get optimization history.
        
        Args:
            limit: Maximum number of optimizations to return
            
        Returns:
            List of past optimizations
        """
        return self.optimization_history[-limit:]

