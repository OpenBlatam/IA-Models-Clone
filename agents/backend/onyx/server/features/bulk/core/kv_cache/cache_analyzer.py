"""
Cache analyzer for optimization and recommendations.

Provides analysis tools to optimize cache configuration.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional
import time

from kv_cache.types import StatsDict

logger = logging.getLogger(__name__)


class CacheAnalyzer:
    """
    Cache analyzer for optimization recommendations.
    
    Analyzes cache performance and provides optimization suggestions.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize cache analyzer.
        
        Args:
            cache: Cache instance to analyze
        """
        self.cache = cache
        self.analysis_history: List[Dict[str, Any]] = []
    
    def analyze_performance(self) -> Dict[str, Any]:
        """
        Analyze cache performance.
        
        Returns:
            Dictionary with performance analysis
        """
        stats = self.cache.get_stats(include_history=True)
        
        analysis = {
            "timestamp": time.time(),
            "hit_rate": stats.get("hit_rate", 0.0),
            "cache_size": stats.get("num_entries", 0),
            "max_tokens": stats.get("max_tokens", 0),
            "memory_usage_mb": stats.get("storage_memory_mb", 0.0),
            "evictions": stats.get("evictions", 0),
            "recommendations": []
        }
        
        # Generate recommendations
        recommendations = []
        
        # Hit rate recommendations
        hit_rate = analysis["hit_rate"]
        if hit_rate < 0.5:
            recommendations.append({
                "type": "low_hit_rate",
                "severity": "high",
                "message": f"Hit rate is {hit_rate:.2%}. Consider increasing cache size or improving eviction strategy.",
                "suggestions": [
                    "Increase max_tokens",
                    "Use adaptive eviction strategy",
                    "Enable cache warmup",
                    "Review access patterns"
                ]
            })
        elif hit_rate < 0.7:
            recommendations.append({
                "type": "moderate_hit_rate",
                "severity": "medium",
                "message": f"Hit rate is {hit_rate:.2%}. Could be improved.",
                "suggestions": [
                    "Consider cache warmup",
                    "Review eviction strategy",
                    "Analyze access patterns"
                ]
            })
        
        # Memory recommendations
        memory_usage = analysis["memory_usage_mb"]
        if memory_usage > 1000:  # 1GB
            recommendations.append({
                "type": "high_memory",
                "severity": "medium",
                "message": f"Memory usage is {memory_usage:.2f} MB. Consider compression or quantization.",
                "suggestions": [
                    "Enable quantization (INT8)",
                    "Enable compression",
                    "Use memory-efficient data types",
                    "Consider reducing cache size"
                ]
            })
        
        # Eviction recommendations
        evictions = analysis["evictions"]
        cache_size = analysis["cache_size"]
        if evictions > cache_size * 0.5:
            recommendations.append({
                "type": "high_evictions",
                "severity": "medium",
                "message": f"High eviction rate ({evictions} evictions). Cache may be too small.",
                "suggestions": [
                    "Increase max_tokens",
                    "Use adaptive eviction strategy",
                    "Enable compression to fit more entries"
                ]
            })
        
        # Utilization recommendations
        utilization = cache_size / analysis["max_tokens"] if analysis["max_tokens"] > 0 else 0.0
        if utilization < 0.3:
            recommendations.append({
                "type": "low_utilization",
                "severity": "low",
                "message": f"Cache utilization is {utilization:.2%}. Cache may be oversized.",
                "suggestions": [
                    "Consider reducing max_tokens",
                    "Review memory usage patterns"
                ]
            })
        
        analysis["recommendations"] = recommendations
        analysis["utilization"] = utilization
        
        self.analysis_history.append(analysis)
        return analysis
    
    def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """
        Get optimization suggestions based on analysis.
        
        Returns:
            List of optimization suggestions
        """
        analysis = self.analyze_performance()
        return analysis.get("recommendations", [])
    
    def compare_strategies(
        self,
        strategies: List[str],
        test_duration: int = 60
    ) -> Dict[str, Any]:
        """
        Compare different cache strategies.
        
        Args:
            strategies: List of strategy names to compare
            test_duration: Test duration in seconds
            
        Returns:
            Dictionary with comparison results
        """
        # This would require modifying cache strategy at runtime
        # For now, return analysis based on current strategy
        current_stats = self.cache.get_stats()
        
        return {
            "current_strategy": self.cache.config.cache_strategy.value,
            "current_performance": {
                "hit_rate": current_stats.get("hit_rate", 0.0),
                "evictions": current_stats.get("evictions", 0),
                "memory_mb": current_stats.get("storage_memory_mb", 0.0)
            },
            "note": "Strategy comparison requires runtime testing",
            "suggested_strategies": [
                {
                    "strategy": "LRU",
                    "best_for": "Sequential access patterns"
                },
                {
                    "strategy": "LFU",
                    "best_for": "Frequent access to same items"
                },
                {
                    "strategy": "ADAPTIVE",
                    "best_for": "Mixed access patterns"
                }
            ]
        }
    
    def estimate_optimal_size(
        self,
        target_hit_rate: float = 0.8,
        sample_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Estimate optimal cache size.
        
        Args:
            target_hit_rate: Target hit rate
            sample_size: Sample size for estimation
            
        Returns:
            Dictionary with size estimation
        """
        stats = self.cache.get_stats()
        current_hit_rate = stats.get("hit_rate", 0.0)
        current_size = stats.get("num_entries", 0)
        max_tokens = stats.get("max_tokens", 0)
        
        # Simple estimation based on current hit rate
        if current_hit_rate > 0:
            # Estimate: if hit_rate scales roughly with size
            estimated_size = int(current_size * (target_hit_rate / current_hit_rate))
            estimated_size = min(estimated_size, max_tokens * 2)  # Cap at 2x current max
        else:
            estimated_size = max_tokens
        
        return {
            "current_size": current_size,
            "current_hit_rate": current_hit_rate,
            "target_hit_rate": target_hit_rate,
            "estimated_optimal_size": estimated_size,
            "recommendation": (
                "increase" if estimated_size > max_tokens
                else "decrease" if estimated_size < max_tokens * 0.5
                else "keep"
            ),
            "note": "Estimation is approximate. Actual optimal size may vary."
        }
    
    def get_analysis_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get analysis history.
        
        Args:
            limit: Maximum number of analyses to return
            
        Returns:
            List of past analyses
        """
        return self.analysis_history[-limit:]

