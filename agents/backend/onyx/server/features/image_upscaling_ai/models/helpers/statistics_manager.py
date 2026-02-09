"""
Statistics Manager
==================

Manage processing statistics for upscaling operations.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class StatisticsManager:
    """Manage processing statistics."""
    
    @staticmethod
    def create_default_stats() -> Dict[str, Any]:
        """Create default statistics dictionary."""
        return {
            "upscales_performed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_time": 0.0,
            "successful_upscales": 0,
            "failed_upscales": 0,
            "images_validated": 0,
            "images_enhanced": 0,
            "validation_failures": 0,
            "method_selections": {},
        }
    
    @staticmethod
    def calculate_average_time(stats: Dict[str, Any]) -> float:
        """Calculate average time per upscale."""
        if stats["upscales_performed"] > 0:
            return stats["total_time"] / stats["upscales_performed"]
        return 0.0
    
    @staticmethod
    def calculate_success_rate(stats: Dict[str, Any]) -> float:
        """Calculate success rate."""
        if stats["upscales_performed"] > 0:
            return stats["successful_upscales"] / stats["upscales_performed"]
        return 0.0
    
    @staticmethod
    def calculate_cache_hit_rate(stats: Dict[str, Any]) -> float:
        """Calculate cache hit rate."""
        total_requests = stats["cache_hits"] + stats["cache_misses"]
        if total_requests > 0:
            return stats["cache_hits"] / total_requests
        return 0.0
    
    @staticmethod
    def get_statistics_summary(stats: Dict[str, Any], cache_stats: Any = None) -> Dict[str, Any]:
        """Get comprehensive statistics summary."""
        return {
            **stats,
            "average_time_per_upscale": round(
                StatisticsManager.calculate_average_time(stats), 4
            ),
            "success_rate": round(
                StatisticsManager.calculate_success_rate(stats), 4
            ),
            "cache_hit_rate": round(
                StatisticsManager.calculate_cache_hit_rate(stats), 4
            ),
            "cache_stats": cache_stats,
        }
    
    @staticmethod
    def reset_statistics() -> Dict[str, Any]:
        """Reset statistics to default values."""
        return StatisticsManager.create_default_stats()
    
    @staticmethod
    def get_derived_stats(
        stats: Dict[str, Any],
        cache: Any = None,
        features: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Get derived statistics with additional metrics."""
        avg_time = StatisticsManager.calculate_average_time(stats)
        success_rate = StatisticsManager.calculate_success_rate(stats)
        cache_hit_rate = StatisticsManager.calculate_cache_hit_rate(stats)
        
        validation_rate = (
            stats["images_validated"] / stats["upscales_performed"]
            if stats["upscales_performed"] > 0 else 0.0
        )
        enhancement_rate = (
            stats["images_enhanced"] / stats["upscales_performed"]
            if stats["upscales_performed"] > 0 else 0.0
        )
        
        result = {
            **stats,
            "average_time_per_upscale": round(avg_time, 4),
            "success_rate": round(success_rate, 4),
            "cache_hit_rate": round(cache_hit_rate, 4),
            "validation_rate": round(validation_rate, 4),
            "enhancement_rate": round(enhancement_rate, 4),
            "cache_stats": cache.get_stats() if cache else None,
        }
        
        if features:
            result["features"] = features
        
        return result
    
    @staticmethod
    def reset_stats() -> Dict[str, Any]:
        """Reset statistics to default values (alias for reset_statistics)."""
        return StatisticsManager.create_default_stats()

