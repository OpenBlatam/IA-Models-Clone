"""
Memory Optimizer for Imagen Video Enhancer AI
=============================================

Memory optimization utilities.
"""

import gc
import logging
import sys
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """
    Optimizes memory usage.
    
    Features:
    - Memory monitoring
    - Automatic cleanup
    - Cache size management
    - Garbage collection
    """
    
    @staticmethod
    def get_memory_usage() -> Dict[str, Any]:
        """
        Get current memory usage.
        
        Returns:
            Dictionary with memory information
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / (1024 * 1024),  # Resident Set Size
                "vms_mb": memory_info.vms / (1024 * 1024),  # Virtual Memory Size
                "percent": process.memory_percent(),
                "available_mb": psutil.virtual_memory().available / (1024 * 1024),
                "total_mb": psutil.virtual_memory().total / (1024 * 1024),
            }
        except ImportError:
            logger.warning("psutil not available, using basic memory info")
            return {
                "rss_mb": 0,
                "vms_mb": 0,
                "percent": 0,
                "available_mb": 0,
                "total_mb": 0,
                "note": "Install psutil for detailed memory information"
            }
    
    @staticmethod
    def optimize_memory(aggressive: bool = False):
        """
        Optimize memory usage.
        
        Args:
            aggressive: Use aggressive garbage collection
        """
        # Force garbage collection
        collected = gc.collect()
        
        if aggressive:
            # Multiple passes for aggressive cleanup
            for _ in range(3):
                gc.collect()
        
        logger.info(f"Memory optimization: collected {collected} objects")
        return collected
    
    @staticmethod
    def clear_cache_if_needed(
        cache_manager,
        max_size_mb: float = 500.0,
        target_size_mb: float = 300.0
    ) -> int:
        """
        Clear cache if memory usage is high.
        
        Args:
            cache_manager: CacheManager instance
            max_size_mb: Maximum cache size in MB
            target_size_mb: Target cache size after cleanup
            
        Returns:
            Number of entries cleaned
        """
        memory = MemoryOptimizer.get_memory_usage()
        
        # Check if memory usage is high
        if memory.get("percent", 0) > 80 or memory.get("rss_mb", 0) > max_size_mb:
            logger.info(f"High memory usage detected: {memory.get('percent', 0):.1f}%")
            
            # Cleanup expired cache entries
            cleaned = 0
            try:
                cleaned = asyncio.run(cache_manager.cleanup_expired())
            except Exception as e:
                logger.warning(f"Error cleaning cache: {e}")
            
            # Force garbage collection
            MemoryOptimizer.optimize_memory(aggressive=True)
            
            return cleaned
        
        return 0
    
    @staticmethod
    def get_memory_recommendations() -> List[str]:
        """
        Get memory optimization recommendations.
        
        Returns:
            List of recommendation strings
        """
        memory = MemoryOptimizer.get_memory_usage()
        recommendations = []
        
        if memory.get("percent", 0) > 80:
            recommendations.append("Memory usage is high (>80%). Consider reducing cache size.")
            recommendations.append("Clear expired cache entries more frequently.")
        
        if memory.get("rss_mb", 0) > 1000:
            recommendations.append("Resident memory is high. Consider optimizing image processing.")
            recommendations.append("Use compression for large results.")
        
        if memory.get("available_mb", float('inf')) < 500:
            recommendations.append("Low available memory. Consider reducing concurrent tasks.")
        
        if not recommendations:
            recommendations.append("Memory usage is optimal.")
        
        return recommendations

