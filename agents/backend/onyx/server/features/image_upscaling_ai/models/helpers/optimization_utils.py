"""
Optimization Utilities
=====================

Utilities for memory optimization, preset optimization, and resolution calculations.
"""

import logging
import gc
from typing import Dict, Any, Tuple, Optional

try:
    import psutil
    import os
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class OptimizationUtils:
    """Utilities for optimization."""
    
    @staticmethod
    def optimize_memory() -> None:
        """Optimize memory usage by forcing garbage collection."""
        gc.collect()
        logger.debug("Memory optimization: garbage collection performed")
    
    @staticmethod
    def get_memory_usage() -> Dict[str, Any]:
        """
        Get current memory usage.
        
        Returns:
            Dictionary with memory usage information
        """
        if not PSUTIL_AVAILABLE:
            return {
                "available": False,
                "message": "psutil not available"
            }
        
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            return {
                "available": True,
                "rss_mb": memory_info.rss / (1024 * 1024),  # Resident Set Size
                "vms_mb": memory_info.vms / (1024 * 1024),  # Virtual Memory Size
                "percent": process.memory_percent(),
            }
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return {
                "available": False,
                "error": str(e)
            }
    
    @staticmethod
    def get_optimal_resolution(
        original_size: Tuple[int, int],
        scale_factor: float,
        max_dimension: Optional[int] = None,
        min_dimension: Optional[int] = None
    ) -> Tuple[int, int]:
        """
        Calculate optimal resolution for upscaling.
        
        Args:
            original_size: Original image size (width, height)
            scale_factor: Scale factor
            max_dimension: Maximum dimension (optional)
            min_dimension: Minimum dimension (optional)
            
        Returns:
            Optimal resolution (width, height)
        """
        width, height = original_size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Apply constraints
        if max_dimension:
            if new_width > max_dimension or new_height > max_dimension:
                # Scale down proportionally
                ratio = min(max_dimension / new_width, max_dimension / new_height)
                new_width = int(new_width * ratio)
                new_height = int(new_height * ratio)
        
        if min_dimension:
            if new_width < min_dimension or new_height < min_dimension:
                # Scale up proportionally
                ratio = max(min_dimension / new_width, min_dimension / new_height)
                new_width = int(new_width * ratio)
                new_height = int(new_height * ratio)
        
        return (new_width, new_height)


