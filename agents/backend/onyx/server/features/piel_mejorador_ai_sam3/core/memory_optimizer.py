"""
Memory Optimizer for Piel Mejorador AI SAM3
===========================================

Optimizes memory usage for large file processing.
"""

import gc
import logging
import psutil
import os
from typing import Optional, Dict, Any
from pathlib import Path
import asyncio

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """
    Optimizes memory usage for large file processing.
    
    Features:
    - Memory monitoring
    - Automatic cleanup
    - Chunked processing
    - Memory pressure detection
    """
    
    def __init__(self, max_memory_percent: float = 80.0):
        """
        Initialize memory optimizer.
        
        Args:
            max_memory_percent: Maximum memory usage percentage before cleanup
        """
        self.max_memory_percent = max_memory_percent
        self.process = psutil.Process(os.getpid())
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get current memory usage.
        
        Returns:
            Dictionary with memory statistics
        """
        memory_info = self.process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            "process_memory_mb": memory_info.rss / (1024 * 1024),
            "process_memory_percent": self.process.memory_percent(),
            "system_memory_percent": system_memory.percent,
            "system_memory_available_mb": system_memory.available / (1024 * 1024),
            "system_memory_total_mb": system_memory.total / (1024 * 1024),
        }
    
    def is_memory_pressure(self) -> bool:
        """
        Check if system is under memory pressure.
        
        Returns:
            True if memory usage is high
        """
        memory_usage = self.get_memory_usage()
        return memory_usage["system_memory_percent"] > self.max_memory_percent
    
    async def optimize(self, force: bool = False):
        """
        Optimize memory usage.
        
        Args:
            force: Force cleanup even if not under pressure
        """
        if force or self.is_memory_pressure():
            logger.info("Optimizing memory...")
            
            # Force garbage collection
            collected = gc.collect()
            logger.debug(f"Garbage collected {collected} objects")
            
            # Clear Python caches if available
            try:
                import sys
                if hasattr(sys, 'intern'):
                    # Clear interned strings cache
                    pass
            except Exception:
                pass
            
            memory_after = self.get_memory_usage()
            logger.info(
                f"Memory optimization complete. "
                f"Process memory: {memory_after['process_memory_mb']:.2f}MB"
            )
    
    async def process_in_chunks(
        self,
        items: list,
        process_func: callable,
        chunk_size: int = 10,
        optimize_between_chunks: bool = True
    ):
        """
        Process items in chunks with memory optimization.
        
        Args:
            items: List of items to process
            process_func: Async function to process items
            chunk_size: Number of items per chunk
            optimize_between_chunks: Optimize memory between chunks
            
        Yields:
            Results from each chunk
        """
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            
            logger.debug(f"Processing chunk {i // chunk_size + 1}/{(len(items) + chunk_size - 1) // chunk_size}")
            
            # Process chunk
            results = await asyncio.gather(*[
                process_func(item) for item in chunk
            ], return_exceptions=True)
            
            yield results
            
            # Optimize memory between chunks
            if optimize_between_chunks:
                await self.optimize()
    
    def get_recommendations(self) -> List[str]:
        """
        Get memory optimization recommendations.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        memory_usage = self.get_memory_usage()
        
        if memory_usage["process_memory_percent"] > 70:
            recommendations.append(
                f"Process memory usage is high ({memory_usage['process_memory_percent']:.1f}%). "
                "Consider processing in smaller batches."
            )
        
        if memory_usage["system_memory_percent"] > 80:
            recommendations.append(
                f"System memory usage is high ({memory_usage['system_memory_percent']:.1f}%). "
                "Consider reducing concurrent tasks."
            )
        
        if memory_usage["system_memory_available_mb"] < 1000:
            recommendations.append(
                f"Low available memory ({memory_usage['system_memory_available_mb']:.0f}MB). "
                "Consider freeing up system resources."
            )
        
        return recommendations




