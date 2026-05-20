"""
Batch optimization utilities for efficient processing.
"""

from typing import List, Callable, Optional, Dict, Any
from pathlib import Path
import numpy as np

from ..logger import logger
from .cache_utils import CacheManager
from .performance import PerformanceMonitor


class BatchOptimizer:
    """Optimize batch processing of audio files."""
    
    def __init__(
        self,
        cache_enabled: bool = True,
        cache_dir: Optional[Path] = None,
        max_workers: Optional[int] = None
    ):
        """
        Initialize batch optimizer.
        
        Args:
            cache_enabled: Enable caching
            cache_dir: Cache directory
            max_workers: Maximum workers for parallel processing
        """
        self.cache_enabled = cache_enabled
        self.cache = CacheManager(cache_dir) if cache_enabled else None
        self.max_workers = max_workers
        self.monitor = PerformanceMonitor()
    
    def optimize_batch(
        self,
        files: List[Path],
        process_func: Callable,
        skip_existing: bool = True
    ) -> Dict[str, Any]:
        """
        Optimize batch processing with caching and parallel execution.
        
        Args:
            files: List of files to process
            process_func: Function to process each file
            skip_existing: Skip files that already have cached results
            
        Returns:
            Dictionary mapping file paths to results
        """
        results = {}
        to_process = []
        
        # Check cache for existing results
        if self.cache_enabled and skip_existing:
            for file_path in files:
                cache_key = {
                    "file": str(file_path),
                    "func": process_func.__name__
                }
                
                if self.cache.exists(cache_key):
                    cached_result = self.cache.get(cache_key)
                    if cached_result:
                        results[str(file_path)] = cached_result
                        logger.debug(f"Using cached result for {file_path}")
                    else:
                        to_process.append(file_path)
                else:
                    to_process.append(file_path)
        else:
            to_process = files
        
        # Process remaining files
        if to_process:
            logger.info(f"Processing {len(to_process)} files")
            
            from .parallel_processing import process_parallel
            
            def process_with_cache(file_path: Path):
                self.monitor.start(str(file_path))
                try:
                    result = process_func(str(file_path))
                    
                    # Cache result
                    if self.cache_enabled:
                        cache_key = {
                            "file": str(file_path),
                            "func": process_func.__name__
                        }
                        self.cache.set(cache_key, result)
                    
                    self.monitor.stop(str(file_path))
                    return str(file_path), result
                except Exception as e:
                    self.monitor.stop(str(file_path))
                    return str(file_path), {"error": str(e)}
            
            processed = process_parallel(
                to_process,
                process_with_cache,
                max_workers=self.max_workers,
                show_progress=True
            )
            
            results.update(dict(processed))
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "performance": self.monitor.get_all_stats(),
            "cache_size": self.cache.get_size() if self.cache else 0
        }

