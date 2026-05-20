"""
Advanced batch operations for KV cache.

This module provides optimized batch operations including
pipelining, batching, and parallel processing.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed


class BatchStrategy(Enum):
    """Batch processing strategies."""
    SEQUENTIAL = "sequential"  # Process one by one
    PARALLEL = "parallel"  # Process in parallel
    PIPELINE = "pipeline"  # Pipeline processing
    CHUNKED = "chunked"  # Process in chunks


@dataclass
class BatchOperation:
    """A batch operation."""
    operation_type: str  # "get", "put", "delete"
    key: str
    value: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Result of batch operation."""
    successful: int
    failed: int
    results: List[Tuple[str, Any]]  # (key, result)
    duration: float
    strategy_used: BatchStrategy


class AdvancedBatchProcessor:
    """Advanced batch processor for cache operations."""
    
    def __init__(self, cache: Any, max_workers: int = 4):
        self.cache = cache
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
    def batch_get(
        self,
        keys: List[str],
        strategy: BatchStrategy = BatchStrategy.PARALLEL
    ) -> BatchResult:
        """Batch get operation."""
        start_time = time.time()
        results = []
        successful = 0
        failed = 0
        
        if strategy == BatchStrategy.PARALLEL:
            results = self._parallel_get(keys)
        elif strategy == BatchStrategy.PIPELINE:
            results = self._pipeline_get(keys)
        elif strategy == BatchStrategy.CHUNKED:
            results = self._chunked_get(keys)
        else:
            results = self._sequential_get(keys)
            
        for key, value in results:
            if value is not None:
                successful += 1
            else:
                failed += 1
                
        duration = time.time() - start_time
        
        return BatchResult(
            successful=successful,
            failed=failed,
            results=results,
            duration=duration,
            strategy_used=strategy
        )
        
    def batch_put(
        self,
        items: Dict[str, Any],
        strategy: BatchStrategy = BatchStrategy.PARALLEL
    ) -> BatchResult:
        """Batch put operation."""
        start_time = time.time()
        results = []
        successful = 0
        failed = 0
        
        if strategy == BatchStrategy.PARALLEL:
            results = self._parallel_put(items)
        elif strategy == BatchStrategy.PIPELINE:
            results = self._pipeline_put(items)
        elif strategy == BatchStrategy.CHUNKED:
            results = self._chunked_put(items)
        else:
            results = self._sequential_put(items)
            
        for key, result in results:
            if result:
                successful += 1
            else:
                failed += 1
                
        duration = time.time() - start_time
        
        return BatchResult(
            successful=successful,
            failed=failed,
            results=results,
            duration=duration,
            strategy_used=strategy
        )
        
    def _sequential_get(self, keys: List[str]) -> List[Tuple[str, Any]]:
        """Sequential get operations."""
        results = []
        for key in keys:
            value = self.cache.get(key)
            results.append((key, value))
        return results
        
    def _parallel_get(self, keys: List[str]) -> List[Tuple[str, Any]]:
        """Parallel get operations."""
        results = []
        futures = {}
        
        for key in keys:
            future = self._executor.submit(self.cache.get, key)
            futures[future] = key
            
        for future in as_completed(futures):
            key = futures[future]
            try:
                value = future.result()
                results.append((key, value))
            except Exception:
                results.append((key, None))
                
        return results
        
    def _pipeline_get(self, keys: List[str]) -> List[Tuple[str, Any]]:
        """Pipelined get operations."""
        # Simplified pipeline - would optimize network calls
        return self._parallel_get(keys)
        
    def _chunked_get(self, keys: List[str], chunk_size: int = 100) -> List[Tuple[str, Any]]:
        """Chunked get operations."""
        results = []
        for i in range(0, len(keys), chunk_size):
            chunk = keys[i:i + chunk_size]
            chunk_results = self._parallel_get(chunk)
            results.extend(chunk_results)
        return results
        
    def _sequential_put(self, items: Dict[str, Any]) -> List[Tuple[str, bool]]:
        """Sequential put operations."""
        results = []
        for key, value in items.items():
            result = self.cache.put(key, value)
            results.append((key, result))
        return results
        
    def _parallel_put(self, items: Dict[str, Any]) -> List[Tuple[str, bool]]:
        """Parallel put operations."""
        results = []
        futures = {}
        
        for key, value in items.items():
            future = self._executor.submit(self.cache.put, key, value)
            futures[future] = key
            
        for future in as_completed(futures):
            key = futures[future]
            try:
                result = future.result()
                results.append((key, result))
            except Exception:
                results.append((key, False))
                
        return results
        
    def _pipeline_put(self, items: Dict[str, Any]) -> List[Tuple[str, bool]]:
        """Pipelined put operations."""
        return self._parallel_put(items)
        
    def _chunked_put(self, items: Dict[str, Any], chunk_size: int = 100) -> List[Tuple[str, bool]]:
        """Chunked put operations."""
        results = []
        items_list = list(items.items())
        for i in range(0, len(items_list), chunk_size):
            chunk = dict(items_list[i:i + chunk_size])
            chunk_results = self._parallel_put(chunk)
            results.extend(chunk_results)
        return results
        
    def batch_delete(
        self,
        keys: List[str],
        strategy: BatchStrategy = BatchStrategy.PARALLEL
    ) -> BatchResult:
        """Batch delete operation."""
        start_time = time.time()
        results = []
        successful = 0
        failed = 0
        
        if strategy == BatchStrategy.PARALLEL:
            futures = {self._executor.submit(self.cache.delete, key): key for key in keys}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    result = future.result()
                    results.append((key, result))
                    if result:
                        successful += 1
                    else:
                        failed += 1
                except Exception:
                    results.append((key, False))
                    failed += 1
        else:
            for key in keys:
                result = self.cache.delete(key)
                results.append((key, result))
                if result:
                    successful += 1
                else:
                    failed += 1
                    
        duration = time.time() - start_time
        
        return BatchResult(
            successful=successful,
            failed=failed,
            results=results,
            duration=duration,
            strategy_used=strategy
        )



