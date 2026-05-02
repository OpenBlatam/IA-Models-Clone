"""
Advanced cache utilities.

Additional utility functions for cache operations.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, Callable
import torch

from kv_cache.types import TensorPair

logger = logging.getLogger(__name__)


class CacheInspector:
    """
    Cache inspector for debugging and analysis.
    
    Provides detailed inspection capabilities.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize cache inspector.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
    
    def inspect_entry(self, position: int) -> Dict[str, Any]:
        """
        Inspect a specific cache entry.
        
        Args:
            position: Cache position
            
        Returns:
            Dictionary with entry details
        """
        entry = self.cache.get(position)
        
        if entry is None:
            return {
                "position": position,
                "exists": False
            }
        
        key, value = entry
        
        return {
            "position": position,
            "exists": True,
            "key_shape": list(key.shape),
            "value_shape": list(value.shape),
            "key_dtype": str(key.dtype),
            "value_dtype": str(value.dtype),
            "key_device": str(key.device),
            "value_device": str(value.device),
            "key_memory_mb": key.numel() * key.element_size() / (1024 * 1024),
            "value_memory_mb": value.numel() * value.element_size() / (1024 * 1024),
            "has_nan": torch.isnan(key).any().item() or torch.isnan(value).any().item(),
            "has_inf": torch.isinf(key).any().item() or torch.isinf(value).any().item(),
            "key_min": key.min().item(),
            "key_max": key.max().item(),
            "value_min": value.min().item(),
            "value_max": value.max().item()
        }
    
    def inspect_cache(self, sample_size: int = 100) -> Dict[str, Any]:
        """
        Inspect cache state.
        
        Args:
            sample_size: Number of entries to sample
            
        Returns:
            Dictionary with cache inspection results
        """
        storage = self.cache.storage
        positions = storage.get_positions()
        
        if not positions:
            return {
                "total_entries": 0,
                "sampled_entries": 0,
                "sample": []
            }
        
        # Sample entries
        import random
        sampled_positions = random.sample(
            positions,
            min(sample_size, len(positions))
        )
        
        sample = [self.inspect_entry(pos) for pos in sampled_positions]
        
        return {
            "total_entries": len(positions),
            "sampled_entries": len(sampled_positions),
            "sample": sample,
            "total_memory_mb": storage.get_total_memory_mb()
        }
    
    def find_entries_by_pattern(
        self,
        pattern_fn: Callable[[TensorPair], bool],
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find entries matching a pattern.
        
        Args:
            pattern_fn: Function that returns True if entry matches pattern
            max_results: Maximum number of results
            
        Returns:
            List of matching entries
        """
        results = []
        positions = self.cache.storage.get_positions()
        
        for pos in positions:
            if len(results) >= max_results:
                break
            
            entry = self.cache.get(pos)
            if entry and pattern_fn(entry):
                results.append({
                    "position": pos,
                    "inspection": self.inspect_entry(pos)
                })
        
        return results


class CacheRepair:
    """
    Cache repair utilities.
    
    Provides repair and recovery capabilities.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize cache repair.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
    
    def repair_invalid_entries(self) -> Dict[str, Any]:
        """
        Repair invalid cache entries (NaN, Inf).
        
        Returns:
            Dictionary with repair results
        """
        repaired = 0
        removed = 0
        positions = self.cache.storage.get_positions()
        
        for pos in positions:
            entry = self.cache.get(pos)
            if entry:
                key, value = entry
                
                # Check for invalid values
                key_has_nan = torch.isnan(key).any()
                key_has_inf = torch.isinf(key).any()
                value_has_nan = torch.isnan(value).any()
                value_has_inf = torch.isinf(value).any()
                
                if key_has_nan or key_has_inf or value_has_nan or value_has_inf:
                    # Try to repair by zeroing invalid values
                    if key_has_nan or key_has_inf:
                        key = torch.where(torch.isnan(key) | torch.isinf(key), torch.zeros_like(key), key)
                        repaired += 1
                    
                    if value_has_nan or value_has_inf:
                        value = torch.where(torch.isnan(value) | torch.isinf(value), torch.zeros_like(value), value)
                        repaired += 1
                    
                    # Update entry
                    self.cache.put(pos, key, value)
        
        return {
            "repaired_entries": repaired,
            "removed_entries": removed,
            "total_checked": len(positions)
        }
    
    def cleanup_orphaned_entries(self) -> Dict[str, Any]:
        """
        Cleanup orphaned or invalid entries.
        
        Returns:
            Dictionary with cleanup results
        """
        removed = 0
        positions = self.cache.storage.get_positions()
        
        for pos in positions:
            entry = self.cache.get(pos)
            if entry is None:
                # Remove orphaned position
                self.cache.storage.remove([pos])
                removed += 1
        
        return {
            "removed_entries": removed,
            "total_checked": len(positions)
        }


class CacheMigrator:
    """
    Cache migration utilities.
    
    Provides migration between different cache configurations.
    """
    
    def __init__(self, source_cache: Any, target_cache: Any):
        """
        Initialize cache migrator.
        
        Args:
            source_cache: Source cache instance
            target_cache: Target cache instance
        """
        self.source_cache = source_cache
        self.target_cache = target_cache
    
    def migrate_entries(
        self,
        positions: Optional[List[int]] = None,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Migrate entries from source to target cache.
        
        Args:
            positions: List of positions to migrate (None = all)
            batch_size: Batch size for migration
            
        Returns:
            Dictionary with migration results
        """
        if positions is None:
            positions = self.source_cache.storage.get_positions()
        
        migrated = 0
        failed = 0
        
        for i in range(0, len(positions), batch_size):
            batch = positions[i:i + batch_size]
            
            for pos in batch:
                try:
                    entry = self.source_cache.get(pos)
                    if entry:
                        key, value = entry
                        self.target_cache.put(pos, key, value)
                        migrated += 1
                except Exception as e:
                    logger.warning(f"Failed to migrate position {pos}: {e}")
                    failed += 1
        
        return {
            "migrated_entries": migrated,
            "failed_entries": failed,
            "total_attempted": len(positions)
        }
    
    def migrate_stats(self) -> Dict[str, Any]:
        """
        Migrate statistics from source to target.
        
        Returns:
            Dictionary with stats migration results
        """
        source_stats = self.source_cache.get_stats()
        target_stats = self.target_cache.get_stats()
        
        # Note: Actual stats migration would depend on implementation
        return {
            "source_stats": source_stats,
            "target_stats": target_stats,
            "note": "Stats migration requires manual implementation"
        }

