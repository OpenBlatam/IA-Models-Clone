"""
Cache sharding system.

Provides sharding capabilities for distributed cache.
"""
from __future__ import annotations

import logging
import hashlib
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ShardConfig:
    """Shard configuration."""
    num_shards: int = 4
    shard_key_func: Optional[Callable] = None


class CacheSharding:
    """
    Cache sharding manager.
    
    Provides sharding for distributed cache.
    """
    
    def __init__(
        self,
        shards: List[Any],
        config: Optional[ShardConfig] = None
    ):
        """
        Initialize sharding.
        
        Args:
            shards: List of cache shard instances
            config: Optional shard configuration
        """
        self.shards = shards
        self.config = config or ShardConfig(num_shards=len(shards))
        self.shard_key_func = self.config.shard_key_func or self._default_shard_key
    
    def _default_shard_key(self, key: Any) -> int:
        """
        Default shard key function.
        
        Args:
            key: Cache key
            
        Returns:
            Shard index
        """
        if isinstance(key, int):
            return key % len(self.shards)
        
        # Hash string keys
        key_str = str(key)
        hash_value = int(hashlib.md5(key_str.encode()).hexdigest(), 16)
        return hash_value % len(self.shards)
    
    def get_shard(self, key: Any) -> Any:
        """
        Get shard for key.
        
        Args:
            key: Cache key
            
        Returns:
            Shard instance
        """
        shard_index = self.shard_key_func(key)
        return self.shards[shard_index]
    
    def get(self, key: Any) -> Optional[Any]:
        """
        Get value from appropriate shard.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        shard = self.get_shard(key)
        return shard.get(key)
    
    def put(self, key: Any, value: Any) -> None:
        """
        Put value to appropriate shard.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        shard = self.get_shard(key)
        shard.put(key, value)
    
    def batch_get(self, keys: List[Any]) -> Dict[Any, Any]:
        """
        Get multiple values from shards.
        
        Args:
            keys: List of keys
            
        Returns:
            Dictionary of key-value pairs
        """
        results = {}
        
        # Group keys by shard
        shard_keys: Dict[int, List[Any]] = {}
        for key in keys:
            shard_index = self.shard_key_func(key)
            if shard_index not in shard_keys:
                shard_keys[shard_index] = []
            shard_keys[shard_index].append(key)
        
        # Get from each shard
        for shard_index, shard_keys_list in shard_keys.items():
            shard = self.shards[shard_index]
            for key in shard_keys_list:
                value = shard.get(key)
                if value is not None:
                    results[key] = value
        
        return results
    
    def batch_put(self, items: Dict[Any, Any]) -> None:
        """
        Put multiple values to shards.
        
        Args:
            items: Dictionary of key-value pairs
        """
        # Group items by shard
        shard_items: Dict[int, Dict[Any, Any]] = {}
        for key, value in items.items():
            shard_index = self.shard_key_func(key)
            if shard_index not in shard_items:
                shard_items[shard_index] = {}
            shard_items[shard_index][key] = value
        
        # Put to each shard
        for shard_index, shard_items_dict in shard_items.items():
            shard = self.shards[shard_index]
            for key, value in shard_items_dict.items():
                shard.put(key, value)
    
    def clear_all(self) -> None:
        """Clear all shards."""
        for shard in self.shards:
            shard.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics from all shards.
        
        Returns:
            Aggregated statistics
        """
        all_stats = [shard.get_stats() for shard in self.shards]
        
        # Aggregate stats
        total_hits = sum(stats.get("hits", 0) for stats in all_stats)
        total_misses = sum(stats.get("misses", 0) for stats in all_stats)
        total_size = sum(stats.get("cache_size", 0) for stats in all_stats)
        total_memory = sum(stats.get("memory_mb", 0.0) for stats in all_stats)
        
        return {
            "num_shards": len(self.shards),
            "total_hits": total_hits,
            "total_misses": total_misses,
            "total_cache_size": total_size,
            "total_memory_mb": total_memory,
            "hit_rate": total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0.0,
            "shard_stats": all_stats
        }


class ConsistentHashingSharding:
    """
    Consistent hashing sharding.
    
    Provides consistent hashing for shard distribution.
    """
    
    def __init__(self, shards: List[Any], virtual_nodes: int = 100):
        """
        Initialize consistent hashing.
        
        Args:
            shards: List of cache shard instances
            virtual_nodes: Number of virtual nodes per shard
        """
        self.shards = shards
        self.virtual_nodes = virtual_nodes
        self.ring: Dict[int, Any] = {}
        
        # Build consistent hash ring
        for shard_index, shard in enumerate(shards):
            for vnode in range(virtual_nodes):
                node_key = f"{shard_index}_{vnode}"
                hash_value = int(hashlib.md5(node_key.encode()).hexdigest(), 16)
                self.ring[hash_value] = shard
        
        # Sort ring
        self.sorted_keys = sorted(self.ring.keys())
    
    def _get_shard_for_key(self, key: Any) -> Any:
        """
        Get shard for key using consistent hashing.
        
        Args:
            key: Cache key
            
        Returns:
            Shard instance
        """
        key_str = str(key)
        key_hash = int(hashlib.md5(key_str.encode()).hexdigest(), 16)
        
        # Find first node >= key_hash
        for node_hash in self.sorted_keys:
            if node_hash >= key_hash:
                return self.ring[node_hash]
        
        # Wrap around to first node
        return self.ring[self.sorted_keys[0]]
    
    def get(self, key: Any) -> Optional[Any]:
        """Get value from shard."""
        shard = self._get_shard_for_key(key)
        return shard.get(key)
    
    def put(self, key: Any, value: Any) -> None:
        """Put value to shard."""
        shard = self._get_shard_for_key(key)
        shard.put(key, value)
    
    def add_shard(self, shard: Any) -> None:
        """
        Add new shard to ring.
        
        Args:
            shard: New shard instance
        """
        shard_index = len(self.shards)
        self.shards.append(shard)
        
        # Add virtual nodes
        for vnode in range(self.virtual_nodes):
            node_key = f"{shard_index}_{vnode}"
            hash_value = int(hashlib.md5(node_key.encode()).hexdigest(), 16)
            self.ring[hash_value] = shard
        
        # Rebuild sorted keys
        self.sorted_keys = sorted(self.ring.keys())
    
    def remove_shard(self, shard: Any) -> None:
        """
        Remove shard from ring.
        
        Args:
            shard: Shard instance to remove
        """
        if shard not in self.shards:
            return
        
        shard_index = self.shards.index(shard)
        
        # Remove virtual nodes
        keys_to_remove = []
        for node_hash, node_shard in self.ring.items():
            if node_shard == shard:
                keys_to_remove.append(node_hash)
        
        for key in keys_to_remove:
            del self.ring[key]
        
        # Remove from shards
        self.shards.remove(shard)
        
        # Rebuild sorted keys
        self.sorted_keys = sorted(self.ring.keys())

