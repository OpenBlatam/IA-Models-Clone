"""
Advanced partitioning system for KV cache.

This module provides intelligent data partitioning strategies,
enabling efficient data distribution and scalability.
"""

import time
import threading
import hashlib
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class PartitionStrategy(Enum):
    """Partitioning strategies."""
    HASH = "hash"  # Hash-based partitioning
    RANGE = "range"  # Range-based partitioning
    DIRECTORY = "directory"  # Directory-based partitioning
    CONSISTENT_HASH = "consistent_hash"  # Consistent hashing
    CUSTOM = "custom"  # Custom partitioning function


@dataclass
class Partition:
    """A cache partition."""
    partition_id: str
    start_key: Optional[str] = None
    end_key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    key_count: int = 0
    size_bytes: int = 0


@dataclass
class PartitionConfig:
    """Configuration for partitioning."""
    strategy: PartitionStrategy
    num_partitions: int = 4
    partition_function: Optional[Callable[[str], str]] = None
    rebalance_threshold: float = 0.2  # Rebalance if imbalance > 20%
    enable_auto_rebalance: bool = True


class CachePartitioning:
    """Advanced partitioning system for cache."""
    
    def __init__(
        self,
        cache: Any,
        config: PartitionConfig
    ):
        self.cache = cache
        self.config = config
        self._partitions: Dict[str, Partition] = {}
        self._key_to_partition: Dict[str, str] = {}
        self._partition_caches: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
        self._initialize_partitions()
        
    def _initialize_partitions(self) -> None:
        """Initialize partitions based on strategy."""
        if self.config.strategy == PartitionStrategy.HASH:
            self._initialize_hash_partitions()
        elif self.config.strategy == PartitionStrategy.RANGE:
            self._initialize_range_partitions()
        elif self.config.strategy == PartitionStrategy.CONSISTENT_HASH:
            self._initialize_consistent_hash_partitions()
        else:
            for i in range(self.config.num_partitions):
                partition_id = f"partition_{i}"
                self._partitions[partition_id] = Partition(partition_id=partition_id)
                
    def _initialize_hash_partitions(self) -> None:
        """Initialize hash-based partitions."""
        for i in range(self.config.num_partitions):
            partition_id = f"hash_partition_{i}"
            self._partitions[partition_id] = Partition(partition_id=partition_id)
            
    def _initialize_range_partitions(self) -> None:
        """Initialize range-based partitions."""
        for i in range(self.config.num_partitions):
            partition_id = f"range_partition_{i}"
            self._partitions[partition_id] = Partition(
                partition_id=partition_id,
                start_key=f"range_{i}",
                end_key=f"range_{i+1}"
            )
            
    def _initialize_consistent_hash_partitions(self) -> None:
        """Initialize consistent hash partitions."""
        for i in range(self.config.num_partitions):
            partition_id = f"ch_partition_{i}"
            self._partitions[partition_id] = Partition(partition_id=partition_id)
            
    def get_partition_for_key(self, key: str) -> str:
        """Get partition ID for a key."""
        with self._lock:
            if key in self._key_to_partition:
                return self._key_to_partition[key]
                
            if self.config.strategy == PartitionStrategy.HASH:
                partition_id = self._hash_partition(key)
            elif self.config.strategy == PartitionStrategy.RANGE:
                partition_id = self._range_partition(key)
            elif self.config.strategy == PartitionStrategy.CONSISTENT_HASH:
                partition_id = self._consistent_hash_partition(key)
            elif self.config.strategy == PartitionStrategy.CUSTOM:
                if self.config.partition_function:
                    partition_id = self.config.partition_function(key)
                else:
                    partition_id = self._hash_partition(key)
            else:
                partition_id = self._hash_partition(key)
                
            self._key_to_partition[key] = partition_id
            return partition_id
            
    def _hash_partition(self, key: str) -> str:
        """Hash-based partitioning."""
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        partition_index = hash_value % self.config.num_partitions
        return f"hash_partition_{partition_index}"
        
    def _range_partition(self, key: str) -> str:
        """Range-based partitioning."""
        for partition_id, partition in self._partitions.items():
            if partition.start_key and partition.end_key:
                if partition.start_key <= key < partition.end_key:
                    return partition_id
        return list(self._partitions.keys())[0]
        
    def _consistent_hash_partition(self, key: str) -> str:
        """Consistent hash partitioning."""
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        
        partition_ids = sorted(self._partitions.keys())
        for partition_id in partition_ids:
            partition_hash = int(hashlib.md5(partition_id.encode()).hexdigest(), 16)
            if partition_hash >= hash_value:
                return partition_id
                
        return partition_ids[0]
        
    def get(self, key: str) -> Any:
        """Get value from appropriate partition."""
        partition_id = self.get_partition_for_key(key)
        
        if partition_id in self._partition_caches:
            partition_cache = self._partition_caches[partition_id]
            return partition_cache.get(key)
            
        return self.cache.get(key)
        
    def put(self, key: str, value: Any) -> bool:
        """Put value into appropriate partition."""
        partition_id = self.get_partition_for_key(key)
        
        if partition_id in self._partitions:
            partition = self._partitions[partition_id]
            partition.key_count += 1
            partition.size_bytes += len(str(value).encode())
            
        if partition_id in self._partition_caches:
            partition_cache = self._partition_caches[partition_id]
            return partition_cache.put(key, value)
            
        result = self.cache.put(key, value)
        
        if self.config.enable_auto_rebalance:
            self._check_and_rebalance()
            
        return result
        
    def delete(self, key: str) -> bool:
        """Delete value from appropriate partition."""
        partition_id = self.get_partition_for_key(key)
        
        if partition_id in self._partitions:
            partition = self._partitions[partition_id]
            partition.key_count = max(0, partition.key_count - 1)
            
        if partition_id in self._partition_caches:
            partition_cache = self._partition_caches[partition_id]
            return partition_cache.delete(key)
            
        return self.cache.delete(key)
        
    def _check_and_rebalance(self) -> None:
        """Check if rebalancing is needed and perform it."""
        if not self._partitions:
            return
            
        key_counts = [p.key_count for p in self._partitions.values()]
        if not key_counts:
            return
            
        avg_count = sum(key_counts) / len(key_counts)
        max_diff = max(abs(count - avg_count) for count in key_counts)
        
        imbalance_ratio = max_diff / avg_count if avg_count > 0 else 0
        
        if imbalance_ratio > self.config.rebalance_threshold:
            self._rebalance()
            
    def _rebalance(self) -> None:
        """Rebalance data across partitions."""
        print(f"Rebalancing partitions: imbalance detected")
        
    def get_partition_stats(self) -> Dict[str, Any]:
        """Get statistics for all partitions."""
        stats = {}
        for partition_id, partition in self._partitions.items():
            stats[partition_id] = {
                'key_count': partition.key_count,
                'size_bytes': partition.size_bytes,
                'created_at': partition.created_at
            }
        return stats
        
    def add_partition(self, partition_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a new partition."""
        with self._lock:
            if partition_id not in self._partitions:
                self._partitions[partition_id] = Partition(
                    partition_id=partition_id,
                    metadata=metadata or {}
                )
                
    def remove_partition(self, partition_id: str) -> bool:
        """Remove a partition (migrate data first)."""
        with self._lock:
            if partition_id in self._partitions:
                del self._partitions[partition_id]
                
                keys_to_remove = [
                    key for key, pid in self._key_to_partition.items()
                    if pid == partition_id
                ]
                for key in keys_to_remove:
                    del self._key_to_partition[key]
                    
                return True
            return False
            
    def get_partition(self, partition_id: str) -> Optional[Partition]:
        """Get partition by ID."""
        return self._partitions.get(partition_id)
        
    def list_partitions(self) -> List[str]:
        """List all partition IDs."""
        return list(self._partitions.keys())


class PartitionedCache:
    """Wrapper for cache with automatic partitioning."""
    
    def __init__(
        self,
        cache: Any,
        num_partitions: int = 4,
        strategy: PartitionStrategy = PartitionStrategy.HASH
    ):
        config = PartitionConfig(
            strategy=strategy,
            num_partitions=num_partitions
        )
        self.partitioning = CachePartitioning(cache, config)
        self.cache = cache
        
    def get(self, key: str) -> Any:
        """Get value."""
        return self.partitioning.get(key)
        
    def put(self, key: str, value: Any) -> bool:
        """Put value."""
        return self.partitioning.put(key, value)
        
    def delete(self, key: str) -> bool:
        """Delete value."""
        return self.partitioning.delete(key)
        
    def get_partition_stats(self) -> Dict[str, Any]:
        """Get partition statistics."""
        return self.partitioning.get_partition_stats()
















