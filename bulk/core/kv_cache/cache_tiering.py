"""
Multi-tier caching system for KV cache.

This module provides tiered caching capabilities, allowing data
to be stored across multiple cache tiers with different characteristics.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict


class TierLevel(Enum):
    """Cache tier levels."""
    L1 = "l1"  # Fastest, smallest (e.g., in-memory)
    L2 = "l2"  # Medium speed, medium size (e.g., SSD)
    L3 = "l3"  # Slower, larger (e.g., disk)
    L4 = "l4"  # Slowest, largest (e.g., network/remote)


class TierPolicy(Enum):
    """Tier promotion/demotion policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns
    MANUAL = "manual"  # Manual promotion/demotion


@dataclass
class TierConfig:
    """Configuration for a cache tier."""
    tier_level: TierLevel
    max_size: int
    max_items: int
    ttl: Optional[float] = None
    policy: TierPolicy = TierPolicy.LRU
    promotion_threshold: int = 10  # Access count to promote
    demotion_threshold: int = 1  # Access count threshold for demotion


@dataclass
class TierStats:
    """Statistics for a cache tier."""
    tier_level: TierLevel
    current_size: int
    current_items: int
    hit_count: int
    miss_count: int
    promotion_count: int
    demotion_count: int
    eviction_count: int


class CacheTier:
    """A single cache tier."""
    
    def __init__(self, config: TierConfig, cache: Any):
        self.config = config
        self.cache = cache
        self._access_counts: Dict[str, int] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.Lock()
        
    def get(self, key: str) -> Any:
        """Get value from tier."""
        value = self.cache.get(key)
        if value is not None:
            with self._lock:
                self._access_counts[key] = self._access_counts.get(key, 0) + 1
                self._access_times[key] = time.time()
        return value
        
    def put(self, key: str, value: Any) -> bool:
        """Put value into tier."""
        result = self.cache.put(key, value)
        if result:
            with self._lock:
                self._access_counts[key] = 0
                self._access_times[key] = time.time()
        return result
        
    def delete(self, key: str) -> bool:
        """Delete value from tier."""
        with self._lock:
            self._access_counts.pop(key, None)
            self._access_times.pop(key, None)
        return self.cache.delete(key)
        
    def get_access_count(self, key: str) -> int:
        """Get access count for a key."""
        return self._access_counts.get(key, 0)
        
    def should_promote(self, key: str) -> bool:
        """Check if key should be promoted to higher tier."""
        return self._access_counts.get(key, 0) >= self.config.promotion_threshold
        
    def get_stats(self) -> TierStats:
        """Get tier statistics."""
        with self._lock:
            return TierStats(
                tier_level=self.config.tier_level,
                current_size=len(self.cache._cache) if hasattr(self.cache, '_cache') else 0,
                current_items=len(self.cache._cache) if hasattr(self.cache, '_cache') else 0,
                hit_count=0,  # Would be tracked in real implementation
                miss_count=0,
                promotion_count=0,
                demotion_count=0,
                eviction_count=0
            )


class MultiTierCache:
    """Multi-tier cache system."""
    
    def __init__(self, tier_configs: List[TierConfig], caches: List[Any]):
        if len(tier_configs) != len(caches):
            raise ValueError("Number of tier configs must match number of caches")
            
        self.tiers: List[CacheTier] = []
        for config, cache in zip(tier_configs, caches):
            self.tiers.append(CacheTier(config, cache))
            
        self._lock = threading.Lock()
        
    def get(self, key: str) -> Any:
        """Get value, checking tiers from fastest to slowest."""
        # Check L1 first, then L2, etc.
        for tier in self.tiers:
            value = tier.get(key)
            if value is not None:
                # Promote to higher tier if needed
                self._try_promote(key, value, tier)
                return value
        return None
        
    def put(self, key: str, value: Any, tier: Optional[TierLevel] = None) -> bool:
        """Put value into specified tier (defaults to L1)."""
        if tier:
            target_tier = self._get_tier(tier)
            if target_tier:
                return target_tier.put(key, value)
        else:
            # Default to L1
            if self.tiers:
                return self.tiers[0].put(key, value)
        return False
        
    def _try_promote(self, key: str, value: Any, current_tier: CacheTier) -> None:
        """Try to promote key to higher tier."""
        current_index = self.tiers.index(current_tier)
        if current_index > 0 and current_tier.should_promote(key):
            # Promote to next higher tier
            higher_tier = self.tiers[current_index - 1]
            higher_tier.put(key, value)
            
    def _get_tier(self, tier_level: TierLevel) -> Optional[CacheTier]:
        """Get tier by level."""
        for tier in self.tiers:
            if tier.config.tier_level == tier_level:
                return tier
        return None
        
    def promote(self, key: str, target_tier: TierLevel) -> bool:
        """Manually promote a key to a target tier."""
        # Find key in any tier
        value = None
        source_tier = None
        
        for tier in self.tiers:
            value = tier.get(key)
            if value is not None:
                source_tier = tier
                break
                
        if value is None:
            return False
            
        # Move to target tier
        target = self._get_tier(target_tier)
        if target:
            target.put(key, value)
            if source_tier:
                source_tier.delete(key)
            return True
        return False
        
    def demote(self, key: str, target_tier: TierLevel) -> bool:
        """Manually demote a key to a target tier."""
        return self.promote(key, target_tier)  # Same logic
        
    def get_tier_stats(self) -> Dict[TierLevel, TierStats]:
        """Get statistics for all tiers."""
        return {
            tier.config.tier_level: tier.get_stats()
            for tier in self.tiers
        }
        
    def clear_tier(self, tier_level: TierLevel) -> None:
        """Clear a specific tier."""
        tier = self._get_tier(tier_level)
        if tier:
            tier.cache.clear()
            
    def optimize_tiers(self) -> Dict[str, Any]:
        """Optimize tier distribution."""
        # Move frequently accessed items to higher tiers
        promotions = 0
        
        for i in range(len(self.tiers) - 1, 0, -1):  # Start from lowest tier
            tier = self.tiers[i]
            # Check all keys in this tier
            if hasattr(tier.cache, '_cache'):
                for key in list(tier.cache._cache.keys()):
                    if tier.should_promote(key):
                        value = tier.get(key)
                        if value is not None:
                            self.promote(key, self.tiers[i-1].config.tier_level)
                            promotions += 1
                            
        return {
            'promotions': promotions,
            'optimization_time': time.time()
        }














