"""
Smart Caching System - Multi-level cache with intelligent invalidation
Optimized for speed and memory efficiency
"""

import asyncio
import time
from typing import Any, Dict, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import hashlib
import json

class CacheStrategy(Enum):
    """Cache invalidation strategies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # ML-based

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float] = None
    size_bytes: int = 0

class SmartCache:
    """Intelligent multi-level cache system"""
    
    def __init__(
        self,
        max_size: int = 1000,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        default_ttl: Optional[float] = None,
        compression: bool = True
    ):
        self.max_size = max_size
        self.strategy = strategy
        self.default_ttl = default_ttl
        self.compression = compression
        
        # Multi-level cache
        self.l1_cache: Dict[str, CacheEntry] = {}  # Hot data
        self.l2_cache: Dict[str, CacheEntry] = {}  # Warm data
        self.l3_cache: Dict[str, CacheEntry] = {}  # Cold data
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        """Start background cleanup task"""
        if not self._running:
            self._running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self):
        """Stop background tasks"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    def _generate_key(self, key: Union[str, tuple, dict]) -> str:
        """Generate consistent cache key"""
        if isinstance(key, str):
            return key
        elif isinstance(key, (tuple, list)):
            return hashlib.md5(str(key).encode()).hexdigest()
        elif isinstance(key, dict):
            return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()
        else:
            return str(key)

    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size in bytes"""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (int, float, bool)):
                return 8
            else:
                return len(str(value).encode())
        except:
            return 1000  # Default estimate

    def _should_evict(self, entry: CacheEntry) -> bool:
        """Check if entry should be evicted"""
        if entry.ttl and time.time() - entry.created_at > entry.ttl:
            return True
        
        # Strategy-based eviction
        if self.strategy == CacheStrategy.LRU:
            return time.time() - entry.last_accessed > 3600  # 1 hour
        elif self.strategy == CacheStrategy.LFU:
            return entry.access_count < 2
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # ML-based: consider access pattern, recency, frequency
            age = time.time() - entry.created_at
            frequency_score = entry.access_count / max(age, 1)
            return frequency_score < 0.1 and age > 1800  # 30 minutes
        
        return False

    def _evict_entries(self, target_size: int):
        """Evict entries based on strategy"""
        all_entries = []
        
        # Collect all entries with metadata
        for level, cache in [("L1", self.l1_cache), ("L2", self.l2_cache), ("L3", self.l3_cache)]:
            for key, entry in cache.items():
                all_entries.append((level, key, entry))
        
        # Sort by eviction priority
        if self.strategy == CacheStrategy.LRU:
            all_entries.sort(key=lambda x: x[2].last_accessed)
        elif self.strategy == CacheStrategy.LFU:
            all_entries.sort(key=lambda x: x[2].access_count)
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Multi-factor scoring
            def score(entry_data):
                level, key, entry = entry_data
                age = time.time() - entry.created_at
                frequency = entry.access_count / max(age, 1)
                recency = 1 / (time.time() - entry.last_accessed + 1)
                level_weight = {"L1": 3, "L2": 2, "L3": 1}[level]
                return frequency * recency * level_weight
            
            all_entries.sort(key=score)
        
        # Evict until target size
        current_size = len(all_entries)
        for level, key, entry in all_entries:
            if current_size <= target_size:
                break
            
            if level == "L1":
                del self.l1_cache[key]
            elif level == "L2":
                del self.l2_cache[key]
            elif level == "L3":
                del self.l3_cache[key]
            
            current_size -= 1
            self.evictions += 1

    async def get(self, key: Union[str, tuple, dict]) -> Optional[Any]:
        """Get value from cache with multi-level lookup"""
        cache_key = self._generate_key(key)
        current_time = time.time()
        
        # L1 cache (hottest)
        if cache_key in self.l1_cache:
            entry = self.l1_cache[cache_key]
            if not self._should_evict(entry):
                entry.last_accessed = current_time
                entry.access_count += 1
                self.hits += 1
                return entry.value
        
        # L2 cache (warm)
        if cache_key in self.l2_cache:
            entry = self.l2_cache[cache_key]
            if not self._should_evict(entry):
                # Promote to L1
                entry.last_accessed = current_time
                entry.access_count += 1
                self.l1_cache[cache_key] = entry
                del self.l2_cache[cache_key]
                self.hits += 1
                return entry.value
        
        # L3 cache (cold)
        if cache_key in self.l3_cache:
            entry = self.l3_cache[cache_key]
            if not self._should_evict(entry):
                # Promote to L2
                entry.last_accessed = current_time
                entry.access_count += 1
                self.l2_cache[cache_key] = entry
                del self.l3_cache[cache_key]
                self.hits += 1
                return entry.value
        
        self.misses += 1
        return None

    async def set(
        self, 
        key: Union[str, tuple, dict], 
        value: Any, 
        ttl: Optional[float] = None
    ) -> None:
        """Set value in cache with intelligent placement"""
        cache_key = self._generate_key(key)
        current_time = time.time()
        
        # Calculate size
        size_bytes = self._calculate_size(value)
        
        # Create entry
        entry = CacheEntry(
            value=value,
            created_at=current_time,
            last_accessed=current_time,
            access_count=1,
            ttl=ttl or self.default_ttl,
            size_bytes=size_bytes
        )
        
        # Check if we need to evict
        total_size = len(self.l1_cache) + len(self.l2_cache) + len(self.l3_cache)
        if total_size >= self.max_size:
            self._evict_entries(int(self.max_size * 0.8))  # Evict to 80%
        
        # Place in appropriate level based on access pattern
        # For now, place new entries in L1
        self.l1_cache[cache_key] = entry

    async def delete(self, key: Union[str, tuple, dict]) -> bool:
        """Delete value from cache"""
        cache_key = self._generate_key(key)
        
        for cache in [self.l1_cache, self.l2_cache, self.l3_cache]:
            if cache_key in cache:
                del cache[cache_key]
                return True
        
        return False

    async def clear(self) -> None:
        """Clear all caches"""
        self.l1_cache.clear()
        self.l2_cache.clear()
        self.l3_cache.clear()

    async def _cleanup_loop(self):
        """Background cleanup task"""
        while self._running:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Clean expired entries
                for cache in [self.l1_cache, self.l2_cache, self.l3_cache]:
                    expired_keys = [
                        key for key, entry in cache.items()
                        if self._should_evict(entry)
                    ]
                    for key in expired_keys:
                        del cache[key]
                        self.evictions += 1
                
                # Rebalance cache levels
                await self._rebalance_caches()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Cache cleanup error: {e}")

    async def _rebalance_caches(self):
        """Rebalance entries between cache levels"""
        # Move cold entries from L1 to L2
        l1_to_l2 = []
        for key, entry in self.l1_cache.items():
            if entry.access_count < 3 and time.time() - entry.last_accessed > 300:  # 5 minutes
                l1_to_l2.append((key, entry))
        
        for key, entry in l1_to_l2:
            self.l2_cache[key] = entry
            del self.l1_cache[key]
        
        # Move cold entries from L2 to L3
        l2_to_l3 = []
        for key, entry in self.l2_cache.items():
            if entry.access_count < 2 and time.time() - entry.last_accessed > 1800:  # 30 minutes
                l2_to_l3.append((key, entry))
        
        for key, entry in l2_to_l3:
            self.l3_cache[key] = entry
            del self.l2_cache[key]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "l1_size": len(self.l1_cache),
            "l2_size": len(self.l2_cache),
            "l3_size": len(self.l3_cache),
            "total_size": len(self.l1_cache) + len(self.l2_cache) + len(self.l3_cache),
            "strategy": self.strategy.value
        }

class CacheOptimizer:
    """Cache optimization and analytics"""
    
    def __init__(self, cache: SmartCache):
        self.cache = cache
        self.access_patterns: Dict[str, list] = {}
        self.optimization_suggestions: list = []

    def track_access(self, key: str, access_time: float):
        """Track access patterns for optimization"""
        if key not in self.access_patterns:
            self.access_patterns[key] = []
        
        self.access_patterns[key].append(access_time)
        
        # Keep only recent accesses (last 1000)
        if len(self.access_patterns[key]) > 1000:
            self.access_patterns[key] = self.access_patterns[key][-1000:]

    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze access patterns and suggest optimizations"""
        analysis = {
            "hot_keys": [],
            "cold_keys": [],
            "burst_patterns": [],
            "suggestions": []
        }
        
        current_time = time.time()
        
        for key, accesses in self.access_patterns.items():
            if not accesses:
                continue
            
            # Calculate metrics
            recent_accesses = [t for t in accesses if current_time - t < 3600]  # Last hour
            frequency = len(recent_accesses)
            avg_interval = 0
            
            if len(accesses) > 1:
                intervals = [accesses[i] - accesses[i-1] for i in range(1, len(accesses))]
                avg_interval = sum(intervals) / len(intervals)
            
            # Categorize keys
            if frequency > 10:  # Hot key
                analysis["hot_keys"].append({
                    "key": key,
                    "frequency": frequency,
                    "avg_interval": avg_interval
                })
            elif frequency < 2:  # Cold key
                analysis["cold_keys"].append({
                    "key": key,
                    "frequency": frequency,
                    "last_access": max(accesses) if accesses else 0
                })
            
            # Detect burst patterns
            if len(recent_accesses) > 5:
                burst_intervals = [recent_accesses[i] - recent_accesses[i-1] for i in range(1, len(recent_accesses))]
                if all(interval < 60 for interval in burst_intervals[-5:]):  # 5 accesses within 5 minutes
                    analysis["burst_patterns"].append({
                        "key": key,
                        "burst_count": len(recent_accesses),
                        "duration": recent_accesses[-1] - recent_accesses[0]
                    })
        
        # Generate suggestions
        if analysis["hot_keys"]:
            analysis["suggestions"].append("Consider increasing L1 cache size for hot keys")
        
        if analysis["cold_keys"]:
            analysis["suggestions"].append("Consider reducing TTL for cold keys to save memory")
        
        if analysis["burst_patterns"]:
            analysis["suggestions"].append("Implement preloading for burst pattern keys")
        
        return analysis

    async def optimize_cache(self) -> None:
        """Apply optimization suggestions"""
        analysis = self.analyze_patterns()
        
        # Implement optimizations based on analysis
        for suggestion in analysis["suggestions"]:
            if "increase L1 cache size" in suggestion:
                # This would require cache reconfiguration
                pass
            elif "reduce TTL" in suggestion:
                # Reduce TTL for cold keys
                for cold_key in analysis["cold_keys"]:
                    key = cold_key["key"]
                    if key in self.cache.l3_cache:
                        entry = self.cache.l3_cache[key]
                        entry.ttl = min(entry.ttl or 3600, 300)  # Max 5 minutes
            elif "preloading" in suggestion:
                # Implement preloading for burst patterns
                for burst in analysis["burst_patterns"]:
                    # This would require application-specific logic
                    pass





