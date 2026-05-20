"""
Statistics module for cache system
"""

import asyncio
from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class CacheStats:
    """Cache statistics with thread-safe updates and latency tracking"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    hit_rate: float = 0.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _latencies: List[float] = field(default_factory=list)
    _max_latency_samples: int = 1000
    
    def update_hit_rate(self):
        """Update hit rate"""
        if self.total_requests > 0:
            self.hit_rate = (self.hits / self.total_requests) * 100
    
    async def increment_hit(self):
        """Thread-safe hit increment"""
        async with self._lock:
            self.hits += 1
            self.total_requests += 1
            self.update_hit_rate()
    
    async def increment_miss(self):
        """Thread-safe miss increment"""
        async with self._lock:
            self.misses += 1
            self.total_requests += 1
            self.update_hit_rate()
    
    async def increment_eviction(self):
        """Thread-safe eviction increment"""
        async with self._lock:
            self.evictions += 1
    
    async def record_latency(self, latency_ms: float):
        """Record operation latency"""
        async with self._lock:
            self._latencies.append(latency_ms)
            if len(self._latencies) > self._max_latency_samples:
                self._latencies.pop(0)
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics"""
        if not self._latencies:
            return {
                "avg": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0
            }
        
        sorted_latencies = sorted(self._latencies)
        n = len(sorted_latencies)
        
        return {
            "avg": sum(sorted_latencies) / n,
            "min": sorted_latencies[0],
            "max": sorted_latencies[-1],
            "p50": sorted_latencies[int(n * 0.50)],
            "p95": sorted_latencies[int(n * 0.95)] if n > 0 else 0.0,
            "p99": sorted_latencies[int(n * 0.99)] if n > 0 else 0.0
        }

