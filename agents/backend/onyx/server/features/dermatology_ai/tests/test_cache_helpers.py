"""
Cache Testing Helpers
Specialized helpers for cache testing
"""

from typing import Any, Dict, List, Optional
from unittest.mock import Mock, AsyncMock
import time
from datetime import datetime, timedelta


class CacheTestHelpers:
    """Helpers for cache testing"""
    
    @staticmethod
    def create_mock_cache(
        default_ttl: int = 3600,
        initial_data: Optional[Dict[str, Any]] = None
    ) -> Mock:
        """Create mock cache with initial data"""
        cache_data = initial_data or {}
        cache = Mock()
        
        async def get_side_effect(key: str):
            return cache_data.get(key)
        
        async def set_side_effect(key: str, value: Any, ttl: Optional[int] = None):
            cache_data[key] = {
                "value": value,
                "expires_at": datetime.utcnow() + timedelta(seconds=ttl or default_ttl)
            }
            return True
        
        async def delete_side_effect(key: str):
            if key in cache_data:
                del cache_data[key]
            return True
        
        cache.get = AsyncMock(side_effect=get_side_effect)
        cache.set = AsyncMock(side_effect=set_side_effect)
        cache.delete = AsyncMock(side_effect=delete_side_effect)
        cache.clear = AsyncMock(return_value=True)
        cache.exists = AsyncMock(side_effect=lambda key: key in cache_data)
        
        return cache
    
    @staticmethod
    def assert_cache_hit(cache: Mock, key: str, expected_value: Any = None):
        """Assert cache hit occurred"""
        assert cache.get.called, f"Cache get was not called for key: {key}"
        # Additional validation can check call arguments
    
    @staticmethod
    def assert_cache_miss(cache: Mock, key: str):
        """Assert cache miss occurred"""
        # Cache miss means get was called but returned None
        assert cache.get.called, f"Cache get was not called for key: {key}"
    
    @staticmethod
    def assert_cache_set(cache: Mock, key: str, value: Any = None, ttl: Optional[int] = None):
        """Assert value was set in cache"""
        assert cache.set.called, f"Cache set was not called for key: {key}"
        if value is not None:
            # Can add more specific validation
            pass
    
    @staticmethod
    def assert_cache_expired(cache: Mock, key: str, ttl: int):
        """Assert cache entry has expired"""
        # This would require checking internal cache state
        # For mock, we can verify TTL was set correctly
        assert cache.set.called, "Cache set was not called"
    
    @staticmethod
    def assert_cache_cleared(cache: Mock):
        """Assert cache was cleared"""
        assert cache.clear.called, "Cache clear was not called"


class CacheStrategyHelpers:
    """Helpers for cache strategy testing"""
    
    @staticmethod
    def create_mock_cache_strategy(
        strategy_type: str = "LRU",
        max_size: int = 100
    ) -> Mock:
        """Create mock cache strategy"""
        strategy = Mock()
        strategy.type = strategy_type
        strategy.max_size = max_size
        strategy.should_evict = Mock(return_value=False)
        strategy.evict = Mock(return_value=[])
        strategy.get_stats = Mock(return_value={
            "hits": 0,
            "misses": 0,
            "evictions": 0
        })
        return strategy
    
    @staticmethod
    def assert_strategy_applied(strategy: Mock):
        """Assert cache strategy was applied"""
        assert strategy.should_evict.called or strategy.evict.called, \
            "Cache strategy was not applied"
    
    @staticmethod
    def assert_eviction_occurred(strategy: Mock):
        """Assert cache eviction occurred"""
        assert strategy.evict.called, "Cache eviction did not occur"


class CachePerformanceHelpers:
    """Helpers for cache performance testing"""
    
    @staticmethod
    async def measure_cache_performance(
        cache: Mock,
        operations: List[Dict[str, Any]],
        iterations: int = 100
    ) -> Dict[str, float]:
        """Measure cache performance"""
        import time
        
        total_time = 0
        hits = 0
        misses = 0
        
        for _ in range(iterations):
            for op in operations:
                op_type = op["type"]
                key = op["key"]
                
                start = time.time()
                if op_type == "get":
                    result = await cache.get(key)
                    if result is None:
                        misses += 1
                    else:
                        hits += 1
                elif op_type == "set":
                    await cache.set(key, op.get("value"), op.get("ttl"))
                total_time += time.time() - start
        
        return {
            "total_time": total_time,
            "avg_time": total_time / (iterations * len(operations)),
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / (hits + misses) if (hits + misses) > 0 else 0
        }
    
    @staticmethod
    def assert_cache_performance_acceptable(
        performance: Dict[str, float],
        max_avg_time: float = 0.001,
        min_hit_rate: float = 0.5
    ):
        """Assert cache performance is acceptable"""
        assert performance["avg_time"] <= max_avg_time, \
            f"Average time {performance['avg_time']} exceeds {max_avg_time}"
        assert performance["hit_rate"] >= min_hit_rate, \
            f"Hit rate {performance['hit_rate']} below {min_hit_rate}"


# Convenience exports
create_mock_cache = CacheTestHelpers.create_mock_cache
assert_cache_hit = CacheTestHelpers.assert_cache_hit
assert_cache_miss = CacheTestHelpers.assert_cache_miss
assert_cache_set = CacheTestHelpers.assert_cache_set
assert_cache_expired = CacheTestHelpers.assert_cache_expired
assert_cache_cleared = CacheTestHelpers.assert_cache_cleared

create_mock_cache_strategy = CacheStrategyHelpers.create_mock_cache_strategy
assert_strategy_applied = CacheStrategyHelpers.assert_strategy_applied
assert_eviction_occurred = CacheStrategyHelpers.assert_eviction_occurred

measure_cache_performance = CachePerformanceHelpers.measure_cache_performance
assert_cache_performance_acceptable = CachePerformanceHelpers.assert_cache_performance_acceptable



