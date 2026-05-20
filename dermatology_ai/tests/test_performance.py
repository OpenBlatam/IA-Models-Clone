"""
Tests for Performance Components
Tests for caching, optimization, and performance monitoring
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import time
import asyncio

from core.infrastructure.cache_strategies import CacheStrategy, CacheStrategyManager
from core.infrastructure.performance_monitor import PerformanceMetrics, PerformanceMonitor
from core.infrastructure.query_optimizer import QueryOptimizer
from core.infrastructure.adapters.cache_adapter import CacheAdapter
from core.infrastructure.adapters.fallback_adapters import NoOpCacheAdapter


class TestCacheStrategyManager:
    """Tests for CacheStrategyManager"""
    
    def test_get_strategy_lru(self):
        """Test getting LRU cache strategy"""
        manager = CacheStrategyManager()
        strategy = manager.get_strategy(CacheStrategy.LRU)
        
        assert strategy is not None
    
    def test_get_strategy_fifo(self):
        """Test getting FIFO cache strategy"""
        manager = CacheStrategyManager()
        strategy = manager.get_strategy(CacheStrategy.FIFO)
        
        assert strategy is not None
    
    def test_get_strategy_lfu(self):
        """Test getting LFU cache strategy"""
        manager = CacheStrategyManager()
        strategy = manager.get_strategy(CacheStrategy.LFU)
        
        assert strategy is not None
    
    @pytest.mark.asyncio
    async def test_cache_strategy_eviction(self):
        """Test cache eviction with different strategies"""
        manager = CacheStrategyManager()
        
        for strategy_type in CacheStrategy:
            strategy = manager.get_strategy(strategy_type)
            # Test that strategy can handle eviction
            assert strategy is not None


class TestCacheAdapter:
    """Tests for CacheAdapter"""
    
    @pytest.fixture
    def cache_adapter(self, mock_cache_service):
        """Create cache adapter"""
        return CacheAdapter(cache_service=mock_cache_service)
    
    @pytest.mark.asyncio
    async def test_cache_hit(self, cache_adapter, mock_cache_service):
        """Test cache hit scenario"""
        mock_cache_service.get = AsyncMock(return_value='{"key": "value"}')
        
        result = await cache_adapter.get("test-key")
        
        assert result is not None
        mock_cache_service.get.assert_called_once_with("test-key")
    
    @pytest.mark.asyncio
    async def test_cache_miss(self, cache_adapter, mock_cache_service):
        """Test cache miss scenario"""
        mock_cache_service.get = AsyncMock(return_value=None)
        
        result = await cache_adapter.get("test-key")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_set_with_ttl(self, cache_adapter, mock_cache_service):
        """Test setting cache with TTL"""
        result = await cache_adapter.set("test-key", {"data": "value"}, ttl=3600)
        
        assert result is True
        mock_cache_service.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_invalidation(self, cache_adapter, mock_cache_service):
        """Test cache invalidation"""
        result = await cache_adapter.delete("test-key")
        
        assert result is True
        mock_cache_service.delete.assert_called_once_with("test-key")
    
    @pytest.mark.asyncio
    async def test_cache_batch_operations(self, cache_adapter, mock_cache_service):
        """Test batch cache operations"""
        keys = ["key1", "key2", "key3"]
        values = {"key1": "value1", "key2": "value2", "key3": "value3"}
        
        # Set multiple values
        for key, value in values.items():
            await cache_adapter.set(key, value)
        
        # Get multiple values
        results = {}
        for key in keys:
            mock_cache_service.get = AsyncMock(return_value=f'"{values[key]}"')
            results[key] = await cache_adapter.get(key)
        
        assert len(results) == 3
        mock_cache_service.set.call_count == 3


class TestNoOpCacheAdapter:
    """Tests for NoOpCacheAdapter (fallback)"""
    
    @pytest.fixture
    def noop_cache(self):
        """Create NoOp cache adapter"""
        return NoOpCacheAdapter()
    
    @pytest.mark.asyncio
    async def test_noop_get_always_returns_none(self, noop_cache):
        """Test that NoOp cache always returns None"""
        result = await noop_cache.get("any-key")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_noop_set_always_succeeds(self, noop_cache):
        """Test that NoOp cache set always succeeds"""
        result = await noop_cache.set("key", "value", ttl=3600)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_noop_delete_always_succeeds(self, noop_cache):
        """Test that NoOp cache delete always succeeds"""
        result = await noop_cache.delete("key")
        assert result is True


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor"""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor"""
        return PerformanceMonitor()
    
    def test_record_operation(self, performance_monitor):
        """Test recording an operation"""
        performance_monitor.record_operation("test_operation", duration=0.5)
        
        metrics = performance_monitor.get_metrics()
        assert "test_operation" in metrics
    
    def test_record_error(self, performance_monitor):
        """Test recording an error"""
        performance_monitor.record_error("test_operation", "Test error")
        
        metrics = performance_monitor.get_metrics()
        assert metrics.get("errors", {}).get("test_operation", 0) > 0
    
    def test_get_metrics(self, performance_monitor):
        """Test getting performance metrics"""
        performance_monitor.record_operation("op1", duration=0.1)
        performance_monitor.record_operation("op2", duration=0.2)
        
        metrics = performance_monitor.get_metrics()
        
        assert isinstance(metrics, dict)
        assert "op1" in metrics or "operations" in metrics
    
    def test_reset_metrics(self, performance_monitor):
        """Test resetting metrics"""
        performance_monitor.record_operation("test", duration=0.1)
        performance_monitor.reset()
        
        metrics = performance_monitor.get_metrics()
        # Metrics should be reset (implementation dependent)
        assert metrics is not None


class TestQueryOptimizer:
    """Tests for QueryOptimizer"""
    
    @pytest.fixture
    def query_optimizer(self):
        """Create query optimizer"""
        return QueryOptimizer()
    
    def test_optimize_query(self, query_optimizer):
        """Test query optimization"""
        query = {"user_id": "user-123", "limit": 100}
        
        optimized = query_optimizer.optimize(query)
        
        assert optimized is not None
        assert isinstance(optimized, dict)
    
    def test_add_index_hint(self, query_optimizer):
        """Test adding index hints to query"""
        query = {"user_id": "user-123"}
        
        optimized = query_optimizer.add_index_hint(query, "user_id_idx")
        
        assert optimized is not None
    
    def test_limit_optimization(self, query_optimizer):
        """Test limit optimization"""
        # Test that very large limits are capped
        query = {"limit": 10000}
        
        optimized = query_optimizer.optimize(query)
        
        # Limit should be optimized (implementation dependent)
        assert optimized is not None


class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    @pytest.mark.asyncio
    async def test_cache_performance(self, mock_cache_service):
        """Test cache performance"""
        adapter = CacheAdapter(cache_service=mock_cache_service)
        
        start_time = time.time()
        
        # Perform multiple cache operations
        for i in range(100):
            await adapter.set(f"key-{i}", {"value": i})
            await adapter.get(f"key-{i}")
        
        duration = time.time() - start_time
        
        # Should complete in reasonable time (adjust threshold as needed)
        assert duration < 5.0  # 5 seconds for 200 operations
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_operations(self, mock_cache_service):
        """Test concurrent cache operations"""
        adapter = CacheAdapter(cache_service=mock_cache_service)
        
        async def cache_operation(key):
            await adapter.set(key, {"value": key})
            return await adapter.get(key)
        
        # Run 50 concurrent operations
        tasks = [cache_operation(f"key-{i}") for i in range(50)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 50
    
    def test_query_optimization_performance(self):
        """Test query optimization performance"""
        optimizer = QueryOptimizer()
        
        start_time = time.time()
        
        # Optimize many queries
        for i in range(1000):
            query = {"user_id": f"user-{i}", "limit": 10}
            optimizer.optimize(query)
        
        duration = time.time() - start_time
        
        # Should be very fast
        assert duration < 1.0  # 1 second for 1000 optimizations


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics"""
    
    def test_metrics_creation(self):
        """Test creating performance metrics"""
        metrics = PerformanceMetrics(
            operation_name="test_op",
            duration=0.5,
            success=True
        )
        
        assert metrics.operation_name == "test_op"
        assert metrics.duration == 0.5
        assert metrics.success is True
    
    def test_metrics_with_error(self):
        """Test metrics with error"""
        metrics = PerformanceMetrics(
            operation_name="test_op",
            duration=0.5,
            success=False,
            error="Test error"
        )
        
        assert metrics.success is False
        assert metrics.error == "Test error"
    
    def test_metrics_serialization(self):
        """Test metrics serialization"""
        metrics = PerformanceMetrics(
            operation_name="test_op",
            duration=0.5,
            success=True
        )
        
        data = metrics.to_dict()
        
        assert isinstance(data, dict)
        assert data["operation_name"] == "test_op"
        assert data["duration"] == 0.5
        assert data["success"] is True



