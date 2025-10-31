from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import pytest
import asyncio
import time
import psutil
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any
from onyx.server.features.ads.performance_optimizer import (
from typing import Any, List, Dict, Optional
import logging
"""
Test suite for Performance Optimization System

This module tests all components of the performance optimization system:
- Memory management
- Caching system
- Async task management
- Database optimization
- Performance monitoring
"""

    PerformanceOptimizer,
    PerformanceConfig,
    MemoryManager,
    AdvancedCache,
    AsyncTaskManager,
    DatabaseOptimizer,
    performance_monitor,
    cache_result,
    performance_context,
    memory_context,
    optimizer
)

class TestPerformanceConfig:
    """Test PerformanceConfig class."""
    
    def test_default_config(self) -> Any:
        """Test default configuration values."""
        config = PerformanceConfig()
        
        assert config.cache_ttl == 3600
        assert config.cache_max_size == 10000
        assert config.max_workers == 10
        assert config.memory_cleanup_threshold == 0.8
        assert config.profiling_enabled is True
        assert config.tracemalloc_enabled is True
    
    def test_custom_config(self) -> Any:
        """Test custom configuration values."""
        config = PerformanceConfig(
            cache_ttl=7200,
            cache_max_size=5000,
            max_workers=5,
            memory_cleanup_threshold=0.7,
            profiling_enabled=False
        )
        
        assert config.cache_ttl == 7200
        assert config.cache_max_size == 5000
        assert config.max_workers == 5
        assert config.memory_cleanup_threshold == 0.7
        assert config.profiling_enabled is False

class TestMemoryManager:
    """Test MemoryManager class."""
    
    def setup_method(self) -> Any:
        """Setup test method."""
        self.config = PerformanceConfig()
        self.memory_manager = MemoryManager(self.config)
    
    def test_get_memory_usage(self) -> Optional[Dict[str, Any]]:
        """Test memory usage retrieval."""
        memory_usage = self.memory_manager.get_memory_usage()
        
        assert 'rss' in memory_usage
        assert 'vms' in memory_usage
        assert 'percent' in memory_usage
        assert 'available' in memory_usage
        assert 'total' in memory_usage
        
        assert memory_usage['rss'] > 0
        assert memory_usage['vms'] > 0
        assert 0 <= memory_usage['percent'] <= 100
        assert memory_usage['available'] > 0
        assert memory_usage['total'] > 0
    
    def test_should_cleanup_memory(self) -> Any:
        """Test memory cleanup threshold check."""
        # Mock high memory usage
        with patch.object(psutil.Process, 'memory_info') as mock_memory_info:
            mock_memory_info.return_value = Mock(
                rss=1024 * 1024 * 1024 * 9,  # 9GB
                vms=1024 * 1024 * 1024 * 10  # 10GB
            )
            
            with patch.object(psutil, 'virtual_memory') as mock_virtual_memory:
                mock_virtual_memory.return_value = Mock(total=1024 * 1024 * 1024 * 10)  # 10GB
                
                should_cleanup = self.memory_manager.should_cleanup_memory()
                assert should_cleanup is True
    
    def test_cleanup_memory(self) -> Any:
        """Test memory cleanup functionality."""
        with patch('gc.collect') as mock_gc_collect:
            mock_gc_collect.return_value = 100
            
            result = self.memory_manager.cleanup_memory(force=True)
            
            assert 'initial_memory' in result
            assert 'final_memory' in result
            assert 'memory_freed' in result
            assert 'gc_collected' in result
            assert 'cleanup_time' in result
            
            mock_gc_collect.assert_called_once()
    
    def test_get_memory_stats(self) -> Optional[Dict[str, Any]]:
        """Test memory statistics retrieval."""
        stats = self.memory_manager.get_memory_stats()
        
        assert 'current' in stats
        assert 'history' in stats
        assert 'gc_stats' in stats
        assert 'warnings' in stats
        
        assert 'counter' in stats['gc_stats']
        assert 'last_gc' in stats['gc_stats']
        assert 'threshold' in stats['gc_stats']

class TestAdvancedCache:
    """Test AdvancedCache class."""
    
    def setup_method(self) -> Any:
        """Setup test method."""
        self.config = PerformanceConfig()
        self.cache = AdvancedCache(self.config)
    
    def test_cache_key_generation(self) -> Any:
        """Test cache key generation."""
        key = self.cache._get_cache_key("test_key", "test_prefix")
        
        assert key.startswith("test_prefix:")
        assert len(key) > len("test_prefix:") + 10  # Should be hashed
    
    def test_data_compression(self) -> Any:
        """Test data compression and decompression."""
        test_data = {"key": "value", "number": 123, "list": [1, 2, 3]}
        
        compressed = self.cache._compress_data(test_data)
        decompressed = self.cache._decompress_data(compressed)
        
        assert isinstance(compressed, bytes)
        assert decompressed == test_data
    
    def test_cache_set_get(self) -> Optional[Dict[str, Any]]:
        """Test cache set and get operations."""
        test_key = "test_key"
        test_value = {"data": "test_value"}
        
        # Test set
        self.cache.set(test_key, test_value, cache_type="l1")
        
        # Test get
        retrieved_value = self.cache.get(test_key, cache_type="l1")
        assert retrieved_value == test_value
    
    def test_cache_miss(self) -> Any:
        """Test cache miss behavior."""
        retrieved_value = self.cache.get("nonexistent_key", cache_type="l1")
        assert retrieved_value is None
    
    def test_cache_clear(self) -> Any:
        """Test cache clearing."""
        # Add some data
        self.cache.set("key1", "value1", cache_type="l1")
        self.cache.set("key2", "value2", cache_type="l2")
        
        # Clear L1 cache
        self.cache.clear(cache_type="l1")
        
        # Check that L1 is cleared but L2 remains
        assert self.cache.get("key1", cache_type="l1") is None
        assert self.cache.get("key2", cache_type="l2") == "value2"
    
    def test_cache_stats(self) -> Any:
        """Test cache statistics."""
        # Add some data to generate stats
        self.cache.set("key1", "value1", cache_type="l1")
        self.cache.set("key2", "value2", cache_type="l2")
        self.cache.get("key1", cache_type="l1")  # Hit
        self.cache.get("nonexistent", cache_type="l1")  # Miss
        
        stats = self.cache.get_stats()
        
        assert 'hits' in stats
        assert 'misses' in stats
        assert 'hit_rate' in stats
        assert 'compression_stats' in stats
        assert 'cache_sizes' in stats
        
        assert stats['hit_rate'] > 0
        assert 'l1' in stats['hits']
        assert 'l1' in stats['misses']

class TestAsyncTaskManager:
    """Test AsyncTaskManager class."""
    
    @pytest.fixture
    def task_manager(self) -> Any:
        """Create task manager fixture."""
        config = PerformanceConfig(max_workers=2)
        return AsyncTaskManager(config)
    
    @pytest.mark.asyncio
    async def test_submit_task(self, task_manager) -> Any:
        """Test task submission."""
        async def test_function(x, y) -> Any:
            await asyncio.sleep(0.1)
            return x + y
        
        task = await task_manager.submit_task(test_function, 2, 3)
        result = await task
        
        assert result == 5
        assert task.done()
    
    @pytest.mark.asyncio
    async def test_batch_submit(self, task_manager) -> Any:
        """Test batch task submission."""
        async def test_function(x) -> Any:
            await asyncio.sleep(0.1)
            return x * 2
        
        tasks_data = [
            (test_function, 1),
            (test_function, 2),
            (test_function, 3)
        ]
        
        submitted_tasks = await task_manager.batch_submit(tasks_data)
        results = await task_manager.wait_for_tasks(submitted_tasks)
        
        assert len(results) == 3
        assert results == [2, 4, 6]
    
    @pytest.mark.asyncio
    async def test_task_timeout(self, task_manager) -> Any:
        """Test task timeout handling."""
        async def slow_function():
            
    """slow_function function."""
await asyncio.sleep(2.0)
            return "done"
        
        task = await task_manager.submit_task(slow_function)
        
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(task, timeout=0.5)
    
    @pytest.mark.asyncio
    async def test_task_error_handling(self, task_manager) -> Any:
        """Test task error handling."""
        async def error_function():
            
    """error_function function."""
raise ValueError("Test error")
        
        task = await task_manager.submit_task(error_function)
        
        with pytest.raises(ValueError):
            await task
    
    def test_get_stats(self, task_manager) -> Optional[Dict[str, Any]]:
        """Test task statistics."""
        stats = task_manager.get_stats()
        
        assert 'running_tasks' in stats
        assert 'task_stats' in stats
        assert 'pool_stats' in stats
        
        assert stats['running_tasks'] == 0
        assert 'thread_pool_size' in stats['pool_stats']
        assert 'process_pool_size' in stats['pool_stats']

class TestDatabaseOptimizer:
    """Test DatabaseOptimizer class."""
    
    def setup_method(self) -> Any:
        """Setup test method."""
        self.config = PerformanceConfig()
        self.db_optimizer = DatabaseOptimizer(self.config)
    
    def test_query_timer_context(self) -> Any:
        """Test query timer context manager."""
        with self.db_optimizer.query_timer("test_query"):
            time.sleep(0.1)  # Simulate query time
        
        stats = self.db_optimizer.get_stats()
        assert 'test_query' in stats['query_stats']
        assert stats['query_stats']['test_query']['count'] == 1
        assert stats['query_stats']['test_query']['avg_time'] > 0
    
    def test_query_caching(self) -> Any:
        """Test query result caching."""
        test_key = "test_query_key"
        test_result = {"data": "test_result"}
        
        # Cache query result
        self.db_optimizer.cache_query_result(test_key, test_result)
        
        # Retrieve cached result
        cached_result = self.db_optimizer.get_cached_query(test_key)
        assert cached_result == test_result
    
    def test_slow_query_detection(self) -> Any:
        """Test slow query detection."""
        # Simulate slow query
        with self.db_optimizer.query_timer("slow_query"):
            time.sleep(1.5)  # Above 1.0 second threshold
        
        stats = self.db_optimizer.get_stats()
        assert len(stats['slow_queries']) == 1
        assert stats['slow_queries'][0]['query_type'] == "slow_query"
        assert stats['slow_queries'][0]['execution_time'] > 1.0

class TestPerformanceDecorators:
    """Test performance decorators."""
    
    @pytest.mark.asyncio
    async def test_performance_monitor_decorator(self) -> Any:
        """Test performance monitor decorator."""
        @performance_monitor("test_operation")
        async def test_function():
            
    """test_function function."""
await asyncio.sleep(0.1)
            return "result"
        
        result = await test_function()
        assert result == "result"
    
    @pytest.mark.asyncio
    async def test_cache_result_decorator(self) -> Any:
        """Test cache result decorator."""
        call_count = 0
        
        @cache_result(ttl=3600, cache_type="l1")
        async def test_function(x) -> Any:
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call should execute function
        result1 = await test_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call should use cache
        result2 = await test_function(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment
    
    @pytest.mark.asyncio
    async def test_performance_context(self) -> Any:
        """Test performance context manager."""
        async with performance_context("test_context"):
            await asyncio.sleep(0.1)
        
        # Context manager should complete without error
    
    def test_memory_context(self) -> Any:
        """Test memory context manager."""
        with memory_context():
            # Simulate memory-intensive operation
            large_list = [i for i in range(10000)]
        
        # Context manager should complete without error

class TestPerformanceOptimizer:
    """Test PerformanceOptimizer class."""
    
    @pytest.fixture
    def performance_optimizer(self) -> Any:
        """Create performance optimizer fixture."""
        config = PerformanceConfig(
            cache_ttl=1800,
            cache_max_size=1000,
            max_workers=2,
            metrics_interval=1
        )
        return PerformanceOptimizer(config)
    
    @pytest.mark.asyncio
    async def test_start_stop(self, performance_optimizer) -> Any:
        """Test optimizer start and stop."""
        # Start optimizer
        await performance_optimizer.start()
        assert performance_optimizer._started is True
        
        # Stop optimizer
        await performance_optimizer.stop()
        assert performance_optimizer._started is False
    
    @pytest.mark.asyncio
    async def test_get_performance_stats(self, performance_optimizer) -> Optional[Dict[str, Any]]:
        """Test performance statistics retrieval."""
        await performance_optimizer.start()
        
        stats = performance_optimizer.get_performance_stats()
        
        assert 'memory' in stats
        assert 'cache' in stats
        assert 'tasks' in stats
        assert 'database' in stats
        assert 'config' in stats
        
        await performance_optimizer.stop()
    
    @pytest.mark.asyncio
    async def test_monitoring_loop(self, performance_optimizer) -> Any:
        """Test monitoring loop functionality."""
        await performance_optimizer.start()
        
        # Let monitoring run for a short time
        await asyncio.sleep(0.1)
        
        # Check that monitoring task is running
        assert performance_optimizer._monitoring_task is not None
        assert not performance_optimizer._monitoring_task.done()
        
        await performance_optimizer.stop()

class TestIntegration:
    """Integration tests for the performance optimization system."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self) -> Any:
        """Test complete performance optimization workflow."""
        # Create optimizer
        config = PerformanceConfig(
            cache_ttl=1800,
            cache_max_size=1000,
            max_workers=2,
            metrics_interval=1
        )
        optimizer = PerformanceOptimizer(config)
        
        # Start optimizer
        await optimizer.start()
        
        # Test memory management
        memory_stats = optimizer.memory_manager.get_memory_stats()
        assert 'current' in memory_stats
        
        # Test caching
        test_data = {"key": "value"}
        optimizer.cache.set("test_key", test_data, cache_type="l1")
        cached_data = optimizer.cache.get("test_key", cache_type="l1")
        assert cached_data == test_data
        
        # Test task management
        async def test_task():
            
    """test_task function."""
await asyncio.sleep(0.1)
            return "task_result"
        
        task = await optimizer.task_manager.submit_task(test_task)
        result = await task
        assert result == "task_result"
        
        # Test database optimization
        with optimizer.db_optimizer.query_timer("test_query"):
            await asyncio.sleep(0.1)
        
        # Get performance stats
        stats = optimizer.get_performance_stats()
        assert 'memory' in stats
        assert 'cache' in stats
        assert 'tasks' in stats
        assert 'database' in stats
        
        # Stop optimizer
        await optimizer.stop()
        assert optimizer._started is False

class TestErrorHandling:
    """Test error handling in performance optimization system."""
    
    @pytest.mark.asyncio
    async def test_cache_error_handling(self) -> Any:
        """Test cache error handling."""
        config = PerformanceConfig()
        cache = AdvancedCache(config)
        
        # Test with invalid data that can't be serialized
        class UnserializableObject:
            pass
        
        # Should handle serialization errors gracefully
        cache.set("test_key", UnserializableObject(), cache_type="l1")
        result = cache.get("test_key", cache_type="l1")
        # Should return None or handle error gracefully
        assert result is None or isinstance(result, UnserializableObject)
    
    @pytest.mark.asyncio
    async def test_task_error_handling(self) -> Any:
        """Test task error handling."""
        config = PerformanceConfig(max_workers=1)
        task_manager = AsyncTaskManager(config)
        
        async def error_task():
            
    """error_task function."""
raise RuntimeError("Task error")
        
        task = await task_manager.submit_task(error_task)
        
        with pytest.raises(RuntimeError):
            await task
    
    @pytest.mark.asyncio
    async def test_memory_cleanup_error_handling(self) -> Any:
        """Test memory cleanup error handling."""
        config = PerformanceConfig()
        memory_manager = MemoryManager(config)
        
        # Should handle cleanup errors gracefully
        with patch('gc.collect', side_effect=Exception("GC error")):
            result = memory_manager.cleanup_memory(force=True)
            assert 'error' in result or result is not None

match __name__:
    case "__main__":
    pytest.main([__file__, "-v"]) 