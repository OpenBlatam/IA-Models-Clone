"""
Performance Benchmarking Tests for HeyGen AI System
==================================================

Advanced performance tests that measure:
- Memory usage patterns
- CPU utilization
- Response time distributions
- Scalability metrics
- Resource efficiency
"""

import asyncio
import logging
import time
import psutil
import pytest
from typing import Dict, List, Any
from unittest.mock import AsyncMock, patch

# Import core components
from core.performance_optimizer import (
    MemoryCache
)

logger = logging.getLogger(__name__)


class TestPerformanceBenchmarks:
    """Performance benchmarking tests"""
    
    @pytest.mark.asyncio
    async def test_memory_usage_patterns(self):
        """Test memory usage patterns under different loads"""
        # Monitor initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create cache with different sizes
        cache_sizes = [100, 500, 1000, 2000]
        memory_usage = []
        
        for size in cache_sizes:
            cache = MemoryCache(CacheConfig(
                level=CacheLevel.L1_MEMORY,
                max_size=size,
                ttl_seconds=60
            ))
            
            # Fill cache
            for i in range(size):
                cache.set(f"key_{i}", f"value_{i}" * 100)  # Large values
            
            # Measure memory
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_usage.append({
                'cache_size': size,
                'memory_mb': current_memory - initial_memory,
                'efficiency': size / (current_memory - initial_memory) if current_memory > initial_memory else 0
            })
            
            # Cleanup
            del cache
        
        # Verify memory usage is reasonable
        for usage in memory_usage:
            assert usage['memory_mb'] >= 0  # Memory should not decrease
            if usage['cache_size'] > 100:
                assert usage['efficiency'] > 0  # Should have some efficiency
    
    @pytest.mark.asyncio
    async def test_cpu_utilization_under_load(self):
        """Test CPU utilization under different loads"""
        # Monitor initial CPU
        initial_cpu = psutil.cpu_percent(interval=0.1)
        
        # Create load balancer with many instances
        lb = LoadBalancer(strategy="round_robin")
        for i in range(100):
            lb.add_instance(f"instance_{i}", f"http://instance_{i}.com", weight=1)
        
        # Simulate high load
        start_time = time.time()
        operations = []
        
        for _ in range(1000):
            instance = lb.get_next_instance()
            operations.append(instance["id"])
        
        load_time = time.time() - start_time
        
        # Measure CPU during load
        cpu_during_load = psutil.cpu_percent(interval=0.1)
        
        # Verify performance
        assert load_time < 1.0  # Should be very fast
        assert len(operations) == 1000
        assert cpu_during_load >= 0  # CPU should be measurable
    
    @pytest.mark.asyncio
    async def test_response_time_distribution(self):
        """Test response time distribution for different operations"""
        # Create cache
        cache = MemoryCache(CacheConfig(
            level=CacheLevel.L1_MEMORY,
            max_size=1000,
            ttl_seconds=60
        ))
        
        # Test different operation types
        operation_times = {
            'cache_set': [],
            'cache_get': [],
            'cache_miss': [],
            'load_balancer': []
        }
        
        # Load balancer
        lb = LoadBalancer(strategy="round_robin")
        lb.add_instance("server1", "http://server1.com", weight=1)
        lb.add_instance("server2", "http://server2.com", weight=1)
        
        # Measure cache operations
        for i in range(100):
            # Cache set
            start_time = time.time()
            cache.set(f"key_{i}", f"value_{i}")
            operation_times['cache_set'].append(time.time() - start_time)
            
            # Cache hit
            start_time = time.time()
            cache.get(f"key_{i}")
            operation_times['cache_get'].append(time.time() - start_time)
            
            # Cache miss
            start_time = time.time()
            cache.get(f"nonexistent_{i}")
            operation_times['cache_miss'].append(time.time() - start_time)
            
            # Load balancer
            start_time = time.time()
            lb.get_next_instance()
            operation_times['load_balancer'].append(time.time() - start_time)
        
        # Calculate statistics
        stats = {}
        for op_type, times in operation_times.items():
            if times:
                stats[op_type] = {
                    'count': len(times),
                    'avg_ms': sum(times) * 1000 / len(times),
                    'min_ms': min(times) * 1000,
                    'max_ms': max(times) * 1000
                }
        
        # Verify performance characteristics
        assert stats['cache_set']['avg_ms'] < 1.0  # Should be very fast
        assert stats['cache_get']['avg_ms'] < 0.5  # Cache hits should be fastest
        assert stats['load_balancer']['avg_ms'] < 0.1  # Load balancer should be extremely fast
    
    @pytest.mark.asyncio
    async def test_scalability_metrics(self):
        """Test scalability metrics with increasing load"""
        # Test different scales
        scales = [10, 50, 100, 200]
        scale_metrics = []
        
        for scale in scales:
            # Create components at this scale
            start_time = time.time()
            
            # Create cache
            cache = MemoryCache(CacheConfig(
                level=CacheLevel.L1_MEMORY,
                max_size=scale * 10,
                ttl_seconds=60
            ))
            
            # Create load balancer
            lb = LoadBalancer(strategy="round_robin")
            for i in range(scale):
                lb.add_instance(f"instance_{i}", f"http://instance_{i}.com", weight=1)
            
            # Create background processor
            processor = BackgroundTaskProcessor(max_workers=min(scale, 10), max_queue_size=scale * 2)
            await processor.start()
            
            setup_time = time.time() - start_time
            
            # Test operations at this scale
            operation_start = time.time()
            
            # Cache operations
            for i in range(scale):
                cache.set(f"key_{i}", f"value_{i}")
                cache.get(f"key_{i}")
            
            # Load balancer operations
            for _ in range(scale):
                lb.get_next_instance()
            
            # Background tasks
            async def dummy_task(task_id: int):
                await asyncio.sleep(0.01)
                return f"Task {task_id} completed"
            
            for i in range(scale):
                await processor.submit_task(dummy_task, i)
            
            # Wait for completion
            await asyncio.sleep(0.1)
            
            operation_time = time.time() - operation_start
            total_time = time.time() - start_time
            
            # Cleanup
            await processor.stop()
            
            scale_metrics.append({
                'scale': scale,
                'setup_time': setup_time,
                'operation_time': operation_time,
                'total_time': total_time,
                'operations_per_second': scale / operation_time if operation_time > 0 else 0
            })
        
        # Verify scalability
        for i, metrics in enumerate(scale_metrics):
            if i > 0:  # Compare with previous scale
                prev_metrics = scale_metrics[i-1]
                # Operations per second should not decrease dramatically
                assert metrics['operations_per_second'] > prev_metrics['operations_per_second'] * 0.5
    
    @pytest.mark.asyncio
    async def test_resource_efficiency(self):
        """Test resource efficiency under sustained load"""
        # Monitor initial resources
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        initial_cpu = psutil.cpu_percent(interval=0.1)
        
        # Create components
        cache = MemoryCache(CacheConfig(
            level=CacheLevel.L1_MEMORY,
            max_size=5000,
            ttl_seconds=60
        ))
        
        lb = LoadBalancer(strategy="round_robin")
        for i in range(50):
            lb.add_instance(f"instance_{i}", f"http://instance_{i}.com", weight=1)
        
        processor = BackgroundTaskProcessor(max_workers=5, max_queue_size=100)
        await processor.start()
        
        # Sustained load simulation
        async def sustained_operation():
            for _ in range(100):
                # Cache operations
                cache.set(f"sustained_{_}", f"value_{_}")
                cache.get(f"sustained_{_}")
                
                # Load balancer
                lb.get_next_instance()
                
                # Background task
                await processor.submit_task(lambda: None)
                
                # Small delay to simulate real workload
                await asyncio.sleep(0.001)
        
        # Run sustained operation
        start_time = time.time()
        await sustained_operation()
        operation_time = time.time() - start_time
        
        # Measure final resources
        final_memory = process.memory_info().rss / 1024 / 1024
        final_cpu = psutil.cpu_percent(interval=0.1)
        
        # Calculate efficiency metrics
        memory_increase = final_memory - initial_memory
        memory_efficiency = 1000 / memory_increase if memory_increase > 0 else float('inf')
        
        # Verify efficiency
        assert operation_time < 10.0  # Should complete in reasonable time
        assert memory_increase >= 0  # Memory should not decrease
        assert memory_efficiency > 0  # Should have some efficiency
        
        # Cleanup
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_load_handling(self):
        """Test handling of concurrent load"""
        # Create components
        cache = MemoryCache(CacheConfig(
            level=CacheLevel.L1_MEMORY,
            max_size=10000,
            ttl_seconds=60
        ))
        
        lb = LoadBalancer(strategy="round_robin")
        for i in range(20):
            lb.add_instance(f"instance_{i}", f"http://instance_{i}.com", weight=1)
        
        # Define concurrent operations
        async def concurrent_cache_operation(worker_id: int):
            """Perform concurrent cache operations"""
            results = []
            for i in range(100):
                key = f"worker_{worker_id}_key_{i}"
                value = f"worker_{worker_id}_value_{i}"
                
                # Set value
                cache.set(key, value)
                
                # Get value
                retrieved = cache.get(key)
                
                results.append({
                    'worker': worker_id,
                    'operation': i,
                    'success': retrieved == value
                })
            
            return results
        
        # Run concurrent operations
        start_time = time.time()
        
        # Create multiple workers
        workers = []
        for i in range(10):
            worker = concurrent_cache_operation(i)
            workers.append(worker)
        
        # Wait for all workers to complete
        all_results = await asyncio.gather(*workers)
        
        completion_time = time.time() - start_time
        
        # Verify results
        total_operations = sum(len(results) for results in all_results)
        successful_operations = sum(
            sum(1 for result in results if result['success'])
            for results in all_results
        )
        
        # Verify performance
        assert completion_time < 5.0  # Should complete in reasonable time
        assert total_operations == 1000  # 10 workers * 100 operations each
        assert successful_operations == total_operations  # All operations should succeed
        
        # Verify load balancer under concurrent load
        lb_results = []
        for _ in range(100):
            instance = lb.get_next_instance()
            lb_results.append(instance["id"])
        
        # Should distribute load across instances
        unique_instances = set(lb_results)
        assert len(unique_instances) > 1  # Should use multiple instances


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
