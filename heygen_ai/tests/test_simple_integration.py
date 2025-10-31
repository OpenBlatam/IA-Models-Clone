"""
Simple Integration Tests for HeyGen AI System
============================================

Basic integration tests that work without complex dependencies:
- Cache + Load balancer integration
- External API + Background tasks
- Simple performance monitoring
"""

import asyncio
import logging
import time
import pytest
from unittest.mock import AsyncMock, patch

# Import core components
from core.external_api_integration import (
    ExternalAPIManager, ServiceConfig
)
from core.performance_optimizer import (
    MemoryCache
)
from core.base_service import ServiceType

logger = logging.getLogger(__name__)


class TestSimpleIntegration:
    """Simple integration tests"""
    
    @pytest.mark.asyncio
    async def test_cache_and_load_balancer_integration(self):
        """Test cache and load balancer working together"""
        # Create cache
        cache = MemoryCache(CacheConfig(
            level=CacheLevel.L1_MEMORY,
            max_size=100,
            ttl_seconds=60
        ))
        
        # Create load balancer
        lb = LoadBalancer(strategy="round_robin")
        lb.add_instance("server1", "http://server1.com", weight=1)
        lb.add_instance("server2", "http://server2.com", weight=1)
        
        # Simulate distributed caching
        async def distributed_cache_operation(key: str, value: str):
            """Simulate operation that uses both cache and load balancer"""
            # Get server from load balancer
            server = lb.get_next_instance()
            
            # Check cache first
            cached_value = cache.get(key)
            if cached_value:
                return {"source": "cache", "value": cached_value, "server": server["id"]}
            
            # Simulate server operation
            await asyncio.sleep(0.05)
            
            # Store in cache
            cache.set(key, value)
            
            return {"source": "server", "value": value, "server": server["id"]}
        
        # Test multiple operations
        results = []
        operations = [
            ("user_123", "John Doe"),
            ("user_456", "Jane Smith"),
            ("user_123", "John Doe"),  # Duplicate for cache test
        ]
        
        for key, value in operations:
            result = await distributed_cache_operation(key, value)
            results.append(result)
        
        # Verify results
        assert len(results) == 3
        assert results[0]["source"] == "server"  # First call hits server
        assert results[2]["source"] == "cache"   # Second call uses cache
        assert results[0]["value"] == "John Doe"
        assert results[2]["value"] == "John Doe"
        
        # Verify load balancer distribution
        server_ids = [r["server"] for r in results]
        assert "server1" in server_ids
        assert "server2" in server_ids
    
    @pytest.mark.asyncio
    async def test_external_api_and_background_tasks(self):
        """Test external API integration with background tasks"""
        # Initialize components
        api_manager = ExternalAPIManager()
        processor = BackgroundTaskProcessor(max_workers=2, max_queue_size=10)
        await processor.start()
        
        # Create service
        config = ServiceConfig(
            service_type=ServiceType.VOICE_SYNTHESIS,
            name="test_service",
            api_key="test_key",
            base_url="https://test.com"
        )
        
        service = ElevenLabsService(config)
        api_manager.register_service(service)
        
        # Define background task
        task_results = []
        
        async def process_api_data(data_id: str):
            """Background task that processes API data"""
            await asyncio.sleep(0.1)  # Simulate processing
            task_results.append(f"Processed {data_id}")
            return f"Result for {data_id}"
        
        # Submit background tasks
        for i in range(5):
            await processor.submit_task(process_api_data, f"data_{i}")
        
        # Wait for completion
        await asyncio.sleep(0.5)
        
        # Verify results
        assert len(task_results) == 5
        assert "Processed data_0" in task_results
        assert "Processed data_4" in task_results
        
        # Verify service registration
        assert "test_service" in api_manager.services
        
        # Cleanup
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_simple_performance_monitoring(self):
        """Test simple performance monitoring without Prometheus"""
        # Create cache for storing metrics
        metrics_cache = MemoryCache(CacheConfig(
            level=CacheLevel.L1_MEMORY,
            max_size=1000,
            ttl_seconds=3600
        ))
        
        # Simple performance monitor
        async def monitor_operation(operation_name: str, operation_func):
            """Monitor operation performance"""
            start_time = time.time()
            
            try:
                result = await operation_func()
                success = True
                error = None
            except Exception as e:
                result = None
                success = False
                error = str(e)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Store metrics
            metrics = {
                "operation": operation_name,
                "duration": duration,
                "success": success,
                "error": error,
                "timestamp": start_time
            }
            
            metrics_cache.set(f"metrics_{operation_name}_{int(start_time)}", metrics)
            return result, metrics
        
        # Test monitored operation
        async def test_operation():
            """Test operation to monitor"""
            await asyncio.sleep(0.1)
            return "operation completed"
        
        # Monitor operation
        result, metrics = await monitor_operation("test_op", test_operation)
        
        # Verify results
        assert result == "operation completed"
        assert metrics["operation"] == "test_op"
        assert metrics["success"] == True
        assert metrics["duration"] > 0
        assert metrics["error"] is None
        
        # Verify metrics are stored
        stored_metrics = metrics_cache.get(f"metrics_test_op_{int(metrics['timestamp'])}")
        assert stored_metrics is not None
        assert stored_metrics["operation"] == "test_op"
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_operations(self):
        """Test concurrent cache operations"""
        # Create cache
        cache = MemoryCache(CacheConfig(
            level=CacheLevel.L1_MEMORY,
            max_size=1000,
            ttl_seconds=60
        ))
        
        # Define cache operation
        async def cache_operation(operation_id: int):
            """Perform cache operation"""
            key = f"key_{operation_id}"
            value = f"value_{operation_id}"
            
            # Set value
            cache.set(key, value)
            
            # Get value
            retrieved = cache.get(key)
            
            # Small delay to simulate work
            await asyncio.sleep(0.01)
            
            return {
                "id": operation_id,
                "key": key,
                "value": value,
                "retrieved": retrieved,
                "success": retrieved == value
            }
        
        # Submit concurrent operations
        start_time = time.time()
        tasks = []
        
        for i in range(50):
            task = cache_operation(i)
            tasks.append(task)
        
        # Wait for all operations to complete
        results = await asyncio.gather(*tasks)
        completion_time = time.time() - start_time
        
        # Verify results
        assert len(results) == 50
        assert all(r["success"] for r in results)
        assert completion_time < 2.0  # Should complete quickly
        
        # Verify cache contents
        for i in range(50):
            cached_value = cache.get(f"key_{i}")
            assert cached_value == f"value_{i}"
    
    @pytest.mark.asyncio
    async def test_load_balancer_health_management(self):
        """Test load balancer health management"""
        lb = LoadBalancer(strategy="round_robin")
        
        # Add instances
        lb.add_instance("healthy1", "http://healthy1.com", weight=1)
        lb.add_instance("healthy2", "http://healthy2.com", weight=1)
        lb.add_instance("unhealthy1", "http://unhealthy1.com", weight=1)
        
        # Mark one instance as unhealthy
        lb.mark_instance_unhealthy("unhealthy1")
        
        # Test distribution (should only use healthy instances)
        instances = []
        for _ in range(10):
            instance = lb.get_next_instance()
            if instance:
                instances.append(instance["id"])
        
        # Verify only healthy instances are used
        assert len(instances) == 10
        assert all(instance_id in ["healthy1", "healthy2"] for instance_id in instances)
        assert "unhealthy1" not in instances
        
        # Mark instance as healthy again
        lb.mark_instance_healthy("unhealthy1")
        
        # Now all instances should be used
        instances = []
        for _ in range(15):
            instance = lb.get_next_instance()
            instances.append(instance["id"])
        
        assert "healthy1" in instances
        assert "healthy2" in instances
        assert "unhealthy1" in instances


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
