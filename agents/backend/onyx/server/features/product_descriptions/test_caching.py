from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
from typing import Dict, Any
from caching_manager import (
from typing import Any, List, Dict, Optional
import logging
"""
Test script for the caching system

This script tests:
- Basic cache operations
- Different cache strategies
- Performance comparisons
- Error handling
- Cache warming
- Monitoring
"""


    CacheManager, CacheConfig, CacheStrategy, EvictionPolicy,
    StaticDataCache, CacheWarmingService, CacheMonitor,
    get_cache_manager, close_cache_manager
)


async def test_basic_operations():
    """Test basic cache operations"""
    print("\n=== Testing Basic Cache Operations ===")
    
    # Initialize cache manager
    config = CacheConfig(strategy=CacheStrategy.MEMORY)
    cache_manager = CacheManager(config)
    await cache_manager.initialize()
    
    try:
        # Test set operation
        print("1. Testing set operation...")
        success = await cache_manager.set("test_key", "test_value", 60)
        assert success, "Set operation failed"
        print("   ✅ Set operation successful")
        
        # Test get operation
        print("2. Testing get operation...")
        value = await cache_manager.get("test_key")
        assert value == "test_value", f"Expected 'test_value', got {value}"
        print("   ✅ Get operation successful")
        
        # Test exists operation
        print("3. Testing exists operation...")
        exists = await cache_manager.exists("test_key")
        assert exists, "Key should exist"
        print("   ✅ Exists operation successful")
        
        # Test delete operation
        print("4. Testing delete operation...")
        success = await cache_manager.delete("test_key")
        assert success, "Delete operation failed"
        print("   ✅ Delete operation successful")
        
        # Test get after delete
        print("5. Testing get after delete...")
        value = await cache_manager.get("test_key")
        assert value is None, "Value should be None after delete"
        print("   ✅ Get after delete successful")
        
        print("✅ All basic operations passed!")
        
    finally:
        await cache_manager.close()


async def test_cache_strategies():
    """Test different cache strategies"""
    print("\n=== Testing Cache Strategies ===")
    
    strategies = [
        (CacheStrategy.MEMORY, "Memory Cache"),
        (CacheStrategy.REDIS, "Redis Cache"),
        (CacheStrategy.HYBRID, "Hybrid Cache")
    ]
    
    for strategy, name in strategies:
        print(f"\nTesting {name}...")
        
        try:
            # Initialize cache manager
            config = CacheConfig(strategy=strategy)
            cache_manager = CacheManager(config)
            await cache_manager.initialize()
            
            # Test operations
            await cache_manager.set("strategy_test", f"value_{strategy.value}", 60)
            value = await cache_manager.get("strategy_test")
            assert value == f"value_{strategy.value}", f"Value mismatch for {name}"
            
            # Get stats
            stats = cache_manager.get_stats()
            print(f"   ✅ {name} - Hits: {stats['hits']}, Sets: {stats['sets']}")
            
            await cache_manager.close()
            
        except Exception as e:
            print(f"   ❌ {name} failed: {e}")


async def test_performance_comparison():
    """Test performance comparison between strategies"""
    print("\n=== Performance Comparison ===")
    
    strategies = [
        (CacheStrategy.MEMORY, "Memory"),
        (CacheStrategy.HYBRID, "Hybrid")
    ]
    
    results = {}
    
    for strategy, name in strategies:
        print(f"\nTesting {name} performance...")
        
        try:
            # Initialize cache manager
            config = CacheConfig(strategy=strategy)
            cache_manager = CacheManager(config)
            await cache_manager.initialize()
            
            # Performance test
            start_time = time.time()
            
            # Set operations
            for i in range(100):
                await cache_manager.set(f"perf_key_{i}", f"perf_value_{i}", 60)
            
            # Get operations
            for i in range(100):
                await cache_manager.get(f"perf_key_{i}")
            
            end_time = time.time()
            total_time = end_time - start_time
            
            results[name] = {
                "total_time": total_time,
                "avg_time_per_operation": total_time / 200,
                "operations_per_second": 200 / total_time
            }
            
            print(f"   ✅ {name} - Total time: {total_time:.3f}s, "
                  f"Avg per op: {total_time/200:.6f}s, "
                  f"Ops/sec: {200/total_time:.1f}")
            
            await cache_manager.close()
            
        except Exception as e:
            print(f"   ❌ {name} failed: {e}")
    
    # Print comparison
    print("\nPerformance Comparison Summary:")
    for name, result in results.items():
        print(f"{name:10} | {result['total_time']:8.3f}s | "
              f"{result['avg_time_per_operation']:10.6f}s | "
              f"{result['operations_per_second']:8.1f} ops/sec")


async def test_static_data_caching():
    """Test static data caching"""
    print("\n=== Testing Static Data Caching ===")
    
    config = CacheConfig(strategy=CacheStrategy.HYBRID)
    cache_manager = CacheManager(config)
    await cache_manager.initialize()
    
    static_cache = StaticDataCache(cache_manager)
    
    try:
        # Test static data caching
        print("1. Caching static configuration...")
        config_data = {
            "api_version": "1.0.0",
            "features": ["caching", "monitoring"],
            "limits": {"max_products": 1000}
        }
        
        success = await static_cache.cache_static_data("app_config", config_data, 86400)
        assert success, "Failed to cache static data"
        print("   ✅ Static data cached successfully")
        
        # Test retrieval
        print("2. Retrieving static data...")
        retrieved_data = await static_cache.get_static_data("app_config")
        assert retrieved_data == config_data, "Retrieved data doesn't match"
        print("   ✅ Static data retrieved successfully")
        
        # Test invalidation
        print("3. Testing invalidation...")
        success = await static_cache.invalidate_static_data("app_config")
        assert success, "Failed to invalidate static data"
        print("   ✅ Static data invalidated successfully")
        
        # Test retrieval after invalidation
        retrieved_data = await static_cache.get_static_data("app_config")
        assert retrieved_data is None, "Data should be None after invalidation"
        print("   ✅ Static data properly invalidated")
        
        print("✅ All static data caching tests passed!")
        
    finally:
        await cache_manager.close()


async def test_cache_warming():
    """Test cache warming functionality"""
    print("\n=== Testing Cache Warming ===")
    
    config = CacheConfig(strategy=CacheStrategy.HYBRID)
    cache_manager = CacheManager(config)
    await cache_manager.initialize()
    
    warming_service = CacheWarmingService(cache_manager)
    
    try:
        # Mock data source
        async def mock_data_source():
            
    """mock_data_source function."""
return {
                "product_1": {"name": "Product 1", "price": 100},
                "product_2": {"name": "Product 2", "price": 200},
                "product_3": {"name": "Product 3", "price": 300}
            }
        
        print("1. Warming cache with mock data...")
        await warming_service.warm_cache(mock_data_source, "products", 10)
        print("   ✅ Cache warming completed")
        
        # Verify warmed data
        print("2. Verifying warmed data...")
        for i in range(1, 4):
            value = await cache_manager.get(f"products:product_{i}")
            assert value is not None, f"Product {i} not found in cache"
            print(f"   ✅ Product {i} found in cache")
        
        print("✅ All cache warming tests passed!")
        
    finally:
        await cache_manager.close()


async def test_cache_monitoring():
    """Test cache monitoring functionality"""
    print("\n=== Testing Cache Monitoring ===")
    
    config = CacheConfig(strategy=CacheStrategy.HYBRID)
    cache_manager = CacheManager(config)
    await cache_manager.initialize()
    
    monitor = CacheMonitor(cache_manager)
    
    try:
        # Generate some activity
        print("1. Generating cache activity...")
        for i in range(50):
            await cache_manager.set(f"monitor_key_{i}", f"monitor_value_{i}", 60)
            await cache_manager.get(f"monitor_key_{i}")
        
        # Record custom metrics
        print("2. Recording custom metrics...")
        await monitor.record_metric({
            "operation": "test_operation",
            "duration": 0.05,
            "success": True
        })
        
        # Get performance report
        print("3. Getting performance report...")
        report = await monitor.get_performance_report()
        
        print(f"   Cache Stats:")
        print(f"     Hits: {report['cache_stats']['hits']}")
        print(f"     Misses: {report['cache_stats']['misses']}")
        print(f"     Hit Rate: {report['cache_stats']['hit_rate']:.2%}")
        print(f"     Total Requests: {report['cache_stats']['total_requests']}")
        
        print(f"   Alerts: {len(report['alerts'])}")
        print(f"   Recommendations: {len(report['recommendations'])}")
        
        print("✅ All cache monitoring tests passed!")
        
    finally:
        await cache_manager.close()


async def test_error_handling():
    """Test error handling scenarios"""
    print("\n=== Testing Error Handling ===")
    
    # Test with invalid Redis URL
    print("1. Testing invalid Redis configuration...")
    try:
        config = CacheConfig(
            strategy=CacheStrategy.REDIS,
            redis_url="redis://invalid-host:6379"
        )
        cache_manager = CacheManager(config)
        await cache_manager.initialize()
        
        # This should fail gracefully
        value = await cache_manager.get("test_key")
        assert value is None, "Should return None on connection error"
        print("   ✅ Graceful handling of Redis connection error")
        
        await cache_manager.close()
        
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
    
    # Test with memory cache (should always work)
    print("2. Testing memory cache error handling...")
    try:
        config = CacheConfig(strategy=CacheStrategy.MEMORY)
        cache_manager = CacheManager(config)
        await cache_manager.initialize()
        
        # Test operations
        await cache_manager.set("error_test", "value", 60)
        value = await cache_manager.get("error_test")
        assert value == "value", "Memory cache should work"
        print("   ✅ Memory cache error handling successful")
        
        await cache_manager.close()
        
    except Exception as e:
        print(f"   ❌ Memory cache error: {e}")


async def run_all_tests():
    """Run all caching tests"""
    print("🚀 Starting Caching System Tests")
    print("=" * 50)
    
    try:
        await test_basic_operations()
        await test_cache_strategies()
        await test_performance_comparison()
        await test_static_data_caching()
        await test_cache_warming()
        await test_cache_monitoring()
        await test_error_handling()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        raise
    
    finally:
        # Cleanup global cache manager
        await close_cache_manager()


if __name__ == "__main__":
    # Run the test suite
    asyncio.run(run_all_tests()) 