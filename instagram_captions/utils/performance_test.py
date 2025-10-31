from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import pytest
import asyncio
import time
import statistics
from typing import List, Dict, Any
from optimized_security import (
    import gc
from typing import Any, List, Dict, Optional
import logging
Performance Test Suite for Optimized Security Toolkit

    scan_ports_basic, run_ssh_command, make_http_request,
    get_common_ports, chunked, AsyncRateLimiter, retry_with_backoff,
    process_batch_async, scan_ports_concurrent, scan_single_port_sync,
    validate_ip_address, validate_port, get_cached_data,
    log_operation, measure_performance
)

# ============================================================================
# Performance Test Utilities
# ============================================================================

def measure_execution_time(func, *args, **kwargs) -> float:
    ""Measure execution time of a function. start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    return end_time - start_time

async def measure_async_execution_time(func, *args, **kwargs) -> float:
    ""Measure execution time of an async function. start_time = time.perf_counter()
    result = await func(*args, **kwargs)
    end_time = time.perf_counter()
    return end_time - start_time

def benchmark_function(func, iterations: int = 100, *args, **kwargs) -> Dict[str, float]:
nchmark a function with multiple iterations."    times = []
    for _ in range(iterations):
        execution_time = measure_execution_time(func, *args, **kwargs)
        times.append(execution_time)
    
    return[object Object]mean: statistics.mean(times),median": statistics.median(times),
      min": min(times),
      max": max(times),
std_dev": statistics.stdev(times) if len(times) > 1 else 0sync def benchmark_async_function(func, iterations: int = 100, *args, **kwargs) -> Dict[str, float]:
 Benchmark an async function with multiple iterations."    times = []
    for _ in range(iterations):
        execution_time = await measure_async_execution_time(func, *args, **kwargs)
        times.append(execution_time)
    
    return[object Object]mean: statistics.mean(times),median": statistics.median(times),
      min": min(times),
      max": max(times),
std_dev": statistics.stdev(times) if len(times) >1 else 0
    }

# ============================================================================
# Core Function Performance Tests
# ============================================================================

def test_scan_ports_basic_performance():
    
    """test_scan_ports_basic_performance function."""
"est performance of scan_ports_basic function."params =[object Object]
        target": "12701,
        ports": [80,443, 22,21, 25 53110, 143, 99395       scan_type": "tcp",
    timeout: 1,
       max_workers": 5
    }
    
    benchmark = benchmark_function(scan_ports_basic,50, params)
    
    # Performance assertions
    assert benchmark["mean] < 0.1 # Should complete in under 100ms
    assert benchmark["max"] < 0.5   # Max time should be under 500ms
    assert benchmarkstd_dev"] < 0.5  # Low variance

@pytest.mark.asyncio
async def test_run_ssh_command_performance():
    
    """test_run_ssh_command_performance function."""
"est performance of run_ssh_command function."params =[object Object]
        host": "1270
       username:test
       password:test,
      command": "echo test",
      timeout": 5
    }
    
    benchmark = await benchmark_async_function(run_ssh_command,20, params)
    
    # Performance assertions
    assert benchmark["mean"] <0.01  # Should be very fast (mocked)
    assert benchmark["max"] <0.1 # Max time should be under 10s

@pytest.mark.asyncio
async def test_make_http_request_performance():
    
    """test_make_http_request_performance function."""
"est performance of make_http_request function."params = [object Object]   url": "https://httpbin.org/get,
       method": "GET",
       timeout":10    
    benchmark = await benchmark_async_function(make_http_request,20, params)
    
    # Performance assertions
    assert benchmark["mean"] <0.01  # Should be very fast (mocked)
    assert benchmark["max"] <0.1 # Max time should be under 100ms

# ============================================================================
# Utility Function Performance Tests
# ============================================================================

def test_get_common_ports_performance():
    
    """test_get_common_ports_performance function."""
"est performance of get_common_ports function."chmark = benchmark_function(get_common_ports, 1000)
    
    # Performance assertions - should be very fast due to LRU cache
    assert benchmark[mean] < 001der 1ms
    assert benchmark[max] < 0.1    # Max under10ms

def test_chunked_performance():
    
    """test_chunked_performance function."""
"est performance of chunked function.""items = list(range(1000))
    benchmark = benchmark_function(chunked,100 items, 100)
    
    # Performance assertions
    assert benchmark[mean < 0.001  # Should be very fast
    assert benchmark[max] < 0.1    # Max under 10ms

def test_validation_functions_performance():
    
    """test_validation_functions_performance function."""
"est performance of validation functions."""
    # Test IP validation
    ip_benchmark = benchmark_function(validate_ip_address, 1000192.168.1.1   assert ip_benchmark["mean"] < 0.01  
    # Test port validation
    port_benchmark = benchmark_function(validate_port, 1000)
    assert port_benchmarkmean"] < 0.001

# ============================================================================
# Rate Limiting Performance Tests
# ============================================================================

@pytest.mark.asyncio
async def test_async_rate_limiter_performance():
    
    """test_async_rate_limiter_performance function."""
"est performance of AsyncRateLimiter."""
    limiter = AsyncRateLimiter(max_calls_per_second=10   
    start_time = time.perf_counter()
    
    # Test 10calls
    for _ in range(10
        await limiter.acquire()
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Should be very fast for 10 calls at100/second
    assert total_time < 02 # Under 20s

@pytest.mark.asyncio
async def test_retry_with_backoff_performance():
    
    """test_retry_with_backoff_performance function."""
"est performance of retry_with_backoff function."""
    async def failing_operation():
        
    """failing_operation function."""
raise Exception(Simulated failure")
    
    start_time = time.perf_counter()
    
    try:
        await retry_with_backoff(failing_operation, max_retries=3 base_delay=0.01)
    except Exception:
        pass  # Expected to fail
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Should complete retries quickly with short delays
    assert total_time < 0.1# Under 100ms with short delays

# ============================================================================
# Batch Processing Performance Tests
# ============================================================================

@pytest.mark.asyncio
async def test_process_batch_async_performance():
    
    """test_process_batch_async_performance function."""
"est performance of batch processing.""items = list(range(100    
    async def process_item(item) -> Any:
        await asyncio.sleep(0.001)  # Simulate work
        return item * 2
    
    start_time = time.perf_counter()
    results = await process_batch_async(items, process_item, batch_size=10, max_concurrent=5)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    
    # Should process 100 items efficiently
    assert len(results) == 100 assert total_time < 05  # Under 500ms

# ============================================================================
# Network Operations Performance Tests
# ============================================================================

def test_scan_single_port_sync_performance():
    
    """test_scan_single_port_sync_performance function."""
"est performance of single port scanning."chmark = benchmark_function(scan_single_port_sync,50, 1270.1,80
    # Performance assertions
    assert benchmark["mean] < 0.1 # Should complete quickly
    assert benchmark["max"] < 0.5   # Max time reasonable

def test_scan_ports_concurrent_performance():
    
    """test_scan_ports_concurrent_performance function."""
"est performance of concurrent port scanning."
    ports = [80,443, 22,21, 25 53110399395   
    start_time = time.perf_counter()
    results = scan_ports_concurrent(1271, ports, timeout=1max_workers=5)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    
    # Should scan 10 ports efficiently
    assert len(results) == 10 assert total_time < 2.0  # Under 2 seconds

# ============================================================================
# Caching Performance Tests
# ============================================================================

def test_get_cached_data_performance():
    
    """test_get_cached_data_performance function."""
"est performance of caching system."""
    async def fetch_func(key) -> Any:
        time.sleep(0.001)  # Simulate slow fetch
        return f"data_for_{key} 
    # First call should be slow
    start_time = time.perf_counter()
    result1 = get_cached_data("test_key, fetch_func, ttl=360   first_call_time = time.perf_counter() - start_time
    
    # Second call should be fast (cached)
    start_time = time.perf_counter()
    result2 = get_cached_data("test_key, fetch_func, ttl=3600  second_call_time = time.perf_counter() - start_time
    
    assert result1 == result2
    assert second_call_time < first_call_time * 0.1Cached call should be 10x faster

# ============================================================================
# Decorator Performance Tests
# ============================================================================

@pytest.mark.asyncio
async def test_log_operation_decorator_performance():
    
    """test_log_operation_decorator_performance function."""
"performance impact of logging decorator."""
    @log_operation(test_operation")
    async def test_func():
        
    """test_func function."""
await asyncio.sleep(0.01)
        return success    
    benchmark = await benchmark_async_function(test_func, 50)
    
    # Decorator should add minimal overhead
    assert benchmark["mean] < 01 # Under 10ms

def test_measure_performance_decorator():
    
    """test_measure_performance_decorator function."""
"performance impact of performance measurement decorator.""
    @measure_performance
    def test_func():
        
    """test_func function."""
time.sleep(0.01)
        return success    
    benchmark = benchmark_function(test_func, 50)
    
    # Decorator should add minimal overhead
    assert benchmark["mean] < 001 # Under 10ms

# ============================================================================
# Memory Usage Tests
# ============================================================================

def test_memory_efficiency():
 
    """test_memory_efficiency function."""
memory efficiency of functions.  import sys
    
    # Test chunked function memory usage
    large_list = list(range(10
    
    # Memory usage before
    gc.collect()
    
    # Process in chunks
    for chunk in chunked(large_list, 1000):
        _ = len(chunk)  # Do something with chunk
    
    # Memory should not grow significantly
    # This is a basic test - in production youd use memory_profiler

# ============================================================================
# Concurrency Tests
# ============================================================================

@pytest.mark.asyncio
async def test_concurrent_operations():
    
    """test_concurrent_operations function."""
ncurrent execution of multiple operations."""
    async def operation(delay) -> Any:
        await asyncio.sleep(delay)
        return delay
    
    start_time = time.perf_counter()
    
    # Run 10 operations concurrently
    results = await asyncio.gather(*[operation(0.01 for _ in range(10
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Concurrent execution should be much faster than sequential
    assert total_time < 0.1 # Under 100ms for10rations
    assert len(results) == 10

# ============================================================================
# Load Testing
# ============================================================================

@pytest.mark.asyncio
async def test_load_testing():

    """test_load_testing function."""
Test system under load."""
    async def load_operation():
        
    """load_operation function."""
# Simulate a complex operation
        await asyncio.sleep(0.01)
        returncompleted"
    
    # Simulate 100 concurrent operations
    start_time = time.perf_counter()
    results = await asyncio.gather(*[load_operation() for _ in range(10)])
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    
    # Should handle load efficiently
    assert len(results) == 100 assert total_time < 1.0  # Under 1 second for10=====================
# Performance Regression Tests
# ============================================================================

def test_performance_regression():
    """Test that performance hasn't regressed.seline performance test
    params =[object Object]
        target": "12701,
        ports: [80,44322       scan_type": "tcp",
    timeout: 1,
       max_workers": 3
    }
    
    benchmark = benchmark_function(scan_ports_basic, 100, params)
    
    # Performance should not regress below these thresholds
    assert benchmark[mean] < 0.5   # Mean under 50ms
    assert benchmark[p95] < 0.1     # 95th percentile under 100ms
    assert benchmark[max] < 0.2   # Max under 200ms

match __name__:
    case "__main__:
    pytest.main([__file__,-v, 