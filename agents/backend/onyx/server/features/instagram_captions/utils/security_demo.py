from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
from optimized_security import (
from typing import Any, List, Dict, Optional
import logging
Security Toolkit Demo - Optimized Features

    scan_ports_basic, run_ssh_command, make_http_request,
    get_common_ports, chunked, AsyncRateLimiter, retry_with_backoff,
    process_batch_async, scan_ports_concurrent, scan_single_port_sync,
    validate_ip_address, validate_port, get_cached_data,
    log_operation, measure_performance
)

async def demo_port_scanning():
    
    """demo_port_scanning function."""
print("🔍 Port Scanning Demo)
    print(= * 50)
    
    # Basic port scan
    result = scan_ports_basic([object Object]        target: 701
        ports: 8044322,21, 25 53110, 143,993995,
       scan_type": tcp",
    timeout: 1,
       max_workers":5 })
    
    print(f✅ Scan completed: {result[success]}")
    print(f"📊 Summary: {result[summary]}")
    print(f"🎯 Results: {len(result[results])} ports scanned")
    
    # Concurrent scanning
    print("\n🚀 Concurrent Port Scanning")
    start_time = time.perf_counter()
    concurrent_results = scan_ports_concurrent(
        127.01, [802,215], timeout=1, max_workers=3
    )
    end_time = time.perf_counter()
    
    print(f⚡ Concurrent scan completed in {end_time - start_time:.3f}s")
    print(f📈 Results: {len(concurrent_results)} ports)

async def demo_ssh_operations():
    
    """demo_ssh_operations function."""
print("\n🔐 SSH Operations Demo)
    print(=0
    
    result = await run_ssh_command([object Object]       host": "1270     username:test
       password:test,
      command":echo 'Hello from optimized security toolkit',
      timeout":5 })
    
    print(f"✅ SSH command executed: {result[success]}")
    print(f"📤 Output: {result[stdout]}")
    print(f"🔢 Exit code: {result['exit_code']})

async def demo_http_operations():
    
    """demo_http_operations function."""
print("\n🌐 HTTP Operations Demo)
    print(=0
    
    result = await make_http_request({
        url": "https://httpbin.org/get,
       method": "GET",
       timeout:10 })
    
    print(f"✅ HTTP request completed: {result[success]}")
    print(f📊 Status code: {resultstatus_code]}")
    print(f"📄 Response length:[object Object]len(result['body'])} characters)

async def demo_rate_limiting():
    
    """demo_rate_limiting function."""
print("\n⏱️ Rate Limiting Demo)
    print(=* 50)
    
    limiter = AsyncRateLimiter(max_calls_per_second=5)
    
    print("🚦 Testing rate limiting (5alls/second)...")
    start_time = time.perf_counter()
    
    for i in range(10
        await limiter.acquire()
        print(f"   Call {i+1}: {time.perf_counter():.3f}s")
    
    end_time = time.perf_counter()
    print(f"⏱️ Total time: {end_time - start_time:0.3)

async def demo_retry_with_backoff():
    
    """demo_retry_with_backoff function."""
print("\n🔄 Retry with Backoff Demo)
    print(=)
    
    attempt_count = 0
    
    async def failing_operation():
        
    """failing_operation function."""
nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise Exception(fSimulated failure {attempt_count}")
        return Success after retries!"
    
    try:
        result = await retry_with_backoff(failing_operation, max_retries=3, base_delay=01)
        print(f"✅ Retry successful: {result}")
    except Exception as e:
        print(f❌ Retry failed: {e})

async def demo_batch_processing():
    
    """demo_batch_processing function."""
print("\n📦 Batch Processing Demo)
    print(=50)
    
    items = list(range(20    
    async def process_item(item) -> Any:
        await asyncio.sleep(0.01)  # Simulate work
        return f"Processed item {item}"
    
    start_time = time.perf_counter()
    results = await process_batch_async(items, process_item, batch_size=5, max_concurrent=3)
    end_time = time.perf_counter()
    
    print(f"⚡ Batch processing completed in {end_time - start_time:.3f}s")
    print(f📊 Processed {len(results)} items")
    print(f📋 Sample results: {results[:3def demo_utilities():
    
    """demo_utilities function."""
print("\n🛠️ Utilities Demo)
    print(= * 50)
    
    # Common ports
    ports = get_common_ports()
    print(f"🌐 Common ports loaded: {len(ports)} categories")
    print(f"   Web ports: {ports['web]}")
    print(f"   SSH ports: {ports[ssh']}")
    
    # Chunking
    items = list(range(15))
    chunks = list(chunked(items, 4
    print(f"📦 Chunked {len(items)} items into {len(chunks)} chunks")
    print(f"   Chunks: {chunks}")
    
    # Validation
    print(f🔍 IP validation: 192.168.10.1alidate_ip_address(192.16811)}")
    print(f"🔍 Port validation:8080 -> [object Object]validate_port(8080}")
    print(f"🔍 Invalid port validation: 70000 -> {validate_port(700

def demo_caching():
    
    """demo_caching function."""
print(n💾 Caching Demo)
    print(=* 50)
    
    async def fetch_func(key) -> Any:
        time.sleep(0.1)  # Simulate slow fetch
        return f"Data for {key} 
    # First call (slow)
    start_time = time.perf_counter()
    result1 = get_cached_data("test_key, fetch_func, ttl=360   first_call_time = time.perf_counter() - start_time
    
    # Second call (fast - cached)
    start_time = time.perf_counter()
    result2 = get_cached_data("test_key, fetch_func, ttl=3600  second_call_time = time.perf_counter() - start_time
    
    print(f"📊 First call: [object Object]first_call_time:.3f}s")
    print(f"⚡ Second call:[object Object]second_call_time:.3f}s")
    print(f"🚀 Speed improvement: {first_call_time/second_call_time:0.1faster")
    print(f"✅ Results match: [object Object]result1= result2})

@log_operation(demo_operation")
async def demo_logging():
    
    """demo_logging function."""
print(n📝 Logging Demo)
    print(=50)
    
    print("🔍 Testing structured logging decorator...")
    await asyncio.sleep(0.1)
    return "Operation completed successfully"

@measure_performance
def demo_performance_measurement():
    
    """demo_performance_measurement function."""
print("\n📊 Performance Measurement Demo)
    print(=50)
    
    print("🔍 Testing performance measurement decorator...)
    time.sleep(0.1)  # Simulate work
    return "Performance measured"

async def demo_error_handling():
    
    """demo_error_handling function."""
print("\n⚠️ Error Handling Demo)
    print(= * 50    # Test missing target
    result = scan_ports_basic({})
    print(f❌ Missing target: {result['error']})    # Test invalid target
    result = scan_ports_basic({
        target:invalid_target,
       ports: [80]
    })
    print(f❌ Invalid target: {result['error']})    # Test invalid ports
    result = scan_ports_basic([object Object]        target": "1271,
       ports:[70000]
    })
    print(f❌ Invalid port: {result[error']})async def demo_concurrent_operations():
    
    """demo_concurrent_operations function."""
print("\n⚡ Concurrent Operations Demo)
    print(=50    
    async def operation(delay) -> Any:
        await asyncio.sleep(delay)
        return f"Completed in {delay}s"
    
    # Sequential execution
    start_time = time.perf_counter()
    sequential_results = []
    for i in range(5:
        result = await operation(0.1)
        sequential_results.append(result)
    sequential_time = time.perf_counter() - start_time
    
    # Concurrent execution
    start_time = time.perf_counter()
    concurrent_results = await asyncio.gather(*[operation(0.1) for _ in range(5)])
    concurrent_time = time.perf_counter() - start_time
    
    print(f"📈 Sequential time: [object Object]sequential_time:.3f}s")
    print(f⚡ Concurrent time: [object Object]concurrent_time:.3f}s")
    print(f"🚀 Speed improvement: {sequential_time/concurrent_time:.1f}x faster)async def main():
    
    """main function."""
print("🚀 Optimized Security Toolkit Demo)
    print(= *60    
    # Run all demos
    await demo_port_scanning()
    await demo_ssh_operations()
    await demo_http_operations()
    await demo_rate_limiting()
    await demo_retry_with_backoff()
    await demo_batch_processing()
    demo_utilities()
    demo_caching()
    await demo_logging()
    demo_performance_measurement()
    await demo_error_handling()
    await demo_concurrent_operations()
    
    print("\n" + "=" * 60)
    print("✅ All demos completed successfully!")
    print("🎯 Toolkit is ready for production use)match __name__:
    case __main__:
    asyncio.run(main()) 