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
import sys
import os
from cybersecurity.async_helpers import get_async_helper, AsyncOperationConfig
from cybersecurity.core.non_blocking_scanner import get_non_blocking_scanner, NonBlockingScanConfig
    import os
    from cybersecurity.async_helpers import get_async_helper
    from cybersecurity.async_helpers import get_async_helper
        import traceback
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Non-Blocking Operations Demo
Demonstrates how to avoid blocking operations in core scanning loops.
"""


# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def demo_async_helpers():
    """Demo async helpers for heavy I/O operations."""
    print("🔧 Demo: Async Helpers for Heavy I/O Operations")
    print("=" * 60)
    
    # Create async helper
    config = AsyncOperationConfig(
        timeout=5.0,
        max_retries=2,
        max_concurrent=10
    )
    helper = get_async_helper(config)
    
    # Demo HTTP requests
    print("\n📡 HTTP Requests (Non-blocking):")
    urls = [
        "http://httpbin.org/get",
        "http://httpbin.org/status/200",
        "http://httpbin.org/status/404"
    ]
    
    start_time = time.time()
    http_tasks = [helper.http_request(url) for url in urls]
    http_results = await asyncio.gather(*http_tasks)
    
    for url, result in zip(urls, http_results):
        status = "✅" if result.success else "❌"
        print(f"  {status} {url}: {result.duration:.3f}s")
    
    print(f"  Total HTTP time: {time.time() - start_time:.3f}s")
    
    # Demo DNS lookups
    print("\n🌐 DNS Lookups (Non-blocking):")
    domains = ["google.com", "github.com", "stackoverflow.com"]
    
    start_time = time.time()
    dns_tasks = [helper.dns_lookup(domain) for domain in domains]
    dns_results = await asyncio.gather(*dns_tasks)
    
    for domain, result in zip(domains, dns_results):
        status = "✅" if result.success else "❌"
        print(f"  {status} {domain}: {result.data if result.success else result.error}")
    
    print(f"  Total DNS time: {time.time() - start_time:.3f}s")
    
    # Demo file operations
    print("\n📁 File Operations (Non-blocking):")
    
    test_file = "test_async_file.txt"
    test_content = "This is a test file for async operations.\n" * 100
    
    # Write file
    write_result = await helper.file_operation(test_file, "write", test_content)
    print(f"  Write file: {'✅' if write_result.success else '❌'}")
    
    # Read file
    read_result = await helper.file_operation(test_file, "read")
    print(f"  Read file: {'✅' if read_result.success else '❌'}")
    print(f"  File size: {len(read_result.data) if read_result.success else 0} bytes")
    
    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)
    
    # Demo crypto operations
    print("\n🔐 Crypto Operations (Non-blocking):")
    
    test_data = b"This is test data for crypto operations"
    
    # Hash operation
    hash_result = await helper.crypto_operation("hash", test_data)
    print(f"  Hash: {'✅' if hash_result.success else '❌'}")
    if hash_result.success:
        print(f"    SHA256: {hash_result.data}")
    
    # Base64 encode
    encode_result = await helper.crypto_operation("base64_encode", test_data)
    print(f"  Base64 encode: {'✅' if encode_result.success else '❌'}")
    
    # Demo batch operations
    print("\n🔄 Batch Operations (Non-blocking):")
    
    async def sample_operation(delay: float):
        
    """sample_operation function."""
await asyncio.sleep(delay)
        return f"Operation completed in {delay}s"
    
    operations = [
        lambda: sample_operation(0.1),
        lambda: sample_operation(0.2),
        lambda: sample_operation(0.3),
        lambda: sample_operation(0.1),
        lambda: sample_operation(0.2)
    ]
    
    start_time = time.time()
    batch_results = await helper.batch_operations(operations, max_concurrent=3)
    
    for i, result in enumerate(batch_results):
        status = "✅" if isinstance(result, str) else "❌"
        print(f"  {status} Batch operation {i+1}: {result}")
    
    print(f"  Total batch time: {time.time() - start_time:.3f}s")
    
    # Get operation stats
    stats = helper.get_operation_stats()
    print(f"\n📊 Operation Statistics:")
    for operation, stat in stats.items():
        print(f"  {operation}: {stat['count']} ops, avg {stat['avg_duration']:.3f}s")
    
    await helper.close()

async def demo_non_blocking_scanner():
    """Demo non-blocking scanner."""
    print("\n\n🔍 Demo: Non-Blocking Scanner")
    print("=" * 60)
    
    # Create non-blocking scanner
    config = NonBlockingScanConfig(
        max_concurrent_scans=20,
        scan_timeout=10.0,
        chunk_size=5,
        enable_dns_cache=True,
        enable_result_cache=True
    )
    scanner = get_non_blocking_scanner(config)
    
    # Demo targets
    targets = [
        "google.com",
        "github.com", 
        "stackoverflow.com",
        "httpbin.org",
        "example.com"
    ]
    
    scan_types = ["dns", "port", "http"]
    
    print(f"\n🎯 Scanning {len(targets)} targets with {len(scan_types)} scan types...")
    
    # Progress callback
    async def progress_callback(progress: float, completed: int, total: int):
        
    """progress_callback function."""
print(f"  Progress: {progress:.1f}% ({completed}/{total})")
    
    start_time = time.time()
    results = await scanner.batch_scan_with_progress(
        targets, 
        scan_types, 
        progress_callback
    )
    total_time = time.time() - start_time
    
    print(f"\n✅ Scan completed in {total_time:.3f}s")
    
    # Display results
    print(f"\n📋 Scan Results:")
    for target, target_results in results.items():
        print(f"\n  🎯 {target}:")
        for result in target_results:
            status = "✅" if result.success else "❌"
            print(f"    {status} {result.scan_type}: {result.duration:.3f}s")
            if result.success and result.data:
                if result.scan_type == "dns":
                    print(f"      IP: {result.data}")
                elif result.scan_type == "port":
                    open_ports = [port for port, status in result.data.items() if status == "open"]
                    if open_ports:
                        print(f"      Open ports: {open_ports}")
                elif result.scan_type == "http":
                    for url, data in result.data.items():
                        if "status_code" in data:
                            print(f"      {url}: {data['status_code']}")
    
    # Get scanner stats
    stats = scanner.get_scan_stats()
    print(f"\n📊 Scanner Statistics:")
    for scan_type, stat in stats.items():
        print(f"  {scan_type}: {stat['total_scans']} scans, avg {stat['avg_duration']:.3f}s")
    
    await scanner.close()

async def demo_concurrent_operations():
    """Demo concurrent operations without blocking."""
    print("\n\n⚡ Demo: Concurrent Operations")
    print("=" * 60)
    
    
    helper = get_async_helper()
    
    # Define different types of operations
    operations = [
        ("dns_lookup", lambda: helper.dns_lookup("google.com")),
        ("http_request", lambda: helper.http_request("http://httpbin.org/get")),
        ("port_scan", lambda: helper.port_scan("google.com", 80)),
        ("file_check", lambda: helper.file_operation("non_existent_file.txt", "exists")),
        ("crypto_hash", lambda: helper.crypto_operation("hash", b"test data"))
    ]
    
    print(f"\n🔄 Executing {len(operations)} different operations concurrently...")
    
    start_time = time.time()
    results = await helper.concurrent_operations(operations)
    total_time = time.time() - start_time
    
    print(f"\n✅ Concurrent operations completed in {total_time:.3f}s")
    
    for operation_name, result in results.items():
        status = "✅" if result.success else "❌"
        print(f"  {status} {operation_name}: {result.duration:.3f}s")
        if result.success and result.data:
            if operation_name == "dns_lookup":
                print(f"    Result: {result.data}")
            elif operation_name == "http_request":
                print(f"    Status: {result.data.get('status_code')}")
            elif operation_name == "port_scan":
                print(f"    Port open: {result.data}")
    
    await helper.close()

async def demo_memory_efficient_processing():
    """Demo memory-efficient processing of large datasets."""
    print("\n\n💾 Demo: Memory-Efficient Processing")
    print("=" * 60)
    
    
    helper = get_async_helper()
    
    # Simulate large dataset
    large_dataset = [f"target_{i}.com" for i in range(1000)]
    
    print(f"\n📊 Processing {len(large_dataset)} targets in chunks...")
    
    async def process_target(target: str):
        """Process a single target."""
        # Simulate some processing
        await asyncio.sleep(0.01)
        return f"Processed {target}"
    
    start_time = time.time()
    results = await helper.stream_processing(large_dataset, process_target, chunk_size=50)
    total_time = time.time() - start_time
    
    print(f"\n✅ Processed {len(results)} targets in {total_time:.3f}s")
    print(f"  Average time per target: {total_time/len(results):.4f}s")
    
    # Show sample results
    print(f"\n📋 Sample Results:")
    for i, result in enumerate(results[:5]):
        print(f"  {i+1}. {result}")
    
    await helper.close()

async def main():
    """Main demo function."""
    print("🚀 Non-Blocking Operations Demo")
    print("=" * 60)
    print("This demo showcases how to avoid blocking operations in core scanning loops")
    print("by extracting heavy I/O to dedicated async helpers.\n")
    
    try:
        # Demo async helpers
        await demo_async_helpers()
        
        # Demo non-blocking scanner
        await demo_non_blocking_scanner()
        
        # Demo concurrent operations
        await demo_concurrent_operations()
        
        # Demo memory-efficient processing
        await demo_memory_efficient_processing()
        
        print("\n\n✅ All demos completed successfully!")
        print("\n🎯 Key Benefits:")
        print("  • Non-blocking I/O operations")
        print("  • Concurrent processing")
        print("  • Memory-efficient batch processing")
        print("  • Dedicated async helpers for heavy operations")
        print("  • Progress tracking and statistics")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # Run the demo
    asyncio.run(main()) 