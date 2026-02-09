from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import sys
import os
from cybersecurity.scanners.async_helpers import AsyncHelperManager, AsyncHelperConfig
from cybersecurity.scanners.port_scanner import PortScanConfig, PortScanner
from cybersecurity.middleware import apply_middleware, get_middleware_manager
        import traceback
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Demo: Non-Blocking Async Scanning with Dedicated Async Helpers
Showcases how to avoid blocking operations in core scanning loops.
"""


# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def demo_basic_async_scanning():
    """Demo basic async scanning without blocking operations."""
    print("🔍 Demo: Basic Async Scanning")
    print("=" * 50)
    
    # Configure async helpers
    async_config = AsyncHelperConfig(
        timeout=5.0,
        max_workers=20,
        retry_attempts=2,
        chunk_size=512
    )
    
    # Create helper manager
    async with AsyncHelperManager(async_config) as helper_manager:
        # Test targets
        targets = ["127.0.0.1", "localhost"]
        ports = [22, 80, 443, 8080, 3306]
        
        print(f"📋 Scanning {len(targets)} targets with {len(ports)} ports each")
        print(f"⚙️  Configuration: timeout={async_config.timeout}s, workers={async_config.max_workers}")
        
        start_time = time.time()
        
        # Scan all targets concurrently
        all_results = {}
        for target in targets:
            print(f"\n🎯 Scanning {target}...")
            results = await helper_manager.comprehensive_scan_async(target, ports)
            all_results[target] = results
            
            # Display results
            open_ports = [r for r in results.get("results", []) if r.get("is_open")]
            print(f"   ✅ Found {len(open_ports)} open ports")
            for result in open_ports:
                print(f"      - Port {result['port']}: {result.get('service_name', 'unknown')}")
        
        total_duration = time.time() - start_time
        print(f"\n⏱️  Total scan duration: {total_duration:.2f}s")
        print(f"🚀 Scan rate: {len(targets) * len(ports) / total_duration:.1f} ports/second")

async def demo_network_io_helpers():
    """Demo dedicated network I/O helpers."""
    print("\n🌐 Demo: Network I/O Helpers")
    print("=" * 50)
    
    async_config = AsyncHelperConfig(timeout=3.0, max_workers=10)
    
    async with AsyncHelperManager(async_config) as helper_manager:
        test_host = "127.0.0.1"
        test_ports = [22, 80, 443, 8080]
        
        print(f"🔌 Testing network I/O helpers with {test_host}")
        
        for port in test_ports:
            print(f"\n📡 Testing port {port}:")
            
            # TCP connection test
            tcp_success, tcp_duration, tcp_error = await helper_manager.network_io.tcp_connect(
                test_host, port
            )
            print(f"   TCP: {'✅' if tcp_success else '❌'} ({tcp_duration:.3f}s)")
            if tcp_error:
                print(f"      Error: {tcp_error}")
            
            if tcp_success:
                # Banner grab test
                banner_success, banner_content, banner_duration = await helper_manager.network_io.banner_grab(
                    test_host, port
                )
                print(f"   Banner: {'✅' if banner_success else '❌'} ({banner_duration:.3f}s)")
                if banner_success and banner_content:
                    print(f"      Content: {banner_content[:100]}...")
                
                # SSL test for SSL ports
                if port in [443, 8443]:
                    ssl_success, ssl_duration, ssl_info = await helper_manager.network_io.ssl_connect(
                        test_host, port
                    )
                    print(f"   SSL: {'✅' if ssl_success else '❌'} ({ssl_duration:.3f}s)")
                    if ssl_success:
                        print(f"      Subject: {ssl_info.get('subject', {}).get('commonName', 'N/A')}")

async def demo_data_processing_helpers():
    """Demo data processing helpers for large datasets."""
    print("\n📊 Demo: Data Processing Helpers")
    print("=" * 50)
    
    async_config = AsyncHelperConfig(chunk_size=100, max_workers=5)
    
    async with AsyncHelperManager(async_config) as helper_manager:
        # Generate large dataset
        large_dataset = [
            {"target": f"192.168.1.{i}", "port": 80, "is_open": i % 2 == 0}
            for i in range(1, 1001)
        ]
        
        print(f"📈 Processing {len(large_dataset)} scan results...")
        
        start_time = time.time()
        
        # Process in chunks (non-blocking)
        processed_results = await helper_manager.data_processing.process_large_dataset(
            large_dataset, chunk_size=200
        )
        
        processing_duration = time.time() - start_time
        
        print(f"✅ Processed {len(processed_results)} items in {processing_duration:.2f}s")
        print(f"🚀 Processing rate: {len(processed_results) / processing_duration:.1f} items/second")
        
        # Analyze results
        analysis = await helper_manager.data_processing.analyze_scan_data(large_dataset)
        print(f"\n📊 Analysis Results:")
        print(f"   Total scans: {analysis['total_scans']}")
        print(f"   Open ports: {analysis['open_ports']}")
        print(f"   Success rate: {analysis['success_rate']:.1%}")
        print(f"   Avg response time: {analysis['avg_response_time']:.3f}s")

async def demo_file_io_helpers():
    """Demo file I/O helpers for async file operations."""
    print("\n📁 Demo: File I/O Helpers")
    print("=" * 50)
    
    async_config = AsyncHelperConfig()
    
    async with AsyncHelperManager(async_config) as helper_manager:
        test_file = "test_scan_results.json"
        test_data = {
            "scan_info": {
                "target": "192.168.1.1",
                "timestamp": time.time(),
                "ports_scanned": [22, 80, 443, 8080]
            },
            "results": [
                {"port": 22, "is_open": True, "service": "ssh"},
                {"port": 80, "is_open": True, "service": "http"},
                {"port": 443, "is_open": False, "service": "https"},
                {"port": 8080, "is_open": False, "service": "http-proxy"}
            ]
        }
        
        print(f"💾 Writing test data to {test_file}...")
        
        # Write JSON file (non-blocking)
        write_success, write_message, write_duration = await helper_manager.file_io.write_json_async(
            test_file, test_data
        )
        
        print(f"   Write: {'✅' if write_success else '❌'} ({write_duration:.3f}s)")
        if not write_success:
            print(f"      Error: {write_message}")
        
        if write_success:
            print(f"📖 Reading data back from {test_file}...")
            
            # Read JSON file (non-blocking)
            read_success, read_data, read_duration = await helper_manager.file_io.read_json_async(test_file)
            
            print(f"   Read: {'✅' if read_success else '❌'} ({read_duration:.3f}s)")
            if read_success:
                print(f"      Data integrity: {'✅' if read_data == test_data else '❌'}")
                print(f"      Items read: {len(read_data.get('results', []))}")
            
            # Cleanup
            try:
                os.remove(test_file)
                print(f"🧹 Cleaned up {test_file}")
            except:
                pass

async def demo_middleware_integration():
    """Demo middleware integration with async scanning."""
    print("\n🔧 Demo: Middleware Integration")
    print("=" * 50)
    
    # Apply middleware to scanning functions
    @apply_middleware(operation_name="async_port_scan")
    async def scan_with_middleware(target: str, port: int) -> dict:
        """Scan function with middleware applied."""
        async_config = AsyncHelperConfig(timeout=2.0)
        async with AsyncHelperManager(async_config) as helper_manager:
            result = await helper_manager.network_io.tcp_connect(target, port)
            return {
                "target": target,
                "port": port,
                "is_open": result[0],
                "duration": result[1],
                "error": result[2]
            }
    
    # Test targets
    test_cases = [
        ("127.0.0.1", 22),
        ("127.0.0.1", 80),
        ("127.0.0.1", 443),
        ("localhost", 8080)
    ]
    
    print(f"🔍 Testing {len(test_cases)} scan operations with middleware...")
    
    # Run scans concurrently
    tasks = [scan_with_middleware(target, port) for target, port in test_cases]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Display results
    successful_scans = 0
    for i, result in enumerate(results):
        if isinstance(result, dict):
            successful_scans += 1
            status = "✅" if result.get("is_open") else "❌"
            print(f"   {status} {result['target']}:{result['port']} ({result['duration']:.3f}s)")
        else:
            print(f"   ❌ Error: {result}")
    
    print(f"\n📊 Scan Summary:")
    print(f"   Successful scans: {successful_scans}/{len(test_cases)}")
    print(f"   Success rate: {successful_scans/len(test_cases):.1%}")
    
    # Get middleware metrics
    manager = get_middleware_manager()
    metrics = manager.metrics.get_metrics_summary()
    
    if "async_port_scan" in metrics.get("operations", {}):
        op_metrics = metrics["operations"]["async_port_scan"]
        print(f"   Avg duration: {op_metrics.get('avg_duration', 0):.3f}s")
        print(f"   Success rate: {op_metrics.get('success_rate', 0):.1%}")

async def demo_performance_comparison():
    """Demo performance comparison between blocking and non-blocking approaches."""
    print("\n⚡ Demo: Performance Comparison")
    print("=" * 50)
    
    test_target = "127.0.0.1"
    test_ports = list(range(80, 90))  # Ports 80-89
    
    print(f"🎯 Testing {len(test_ports)} ports on {test_target}")
    
    # Test 1: Traditional blocking approach (simulated)
    print("\n🔴 Traditional Blocking Approach (simulated):")
    start_time = time.time()
    
    # Simulate blocking operations
    for port in test_ports:
        # Simulate blocking socket operation
        await asyncio.sleep(0.1)  # Simulate blocking I/O
    
    blocking_duration = time.time() - start_time
    print(f"   Duration: {blocking_duration:.2f}s")
    print(f"   Rate: {len(test_ports) / blocking_duration:.1f} ports/second")
    
    # Test 2: Non-blocking async approach
    print("\n🟢 Non-Blocking Async Approach:")
    start_time = time.time()
    
    async_config = AsyncHelperConfig(timeout=1.0, max_workers=len(test_ports))
    async with AsyncHelperManager(async_config) as helper_manager:
        # All operations run concurrently
        tasks = [
            helper_manager.network_io.tcp_connect(test_target, port)
            for port in test_ports
        ]
        results = await asyncio.gather(*tasks)
    
    async_duration = time.time() - start_time
    print(f"   Duration: {async_duration:.2f}s")
    print(f"   Rate: {len(test_ports) / async_duration:.1f} ports/second")
    
    # Performance improvement
    improvement = (blocking_duration - async_duration) / blocking_duration * 100
    print(f"\n📈 Performance Improvement: {improvement:.1f}% faster")
    
    # Success rate
    successful_scans = len([r for r in results if r[0]])
    print(f"📊 Success Rate: {successful_scans}/{len(test_ports)} ({successful_scans/len(test_ports):.1%})")

async def main():
    """Run all demos."""
    print("🚀 Non-Blocking Async Scanning Demo")
    print("=" * 60)
    print("This demo showcases how to avoid blocking operations in core scanning loops")
    print("by extracting heavy I/O operations to dedicated async helpers.\n")
    
    try:
        # Run all demos
        await demo_basic_async_scanning()
        await demo_network_io_helpers()
        await demo_data_processing_helpers()
        await demo_file_io_helpers()
        await demo_middleware_integration()
        await demo_performance_comparison()
        
        print("\n✅ All demos completed successfully!")
        print("\n🎯 Key Benefits:")
        print("   • Non-blocking I/O operations")
        print("   • Concurrent processing")
        print("   • Better resource utilization")
        print("   • Improved performance")
        print("   • Scalable architecture")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        traceback.print_exc()

match __name__:
    case "__main__":
    asyncio.run(main()) 