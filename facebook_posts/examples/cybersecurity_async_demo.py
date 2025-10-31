from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
from cybersecurity_toolkit import (
from typing import Any, List, Dict, Optional
import logging
"""
Cybersecurity Async/Await Demo
Demonstrates proper use of 'def' for CPU-bound and 'async def' for I/O-bound operations
"""

    SecurityConfig, ScanResult, hash_password, verify_password,
    generate_secure_token, validate_ip_address, scan_single_port,
    scan_port_range, check_ssl_certificate, fetch_http_headers,
    scan_common_ports, check_dns_records, read_file_async,
    write_file_async, execute_command_async, scan_network_ports,
    check_ssl_security, analyze_web_security
)


async def demo_cpu_bound_operations():
    """Demo CPU-bound operations using 'def' functions."""
    print("=== CPU-Bound Operations Demo ===")
    
    config = SecurityConfig(iterations=10000)  # Reduced for demo
    
    # Password hashing (CPU intensive)
    start_time = time.time()
    password = "secure_password_123"
    hashed = hash_password(password, config)
    hash_time = time.time() - start_time
    print(f"Password hashing: {hash_time:.4f}s")
    
    # Password verification (CPU intensive)
    start_time = time.time()
    is_valid = verify_password(password, hashed, config)
    verify_time = time.time() - start_time
    print(f"Password verification: {verify_time:.4f}s")
    print(f"Password valid: {is_valid}")
    
    # Token generation (CPU intensive)
    start_time = time.time()
    token = generate_secure_token(32)
    token_time = time.time() - start_time
    print(f"Token generation: {token_time:.4f}s")
    print(f"Generated token: {token[:20]}...")
    
    # IP validation (CPU intensive)
    test_ips = ["192.168.1.1", "256.256.256.256", "localhost", "2001:db8::1"]
    for ip in test_ips:
        is_valid_ip = validate_ip_address(ip)
        print(f"IP {ip}: {'Valid' if is_valid_ip else 'Invalid'}")


async def demo_io_bound_operations():
    """Demo I/O-bound operations using 'async def' functions."""
    print("\n=== I/O-Bound Operations Demo ===")
    
    config = SecurityConfig(timeout=5.0)  # Shorter timeout for demo
    
    # Port scanning (I/O intensive)
    print("Scanning common ports on localhost...")
    start_time = time.time()
    open_ports = await scan_common_ports("localhost", config)
    scan_time = time.time() - start_time
    print(f"Port scan completed in {scan_time:.4f}s")
    print(f"Open ports found: {len(open_ports)}")
    for port_result in open_ports:
        print(f"  Port {port_result.port}: {port_result.response_time:.3f}s")
    
    # SSL certificate check (I/O intensive)
    print("\nChecking SSL certificate for google.com...")
    start_time = time.time()
    ssl_result = await check_ssl_certificate("google.com", 443, config)
    ssl_time = time.time() - start_time
    print(f"SSL check completed in {ssl_time:.4f}s")
    print(f"SSL valid: {ssl_result.get('is_valid', False)}")
    
    # HTTP headers fetch (I/O intensive)
    print("\nFetching HTTP headers from example.com...")
    start_time = time.time()
    headers_result = await fetch_http_headers("https://example.com", config)
    headers_time = time.time() - start_time
    print(f"Headers fetch completed in {headers_time:.4f}s")
    print(f"Status code: {headers_result.get('status_code', 'N/A')}")
    
    # DNS resolution (I/O intensive)
    print("\nResolving DNS for google.com...")
    start_time = time.time()
    dns_result = await check_dns_records("google.com", config)
    dns_time = time.time() - start_time
    print(f"DNS resolution completed in {dns_time:.4f}s")
    print(f"IP address: {dns_result.get('ip_address', 'N/A')}")


async def demo_concurrent_operations():
    """Demo concurrent async operations."""
    print("\n=== Concurrent Operations Demo ===")
    
    config = SecurityConfig(timeout=3.0)
    
    # Concurrent port scanning
    hosts = ["localhost", "127.0.0.1"]
    ports = [80, 443, 22, 21, 25]
    
    print("Concurrent port scanning...")
    start_time = time.time()
    
    tasks = []
    for host in hosts:
        for port in ports:
            task = scan_single_port(host, port, config)
            tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    concurrent_time = time.time() - start_time
    
    open_count = sum(1 for r in results if isinstance(r, ScanResult) and r.is_open)
    print(f"Concurrent scan completed in {concurrent_time:.4f}s")
    print(f"Total scans: {len(tasks)}, Open ports: {open_count}")
    
    # Concurrent SSL checks
    ssl_hosts = ["google.com", "github.com", "stackoverflow.com"]
    print("\nConcurrent SSL certificate checks...")
    start_time = time.time()
    
    ssl_tasks = [check_ssl_certificate(host, 443, config) for host in ssl_hosts]
    ssl_results = await asyncio.gather(*ssl_tasks, return_exceptions=True)
    ssl_concurrent_time = time.time() - start_time
    
    valid_count = sum(1 for r in ssl_results if isinstance(r, dict) and r.get('is_valid', False))
    print(f"Concurrent SSL checks completed in {ssl_concurrent_time:.4f}s")
    print(f"Valid certificates: {valid_count}/{len(ssl_hosts)}")


async def demo_file_operations():
    """Demo async file operations."""
    print("\n=== Async File Operations Demo ===")
    
    # Write file asynchronously
    test_content = "This is a test file created by async file operations.\n"
    test_content += "Line 2: Testing async I/O operations.\n"
    test_content += "Line 3: Cybersecurity toolkit demo.\n"
    
    print("Writing test file asynchronously...")
    start_time = time.time()
    write_success = await write_file_async("test_async_file.txt", test_content)
    write_time = time.time() - start_time
    print(f"File write completed in {write_time:.4f}s")
    print(f"Write success: {write_success}")
    
    # Read file asynchronously
    print("Reading test file asynchronously...")
    start_time = time.time()
    read_content = await read_file_async("test_async_file.txt")
    read_time = time.time() - start_time
    print(f"File read completed in {read_time:.4f}s")
    print(f"Content length: {len(read_content)} characters")
    print(f"Content preview: {read_content[:50]}...")


async def demo_roro_pattern():
    """Demo RORO pattern usage."""
    print("\n=== RORO Pattern Demo ===")
    
    # Network port scanning with RORO
    print("Network port scanning using RORO pattern...")
    scan_params = {
        'host': 'localhost',
        'start_port': 80,
        'end_port': 90,
        'config': SecurityConfig(timeout=2.0)
    }
    
    start_time = time.time()
    scan_result = scan_network_ports(scan_params)
    roro_time = time.time() - start_time
    
    print(f"RORO scan completed in {roro_time:.4f}s")
    print(f"Open ports found: {scan_result['open_ports']}")
    
    # SSL security check with RORO
    print("\nSSL security check using RORO pattern...")
    ssl_params = {
        'host': 'google.com',
        'port': 443,
        'config': SecurityConfig(timeout=5.0)
    }
    
    start_time = time.time()
    ssl_result = check_ssl_security(ssl_params)
    ssl_roro_time = time.time() - start_time
    
    print(f"RORO SSL check completed in {ssl_roro_time:.4f}s")
    print(f"SSL secure: {ssl_result['is_secure']}")
    
    # Web security analysis with RORO
    print("\nWeb security analysis using RORO pattern...")
    web_params = {
        'url': 'https://example.com',
        'config': SecurityConfig(timeout=5.0)
    }
    
    start_time = time.time()
    web_result = analyze_web_security(web_params)
    web_roro_time = time.time() - start_time
    
    print(f"RORO web analysis completed in {web_roro_time:.4f}s")
    print(f"Website accessible: {web_result['is_accessible']}")


async def demo_performance_comparison():
    """Demo performance comparison between sync and async operations."""
    print("\n=== Performance Comparison Demo ===")
    
    config = SecurityConfig(timeout=2.0)
    ports = [80, 443, 22, 21, 25, 53, 110, 143, 993, 995]
    
    # Sequential async operations
    print("Sequential async port scanning...")
    start_time = time.time()
    for port in ports:
        await scan_single_port("localhost", port, config)
    sequential_time = time.time() - start_time
    print(f"Sequential time: {sequential_time:.4f}s")
    
    # Concurrent async operations
    print("Concurrent async port scanning...")
    start_time = time.time()
    tasks = [scan_single_port("localhost", port, config) for port in ports]
    await asyncio.gather(*tasks, return_exceptions=True)
    concurrent_time = time.time() - start_time
    print(f"Concurrent time: {concurrent_time:.4f}s")
    
    speedup = sequential_time / concurrent_time if concurrent_time > 0 else 0
    print(f"Speedup: {speedup:.2f}x")


async def main():
    """Main demo function."""
    print("Cybersecurity Toolkit - Async/Await Patterns Demo")
    print("=" * 50)
    
    try:
        await demo_cpu_bound_operations()
        await demo_io_bound_operations()
        await demo_concurrent_operations()
        await demo_file_operations()
        await demo_roro_pattern()
        await demo_performance_comparison()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        
    except Exception as e:
        print(f"Demo error: {e}")


match __name__:
    case "__main__":
    asyncio.run(main()) 