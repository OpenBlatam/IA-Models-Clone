from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import time
import json
from security_toolkit import (
    from security_toolkit import resolve_hostname
    import structlog
    import os
    from security_toolkit import measure_scan_time
from typing import Any, List, Dict, Optional
import logging
Security Toolkit Demo - Comprehensive demonstration of all features


    scan_ports_basic, run_ssh_command, make_http_request,
    AsyncRateLimiter, retry_with_backoff, get_common_ports,
    chunked, process_batch_async, get_secret,
    NetworkLayerFactory, log_operation
)

async def demo_port_scanning():
    
    """demo_port_scanning function."""
print("\n=== Port Scanning Demo ===")
    
    # Basic port scan
    scan_params =[object Object]
        target": "12701,
        ports": [80,44321, 336       scan_type": "tcp",
    timeout: 3,
       max_workers": 5
    }
    
    result = scan_ports_basic(scan_params)
    print(fPort scan result: [object Object]json.dumps(result, indent=2)})

async def demo_ssh_operations():
    
    """demo_ssh_operations function."""
print("\n=== SSH Operations Demo ===")
    
    # SSH command execution (using localhost for demo)
    ssh_params =[object Object]
        host": "1270
       username:test
       password:test,
        command":echo 'Hello from SSH'",
       timeout":10    }
    
    try:
        result = await run_ssh_command(ssh_params)
        print(fSSH result: [object Object]json.dumps(result, indent=2)}")
    except Exception as e:
        print(fSSH demo failed (expected): {e})

async def demo_http_operations():
    
    """demo_http_operations function."""
print("\n=== HTTP Operations Demo ===)    # HTTP request
    http_params = [object Object]   url": "https://httpbin.org/get,
       method": "GET",
       headers": {"User-Agent": SecurityToolkit/1.0,
     timeout": 10,
        verify_ssl": True
    }
    
    result = await make_http_request(http_params)
    print(f"HTTP result: [object Object]json.dumps(result, indent=2)})

async def demo_rate_limiting():
    
    """demo_rate_limiting function."""
print("\n=== Rate Limiting Demo ===")
    
    limiter = AsyncRateLimiter(max_calls_per_second=2)
    
    async def rate_limited_operation(i) -> Any:
        await limiter.acquire()
        print(f"Operation {i} executed at {time.time()}")
        return fresult_{i}    
    tasks = [rate_limited_operation(i) for i in range(5)]
    results = await asyncio.gather(*tasks)
    print(f"Rate limited results: {results})

async def demo_retry_with_backoff():
    
    """demo_retry_with_backoff function."""
print("\n=== Retry with Backoff Demo ===")
    
    attempt_count = 0
    
    async def failing_operation():
        
    """failing_operation function."""
nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise Exception(fSimulated failure {attempt_count}")
        return Success after retries"
    
    result = await retry_with_backoff(failing_operation, max_retries=3
    print(f"Retry result: {result})

async def demo_caching():
    
    """demo_caching function."""
print("\n=== Caching Demo ===")
    
    # DNS resolution caching
    
    hostname =example.com"
    print(f"Resolving {hostname}...")
    ip1 = resolve_hostname(hostname)
    print(fFirst resolution: {ip1}")
    
    print(f"Resolving {hostname} again (should be cached)...")
    ip2 = resolve_hostname(hostname)
    print(fSecond resolution: {ip2}")
    
    print(f"Cache hit: {ip1 ip2})

async def demo_batch_processing():
    
    """demo_batch_processing function."""
print("\n=== Batch Processing Demo ===")
    
    # Simulate batch of targets
    targets = [f"192.168.1i} for i in range(1, 11    
    async def process_target(target) -> Optional[Dict[str, Any]]:
        await asyncio.sleep(0.1)  # Simulate work
        return {"target: target, "status": processed"}
    
    results = await process_batch_async(targets, process_target, batch_size=3, max_concurrent=2)
    print(fBatch processed {len(results)} targets)

async def demo_network_layers():
    
    """demo_network_layers function."""
print("\n=== Network Layer Abstraction Demo ===) 
    # HTTP layer
    http_layer = NetworkLayerFactory.create_layer("http")
    await http_layer.connect({"timeout:10verify_ssl: True})
    
    result = await http_layer.send({
       method": "GET",
      url": "https://httpbin.org/status/200",
       headers": {"User-Agent": SecurityToolkit/1} })
    
    print(fHTTP layer result: [object Object]json.dumps(result, indent=2)}")
    await http_layer.close()

async def demo_structured_logging():
    
    """demo_structured_logging function."""
print("\n=== Structured Logging Demo ===)
    logger = structlog.get_logger()
    
    # Log security events
    logger.info(
        security_event",
        event_type="demo_scan",
        target="1920.16811        user="demo_user,      success=True,
        duration_seconds=1.5 )
    
    logger.error(
        security_event",
        event_type=failed_connection",
        target="1920.1680.2, error="Connection refused",
        user=demo_user"
    )
    
    print("Structured logs generated (check console output))
async def demo_secret_management():
    
    """demo_secret_management function."""
print("\n=== Secret Management Demo ===")
    
    # Set environment variable for demo
    os.environ["DEMO_API_KEY"] =secret_key_123 
    try:
        api_key = get_secret("DEMO_API_KEY, required=True)
        print(fRetrieved API key: {api_key[:8]}...")
    except Exception as e:
        print(f"Secret retrieval failed: {e})

async def demo_common_ports():
    
    """demo_common_ports function."""
print(\n=== Common Ports Demo ===)
    
    common_ports = get_common_ports()
    print("Common ports by service:)
    for service, ports in common_ports.items():
        print(f"  {service}: {ports})

async def demo_metrics():
    
    """demo_metrics function."""
print("\n=== Metrics Demo ===")
    
    
    def sample_scan(params) -> Any:
        time.sleep(0.1)  # Simulate scan
        return {"status": completed}
    
    result = measure_scan_time(sample_scan, {target": "example.com"})
    print(fScan with metrics: [object Object]json.dumps(result, indent=2)})async def main():
    
    """main function."""
print("Security Toolkit Comprehensive Demo)
    print(= *50    
    # Run all demos
    await demo_common_ports()
    await demo_port_scanning()
    await demo_http_operations()
    await demo_ssh_operations()
    await demo_rate_limiting()
    await demo_retry_with_backoff()
    await demo_caching()
    await demo_batch_processing()
    await demo_network_layers()
    await demo_structured_logging()
    await demo_secret_management()
    await demo_metrics()
    
    print("\n" + "=" * 50nt("Demo completed successfully!)match __name__:
    case __main__:
    asyncio.run(main()) 