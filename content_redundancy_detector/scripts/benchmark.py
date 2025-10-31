#!/usr/bin/env python3
"""
Performance Benchmark Script
Tests API endpoints and measures performance metrics
"""

import asyncio
import time
import statistics
from typing import List, Dict
import httpx


class BenchmarkResult:
    """Stores benchmark results"""
    def __init__(self, endpoint: str, method: str = "GET"):
        self.endpoint = endpoint
        self.method = method
        self.requests: List[float] = []
        self.errors: int = 0
        self.total_time: float = 0.0
    
    def add_request(self, duration: float, success: bool = True):
        """Add a request result"""
        if success:
            self.requests.append(duration)
        else:
            self.errors += 1
    
    def get_stats(self) -> Dict:
        """Get statistics"""
        if not self.requests:
            return {
                "endpoint": self.endpoint,
                "method": self.method,
                "requests": 0,
                "errors": self.errors,
                "error": "No successful requests"
            }
        
        return {
            "endpoint": self.endpoint,
            "method": self.method,
            "total_requests": len(self.requests),
            "errors": self.errors,
            "success_rate": len(self.requests) / (len(self.requests) + self.errors) * 100,
            "mean": statistics.mean(self.requests),
            "median": statistics.median(self.requests),
            "p95": statistics.quantiles(self.requests, n=20)[18] if len(self.requests) > 1 else self.requests[0],
            "p99": statistics.quantiles(self.requests, n=100)[98] if len(self.requests) > 1 else self.requests[0],
            "min": min(self.requests),
            "max": max(self.requests),
            "stdev": statistics.stdev(self.requests) if len(self.requests) > 1 else 0.0
        }


async def benchmark_endpoint(
    client: httpx.AsyncClient,
    endpoint: str,
    method: str = "GET",
    payload: dict = None,
    iterations: int = 100,
    concurrency: int = 10
) -> BenchmarkResult:
    """Benchmark a single endpoint"""
    result = BenchmarkResult(endpoint, method)
    
    async def single_request():
        """Execute a single request"""
        start = time.time()
        try:
            if method == "GET":
                response = await client.get(endpoint)
            elif method == "POST":
                response = await client.post(endpoint, json=payload)
            else:
                response = await client.request(method, endpoint, json=payload)
            
            duration = time.time() - start
            success = response.status_code < 400
            result.add_request(duration, success)
        except Exception as e:
            duration = time.time() - start
            result.add_request(duration, False)
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(concurrency)
    
    async def bounded_request():
        async with semaphore:
            await single_request()
    
    # Run concurrent requests
    tasks = [bounded_request() for _ in range(iterations)]
    await asyncio.gather(*tasks)
    
    return result


async def run_benchmarks(base_url: str = "http://127.0.0.1:8000"):
    """Run all benchmarks"""
    print("=" * 60)
    print("🚀 Performance Benchmark Suite")
    print("=" * 60)
    print(f"\nBase URL: {base_url}\n")
    
    endpoints = [
        ("/health", "GET"),
        ("/api/v1/health", "GET"),
        ("/api/v1/metrics", "GET"),
        ("/policies/summary", "GET"),
        ("/cache/stats", "GET"),
        ("/rate-limit/status", "GET"),
    ]
    
    # Add POST endpoints if needed
    # endpoints.append(("/api/v1/analyze", "POST", {"content": "Test content"}))
    
    results = []
    
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        for endpoint_config in endpoints:
            endpoint = endpoint_config[0]
            method = endpoint_config[1]
            payload = endpoint_config[2] if len(endpoint_config) > 2 else None
            
            print(f"\n📊 Benchmarking: {method} {endpoint}")
            print("-" * 60)
            
            result = await benchmark_endpoint(
                client,
                endpoint,
                method,
                payload,
                iterations=100,
                concurrency=10
            )
            
            stats = result.get_stats()
            results.append(stats)
            
            if "error" in stats:
                print(f"❌ {stats['error']}")
            else:
                print(f"✅ Total Requests: {stats['total_requests']}")
                print(f"   Errors: {stats['errors']}")
                print(f"   Success Rate: {stats['success_rate']:.2f}%")
                print(f"   Mean: {stats['mean']*1000:.2f}ms")
                print(f"   Median: {stats['median']*1000:.2f}ms")
                print(f"   P95: {stats['p95']*1000:.2f}ms")
                print(f"   P99: {stats['p99']*1000:.2f}ms")
                print(f"   Min: {stats['min']*1000:.2f}ms")
                print(f"   Max: {stats['max']*1000:.2f}ms")
    
    # Summary
    print("\n" + "=" * 60)
    print("📈 Summary")
    print("=" * 60)
    
    for stats in results:
        if "mean" in stats:
            print(f"\n{stats['endpoint']}:")
            print(f"  Mean Response Time: {stats['mean']*1000:.2f}ms")
            print(f"  P95 Response Time: {stats['p95']*1000:.2f}ms")
            print(f"  Success Rate: {stats['success_rate']:.2f}%")


if __name__ == "__main__":
    import sys
    
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    
    asyncio.run(run_benchmarks(base_url))


