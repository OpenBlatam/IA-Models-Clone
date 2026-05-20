#!/usr/bin/env python3
"""
Load Testing Script
Tests API performance under load
"""

import asyncio
import aiohttp
import time
from typing import List, Dict, Any
import argparse
import statistics


async def make_request(
    session: aiohttp.ClientSession,
    url: str,
    method: str = "GET",
    data: Any = None
) -> tuple[bool, float]:
    """Make HTTP request and measure duration"""
    start_time = time.time()
    
    try:
        if method == "GET":
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                await resp.read()
                success = resp.status == 200
        elif method == "POST":
            async with session.post(url, json=data, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                await resp.read()
                success = resp.status in (200, 201)
        else:
            success = False
        
        duration = time.time() - start_time
        return success, duration
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"Request failed: {e}")
        return False, duration


async def run_load_test(
    url: str,
    num_requests: int,
    concurrent: int,
    method: str = "GET",
    data: Any = None
) -> Dict[str, Any]:
    """Run load test"""
    print(f"Starting load test: {num_requests} requests, {concurrent} concurrent")
    
    async with aiohttp.ClientSession() as session:
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrent)
        
        async def bounded_request():
            async with semaphore:
                return await make_request(session, url, method, data)
        
        # Run requests
        start_time = time.time()
        tasks = [bounded_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Calculate statistics
        successes = [r[0] for r in results]
        durations = [r[1] for r in results if r[0]]  # Only successful requests
        
        success_count = sum(successes)
        failure_count = num_requests - success_count
        
        stats = {
            "total_requests": num_requests,
            "successful_requests": success_count,
            "failed_requests": failure_count,
            "success_rate": success_count / num_requests * 100,
            "total_time": total_time,
            "requests_per_second": num_requests / total_time,
            "avg_response_time": statistics.mean(durations) if durations else 0,
            "min_response_time": min(durations) if durations else 0,
            "max_response_time": max(durations) if durations else 0,
            "median_response_time": statistics.median(durations) if durations else 0,
            "p95_response_time": statistics.quantiles(durations, n=20)[18] if len(durations) > 20 else 0,
            "p99_response_time": statistics.quantiles(durations, n=100)[98] if len(durations) > 100 else 0,
        }
        
        return stats


def main():
    parser = argparse.ArgumentParser(description="Load test for Dermatology AI API")
    parser.add_argument("--url", default="http://localhost:8006/health", help="URL to test")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests")
    parser.add_argument("--concurrent", type=int, default=10, help="Concurrent requests")
    parser.add_argument("--method", default="GET", help="HTTP method")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Load Test Configuration")
    print(f"{'='*60}")
    print(f"URL: {args.url}")
    print(f"Requests: {args.requests}")
    print(f"Concurrent: {args.concurrent}")
    print(f"Method: {args.method}")
    print(f"{'='*60}\n")
    
    # Run test
    stats = asyncio.run(run_load_test(
        args.url,
        args.requests,
        args.concurrent,
        args.method
    ))
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Load Test Results")
    print(f"{'='*60}")
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Successful: {stats['successful_requests']}")
    print(f"Failed: {stats['failed_requests']}")
    print(f"Success Rate: {stats['success_rate']:.2f}%")
    print(f"\nPerformance:")
    print(f"  Total Time: {stats['total_time']:.2f}s")
    print(f"  Requests/sec: {stats['requests_per_second']:.2f}")
    print(f"  Avg Response: {stats['avg_response_time']*1000:.2f}ms")
    print(f"  Min Response: {stats['min_response_time']*1000:.2f}ms")
    print(f"  Max Response: {stats['max_response_time']*1000:.2f}ms")
    print(f"  Median Response: {stats['median_response_time']*1000:.2f}ms")
    print(f"  P95 Response: {stats['p95_response_time']*1000:.2f}ms")
    print(f"  P99 Response: {stats['p99_response_time']*1000:.2f}ms")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()















