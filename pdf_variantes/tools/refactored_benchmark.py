"""
Refactored Benchmark
====================
Benchmark tool using base classes and improved structure.
"""

import time
import statistics
import concurrent.futures
from typing import Dict, Any, List
from .base import BaseAPITool, ToolResult
from .config import get_config
from .utils import format_response_time, print_success, print_error


class Benchmark(BaseAPITool):
    """Refactored benchmark tool."""
    
    def benchmark_endpoint(
        self,
        method: str,
        endpoint: str,
        iterations: int = 100,
        concurrent: int = 1
    ) -> Dict[str, Any]:
        """Benchmark an endpoint."""
        times = []
        errors = []
        success_count = 0
        
        def make_request():
            start = time.time()
            try:
                response = self.make_request(method, endpoint)
                elapsed = (time.time() - start) * 1000
                
                if 200 <= response.status_code < 300:
                    times.append(elapsed)
                    return True
                else:
                    errors.append(f"Status {response.status_code}")
                    return False
            except Exception as e:
                errors.append(str(e))
                return False
        
        start_total = time.time()
        
        if concurrent > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent) as executor:
                futures = [executor.submit(make_request) for _ in range(iterations)]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
                success_count = sum(results)
        else:
            for _ in range(iterations):
                if make_request():
                    success_count += 1
        
        total_time = (time.time() - start_total) * 1000
        
        if not times:
            return {
                "endpoint": endpoint,
                "method": method,
                "iterations": iterations,
                "success_count": 0,
                "error_count": iterations,
                "errors": errors[:10]
            }
        
        sorted_times = sorted(times)
        
        return {
            "endpoint": endpoint,
            "method": method,
            "iterations": iterations,
            "success_count": success_count,
            "error_count": iterations - success_count,
            "total_time": total_time,
            "avg_time": statistics.mean(times),
            "min_time": min(times),
            "max_time": max(times),
            "median_time": statistics.median(times),
            "p95_time": sorted_times[int(len(sorted_times) * 0.95)] if len(sorted_times) > 0 else 0,
            "p99_time": sorted_times[int(len(sorted_times) * 0.99)] if len(sorted_times) > 0 else 0,
            "success_rate": (success_count / iterations) * 100
        }
    
    def run(
        self,
        endpoint: str = "/health",
        method: str = "GET",
        iterations: int = 100,
        concurrent: int = 1,
        **kwargs
    ) -> ToolResult:
        """Run benchmark."""
        print(f"🔥 Benchmarking {method} {endpoint} ({iterations} iterations, {concurrent} concurrent)...")
        
        result = self.benchmark_endpoint(method, endpoint, iterations, concurrent)
        self.results.append(result)
        
        if result["success_count"] == 0:
            print_error("All requests failed")
            return ToolResult(
                success=False,
                message="Benchmark failed: all requests failed",
                data=result
            )
        
        print_success(f"Benchmark completed: {result['success_rate']:.2f}% success rate")
        print(f"   Average: {format_response_time(result['avg_time'])}")
        print(f"   P95: {format_response_time(result['p95_time'])}")
        
        return ToolResult(
            success=True,
            message=f"Benchmark completed: {result['success_rate']:.2f}% success",
            data=result
        )


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark (Refactored)")
    parser.add_argument("--url", help="API base URL")
    parser.add_argument("--endpoint", default="/health", help="Endpoint to benchmark")
    parser.add_argument("--method", default="GET", help="HTTP method")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--concurrent", type=int, default=1, help="Concurrent requests")
    parser.add_argument("--export", help="Export results")
    
    args = parser.parse_args()
    
    config = get_config()
    base_url = args.url or config.base_url
    
    benchmark = Benchmark(base_url=base_url)
    result = benchmark.run(
        endpoint=args.endpoint,
        method=args.method,
        iterations=args.iterations,
        concurrent=args.concurrent
    )
    
    if args.export:
        benchmark.export_results(args.export)
    
    return 0 if result.success else 1


if __name__ == "__main__":
    exit(main())



