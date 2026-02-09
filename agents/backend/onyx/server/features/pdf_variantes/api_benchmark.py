#!/usr/bin/env python3
"""
API Benchmark
=============
Performance benchmarking tool for API endpoints.

⚠️ DEPRECATED: This file is deprecated. Use `tools.refactored_benchmark.Benchmark` instead.

For new code, use:
    from tools.refactored_benchmark import Benchmark
    # or
    from tools.manager import ToolManager
    manager = ToolManager()
    result = manager.run_tool("benchmark")
"""
import warnings

warnings.warn(
    "api_benchmark.py is deprecated. Use 'tools.refactored_benchmark.Benchmark' instead.",
    DeprecationWarning,
    stacklevel=2
)

import time
import statistics
import requests
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import concurrent.futures


@dataclass
class BenchmarkResult:
    """Benchmark result."""
    endpoint: str
    method: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    median_time: float
    p95_time: float
    p99_time: float
    success_rate: float
    errors: List[str]
    timestamp: str


class APIBenchmark:
    """API benchmarking tool."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json"
        })
        self.results: List[BenchmarkResult] = []
    
    def benchmark_endpoint(
        self,
        method: str,
        endpoint: str,
        iterations: int = 100,
        concurrent: int = 1,
        **kwargs
    ) -> BenchmarkResult:
        """Benchmark an endpoint."""
        print(f"🔥 Benchmarking {method} {endpoint} ({iterations} iterations, {concurrent} concurrent)...")
        
        times = []
        errors = []
        success_count = 0
        
        def make_request():
            start = time.time()
            try:
                url = f"{self.base_url}{endpoint}"
                
                if method.upper() == "GET":
                    response = self.session.get(url, timeout=10, **kwargs)
                elif method.upper() == "POST":
                    response = self.session.post(url, timeout=10, **kwargs)
                elif method.upper() == "PUT":
                    response = self.session.put(url, timeout=10, **kwargs)
                elif method.upper() == "DELETE":
                    response = self.session.delete(url, timeout=10, **kwargs)
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                elapsed = (time.time() - start) * 1000  # ms
                
                if 200 <= response.status_code < 300:
                    times.append(elapsed)
                    success_count += 1
                else:
                    errors.append(f"Status {response.status_code}")
            
            except Exception as e:
                errors.append(str(e))
        
        start_total = time.time()
        
        if concurrent > 1:
            # Concurrent execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent) as executor:
                futures = [executor.submit(make_request) for _ in range(iterations)]
                concurrent.futures.wait(futures)
        else:
            # Sequential execution
            for _ in range(iterations):
                make_request()
        
        total_time = (time.time() - start_total) * 1000
        
        if not times:
            return BenchmarkResult(
                endpoint=endpoint,
                method=method.upper(),
                iterations=iterations,
                total_time=total_time,
                avg_time=0,
                min_time=0,
                max_time=0,
                median_time=0,
                p95_time=0,
                p99_time=0,
                success_rate=0,
                errors=errors,
                timestamp=datetime.now().isoformat()
            )
        
        sorted_times = sorted(times)
        success_rate = (success_count / iterations) * 100
        
        result = BenchmarkResult(
            endpoint=endpoint,
            method=method.upper(),
            iterations=iterations,
            total_time=total_time,
            avg_time=statistics.mean(times),
            min_time=min(times),
            max_time=max(times),
            median_time=statistics.median(times),
            p95_time=sorted_times[int(len(sorted_times) * 0.95)] if len(sorted_times) > 0 else 0,
            p99_time=sorted_times[int(len(sorted_times) * 0.99)] if len(sorted_times) > 0 else 0,
            success_rate=success_rate,
            errors=errors[:10],  # First 10 errors
            timestamp=datetime.now().isoformat()
        )
        
        self.results.append(result)
        return result
    
    def print_result(self, result: BenchmarkResult):
        """Print benchmark result."""
        print("\n" + "=" * 70)
        print(f"📊 Benchmark Results: {result.method} {result.endpoint}")
        print("=" * 70)
        print(f"Iterations: {result.iterations}")
        print(f"Total Time: {result.total_time:.2f}ms")
        print(f"Success Rate: {result.success_rate:.2f}%")
        print()
        print("Response Time Statistics:")
        print(f"  Average: {result.avg_time:.2f}ms")
        print(f"  Median:  {result.median_time:.2f}ms")
        print(f"  Min:     {result.min_time:.2f}ms")
        print(f"  Max:     {result.max_time:.2f}ms")
        print(f"  P95:     {result.p95_time:.2f}ms")
        print(f"  P99:     {result.p99_time:.2f}ms")
        
        if result.errors:
            print(f"\nErrors ({len(result.errors)}):")
            for error in result.errors[:5]:
                print(f"  - {error}")
        
        print("=" * 70)
    
    def compare_results(self, result1: BenchmarkResult, result2: BenchmarkResult):
        """Compare two benchmark results."""
        print("\n" + "=" * 70)
        print("📊 Benchmark Comparison")
        print("=" * 70)
        
        print(f"\n{result1.method} {result1.endpoint} vs {result2.method} {result2.endpoint}")
        print()
        
        # Compare average time
        avg_diff = result2.avg_time - result1.avg_time
        avg_pct = (avg_diff / result1.avg_time * 100) if result1.avg_time > 0 else 0
        print(f"Average Time:")
        print(f"  Result 1: {result1.avg_time:.2f}ms")
        print(f"  Result 2: {result2.avg_time:.2f}ms")
        print(f"  Difference: {avg_diff:+.2f}ms ({avg_pct:+.2f}%)")
        
        # Compare P95
        p95_diff = result2.p95_time - result1.p95_time
        p95_pct = (p95_diff / result1.p95_time * 100) if result1.p95_time > 0 else 0
        print(f"\nP95 Time:")
        print(f"  Result 1: {result1.p95_time:.2f}ms")
        print(f"  Result 2: {result2.p95_time:.2f}ms")
        print(f"  Difference: {p95_diff:+.2f}ms ({p95_pct:+.2f}%)")
        
        # Compare success rate
        success_diff = result2.success_rate - result1.success_rate
        print(f"\nSuccess Rate:")
        print(f"  Result 1: {result1.success_rate:.2f}%")
        print(f"  Result 2: {result2.success_rate:.2f}%")
        print(f"  Difference: {success_diff:+.2f}%")
        
        print("=" * 70)
    
    def export_results(self, file_path: Path):
        """Export benchmark results."""
        data = {
            "results": [asdict(r) for r in self.results],
            "exported_at": datetime.now().isoformat()
        }
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Benchmark results exported to {file_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="API Benchmark")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--endpoint", default="/health", help="Endpoint to benchmark")
    parser.add_argument("--method", default="GET", help="HTTP method")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--concurrent", type=int, default=1, help="Concurrent requests")
    parser.add_argument("--export", help="Export results to file")
    
    args = parser.parse_args()
    
    benchmark = APIBenchmark(base_url=args.url)
    
    result = benchmark.benchmark_endpoint(
        method=args.method,
        endpoint=args.endpoint,
        iterations=args.iterations,
        concurrent=args.concurrent
    )
    
    benchmark.print_result(result)
    
    if args.export:
        benchmark.export_results(Path(args.export))


if __name__ == "__main__":
    main()



