"""
🚀 PERFORMANCE BENCHMARKING SCRIPT
==================================

Comprehensive performance testing for the optimized blog system:
- Load testing with multiple concurrent requests
- Memory usage monitoring
- Response time analysis
- Cache hit/miss ratio testing
- Database performance testing
- Throughput measurement
"""

import asyncio
import time
import statistics
import psutil
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import uvicorn
from pathlib import Path

# Import the optimized blog system
from optimized_blog_system_v2 import (
    create_optimized_blog_system,
    Config,
    DatabaseConfig,
    CacheConfig,
    PerformanceConfig
)

# ============================================================================
# BENCHMARK CONFIGURATION
# ============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarks."""
    base_url: str = "http://localhost:8000"
    concurrent_users: int = 50
    requests_per_user: int = 100
    warmup_requests: int = 10
    test_duration: int = 60  # seconds
    cache_enabled: bool = True
    database_enabled: bool = True
    monitoring_enabled: bool = True

@dataclass
class BenchmarkResult:
    """Results from a performance benchmark."""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    min_response_time: float
    max_response_time: float
    requests_per_second: float
    memory_usage_mb: float
    cpu_usage_percent: float
    cache_hit_ratio: Optional[float] = None
    database_queries: Optional[int] = None

# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

class PerformanceMonitor:
    """Monitor system performance during benchmarks."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_samples = []
        self.cpu_samples = []
        self.response_times = []
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.memory_samples = []
        self.cpu_samples = []
        self.response_times = []
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
    
    def record_request(self, response_time: float, success: bool = True):
        """Record a request result."""
        self.response_times.append(response_time)
        self.request_count += 1
        
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    def sample_system_metrics(self):
        """Sample current system metrics."""
        try:
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent()
            
            self.memory_samples.append(memory.used / 1024 / 1024)  # MB
            self.cpu_samples.append(cpu)
        except Exception as e:
            print(f"Warning: Could not sample system metrics: {e}")
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return results."""
        self.end_time = time.time()
        
        if not self.response_times:
            return {}
        
        return {
            "total_requests": self.request_count,
            "successful_requests": self.success_count,
            "failed_requests": self.error_count,
            "success_rate": self.success_count / self.request_count if self.request_count > 0 else 0,
            "average_response_time": statistics.mean(self.response_times),
            "median_response_time": statistics.median(self.response_times),
            "p95_response_time": self._percentile(self.response_times, 95),
            "p99_response_time": self._percentile(self.response_times, 99),
            "min_response_time": min(self.response_times),
            "max_response_time": max(self.response_times),
            "requests_per_second": self.request_count / (self.end_time - self.start_time),
            "average_memory_mb": statistics.mean(self.memory_samples) if self.memory_samples else 0,
            "max_memory_mb": max(self.memory_samples) if self.memory_samples else 0,
            "average_cpu_percent": statistics.mean(self.cpu_samples) if self.cpu_samples else 0,
            "max_cpu_percent": max(self.cpu_samples) if self.cpu_samples else 0,
            "test_duration": self.end_time - self.start_time
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))

# ============================================================================
# BENCHMARK TESTS
# ============================================================================

class BlogSystemBenchmark:
    """Comprehensive benchmark tests for the blog system."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.monitor = PerformanceMonitor()
        self.session = None
    
    async def setup(self):
        """Setup benchmark environment."""
        self.session = aiohttp.ClientSession()
    
    async def cleanup(self):
        """Cleanup benchmark environment."""
        if self.session:
            await self.session.close()
    
    async def make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict[str, Any]:
        """Make a single HTTP request."""
        url = f"{self.config.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            if method == "GET":
                async with self.session.get(url) as response:
                    response_time = time.time() - start_time
                    success = response.status < 400
                    self.monitor.record_request(response_time, success)
                    return {
                        "status": response.status,
                        "response_time": response_time,
                        "success": success,
                        "data": await response.json() if response.status < 400 else None
                    }
            
            elif method == "POST":
                async with self.session.post(url, json=data) as response:
                    response_time = time.time() - start_time
                    success = response.status < 400
                    self.monitor.record_request(response_time, success)
                    return {
                        "status": response.status,
                        "response_time": response_time,
                        "success": success,
                        "data": await response.json() if response.status < 400 else None
                    }
            
            elif method == "PATCH":
                async with self.session.patch(url, json=data) as response:
                    response_time = time.time() - start_time
                    success = response.status < 400
                    self.monitor.record_request(response_time, success)
                    return {
                        "status": response.status,
                        "response_time": response_time,
                        "success": success,
                        "data": await response.json() if response.status < 400 else None
                    }
            
            elif method == "DELETE":
                async with self.session.delete(url) as response:
                    response_time = time.time() - start_time
                    success = response.status < 400
                    self.monitor.record_request(response_time, success)
                    return {
                        "status": response.status,
                        "response_time": response_time,
                        "success": success,
                        "data": None
                    }
        
        except Exception as e:
            response_time = time.time() - start_time
            self.monitor.record_request(response_time, False)
            return {
                "status": 0,
                "response_time": response_time,
                "success": False,
                "error": str(e)
            }
    
    async def warmup_test(self) -> BenchmarkResult:
        """Warmup test to initialize the system."""
        print("🔥 Running warmup test...")
        
        self.monitor.start_monitoring()
        
        # Create some test posts
        test_posts = []
        for i in range(5):
            post_data = {
                "title": f"Test Post {i}",
                "content": f"This is test content for post {i}",
                "tags": ["test", f"post-{i}"],
                "is_published": True
            }
            result = await self.make_request("POST", "/posts", post_data)
            if result["success"]:
                test_posts.append(result["data"]["id"])
        
        # Make some read requests
        for _ in range(self.config.warmup_requests):
            await self.make_request("GET", "/posts")
            if test_posts:
                await self.make_request("GET", f"/posts/{test_posts[0]}")
        
        results = self.monitor.stop_monitoring()
        return BenchmarkResult(
            test_name="Warmup Test",
            total_requests=results["total_requests"],
            successful_requests=results["successful_requests"],
            failed_requests=results["failed_requests"],
            average_response_time=results["average_response_time"],
            median_response_time=results["median_response_time"],
            p95_response_time=results["p95_response_time"],
            p99_response_time=results["p99_response_time"],
            min_response_time=results["min_response_time"],
            max_response_time=results["max_response_time"],
            requests_per_second=results["requests_per_second"],
            memory_usage_mb=results["average_memory_mb"],
            cpu_usage_percent=results["average_cpu_percent"]
        )
    
    async def read_heavy_test(self) -> BenchmarkResult:
        """Test read-heavy workload."""
        print("📖 Running read-heavy test...")
        
        self.monitor.start_monitoring()
        
        async def read_worker():
            """Worker for read operations."""
            for _ in range(self.config.requests_per_user):
                # List posts
                await self.make_request("GET", "/posts")
                
                # Get specific post (simulate random access)
                post_id = (hash(str(time.time())) % 5) + 1
                await self.make_request("GET", f"/posts/{post_id}")
                
                # Get health check
                await self.make_request("GET", "/health")
                
                # Small delay to simulate real usage
                await asyncio.sleep(0.01)
        
        # Run concurrent workers
        tasks = [read_worker() for _ in range(self.config.concurrent_users)]
        await asyncio.gather(*tasks)
        
        results = self.monitor.stop_monitoring()
        return BenchmarkResult(
            test_name="Read-Heavy Test",
            total_requests=results["total_requests"],
            successful_requests=results["successful_requests"],
            failed_requests=results["failed_requests"],
            average_response_time=results["average_response_time"],
            median_response_time=results["median_response_time"],
            p95_response_time=results["p95_response_time"],
            p99_response_time=results["p99_response_time"],
            min_response_time=results["min_response_time"],
            max_response_time=results["max_response_time"],
            requests_per_second=results["requests_per_second"],
            memory_usage_mb=results["average_memory_mb"],
            cpu_usage_percent=results["average_cpu_percent"]
        )
    
    async def write_heavy_test(self) -> BenchmarkResult:
        """Test write-heavy workload."""
        print("✍️ Running write-heavy test...")
        
        self.monitor.start_monitoring()
        
        async def write_worker():
            """Worker for write operations."""
            for i in range(self.config.requests_per_user):
                # Create post
                post_data = {
                    "title": f"Benchmark Post {i}",
                    "content": f"This is benchmark content for post {i}",
                    "tags": ["benchmark", f"post-{i}"],
                    "is_published": True
                }
                result = await self.make_request("POST", "/posts", post_data)
                
                if result["success"] and result["data"]:
                    post_id = result["data"]["id"]
                    
                    # Update post
                    update_data = {
                        "title": f"Updated Benchmark Post {i}",
                        "content": f"This is updated benchmark content for post {i}"
                    }
                    await self.make_request("PATCH", f"/posts/{post_id}", update_data)
                    
                    # Delete post
                    await self.make_request("DELETE", f"/posts/{post_id}")
                
                # Small delay
                await asyncio.sleep(0.01)
        
        # Run concurrent workers
        tasks = [write_worker() for _ in range(self.config.concurrent_users)]
        await asyncio.gather(*tasks)
        
        results = self.monitor.stop_monitoring()
        return BenchmarkResult(
            test_name="Write-Heavy Test",
            total_requests=results["total_requests"],
            successful_requests=results["successful_requests"],
            failed_requests=results["failed_requests"],
            average_response_time=results["average_response_time"],
            median_response_time=results["median_response_time"],
            p95_response_time=results["p95_response_time"],
            p99_response_time=results["p99_response_time"],
            min_response_time=results["min_response_time"],
            max_response_time=results["max_response_time"],
            requests_per_second=results["requests_per_second"],
            memory_usage_mb=results["average_memory_mb"],
            cpu_usage_percent=results["average_cpu_percent"]
        )
    
    async def mixed_workload_test(self) -> BenchmarkResult:
        """Test mixed read/write workload."""
        print("🔄 Running mixed workload test...")
        
        self.monitor.start_monitoring()
        
        async def mixed_worker():
            """Worker for mixed operations."""
            for i in range(self.config.requests_per_user):
                # 70% reads, 30% writes
                operation = hash(f"{time.time()}{i}") % 10
                
                if operation < 7:  # 70% reads
                    await self.make_request("GET", "/posts")
                    post_id = (hash(str(time.time())) % 5) + 1
                    await self.make_request("GET", f"/posts/{post_id}")
                else:  # 30% writes
                    post_data = {
                        "title": f"Mixed Post {i}",
                        "content": f"This is mixed content for post {i}",
                        "tags": ["mixed", f"post-{i}"],
                        "is_published": True
                    }
                    result = await self.make_request("POST", "/posts", post_data)
                    
                    if result["success"] and result["data"]:
                        post_id = result["data"]["id"]
                        await self.make_request("DELETE", f"/posts/{post_id}")
                
                await asyncio.sleep(0.01)
        
        # Run concurrent workers
        tasks = [mixed_worker() for _ in range(self.config.concurrent_users)]
        await asyncio.gather(*tasks)
        
        results = self.monitor.stop_monitoring()
        return BenchmarkResult(
            test_name="Mixed Workload Test",
            total_requests=results["total_requests"],
            successful_requests=results["successful_requests"],
            failed_requests=results["failed_requests"],
            average_response_time=results["average_response_time"],
            median_response_time=results["median_response_time"],
            p95_response_time=results["p95_response_time"],
            p99_response_time=results["p99_response_time"],
            min_response_time=results["min_response_time"],
            max_response_time=results["max_response_time"],
            requests_per_second=results["requests_per_second"],
            memory_usage_mb=results["average_memory_mb"],
            cpu_usage_percent=results["average_cpu_percent"]
        )
    
    async def stress_test(self) -> BenchmarkResult:
        """Stress test with high concurrency."""
        print("💪 Running stress test...")
        
        self.monitor.start_monitoring()
        
        async def stress_worker():
            """Worker for stress testing."""
            for _ in range(self.config.requests_per_user * 2):  # Double the load
                # Random operations
                operation = hash(f"{time.time()}") % 4
                
                if operation == 0:
                    await self.make_request("GET", "/posts")
                elif operation == 1:
                    await self.make_request("GET", "/health")
                elif operation == 2:
                    post_data = {
                        "title": f"Stress Post {time.time()}",
                        "content": f"Stress content {time.time()}",
                        "tags": ["stress"],
                        "is_published": True
                    }
                    await self.make_request("POST", "/posts", post_data)
                else:
                    await self.make_request("GET", "/metrics")
        
        # Run with higher concurrency
        tasks = [stress_worker() for _ in range(self.config.concurrent_users * 2)]
        await asyncio.gather(*tasks)
        
        results = self.monitor.stop_monitoring()
        return BenchmarkResult(
            test_name="Stress Test",
            total_requests=results["total_requests"],
            successful_requests=results["successful_requests"],
            failed_requests=results["failed_requests"],
            average_response_time=results["average_response_time"],
            median_response_time=results["median_response_time"],
            p95_response_time=results["p95_response_time"],
            p99_response_time=results["p99_response_time"],
            min_response_time=results["min_response_time"],
            max_response_time=results["max_response_time"],
            requests_per_second=results["requests_per_second"],
            memory_usage_mb=results["average_memory_mb"],
            cpu_usage_percent=results["average_cpu_percent"]
        )

# ============================================================================
# RESULTS REPORTING
# ============================================================================

class BenchmarkReporter:
    """Generate comprehensive benchmark reports."""
    
    @staticmethod
    def print_results(results: List[BenchmarkResult]):
        """Print benchmark results in a formatted way."""
        print("\n" + "="*80)
        print("🚀 BENCHMARK RESULTS")
        print("="*80)
        
        for result in results:
            print(f"\n📊 {result.test_name}")
            print("-" * 50)
            print(f"Total Requests:     {result.total_requests:,}")
            print(f"Successful:         {result.successful_requests:,}")
            print(f"Failed:             {result.failed_requests:,}")
            print(f"Success Rate:       {result.successful_requests/result.total_requests*100:.2f}%")
            print(f"Requests/Second:    {result.requests_per_second:.2f}")
            print(f"Avg Response Time:  {result.average_response_time*1000:.2f}ms")
            print(f"Median Response:    {result.median_response_time*1000:.2f}ms")
            print(f"P95 Response:       {result.p95_response_time*1000:.2f}ms")
            print(f"P99 Response:       {result.p99_response_time*1000:.2f}ms")
            print(f"Min Response:       {result.min_response_time*1000:.2f}ms")
            print(f"Max Response:       {result.max_response_time*1000:.2f}ms")
            print(f"Memory Usage:       {result.memory_usage_mb:.2f} MB")
            print(f"CPU Usage:          {result.cpu_usage_percent:.2f}%")
    
    @staticmethod
    def save_results(results: List[BenchmarkResult], filename: str = "benchmark_results.json"):
        """Save results to JSON file."""
        data = []
        for result in results:
            data.append({
                "test_name": result.test_name,
                "total_requests": result.total_requests,
                "successful_requests": result.successful_requests,
                "failed_requests": result.failed_requests,
                "success_rate": result.successful_requests / result.total_requests,
                "average_response_time": result.average_response_time,
                "median_response_time": result.median_response_time,
                "p95_response_time": result.p95_response_time,
                "p99_response_time": result.p99_response_time,
                "min_response_time": result.min_response_time,
                "max_response_time": result.max_response_time,
                "requests_per_second": result.requests_per_second,
                "memory_usage_mb": result.memory_usage_mb,
                "cpu_usage_percent": result.cpu_usage_percent
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Results saved to {filename}")

# ============================================================================
# MAIN BENCHMARK RUNNER
# ============================================================================

async def run_benchmarks(config: BenchmarkConfig = None) -> List[BenchmarkResult]:
    """Run all benchmark tests."""
    if config is None:
        config = BenchmarkConfig()
    
    benchmark = BlogSystemBenchmark(config)
    results = []
    
    try:
        await benchmark.setup()
        
        # Run all tests
        tests = [
            benchmark.warmup_test(),
            benchmark.read_heavy_test(),
            benchmark.write_heavy_test(),
            benchmark.mixed_workload_test(),
            benchmark.stress_test()
        ]
        
        for test in tests:
            result = await test
            results.append(result)
    
    finally:
        await benchmark.cleanup()
    
    return results

def main():
    """Main benchmark runner."""
    print("🚀 Starting Blog System Performance Benchmark")
    print("=" * 60)
    
    # Configuration
    config = BenchmarkConfig(
        concurrent_users=20,
        requests_per_user=50,
        warmup_requests=20,
        test_duration=30
    )
    
    # Run benchmarks
    results = asyncio.run(run_benchmarks(config))
    
    # Report results
    BenchmarkReporter.print_results(results)
    BenchmarkReporter.save_results(results)
    
    print("\n✅ Benchmark completed!")

if __name__ == "__main__":
    main() 
 
 