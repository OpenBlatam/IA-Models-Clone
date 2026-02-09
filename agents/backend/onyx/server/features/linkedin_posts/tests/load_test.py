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
import statistics
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import csv
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import httpx
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.panel import Panel
from .debug_tools import APIDebugger, PerformanceProfiler, print_debug_info
from typing import Any, List, Dict, Optional
import logging
"""
Load Testing Suite for LinkedIn Posts API
=========================================

Comprehensive load testing with various scenarios and metrics collection.
"""




@dataclass
class LoadTestResult:
    """Result of a load test."""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    success_rate: float
    error_details: List[Dict[str, Any]]
    timestamp: datetime


class LoadTester:
    """
    Advanced load tester for the LinkedIn Posts API.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000/api/v2"):
        
    """__init__ function."""
self.base_url = base_url
        self.console = Console()
        self.debugger = APIDebugger()
        self.profiler = PerformanceProfiler()
        self.results: List[LoadTestResult] = []
        
        # Test data
        self.sample_post_data = {
            "content": "Load test post content",
            "post_type": "educational",
            "tone": "professional",
            "target_audience": "developers",
            "industry": "technology"
        }
        
        self.auth_headers = {
            "Authorization": "Bearer test-jwt-token",
            "Content-Type": "application/json"
        }
    
    async def run_basic_load_test(
        self,
        num_requests: int = 100,
        concurrent_users: int = 10,
        test_name: str = "Basic Load Test"
    ) -> LoadTestResult:
        """Run a basic load test."""
        self.console.print(f"\n🚀 Running {test_name}")
        self.console.print(f"   Requests: {num_requests}")
        self.console.print(f"   Concurrent Users: {concurrent_users}")
        
        # Prepare test data
        test_posts = []
        for i in range(num_requests):
            post_data = {
                **self.sample_post_data,
                "content": f"Load test post {i} - {self.sample_post_data['content']}"
            }
            test_posts.append(post_data)
        
        # Run test
        start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Running load test...", total=num_requests)
            
            # Create semaphore to limit concurrency
            semaphore = asyncio.Semaphore(concurrent_users)
            
            async async def make_request(post_data, request_id) -> Any:
                async with semaphore:
                    try:
                        async with httpx.AsyncClient(
                            base_url=self.base_url,
                            timeout=30.0
                        ) as client:
                            request_start = time.time()
                            
                            response = await client.post(
                                "/linkedin-posts/",
                                json=post_data,
                                headers=self.auth_headers
                            )
                            
                            request_time = time.time() - request_start
                            
                            progress.advance(task)
                            
                            return {
                                "request_id": request_id,
                                "success": response.status_code == 201,
                                "status_code": response.status_code,
                                "response_time": request_time,
                                "error": None if response.status_code == 201 else response.text
                            }
                    
                    except Exception as e:
                        progress.advance(task)
                        return {
                            "request_id": request_id,
                            "success": False,
                            "status_code": None,
                            "response_time": 0,
                            "error": str(e)
                        }
            
            # Execute requests
            tasks = [
                make_request(post_data, i)
                for i, post_data in enumerate(test_posts)
            ]
            
            results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        response_times = [r["response_time"] for r in successful_requests]
        
        if response_times:
            result = LoadTestResult(
                test_name=test_name,
                total_requests=num_requests,
                successful_requests=len(successful_requests),
                failed_requests=len(failed_requests),
                total_time=total_time,
                avg_response_time=statistics.mean(response_times),
                min_response_time=min(response_times),
                max_response_time=max(response_times),
                p50_response_time=statistics.quantiles(response_times, n=2)[0],
                p95_response_time=statistics.quantiles(response_times, n=20)[18],
                p99_response_time=statistics.quantiles(response_times, n=100)[98],
                requests_per_second=num_requests / total_time,
                success_rate=len(successful_requests) / num_requests,
                error_details=failed_requests,
                timestamp=datetime.utcnow()
            )
        else:
            result = LoadTestResult(
                test_name=test_name,
                total_requests=num_requests,
                successful_requests=0,
                failed_requests=num_requests,
                total_time=total_time,
                avg_response_time=0,
                min_response_time=0,
                max_response_time=0,
                p50_response_time=0,
                p95_response_time=0,
                p99_response_time=0,
                requests_per_second=0,
                success_rate=0,
                error_details=failed_requests,
                timestamp=datetime.utcnow()
            )
        
        self.results.append(result)
        return result
    
    async def run_stress_test(
        self,
        max_requests: int = 1000,
        ramp_up_time: int = 60,
        test_name: str = "Stress Test"
    ) -> LoadTestResult:
        """Run a stress test with gradual ramp-up."""
        self.console.print(f"\n🔥 Running {test_name}")
        self.console.print(f"   Max Requests: {max_requests}")
        self.console.print(f"   Ramp-up Time: {ramp_up_time}s")
        
        start_time = time.time()
        results = []
        current_requests = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Running stress test...", total=max_requests)
            
            while current_requests < max_requests:
                # Calculate current concurrency based on ramp-up
                elapsed_time = time.time() - start_time
                progress_ratio = min(elapsed_time / ramp_up_time, 1.0)
                current_concurrency = int(10 + (progress_ratio * 90))  # 10 to 100 users
                
                # Create batch of requests
                batch_size = min(current_concurrency, max_requests - current_requests)
                
                async async def make_stress_request(request_id) -> Any:
                    try:
                        async with httpx.AsyncClient(
                            base_url=self.base_url,
                            timeout=30.0
                        ) as client:
                            request_start = time.time()
                            
                            response = await client.get(
                                "/linkedin-posts/health",
                                headers=self.auth_headers
                            )
                            
                            request_time = time.time() - request_start
                            
                            return {
                                "request_id": request_id,
                                "success": response.status_code == 200,
                                "status_code": response.status_code,
                                "response_time": request_time,
                                "error": None if response.status_code == 200 else response.text
                            }
                    
                    except Exception as e:
                        return {
                            "request_id": request_id,
                            "success": False,
                            "status_code": None,
                            "response_time": 0,
                            "error": str(e)
                        }
                
                # Execute batch
                batch_tasks = [
                    make_stress_request(current_requests + i)
                    for i in range(batch_size)
                ]
                
                batch_results = await asyncio.gather(*batch_tasks)
                results.extend(batch_results)
                
                current_requests += batch_size
                progress.advance(task, batch_size)
                
                # Small delay between batches
                await asyncio.sleep(0.1)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        response_times = [r["response_time"] for r in successful_requests]
        
        if response_times:
            result = LoadTestResult(
                test_name=test_name,
                total_requests=len(results),
                successful_requests=len(successful_requests),
                failed_requests=len(failed_requests),
                total_time=total_time,
                avg_response_time=statistics.mean(response_times),
                min_response_time=min(response_times),
                max_response_time=max(response_times),
                p50_response_time=statistics.quantiles(response_times, n=2)[0],
                p95_response_time=statistics.quantiles(response_times, n=20)[18],
                p99_response_time=statistics.quantiles(response_times, n=100)[98],
                requests_per_second=len(results) / total_time,
                success_rate=len(successful_requests) / len(results),
                error_details=failed_requests,
                timestamp=datetime.utcnow()
            )
        else:
            result = LoadTestResult(
                test_name=test_name,
                total_requests=len(results),
                successful_requests=0,
                failed_requests=len(results),
                total_time=total_time,
                avg_response_time=0,
                min_response_time=0,
                max_response_time=0,
                p50_response_time=0,
                p95_response_time=0,
                p99_response_time=0,
                requests_per_second=0,
                success_rate=0,
                error_details=failed_requests,
                timestamp=datetime.utcnow()
            )
        
        self.results.append(result)
        return result
    
    async def run_batch_operations_test(
        self,
        num_batches: int = 10,
        batch_size: int = 20,
        test_name: str = "Batch Operations Test"
    ) -> LoadTestResult:
        """Test batch operations performance."""
        self.console.print(f"\n📦 Running {test_name}")
        self.console.print(f"   Batches: {num_batches}")
        self.console.print(f"   Batch Size: {batch_size}")
        
        start_time = time.time()
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Running batch operations...", total=num_batches)
            
            for batch_num in range(num_batches):
                # Create batch data
                batch_data = []
                for i in range(batch_size):
                    post_data = {
                        **self.sample_post_data,
                        "content": f"Batch {batch_num} - Post {i} - {self.sample_post_data['content']}"
                    }
                    batch_data.append(post_data)
                
                # Execute batch request
                try:
                    async with httpx.AsyncClient(
                        base_url=self.base_url,
                        timeout=60.0
                    ) as client:
                        request_start = time.time()
                        
                        response = await client.post(
                            "/linkedin-posts/batch?parallel_processing=true",
                            json=batch_data,
                            headers=self.auth_headers
                        )
                        
                        request_time = time.time() - request_start
                        
                        results.append({
                            "batch_id": batch_num,
                            "success": response.status_code == 200,
                            "status_code": response.status_code,
                            "response_time": request_time,
                            "batch_size": batch_size,
                            "error": None if response.status_code == 200 else response.text
                        })
                
                except Exception as e:
                    results.append({
                        "batch_id": batch_num,
                        "success": False,
                        "status_code": None,
                        "response_time": 0,
                        "batch_size": batch_size,
                        "error": str(e)
                    })
                
                progress.advance(task)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Analyze results
        successful_batches = [r for r in results if r["success"]]
        failed_batches = [r for r in results if not r["success"]]
        
        response_times = [r["response_time"] for r in successful_batches]
        
        if response_times:
            result = LoadTestResult(
                test_name=test_name,
                total_requests=num_batches,
                successful_requests=len(successful_batches),
                failed_requests=len(failed_batches),
                total_time=total_time,
                avg_response_time=statistics.mean(response_times),
                min_response_time=min(response_times),
                max_response_time=max(response_times),
                p50_response_time=statistics.quantiles(response_times, n=2)[0],
                p95_response_time=statistics.quantiles(response_times, n=20)[18],
                p99_response_time=statistics.quantiles(response_times, n=100)[98],
                requests_per_second=num_batches / total_time,
                success_rate=len(successful_batches) / num_batches,
                error_details=failed_batches,
                timestamp=datetime.utcnow()
            )
        else:
            result = LoadTestResult(
                test_name=test_name,
                total_requests=num_batches,
                successful_requests=0,
                failed_requests=num_batches,
                total_time=total_time,
                avg_response_time=0,
                min_response_time=0,
                max_response_time=0,
                p50_response_time=0,
                p95_response_time=0,
                p99_response_time=0,
                requests_per_second=0,
                success_rate=0,
                error_details=failed_batches,
                timestamp=datetime.utcnow()
            )
        
        self.results.append(result)
        return result
    
    async def run_cache_performance_test(
        self,
        num_requests: int = 100,
        test_name: str = "Cache Performance Test"
    ) -> Dict[str, LoadTestResult]:
        """Test cache performance impact."""
        self.console.print(f"\n💾 Running {test_name}")
        
        # Test without cache
        self.console.print("  🔍 Testing without cache...")
        no_cache_result = await self._run_cache_test(
            num_requests, use_cache=False, test_name="No Cache"
        )
        
        # Test with cache
        self.console.print("  🔍 Testing with cache...")
        with_cache_result = await self._run_cache_test(
            num_requests, use_cache=True, test_name="With Cache"
        )
        
        return {
            "no_cache": no_cache_result,
            "with_cache": with_cache_result
        }
    
    async def _run_cache_test(
        self,
        num_requests: int,
        use_cache: bool,
        test_name: str
    ) -> LoadTestResult:
        """Run a cache test."""
        # First, create a post
        async with httpx.AsyncClient(base_url=self.base_url) as client:
            create_response = await client.post(
                "/linkedin-posts/",
                json=self.sample_post_data,
                headers=self.auth_headers
            )
            
            if create_response.status_code != 201:
                raise Exception("Failed to create test post")
            
            post_id = create_response.json()["id"]
        
        # Now test reading the post multiple times
        start_time = time.time()
        results = []
        
        for i in range(num_requests):
            try:
                async with httpx.AsyncClient(base_url=self.base_url) as client:
                    request_start = time.time()
                    
                    response = await client.get(
                        f"/linkedin-posts/{post_id}?use_cache={str(use_cache).lower()}",
                        headers=self.auth_headers
                    )
                    
                    request_time = time.time() - request_start
                    
                    results.append({
                        "request_id": i,
                        "success": response.status_code == 200,
                        "status_code": response.status_code,
                        "response_time": request_time,
                        "cache_hit": response.headers.get("X-Cache") == "HIT",
                        "error": None if response.status_code == 200 else response.text
                    })
            
            except Exception as e:
                results.append({
                    "request_id": i,
                    "success": False,
                    "status_code": None,
                    "response_time": 0,
                    "cache_hit": False,
                    "error": str(e)
                })
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        response_times = [r["response_time"] for r in successful_requests]
        
        if response_times:
            result = LoadTestResult(
                test_name=f"{test_name} - {num_requests} requests",
                total_requests=num_requests,
                successful_requests=len(successful_requests),
                failed_requests=len(failed_requests),
                total_time=total_time,
                avg_response_time=statistics.mean(response_times),
                min_response_time=min(response_times),
                max_response_time=max(response_times),
                p50_response_time=statistics.quantiles(response_times, n=2)[0],
                p95_response_time=statistics.quantiles(response_times, n=20)[18],
                p99_response_time=statistics.quantiles(response_times, n=100)[98],
                requests_per_second=num_requests / total_time,
                success_rate=len(successful_requests) / num_requests,
                error_details=failed_requests,
                timestamp=datetime.utcnow()
            )
        else:
            result = LoadTestResult(
                test_name=f"{test_name} - {num_requests} requests",
                total_requests=num_requests,
                successful_requests=0,
                failed_requests=num_requests,
                total_time=total_time,
                avg_response_time=0,
                min_response_time=0,
                max_response_time=0,
                p50_response_time=0,
                p95_response_time=0,
                p99_response_time=0,
                requests_per_second=0,
                success_rate=0,
                error_details=failed_requests,
                timestamp=datetime.utcnow()
            )
        
        # Clean up
        async with httpx.AsyncClient(base_url=self.base_url) as client:
            await client.delete(f"/linkedin-posts/{post_id}", headers=self.auth_headers)
        
        return result
    
    def print_results(self, results: Optional[List[LoadTestResult]] = None):
        """Print test results in a formatted table."""
        if results is None:
            results = self.results
        
        if not results:
            self.console.print("No test results to display.")
            return
        
        table = Table(title="Load Test Results")
        table.add_column("Test Name", style="cyan")
        table.add_column("Total Requests", style="magenta")
        table.add_column("Success Rate", style="green")
        table.add_column("Avg Response Time", style="yellow")
        table.add_column("P95 Response Time", style="yellow")
        table.add_column("Requests/sec", style="blue")
        table.add_column("Total Time", style="red")
        
        for result in results:
            table.add_row(
                result.test_name,
                str(result.total_requests),
                f"{result.success_rate:.1%}",
                f"{result.avg_response_time:.3f}s",
                f"{result.p95_response_time:.3f}s",
                f"{result.requests_per_second:.1f}",
                f"{result.total_time:.1f}s"
            )
        
        self.console.print(table)
    
    def save_results(self, filename: str = None):
        """Save test results to file."""
        if filename is None:
            filename = f"load_test_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert results to dict for JSON serialization
        results_data = []
        for result in self.results:
            results_data.append({
                "test_name": result.test_name,
                "total_requests": result.total_requests,
                "successful_requests": result.successful_requests,
                "failed_requests": result.failed_requests,
                "total_time": result.total_time,
                "avg_response_time": result.avg_response_time,
                "min_response_time": result.min_response_time,
                "max_response_time": result.max_response_time,
                "p50_response_time": result.p50_response_time,
                "p95_response_time": result.p95_response_time,
                "p99_response_time": result.p99_response_time,
                "requests_per_second": result.requests_per_second,
                "success_rate": result.success_rate,
                "error_details": result.error_details,
                "timestamp": result.timestamp.isoformat()
            })
        
        with open(filename, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(results_data, f, indent=2)
        
        self.console.print(f"Results saved to {filename}")
        return filename
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive load test report."""
        if not self.results:
            return {"error": "No test results available"}
        
        # Calculate overall statistics
        total_requests = sum(r.total_requests for r in self.results)
        total_successful = sum(r.successful_requests for r in self.results)
        total_failed = sum(r.failed_requests for r in self.results)
        overall_success_rate = total_successful / total_requests if total_requests > 0 else 0
        
        # Performance statistics
        all_response_times = []
        for result in self.results:
            if result.successful_requests > 0:
                # Estimate response times based on average
                estimated_times = [result.avg_response_time] * result.successful_requests
                all_response_times.extend(estimated_times)
        
        if all_response_times:
            performance_stats = {
                "avg_response_time": statistics.mean(all_response_times),
                "min_response_time": min(all_response_times),
                "max_response_time": max(all_response_times),
                "p50_response_time": statistics.quantiles(all_response_times, n=2)[0],
                "p95_response_time": statistics.quantiles(all_response_times, n=20)[18],
                "p99_response_time": statistics.quantiles(all_response_times, n=100)[98]
            }
        else:
            performance_stats = {
                "avg_response_time": 0,
                "min_response_time": 0,
                "max_response_time": 0,
                "p50_response_time": 0,
                "p95_response_time": 0,
                "p99_response_time": 0
            }
        
        report = {
            "summary": {
                "total_tests": len(self.results),
                "total_requests": total_requests,
                "total_successful": total_successful,
                "total_failed": total_failed,
                "overall_success_rate": overall_success_rate,
                "timestamp": datetime.utcnow().isoformat()
            },
            "performance": performance_stats,
            "test_results": [
                {
                    "test_name": r.test_name,
                    "success_rate": r.success_rate,
                    "requests_per_second": r.requests_per_second,
                    "avg_response_time": r.avg_response_time
                }
                for r in self.results
            ],
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        for result in self.results:
            if result.success_rate < 0.95:
                recommendations.append(
                    f"Improve reliability for {result.test_name}: "
                    f"Success rate is {result.success_rate:.1%}"
                )
            
            if result.avg_response_time > 1.0:
                recommendations.append(
                    f"Optimize performance for {result.test_name}: "
                    f"Average response time is {result.avg_response_time:.3f}s"
                )
            
            if result.p95_response_time > 2.0:
                recommendations.append(
                    f"Address latency spikes in {result.test_name}: "
                    f"P95 response time is {result.p95_response_time:.3f}s"
                )
        
        if not recommendations:
            recommendations.append("All tests passed performance and reliability thresholds!")
        
        return recommendations


async def run_comprehensive_load_test():
    """Run a comprehensive load test suite."""
    console = Console()
    console.print("🚀 Starting Comprehensive Load Test Suite")
    console.print("=" * 60)
    
    # Initialize load tester
    load_tester = LoadTester()
    
    try:
        # 1. Basic load test
        basic_result = await load_tester.run_basic_load_test(
            num_requests=100,
            concurrent_users=10
        )
        
        # 2. Stress test
        stress_result = await load_tester.run_stress_test(
            max_requests=500,
            ramp_up_time=30
        )
        
        # 3. Batch operations test
        batch_result = await load_tester.run_batch_operations_test(
            num_batches=5,
            batch_size=10
        )
        
        # 4. Cache performance test
        cache_results = await load_tester.run_cache_performance_test(
            num_requests=50
        )
        
        # Print results
        console.print("\n📊 Test Results Summary")
        console.print("=" * 60)
        load_tester.print_results()
        
        # Generate and save report
        report = load_tester.generate_report()
        report_filename = load_tester.save_results()
        
        console.print("\n📋 Recommendations")
        console.print("=" * 60)
        for recommendation in report["recommendations"]:
            console.print(f"• {recommendation}")
        
        console.print(f"\n✅ Load test suite completed!")
        console.print(f"📄 Detailed report saved to: {report_filename}")
        
        return report
        
    except Exception as e:
        console.print(f"❌ Load test failed: {e}")
        raise


match __name__:
    case "__main__":
    asyncio.run(run_comprehensive_load_test()) 