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
import aiohttp
import statistics
from typing import List, Dict, Any
import json
from dataclasses import dataclass
import argparse
from typing import Any, List, Dict, Optional
import logging
"""
Instagram Captions API v14.0 - Performance Tests
Comprehensive load and stress testing
"""


@dataclass
class TestResult:
    """Test result data"""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    requests_per_second: float
    success_rate: float

class PerformanceTester:
    """Performance testing framework"""
    
    def __init__(self, base_url: str = "http://localhost:8140", api_key: str = "optimized-v14-key"):
        
    """__init__ function."""
self.base_url = base_url
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.results: List[TestResult] = []
    
    async async def single_request_test(self, num_requests: int = 100) -> TestResult:
        """Test single request performance"""
        print(f"🧪 Running single request test with {num_requests} requests...")
        
        response_times = []
        successful_requests = 0
        failed_requests = 0
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            for i in range(num_requests):
                request_data = {
                    "content_description": f"Beautiful sunset over the ocean {i}",
                    "style": "inspirational",
                    "hashtag_count": 15,
                    "optimization_level": "ultra_fast"
                }
                
                request_start = time.time()
                try:
                    async with session.post(
                        f"{self.base_url}/api/v14/generate",
                        headers=self.headers,
                        json=request_data
                    ) as response:
                        if response.status == 200:
                            successful_requests += 1
                        else:
                            failed_requests += 1
                except Exception:
                    failed_requests += 1
                
                response_time = time.time() - request_start
                response_times.append(response_time)
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"   Processed {i + 1}/{num_requests} requests...")
        
        total_time = time.time() - start_time
        
        return TestResult(
            test_name="Single Request Test",
            total_requests=num_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_time=total_time,
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            p95_response_time=statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else 0,
            requests_per_second=successful_requests / total_time if total_time > 0 else 0,
            success_rate=successful_requests / num_requests * 100
        )
    
    async async def batch_request_test(self, num_batches: int = 10, batch_size: int = 10) -> TestResult:
        """Test batch request performance"""
        print(f"🧪 Running batch request test with {num_batches} batches of {batch_size} requests...")
        
        response_times = []
        successful_requests = 0
        failed_requests = 0
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            for batch_num in range(num_batches):
                batch_requests = []
                for i in range(batch_size):
                    batch_requests.append({
                        "content_description": f"Amazing landscape photography {batch_num}-{i}",
                        "style": "professional",
                        "hashtag_count": 20,
                        "optimization_level": "ultra_fast"
                    })
                
                request_start = time.time()
                try:
                    async with session.post(
                        f"{self.base_url}/api/v14/batch",
                        headers=self.headers,
                        json=batch_requests
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            successful_requests += result.get("successful_requests", 0)
                            failed_requests += result.get("total_requests", 0) - result.get("successful_requests", 0)
                        else:
                            failed_requests += batch_size
                except Exception:
                    failed_requests += batch_size
                
                response_time = time.time() - request_start
                response_times.append(response_time)
                
                print(f"   Processed batch {batch_num + 1}/{num_batches}...")
        
        total_time = time.time() - start_time
        total_requests = num_batches * batch_size
        
        return TestResult(
            test_name="Batch Request Test",
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_time=total_time,
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            p95_response_time=statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else 0,
            requests_per_second=successful_requests / total_time if total_time > 0 else 0,
            success_rate=successful_requests / total_requests * 100
        )
    
    async def concurrent_load_test(self, num_concurrent: int = 50, duration: int = 60) -> TestResult:
        """Test concurrent load performance"""
        print(f"🧪 Running concurrent load test with {num_concurrent} concurrent users for {duration} seconds...")
        
        response_times = []
        successful_requests = 0
        failed_requests = 0
        start_time = time.time()
        
        async def worker(worker_id: int):
            
    """worker function."""
nonlocal successful_requests, failed_requests
            async with aiohttp.ClientSession() as session:
                while time.time() - start_time < duration:
                    request_data = {
                        "content_description": f"Concurrent test content from worker {worker_id}",
                        "style": "casual",
                        "hashtag_count": 10,
                        "optimization_level": "ultra_fast"
                    }
                    
                    request_start = time.time()
                    try:
                        async with session.post(
                            f"{self.base_url}/api/v14/generate",
                            headers=self.headers,
                            json=request_data
                        ) as response:
                            if response.status == 200:
                                successful_requests += 1
                            else:
                                failed_requests += 1
                    except Exception:
                        failed_requests += 1
                    
                    response_time = time.time() - request_start
                    response_times.append(response_time)
                    
                    # Small delay to prevent overwhelming
                    await asyncio.sleep(0.1)
        
        # Start concurrent workers
        workers = [worker(i) for i in range(num_concurrent)]
        await asyncio.gather(*workers)
        
        total_time = time.time() - start_time
        total_requests = successful_requests + failed_requests
        
        return TestResult(
            test_name="Concurrent Load Test",
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_time=total_time,
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            p95_response_time=statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else 0,
            requests_per_second=successful_requests / total_time if total_time > 0 else 0,
            success_rate=successful_requests / total_requests * 100 if total_requests > 0 else 0
        )
    
    async def stress_test(self, max_concurrent: int = 200, ramp_up_time: int = 30) -> TestResult:
        """Stress test with gradual ramp-up"""
        print(f"🧪 Running stress test with max {max_concurrent} concurrent users...")
        
        response_times = []
        successful_requests = 0
        failed_requests = 0
        start_time = time.time()
        
        async def stress_worker(worker_id: int):
            
    """stress_worker function."""
nonlocal successful_requests, failed_requests
            async with aiohttp.ClientSession() as session:
                while time.time() - start_time < ramp_up_time * 2:  # Run for 2x ramp-up time
                    request_data = {
                        "content_description": f"Stress test content {worker_id}",
                        "style": "professional",
                        "hashtag_count": 15,
                        "optimization_level": "ultra_fast"
                    }
                    
                    request_start = time.time()
                    try:
                        async with session.post(
                            f"{self.base_url}/api/v14/generate",
                            headers=self.headers,
                            json=request_data
                        ) as response:
                            if response.status == 200:
                                successful_requests += 1
                            else:
                                failed_requests += 1
                    except Exception:
                        failed_requests += 1
                    
                    response_time = time.time() - request_start
                    response_times.append(response_time)
                    
                    await asyncio.sleep(0.05)  # Faster requests for stress test
        
        # Gradually increase load
        current_concurrent = 1
        while current_concurrent <= max_concurrent and time.time() - start_time < ramp_up_time:
            workers = [stress_worker(i) for i in range(current_concurrent)]
            await asyncio.gather(*workers)
            current_concurrent = min(current_concurrent * 2, max_concurrent)
        
        # Continue with max load
        workers = [stress_worker(i) for i in range(max_concurrent)]
        await asyncio.gather(*workers)
        
        total_time = time.time() - start_time
        total_requests = successful_requests + failed_requests
        
        return TestResult(
            test_name="Stress Test",
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_time=total_time,
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            p95_response_time=statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else 0,
            requests_per_second=successful_requests / total_time if total_time > 0 else 0,
            success_rate=successful_requests / total_requests * 100 if total_requests > 0 else 0
        )
    
    async def cache_performance_test(self, num_requests: int = 100) -> TestResult:
        """Test cache performance with repeated requests"""
        print(f"🧪 Running cache performance test with {num_requests} requests...")
        
        response_times = []
        successful_requests = 0
        failed_requests = 0
        
        # Generate unique content for first half, repeat for second half
        content_list = [f"Cache test content {i}" for i in range(num_requests // 2)]
        content_list.extend(content_list)  # Repeat content for cache hits
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            for i, content in enumerate(content_list):
                request_data = {
                    "content_description": content,
                    "style": "casual",
                    "hashtag_count": 10,
                    "optimization_level": "ultra_fast"
                }
                
                request_start = time.time()
                try:
                    async with session.post(
                        f"{self.base_url}/api/v14/generate",
                        headers=self.headers,
                        json=request_data
                    ) as response:
                        if response.status == 200:
                            successful_requests += 1
                        else:
                            failed_requests += 1
                except Exception:
                    failed_requests += 1
                
                response_time = time.time() - request_start
                response_times.append(response_time)
        
        total_time = time.time() - start_time
        
        return TestResult(
            test_name="Cache Performance Test",
            total_requests=num_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_time=total_time,
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            p95_response_time=statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else 0,
            requests_per_second=successful_requests / total_time if total_time > 0 else 0,
            success_rate=successful_requests / num_requests * 100
        )
    
    def print_results(self) -> Any:
        """Print test results"""
        print("\n" + "="*80)
        print("📊 PERFORMANCE TEST RESULTS")
        print("="*80)
        
        for result in self.results:
            print(f"\n🧪 {result.test_name}")
            print("-" * 60)
            print(f"   Total Requests: {result.total_requests}")
            print(f"   Successful: {result.successful_requests}")
            print(f"   Failed: {result.failed_requests}")
            print(f"   Success Rate: {result.success_rate:.2f}%")
            print(f"   Total Time: {result.total_time:.2f}s")
            print(f"   Requests/Second: {result.requests_per_second:.2f}")
            print(f"   Avg Response Time: {result.avg_response_time*1000:.2f}ms")
            print(f"   Min Response Time: {result.min_response_time*1000:.2f}ms")
            print(f"   Max Response Time: {result.max_response_time*1000:.2f}ms")
            print(f"   P95 Response Time: {result.p95_response_time*1000:.2f}ms")
    
    def save_results(self, filename: str = "performance_results.json"):
        """Save results to JSON file"""
        results_dict = []
        for result in self.results:
            results_dict.append({
                "test_name": result.test_name,
                "total_requests": result.total_requests,
                "successful_requests": result.successful_requests,
                "failed_requests": result.failed_requests,
                "total_time": result.total_time,
                "avg_response_time": result.avg_response_time,
                "min_response_time": result.min_response_time,
                "max_response_time": result.max_response_time,
                "p95_response_time": result.p95_response_time,
                "requests_per_second": result.requests_per_second,
                "success_rate": result.success_rate
            })
        
        with open(filename, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(results_dict, f, indent=2)
        
        print(f"\n💾 Results saved to {filename}")

async def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Performance testing for Instagram Captions API v14.0")
    parser.add_argument("--base-url", default="http://localhost:8140", help="API base URL")
    parser.add_argument("--api-key", default="optimized-v14-key", help="API key")
    parser.add_argument("--load-test", action="store_true", help="Run load test")
    parser.add_argument("--stress-test", action="store_true", help="Run stress test")
    parser.add_argument("--duration", type=int, default=300, help="Test duration in seconds")
    parser.add_argument("--users", type=int, default=1000, help="Number of concurrent users")
    parser.add_argument("--max-users", type=int, default=5000, help="Maximum concurrent users for stress test")
    
    args = parser.parse_args()
    
    tester = PerformanceTester(args.base_url, args.api_key)
    
    print("🚀 Instagram Captions API v14.0 - Performance Testing")
    print("="*60)
    
    # Run tests based on arguments
    if args.load_test:
        result = await tester.concurrent_load_test(args.users, args.duration)
        tester.results.append(result)
    
    if args.stress_test:
        result = await tester.stress_test(args.max_users, args.duration // 2)
        tester.results.append(result)
    
    # Always run basic tests
    if not args.load_test and not args.stress_test:
        tester.results.append(await tester.single_request_test(100))
        tester.results.append(await tester.batch_request_test(10, 10))
        tester.results.append(await tester.cache_performance_test(100))
    
    # Print and save results
    tester.print_results()
    tester.save_results()

match __name__:
    case "__main__":
    asyncio.run(main()) 