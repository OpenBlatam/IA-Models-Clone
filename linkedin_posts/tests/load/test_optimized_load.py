from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import pytest
import asyncio
import time
import statistics
import psutil
import threading
from typing import Dict, Any, List, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import AsyncMock, patch
from ..conftest_optimized import (
from typing import Any, List, Dict, Optional
import logging
"""
Optimized Load Tests
===================

Clean, fast, and efficient load testing with minimal dependencies.
"""


# Import our optimized fixtures
    test_data_generator,
    performance_monitor,
    test_utils,
    async_utils
)


class OptimizedLoadTester:
    """Optimized load testing utility."""
    
    def __init__(self) -> Any:
        self.process = psutil.Process()
        self.results = []
        self.errors = []
    
    async def run_single_load_test(
        self,
        operation_func: Callable,
        duration: float = 30.0,
        target_rps: float = 100.0,
        max_concurrent: int = 50
    ) -> Dict[str, Any]:
        """Run a single load test with optimized performance."""
        start_time = time.time()
        end_time = start_time + duration
        
        # Calculate request interval
        interval = 1.0 / target_rps
        
        # Track metrics
        request_times = []
        response_times = []
        error_count = 0
        success_count = 0
        
        # Semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def make_request():
            
    """make_request function."""
nonlocal error_count, success_count
            
            async with semaphore:
                request_start = time.time()
                
                try:
                    result = await operation_func()
                    request_end = time.time()
                    
                    request_times.append(request_start)
                    response_times.append(request_end - request_start)
                    success_count += 1
                    
                    return result
                except Exception as e:
                    error_count += 1
                    self.errors.append(str(e))
                    return None
        
        # Create tasks
        tasks = []
        current_time = time.time()
        
        while current_time < end_time:
            task = asyncio.create_task(make_request())
            tasks.append(task)
            
            # Wait for next request interval
            await asyncio.sleep(interval)
            current_time = time.time()
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate metrics
        total_requests = len(request_times)
        total_time = time.time() - start_time
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p50_response_time = statistics.quantiles(response_times, n=2)[0] if len(response_times) > 1 else response_times[0]
            p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 19 else response_times[-1]
            p99_response_time = statistics.quantiles(response_times, n=100)[98] if len(response_times) > 99 else response_times[-1]
        else:
            avg_response_time = p50_response_time = p95_response_time = p99_response_time = 0
        
        # Memory usage
        memory_usage_mb = self.process.memory_info().rss / 1024 / 1024
        cpu_usage = self.process.cpu_percent()
        
        return {
            "duration": total_time,
            "total_requests": total_requests,
            "successful_requests": success_count,
            "failed_requests": error_count,
            "success_rate": success_count / total_requests if total_requests > 0 else 0,
            "requests_per_second": total_requests / total_time if total_time > 0 else 0,
            "avg_response_time": avg_response_time,
            "p50_response_time": p50_response_time,
            "p95_response_time": p95_response_time,
            "p99_response_time": p99_response_time,
            "memory_usage_mb": memory_usage_mb,
            "cpu_usage": cpu_usage,
            "error_rate": error_count / total_requests if total_requests > 0 else 0
        }
    
    def run_threaded_load_test(
        self,
        operation_func: Callable,
        num_threads: int = 10,
        requests_per_thread: int = 100,
        max_workers: int = 50
    ) -> Dict[str, Any]:
        """Run threaded load test for synchronous operations."""
        start_time = time.time()
        
        # Track metrics
        response_times = []
        error_count = 0
        success_count = 0
        
        def worker():
            
    """worker function."""
nonlocal error_count, success_count
            
            for _ in range(requests_per_thread):
                request_start = time.time()
                
                try:
                    result = operation_func()
                    request_end = time.time()
                    
                    response_times.append(request_end - request_start)
                    success_count += 1
                    
                except Exception as e:
                    error_count += 1
                    self.errors.append(str(e))
        
        # Run threads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker) for _ in range(num_threads)]
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    error_count += 1
                    self.errors.append(str(e))
        
        # Calculate metrics
        total_time = time.time() - start_time
        total_requests = num_threads * requests_per_thread
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p50_response_time = statistics.quantiles(response_times, n=2)[0] if len(response_times) > 1 else response_times[0]
            p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 19 else response_times[-1]
        else:
            avg_response_time = p50_response_time = p95_response_time = 0
        
        # Memory usage
        memory_usage_mb = self.process.memory_info().rss / 1024 / 1024
        cpu_usage = self.process.cpu_percent()
        
        return {
            "duration": total_time,
            "total_requests": total_requests,
            "successful_requests": success_count,
            "failed_requests": error_count,
            "success_rate": success_count / total_requests if total_requests > 0 else 0,
            "requests_per_second": total_requests / total_time if total_time > 0 else 0,
            "avg_response_time": avg_response_time,
            "p50_response_time": p50_response_time,
            "p95_response_time": p95_response_time,
            "memory_usage_mb": memory_usage_mb,
            "cpu_usage": cpu_usage,
            "error_rate": error_count / total_requests if total_requests > 0 else 0
        }


class TestOptimizedLoadTesting:
    """Optimized load testing tests."""
    
    @pytest.fixture
    def load_tester(self) -> Any:
        """Load tester fixture."""
        return OptimizedLoadTester()
    
    @pytest.mark.asyncio
    async def test_low_load_performance(self, load_tester) -> Any:
        """Test low load performance (10 RPS)."""
        async def mock_operation():
            
    """mock_operation function."""
await asyncio.sleep(0.01)  # Simulate 10ms operation
            return {"status": "success"}
        
        # Run low load test
        results = await load_tester.run_single_load_test(
            mock_operation,
            duration=10.0,  # 10 seconds
            target_rps=10.0,  # 10 requests per second
            max_concurrent=5
        )
        
        # Verify results
        assert results["total_requests"] > 0
        assert results["success_rate"] > 0.9  # 90% success rate
        assert results["requests_per_second"] > 5  # At least 5 RPS
        assert results["avg_response_time"] > 0.01  # At least 10ms
        assert results["memory_usage_mb"] > 0
        assert results["cpu_usage"] >= 0
    
    @pytest.mark.asyncio
    async def test_medium_load_performance(self, load_tester) -> Any:
        """Test medium load performance (50 RPS)."""
        async def mock_operation():
            
    """mock_operation function."""
await asyncio.sleep(0.005)  # Simulate 5ms operation
            return {"status": "success"}
        
        # Run medium load test
        results = await load_tester.run_single_load_test(
            mock_operation,
            duration=15.0,  # 15 seconds
            target_rps=50.0,  # 50 requests per second
            max_concurrent=20
        )
        
        # Verify results
        assert results["total_requests"] > 0
        assert results["success_rate"] > 0.8  # 80% success rate
        assert results["requests_per_second"] > 20  # At least 20 RPS
        assert results["avg_response_time"] > 0.005  # At least 5ms
        assert results["p95_response_time"] > results["avg_response_time"]
    
    @pytest.mark.asyncio
    async def test_high_load_performance(self, load_tester) -> Any:
        """Test high load performance (100 RPS)."""
        async def mock_operation():
            
    """mock_operation function."""
await asyncio.sleep(0.002)  # Simulate 2ms operation
            return {"status": "success"}
        
        # Run high load test
        results = await load_tester.run_single_load_test(
            mock_operation,
            duration=20.0,  # 20 seconds
            target_rps=100.0,  # 100 requests per second
            max_concurrent=50
        )
        
        # Verify results
        assert results["total_requests"] > 0
        assert results["success_rate"] > 0.7  # 70% success rate
        assert results["requests_per_second"] > 50  # At least 50 RPS
        assert results["p99_response_time"] > results["p95_response_time"]
    
    def test_threaded_load_performance(self, load_tester) -> Any:
        """Test threaded load performance."""
        def mock_sync_operation():
            
    """mock_sync_operation function."""
time.sleep(0.001)  # Simulate 1ms operation
            return {"status": "success"}
        
        # Run threaded load test
        results = load_tester.run_threaded_load_test(
            mock_sync_operation,
            num_threads=5,
            requests_per_thread=20,
            max_workers=10
        )
        
        # Verify results
        assert results["total_requests"] == 100  # 5 threads * 20 requests
        assert results["success_rate"] > 0.9  # 90% success rate
        assert results["requests_per_second"] > 10  # At least 10 RPS
        assert results["avg_response_time"] > 0.001  # At least 1ms


class TestOptimizedStressTesting:
    """Optimized stress testing tests."""
    
    @pytest.fixture
    def stress_tester(self) -> Any:
        """Stress tester fixture."""
        return OptimizedLoadTester()
    
    @pytest.mark.asyncio
    async def test_stress_test_with_errors(self, stress_tester) -> Any:
        """Test stress test with simulated errors."""
        error_rate = 0.1  # 10% error rate
        
        async def mock_operation_with_errors():
            
    """mock_operation_with_errors function."""
await asyncio.sleep(0.01)
            
            # Simulate random errors
            if asyncio.get_event_loop().time() % 10 < error_rate * 10:
                raise Exception("Simulated error")
            
            return {"status": "success"}
        
        # Run stress test
        results = await stress_tester.run_single_load_test(
            mock_operation_with_errors,
            duration=10.0,
            target_rps=30.0,
            max_concurrent=15
        )
        
        # Verify results
        assert results["total_requests"] > 0
        assert results["error_rate"] > 0  # Should have some errors
        assert results["success_rate"] < 1.0  # Should not be 100% success
        assert results["failed_requests"] > 0
    
    @pytest.mark.asyncio
    async def test_stress_test_memory_usage(self, stress_tester) -> Any:
        """Test stress test memory usage."""
        # Track memory usage
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        async def memory_intensive_operation():
            
    """memory_intensive_operation function."""
# Create some data to consume memory
            data = [i for i in range(1000)]
            await asyncio.sleep(0.01)
            return {"data": data}
        
        # Run stress test
        results = await stress_tester.run_single_load_test(
            memory_intensive_operation,
            duration=15.0,
            target_rps=20.0,
            max_concurrent=10
        )
        
        # Verify memory usage
        final_memory = results["memory_usage_mb"]
        memory_increase = final_memory - initial_memory
        
        assert memory_increase > 0  # Should use some memory
        assert memory_increase < 100  # Should not use excessive memory (less than 100MB)
    
    @pytest.mark.asyncio
    async def test_stress_test_cpu_usage(self, stress_tester) -> Any:
        """Test stress test CPU usage."""
        async def cpu_intensive_operation():
            
    """cpu_intensive_operation function."""
# Simulate CPU work
            start_time = time.time()
            while time.time() - start_time < 0.01:
                _ = sum(i for i in range(1000))
            return {"status": "success"}
        
        # Run stress test
        results = await stress_tester.run_single_load_test(
            cpu_intensive_operation,
            duration=10.0,
            target_rps=15.0,
            max_concurrent=8
        )
        
        # Verify CPU usage
        assert results["cpu_usage"] > 0  # Should use some CPU
        assert results["cpu_usage"] < 100  # Should not use 100% CPU


class TestOptimizedEnduranceTesting:
    """Optimized endurance testing tests."""
    
    @pytest.fixture
    def endurance_tester(self) -> Any:
        """Endurance tester fixture."""
        return OptimizedLoadTester()
    
    @pytest.mark.asyncio
    async def test_endurance_test_long_duration(self, endurance_tester) -> Any:
        """Test endurance with long duration."""
        async def stable_operation():
            
    """stable_operation function."""
await asyncio.sleep(0.02)  # 20ms operation
            return {"status": "success", "timestamp": time.time()}
        
        # Run endurance test
        results = await endurance_tester.run_single_load_test(
            stable_operation,
            duration=30.0,  # 30 seconds
            target_rps=25.0,  # 25 requests per second
            max_concurrent=20
        )
        
        # Verify endurance results
        assert results["duration"] >= 30.0
        assert results["total_requests"] > 500  # Should handle many requests
        assert results["success_rate"] > 0.95  # High success rate over time
        assert results["requests_per_second"] > 15  # Consistent throughput
    
    @pytest.mark.asyncio
    async def test_endurance_test_memory_stability(self, endurance_tester) -> Any:
        """Test memory stability over time."""
        memory_samples = []
        
        async def memory_monitored_operation():
            
    """memory_monitored_operation function."""
# Record memory usage
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            memory_samples.append(memory_mb)
            
            await asyncio.sleep(0.01)
            return {"memory": memory_mb}
        
        # Run endurance test
        results = await endurance_tester.run_single_load_test(
            memory_monitored_operation,
            duration=20.0,
            target_rps=30.0,
            max_concurrent=15
        )
        
        # Verify memory stability
        if len(memory_samples) > 10:
            memory_variance = statistics.variance(memory_samples)
            assert memory_variance < 100  # Memory should be relatively stable
    
    @pytest.mark.asyncio
    async def test_endurance_test_response_time_stability(self, endurance_tester) -> Any:
        """Test response time stability over time."""
        response_times = []
        
        async def timing_monitored_operation():
            
    """timing_monitored_operation function."""
start_time = time.time()
            await asyncio.sleep(0.01)
            end_time = time.time()
            
            response_time = end_time - start_time
            response_times.append(response_time)
            
            return {"response_time": response_time}
        
        # Run endurance test
        results = await endurance_tester.run_single_load_test(
            timing_monitored_operation,
            duration=25.0,
            target_rps=20.0,
            max_concurrent=12
        )
        
        # Verify response time stability
        if len(response_times) > 10:
            response_time_variance = statistics.variance(response_times)
            assert response_time_variance < 0.001  # Response times should be stable


class TestOptimizedScalabilityTesting:
    """Optimized scalability testing tests."""
    
    @pytest.fixture
    def scalability_tester(self) -> Any:
        """Scalability tester fixture."""
        return OptimizedLoadTester()
    
    @pytest.mark.asyncio
    async def test_scalability_concurrent_users(self, scalability_tester) -> Any:
        """Test scalability with increasing concurrent users."""
        concurrency_levels = [5, 10, 20, 30]
        results_by_level = {}
        
        async def scalable_operation():
            
    """scalable_operation function."""
await asyncio.sleep(0.01)
            return {"status": "success"}
        
        # Test different concurrency levels
        for concurrency in concurrency_levels:
            results = await scalability_tester.run_single_load_test(
                scalable_operation,
                duration=10.0,
                target_rps=50.0,
                max_concurrent=concurrency
            )
            
            results_by_level[concurrency] = results
        
        # Verify scalability
        for concurrency in concurrency_levels:
            results = results_by_level[concurrency]
            assert results["success_rate"] > 0.8  # Good success rate at all levels
            assert results["requests_per_second"] > 10  # Reasonable throughput
    
    @pytest.mark.asyncio
    async async def test_scalability_request_rate(self, scalability_tester) -> Any:
        """Test scalability with increasing request rates."""
        rps_levels = [10, 25, 50, 75]
        results_by_rps = {}
        
        async def rate_scalable_operation():
            
    """rate_scalable_operation function."""
await asyncio.sleep(0.005)
            return {"status": "success"}
        
        # Test different request rates
        for rps in rps_levels:
            results = await scalability_tester.run_single_load_test(
                rate_scalable_operation,
                duration=15.0,
                target_rps=rps,
                max_concurrent=25
            )
            
            results_by_rps[rps] = results
        
        # Verify scalability
        for rps in rps_levels:
            results = results_by_rps[rps]
            assert results["success_rate"] > 0.7  # Reasonable success rate
            assert results["requests_per_second"] > rps * 0.5  # At least 50% of target RPS


# Export test classes
__all__ = [
    "OptimizedLoadTester",
    "TestOptimizedLoadTesting",
    "TestOptimizedStressTesting",
    "TestOptimizedEnduranceTesting",
    "TestOptimizedScalabilityTesting"
] 