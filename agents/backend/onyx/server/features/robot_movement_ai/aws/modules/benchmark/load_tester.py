"""
Load Tester
===========

Load testing utilities.
"""

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class LoadTestResult:
    """Load test result."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    requests_per_second: float
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    p99_response_time: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class LoadTester:
    """Load tester."""
    
    def __init__(self):
        self._results: Dict[str, LoadTestResult] = {}
    
    async def run_load_test(
        self,
        name: str,
        func: Callable,
        concurrent_users: int = 10,
        duration_seconds: int = 60,
        *args,
        **kwargs
    ) -> LoadTestResult:
        """Run load test."""
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        response_times = []
        successful = 0
        failed = 0
        
        async def worker():
            nonlocal successful, failed
            while time.time() < end_time:
                try:
                    request_start = time.time()
                    
                    if asyncio.iscoroutinefunction(func):
                        await func(*args, **kwargs)
                    else:
                        await asyncio.to_thread(func, *args, **kwargs)
                    
                    response_time = time.time() - request_start
                    response_times.append(response_time)
                    successful += 1
                
                except Exception as e:
                    failed += 1
                    logger.error(f"Load test request failed: {e}")
        
        # Start concurrent workers
        workers = [asyncio.create_task(worker()) for _ in range(concurrent_users)]
        await asyncio.gather(*workers)
        
        total_time = time.time() - start_time
        total_requests = successful + failed
        
        if not response_times:
            raise ValueError("No successful requests")
        
        import statistics
        avg_response_time = statistics.mean(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        
        sorted_times = sorted(response_times)
        p95_index = int(len(sorted_times) * 0.95)
        p99_index = int(len(sorted_times) * 0.99)
        p95_response_time = sorted_times[min(p95_index, len(sorted_times) - 1)]
        p99_response_time = sorted_times[min(p99_index, len(sorted_times) - 1)]
        
        requests_per_second = total_requests / total_time if total_time > 0 else 0
        
        result = LoadTestResult(
            total_requests=total_requests,
            successful_requests=successful,
            failed_requests=failed,
            total_time=total_time,
            requests_per_second=requests_per_second,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time
        )
        
        self._results[name] = result
        logger.info(f"Load test {name} completed: {requests_per_second:.2f} req/s, {avg_response_time:.4f}s avg")
        
        return result
    
    def get_result(self, name: str) -> Optional[LoadTestResult]:
        """Get load test result."""
        return self._results.get(name)
    
    def get_all_results(self) -> Dict[str, LoadTestResult]:
        """Get all load test results."""
        return self._results.copy()















