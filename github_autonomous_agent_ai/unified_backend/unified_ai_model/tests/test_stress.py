"""
Stress tests for Unified AI Model API
Tests the API under heavy load to verify performance and stability.

Run with:
    $env:DEEPSEEK_API_KEY="sk-xxx"; python -m pytest unified_ai_model/tests/test_stress.py -v -s

Note: These tests make many API calls and may take several minutes.
"""

import os
import time
import asyncio
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
import pytest
import httpx

# Configuration
API_BASE_URL = "http://localhost:8050/api/v1"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
HAS_API_KEY = bool(DEEPSEEK_API_KEY) and DEEPSEEK_API_KEY != "sk-test-key-12345"

# Skip if no API key
pytestmark = pytest.mark.skipif(
    not HAS_API_KEY,
    reason="No valid API key configured"
)


class StressTestMetrics:
    """Collect and analyze stress test metrics."""
    
    def __init__(self):
        self.response_times: List[float] = []
        self.success_count = 0
        self.error_count = 0
        self.errors: List[str] = []
        self.start_time = 0.0
        self.end_time = 0.0
    
    def record_success(self, response_time: float):
        self.response_times.append(response_time)
        self.success_count += 1
    
    def record_error(self, error: str):
        self.error_count += 1
        self.errors.append(error)
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        self.end_time = time.time()
    
    @property
    def total_time(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def total_requests(self) -> int:
        return self.success_count + self.error_count
    
    @property
    def requests_per_second(self) -> float:
        if self.total_time > 0:
            return self.total_requests / self.total_time
        return 0.0
    
    @property
    def success_rate(self) -> float:
        if self.total_requests > 0:
            return (self.success_count / self.total_requests) * 100
        return 0.0
    
    @property
    def avg_response_time(self) -> float:
        if self.response_times:
            return statistics.mean(self.response_times)
        return 0.0
    
    @property
    def p50_response_time(self) -> float:
        if self.response_times:
            return statistics.median(self.response_times)
        return 0.0
    
    @property
    def p95_response_time(self) -> float:
        if len(self.response_times) >= 20:
            sorted_times = sorted(self.response_times)
            idx = int(len(sorted_times) * 0.95)
            return sorted_times[idx]
        return max(self.response_times) if self.response_times else 0.0
    
    @property
    def p99_response_time(self) -> float:
        if len(self.response_times) >= 100:
            sorted_times = sorted(self.response_times)
            idx = int(len(sorted_times) * 0.99)
            return sorted_times[idx]
        return max(self.response_times) if self.response_times else 0.0
    
    def report(self) -> str:
        return f"""
========================================
STRESS TEST RESULTS
========================================
Total Requests:     {self.total_requests}
Successful:         {self.success_count}
Failed:             {self.error_count}
Success Rate:       {self.success_rate:.2f}%
Total Time:         {self.total_time:.2f}s
Requests/Second:    {self.requests_per_second:.2f}

Response Times:
  Average:          {self.avg_response_time:.2f}ms
  Median (P50):     {self.p50_response_time:.2f}ms
  P95:              {self.p95_response_time:.2f}ms
  P99:              {self.p99_response_time:.2f}ms
  Min:              {min(self.response_times) if self.response_times else 0:.2f}ms
  Max:              {max(self.response_times) if self.response_times else 0:.2f}ms
========================================
"""


def sync_request(url: str, method: str = "GET", data: Dict = None) -> Tuple[bool, float, str]:
    """Make a synchronous HTTP request."""
    start = time.time()
    try:
        with httpx.Client(timeout=60.0) as client:
            if method == "GET":
                response = client.get(url)
            else:
                response = client.post(url, json=data)
            
            elapsed = (time.time() - start) * 1000  # ms
            
            if response.status_code == 200:
                return True, elapsed, ""
            else:
                return False, elapsed, f"Status {response.status_code}"
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return False, elapsed, str(e)


async def async_request(client: httpx.AsyncClient, url: str, method: str = "GET", data: Dict = None) -> Tuple[bool, float, str]:
    """Make an async HTTP request."""
    start = time.time()
    try:
        if method == "GET":
            response = await client.get(url)
        else:
            response = await client.post(url, json=data)
        
        elapsed = (time.time() - start) * 1000  # ms
        
        if response.status_code == 200:
            return True, elapsed, ""
        else:
            return False, elapsed, f"Status {response.status_code}"
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return False, elapsed, str(e)


class TestStressHealth:
    """Stress tests for health endpoint (lightweight)."""
    
    def test_health_burst_requests(self):
        """Test burst of health check requests."""
        metrics = StressTestMetrics()
        num_requests = 100
        max_workers = 20
        
        print(f"\n  Running {num_requests} health requests with {max_workers} workers...")
        
        metrics.start()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(sync_request, f"{API_BASE_URL}/health")
                for _ in range(num_requests)
            ]
            
            for future in as_completed(futures):
                success, elapsed, error = future.result()
                if success:
                    metrics.record_success(elapsed)
                else:
                    metrics.record_error(error)
        
        metrics.stop()
        print(metrics.report())
        
        # Assertions
        assert metrics.success_rate >= 95.0, f"Success rate too low: {metrics.success_rate}%"
        # Health endpoints should be fast, but under load we allow up to 10s
        assert metrics.avg_response_time < 10000, f"Avg response time too high: {metrics.avg_response_time}ms"
        print(f"  [OK] Success rate: {metrics.success_rate:.1f}%, Avg time: {metrics.avg_response_time:.0f}ms")


class TestStressChat:
    """Stress tests for chat endpoint."""
    
    def test_sequential_chat_requests(self):
        """Test sequential chat requests to measure baseline performance."""
        metrics = StressTestMetrics()
        num_requests = 10
        
        print(f"\n  Running {num_requests} sequential chat requests...")
        
        metrics.start()
        for i in range(num_requests):
            success, elapsed, error = sync_request(
                f"{API_BASE_URL}/chat",
                method="POST",
                data={"message": f"Di solo: {i}"}
            )
            if success:
                metrics.record_success(elapsed)
            else:
                metrics.record_error(error)
        
        metrics.stop()
        print(metrics.report())
        
        assert metrics.success_rate >= 80.0, f"Success rate too low: {metrics.success_rate}%"
    
    def test_concurrent_chat_requests(self):
        """Test concurrent chat requests."""
        metrics = StressTestMetrics()
        num_requests = 20
        max_workers = 5
        
        print(f"\n  Running {num_requests} concurrent chat requests with {max_workers} workers...")
        
        metrics.start()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    sync_request,
                    f"{API_BASE_URL}/chat",
                    "POST",
                    {"message": f"Di solo el numero: {i}"}
                )
                for i in range(num_requests)
            ]
            
            for future in as_completed(futures):
                success, elapsed, error = future.result()
                if success:
                    metrics.record_success(elapsed)
                else:
                    metrics.record_error(error)
        
        metrics.stop()
        print(metrics.report())
        
        assert metrics.success_rate >= 70.0, f"Success rate too low: {metrics.success_rate}%"


class TestStressAgents:
    """Stress tests for agent operations."""
    
    def test_rapid_agent_creation(self):
        """Test rapid creation of multiple agents."""
        metrics = StressTestMetrics()
        num_agents = 20
        
        print(f"\n  Creating {num_agents} agents rapidly...")
        
        agent_ids = []
        metrics.start()
        
        for i in range(num_agents):
            success, elapsed, error = sync_request(
                f"{API_BASE_URL}/agents",
                method="POST",
                data={"name": f"StressAgent_{i}"}
            )
            if success:
                metrics.record_success(elapsed)
            else:
                metrics.record_error(error)
        
        metrics.stop()
        print(metrics.report())
        
        # Cleanup - stop all agents
        try:
            with httpx.Client(timeout=30.0) as client:
                client.post(f"{API_BASE_URL}/agents/stop-all")
        except Exception:
            pass
        
        assert metrics.success_rate >= 90.0
    
    def test_agent_task_throughput(self):
        """Test throughput of task submissions to a single agent."""
        metrics = StressTestMetrics()
        num_tasks = 50
        
        # Create agent
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{API_BASE_URL}/agents",
                json={"name": "ThroughputAgent", "enable_parallel": True}
            )
            agent_id = resp.json()["agent_id"]
        
        print(f"\n  Submitting {num_tasks} tasks to agent...")
        
        metrics.start()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(
                    sync_request,
                    f"{API_BASE_URL}/agents/{agent_id}/tasks",
                    "POST",
                    {"description": f"Task {i}", "priority": i % 10}
                )
                for i in range(num_tasks)
            ]
            
            for future in as_completed(futures):
                success, elapsed, error = future.result()
                if success:
                    metrics.record_success(elapsed)
                else:
                    metrics.record_error(error)
        
        metrics.stop()
        print(metrics.report())
        
        # Cleanup
        try:
            with httpx.Client(timeout=30.0) as client:
                client.post(f"{API_BASE_URL}/agents/{agent_id}/stop")
        except Exception:
            pass
        
        assert metrics.success_rate >= 90.0


class TestStressConversations:
    """Stress tests for conversation management."""
    
    def test_multiple_conversations(self):
        """Test creating and using multiple conversations simultaneously."""
        metrics = StressTestMetrics()
        num_conversations = 10
        messages_per_conv = 3
        
        print(f"\n  Creating {num_conversations} conversations with {messages_per_conv} messages each...")
        
        def conversation_workflow(conv_id: int) -> List[Tuple[bool, float, str]]:
            results = []
            with httpx.Client(timeout=60.0) as client:
                # Create conversation
                start = time.time()
                try:
                    resp = client.post(
                        f"{API_BASE_URL}/conversations",
                        json={"system_prompt": f"Conv {conv_id}"}
                    )
                    elapsed = (time.time() - start) * 1000
                    if resp.status_code == 200:
                        results.append((True, elapsed, ""))
                        conversation_id = resp.json()["conversation_id"]
                        
                        # Send messages
                        for msg in range(messages_per_conv):
                            start = time.time()
                            resp = client.post(
                                f"{API_BASE_URL}/chat",
                                json={
                                    "message": f"Msg {msg}",
                                    "conversation_id": conversation_id
                                }
                            )
                            elapsed = (time.time() - start) * 1000
                            if resp.status_code == 200:
                                results.append((True, elapsed, ""))
                            else:
                                results.append((False, elapsed, f"Status {resp.status_code}"))
                    else:
                        results.append((False, elapsed, f"Status {resp.status_code}"))
                except Exception as e:
                    elapsed = (time.time() - start) * 1000
                    results.append((False, elapsed, str(e)))
            return results
        
        metrics.start()
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(conversation_workflow, i)
                for i in range(num_conversations)
            ]
            
            for future in as_completed(futures):
                for success, elapsed, error in future.result():
                    if success:
                        metrics.record_success(elapsed)
                    else:
                        metrics.record_error(error)
        
        metrics.stop()
        print(metrics.report())
        
        assert metrics.success_rate >= 70.0


class TestStressAsync:
    """Async stress tests for maximum concurrency."""
    
    @pytest.mark.asyncio
    async def test_async_health_flood(self):
        """Flood health endpoint with async requests."""
        metrics = StressTestMetrics()
        num_requests = 200
        
        print(f"\n  Flooding health endpoint with {num_requests} async requests...")
        
        metrics.start()
        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = [
                async_request(client, f"{API_BASE_URL}/health")
                for _ in range(num_requests)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    metrics.record_error(str(result))
                else:
                    success, elapsed, error = result
                    if success:
                        metrics.record_success(elapsed)
                    else:
                        metrics.record_error(error)
        
        metrics.stop()
        print(metrics.report())
        
        assert metrics.success_rate >= 90.0
        assert metrics.requests_per_second >= 10.0
    
    @pytest.mark.asyncio
    async def test_async_mixed_workload(self):
        """Test mixed workload of different endpoint types."""
        metrics = StressTestMetrics()
        
        print("\n  Running mixed async workload...")
        
        async def mixed_requests(client: httpx.AsyncClient):
            # Health check
            result = await async_request(client, f"{API_BASE_URL}/health")
            yield result
            
            # Stats
            result = await async_request(client, f"{API_BASE_URL}/stats")
            yield result
            
            # Chat (lightweight)
            result = await async_request(
                client, f"{API_BASE_URL}/chat", "POST",
                {"message": "OK"}
            )
            yield result
        
        metrics.start()
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Run 10 parallel mixed workloads
            async def run_workload():
                results = []
                async for result in mixed_requests(client):
                    results.append(result)
                return results
            
            tasks = [run_workload() for _ in range(10)]
            all_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for workload_results in all_results:
                if isinstance(workload_results, Exception):
                    metrics.record_error(str(workload_results))
                else:
                    for success, elapsed, error in workload_results:
                        if success:
                            metrics.record_success(elapsed)
                        else:
                            metrics.record_error(error)
        
        metrics.stop()
        print(metrics.report())
        
        assert metrics.success_rate >= 70.0


class TestStressEndurance:
    """Endurance tests for sustained load."""
    
    def test_sustained_load(self):
        """Test API under sustained load for extended period."""
        metrics = StressTestMetrics()
        duration_seconds = 30  # Run for 30 seconds
        requests_per_second = 2
        
        print(f"\n  Running sustained load test for {duration_seconds}s at {requests_per_second} req/s...")
        
        metrics.start()
        end_time = time.time() + duration_seconds
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            
            while time.time() < end_time:
                # Submit requests at target rate
                for _ in range(requests_per_second):
                    futures.append(
                        executor.submit(sync_request, f"{API_BASE_URL}/health")
                    )
                time.sleep(1.0)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    success, elapsed, error = future.result(timeout=10)
                    if success:
                        metrics.record_success(elapsed)
                    else:
                        metrics.record_error(error)
                except Exception as e:
                    metrics.record_error(str(e))
        
        metrics.stop()
        print(metrics.report())
        
        assert metrics.success_rate >= 95.0
        assert metrics.total_requests >= duration_seconds * requests_per_second * 0.8


class TestStressRecovery:
    """Test API recovery after stress."""
    
    def test_recovery_after_burst(self):
        """Test that API recovers properly after burst load."""
        # Phase 1: Burst load
        print("\n  Phase 1: Applying burst load...")
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(sync_request, f"{API_BASE_URL}/health")
                for _ in range(50)
            ]
            for future in as_completed(futures):
                future.result()
        
        # Wait for recovery
        print("  Waiting for recovery...")
        time.sleep(2)
        
        # Phase 2: Check normal operation
        print("  Phase 2: Checking normal operation...")
        metrics = StressTestMetrics()
        metrics.start()
        
        for _ in range(10):
            success, elapsed, error = sync_request(f"{API_BASE_URL}/health")
            if success:
                metrics.record_success(elapsed)
            else:
                metrics.record_error(error)
        
        metrics.stop()
        print(metrics.report())
        
        assert metrics.success_rate >= 90.0, "API did not recover properly"
        assert metrics.avg_response_time < 10000, "Response times still elevated"
        print(f"  [OK] API recovered - Success: {metrics.success_rate:.1f}%, Avg: {metrics.avg_response_time:.0f}ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
