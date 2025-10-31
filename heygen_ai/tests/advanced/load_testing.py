"""
Advanced Load Testing Framework for HeyGen AI system.
Comprehensive load testing including stress testing, performance testing,
and scalability analysis.
"""

import asyncio
import aiohttp
import time
import statistics
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import matplotlib.pyplot as plt
from collections import defaultdict, deque

@dataclass
class LoadTestConfig:
    """Configuration for load testing."""
    base_url: str = "http://localhost:8000"
    max_concurrent_users: int = 100
    test_duration: int = 300  # seconds
    ramp_up_time: int = 60  # seconds
    ramp_down_time: int = 30  # seconds
    think_time: float = 1.0  # seconds between requests
    timeout: int = 30  # seconds
    success_criteria: Dict[str, float] = field(default_factory=lambda: {
        "response_time_p95": 2.0,  # 95th percentile response time
        "error_rate": 0.01,  # 1% error rate
        "throughput": 100  # requests per second
    })

@dataclass
class LoadTestResult:
    """Result of a load test execution."""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration: float
    average_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    p99_response_time: float
    throughput: float  # requests per second
    error_rate: float
    concurrent_users: int
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ResourceMetrics:
    """System resource metrics during load testing."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float

class LoadTestScenario:
    """Represents a load test scenario."""
    
    def __init__(self, name: str, endpoint: str, method: str = "GET", 
                 headers: Dict[str, str] = None, payload: Dict[str, Any] = None,
                 weight: float = 1.0):
        self.name = name
        self.endpoint = endpoint
        self.method = method
        self.headers = headers or {}
        self.payload = payload or {}
        self.weight = weight
        self.results = []
    
    async def execute_request(self, session: aiohttp.ClientSession, base_url: str) -> Dict[str, Any]:
        """Execute a single request for this scenario."""
        start_time = time.time()
        
        try:
            url = f"{base_url.rstrip('/')}/{self.endpoint.lstrip('/')}"
            
            async with session.request(
                method=self.method,
                url=url,
                headers=self.headers,
                json=self.payload if self.method in ["POST", "PUT", "PATCH"] else None,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response_time = time.time() - start_time
                
                result = {
                    "scenario": self.name,
                    "status_code": response.status_code,
                    "response_time": response_time,
                    "success": 200 <= response.status_code < 400,
                    "timestamp": datetime.now(),
                    "content_length": response.headers.get("content-length", 0)
                }
                
                # Read response body for content length
                try:
                    body = await response.text()
                    result["content_length"] = len(body)
                except:
                    pass
                
                return result
        
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "scenario": self.name,
                "status_code": 0,
                "response_time": response_time,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now(),
                "content_length": 0
            }

class ResourceMonitor:
    """Monitors system resources during load testing."""
    
    def __init__(self, monitor_interval: float = 1.0):
        self.monitor_interval = monitor_interval
        self.metrics: List[ResourceMetrics] = []
        self.monitoring = False
        self.monitor_thread = None
        self.start_time = None
    
    def start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_resources(self):
        """Monitor system resources."""
        process = psutil.Process()
        initial_network = psutil.net_io_counters()
        initial_disk = psutil.disk_io_counters()
        
        while self.monitoring:
            try:
                # CPU and memory
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_used_mb = memory.used / 1024 / 1024
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                disk_io_read_mb = (disk_io.read_bytes - initial_disk.read_bytes) / 1024 / 1024
                disk_io_write_mb = (disk_io.write_bytes - initial_disk.write_bytes) / 1024 / 1024
                
                # Network I/O
                network_io = psutil.net_io_counters()
                network_sent_mb = (network_io.bytes_sent - initial_network.bytes_sent) / 1024 / 1024
                network_recv_mb = (network_io.bytes_recv - initial_network.bytes_recv) / 1024 / 1024
                
                metric = ResourceMetrics(
                    timestamp=datetime.now(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    memory_used_mb=memory_used_mb,
                    disk_io_read_mb=disk_io_read_mb,
                    disk_io_write_mb=disk_io_write_mb,
                    network_sent_mb=network_sent_mb,
                    network_recv_mb=network_recv_mb
                )
                
                self.metrics.append(metric)
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logging.error(f"Error monitoring resources: {e}")
                time.sleep(self.monitor_interval)
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average resource metrics."""
        if not self.metrics:
            return {}
        
        return {
            "avg_cpu_percent": statistics.mean(m.cpu_percent for m in self.metrics),
            "max_cpu_percent": max(m.cpu_percent for m in self.metrics),
            "avg_memory_percent": statistics.mean(m.memory_percent for m in self.metrics),
            "max_memory_percent": max(m.memory_percent for m in self.metrics),
            "avg_memory_used_mb": statistics.mean(m.memory_used_mb for m in self.metrics),
            "max_memory_used_mb": max(m.memory_used_mb for m in self.metrics),
            "total_disk_read_mb": self.metrics[-1].disk_io_read_mb if self.metrics else 0,
            "total_disk_write_mb": self.metrics[-1].disk_io_write_mb if self.metrics else 0,
            "total_network_sent_mb": self.metrics[-1].network_sent_mb if self.metrics else 0,
            "total_network_recv_mb": self.metrics[-1].network_recv_mb if self.metrics else 0
        }

class LoadTestRunner:
    """Main load test runner."""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.scenarios: List[LoadTestScenario] = []
        self.resource_monitor = ResourceMonitor()
        self.results: List[Dict[str, Any]] = []
    
    def add_scenario(self, scenario: LoadTestScenario):
        """Add a test scenario."""
        self.scenarios.append(scenario)
    
    async def run_load_test(self) -> LoadTestResult:
        """Run the load test."""
        print(f"🚀 Starting Load Test")
        print(f"   Target: {self.config.base_url}")
        print(f"   Max Users: {self.config.max_concurrent_users}")
        print(f"   Duration: {self.config.test_duration}s")
        print("=" * 50)
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        start_time = time.time()
        
        # Create HTTP session
        connector = aiohttp.TCPConnector(limit=self.config.max_concurrent_users * 2)
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Run load test
            await self._execute_load_test(session)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Stop resource monitoring
        self.resource_monitor.stop_monitoring()
        
        # Calculate results
        result = self._calculate_results(total_duration)
        
        # Print results
        self._print_results(result)
        
        return result
    
    async def _execute_load_test(self, session: aiohttp.ClientSession):
        """Execute the actual load test."""
        # Calculate ramp-up schedule
        ramp_up_steps = 10
        users_per_step = self.config.max_concurrent_users // ramp_up_steps
        
        # Ramp up phase
        print("📈 Ramp-up phase...")
        for step in range(ramp_up_steps):
            current_users = users_per_step * (step + 1)
            step_duration = self.config.ramp_up_time / ramp_up_steps
            
            print(f"   Step {step + 1}/{ramp_up_steps}: {current_users} users")
            
            # Create tasks for this step
            tasks = []
            for _ in range(users_per_step):
                task = self._run_user_session(session, step_duration)
                tasks.append(task)
            
            # Run tasks concurrently
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Steady state phase
        print("⚡ Steady state phase...")
        steady_duration = self.config.test_duration - self.config.ramp_up_time - self.config.ramp_down_time
        
        if steady_duration > 0:
            tasks = []
            for _ in range(self.config.max_concurrent_users):
                task = self._run_user_session(session, steady_duration)
                tasks.append(task)
            
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Ramp down phase
        print("📉 Ramp-down phase...")
        ramp_down_steps = 5
        users_per_step = self.config.max_concurrent_users // ramp_down_steps
        
        for step in range(ramp_down_steps):
            current_users = self.config.max_concurrent_users - (users_per_step * (step + 1))
            step_duration = self.config.ramp_down_time / ramp_down_steps
            
            if current_users > 0:
                print(f"   Step {step + 1}/{ramp_down_steps}: {current_users} users")
                
                tasks = []
                for _ in range(current_users):
                    task = self._run_user_session(session, step_duration)
                    tasks.append(task)
                
                await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _run_user_session(self, session: aiohttp.ClientSession, duration: float):
        """Run a single user session."""
        end_time = time.time() + duration
        
        while time.time() < end_time:
            # Select scenario based on weight
            scenario = self._select_scenario()
            
            # Execute request
            result = await scenario.execute_request(session, self.config.base_url)
            self.results.append(result)
            
            # Think time
            await asyncio.sleep(self.config.think_time)
    
    def _select_scenario(self) -> LoadTestScenario:
        """Select a scenario based on weights."""
        if not self.scenarios:
            raise ValueError("No scenarios defined")
        
        total_weight = sum(s.weight for s in self.scenarios)
        random_value = np.random.random() * total_weight
        
        current_weight = 0
        for scenario in self.scenarios:
            current_weight += scenario.weight
            if random_value <= current_weight:
                return scenario
        
        return self.scenarios[0]  # Fallback
    
    def _calculate_results(self, total_duration: float) -> LoadTestResult:
        """Calculate test results."""
        if not self.results:
            return LoadTestResult(
                test_name="load_test",
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                total_duration=total_duration,
                average_response_time=0.0,
                min_response_time=0.0,
                max_response_time=0.0,
                p95_response_time=0.0,
                p99_response_time=0.0,
                throughput=0.0,
                error_rate=0.0,
                concurrent_users=self.config.max_concurrent_users
            )
        
        # Basic statistics
        total_requests = len(self.results)
        successful_requests = sum(1 for r in self.results if r.get("success", False))
        failed_requests = total_requests - successful_requests
        
        # Response time statistics
        response_times = [r["response_time"] for r in self.results]
        response_times.sort()
        
        average_response_time = statistics.mean(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        
        # Percentiles
        p95_index = int(len(response_times) * 0.95)
        p99_index = int(len(response_times) * 0.99)
        
        p95_response_time = response_times[p95_index] if p95_index < len(response_times) else max_response_time
        p99_response_time = response_times[p99_index] if p99_index < len(response_times) else max_response_time
        
        # Throughput and error rate
        throughput = total_requests / total_duration if total_duration > 0 else 0
        error_rate = failed_requests / total_requests if total_requests > 0 else 0
        
        return LoadTestResult(
            test_name="load_test",
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_duration=total_duration,
            average_response_time=average_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            throughput=throughput,
            error_rate=error_rate,
            concurrent_users=self.config.max_concurrent_users
        )
    
    def _print_results(self, result: LoadTestResult):
        """Print test results."""
        print("\n" + "=" * 60)
        print("📊 LOAD TEST RESULTS")
        print("=" * 60)
        
        print(f"⏱️  Total Duration: {result.total_duration:.2f}s")
        print(f"👥 Concurrent Users: {result.concurrent_users}")
        print(f"📈 Total Requests: {result.total_requests}")
        print(f"✅ Successful: {result.successful_requests}")
        print(f"❌ Failed: {result.failed_requests}")
        print(f"📊 Error Rate: {result.error_rate:.2%}")
        print(f"🚀 Throughput: {result.throughput:.2f} req/s")
        
        print(f"\n⏱️  Response Times:")
        print(f"   Average: {result.average_response_time:.3f}s")
        print(f"   Min: {result.min_response_time:.3f}s")
        print(f"   Max: {result.max_response_time:.3f}s")
        print(f"   95th percentile: {result.p95_response_time:.3f}s")
        print(f"   99th percentile: {result.p99_response_time:.3f}s")
        
        # Resource metrics
        resource_metrics = self.resource_monitor.get_average_metrics()
        if resource_metrics:
            print(f"\n💻 System Resources:")
            print(f"   CPU: {resource_metrics['avg_cpu_percent']:.1f}% avg, {resource_metrics['max_cpu_percent']:.1f}% max")
            print(f"   Memory: {resource_metrics['avg_memory_percent']:.1f}% avg, {resource_metrics['max_memory_percent']:.1f}% max")
            print(f"   Memory Used: {resource_metrics['avg_memory_used_mb']:.1f}MB avg, {resource_metrics['max_memory_used_mb']:.1f}MB max")
        
        # Success criteria check
        print(f"\n🎯 Success Criteria:")
        criteria = self.config.success_criteria
        p95_pass = result.p95_response_time <= criteria.get("response_time_p95", float('inf'))
        error_pass = result.error_rate <= criteria.get("error_rate", 1.0)
        throughput_pass = result.throughput >= criteria.get("throughput", 0)
        
        print(f"   P95 Response Time ≤ {criteria.get('response_time_p95', 'N/A')}s: {'✅' if p95_pass else '❌'}")
        print(f"   Error Rate ≤ {criteria.get('error_rate', 'N/A'):.1%}: {'✅' if error_pass else '❌'}")
        print(f"   Throughput ≥ {criteria.get('throughput', 'N/A')} req/s: {'✅' if throughput_pass else '❌'}")
        
        overall_pass = p95_pass and error_pass and throughput_pass
        print(f"\n🎯 Overall Result: {'✅ PASSED' if overall_pass else '❌ FAILED'}")
        
        print("=" * 60)

class StressTestRunner:
    """Runs stress tests to find breaking points."""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.load_runner = LoadTestRunner(config)
    
    async def run_stress_test(self, max_users: int = 1000, step_size: int = 50) -> List[LoadTestResult]:
        """Run stress test with increasing load."""
        print("💥 Starting Stress Test")
        print(f"   Max Users: {max_users}")
        print(f"   Step Size: {step_size}")
        print("=" * 50)
        
        results = []
        current_users = step_size
        
        while current_users <= max_users:
            print(f"\n🔥 Testing with {current_users} users...")
            
            # Update config
            self.load_runner.config.max_concurrent_users = current_users
            self.load_runner.config.test_duration = 60  # 1 minute per step
            
            # Run test
            result = await self.load_runner.run_load_test()
            results.append(result)
            
            # Check if system is breaking
            if result.error_rate > 0.1:  # 10% error rate threshold
                print(f"⚠️  High error rate detected: {result.error_rate:.2%}")
                break
            
            if result.p95_response_time > 10.0:  # 10 second response time threshold
                print(f"⚠️  High response time detected: {result.p95_response_time:.2f}s")
                break
            
            current_users += step_size
        
        # Analyze results
        self._analyze_stress_results(results)
        
        return results
    
    def _analyze_stress_results(self, results: List[LoadTestResult]):
        """Analyze stress test results."""
        if not results:
            return
        
        print("\n📊 STRESS TEST ANALYSIS")
        print("=" * 40)
        
        # Find breaking point
        breaking_point = None
        for i, result in enumerate(results):
            if result.error_rate > 0.05 or result.p95_response_time > 5.0:
                breaking_point = result.concurrent_users
                break
        
        if breaking_point:
            print(f"💥 Breaking Point: {breaking_point} concurrent users")
        else:
            print(f"💪 System handled up to {results[-1].concurrent_users} users without breaking")
        
        # Performance degradation analysis
        if len(results) > 1:
            first_result = results[0]
            last_result = results[-1]
            
            response_time_increase = (last_result.average_response_time - first_result.average_response_time) / first_result.average_response_time
            throughput_change = (last_result.throughput - first_result.throughput) / first_result.throughput
            
            print(f"📈 Response Time Increase: {response_time_increase:.1%}")
            print(f"📊 Throughput Change: {throughput_change:.1%}")

class LoadTestingFramework:
    """Main load testing framework."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.config = LoadTestConfig(base_url=base_url)
        self.load_runner = LoadTestRunner(self.config)
        self.stress_runner = StressTestRunner(self.config)
    
    def add_scenario(self, name: str, endpoint: str, method: str = "GET", 
                    headers: Dict[str, str] = None, payload: Dict[str, Any] = None,
                    weight: float = 1.0):
        """Add a test scenario."""
        scenario = LoadTestScenario(name, endpoint, method, headers, payload, weight)
        self.load_runner.add_scenario(scenario)
        return scenario
    
    async def run_load_test(self, duration: int = 300, max_users: int = 100) -> LoadTestResult:
        """Run a load test."""
        self.config.test_duration = duration
        self.config.max_concurrent_users = max_users
        
        return await self.load_runner.run_load_test()
    
    async def run_stress_test(self, max_users: int = 1000) -> List[LoadTestResult]:
        """Run a stress test."""
        return await self.stress_runner.run_stress_test(max_users)
    
    def generate_report(self, results: List[LoadTestResult], output_file: str = "load_test_report.json"):
        """Generate a detailed report."""
        report = {
            "test_configuration": {
                "base_url": self.config.base_url,
                "max_concurrent_users": self.config.max_concurrent_users,
                "test_duration": self.config.test_duration,
                "success_criteria": self.config.success_criteria
            },
            "results": [
                {
                    "test_name": r.test_name,
                    "total_requests": r.total_requests,
                    "successful_requests": r.successful_requests,
                    "failed_requests": r.failed_requests,
                    "total_duration": r.total_duration,
                    "average_response_time": r.average_response_time,
                    "p95_response_time": r.p95_response_time,
                    "p99_response_time": r.p99_response_time,
                    "throughput": r.throughput,
                    "error_rate": r.error_rate,
                    "concurrent_users": r.concurrent_users,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in results
            ],
            "summary": {
                "total_tests": len(results),
                "average_throughput": statistics.mean(r.throughput for r in results) if results else 0,
                "average_error_rate": statistics.mean(r.error_rate for r in results) if results else 0,
                "max_concurrent_users": max(r.concurrent_users for r in results) if results else 0
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Report saved to: {output_file}")

# Example usage and demo
async def demo_load_testing():
    """Demonstrate load testing capabilities."""
    print("🚀 Load Testing Framework Demo")
    print("=" * 40)
    
    # Create load testing framework
    framework = LoadTestingFramework("http://localhost:8000")
    
    # Add test scenarios
    framework.add_scenario("Home Page", "/", "GET", weight=0.3)
    framework.add_scenario("API Health", "/health", "GET", weight=0.2)
    framework.add_scenario("User Login", "/api/login", "POST", 
                          payload={"username": "test", "password": "test"}, weight=0.3)
    framework.add_scenario("Data Fetch", "/api/data", "GET", weight=0.2)
    
    # Run load test
    print("\n🔥 Running Load Test...")
    result = await framework.run_load_test(duration=60, max_users=50)
    
    # Run stress test
    print("\n💥 Running Stress Test...")
    stress_results = await framework.run_stress_test(max_users=200)
    
    # Generate report
    all_results = [result] + stress_results
    framework.generate_report(all_results)

if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_load_testing())
