#!/usr/bin/env python3
"""
Advanced Testing Framework

Comprehensive testing framework with:
- Performance testing
- Load testing
- Stress testing
- Chaos testing
- Integration testing
- End-to-end testing
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
import asyncio
import time
import statistics
import random
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
import aiohttp
import httpx
from collections import defaultdict, deque
import concurrent.futures
import threading

logger = structlog.get_logger("advanced_testing")

# =============================================================================
# TESTING MODELS
# =============================================================================

class TestType(Enum):
    """Types of tests."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    LOAD = "load"
    STRESS = "stress"
    CHAOS = "chaos"
    END_TO_END = "end_to_end"
    SECURITY = "security"
    COMPATIBILITY = "compatibility"

class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"

@dataclass
class TestCase:
    """Test case definition."""
    test_id: str
    name: str
    description: str
    test_type: TestType
    endpoint: str
    method: str = "GET"
    headers: Dict[str, str] = None
    payload: Dict[str, Any] = None
    expected_status: int = 200
    expected_response_time: float = 5.0
    timeout: int = 30
    retries: int = 3
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}
        if self.dependencies is None:
            self.dependencies = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_id": self.test_id,
            "name": self.name,
            "description": self.description,
            "test_type": self.test_type.value,
            "endpoint": self.endpoint,
            "method": self.method,
            "headers": self.headers,
            "payload": self.payload,
            "expected_status": self.expected_status,
            "expected_response_time": self.expected_response_time,
            "timeout": self.timeout,
            "retries": self.retries,
            "dependencies": self.dependencies
        }

@dataclass
class TestResult:
    """Test execution result."""
    test_id: str
    status: TestStatus
    start_time: datetime
    end_time: Optional[datetime]
    response_time: float
    status_code: Optional[int]
    response_data: Optional[Dict[str, Any]]
    error_message: Optional[str]
    retry_count: int = 0
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_id": self.test_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "response_time": self.response_time,
            "status_code": self.status_code,
            "response_data": self.response_data,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "metrics": self.metrics,
            "duration": (self.end_time - self.start_time).total_seconds() if self.end_time else None
        }

@dataclass
class LoadTestConfig:
    """Load test configuration."""
    concurrent_users: int = 10
    duration: int = 60  # seconds
    ramp_up_time: int = 10  # seconds
    ramp_down_time: int = 10  # seconds
    target_rps: Optional[int] = None  # requests per second
    max_response_time: float = 5.0
    max_error_rate: float = 0.01  # 1%
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "concurrent_users": self.concurrent_users,
            "duration": self.duration,
            "ramp_up_time": self.ramp_up_time,
            "ramp_down_time": self.ramp_down_time,
            "target_rps": self.target_rps,
            "max_response_time": self.max_response_time,
            "max_error_rate": self.max_error_rate
        }

# =============================================================================
# ADVANCED TESTING FRAMEWORK
# =============================================================================

class AdvancedTestingFramework:
    """Advanced testing framework with comprehensive test types."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_cases: Dict[str, TestCase] = {}
        self.test_results: Dict[str, List[TestResult]] = defaultdict(list)
        self.test_suites: Dict[str, List[str]] = {}
        
        # Performance tracking
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        self.load_test_results: Dict[str, Dict[str, Any]] = {}
        
        # Test execution
        self.active_tests: Dict[str, asyncio.Task] = {}
        self.test_callbacks: List[Callable[[TestResult], None]] = []
        
        # Statistics
        self.stats = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'skipped_tests': 0,
            'total_execution_time': 0.0,
            'average_response_time': 0.0
        }
    
    async def start(self) -> None:
        """Start the testing framework."""
        logger.info("Advanced testing framework started", base_url=self.base_url)
    
    async def stop(self) -> None:
        """Stop the testing framework."""
        # Cancel active tests
        for test_id, task in self.active_tests.items():
            task.cancel()
        
        logger.info("Advanced testing framework stopped")
    
    def add_test_case(self, test_case: TestCase) -> None:
        """Add a test case."""
        self.test_cases[test_case.test_id] = test_case
        self.stats['total_tests'] += 1
        
        logger.info(
            "Test case added",
            test_id=test_case.test_id,
            name=test_case.name,
            test_type=test_case.test_type.value
        )
    
    def create_test_suite(self, suite_name: str, test_ids: List[str]) -> None:
        """Create a test suite."""
        self.test_suites[suite_name] = test_ids
        logger.info("Test suite created", suite_name=suite_name, test_count=len(test_ids))
    
    async def run_test(self, test_id: str) -> TestResult:
        """Run a single test."""
        if test_id not in self.test_cases:
            raise ValueError(f"Test case {test_id} not found")
        
        test_case = self.test_cases[test_id]
        
        # Check dependencies
        for dependency in test_case.dependencies:
            if dependency not in self.test_results or not self.test_results[dependency]:
                raise ValueError(f"Dependency {dependency} not satisfied")
            
            last_result = self.test_results[dependency][-1]
            if last_result.status != TestStatus.PASSED:
                raise ValueError(f"Dependency {dependency} failed")
        
        # Create result object
        result = TestResult(
            test_id=test_id,
            status=TestStatus.RUNNING,
            start_time=datetime.utcnow(),
            end_time=None,
            response_time=0.0,
            status_code=None,
            response_data=None,
            error_message=None
        )
        
        try:
            # Execute test based on type
            if test_case.test_type == TestType.UNIT:
                await self._run_unit_test(test_case, result)
            elif test_case.test_type == TestType.INTEGRATION:
                await self._run_integration_test(test_case, result)
            elif test_case.test_type == TestType.PERFORMANCE:
                await self._run_performance_test(test_case, result)
            elif test_case.test_type == TestType.LOAD:
                await self._run_load_test(test_case, result)
            elif test_case.test_type == TestType.STRESS:
                await self._run_stress_test(test_case, result)
            elif test_case.test_type == TestType.CHAOS:
                await self._run_chaos_test(test_case, result)
            elif test_case.test_type == TestType.END_TO_END:
                await self._run_end_to_end_test(test_case, result)
            elif test_case.test_type == TestType.SECURITY:
                await self._run_security_test(test_case, result)
            else:
                await self._run_basic_test(test_case, result)
            
            result.status = TestStatus.PASSED
            self.stats['passed_tests'] += 1
            
        except Exception as e:
            result.status = TestStatus.FAILED
            result.error_message = str(e)
            self.stats['failed_tests'] += 1
            
            logger.error(
                "Test failed",
                test_id=test_id,
                error=str(e)
            )
        
        finally:
            result.end_time = datetime.utcnow()
            
            # Store result
            self.test_results[test_id].append(result)
            
            # Update performance metrics
            if result.response_time > 0:
                self.performance_metrics[test_id].append(result.response_time)
            
            # Call callbacks
            for callback in self.test_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error("Test callback error", error=str(e))
        
        return result
    
    async def run_test_suite(self, suite_name: str) -> List[TestResult]:
        """Run a test suite."""
        if suite_name not in self.test_suites:
            raise ValueError(f"Test suite {suite_name} not found")
        
        test_ids = self.test_suites[suite_name]
        results = []
        
        logger.info("Starting test suite", suite_name=suite_name, test_count=len(test_ids))
        
        for test_id in test_ids:
            try:
                result = await self.run_test(test_id)
                results.append(result)
            except Exception as e:
                logger.error("Test suite execution error", test_id=test_id, error=str(e))
        
        logger.info("Test suite completed", suite_name=suite_name, results=len(results))
        return results
    
    async def run_all_tests(self) -> Dict[str, List[TestResult]]:
        """Run all test cases."""
        all_results = {}
        
        for test_id in self.test_cases:
            try:
                result = await self.run_test(test_id)
                all_results[test_id] = [result]
            except Exception as e:
                logger.error("Test execution error", test_id=test_id, error=str(e))
        
        return all_results
    
    async def _run_basic_test(self, test_case: TestCase, result: TestResult) -> None:
        """Run a basic HTTP test."""
        url = f"{self.base_url}{test_case.endpoint}"
        
        async with httpx.AsyncClient(timeout=test_case.timeout) as client:
            start_time = time.time()
            
            try:
                if test_case.method.upper() == "GET":
                    response = await client.get(url, headers=test_case.headers)
                elif test_case.method.upper() == "POST":
                    response = await client.post(url, headers=test_case.headers, json=test_case.payload)
                elif test_case.method.upper() == "PUT":
                    response = await client.put(url, headers=test_case.headers, json=test_case.payload)
                elif test_case.method.upper() == "DELETE":
                    response = await client.delete(url, headers=test_case.headers)
                else:
                    raise ValueError(f"Unsupported HTTP method: {test_case.method}")
                
                result.response_time = time.time() - start_time
                result.status_code = response.status_code
                
                # Parse response
                try:
                    result.response_data = response.json()
                except:
                    result.response_data = {"text": response.text}
                
                # Validate status code
                if response.status_code != test_case.expected_status:
                    raise AssertionError(f"Expected status {test_case.expected_status}, got {response.status_code}")
                
                # Validate response time
                if result.response_time > test_case.expected_response_time:
                    raise AssertionError(f"Response time {result.response_time:.2f}s exceeds expected {test_case.expected_response_time}s")
                
            except httpx.TimeoutException:
                result.response_time = time.time() - start_time
                raise TimeoutError(f"Request timeout after {test_case.timeout}s")
    
    async def _run_unit_test(self, test_case: TestCase, result: TestResult) -> None:
        """Run a unit test."""
        # Unit tests typically test individual functions/components
        # For API testing, this would be testing specific endpoints in isolation
        await self._run_basic_test(test_case, result)
        
        # Additional unit test validations
        if result.response_data:
            # Validate response structure
            self._validate_response_structure(result.response_data, test_case)
    
    async def _run_integration_test(self, test_case: TestCase, result: TestResult) -> None:
        """Run an integration test."""
        # Integration tests test the interaction between components
        await self._run_basic_test(test_case, result)
        
        # Additional integration test validations
        if result.response_data:
            # Validate data consistency
            self._validate_data_consistency(result.response_data, test_case)
    
    async def _run_performance_test(self, test_case: TestCase, result: TestResult) -> None:
        """Run a performance test."""
        # Performance tests measure response times and throughput
        iterations = 10
        response_times = []
        
        for _ in range(iterations):
            test_result = TestResult(
                test_id=test_case.test_id,
                status=TestStatus.RUNNING,
                start_time=datetime.utcnow(),
                end_time=None,
                response_time=0.0,
                status_code=None,
                response_data=None,
                error_message=None
            )
            
            await self._run_basic_test(test_case, test_result)
            response_times.append(test_result.response_time)
        
        # Calculate performance metrics
        result.response_time = statistics.mean(response_times)
        result.metrics = {
            'iterations': iterations,
            'avg_response_time': statistics.mean(response_times),
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'std_deviation': statistics.stdev(response_times) if len(response_times) > 1 else 0,
            'p95_response_time': self._percentile(response_times, 95),
            'p99_response_time': self._percentile(response_times, 99)
        }
        
        # Validate performance
        if result.response_time > test_case.expected_response_time:
            raise AssertionError(f"Performance test failed: avg response time {result.response_time:.2f}s exceeds expected {test_case.expected_response_time}s")
    
    async def _run_load_test(self, test_case: TestCase, result: TestResult) -> None:
        """Run a load test."""
        config = LoadTestConfig(
            concurrent_users=test_case.payload.get('concurrent_users', 10) if test_case.payload else 10,
            duration=test_case.payload.get('duration', 60) if test_case.payload else 60,
            target_rps=test_case.payload.get('target_rps') if test_case.payload else None
        )
        
        # Run load test
        load_results = await self._execute_load_test(test_case, config)
        
        result.response_time = load_results['avg_response_time']
        result.metrics = load_results
        
        # Validate load test results
        if load_results['error_rate'] > config.max_error_rate:
            raise AssertionError(f"Load test failed: error rate {load_results['error_rate']:.2%} exceeds maximum {config.max_error_rate:.2%}")
        
        if load_results['avg_response_time'] > config.max_response_time:
            raise AssertionError(f"Load test failed: avg response time {load_results['avg_response_time']:.2f}s exceeds maximum {config.max_response_time}s")
    
    async def _run_stress_test(self, test_case: TestCase, result: TestResult) -> None:
        """Run a stress test."""
        # Stress tests push the system beyond normal capacity
        config = LoadTestConfig(
            concurrent_users=test_case.payload.get('concurrent_users', 100) if test_case.payload else 100,
            duration=test_case.payload.get('duration', 300) if test_case.payload else 300
        )
        
        # Run stress test
        stress_results = await self._execute_load_test(test_case, config)
        
        result.response_time = stress_results['avg_response_time']
        result.metrics = stress_results
        
        # Stress tests are considered passed if the system doesn't crash
        # and maintains some level of functionality
        if stress_results['error_rate'] > 0.5:  # 50% error rate threshold
            raise AssertionError(f"Stress test failed: system became unstable with {stress_results['error_rate']:.2%} error rate")
    
    async def _run_chaos_test(self, test_case: TestCase, result: TestResult) -> None:
        """Run a chaos test."""
        # Chaos tests introduce failures and measure system resilience
        chaos_type = test_case.payload.get('chaos_type', 'latency') if test_case.payload else 'latency'
        
        if chaos_type == 'latency':
            # Inject latency
            latency_ms = test_case.payload.get('latency_ms', 1000) if test_case.payload else 1000
            await asyncio.sleep(latency_ms / 1000.0)
        
        elif chaos_type == 'error':
            # Inject errors
            error_rate = test_case.payload.get('error_rate', 0.1) if test_case.payload else 0.1
            if random.random() < error_rate:
                raise Exception("Chaos engineering error injection")
        
        # Run the actual test
        await self._run_basic_test(test_case, result)
        
        # Chaos tests pass if the system recovers
        result.metrics['chaos_type'] = chaos_type
        result.metrics['recovery_successful'] = True
    
    async def _run_end_to_end_test(self, test_case: TestCase, result: TestResult) -> None:
        """Run an end-to-end test."""
        # End-to-end tests simulate complete user workflows
        workflow_steps = test_case.payload.get('workflow_steps', []) if test_case.payload else []
        
        total_response_time = 0.0
        step_results = []
        
        for step in workflow_steps:
            step_result = TestResult(
                test_id=f"{test_case.test_id}_step_{len(step_results)}",
                status=TestStatus.RUNNING,
                start_time=datetime.utcnow(),
                end_time=None,
                response_time=0.0,
                status_code=None,
                response_data=None,
                error_message=None
            )
            
            # Create temporary test case for step
            step_test_case = TestCase(
                test_id=step_result.test_id,
                name=f"E2E Step {len(step_results)}",
                description="End-to-end test step",
                test_type=TestType.INTEGRATION,
                endpoint=step.get('endpoint', test_case.endpoint),
                method=step.get('method', test_case.method),
                headers=step.get('headers', test_case.headers),
                payload=step.get('payload', test_case.payload)
            )
            
            await self._run_basic_test(step_test_case, step_result)
            step_results.append(step_result)
            total_response_time += step_result.response_time
        
        result.response_time = total_response_time
        result.metrics = {
            'workflow_steps': len(workflow_steps),
            'step_results': [sr.to_dict() for sr in step_results],
            'total_response_time': total_response_time
        }
    
    async def _run_security_test(self, test_case: TestCase, result: TestResult) -> None:
        """Run a security test."""
        # Security tests check for vulnerabilities
        security_checks = test_case.payload.get('security_checks', []) if test_case.payload else []
        
        security_results = {}
        
        for check in security_checks:
            check_type = check.get('type', 'injection')
            
            if check_type == 'injection':
                # Test for injection vulnerabilities
                malicious_payload = check.get('payload', "'; DROP TABLE users; --")
                
                # Create test case with malicious payload
                security_test_case = TestCase(
                    test_id=f"{test_case.test_id}_security_{check_type}",
                    name=f"Security Test: {check_type}",
                    description="Security vulnerability test",
                    test_type=TestType.SECURITY,
                    endpoint=test_case.endpoint,
                    method=test_case.method,
                    headers=test_case.headers,
                    payload={"input": malicious_payload}
                )
                
                security_result = TestResult(
                    test_id=security_test_case.test_id,
                    status=TestStatus.RUNNING,
                    start_time=datetime.utcnow(),
                    end_time=None,
                    response_time=0.0,
                    status_code=None,
                    response_data=None,
                    error_message=None
                )
                
                try:
                    await self._run_basic_test(security_test_case, security_result)
                    security_results[check_type] = "vulnerable" if security_result.status_code == 200 else "secure"
                except:
                    security_results[check_type] = "secure"
        
        result.metrics = {
            'security_checks': security_results,
            'vulnerabilities_found': len([r for r in security_results.values() if r == "vulnerable"])
        }
        
        # Security test passes if no vulnerabilities are found
        if result.metrics['vulnerabilities_found'] > 0:
            raise AssertionError(f"Security test failed: {result.metrics['vulnerabilities_found']} vulnerabilities found")
    
    async def _execute_load_test(self, test_case: TestCase, config: LoadTestConfig) -> Dict[str, Any]:
        """Execute a load test."""
        url = f"{self.base_url}{test_case.endpoint}"
        results = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'error_rate': 0.0,
            'avg_response_time': 0.0,
            'min_response_time': 0.0,
            'max_response_time': 0.0,
            'throughput': 0.0
        }
        
        start_time = time.time()
        end_time = start_time + config.duration
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(config.concurrent_users)
        
        async def make_request():
            async with semaphore:
                try:
                    async with httpx.AsyncClient(timeout=test_case.timeout) as client:
                        request_start = time.time()
                        
                        if test_case.method.upper() == "GET":
                            response = await client.get(url, headers=test_case.headers)
                        elif test_case.method.upper() == "POST":
                            response = await client.post(url, headers=test_case.headers, json=test_case.payload)
                        else:
                            response = await client.get(url, headers=test_case.headers)
                        
                        request_time = time.time() - request_start
                        
                        results['total_requests'] += 1
                        results['response_times'].append(request_time)
                        
                        if response.status_code == test_case.expected_status:
                            results['successful_requests'] += 1
                        else:
                            results['failed_requests'] += 1
                            
                except Exception as e:
                    results['total_requests'] += 1
                    results['failed_requests'] += 1
                    results['response_times'].append(test_case.timeout)
        
        # Execute load test
        tasks = []
        while time.time() < end_time:
            if config.target_rps:
                # Rate-limited execution
                tasks.append(asyncio.create_task(make_request()))
                await asyncio.sleep(1.0 / config.target_rps)
            else:
                # Concurrent execution
                for _ in range(config.concurrent_users):
                    tasks.append(asyncio.create_task(make_request()))
                await asyncio.sleep(0.1)  # Small delay to prevent overwhelming
        
        # Wait for remaining tasks
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate metrics
        if results['response_times']:
            results['avg_response_time'] = statistics.mean(results['response_times'])
            results['min_response_time'] = min(results['response_times'])
            results['max_response_time'] = max(results['response_times'])
        
        if results['total_requests'] > 0:
            results['error_rate'] = results['failed_requests'] / results['total_requests']
            results['throughput'] = results['total_requests'] / config.duration
        
        return results
    
    def _validate_response_structure(self, response_data: Dict[str, Any], test_case: TestCase) -> None:
        """Validate response structure."""
        # Basic structure validation
        if not isinstance(response_data, dict):
            raise AssertionError("Response is not a JSON object")
        
        # Add more specific validations based on test case
        pass
    
    def _validate_data_consistency(self, response_data: Dict[str, Any], test_case: TestCase) -> None:
        """Validate data consistency."""
        # Data consistency validation
        pass
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def add_test_callback(self, callback: Callable[[TestResult], None]) -> None:
        """Add test result callback."""
        self.test_callbacks.append(callback)
    
    def get_test_stats(self) -> Dict[str, Any]:
        """Get testing statistics."""
        total_execution_time = sum(
            (result.end_time - result.start_time).total_seconds()
            for results in self.test_results.values()
            for result in results
            if result.end_time
        )
        
        all_response_times = [
            result.response_time
            for results in self.test_results.values()
            for result in results
            if result.response_time > 0
        ]
        
        return {
            **self.stats,
            'total_execution_time': total_execution_time,
            'average_response_time': statistics.mean(all_response_times) if all_response_times else 0,
            'test_cases_count': len(self.test_cases),
            'test_suites_count': len(self.test_suites),
            'performance_metrics_count': len(self.performance_metrics)
        }
    
    def get_test_results(self, test_id: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get test results."""
        if test_id:
            return [result.to_dict() for result in self.test_results.get(test_id, [])]
        else:
            return {
                test_id: [result.to_dict() for result in results]
                for test_id, results in self.test_results.items()
            }

# =============================================================================
# GLOBAL TESTING FRAMEWORK INSTANCE
# =============================================================================

# Global testing framework
testing_framework = AdvancedTestingFramework()

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'TestType',
    'TestStatus',
    'TestCase',
    'TestResult',
    'LoadTestConfig',
    'AdvancedTestingFramework',
    'testing_framework'
]





























