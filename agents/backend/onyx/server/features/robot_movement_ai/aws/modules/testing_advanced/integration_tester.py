"""
Integration Tester
=================

Integration testing utilities.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Integration test case."""
    name: str
    test_func: Callable
    dependencies: List[str] = None
    timeout: Optional[float] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class TestResult:
    """Test result."""
    test_name: str
    passed: bool
    duration: float
    error: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class IntegrationTester:
    """Integration tester."""
    
    def __init__(self):
        self._test_cases: Dict[str, TestCase] = {}
        self._results: List[TestResult] = []
        self._fixtures: Dict[str, Any] = {}
    
    def register_test(
        self,
        name: str,
        test_func: Callable,
        dependencies: Optional[List[str]] = None,
        timeout: Optional[float] = None
    ):
        """Register integration test."""
        test_case = TestCase(
            name=name,
            test_func=test_func,
            dependencies=dependencies or [],
            timeout=timeout
        )
        
        self._test_cases[name] = test_case
        logger.info(f"Registered integration test: {name}")
    
    def register_fixture(self, name: str, fixture: Any):
        """Register test fixture."""
        self._fixtures[name] = fixture
        logger.info(f"Registered fixture: {name}")
    
    async def run_test(self, test_name: str) -> TestResult:
        """Run single test."""
        if test_name not in self._test_cases:
            raise ValueError(f"Test {test_name} not found")
        
        test_case = self._test_cases[test_name]
        start_time = asyncio.get_event_loop().time()
        
        try:
            if test_case.timeout:
                await asyncio.wait_for(
                    self._execute_test(test_case),
                    timeout=test_case.timeout
                )
            else:
                await self._execute_test(test_case)
            
            duration = asyncio.get_event_loop().time() - start_time
            result = TestResult(
                test_name=test_name,
                passed=True,
                duration=duration
            )
        
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            result = TestResult(
                test_name=test_name,
                passed=False,
                duration=duration,
                error=str(e)
            )
        
        self._results.append(result)
        return result
    
    async def _execute_test(self, test_case: TestCase):
        """Execute test case."""
        # Resolve dependencies
        fixtures = {
            name: self._fixtures[name]
            for name in test_case.dependencies
            if name in self._fixtures
        }
        
        # Execute test
        if asyncio.iscoroutinefunction(test_case.test_func):
            await test_case.test_func(**fixtures)
        else:
            await asyncio.to_thread(test_case.test_func, **fixtures)
    
    async def run_all_tests(self) -> List[TestResult]:
        """Run all tests."""
        results = []
        
        for test_name in self._test_cases.keys():
            result = await self.run_test(test_name)
            results.append(result)
        
        return results
    
    def get_results(self, test_name: Optional[str] = None) -> List[TestResult]:
        """Get test results."""
        results = self._results
        
        if test_name:
            results = [r for r in results if r.test_name == test_name]
        
        return results
    
    def get_test_stats(self) -> Dict[str, Any]:
        """Get test statistics."""
        return {
            "total_tests": len(self._test_cases),
            "total_results": len(self._results),
            "passed": sum(1 for r in self._results if r.passed),
            "failed": sum(1 for r in self._results if not r.passed),
            "pass_rate": (
                sum(1 for r in self._results if r.passed) / len(self._results) * 100
                if self._results else 0
            )
        }















