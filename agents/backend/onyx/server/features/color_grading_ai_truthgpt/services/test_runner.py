"""
Test Runner for Color Grading AI
==================================

Automated testing framework for services and components.
"""

import logging
import asyncio
import inspect
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Test status."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestResult:
    """Test result."""
    test_name: str
    status: TestStatus
    duration: float
    error: Optional[str] = None
    assertions: int = 0
    passed_assertions: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuite:
    """Test suite."""
    name: str
    tests: List[Callable]
    setup: Optional[Callable] = None
    teardown: Optional[Callable] = None


class TestRunner:
    """
    Test runner for automated testing.
    
    Features:
    - Unit test execution
    - Integration test execution
    - Test suites
    - Setup/teardown hooks
    - Assertion tracking
    - Test reporting
    - Parallel execution
    """
    
    def __init__(self):
        """Initialize test runner."""
        self._test_suites: Dict[str, TestSuite] = {}
        self._test_results: List[TestResult] = []
        self._assertions_count = 0
        self._passed_assertions = 0
    
    def register_suite(
        self,
        name: str,
        tests: List[Callable],
        setup: Optional[Callable] = None,
        teardown: Optional[Callable] = None
    ):
        """
        Register test suite.
        
        Args:
            name: Suite name
            tests: List of test functions
            setup: Optional setup function
            teardown: Optional teardown function
        """
        suite = TestSuite(
            name=name,
            tests=tests,
            setup=setup,
            teardown=teardown
        )
        self._test_suites[name] = suite
        logger.info(f"Registered test suite: {name} with {len(tests)} tests")
    
    async def run_test(
        self,
        test_func: Callable,
        suite_name: Optional[str] = None
    ) -> TestResult:
        """
        Run a single test.
        
        Args:
            test_func: Test function
            suite_name: Optional suite name
            
        Returns:
            Test result
        """
        test_name = test_func.__name__
        start_time = datetime.now()
        
        try:
            # Run test
            if asyncio.iscoroutinefunction(test_func):
                await test_func()
            else:
                test_func()
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return TestResult(
                test_name=test_name,
                status=TestStatus.PASSED,
                duration=duration,
                assertions=self._assertions_count,
                passed_assertions=self._passed_assertions
            )
        
        except AssertionError as e:
            duration = (datetime.now() - start_time).total_seconds()
            return TestResult(
                test_name=test_name,
                status=TestStatus.FAILED,
                duration=duration,
                error=str(e),
                assertions=self._assertions_count,
                passed_assertions=self._passed_assertions
            )
        
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                duration=duration,
                error=str(e),
                assertions=self._assertions_count,
                passed_assertions=self._passed_assertions
            )
    
    async def run_suite(
        self,
        suite_name: str,
        parallel: bool = False
    ) -> List[TestResult]:
        """
        Run test suite.
        
        Args:
            suite_name: Suite name
            parallel: Whether to run tests in parallel
            
        Returns:
            List of test results
        """
        suite = self._test_suites.get(suite_name)
        if not suite:
            logger.error(f"Test suite not found: {suite_name}")
            return []
        
        results = []
        
        # Run setup
        if suite.setup:
            try:
                if asyncio.iscoroutinefunction(suite.setup):
                    await suite.setup()
                else:
                    suite.setup()
            except Exception as e:
                logger.error(f"Setup failed for suite {suite_name}: {e}")
                return results
        
        # Run tests
        if parallel:
            tasks = [self.run_test(test, suite_name) for test in suite.tests]
            results = await asyncio.gather(*tasks)
        else:
            for test in suite.tests:
                result = await self.run_test(test, suite_name)
                results.append(result)
        
        # Run teardown
        if suite.teardown:
            try:
                if asyncio.iscoroutinefunction(suite.teardown):
                    await suite.teardown()
                else:
                    suite.teardown()
            except Exception as e:
                logger.error(f"Teardown failed for suite {suite_name}: {e}")
        
        self._test_results.extend(results)
        return results
    
    async def run_all_suites(self, parallel: bool = False) -> Dict[str, List[TestResult]]:
        """
        Run all test suites.
        
        Args:
            parallel: Whether to run suites in parallel
            
        Returns:
            Dictionary of suite results
        """
        all_results = {}
        
        if parallel:
            tasks = {
                name: self.run_suite(name, parallel=False)
                for name in self._test_suites.keys()
            }
            results = await asyncio.gather(*tasks.values())
            all_results = dict(zip(tasks.keys(), results))
        else:
            for suite_name in self._test_suites.keys():
                results = await self.run_suite(suite_name, parallel=False)
                all_results[suite_name] = results
        
        return all_results
    
    def assert_true(self, condition: bool, message: str = ""):
        """Assert that condition is True."""
        self._assertions_count += 1
        if condition:
            self._passed_assertions += 1
        else:
            raise AssertionError(f"Assertion failed: {message}")
    
    def assert_equal(self, actual: Any, expected: Any, message: str = ""):
        """Assert that actual equals expected."""
        self._assertions_count += 1
        if actual == expected:
            self._passed_assertions += 1
        else:
            raise AssertionError(f"Assertion failed: {actual} != {expected}. {message}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get test statistics."""
        total_tests = len(self._test_results)
        passed = sum(1 for r in self._test_results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self._test_results if r.status == TestStatus.FAILED)
        errors = sum(1 for r in self._test_results if r.status == TestStatus.ERROR)
        
        return {
            "total_tests": total_tests,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "success_rate": passed / total_tests if total_tests > 0 else 0.0,
            "total_assertions": self._assertions_count,
            "passed_assertions": self._passed_assertions,
        }


