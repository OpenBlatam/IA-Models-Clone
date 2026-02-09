"""
Test Runner for Flux2 Clothing Changer
=======================================

Automated testing system for model and API testing.
"""

import unittest
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Test status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TestResult:
    """Test result."""
    name: str
    status: TestStatus
    duration: float
    error: Optional[str] = None
    output: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TestRunner:
    """Automated test runner."""
    
    def __init__(self):
        """Initialize test runner."""
        self.tests: List[Callable] = []
        self.results: List[TestResult] = []
        self.on_test_complete: Optional[Callable] = None
    
    def register_test(self, test_func: Callable) -> None:
        """
        Register a test function.
        
        Args:
            test_func: Test function to register
        """
        self.tests.append(test_func)
        logger.debug(f"Registered test: {test_func.__name__}")
    
    def run_test(self, test_func: Callable) -> TestResult:
        """
        Run a single test.
        
        Args:
            test_func: Test function to run
            
        Returns:
            Test result
        """
        test_name = test_func.__name__
        start_time = time.time()
        
        result = TestResult(
            name=test_name,
            status=TestStatus.RUNNING,
            duration=0.0,
        )
        
        try:
            # Run test
            output = test_func()
            duration = time.time() - start_time
            
            result.status = TestStatus.PASSED
            result.duration = duration
            result.output = str(output) if output else None
            
            logger.info(f"Test passed: {test_name} ({duration:.2f}s)")
            
        except AssertionError as e:
            duration = time.time() - start_time
            result.status = TestStatus.FAILED
            result.duration = duration
            result.error = str(e)
            logger.error(f"Test failed: {test_name} - {e}")
            
        except Exception as e:
            duration = time.time() - start_time
            result.status = TestStatus.FAILED
            result.duration = duration
            result.error = str(e)
            logger.error(f"Test error: {test_name} - {e}")
        
        self.results.append(result)
        
        # Callback
        if self.on_test_complete:
            try:
                self.on_test_complete(result)
            except Exception as e:
                logger.error(f"Test callback error: {e}")
        
        return result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all registered tests.
        
        Returns:
            Test summary
        """
        logger.info(f"Running {len(self.tests)} tests...")
        
        start_time = time.time()
        
        for test_func in self.tests:
            self.run_test(test_func)
        
        total_duration = time.time() - start_time
        
        # Calculate statistics
        passed = sum(1 for r in self.results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAILED)
        skipped = sum(1 for r in self.results if r.status == TestStatus.SKIPPED)
        
        summary = {
            "total": len(self.tests),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "duration": total_duration,
            "success_rate": (passed / len(self.tests) * 100) if self.tests else 0.0,
            "results": self.results,
        }
        
        logger.info(f"Tests completed: {passed} passed, {failed} failed, {skipped} skipped")
        
        return summary
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get test statistics."""
        if not self.results:
            return {
                "total_tests": 0,
                "registered_tests": len(self.tests),
            }
        
        passed = sum(1 for r in self.results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAILED)
        avg_duration = sum(r.duration for r in self.results) / len(self.results)
        
        return {
            "total_tests": len(self.results),
            "registered_tests": len(self.tests),
            "passed": passed,
            "failed": failed,
            "success_rate": (passed / len(self.results) * 100) if self.results else 0.0,
            "average_duration": avg_duration,
        }

