"""
Test Runner

Utilities for running tests.
"""

import logging
import unittest
from typing import List, Any, Optional

logger = logging.getLogger(__name__)


class TestRunner:
    """Run test suites."""
    
    def __init__(self):
        """Initialize test runner."""
        self.suite = unittest.TestSuite()
    
    def add_test(self, test_case: unittest.TestCase) -> None:
        """
        Add test case.
        
        Args:
            test_case: Test case
        """
        self.suite.addTest(test_case)
    
    def add_tests(self, test_cases: List[unittest.TestCase]) -> None:
        """
        Add multiple test cases.
        
        Args:
            test_cases: List of test cases
        """
        for test_case in test_cases:
            self.add_test(test_case)
    
    def run(
        self,
        verbosity: int = 2
    ) -> unittest.TestResult:
        """
        Run tests.
        
        Args:
            verbosity: Verbosity level
            
        Returns:
            Test result
        """
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(self.suite)
        
        logger.info(f"Tests run: {result.testsRun}")
        logger.info(f"Failures: {len(result.failures)}")
        logger.info(f"Errors: {len(result.errors)}")
        
        return result


def run_tests(
    test_cases: List[unittest.TestCase],
    verbosity: int = 2
) -> unittest.TestResult:
    """
    Run test cases.
    
    Args:
        test_cases: List of test cases
        verbosity: Verbosity level
        
    Returns:
        Test result
    """
    runner = TestRunner()
    runner.add_tests(test_cases)
    return runner.run(verbosity)


def create_test_suite() -> TestRunner:
    """Create test suite."""
    return TestRunner()



