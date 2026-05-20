"""
Testing Helpers
===============

Advanced testing utilities and fixtures.
"""

import asyncio
import logging
import tempfile
import shutil
from typing import Dict, Any, Optional, List, Callable, Awaitable
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Test result."""
    name: str
    passed: bool
    duration: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class TestRunner:
    """Test runner with reporting."""
    
    def __init__(self):
        """Initialize test runner."""
        self.results: List[TestResult] = []
    
    async def run_test(
        self,
        name: str,
        test_func: Callable[[], Awaitable[Any]],
        timeout: Optional[float] = None
    ) -> TestResult:
        """
        Run a single test.
        
        Args:
            name: Test name
            test_func: Test function
            timeout: Optional timeout
            
        Returns:
            Test result
        """
        start = datetime.now()
        
        try:
            if timeout:
                await asyncio.wait_for(test_func(), timeout=timeout)
            else:
                await test_func()
            
            duration = (datetime.now() - start).total_seconds()
            result = TestResult(
                name=name,
                passed=True,
                duration=duration
            )
        except asyncio.TimeoutError:
            duration = (datetime.now() - start).total_seconds()
            result = TestResult(
                name=name,
                passed=False,
                duration=duration,
                error=f"Test timed out after {timeout}s"
            )
        except Exception as e:
            duration = (datetime.now() - start).total_seconds()
            result = TestResult(
                name=name,
                passed=False,
                duration=duration,
                error=str(e)
            )
        
        self.results.append(result)
        return result
    
    async def run_tests(
        self,
        tests: List[tuple[str, Callable[[], Awaitable[Any]]]],
        parallel: bool = False
    ) -> List[TestResult]:
        """
        Run multiple tests.
        
        Args:
            tests: List of (name, test_func) tuples
            parallel: Whether to run in parallel
            
        Returns:
            List of test results
        """
        if parallel:
            tasks = [self.run_test(name, func) for name, func in tests]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for name, func in tests:
                result = await self.run_test(name, func)
                results.append(result)
            return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test summary."""
        total = len(self.results)
        passed = len([r for r in self.results if r.passed])
        failed = total - passed
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "success_rate": passed / total if total > 0 else 0.0,
            "total_duration": sum(r.duration for r in self.results),
            "avg_duration": sum(r.duration for r in self.results) / total if total > 0 else 0.0
        }
    
    def get_failed_tests(self) -> List[TestResult]:
        """Get failed tests."""
        return [r for r in self.results if not r.passed]


@asynccontextmanager
async def temp_directory():
    """Context manager for temporary directory."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@asynccontextmanager
async def mock_service(service_type: str, mock_func: Callable):
    """Context manager for mocking a service."""
    # In production, integrate with actual mocking library
    original = None
    try:
        # Mock setup would go here
        yield
    finally:
        # Mock teardown would go here
        pass


class AsyncTestCase:
    """Base class for async test cases."""
    
    def __init__(self):
        """Initialize test case."""
        self.setup_called = False
        self.teardown_called = False
    
    async def setup(self):
        """Setup method called before tests."""
        self.setup_called = True
    
    async def teardown(self):
        """Teardown method called after tests."""
        self.teardown_called = True
    
    async def run(self):
        """Run test case."""
        await self.setup()
        try:
            await self.test()
        finally:
            await self.teardown()
    
    async def test(self):
        """Test method to override."""
        raise NotImplementedError("Subclasses must implement test()")




