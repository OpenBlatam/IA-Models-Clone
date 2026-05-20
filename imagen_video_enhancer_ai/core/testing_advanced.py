"""
Advanced Testing System
========================

Advanced testing system with fixtures, mocks, and test utilities.
"""

import asyncio
import pytest
import logging
from typing import Dict, Any, Optional, List, Callable, Type
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class TestConfig:
    """Test configuration."""
    async_mode: bool = True
    timeout: Optional[float] = None
    retry_count: int = 0
    cleanup_after: bool = True
    verbose: bool = False
    capture_logs: bool = True


@dataclass
class TestResult:
    """Test result."""
    name: str
    passed: bool
    duration: float
    error: Optional[str] = None
    assertions: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TestFixture:
    """Test fixture manager."""
    
    def __init__(self):
        """Initialize test fixture."""
        self.fixtures: Dict[str, Any] = {}
        self.cleanup_handlers: List[Callable] = []
    
    def register(self, name: str, fixture: Any, cleanup: Optional[Callable] = None):
        """
        Register a fixture.
        
        Args:
            name: Fixture name
            fixture: Fixture object
            cleanup: Optional cleanup function
        """
        self.fixtures[name] = fixture
        if cleanup:
            self.cleanup_handlers.append(cleanup)
    
    def get(self, name: str) -> Any:
        """
        Get fixture by name.
        
        Args:
            name: Fixture name
            
        Returns:
            Fixture object
        """
        return self.fixtures.get(name)
    
    def cleanup(self):
        """Run cleanup handlers."""
        for handler in self.cleanup_handlers:
            try:
                handler()
            except Exception as e:
                logger.warning(f"Cleanup handler failed: {e}")


class MockBuilder:
    """Builder for creating mocks."""
    
    @staticmethod
    def create_async_mock(return_value: Any = None, side_effect: Any = None) -> AsyncMock:
        """
        Create async mock.
        
        Args:
            return_value: Return value
            side_effect: Side effect
            
        Returns:
            Async mock
        """
        mock = AsyncMock()
        if return_value is not None:
            mock.return_value = return_value
        if side_effect is not None:
            mock.side_effect = side_effect
        return mock
    
    @staticmethod
    def create_mock(return_value: Any = None, side_effect: Any = None) -> Mock:
        """
        Create mock.
        
        Args:
            return_value: Return value
            side_effect: Side effect
            
        Returns:
            Mock
        """
        mock = Mock()
        if return_value is not None:
            mock.return_value = return_value
        if side_effect is not None:
            mock.side_effect = side_effect
        return mock
    
    @staticmethod
    def create_magic_mock(return_value: Any = None, side_effect: Any = None) -> MagicMock:
        """
        Create magic mock.
        
        Args:
            return_value: Return value
            side_effect: Side effect
            
        Returns:
            Magic mock
        """
        mock = MagicMock()
        if return_value is not None:
            mock.return_value = return_value
        if side_effect is not None:
            mock.side_effect = side_effect
        return mock


class AsyncTestCase:
    """Base class for async test cases."""
    
    def __init__(self, config: Optional[TestConfig] = None):
        """
        Initialize async test case.
        
        Args:
            config: Test configuration
        """
        self.config = config or TestConfig()
        self.fixture = TestFixture()
        self.results: List[TestResult] = []
    
    async def setup(self):
        """Setup test case."""
        pass
    
    async def teardown(self):
        """Teardown test case."""
        if self.config.cleanup_after:
            self.fixture.cleanup()
    
    async def run_test(self, test_func: Callable, name: Optional[str] = None) -> TestResult:
        """
        Run a test function.
        
        Args:
            test_func: Test function
            name: Optional test name
            
        Returns:
            Test result
        """
        test_name = name or test_func.__name__
        start_time = datetime.now()
        
        try:
            await self.setup()
            
            if self.config.timeout:
                await asyncio.wait_for(test_func(), timeout=self.config.timeout)
            else:
                await test_func()
            
            duration = (datetime.now() - start_time).total_seconds()
            result = TestResult(
                name=test_name,
                passed=True,
                duration=duration
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            result = TestResult(
                name=test_name,
                passed=False,
                duration=duration,
                error=str(e)
            )
        finally:
            await self.teardown()
        
        self.results.append(result)
        return result
    
    def assert_async(self, coro, expected_result: Any = None, timeout: Optional[float] = None):
        """
        Assert async operation.
        
        Args:
            coro: Coroutine
            expected_result: Expected result
            timeout: Optional timeout
        """
        if timeout:
            result = asyncio.run(asyncio.wait_for(coro, timeout=timeout))
        else:
            result = asyncio.run(coro)
        
        if expected_result is not None:
            assert result == expected_result, f"Expected {expected_result}, got {result}"
        
        return result
    
    def assert_raises_async(self, exception_type: Type[Exception], coro):
        """
        Assert that async operation raises exception.
        
        Args:
            exception_type: Exception type
            coro: Coroutine
        """
        try:
            asyncio.run(coro)
            assert False, f"Expected {exception_type.__name__} to be raised"
        except exception_type:
            pass


@contextmanager
def temp_directory():
    """Context manager for temporary directory."""
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    try:
        yield Path(temp_dir)
    finally:
        shutil.rmtree(temp_dir)


@contextmanager
def temp_file(content: str = ""):
    """Context manager for temporary file."""
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(content)
        temp_path = Path(f.name)
    
    try:
        yield temp_path
    finally:
        if temp_path.exists():
            temp_path.unlink()


class TestRunner:
    """Test runner with reporting."""
    
    def __init__(self, config: Optional[TestConfig] = None):
        """
        Initialize test runner.
        
        Args:
            config: Test configuration
        """
        self.config = config or TestConfig()
        self.test_cases: List[AsyncTestCase] = []
        self.results: List[TestResult] = []
    
    def add_test_case(self, test_case: AsyncTestCase):
        """
        Add test case.
        
        Args:
            test_case: Test case
        """
        self.test_cases.append(test_case)
    
    async def run_all(self) -> Dict[str, Any]:
        """
        Run all test cases.
        
        Returns:
            Test results summary
        """
        all_results = []
        
        for test_case in self.test_cases:
            for result in test_case.results:
                all_results.append(result)
        
        self.results = all_results
        
        passed = sum(1 for r in all_results if r.passed)
        failed = len(all_results) - passed
        total_duration = sum(r.duration for r in all_results)
        
        return {
            "total": len(all_results),
            "passed": passed,
            "failed": failed,
            "success_rate": (passed / len(all_results) * 100) if all_results else 0,
            "total_duration": total_duration,
            "average_duration": (total_duration / len(all_results)) if all_results else 0,
            "results": [r.__dict__ for r in all_results]
        }
    
    def generate_report(self, output_file: Optional[Path] = None) -> str:
        """
        Generate test report.
        
        Args:
            output_file: Optional output file path
            
        Returns:
            Report string
        """
        summary = {
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": len(self.results) - sum(1 for r in self.results if r.passed)
        }
        
        report = f"""Test Report
===========

Total Tests: {summary['total']}
Passed: {summary['passed']}
Failed: {summary['failed']}
Success Rate: {(summary['passed'] / summary['total'] * 100) if summary['total'] > 0 else 0:.2f}%

Results:
"""
        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            report += f"\n{status}: {result.name} ({result.duration:.3f}s)"
            if result.error:
                report += f"\n  Error: {result.error}"
        
        if output_file:
            output_file.write_text(report, encoding='utf-8')
        
        return report


class TestDecorator:
    """Decorators for tests."""
    
    @staticmethod
    def async_test(timeout: Optional[float] = None, retry: int = 0):
        """
        Decorator for async tests.
        
        Args:
            timeout: Optional timeout
            retry: Retry count
        """
        def decorator(func: Callable):
            async def wrapper(*args, **kwargs):
                for attempt in range(retry + 1):
                    try:
                        if timeout:
                            await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                        else:
                            await func(*args, **kwargs)
                        return
                    except Exception as e:
                        if attempt == retry:
                            raise
                        await asyncio.sleep(0.1 * (attempt + 1))
            return wrapper
        return decorator
    
    @staticmethod
    def mock_patch(target: str, **mock_kwargs):
        """
        Decorator for patching with mock.
        
        Args:
            target: Target to patch
            **mock_kwargs: Mock arguments
        """
        def decorator(func: Callable):
            @patch(target, **mock_kwargs)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator



