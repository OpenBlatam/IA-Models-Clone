"""
Retry Testing Helpers
Specialized helpers for retry mechanism testing
"""

from typing import Any, Dict, List, Optional, Callable
from unittest.mock import Mock, AsyncMock
import asyncio
import time


class RetryTestHelpers:
    """Helpers for retry testing"""
    
    @staticmethod
    def create_mock_retry_strategy(
        max_retries: int = 3,
        retry_delay: float = 0.1,
        backoff_multiplier: float = 2.0
    ) -> Mock:
        """Create mock retry strategy"""
        strategy = Mock()
        strategy.max_retries = max_retries
        strategy.retry_delay = retry_delay
        strategy.backoff_multiplier = backoff_multiplier
        strategy.should_retry = Mock(return_value=True)
        strategy.get_delay = Mock(return_value=retry_delay)
        return strategy
    
    @staticmethod
    async def test_retry_mechanism(
        func: Callable,
        max_retries: int = 3,
        retry_delay: float = 0.1,
        should_succeed: bool = True
    ) -> Dict[str, Any]:
        """Test retry mechanism"""
        attempts = 0
        errors = []
        
        async def retry_wrapper():
            nonlocal attempts
            attempts += 1
            
            if asyncio.iscoroutinefunction(func):
                result = await func()
            else:
                result = func()
            
            if not should_succeed and attempts < max_retries:
                raise Exception(f"Attempt {attempts} failed")
            
            return result
        
        try:
            for attempt in range(max_retries):
                try:
                    result = await retry_wrapper()
                    return {
                        "success": True,
                        "attempts": attempts,
                        "result": result
                    }
                except Exception as e:
                    errors.append(str(e))
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                    else:
                        raise
            
            return {
                "success": False,
                "attempts": attempts,
                "errors": errors
            }
        except Exception as e:
            return {
                "success": False,
                "attempts": attempts,
                "errors": errors + [str(e)]
            }
    
    @staticmethod
    def assert_retry_occurred(
        retry_result: Dict[str, Any],
        expected_attempts: Optional[int] = None
    ):
        """Assert retry occurred"""
        assert retry_result["attempts"] > 1, "Retry did not occur"
        if expected_attempts:
            assert retry_result["attempts"] == expected_attempts, \
                f"Expected {expected_attempts} attempts, got {retry_result['attempts']}"


class ExponentialBackoffHelpers:
    """Helpers for exponential backoff testing"""
    
    @staticmethod
    def calculate_backoff_delay(
        attempt: int,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0
    ) -> float:
        """Calculate exponential backoff delay"""
        delay = base_delay * (multiplier ** attempt)
        return min(delay, max_delay)
    
    @staticmethod
    def assert_backoff_correct(
        delays: List[float],
        base_delay: float = 1.0,
        multiplier: float = 2.0
    ):
        """Assert backoff delays are correct"""
        for i, delay in enumerate(delays):
            expected = ExponentialBackoffHelpers.calculate_backoff_delay(
                i, base_delay, multiplier=multiplier
            )
            assert abs(delay - expected) < 0.1, \
                f"Backoff delay {delay} does not match expected {expected}"


# Convenience exports
create_mock_retry_strategy = RetryTestHelpers.create_mock_retry_strategy
test_retry_mechanism = RetryTestHelpers.test_retry_mechanism
assert_retry_occurred = RetryTestHelpers.assert_retry_occurred

calculate_backoff_delay = ExponentialBackoffHelpers.calculate_backoff_delay
assert_backoff_correct = ExponentialBackoffHelpers.assert_backoff_correct



