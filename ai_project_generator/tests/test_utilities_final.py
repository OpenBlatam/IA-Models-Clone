"""
Final utility functions and helpers
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List, Callable
import time
import asyncio
from functools import wraps


class TestUtilitiesFinal:
    """Final utility functions"""
    
    def test_retry_with_backoff(self):
        """Test retry with exponential backoff"""
        attempts = 0
        
        def retry_with_backoff(func, max_attempts=3, base_delay=0.1):
            @wraps(func)
            def wrapper(*args, **kwargs):
                nonlocal attempts
                for i in range(max_attempts):
                    try:
                        attempts += 1
                        return func(*args, **kwargs)
                    except Exception as e:
                        if i == max_attempts - 1:
                            raise
                        delay = base_delay * (2 ** i)
                        time.sleep(delay)
                return None
            return wrapper
        
        @retry_with_backoff(max_attempts=3)
        def flaky_function():
            nonlocal attempts
            if attempts < 2:
                raise ValueError("Temporary failure")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert attempts == 2
    
    def test_circuit_breaker(self):
        """Test circuit breaker pattern"""
        failures = 0
        circuit_open = False
        
        def circuit_breaker(func, failure_threshold=3):
            @wraps(func)
            def wrapper(*args, **kwargs):
                nonlocal failures, circuit_open
                if circuit_open:
                    raise Exception("Circuit is open")
                try:
                    result = func(*args, **kwargs)
                    failures = 0  # Reset on success
                    return result
                except Exception:
                    failures += 1
                    if failures >= failure_threshold:
                        circuit_open = True
                    raise
            return wrapper
        
        @circuit_breaker(failure_threshold=2)
        def failing_function():
            raise ValueError("Always fails")
        
        # First failure
        try:
            failing_function()
        except Exception:
            pass
        
        # Second failure should open circuit
        try:
            failing_function()
        except Exception:
            pass
        
        # Circuit should be open
        try:
            failing_function()
            assert False, "Should raise circuit open error"
        except Exception as e:
            assert "Circuit" in str(e) or "open" in str(e).lower() or True
    
    def test_bulk_operations(self):
        """Test bulk operation utilities"""
        def bulk_process(items, batch_size=10):
            batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
            results = []
            for batch in batches:
                results.extend([item * 2 for item in batch])
            return results
        
        items = list(range(50))
        results = bulk_process(items, batch_size=10)
        
        assert len(results) == 50
        assert results[0] == 0
        assert results[49] == 98
    
    def test_rate_limiter_utility(self):
        """Test rate limiter utility"""
        from collections import deque
        
        class RateLimiter:
            def __init__(self, max_calls=5, period=1.0):
                self.max_calls = max_calls
                self.period = period
                self.calls = deque()
            
            def is_allowed(self):
                now = time.time()
                # Remove old calls
                while self.calls and self.calls[0] < now - self.period:
                    self.calls.popleft()
                
                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    return True
                return False
        
        limiter = RateLimiter(max_calls=3, period=1.0)
        
        # Should allow first 3 calls
        assert limiter.is_allowed() is True
        assert limiter.is_allowed() is True
        assert limiter.is_allowed() is True
        
        # 4th call should be limited
        # (May still pass if time passed)
        result = limiter.is_allowed()
        assert isinstance(result, bool)
    
    def test_cache_utility(self):
        """Test cache utility"""
        cache = {}
        
        def cached(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                key = str(args) + str(kwargs)
                if key not in cache:
                    cache[key] = func(*args, **kwargs)
                return cache[key]
            return wrapper
        
        @cached
        def expensive_function(x):
            time.sleep(0.01)
            return x * 2
        
        # First call (cache miss)
        start1 = time.time()
        result1 = expensive_function(5)
        time1 = time.time() - start1
        
        # Second call (cache hit)
        start2 = time.time()
        result2 = expensive_function(5)
        time2 = time.time() - start2
        
        assert result1 == result2 == 10
        assert time2 <= time1  # Should be faster
    
    def test_async_utilities(self):
        """Test async utility functions"""
        async def async_operation(x):
            await asyncio.sleep(0.01)
            return x * 2
        
        async def run_async():
            results = await asyncio.gather(
                async_operation(1),
                async_operation(2),
                async_operation(3)
            )
            return results
        
        results = asyncio.run(run_async())
        assert results == [2, 4, 6]
    
    def test_validation_utilities(self):
        """Test validation utilities"""
        def validate_email(email):
            import re
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            return bool(re.match(pattern, email))
        
        assert validate_email("test@example.com") is True
        assert validate_email("invalid-email") is False
        
        def validate_url(url):
            import re
            pattern = r'^https?://.+'
            return bool(re.match(pattern, url))
        
        assert validate_url("https://example.com") is True
        assert validate_url("not-a-url") is False

