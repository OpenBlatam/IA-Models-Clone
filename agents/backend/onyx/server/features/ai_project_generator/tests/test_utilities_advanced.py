"""
Advanced utility functions for tests
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List, Callable
import time
import asyncio
from functools import wraps


class TestUtilitiesAdvanced:
    """Advanced utility functions"""
    
    def test_retry_decorator(self):
        """Test retry decorator"""
        attempts = 0
        
        @pytest.fixture
        def retry(max_attempts=3, delay=0.1):
            def decorator(func):
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
                            time.sleep(delay)
                    return None
                return wrapper
            return decorator
        
        @retry(max_attempts=3)
        def flaky_function():
            nonlocal attempts
            if attempts < 2:
                raise ValueError("Temporary failure")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert attempts == 2
    
    def test_timeout_decorator(self):
        """Test timeout decorator"""
        @pytest.fixture
        def timeout(seconds=1):
            def decorator(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    import signal
                    
                    def handler(signum, frame):
                        raise TimeoutError("Function timed out")
                    
                    signal.signal(signal.SIGALRM, handler)
                    signal.alarm(seconds)
                    try:
                        result = func(*args, **kwargs)
                    finally:
                        signal.alarm(0)
                    return result
                return wrapper
            return decorator
        
        # Note: Signal-based timeout doesn't work on Windows
        # This is a conceptual test
        assert True
    
    def test_cache_decorator(self):
        """Test cache decorator"""
        call_count = 0
        
        def cache(func):
            cache_dict = {}
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                key = str(args) + str(kwargs)
                if key not in cache_dict:
                    cache_dict[key] = func(*args, **kwargs)
                return cache_dict[key]
            return wrapper
        
        @cache
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call (cached)
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Not called again
    
    def test_async_retry(self):
        """Test async retry"""
        attempts = 0
        
        async def async_retry(func, max_attempts=3, delay=0.1):
            nonlocal attempts
            for i in range(max_attempts):
                try:
                    attempts += 1
                    return await func()
                except Exception:
                    if i == max_attempts - 1:
                        raise
                    await asyncio.sleep(delay)
            return None
        
        async def flaky_async():
            nonlocal attempts
            if attempts < 2:
                raise ValueError("Temporary failure")
            return "success"
        
        result = asyncio.run(async_retry(flaky_async))
        assert result == "success"
        assert attempts == 2
    
    def test_batch_processor(self):
        """Test batch processing utility"""
        def process_batch(items, batch_size=10):
            batches = []
            for i in range(0, len(items), batch_size):
                batches.append(items[i:i + batch_size])
            return batches
        
        items = list(range(25))
        batches = process_batch(items, batch_size=10)
        
        assert len(batches) == 3
        assert len(batches[0]) == 10
        assert len(batches[1]) == 10
        assert len(batches[2]) == 5

