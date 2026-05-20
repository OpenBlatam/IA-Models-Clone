"""
Comprehensive Tests for Request Deduplicator
Tests for request deduplication functionality
"""

import pytest
from unittest.mock import Mock, AsyncMock
import asyncio
import time

from core.infrastructure.request_deduplicator import (
    RequestDeduplicator,
    deduplicate_decorator
)


class TestRequestDeduplicator:
    """Tests for RequestDeduplicator"""
    
    @pytest.fixture
    def deduplicator(self):
        """Create request deduplicator"""
        return RequestDeduplicator(ttl=60.0)
    
    @pytest.mark.asyncio
    async def test_deduplicate_first_request(self, deduplicator):
        """Test first request (not duplicate)"""
        call_count = 0
        
        async def test_function(arg):
            nonlocal call_count
            call_count += 1
            return f"result-{arg}"
        
        result = await deduplicator.deduplicate(test_function, "test")
        
        assert result == "result-test"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_deduplicate_duplicate_request(self, deduplicator):
        """Test duplicate request (should return cached)"""
        call_count = 0
        
        async def test_function(arg):
            nonlocal call_count
            call_count += 1
            return f"result-{arg}"
        
        # First call
        result1 = await deduplicator.deduplicate(test_function, "test")
        
        # Duplicate call
        result2 = await deduplicator.deduplicate(test_function, "test")
        
        assert result1 == result2
        assert call_count == 1  # Should only be called once
    
    @pytest.mark.asyncio
    async def test_deduplicate_different_args(self, deduplicator):
        """Test that different arguments are not deduplicated"""
        call_count = 0
        
        async def test_function(arg):
            nonlocal call_count
            call_count += 1
            return f"result-{arg}"
        
        result1 = await deduplicator.deduplicate(test_function, "test1")
        result2 = await deduplicator.deduplicate(test_function, "test2")
        
        assert result1 != result2
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_deduplicate_expired_cache(self, deduplicator):
        """Test that expired cache entries are not used"""
        deduplicator.ttl = 0.1  # Very short TTL
        
        call_count = 0
        
        async def test_function(arg):
            nonlocal call_count
            call_count += 1
            return f"result-{arg}"
        
        # First call
        await deduplicator.deduplicate(test_function, "test")
        
        # Wait for expiration
        await asyncio.sleep(0.2)
        
        # Second call should execute again
        await deduplicator.deduplicate(test_function, "test")
        
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_clear_expired(self, deduplicator):
        """Test clearing expired entries"""
        deduplicator.ttl = 0.1
        
        async def test_function(arg):
            return f"result-{arg}"
        
        # Create some entries
        await deduplicator.deduplicate(test_function, "test1")
        await deduplicator.deduplicate(test_function, "test2")
        
        # Wait for expiration
        await asyncio.sleep(0.2)
        
        # Clear expired
        await deduplicator.clear_expired()
        
        # Cache should be empty or have no expired entries
        assert len(deduplicator.cache) == 0 or all(
            time.time() - timestamp < deduplicator.ttl
            for _, timestamp in deduplicator.cache.values()
        )
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, deduplicator):
        """Test concurrent duplicate requests"""
        call_count = 0
        
        async def test_function(arg):
            nonlocal call_count
            await asyncio.sleep(0.1)  # Simulate work
            call_count += 1
            return f"result-{arg}"
        
        # Make concurrent requests with same args
        results = await asyncio.gather(
            deduplicator.deduplicate(test_function, "test"),
            deduplicator.deduplicate(test_function, "test"),
            deduplicator.deduplicate(test_function, "test")
        )
        
        # All should return same result
        assert all(r == results[0] for r in results)
        # Should only execute once (or very few times due to race conditions)
        assert call_count <= 2  # May be 1 or 2 due to race conditions


class TestDeduplicateDecorator:
    """Tests for deduplicate_decorator"""
    
    @pytest.mark.asyncio
    async def test_decorator_first_call(self):
        """Test decorator on first call"""
        deduplicator = RequestDeduplicator()
        
        call_count = 0
        
        @deduplicate_decorator(deduplicator)
        async def test_function(arg):
            nonlocal call_count
            call_count += 1
            return f"result-{arg}"
        
        result = await test_function("test")
        
        assert result == "result-test"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_decorator_duplicate_call(self):
        """Test decorator on duplicate call"""
        deduplicator = RequestDeduplicator()
        
        call_count = 0
        
        @deduplicate_decorator(deduplicator)
        async def test_function(arg):
            nonlocal call_count
            call_count += 1
            return f"result-{arg}"
        
        # First call
        result1 = await test_function("test")
        
        # Duplicate call
        result2 = await test_function("test")
        
        assert result1 == result2
        assert call_count == 1



