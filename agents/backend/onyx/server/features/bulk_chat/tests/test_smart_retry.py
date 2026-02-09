"""
Tests for Smart Retry Manager
==============================
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from ..core.smart_retry_manager import SmartRetryManager, RetryStrategy


@pytest.fixture
def retry_manager():
    """Create smart retry manager for testing."""
    return SmartRetryManager()


@pytest.mark.asyncio
async def test_execute_with_retry_success(retry_manager):
    """Test successful execution with retry."""
    async def success_func():
        return "success"
    
    result = await retry_manager.execute_with_retry(
        operation_id="test_op",
        operation=success_func,
        max_retries=3
    )
    
    assert result == "success"


@pytest.mark.asyncio
async def test_execute_with_retry_failure(retry_manager):
    """Test retry on failure."""
    call_count = 0
    
    async def fail_then_succeed():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Temporary failure")
        return "success"
    
    result = await retry_manager.execute_with_retry(
        operation_id="test_op",
        operation=fail_then_succeed,
        max_retries=5
    )
    
    assert result == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_execute_with_retry_max_retries_exceeded(retry_manager):
    """Test retry when max retries exceeded."""
    async def always_fail():
        raise Exception("Always fails")
    
    with pytest.raises(Exception):
        await retry_manager.execute_with_retry(
            operation_id="test_op",
            operation=always_fail,
            max_retries=3
        )


@pytest.mark.asyncio
async def test_get_retry_stats(retry_manager):
    """Test getting retry statistics."""
    async def success_func():
        return "success"
    
    async def fail_func():
        raise Exception("fail")
    
    try:
        await retry_manager.execute_with_retry("op1", success_func, max_retries=3)
    except:
        pass
    
    try:
        await retry_manager.execute_with_retry("op2", fail_func, max_retries=3)
    except:
        pass
    
    stats = retry_manager.get_retry_stats()
    
    assert stats is not None
    assert "total_operations" in stats or "successful" in stats or "failed" in stats


@pytest.mark.asyncio
async def test_get_operation_history(retry_manager):
    """Test getting operation retry history."""
    async def func():
        return "result"
    
    await retry_manager.execute_with_retry("test_op", func, max_retries=3)
    
    history = retry_manager.get_operation_history("test_op")
    
    assert history is not None
    assert isinstance(history, list) or isinstance(history, dict)


