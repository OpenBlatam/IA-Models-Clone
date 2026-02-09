"""
Comprehensive Unit Tests for Async Helpers

Tests cover async utility functions with diverse test cases
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from utils.async_helpers import run_in_executor, batch_process, to_async


class TestRunInExecutor:
    """Test cases for run_in_executor function"""
    
    @pytest.mark.asyncio
    async def test_run_in_executor_simple_function(self):
        """Test running simple synchronous function"""
        def sync_func(x, y):
            return x + y
        
        result = await run_in_executor(sync_func, 5, 3)
        assert result == 8
    
    @pytest.mark.asyncio
    async def test_run_in_executor_with_kwargs(self):
        """Test running function with keyword arguments"""
        def sync_func(a, b, c=0):
            return a + b + c
        
        result = await run_in_executor(sync_func, 1, 2, c=3)
        assert result == 6
    
    @pytest.mark.asyncio
    async def test_run_in_executor_raises_exception(self):
        """Test executor raises exception from function"""
        def sync_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            await run_in_executor(sync_func)
    
    @pytest.mark.asyncio
    async def test_run_in_executor_returns_none(self):
        """Test function that returns None"""
        def sync_func():
            return None
        
        result = await run_in_executor(sync_func)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_run_in_executor_complex_return(self):
        """Test function returning complex object"""
        def sync_func():
            return {"key": "value", "list": [1, 2, 3]}
        
        result = await run_in_executor(sync_func)
        assert result == {"key": "value", "list": [1, 2, 3]}


class TestBatchProcess:
    """Test cases for batch_process function"""
    
    @pytest.mark.asyncio
    async def test_batch_process_simple(self):
        """Test processing simple items"""
        items = [1, 2, 3, 4, 5]
        
        async def processor(item):
            return item * 2
        
        results = await batch_process(items, processor, batch_size=2)
        assert results == [2, 4, 6, 8, 10]
    
    @pytest.mark.asyncio
    async def test_batch_process_empty_list(self):
        """Test processing empty list"""
        async def processor(item):
            return item
        
        results = await batch_process([], processor)
        assert results == []
    
    @pytest.mark.asyncio
    async def test_batch_process_single_item(self):
        """Test processing single item"""
        async def processor(item):
            return item * 2
        
        results = await batch_process([5], processor)
        assert results == [5 * 2]
    
    @pytest.mark.asyncio
    async def test_batch_process_custom_batch_size(self):
        """Test with custom batch size"""
        items = list(range(10))
        
        async def processor(item):
            return item
        
        results = await batch_process(items, processor, batch_size=3)
        assert len(results) == 10
        assert results == list(range(10))
    
    @pytest.mark.asyncio
    async def test_batch_process_max_concurrent(self):
        """Test with max concurrent limit"""
        items = list(range(10))
        call_count = []
        
        async def processor(item):
            call_count.append(item)
            await asyncio.sleep(0.01)  # Small delay
            return item
        
        results = await batch_process(items, processor, batch_size=10, max_concurrent=3)
        assert len(results) == 10
        # Verify concurrency was limited (all items processed)
        assert len(call_count) == 10
    
    @pytest.mark.asyncio
    async def test_batch_process_sync_function(self):
        """Test processing with synchronous function"""
        def sync_processor(item):
            return item * 2
        
        items = [1, 2, 3]
        results = await batch_process(items, sync_processor)
        assert results == [2, 4, 6]
    
    @pytest.mark.asyncio
    async def test_batch_process_raises_exception(self):
        """Test processor that raises exception"""
        async def processor(item):
            if item == 2:
                raise ValueError("Error on item 2")
            return item
        
        items = [1, 2, 3]
        with pytest.raises(ValueError, match="Error on item 2"):
            await batch_process(items, processor)
    
    @pytest.mark.asyncio
    async def test_batch_process_large_batch(self):
        """Test processing large batch"""
        items = list(range(100))
        
        async def processor(item):
            return item * 2
        
        results = await batch_process(items, processor, batch_size=20)
        assert len(results) == 100
        assert results == [i * 2 for i in range(100)]
    
    @pytest.mark.asyncio
    async def test_batch_process_uneven_batches(self):
        """Test processing with uneven batch division"""
        items = list(range(7))
        
        async def processor(item):
            return item
        
        results = await batch_process(items, processor, batch_size=3)
        assert len(results) == 7
        assert results == list(range(7))


class TestToAsync:
    """Test cases for to_async decorator"""
    
    @pytest.mark.asyncio
    async def test_to_async_simple_function(self):
        """Test converting simple function to async"""
        def sync_func(x):
            return x * 2
        
        async_func = to_async(sync_func)
        result = await async_func(5)
        assert result == 10
    
    @pytest.mark.asyncio
    async def test_to_async_with_args_and_kwargs(self):
        """Test async function with args and kwargs"""
        def sync_func(a, b, c=0):
            return a + b + c
        
        async_func = to_async(sync_func)
        result = await async_func(1, 2, c=3)
        assert result == 6
    
    @pytest.mark.asyncio
    async def test_to_async_preserves_function_name(self):
        """Test that function name is preserved"""
        def my_function():
            return "test"
        
        async_func = to_async(my_function)
        assert async_func.__name__ == "my_function"
    
    @pytest.mark.asyncio
    async def test_to_async_raises_exception(self):
        """Test async function raises exception"""
        def sync_func():
            raise ValueError("Test error")
        
        async_func = to_async(sync_func)
        with pytest.raises(ValueError, match="Test error"):
            await async_func()
    
    @pytest.mark.asyncio
    async def test_to_async_returns_none(self):
        """Test async function that returns None"""
        def sync_func():
            return None
        
        async_func = to_async(sync_func)
        result = await async_func()
        assert result is None
    
    @pytest.mark.asyncio
    async def test_to_async_complex_return(self):
        """Test async function with complex return value"""
        def sync_func():
            return {"result": [1, 2, 3], "status": "ok"}
        
        async_func = to_async(sync_func)
        result = await async_func()
        assert result == {"result": [1, 2, 3], "status": "ok"}
    
    @pytest.mark.asyncio
    async def test_to_async_multiple_calls(self):
        """Test calling async function multiple times"""
        call_count = 0
        
        def sync_func():
            nonlocal call_count
            call_count += 1
            return call_count
        
        async_func = to_async(sync_func)
        
        result1 = await async_func()
        result2 = await async_func()
        result3 = await async_func()
        
        assert result1 == 1
        assert result2 == 2
        assert result3 == 3















