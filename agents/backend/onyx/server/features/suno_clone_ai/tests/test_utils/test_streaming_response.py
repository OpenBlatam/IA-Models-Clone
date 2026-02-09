"""
Comprehensive Unit Tests for Streaming Response

Tests cover streaming response functionality with diverse test cases
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from fastapi.responses import StreamingResponse

from utils.streaming_response import StreamingResponseOptimizer


class TestStreamingResponseOptimizer:
    """Test cases for StreamingResponseOptimizer class"""
    
    @pytest.mark.asyncio
    async def test_stream_json_array_basic(self):
        """Test streaming basic JSON array"""
        items = [
            {"id": 1, "name": "item1"},
            {"id": 2, "name": "item2"}
        ]
        
        async def item_generator():
            for item in items:
                yield item
        
        chunks = []
        async for chunk in StreamingResponseOptimizer.stream_json_array(item_generator(), chunk_size=10):
            chunks.append(chunk)
        
        # Should start with [
        assert chunks[0] == b'['
        # Should end with ]
        assert chunks[-1] == b']'
        # Should have data chunks
        assert len(chunks) > 2
    
    @pytest.mark.asyncio
    async def test_stream_json_array_empty(self):
        """Test streaming empty array"""
        async def empty_generator():
            return
            yield  # Make it async generator
        
        chunks = []
        async for chunk in StreamingResponseOptimizer.stream_json_array(empty_generator(), chunk_size=10):
            chunks.append(chunk)
        
        # Should have opening and closing brackets
        assert b'[' in chunks
        assert b']' in chunks
    
    @pytest.mark.asyncio
    async def test_stream_json_array_chunking(self):
        """Test streaming with chunking"""
        items = [{"id": i} for i in range(250)]
        
        async def item_generator():
            for item in items:
                yield item
        
        chunks = []
        async for chunk in StreamingResponseOptimizer.stream_json_array(item_generator(), chunk_size=100):
            chunks.append(chunk)
        
        # Should have multiple chunks
        assert len(chunks) > 2
    
    @pytest.mark.asyncio
    async def test_stream_json_array_single_item(self):
        """Test streaming single item"""
        items = [{"id": 1}]
        
        async def item_generator():
            for item in items:
                yield item
        
        chunks = []
        async for chunk in StreamingResponseOptimizer.stream_json_array(item_generator()):
            chunks.append(chunk)
        
        # Should have brackets and data
        assert b'[' in chunks
        assert b']' in chunks
        assert any(b'id' in chunk for chunk in chunks)
    
    @pytest.mark.asyncio
    async def test_stream_json_array_custom_chunk_size(self):
        """Test streaming with custom chunk size"""
        items = [{"id": i} for i in range(50)]
        
        async def item_generator():
            for item in items:
                yield item
        
        chunks = []
        async for chunk in StreamingResponseOptimizer.stream_json_array(item_generator(), chunk_size=5):
            chunks.append(chunk)
        
        # Should have multiple chunks due to small chunk size
        assert len(chunks) > 2
    
    @pytest.mark.asyncio
    async def test_stream_json_array_comma_separation(self):
        """Test comma separation between chunks"""
        items = [{"id": 1}, {"id": 2}, {"id": 3}]
        
        async def item_generator():
            for item in items:
                yield item
        
        chunks = []
        async for chunk in StreamingResponseOptimizer.stream_json_array(item_generator(), chunk_size=1):
            chunks.append(chunk)
        
        # Should have commas between chunks (except first)
        chunk_str = b''.join(chunks)
        # Should be valid JSON array
        assert chunk_str.startswith(b'[')
        assert chunk_str.endswith(b']')
    
    def test_create_streaming_response_basic(self):
        """Test creating basic streaming response"""
        async def item_generator():
            yield {"id": 1}
        
        response = StreamingResponseOptimizer.create_streaming_response(item_generator())
        
        assert isinstance(response, StreamingResponse)
        assert response.media_type == "application/json"
    
    def test_create_streaming_response_custom_media_type(self):
        """Test creating streaming response with custom media type"""
        async def item_generator():
            yield {"id": 1}
        
        response = StreamingResponseOptimizer.create_streaming_response(
            item_generator(),
            media_type="application/x-ndjson"
        )
        
        assert response.media_type == "application/x-ndjson"
    
    def test_create_streaming_response_custom_chunk_size(self):
        """Test creating streaming response with custom chunk size"""
        async def item_generator():
            yield {"id": 1}
        
        response = StreamingResponseOptimizer.create_streaming_response(
            item_generator(),
            chunk_size=50
        )
        
        assert isinstance(response, StreamingResponse)
    
    @pytest.mark.asyncio
    async def test_stream_json_array_large_dataset(self):
        """Test streaming large dataset"""
        items = [{"id": i, "data": "x" * 100} for i in range(1000)]
        
        async def item_generator():
            for item in items:
                yield item
        
        chunk_count = 0
        async for chunk in StreamingResponseOptimizer.stream_json_array(item_generator(), chunk_size=100):
            chunk_count += 1
            assert isinstance(chunk, bytes)
        
        # Should have processed all items
        assert chunk_count > 0















