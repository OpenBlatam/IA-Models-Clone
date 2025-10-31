from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from onyx.server.features.key_messages.service import KeyMessageService
from onyx.server.features.key_messages.models import (
from onyx.core.exceptions import ValidationException, ServiceException
from typing import Any, List, Dict, Optional
import logging
"""
Tests for Key Messages service.
"""
    KeyMessageRequest,
    MessageType,
    MessageTone,
    BatchKeyMessageRequest
)

class TestKeyMessageService:
    """Test KeyMessageService."""
    
    @pytest.fixture
    def service(self) -> Any:
        """Create a service instance for testing."""
        return KeyMessageService()
    
    @pytest.fixture
    async def valid_request(self) -> Any:
        """Create a valid request for testing."""
        return KeyMessageRequest(
            message="Test message",
            message_type=MessageType.MARKETING,
            tone=MessageTone.PROFESSIONAL,
            target_audience="Test audience",
            keywords=["test", "keyword"]
        )
    
    @pytest.mark.asyncio
    async def test_generate_response_success(self, service, valid_request) -> Any:
        """Test successful response generation."""
        with patch.object(service, '_generate_with_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Generated response"
            
            response = await service.generate_response(valid_request)
            
            assert response.success is True
            assert response.data is not None
            assert response.data.response == "Generated response"
            assert response.data.original_message == "Test message"
            assert response.data.message_type == MessageType.MARKETING
            assert response.data.tone == MessageTone.PROFESSIONAL
            assert response.error is None
            assert response.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_generate_response_empty_message(self, service) -> Any:
        """Test response generation with empty message."""
        request = KeyMessageRequest(message="")
        
        response = await service.generate_response(request)
        
        assert response.success is False
        assert "Message cannot be empty" in response.error
    
    @pytest.mark.asyncio
    async def test_generate_response_common_response(self, service) -> Any:
        """Test response generation with common response."""
        request = KeyMessageRequest(
            message="test_message",
            message_type=MessageType.INFORMATIONAL,
            tone=MessageTone.PROFESSIONAL
        )
        
        response = await service.generate_response(request)
        
        assert response.success is True
        assert response.data is not None
        assert "Test message received" in response.data.response
        assert response.data.metadata.get("from_cache") is True
    
    @pytest.mark.asyncio
    async def test_generate_response_cache(self, service, valid_request) -> Any:
        """Test response generation with caching."""
        # First call
        with patch.object(service, '_generate_with_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "First response"
            response1 = await service.generate_response(valid_request)
        
        # Second call should use cache
        response2 = await service.generate_response(valid_request)
        
        assert response1.success is True
        assert response2.success is True
        assert response1.data.response == response2.data.response
        assert response2.data.metadata.get("from_cache") is True
    
    @pytest.mark.asyncio
    async def test_analyze_message_success(self, service, valid_request) -> Any:
        """Test successful message analysis."""
        with patch.object(service, '_analyze_with_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Analysis result"
            
            response = await service.analyze_message(valid_request)
            
            assert response.success is True
            assert response.data is not None
            assert response.data.response == "Analysis result"
            assert response.data.metadata.get("analysis") is True
            assert response.error is None
            assert response.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_analyze_message_empty_message(self, service) -> Any:
        """Test message analysis with empty message."""
        request = KeyMessageRequest(message="")
        
        response = await service.analyze_message(request)
        
        assert response.success is False
        assert "Message cannot be empty" in response.error
    
    @pytest.mark.asyncio
    async def test_analyze_message_cache(self, service, valid_request) -> Any:
        """Test message analysis with caching."""
        # First call
        with patch.object(service, '_analyze_with_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "First analysis"
            response1 = await service.analyze_message(valid_request)
        
        # Second call should use cache
        response2 = await service.analyze_message(valid_request)
        
        assert response1.success is True
        assert response2.success is True
        assert response1.data.response == response2.data.response
        assert response2.data.metadata.get("from_cache") is True
    
    @pytest.mark.asyncio
    async def test_generate_batch_success(self, service) -> Any:
        """Test successful batch generation."""
        messages = [
            KeyMessageRequest(message="Message 1"),
            KeyMessageRequest(message="Message 2")
        ]
        request = BatchKeyMessageRequest(messages=messages, batch_size=10)
        
        with patch.object(service, '_generate_with_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Generated response"
            
            response = await service.generate_batch(request)
            
            assert response.success is True
            assert len(response.results) == 2
            assert response.total_processed == 2
            assert response.failed_count == 0
            assert all(r.success for r in response.results)
            assert response.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_generate_batch_with_failures(self, service) -> Any:
        """Test batch generation with some failures."""
        messages = [
            KeyMessageRequest(message="Message 1"),
            KeyMessageRequest(message="")  # This will fail
        ]
        request = BatchKeyMessageRequest(messages=messages, batch_size=10)
        
        with patch.object(service, '_generate_with_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Generated response"
            
            response = await service.generate_batch(request)
            
            assert response.success is False
            assert len(response.results) == 2
            assert response.total_processed == 2
            assert response.failed_count == 1
            assert response.results[0].success is True
            assert response.results[1].success is False
    
    @pytest.mark.asyncio
    async def test_generate_batch_size_limit(self, service) -> Any:
        """Test batch generation with size limit."""
        messages = [KeyMessageRequest(message=f"Message {i}") for i in range(100)]
        request = BatchKeyMessageRequest(messages=messages, batch_size=50)
        
        with patch.object(service, '_generate_with_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Generated response"
            
            response = await service.generate_batch(request)
            
            assert response.total_processed == 50  # Limited by batch_size
            assert len(response.results) == 50
    
    @pytest.mark.asyncio
    async def test_generate_with_llm_success(self, service, valid_request) -> Any:
        """Test successful LLM generation."""
        with patch.object(service, '_call_llm_api', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = "LLM generated response"
            
            result = await service._generate_with_llm(valid_request)
            
            assert result == "LLM generated response"
            mock_api.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_with_llm_failure(self, service, valid_request) -> Any:
        """Test LLM generation failure."""
        with patch.object(service, '_call_llm_api', new_callable=AsyncMock) as mock_api:
            mock_api.side_effect = Exception("LLM error")
            
            with pytest.raises(ServiceException):
                await service._generate_with_llm(valid_request)
    
    @pytest.mark.asyncio
    async def test_analyze_with_llm_success(self, service, valid_request) -> Any:
        """Test successful LLM analysis."""
        with patch.object(service, '_call_llm_api', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = "LLM analysis result"
            
            result = await service._analyze_with_llm(valid_request)
            
            assert result == "LLM analysis result"
            mock_api.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_with_llm_failure(self, service, valid_request) -> Any:
        """Test LLM analysis failure."""
        with patch.object(service, '_call_llm_api', new_callable=AsyncMock) as mock_api:
            mock_api.side_effect = Exception("LLM error")
            
            with pytest.raises(ServiceException):
                await service._analyze_with_llm(valid_request)
    
    @pytest.mark.asyncio
    async async def test_call_llm_api(self, service) -> Any:
        """Test LLM API call."""
        result = await service._call_llm_api("Test prompt")
        
        assert isinstance(result, str)
        assert "Generated response for:" in result
    
    def test_generate_cache_key(self, service) -> Any:
        """Test cache key generation."""
        data1 = {"message": "test", "type": "marketing"}
        data2 = {"message": "test", "type": "marketing"}
        data3 = {"message": "different", "type": "marketing"}
        
        key1 = service._generate_cache_key(data1)
        key2 = service._generate_cache_key(data2)
        key3 = service._generate_cache_key(data3)
        
        assert key1 == key2
        assert key1 != key3
        assert isinstance(key1, str)
        assert len(key1) == 32  # MD5 hash length
    
    def test_cache_operations(self, service) -> Any:
        """Test cache operations."""
        # Test setting and getting cache
        service._cache_response("test_key", "test_response")
        cached = service._get_cached_response("test_key")
        
        assert cached == "test_response"
        
        # Test non-existent key
        cached = service._get_cached_response("non_existent")
        assert cached is None
    
    @pytest.mark.asyncio
    async def test_clear_cache(self, service) -> Any:
        """Test cache clearing."""
        # Add some data to cache
        service._cache_response("test_key", "test_response")
        assert len(service.cache) == 1
        
        # Clear cache
        await service.clear_cache()
        assert len(service.cache) == 0
    
    @pytest.mark.asyncio
    async def test_get_cache_stats(self, service) -> Optional[Dict[str, Any]]:
        """Test cache statistics."""
        # Add some data to cache
        service._cache_response("test_key", "test_response")
        
        stats = await service.get_cache_stats()
        
        assert "cache_size" in stats
        assert "cache_ttl_hours" in stats
        assert "cache_keys" in stats
        assert stats["cache_size"] == 1
        assert stats["cache_ttl_hours"] == 24
        assert "test_key" in stats["cache_keys"]
    
    @pytest.mark.asyncio
    async def test_create_response(self, service, valid_request) -> Any:
        """Test response creation."""
        start_time = 0.0
        response_text = "Test response"
        
        response = await service._create_response(
            valid_request, response_text, start_time, from_cache=True
        )
        
        assert response.success is True
        assert response.data is not None
        assert response.data.response == response_text
        assert response.data.original_message == valid_request.message
        assert response.data.metadata.get("from_cache") is True
        assert response.processing_time >= 0
    
    @pytest.mark.asyncio
    async def test_create_analysis_response(self, service, valid_request) -> Any:
        """Test analysis response creation."""
        start_time = 0.0
        analysis_text = "Analysis result"
        
        response = await service._create_analysis_response(
            valid_request, analysis_text, start_time, from_cache=True
        )
        
        assert response.success is True
        assert response.data is not None
        assert response.data.response == analysis_text
        assert response.data.original_message == valid_request.message
        assert response.data.metadata.get("from_cache") is True
        assert response.data.metadata.get("analysis") is True
        assert response.processing_time >= 0 