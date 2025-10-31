from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import pytest
from datetime import datetime
from onyx.server.features.key_messages.models import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Tests for Key Messages models.
"""
    KeyMessageRequest,
    GeneratedResponse,
    KeyMessageResponse,
    MessageType,
    MessageTone,
    MessageAnalysis,
    BatchKeyMessageRequest,
    BatchKeyMessageResponse
)

class TestMessageType:
    """Test MessageType enum."""
    
    def test_message_types(self) -> Any:
        """Test that all message types are available."""
        types = [mt.value for mt in MessageType]
        expected_types = [
            "marketing", "educational", "promotional", "informational",
            "call_to_action", "social_media", "email", "website"
        ]
        assert types == expected_types

class TestMessageTone:
    """Test MessageTone enum."""
    
    def test_message_tones(self) -> Any:
        """Test that all message tones are available."""
        tones = [mt.value for mt in MessageTone]
        expected_tones = [
            "professional", "casual", "friendly", "authoritative",
            "conversational", "enthusiastic", "urgent", "calm"
        ]
        assert tones == expected_tones

class TestKeyMessageRequest:
    """Test KeyMessageRequest model."""
    
    async def test_valid_request(self) -> Any:
        """Test creating a valid request."""
        request = KeyMessageRequest(
            message="Test message",
            message_type=MessageType.MARKETING,
            tone=MessageTone.PROFESSIONAL,
            target_audience="Test audience",
            keywords=["test", "keyword"]
        )
        
        assert request.message == "Test message"
        assert request.message_type == MessageType.MARKETING
        assert request.tone == MessageTone.PROFESSIONAL
        assert request.target_audience == "Test audience"
        assert request.keywords == ["test", "keyword"]
    
    def test_default_values(self) -> Any:
        """Test default values."""
        request = KeyMessageRequest(message="Test")
        
        assert request.message_type == MessageType.INFORMATIONAL
        assert request.tone == MessageTone.PROFESSIONAL
        assert request.keywords == []
        assert request.target_audience is None
        assert request.context is None
        assert request.max_length is None
    
    def test_message_validation(self) -> Any:
        """Test message validation."""
        # Empty message should raise validation error
        with pytest.raises(ValueError):
            KeyMessageRequest(message="")
        
        # Message too long should raise validation error
        with pytest.raises(ValueError):
            KeyMessageRequest(message="x" * 10001)

class TestGeneratedResponse:
    """Test GeneratedResponse model."""
    
    def test_valid_response(self) -> Any:
        """Test creating a valid response."""
        response = GeneratedResponse(
            id="test-id",
            original_message="Original",
            response="Generated response",
            message_type=MessageType.MARKETING,
            tone=MessageTone.PROFESSIONAL,
            word_count=2,
            character_count=18,
            processing_time=0.5
        )
        
        assert response.id == "test-id"
        assert response.original_message == "Original"
        assert response.response == "Generated response"
        assert response.message_type == MessageType.MARKETING
        assert response.tone == MessageTone.PROFESSIONAL
        assert response.word_count == 2
        assert response.character_count == 18
        assert response.processing_time == 0.5
    
    def test_default_values(self) -> Any:
        """Test default values."""
        response = GeneratedResponse(
            id="test-id",
            original_message="Original",
            response="Generated",
            message_type=MessageType.MARKETING,
            tone=MessageTone.PROFESSIONAL,
            word_count=1,
            character_count=9,
            processing_time=0.1
        )
        
        assert response.keywords_used == []
        assert response.sentiment_score is None
        assert response.readability_score is None
        assert response.suggestions == []
        assert isinstance(response.created_at, datetime)

class TestKeyMessageResponse:
    """Test KeyMessageResponse model."""
    
    def test_successful_response(self) -> Any:
        """Test successful response."""
        generated_response = GeneratedResponse(
            id="test-id",
            original_message="Original",
            response="Generated",
            message_type=MessageType.MARKETING,
            tone=MessageTone.PROFESSIONAL,
            word_count=1,
            character_count=9,
            processing_time=0.1
        )
        
        response = KeyMessageResponse(
            success=True,
            data=generated_response,
            processing_time=0.5
        )
        
        assert response.success is True
        assert response.data == generated_response
        assert response.error is None
        assert response.processing_time == 0.5
        assert response.suggestions == []
    
    def test_error_response(self) -> Any:
        """Test error response."""
        response = KeyMessageResponse(
            success=False,
            error="Test error",
            processing_time=0.1
        )
        
        assert response.success is False
        assert response.error == "Test error"
        assert response.data is None
        assert response.processing_time == 0.1

class TestBatchKeyMessageRequest:
    """Test BatchKeyMessageRequest model."""
    
    async def test_valid_batch_request(self) -> Any:
        """Test creating a valid batch request."""
        messages = [
            KeyMessageRequest(message="Message 1"),
            KeyMessageRequest(message="Message 2")
        ]
        
        request = BatchKeyMessageRequest(
            messages=messages,
            batch_size=10
        )
        
        assert len(request.messages) == 2
        assert request.batch_size == 10
    
    def test_default_batch_size(self) -> Any:
        """Test default batch size."""
        messages = [KeyMessageRequest(message="Test")]
        request = BatchKeyMessageRequest(messages=messages)
        
        assert request.batch_size == 10

class TestBatchKeyMessageResponse:
    """Test BatchKeyMessageResponse model."""
    
    def test_successful_batch_response(self) -> Any:
        """Test successful batch response."""
        results = [
            KeyMessageResponse(success=True, processing_time=0.1),
            KeyMessageResponse(success=True, processing_time=0.2)
        ]
        
        response = BatchKeyMessageResponse(
            success=True,
            results=results,
            total_processed=2,
            failed_count=0,
            processing_time=0.5
        )
        
        assert response.success is True
        assert len(response.results) == 2
        assert response.total_processed == 2
        assert response.failed_count == 0
        assert response.processing_time == 0.5
    
    def test_failed_batch_response(self) -> Any:
        """Test failed batch response."""
        results = [
            KeyMessageResponse(success=True, processing_time=0.1),
            KeyMessageResponse(success=False, error="Error", processing_time=0.1)
        ]
        
        response = BatchKeyMessageResponse(
            success=False,
            results=results,
            total_processed=2,
            failed_count=1,
            processing_time=0.3
        )
        
        assert response.success is False
        assert response.failed_count == 1
        assert response.total_processed == 2

class TestMessageAnalysis:
    """Test MessageAnalysis model."""
    
    def test_valid_analysis(self) -> Any:
        """Test creating a valid analysis."""
        analysis = MessageAnalysis(
            sentiment="positive",
            tone_consistency=0.8,
            clarity_score=0.9,
            engagement_potential=0.7,
            keyword_optimization=0.6,
            suggestions=["Suggestion 1", "Suggestion 2"]
        )
        
        assert analysis.sentiment == "positive"
        assert analysis.tone_consistency == 0.8
        assert analysis.clarity_score == 0.9
        assert analysis.engagement_potential == 0.7
        assert analysis.keyword_optimization == 0.6
        assert len(analysis.suggestions) == 2
    
    def test_default_suggestions(self) -> Any:
        """Test default suggestions."""
        analysis = MessageAnalysis(
            sentiment="neutral",
            tone_consistency=0.5,
            clarity_score=0.5,
            engagement_potential=0.5,
            keyword_optimization=0.5
        )
        
        assert analysis.suggestions == [] 