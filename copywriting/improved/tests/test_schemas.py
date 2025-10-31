"""
Test Schemas
===========

Test Pydantic schemas for validation and serialization.
"""

import pytest
from datetime import datetime
from uuid import uuid4

from ..schemas import (
    CopywritingRequest,
    CopywritingResponse,
    CopywritingVariant,
    FeedbackRequest,
    FeedbackResponse,
    BatchCopywritingRequest,
    BatchCopywritingResponse,
    HealthCheckResponse,
    ErrorResponse,
    CopywritingTone,
    CopywritingStyle,
    CopywritingPurpose
)


class TestCopywritingRequest:
    """Test CopywritingRequest schema"""
    
    def test_valid_request(self):
        """Test valid request creation"""
        request = CopywritingRequest(
            topic="Test Topic",
            target_audience="Test Audience",
            key_points=["Point 1", "Point 2"],
            tone=CopywritingTone.PROFESSIONAL,
            style=CopywritingStyle.DIRECT_RESPONSE,
            purpose=CopywritingPurpose.SALES
        )
        
        assert request.topic == "Test Topic"
        assert request.target_audience == "Test Audience"
        assert request.key_points == ["Point 1", "Point 2"]
        assert request.tone == CopywritingTone.PROFESSIONAL
        assert request.style == CopywritingStyle.DIRECT_RESPONSE
        assert request.purpose == CopywritingPurpose.SALES
        assert request.include_cta is True
        assert request.language == "en"
        assert request.creativity_level == 0.7
        assert request.variants_count == 3
    
    def test_minimal_request(self):
        """Test minimal request with only required fields"""
        request = CopywritingRequest(
            topic="Minimal Topic",
            target_audience="Minimal Audience"
        )
        
        assert request.topic == "Minimal Topic"
        assert request.target_audience == "Minimal Audience"
        assert request.key_points == []
        assert request.tone == CopywritingTone.PROFESSIONAL
        assert request.style == CopywritingStyle.DIRECT_RESPONSE
        assert request.purpose == CopywritingPurpose.SALES
    
    def test_invalid_topic(self):
        """Test invalid topic validation"""
        with pytest.raises(ValueError):
            CopywritingRequest(
                topic="",  # Empty topic
                target_audience="Test Audience"
            )
    
    def test_invalid_word_count(self):
        """Test invalid word count validation"""
        with pytest.raises(ValueError):
            CopywritingRequest(
                topic="Test Topic",
                target_audience="Test Audience",
                word_count=10  # Too low
            )
    
    def test_invalid_creativity_level(self):
        """Test invalid creativity level validation"""
        with pytest.raises(ValueError):
            CopywritingRequest(
                topic="Test Topic",
                target_audience="Test Audience",
                creativity_level=1.5  # Too high
            )
    
    def test_key_points_validation(self):
        """Test key points validation and cleaning"""
        request = CopywritingRequest(
            topic="Test Topic",
            target_audience="Test Audience",
            key_points=["  Point 1  ", "", "Point 2", "   "]
        )
        
        # Should strip whitespace and remove empty strings
        assert request.key_points == ["Point 1", "Point 2"]


class TestCopywritingVariant:
    """Test CopywritingVariant schema"""
    
    def test_valid_variant(self):
        """Test valid variant creation"""
        variant = CopywritingVariant(
            title="Test Title",
            content="This is test content with enough words to meet the minimum requirement.",
            word_count=15,
            confidence_score=0.85
        )
        
        assert variant.title == "Test Title"
        assert variant.content == "This is test content with enough words to meet the minimum requirement."
        assert variant.word_count == 15
        assert variant.confidence_score == 0.85
        assert variant.cta is None
        assert isinstance(variant.id, type(uuid4()))
        assert isinstance(variant.created_at, datetime)
    
    def test_invalid_content_length(self):
        """Test invalid content length validation"""
        with pytest.raises(ValueError):
            CopywritingVariant(
                title="Test Title",
                content="Short",  # Too short
                word_count=1,
                confidence_score=0.85
            )
    
    def test_invalid_confidence_score(self):
        """Test invalid confidence score validation"""
        with pytest.raises(ValueError):
            CopywritingVariant(
                title="Test Title",
                content="This is test content with enough words to meet the minimum requirement.",
                word_count=15,
                confidence_score=1.5  # Too high
            )


class TestCopywritingResponse:
    """Test CopywritingResponse schema"""
    
    def test_valid_response(self):
        """Test valid response creation"""
        variant = CopywritingVariant(
            title="Test Title",
            content="This is test content with enough words to meet the minimum requirement.",
            word_count=15,
            confidence_score=0.85
        )
        
        response = CopywritingResponse(
            variants=[variant],
            processing_time_ms=1000
        )
        
        assert len(response.variants) == 1
        assert response.processing_time_ms == 1000
        assert response.total_variants == 1
        assert response.best_variant == variant
        assert isinstance(response.request_id, type(uuid4()))
        assert isinstance(response.created_at, datetime)
    
    def test_best_variant_selection(self):
        """Test best variant selection logic"""
        variant1 = CopywritingVariant(
            title="Variant 1",
            content="This is test content with enough words to meet the minimum requirement.",
            word_count=15,
            confidence_score=0.7
        )
        
        variant2 = CopywritingVariant(
            title="Variant 2",
            content="This is test content with enough words to meet the minimum requirement.",
            word_count=15,
            confidence_score=0.9
        )
        
        response = CopywritingResponse(
            variants=[variant1, variant2],
            processing_time_ms=1000
        )
        
        assert response.best_variant == variant2  # Higher confidence score


class TestFeedbackRequest:
    """Test FeedbackRequest schema"""
    
    def test_valid_feedback(self):
        """Test valid feedback creation"""
        feedback = FeedbackRequest(
            variant_id=uuid4(),
            rating=4,
            feedback_text="Great content!",
            improvements=["More specific examples"],
            is_helpful=True
        )
        
        assert feedback.rating == 4
        assert feedback.feedback_text == "Great content!"
        assert feedback.improvements == ["More specific examples"]
        assert feedback.is_helpful is True
    
    def test_invalid_rating(self):
        """Test invalid rating validation"""
        with pytest.raises(ValueError):
            FeedbackRequest(
                variant_id=uuid4(),
                rating=6,  # Too high
                is_helpful=True
            )
    
    def test_minimal_feedback(self):
        """Test minimal feedback with only required fields"""
        feedback = FeedbackRequest(
            variant_id=uuid4(),
            rating=3,
            is_helpful=False
        )
        
        assert feedback.rating == 3
        assert feedback.is_helpful is False
        assert feedback.feedback_text is None
        assert feedback.improvements is None


class TestBatchCopywritingRequest:
    """Test BatchCopywritingRequest schema"""
    
    def test_valid_batch_request(self):
        """Test valid batch request creation"""
        request1 = CopywritingRequest(
            topic="Topic 1",
            target_audience="Audience 1"
        )
        
        request2 = CopywritingRequest(
            topic="Topic 2",
            target_audience="Audience 2"
        )
        
        batch_request = BatchCopywritingRequest(
            requests=[request1, request2]
        )
        
        assert len(batch_request.requests) == 2
        assert isinstance(batch_request.batch_id, type(uuid4()))
    
    def test_invalid_batch_size(self):
        """Test invalid batch size validation"""
        requests = [
            CopywritingRequest(topic=f"Topic {i}", target_audience=f"Audience {i}")
            for i in range(51)  # Too many requests
        ]
        
        with pytest.raises(ValueError):
            BatchCopywritingRequest(requests=requests)


class TestHealthCheckResponse:
    """Test HealthCheckResponse schema"""
    
    def test_valid_health_check(self):
        """Test valid health check response"""
        health = HealthCheckResponse(
            status="healthy",
            uptime_seconds=3600.0,
            dependencies={"database": "healthy", "redis": "healthy"}
        )
        
        assert health.status == "healthy"
        assert health.uptime_seconds == 3600.0
        assert health.dependencies["database"] == "healthy"
        assert health.dependencies["redis"] == "healthy"
        assert health.version == "2.0.0"
        assert isinstance(health.timestamp, datetime)


class TestErrorResponse:
    """Test ErrorResponse schema"""
    
    def test_valid_error_response(self):
        """Test valid error response creation"""
        error = ErrorResponse(
            error_code="TEST_ERROR",
            error_message="Test error message",
            error_details={"key": "value"},
            request_id=uuid4()
        )
        
        assert error.error_code == "TEST_ERROR"
        assert error.error_message == "Test error message"
        assert error.error_details["key"] == "value"
        assert isinstance(error.request_id, type(uuid4()))
        assert isinstance(error.timestamp, datetime)
    
    def test_minimal_error_response(self):
        """Test minimal error response with only required fields"""
        error = ErrorResponse(
            error_code="MINIMAL_ERROR",
            error_message="Minimal error message"
        )
        
        assert error.error_code == "MINIMAL_ERROR"
        assert error.error_message == "Minimal error message"
        assert error.error_details is None
        assert error.request_id is None






























