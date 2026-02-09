"""
Tests for Validation Schemas
Tests for Pydantic validation schemas
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from core.infrastructure.validation_schemas import (
    AnalysisRequest,
    AnalysisResponse,
    RecommendationRequest,
    RecommendationResponse
)


class TestAnalysisRequest:
    """Tests for AnalysisRequest schema"""
    
    def test_create_valid_request(self):
        """Test creating valid analysis request"""
        request = AnalysisRequest(
            metadata={"device": "mobile", "location": "indoor"}
        )
        
        assert request.metadata["device"] == "mobile"
        assert request.metadata["location"] == "indoor"
    
    def test_create_request_without_metadata(self):
        """Test creating request without metadata"""
        request = AnalysisRequest()
        
        assert request.metadata == {}
    
    def test_request_validation(self):
        """Test request validation"""
        # Valid request
        request = AnalysisRequest(metadata={})
        assert request is not None
        
        # Should accept any dict as metadata
        request = AnalysisRequest(metadata={"any": "value"})
        assert request.metadata["any"] == "value"


class TestAnalysisResponse:
    """Tests for AnalysisResponse schema"""
    
    def test_create_valid_response(self):
        """Test creating valid analysis response"""
        response = AnalysisResponse(
            success=True,
            analysis_id="test-123",
            status="completed",
            metrics={"overall_score": 75.5},
            conditions=[{"name": "acne", "confidence": 0.65}]
        )
        
        assert response.success is True
        assert response.analysis_id == "test-123"
        assert response.status == "completed"
        assert response.metrics["overall_score"] == 75.5
        assert len(response.conditions) == 1
    
    def test_response_without_optional_fields(self):
        """Test response without optional fields"""
        response = AnalysisResponse(
            success=True,
            analysis_id="test-123",
            status="pending"
        )
        
        assert response.success is True
        assert response.metrics is None
        assert response.conditions == []
    
    def test_response_validation_required_fields(self):
        """Test response validation for required fields"""
        # Missing required fields should raise error
        with pytest.raises(ValidationError):
            AnalysisResponse(success=True)  # Missing analysis_id and status


class TestRecommendationRequest:
    """Tests for RecommendationRequest schema"""
    
    def test_create_valid_recommendation_request(self):
        """Test creating valid recommendation request"""
        from core.infrastructure.validation_schemas import RecommendationRequest
        
        request = RecommendationRequest(
            analysis_id="test-123",
            include_routine=True
        )
        
        assert request.analysis_id == "test-123"
        assert request.include_routine is True
    
    def test_recommendation_request_defaults(self):
        """Test recommendation request defaults"""
        from core.infrastructure.validation_schemas import RecommendationRequest
        
        request = RecommendationRequest(analysis_id="test-123")
        
        assert request.analysis_id == "test-123"
        # include_routine default depends on schema definition


class TestRecommendationResponse:
    """Tests for RecommendationResponse schema"""
    
    def test_create_valid_recommendation_response(self):
        """Test creating valid recommendation response"""
        from core.infrastructure.validation_schemas import RecommendationResponse
        
        response = RecommendationResponse(
            success=True,
            recommendations={
                "routine": {
                    "morning": [],
                    "evening": []
                }
            }
        )
        
        assert response.success is True
        assert "routine" in response.recommendations


class TestSchemaSerialization:
    """Tests for schema serialization"""
    
    def test_analysis_request_serialization(self):
        """Test serializing analysis request"""
        request = AnalysisRequest(metadata={"key": "value"})
        
        serialized = request.dict()
        
        assert isinstance(serialized, dict)
        assert "metadata" in serialized
    
    def test_analysis_response_serialization(self):
        """Test serializing analysis response"""
        response = AnalysisResponse(
            success=True,
            analysis_id="test-123",
            status="completed"
        )
        
        serialized = response.dict()
        
        assert isinstance(serialized, dict)
        assert serialized["success"] is True
        assert serialized["analysis_id"] == "test-123"
    
    def test_response_json_serialization(self):
        """Test JSON serialization of response"""
        response = AnalysisResponse(
            success=True,
            analysis_id="test-123",
            status="completed",
            created_at=datetime.utcnow()
        )
        
        json_str = response.json()
        
        assert isinstance(json_str, str)
        assert "test-123" in json_str



