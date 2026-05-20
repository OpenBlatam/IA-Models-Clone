"""
Refactored unit tests for copywriting models using base classes.
"""
import pytest
from typing import Dict, Any, List

from tests.base import BaseTestClass, TestAssertions, TestConfig
from agents.backend.onyx.server.features.copywriting.models import (
    CopywritingRequest,
    CopywritingResponse,
    BatchCopywritingRequest,
    FeedbackRequest,
    TaskStatus
)


class TestCopywritingRequestRefactored(BaseTestClass):
    """Refactored test cases for CopywritingRequest model."""
    
    def test_valid_request_creation(self):
        """Test creating a valid copywriting request."""
        request = self.create_request()
        
        assert request.product_description == "Zapatos deportivos de alta gama"
        assert request.target_platform == "Instagram"
        assert request.tone == "inspirational"
        assert request.creativity_level == 0.8
        assert request.language == "es"
    
    def test_request_with_minimal_data(self):
        """Test creating request with only required fields."""
        request = self.create_request(
            target_audience=None,
            key_points=None,
            instructions=None,
            restrictions=None
        )
        
        assert request.product_description == "Zapatos deportivos de alta gama"
        assert request.target_audience is None
        assert request.key_points is None
        assert request.creativity_level == 0.8  # default value
    
    def test_invalid_creativity_level(self):
        """Test validation of creativity level bounds."""
        with pytest.raises(ValueError):
            self.create_request(creativity_level=1.5)  # Invalid: > 1.0
        
        with pytest.raises(ValueError):
            self.create_request(creativity_level=-0.1)  # Invalid: < 0.0
    
    def test_invalid_language_code(self):
        """Test validation of language codes."""
        with pytest.raises(ValueError):
            self.create_request(language="invalid_lang")
    
    def test_request_serialization(self):
        """Test request serialization to dict."""
        request = self.create_request()
        request_dict = request.dict()
        
        assert isinstance(request_dict, dict)
        assert request_dict["product_description"] == request.product_description
        assert request_dict["target_platform"] == request.target_platform
        assert request_dict["tone"] == request.tone
    
    def test_request_validation_edge_cases(self):
        """Test edge cases in request validation."""
        # Test empty strings
        with pytest.raises(ValueError):
            self.create_request(product_description="")
        
        # Test very long strings
        long_description = "A" * (TestConfig.MAX_INPUT_LENGTH + 1)
        with pytest.raises(ValueError):
            self.create_request(product_description=long_description)
    
    @pytest.mark.parametrize("tone", ["inspirational", "informative", "playful", "professional"])
    def test_valid_tones(self, tone):
        """Test valid tone values."""
        request = self.create_request(tone=tone)
        assert request.tone == tone
    
    @pytest.mark.parametrize("platform", ["Instagram", "Facebook", "Twitter", "LinkedIn"])
    def test_valid_platforms(self, platform):
        """Test valid platform values."""
        request = self.create_request(target_platform=platform)
        assert request.target_platform == platform


class TestCopywritingResponseRefactored(BaseTestClass):
    """Refactored test cases for CopywritingResponse model."""
    
    def test_response_creation(self):
        """Test creating a valid copywriting response."""
        response = self.create_response()
        
        TestAssertions.assert_valid_copywriting_response(response)
        assert response.model_used == "gpt-3.5-turbo"
        assert response.generation_time == 2.5
    
    def test_response_with_multiple_variants(self):
        """Test response with multiple variants."""
        variants = [
            {"headline": "Headline 1", "primary_text": "Content 1"},
            {"headline": "Headline 2", "primary_text": "Content 2"}
        ]
        
        response = self.create_response(variants=variants)
        
        assert len(response.variants) == 2
        TestAssertions.assert_valid_copywriting_response(response)
    
    def test_response_serialization(self):
        """Test response serialization to dict."""
        response = self.create_response()
        response_dict = response.dict()
        
        assert isinstance(response_dict, dict)
        assert "variants" in response_dict
        assert "model_used" in response_dict
        assert "generation_time" in response_dict
    
    def test_response_validation(self):
        """Test response validation."""
        # Test empty variants
        with pytest.raises(ValueError):
            self.create_response(variants=[])
        
        # Test invalid generation time
        with pytest.raises(ValueError):
            self.create_response(generation_time=-1.0)


class TestBatchCopywritingRequestRefactored(BaseTestClass):
    """Refactored test cases for BatchCopywritingRequest model."""
    
    def test_batch_request_creation(self):
        """Test creating a valid batch request."""
        batch_request = self.create_batch_request(count=3)
        
        assert len(batch_request.requests) == 3
        assert all(isinstance(req, CopywritingRequest) for req in batch_request.requests)
    
    def test_batch_request_validation(self):
        """Test batch request size validation."""
        # Test maximum batch size
        batch_request = self.create_batch_request(count=TestConfig.MAX_BATCH_SIZE)
        assert len(batch_request.requests) == TestConfig.MAX_BATCH_SIZE
        
        # Test exceeding maximum batch size
        with pytest.raises(ValueError):
            self.create_batch_request(count=TestConfig.MAX_BATCH_SIZE + 1)
    
    def test_batch_request_serialization(self):
        """Test batch request serialization."""
        batch_request = self.create_batch_request(count=2)
        batch_dict = batch_request.dict()
        
        assert isinstance(batch_dict, dict)
        assert "requests" in batch_dict
        assert len(batch_dict["requests"]) == 2


class TestFeedbackRequestRefactored(BaseTestClass):
    """Refactored test cases for FeedbackRequest model."""
    
    def test_feedback_creation(self):
        """Test creating a valid feedback request."""
        feedback_request = self.create_feedback_request()
        
        assert feedback_request.variant_id == "variant_1"
        assert feedback_request.feedback["type"] == "human"
        assert feedback_request.feedback["score"] == 0.9
    
    def test_feedback_score_validation(self):
        """Test feedback score validation."""
        with pytest.raises(ValueError):
            self.create_feedback_request(feedback={"score": 1.5})  # Invalid: > 1.0
        
        with pytest.raises(ValueError):
            self.create_feedback_request(feedback={"score": -0.1})  # Invalid: < 0.0
    
    def test_feedback_serialization(self):
        """Test feedback request serialization."""
        feedback_request = self.create_feedback_request()
        feedback_dict = feedback_request.dict()
        
        assert isinstance(feedback_dict, dict)
        assert "variant_id" in feedback_dict
        assert "feedback" in feedback_dict


class TestTaskStatusRefactored(BaseTestClass):
    """Refactored test cases for TaskStatus model."""
    
    def test_task_status_creation(self):
        """Test creating task status responses."""
        # Success status
        success_status = TaskStatus(
            status="SUCCESS",
            result={"variants": []},
            error=None
        )
        
        assert success_status.status == "SUCCESS"
        assert success_status.result == {"variants": []}
        assert success_status.error is None
        
        # Failure status
        failure_status = TaskStatus(
            status="FAILURE",
            result=None,
            error="Test error message"
        )
        
        assert failure_status.status == "FAILURE"
        assert failure_status.result is None
        assert failure_status.error == "Test error message"
    
    def test_task_status_validation(self):
        """Test task status validation."""
        with pytest.raises(ValueError):
            TaskStatus(
                status="INVALID_STATUS",
                result=None,
                error=None
            )
    
    @pytest.mark.parametrize("status", ["PENDING", "SUCCESS", "FAILURE", "RETRY"])
    def test_valid_status_values(self, status):
        """Test valid status values."""
        task_status = TaskStatus(
            status=status,
            result=None,
            error=None
        )
        assert task_status.status == status


class TestModelIntegrationRefactored(BaseTestClass):
    """Refactored integration tests for models."""
    
    def test_request_to_response_flow(self):
        """Test complete flow from request to response."""
        request = self.create_request()
        response = self.create_response()
        
        # Validate that request and response are compatible
        assert isinstance(request, CopywritingRequest)
        assert isinstance(response, CopywritingResponse)
        
        # Test that response can be created from request data
        response_from_request = self.create_response(
            extra_metadata={"request_id": "test_123"}
        )
        assert response_from_request.extra_metadata["request_id"] == "test_123"
    
    def test_batch_processing_flow(self):
        """Test batch processing flow."""
        batch_request = self.create_batch_request(count=3)
        
        # Simulate batch processing
        batch_responses = [
            self.create_response(
                extra_metadata={"batch_index": i}
            )
            for i in range(len(batch_request.requests))
        ]
        
        assert len(batch_responses) == len(batch_request.requests)
        for i, response in enumerate(batch_responses):
            assert response.extra_metadata["batch_index"] == i
    
    def test_feedback_integration(self):
        """Test feedback integration with responses."""
        response = self.create_response()
        feedback = self.create_feedback_request(
            variant_id="test_variant",
            feedback={"score": 0.8, "comments": "Good response"}
        )
        
        # Validate feedback can be associated with response
        assert feedback.variant_id == "test_variant"
        assert feedback.feedback["score"] == 0.8
        assert "Good response" in feedback.feedback["comments"]


class TestModelValidationRefactored(BaseTestClass):
    """Refactored validation tests for models."""
    
    def test_comprehensive_request_validation(self):
        """Test comprehensive request validation."""
        # Test all required fields
        required_fields = ["product_description", "target_platform", "tone", "language"]
        
        for field in required_fields:
            with pytest.raises(ValueError):
                data = self.sample_request_data.copy()
                del data[field]
                CopywritingRequest(**data)
    
    def test_comprehensive_response_validation(self):
        """Test comprehensive response validation."""
        # Test required fields
        required_fields = ["variants", "model_used", "generation_time", "extra_metadata"]
        
        for field in required_fields:
            with pytest.raises(ValueError):
                data = self.sample_response_data.copy()
                del data[field]
                CopywritingResponse(**data)
    
    def test_edge_case_validation(self):
        """Test edge case validation."""
        # Test boundary values
        boundary_tests = [
            ("creativity_level", 0.0, True),
            ("creativity_level", 1.0, True),
            ("creativity_level", -0.1, False),
            ("creativity_level", 1.1, False),
        ]
        
        for field, value, should_pass in boundary_tests:
            if should_pass:
                request = self.create_request(**{field: value})
                assert getattr(request, field) == value
            else:
                with pytest.raises(ValueError):
                    self.create_request(**{field: value})
