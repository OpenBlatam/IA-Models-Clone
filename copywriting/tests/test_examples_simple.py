"""
Simple example test cases demonstrating best practices for copywriting service testing.
"""
import pytest
import time
from typing import Dict, Any, List
from unittest.mock import Mock, patch
import asyncio

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_utils import TestDataFactory, MockAIService, TestAssertions
from models import (
    CopywritingInput,
    CopywritingOutput,
    Feedback,
    SectionFeedback
)


class TestExamples:
    """Example test cases showing best practices."""
    
    def test_basic_request_creation_example(self):
        """Example: Creating a basic copywriting request."""
        # Using the factory for consistent test data
        request = TestDataFactory.create_copywriting_input(
            product_description="Smartphone de última generación",
            target_platform="instagram",
            tone="inspirational",
            creativity_level="innovative"
        )
        
        # Basic assertions
        assert request.product_description == "Smartphone de última generación"
        assert request.target_platform == "instagram"
        assert request.tone == "inspirational"
        assert request.creativity_level == "innovative"
        assert request.language == "es"  # Default value
    
    def test_batch_request_example(self):
        """Example: Creating and validating batch requests."""
        # Create multiple requests
        requests = TestDataFactory.create_batch_inputs(3)
        
        # Validate batch structure
        assert len(requests) == 3
        assert requests[0].product_description == "Test product 1"
        assert requests[1].product_description == "Test product 2"
        assert requests[2].product_description == "Test product 3"
    
    def test_feedback_creation_example(self):
        """Example: Creating and validating feedback."""
        # Create feedback
        feedback = TestDataFactory.create_feedback(
            feedback_type="human",
            score=0.9,
            comments="Excelente copy, muy creativo"
        )
        
        # Validate feedback
        assert feedback.type == "human"
        assert feedback.score == 0.9
        assert feedback.comments == "Excelente copy, muy creativo"
        assert feedback.user_id == "test_user_123"  # Default from factory
    
    def test_section_feedback_example(self):
        """Example: Creating section-specific feedback."""
        # Create base feedback
        base_feedback = TestDataFactory.create_feedback(
            feedback_type="model",
            score=0.8,
            comments="Good headline but could be more engaging"
        )
        
        # Create section feedback
        section_feedback = TestDataFactory.create_section_feedback(
            section="headline",
            feedback=base_feedback,
            suggestions=["Add emotional appeal", "Use action words"]
        )
        
        # Validate section feedback
        assert section_feedback.section == "headline"
        assert section_feedback.feedback.type == "model"
        assert section_feedback.feedback.score == 0.8
        assert len(section_feedback.suggestions) == 2
        assert "Add emotional appeal" in section_feedback.suggestions
    
    def test_model_validation_example(self):
        """Example: Testing model validation."""
        # Test valid input
        valid_request = TestDataFactory.create_copywriting_input(
            product_description="Valid product",
            target_platform="instagram",
            tone="inspirational",
            use_case="product_launch",
            content_type="social_post"
        )
        
        # Should not raise any validation errors
        assert valid_request.product_description == "Valid product"
        assert valid_request.target_platform == "instagram"
        
        # Test invalid input (should raise validation error)
        with pytest.raises(Exception):  # Pydantic validation error
            TestDataFactory.create_copywriting_input(
                product_description="",  # Empty description should fail
                target_platform="invalid_platform",  # Invalid platform
                tone="invalid_tone",  # Invalid tone
                use_case="invalid_use_case",  # Invalid use case
                content_type="invalid_type"  # Invalid content type
            )
    
    def test_serialization_example(self):
        """Example: Testing model serialization and deserialization."""
        # Create a request
        original_request = TestDataFactory.create_copywriting_input(
            product_description="Test product for serialization",
            target_platform="facebook",
            tone="professional"
        )
        
        # Serialize to JSON
        json_str = original_request.model_dump_json()
        assert isinstance(json_str, str)
        assert "Test product for serialization" in json_str
        
        # Deserialize from JSON
        parsed_request = CopywritingInput.model_validate_json(json_str)
        assert parsed_request.product_description == original_request.product_description
        assert parsed_request.target_platform == original_request.target_platform
        assert parsed_request.tone == original_request.tone
    
    def test_computed_fields_example(self):
        """Example: Testing computed fields."""
        # Create a request with tracking ID
        request = TestDataFactory.create_copywriting_input(
            product_description="Product with tracking",
            target_platform="twitter"
        )
        
        # Check computed fields
        assert hasattr(request, 'tracking_id')
        assert request.tracking_id is not None
        assert len(request.tracking_id) > 0
        
        # Check other computed fields if they exist
        if hasattr(request, 'created_at'):
            assert request.created_at is not None
    
    def test_enum_values_example(self):
        """Example: Testing enum values and validation."""
        # Test valid enum values
        valid_request = TestDataFactory.create_copywriting_input(
            product_description="Test product",
            target_platform="instagram",  # Valid platform
            tone="inspirational",  # Valid tone
            creativity_level="balanced"  # Valid creativity level
        )
        
        assert valid_request.target_platform == "instagram"
        assert valid_request.tone == "inspirational"
        assert valid_request.creativity_level == "balanced"
        
        # Test that invalid enum values raise validation errors
        with pytest.raises(Exception):
            TestDataFactory.create_copywriting_input(
                product_description="Test product",
                target_platform="invalid_platform",  # Invalid platform
                tone="invalid_tone",  # Invalid tone
                creativity_level="invalid_level"  # Invalid creativity level
            )
    
    def test_optional_fields_example(self):
        """Example: Testing optional fields and defaults."""
        # Create request with minimal required fields
        minimal_request = TestDataFactory.create_copywriting_input(
            product_description="Minimal product",
            target_platform="instagram",
            tone="casual"
        )
        
        # Check that optional fields have sensible defaults
        assert minimal_request.language == "es"  # Default language
        assert minimal_request.key_points is not None
        assert minimal_request.instructions is not None
        assert minimal_request.restrictions is not None
        
        # Create request with all optional fields specified
        full_request = TestDataFactory.create_copywriting_input(
            product_description="Full product",
            target_platform="facebook",
            tone="professional",
            key_points=["Quality", "Innovation", "Value"],
            instructions="Create engaging content",
            restrictions=["No price mentions"],
            language="en"
        )
        
        # Check that all fields are set correctly
        assert full_request.key_points == ["Quality", "Innovation", "Value"]
        assert full_request.instructions == "Create engaging content"
        assert full_request.restrictions == ["No price mentions"]
        assert full_request.language == "en"
    
    def test_data_factory_example(self):
        """Example: Using the TestDataFactory for different scenarios."""
        # Create a request for social media
        social_request = TestDataFactory.create_copywriting_input(
            product_description="Social media product",
            target_platform="instagram",
            tone="playful",
            content_type="social_post"
        )
        
        # Create a request for email marketing
        email_request = TestDataFactory.create_copywriting_input(
            product_description="Email marketing product",
            target_platform="email",
            tone="professional",
            content_type="email_subject"
        )
        
        # Create a request for advertising
        ad_request = TestDataFactory.create_copywriting_input(
            product_description="Advertising product",
            target_platform="facebook",
            tone="persuasive",
            content_type="ad_copy"
        )
        
        # Validate all requests
        assert social_request.content_type == "social_post"
        assert email_request.content_type == "email_subject"
        assert ad_request.content_type == "ad_copy"
        
        # All should have valid tracking IDs
        assert social_request.tracking_id is not None
        assert email_request.tracking_id is not None
        assert ad_request.tracking_id is not None
    
    def test_assertions_example(self):
        """Example: Using custom test assertions."""
        # Create test data
        request = TestDataFactory.create_copywriting_input()
        feedback = TestDataFactory.create_feedback()
        
        # Use custom assertions
        TestAssertions.assert_valid_copywriting_input(request)
        TestAssertions.assert_valid_feedback(feedback)
        
        # Test performance assertion
        execution_time = 0.1  # Simulated execution time
        TestAssertions.assert_performance_threshold(execution_time, 1.0)
    
    def test_mock_service_example(self):
        """Example: Using mock services for testing."""
        # Create mock AI service
        mock_ai = MockAIService(
            delay=0.05,
            should_fail=False,
            response_data={
                "variants": [{"headline": "Mock Headline", "primary_text": "Mock Content"}],
                "model_used": "gpt-3.5-turbo",
                "generation_time": 0.05,
                "extra_metadata": {"tokens_used": 50}
            }
        )
        
        # Test mock service
        request = TestDataFactory.create_copywriting_input()
        
        # Simulate async call
        async def test_mock_call():
            result = await mock_ai.mock_call(request, "gpt-3.5-turbo")
            return result
        
        # Run the test
        result = asyncio.run(test_mock_call())
        
        # Validate mock response
        assert result["model_used"] == "gpt-3.5-turbo"
        assert result["generation_time"] == 0.05
        assert len(result["variants"]) == 1
        assert "Mock Headline" in result["variants"][0]["headline"]
        assert mock_ai.call_count == 1
    
    def test_error_handling_example(self):
        """Example: Testing error handling scenarios."""
        # Test with mock service that fails
        failing_mock = MockAIService(should_fail=True)
        
        async def test_failing_call():
            with pytest.raises(Exception, match="Mock AI service error"):
                await failing_mock.mock_call(
                    TestDataFactory.create_copywriting_input(),
                    "gpt-3.5-turbo"
                )
        
        # Run the test
        asyncio.run(test_failing_call())
    
    def test_performance_example(self):
        """Example: Testing performance characteristics."""
        import time
        
        # Test model creation performance
        start_time = time.time()
        
        # Create 100 models
        for i in range(100):
            TestDataFactory.create_copywriting_input(
                product_description=f"Product {i}",
                target_platform="instagram"
            )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should create 100 models in less than 1 second
        TestAssertions.assert_performance_threshold(execution_time, 1.0)
        print(f"Created 100 models in {execution_time:.3f}s")
    
    def test_batch_processing_example(self):
        """Example: Testing batch processing."""
        # Create batch of requests
        batch_requests = TestDataFactory.create_batch_inputs(5)
        
        # Process batch
        results = []
        for request in batch_requests:
            # Simulate processing
            result = {
                "input_id": request.tracking_id,
                "processed_at": time.time(),
                "status": "success"
            }
            results.append(result)
        
        # Validate batch processing
        assert len(results) == 5
        assert all(result["status"] == "success" for result in results)
        assert all(result["input_id"] is not None for result in results)
