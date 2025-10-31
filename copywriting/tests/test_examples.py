"""
Example test cases demonstrating best practices for copywriting service testing.
"""
import pytest
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
            tone="modern",
            creativity_level="innovative"
        )
        
        # Basic assertions
        assert request.product_description == "Smartphone de última generación"
        assert request.target_platform == "instagram"
        assert request.tone == "modern"
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
    
    @pytest.mark.asyncio
    async def test_async_service_call_example(self):
        """Example: Testing async service calls with mocking."""
        from service import CopywritingService
        
        service = CopywritingService()
        request = TestDataFactory.create_copywriting_input()
        
        # Mock the AI service
        mock_ai = MockAIService(delay=0.1)
        
        with patch.object(service, '_call_ai_model', mock_ai.mock_call):
            result = await service.generate_copywriting(request, "gpt-3.5-turbo")
        
        # Validate response
        TestAssertions.assert_valid_copywriting_response(result)
        assert result.model_used == "gpt-3.5-turbo"
        assert len(result.variants) > 0
    
    def test_error_handling_example(self):
        """Example: Testing error handling scenarios."""
        # Test invalid creativity level
        with pytest.raises(ValueError, match="Creativity level must be between 0.0 and 1.0"):
            TestDataFactory.create_sample_request(creativity_level=1.5)
        
        # Test invalid language
        with pytest.raises(ValueError, match="Invalid language code"):
            TestDataFactory.create_sample_request(language="invalid_lang")
    
    def test_custom_assertions_example(self):
        """Example: Using custom assertions for domain validation."""
        # Create a response
        response = TestDataFactory.create_sample_response(
            variants=[
                {
                    "headline": "¡Descubre la Innovación!",
                    "primary_text": "Producto revolucionario para tu vida",
                    "call_to_action": "Compra ahora",
                    "hashtags": ["#innovación", "#producto"]
                }
            ]
        )
        
        # Use custom assertions
        TestAssertions.assert_valid_copywriting_response(response)
        
        # Additional specific validations
        assert response.variants[0]["headline"].startswith("¡")
        assert len(response.variants[0]["hashtags"]) == 2
        assert "#innovación" in response.variants[0]["hashtags"]
    
    def test_mock_ai_service_example(self):
        """Example: Using mock AI service for testing."""
        # Create mock service with specific behavior
        mock_ai = MockAIService(delay=0.05, should_fail=False)
        
        # Test successful call
        request = TestDataFactory.create_sample_request()
        result = asyncio.run(mock_ai.mock_call(request, "gpt-3.5-turbo"))
        
        assert "variants" in result
        assert result["model_used"] == "gpt-3.5-turbo"
        assert result["generation_time"] == 0.05
        
        # Test failing call
        failing_mock = MockAIService(should_fail=True)
        
        with pytest.raises(Exception, match="Mock AI service error"):
            asyncio.run(failing_mock.mock_call(request, "gpt-3.5-turbo"))
    
    def test_data_factory_patterns_example(self):
        """Example: Using data factory patterns for test data."""
        # Create variations of test data
        variations = [
            TestDataFactory.create_sample_request(tone="inspirational"),
            TestDataFactory.create_sample_request(tone="informative"),
            TestDataFactory.create_sample_request(tone="playful"),
        ]
        
        # Validate all variations
        for request in variations:
            assert request.tone in ["inspirational", "informative", "playful"]
            assert request.product_description is not None
            assert request.target_platform is not None
    
    def test_edge_cases_example(self):
        """Example: Testing edge cases and boundary conditions."""
        # Test minimum valid request
        minimal_request = TestDataFactory.create_sample_request(
            product_description="Test",
            target_platform="Instagram",
            tone="informative",
            language="es"
        )
        assert minimal_request.creativity_level == 0.5  # Default value
        
        # Test maximum creativity level
        max_creativity_request = TestDataFactory.create_sample_request(
            creativity_level=1.0
        )
        assert max_creativity_request.creativity_level == 1.0
        
        # Test minimum creativity level
        min_creativity_request = TestDataFactory.create_sample_request(
            creativity_level=0.0
        )
        assert min_creativity_request.creativity_level == 0.0
    
    def test_validation_examples(self):
        """Example: Testing various validation scenarios."""
        # Valid request
        valid_request = TestDataFactory.create_sample_request()
        assert valid_request.product_description is not None
        assert valid_request.target_platform is not None
        assert valid_request.tone is not None
        assert valid_request.language is not None
        
        # Test with optional fields
        request_with_optionals = TestDataFactory.create_sample_request(
            target_audience="Jóvenes profesionales",
            key_points=["Calidad", "Precio", "Diseño"],
            instructions="Enfatiza la innovación",
            restrictions=["No mencionar precio"]
        )
        
        assert request_with_optionals.target_audience == "Jóvenes profesionales"
        assert len(request_with_optionals.key_points) == 3
        assert request_with_optionals.instructions == "Enfatiza la innovación"
        assert len(request_with_optionals.restrictions) == 1


class TestPerformanceExamples:
    """Example performance testing patterns."""
    
    def test_response_time_measurement_example(self):
        """Example: Measuring response times in tests."""
        import time
        
        # Simulate a service call
        start_time = time.time()
        
        # Mock service call
        mock_ai = MockAIService(delay=0.1)
        request = TestDataFactory.create_sample_request()
        result = asyncio.run(mock_ai.mock_call(request, "gpt-3.5-turbo"))
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Assert reasonable response time
        assert response_time < 1.0  # Should be fast
        assert result is not None
    
    def test_memory_usage_example(self):
        """Example: Testing memory usage patterns."""
        import psutil
        import os
        
        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple requests (simulate load)
        requests = [
            TestDataFactory.create_sample_request(
                product_description=f"Producto {i}"
            )
            for i in range(100)
        ]
        
        # Get final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Assert reasonable memory usage
        assert memory_increase < 50.0  # Should not use more than 50MB
        assert len(requests) == 100


class TestIntegrationExamples:
    """Example integration testing patterns."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow_example(self):
        """Example: Testing complete workflow from request to response."""
        from agents.backend.onyx.server.features.copywriting.service import CopywritingService
        
        # Create service and request
        service = CopywritingService()
        request = TestDataFactory.create_sample_request(
            product_description="Laptop gaming de alta gama",
            target_platform="Instagram",
            tone="energético"
        )
        
        # Mock AI service
        mock_ai = MockAIService(delay=0.1)
        
        with patch.object(service, '_call_ai_model', mock_ai.mock_call):
            # Generate copywriting
            result = await service.generate_copywriting(request, "gpt-3.5-turbo")
            
            # Validate complete workflow
            TestAssertions.assert_valid_copywriting_response(result)
            assert result.model_used == "gpt-3.5-turbo"
            assert len(result.variants) > 0
            
            # Validate content quality
            for variant in result.variants:
                assert len(variant["headline"]) > 0
                assert len(variant["primary_text"]) > 0
                assert "gaming" in variant["headline"].lower() or "gaming" in variant["primary_text"].lower()
    
    def test_batch_processing_example(self):
        """Example: Testing batch processing workflows."""
        # Create batch request
        batch_request = TestDataFactory.create_batch_request(num_requests=5)
        
        # Validate batch structure
        assert len(batch_request.requests) == 5
        
        # Validate each request in batch
        for i, request in enumerate(batch_request.requests):
            assert request.product_description == f"Producto {i}"
            assert request.target_platform in ["Instagram", "Facebook", "Twitter"]
            assert request.tone in ["inspirational", "informative", "playful"]


class TestMockingExamples:
    """Example mocking patterns for testing."""
    
    def test_mock_external_service_example(self):
        """Example: Mocking external services."""
        # Mock external API call
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "variants": [{"headline": "Test", "primary_text": "Content"}]
            }
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            
            # Test code that uses external service
            # (This would be your actual service call)
            response = mock_post.return_value
            
            assert response.status_code == 200
            assert "variants" in response.json()
    
    def test_mock_async_service_example(self):
        """Example: Mocking async services."""
        async def mock_async_call():
            return {"result": "mocked"}
        
        # Use in async test
        result = asyncio.run(mock_async_call())
        assert result["result"] == "mocked"
    
    def test_mock_with_side_effects_example(self):
        """Example: Mocking with side effects."""
        mock_service = Mock()
        mock_service.side_effect = [
            {"status": "success", "data": "first_call"},
            {"status": "success", "data": "second_call"},
            Exception("Service error")
        ]
        
        # First two calls succeed
        assert mock_service()["status"] == "success"
        assert mock_service()["data"] == "second_call"
        
        # Third call fails
        with pytest.raises(Exception, match="Service error"):
            mock_service()


# Pytest markers for example tests
@pytest.mark.example
class TestExampleMarkers:
    """Example tests with custom markers."""
    
    def test_example_marker(self):
        """This test is marked as an example."""
        assert True
    
    @pytest.mark.slow
    def test_slow_example(self):
        """This test is marked as slow."""
        import time
        time.sleep(0.1)  # Simulate slow operation
        assert True
    
    @pytest.mark.integration
    def test_integration_example(self):
        """This test is marked as integration."""
        assert True

