"""
Refactored unit tests for copywriting service using base classes.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List
import asyncio

from tests.base import BaseTestClass, MockAIService, TestAssertions, TestConfig, PerformanceMixin
from agents.backend.onyx.server.features.copywriting.service import CopywritingService
from agents.backend.onyx.server.features.copywriting.models import (
    CopywritingRequest,
    CopywritingResponse,
    BatchCopywritingRequest
)


class TestCopywritingServiceRefactored(BaseTestClass, PerformanceMixin):
    """Refactored test cases for CopywritingService."""
    
    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        return CopywritingService()
    
    @pytest.mark.asyncio
    async def test_generate_copywriting_success(self, service):
        """Test successful copywriting generation."""
        request = self.create_request()
        mock_ai = MockAIService(delay=0.1)
        
        with patch.object(service, '_call_ai_model', mock_ai.mock_call):
            result = await service.generate_copywriting(request, "gpt-3.5-turbo")
            
            TestAssertions.assert_valid_copywriting_response(result)
            assert result.model_used == "gpt-3.5-turbo"
            assert mock_ai.call_count == 1
    
    @pytest.mark.asyncio
    async def test_generate_copywriting_invalid_model(self, service):
        """Test copywriting generation with invalid model."""
        request = self.create_request()
        
        with pytest.raises(ValueError, match="Modelo no soportado"):
            await service.generate_copywriting(request, "invalid_model")
    
    @pytest.mark.asyncio
    async def test_generate_copywriting_ai_error(self, service):
        """Test copywriting generation with AI model error."""
        request = self.create_request()
        mock_ai = MockAIService(should_fail=True)
        
        with patch.object(service, '_call_ai_model', mock_ai.mock_call):
            with pytest.raises(Exception, match="Mock AI service error"):
                await service.generate_copywriting(request, "gpt-3.5-turbo")
    
    @pytest.mark.asyncio
    async def test_batch_generate_copywriting_success(self, service):
        """Test successful batch copywriting generation."""
        batch_request = self.create_batch_request(count=3)
        
        with patch.object(service, 'generate_copywriting') as mock_generate:
            mock_generate.side_effect = [
                self.create_response(extra_metadata={"batch_index": i})
                for i in range(3)
            ]
            
            result = await service.batch_generate_copywriting(batch_request)
            
            TestAssertions.assert_valid_batch_response(result, 3)
            assert mock_generate.call_count == 3
    
    @pytest.mark.asyncio
    async def test_batch_generate_copywriting_too_large(self, service):
        """Test batch generation with too many requests."""
        batch_request = self.create_batch_request(count=TestConfig.MAX_BATCH_SIZE + 1)
        
        with pytest.raises(ValueError, match="batch máximo"):
            await service.batch_generate_copywriting(batch_request)
    
    def test_get_available_models(self, service):
        """Test getting available models."""
        models = service.get_available_models()
        
        assert isinstance(models, list)
        assert "gpt-3.5-turbo" in models
        assert "gpt-4" in models
        assert len(models) > 0
    
    def test_validate_model_success(self, service):
        """Test model validation with valid model."""
        assert service.validate_model("gpt-3.5-turbo") is True
        assert service.validate_model("gpt-4") is True
    
    def test_validate_model_failure(self, service):
        """Test model validation with invalid model."""
        assert service.validate_model("invalid_model") is False
        assert service.validate_model("") is False
        assert service.validate_model(None) is False
    
    @pytest.mark.asyncio
    async def test_call_ai_model_success(self, service):
        """Test successful AI model call."""
        request = self.create_request()
        
        with patch('openai.ChatCompletion.create') as mock_create:
            mock_create.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": '{"variants": [{"headline": "Test", "primary_text": "Content"}]}'
                        }
                    }
                ],
                "usage": {"total_tokens": 100}
            }
            
            result = await service._call_ai_model(request, "gpt-3.5-turbo")
            
            assert "variants" in result
            assert len(result["variants"]) == 1
            assert result["variants"][0]["headline"] == "Test"
    
    @pytest.mark.asyncio
    async def test_call_ai_model_invalid_response(self, service):
        """Test AI model call with invalid response format."""
        request = self.create_request()
        
        with patch('openai.ChatCompletion.create') as mock_create:
            mock_create.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": "Invalid JSON response"
                        }
                    }
                ]
            }
            
            with pytest.raises(ValueError, match="Respuesta inválida"):
                await service._call_ai_model(request, "gpt-3.5-turbo")
    
    @pytest.mark.asyncio
    async def test_call_ai_model_api_error(self, service):
        """Test AI model call with API error."""
        request = self.create_request()
        
        with patch('openai.ChatCompletion.create') as mock_create:
            mock_create.side_effect = Exception("API Error")
            
            with pytest.raises(Exception, match="API Error"):
                await service._call_ai_model(request, "gpt-3.5-turbo")
    
    def test_format_prompt(self, service):
        """Test prompt formatting."""
        request = self.create_request()
        prompt = service._format_prompt(request)
        
        assert isinstance(prompt, str)
        assert request.product_description in prompt
        assert request.target_platform in prompt
        assert request.tone in prompt
    
    def test_format_prompt_minimal(self, service):
        """Test prompt formatting with minimal data."""
        request = self.create_request(
            target_audience=None,
            key_points=None,
            instructions=None,
            restrictions=None
        )
        
        prompt = service._format_prompt(request)
        
        assert isinstance(prompt, str)
        assert request.product_description in prompt
        assert request.target_platform in prompt
        assert request.tone in prompt
    
    def test_parse_ai_response_success(self, service):
        """Test successful AI response parsing."""
        response_text = '{"variants": [{"headline": "Test", "primary_text": "Content"}]}'
        
        result = service._parse_ai_response(response_text)
        
        assert "variants" in result
        assert len(result["variants"]) == 1
        assert result["variants"][0]["headline"] == "Test"
    
    def test_parse_ai_response_invalid_json(self, service):
        """Test AI response parsing with invalid JSON."""
        response_text = "Invalid JSON"
        
        with pytest.raises(ValueError, match="JSON inválido"):
            service._parse_ai_response(response_text)
    
    def test_parse_ai_response_missing_variants(self, service):
        """Test AI response parsing with missing variants."""
        response_text = '{"other_field": "value"}'
        
        with pytest.raises(ValueError, match="variants"):
            service._parse_ai_response(response_text)


class TestServicePerformanceRefactored(BaseTestClass, PerformanceMixin):
    """Refactored performance tests for CopywritingService."""
    
    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        return CopywritingService()
    
    @pytest.mark.asyncio
    async def test_single_request_performance(self, service):
        """Test single request performance."""
        request = self.create_request()
        mock_ai = MockAIService(delay=0.1)
        
        with patch.object(service, '_call_ai_model', mock_ai.mock_call):
            result, execution_time = await self.measure_async_execution_time(
                service.generate_copywriting(request, "gpt-3.5-turbo")
            )
            
            self.assert_performance_threshold(execution_time, TestConfig.SINGLE_REQUEST_MAX_TIME)
            TestAssertions.assert_valid_copywriting_response(result)
    
    @pytest.mark.asyncio
    async def test_batch_request_performance(self, service):
        """Test batch request performance."""
        batch_request = self.create_batch_request(count=5)
        
        with patch.object(service, 'generate_copywriting') as mock_generate:
            mock_generate.return_value = self.create_response()
            
            result, execution_time = await self.measure_async_execution_time(
                service.batch_generate_copywriting(batch_request)
            )
            
            self.assert_performance_threshold(execution_time, TestConfig.BATCH_REQUEST_MAX_TIME)
            TestAssertions.assert_valid_batch_response(result, 5)
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_performance(self, service):
        """Test concurrent request performance."""
        requests = [self.create_request() for _ in range(5)]
        mock_ai = MockAIService(delay=0.1)
        
        async def process_request(request):
            with patch.object(service, '_call_ai_model', mock_ai.mock_call):
                return await service.generate_copywriting(request, "gpt-3.5-turbo")
        
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*[process_request(req) for req in requests])
        end_time = asyncio.get_event_loop().time()
        
        execution_time = end_time - start_time
        self.assert_performance_threshold(execution_time, TestConfig.CONCURRENT_REQUEST_MAX_TIME)
        
        assert len(results) == 5
        for result in results:
            TestAssertions.assert_valid_copywriting_response(result)


class TestServiceErrorHandlingRefactored(BaseTestClass):
    """Refactored error handling tests for CopywritingService."""
    
    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        return CopywritingService()
    
    @pytest.mark.asyncio
    async def test_network_error_handling(self, service):
        """Test handling of network errors."""
        request = self.create_request()
        
        with patch('openai.ChatCompletion.create') as mock_create:
            mock_create.side_effect = ConnectionError("Network error")
            
            with pytest.raises(ConnectionError, match="Network error"):
                await service._call_ai_model(request, "gpt-3.5-turbo")
    
    @pytest.mark.asyncio
    async def test_timeout_error_handling(self, service):
        """Test handling of timeout errors."""
        request = self.create_request()
        
        with patch('openai.ChatCompletion.create') as mock_create:
            mock_create.side_effect = TimeoutError("Request timeout")
            
            with pytest.raises(TimeoutError, match="Request timeout"):
                await service._call_ai_model(request, "gpt-3.5-turbo")
    
    @pytest.mark.asyncio
    async def test_rate_limit_error_handling(self, service):
        """Test handling of rate limit errors."""
        request = self.create_request()
        
        with patch('openai.ChatCompletion.create') as mock_create:
            mock_create.side_effect = Exception("Rate limit exceeded")
            
            with pytest.raises(Exception, match="Rate limit exceeded"):
                await service._call_ai_model(request, "gpt-3.5-turbo")
    
    def test_invalid_input_handling(self, service):
        """Test handling of invalid input."""
        # Test with None input
        with pytest.raises((TypeError, ValueError)):
            service._format_prompt(None)
        
        # Test with invalid request type
        with pytest.raises(AttributeError):
            service._format_prompt("invalid_request")


class TestServiceIntegrationRefactored(BaseTestClass):
    """Refactored integration tests for CopywritingService."""
    
    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        return CopywritingService()
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, service):
        """Test complete end-to-end workflow."""
        request = self.create_request()
        mock_ai = MockAIService(delay=0.1)
        
        with patch.object(service, '_call_ai_model', mock_ai.mock_call):
            # Generate copywriting
            response = await service.generate_copywriting(request, "gpt-3.5-turbo")
            
            # Validate response
            TestAssertions.assert_valid_copywriting_response(response)
            assert response.model_used == "gpt-3.5-turbo"
            
            # Validate content quality
            for variant in response.variants:
                assert len(variant["headline"]) > 0
                assert len(variant["primary_text"]) > 0
    
    @pytest.mark.asyncio
    async def test_batch_workflow(self, service):
        """Test complete batch workflow."""
        batch_request = self.create_batch_request(count=3)
        
        with patch.object(service, 'generate_copywriting') as mock_generate:
            mock_generate.side_effect = [
                self.create_response(extra_metadata={"batch_index": i})
                for i in range(3)
            ]
            
            batch_response = await service.batch_generate_copywriting(batch_request)
            
            TestAssertions.assert_valid_batch_response(batch_response, 3)
            
            # Validate individual results
            for i, result in enumerate(batch_response.results):
                assert result.extra_metadata["batch_index"] == i
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, service):
        """Test error recovery workflow."""
        request = self.create_request()
        
        # Mock AI service with retry logic
        call_count = 0
        
        async def mock_ai_call_with_retry(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail first two calls
                raise Exception("AI service temporarily unavailable")
            return {
                "variants": [{"headline": "Recovery Success", "primary_text": "Content"}],
                "model_used": "gpt-3.5-turbo",
                "generation_time": 1.0,
                "extra_metadata": {}
            }
        
        with patch.object(service, '_call_ai_model', mock_ai_call_with_retry):
            # Should eventually succeed after retries
            response = await service.generate_copywriting(request, "gpt-3.5-turbo")
            
            assert response is not None
            assert response.variants[0]["headline"] == "Recovery Success"
            assert call_count == 3  # Should have retried twice


class TestServiceValidationRefactored(BaseTestClass):
    """Refactored validation tests for CopywritingService."""
    
    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        return CopywritingService()
    
    def test_model_validation_comprehensive(self, service):
        """Test comprehensive model validation."""
        valid_models = ["gpt-3.5-turbo", "gpt-4", "claude-3"]
        invalid_models = ["invalid_model", "", None, "gpt-5"]
        
        for model in valid_models:
            assert service.validate_model(model) is True
        
        for model in invalid_models:
            assert service.validate_model(model) is False
    
    def test_prompt_formatting_validation(self, service):
        """Test prompt formatting validation."""
        request = self.create_request()
        prompt = service._format_prompt(request)
        
        # Validate prompt contains all necessary information
        assert request.product_description in prompt
        assert request.target_platform in prompt
        assert request.tone in prompt
        assert request.language in prompt
        
        # Validate prompt structure
        assert isinstance(prompt, str)
        assert len(prompt) > 0
    
    def test_response_parsing_validation(self, service):
        """Test response parsing validation."""
        valid_responses = [
            '{"variants": [{"headline": "Test", "primary_text": "Content"}]}',
            '{"variants": [{"headline": "Test", "primary_text": "Content", "call_to_action": "Buy now"}]}'
        ]
        
        invalid_responses = [
            "Invalid JSON",
            '{"other_field": "value"}',
            '{"variants": []}',
            '{"variants": [{"headline": "Test"}]}'  # Missing primary_text
        ]
        
        for response in valid_responses:
            result = service._parse_ai_response(response)
            assert "variants" in result
            assert len(result["variants"]) > 0
        
        for response in invalid_responses:
            with pytest.raises(ValueError):
                service._parse_ai_response(response)
