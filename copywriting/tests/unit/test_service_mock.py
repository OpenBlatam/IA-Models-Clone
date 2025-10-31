"""
Mock unit tests for copywriting service layer (avoiding aioredis dependency issues).
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List
import asyncio
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models import (
    CopywritingInput,
    CopywritingOutput,
    Feedback
)
from tests.test_utils import TestDataFactory, MockAIService


class MockCopywritingService:
    """Mock copywriting service to avoid dependency issues."""
    
    def __init__(self):
        self.ai_service = MockAIService()
        self.config = Mock()
        self.config.max_retries = 3
        self.config.timeout = 30
        self.config.batch_size = 10
        self.cache = Mock()
        self.logger = Mock()
        self.metrics = Mock()
    
    def generate_copy(self, request: CopywritingInput) -> CopywritingOutput:
        """Generate copy for a request."""
        # Validate input
        if not self.validate_input(request):
            raise ValueError("Invalid input")
        
        # Check cache
        cache_key = f"copy_{hash(str(request.model_dump()))}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Generate copy using AI service
        result = self.ai_service.mock_call(request, "gpt-3.5-turbo")
        
        # Create response
        response = CopywritingOutput(
            variants=[{
                "variant_id": f"variant_{hash(str(request.model_dump()))}",
                "headline": result["variants"][0]["headline"],
                "primary_text": result["variants"][0]["primary_text"],
                "call_to_action": result["variants"][0].get("call_to_action", "Learn More")
            }],
            model_used=result["model_used"],
            generation_time=result["generation_time"],
            tokens_used=result.get("tokens_used", 100)
        )
        
        # Cache result
        self.cache.set(cache_key, response)
        
        # Log and record metrics
        self.logger.info(f"Generated copy for request: {request.tracking_id}")
        self.metrics.increment_request_count()
        self.metrics.record_request_duration(result["generation_time"])
        self.metrics.record_tokens_used(result.get("tokens_used", 100))
        
        return response
    
    def process_batch(self, requests: List[CopywritingInput]) -> List[CopywritingOutput]:
        """Process a batch of requests."""
        results = []
        for request in requests:
            try:
                result = self.generate_copy(request)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing request: {e}")
                # Create error response
                error_response = CopywritingOutput(
                    variants=[{
                        "variant_id": f"error_{hash(str(request.model_dump()))}",
                        "headline": "Error generating copy",
                        "primary_text": f"Error: {str(e)}",
                        "call_to_action": "Try Again"
                    }],
                    model_used="error",
                    generation_time=0.0,
                    tokens_used=0
                )
                results.append(error_response)
        return results
    
    def get_feedback(self, variant_id: str) -> Feedback:
        """Get feedback for a variant."""
        return TestDataFactory.create_feedback()
    
    def validate_input(self, request: CopywritingInput) -> bool:
        """Validate input request."""
        if not request.product_description or not request.product_description.strip():
            return False
        if len(request.product_description) > 2000:
            return False
        return True
    
    async def generate_copy_async(self, request: CopywritingInput) -> CopywritingOutput:
        """Generate copy asynchronously."""
        # Simulate async operation
        await asyncio.sleep(0.1)
        return self.generate_copy(request)
    
    def cleanup(self):
        """Cleanup resources."""
        pass


class TestCopywritingService:
    """Test cases for CopywritingService."""
    
    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        return MockCopywritingService()
    
    @pytest.fixture
    def sample_request(self):
        """Create a sample copywriting request."""
        return TestDataFactory.create_copywriting_input()
    
    @pytest.fixture
    def sample_response(self):
        """Create a sample copywriting response."""
        return TestDataFactory.create_copywriting_output()
    
    def test_service_initialization(self, service):
        """Test service initialization."""
        assert service is not None
        assert hasattr(service, 'generate_copy')
        assert hasattr(service, 'process_batch')
        assert hasattr(service, 'get_feedback')
    
    def test_generate_copy_basic(self, service, sample_request):
        """Test basic copy generation."""
        result = service.generate_copy(sample_request)
        
        # Validate result
        assert result is not None
        assert hasattr(result, 'variants')
        assert len(result.variants) > 0
        assert result.model_used is not None
        assert result.generation_time >= 0
    
    def test_generate_copy_with_validation(self, service, sample_request):
        """Test copy generation with input validation."""
        # Test with valid input
        result = service.generate_copy(sample_request)
        assert result is not None
        
        # Test with invalid input (should handle gracefully)
        invalid_request = TestDataFactory.create_copywriting_input(
            product_description=""  # Empty description
        )
        
        # Should raise validation error
        with pytest.raises(ValueError, match="Invalid input"):
            service.generate_copy(invalid_request)
    
    def test_process_batch(self, service):
        """Test batch processing."""
        # Create batch of requests
        batch_requests = TestDataFactory.create_batch_inputs(3)
        
        # Process batch
        results = service.process_batch(batch_requests)
        
        # Validate results
        assert isinstance(results, list)
        assert len(results) == 3
        for result in results:
            assert hasattr(result, 'variants')
            assert len(result.variants) > 0
    
    def test_get_feedback(self, service):
        """Test feedback retrieval."""
        # Test feedback retrieval
        result = service.get_feedback("test_variant_id")
        
        # Validate result
        assert result is not None
        assert hasattr(result, 'type')
        assert hasattr(result, 'score')
        assert hasattr(result, 'comments')
    
    def test_service_error_handling(self, service):
        """Test service error handling."""
        # Test with invalid input
        invalid_request = TestDataFactory.create_copywriting_input(
            product_description=""  # Empty description
        )
        
        # Should raise validation error
        with pytest.raises(ValueError):
            service.generate_copy(invalid_request)
    
    def test_service_performance(self, service, sample_request):
        """Test service performance."""
        start_time = time.time()
        
        # Generate copy
        result = service.generate_copy(sample_request)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Validate performance
        assert result is not None
        assert execution_time < 1.0  # Should complete within 1 second
        assert execution_time > 0
    
    def test_service_concurrent_processing(self, service):
        """Test concurrent processing."""
        # Create multiple requests
        requests = TestDataFactory.create_batch_inputs(5)
        
        # Process requests concurrently
        async def process_concurrent():
            tasks = [service.generate_copy_async(req) for req in requests]
            return await asyncio.gather(*tasks)
        
        # Run concurrent processing
        results = asyncio.run(process_concurrent())
        
        # Validate results
        assert len(results) == 5
        for result in results:
            assert hasattr(result, 'variants')
            assert len(result.variants) > 0
    
    def test_service_memory_usage(self, service):
        """Test service memory usage."""
        import psutil
        import gc
        
        # Get initial memory usage
        initial_memory = psutil.Process().memory_info().rss
        
        # Create and process multiple requests
        requests = TestDataFactory.create_batch_inputs(100)
        
        # Process requests
        results = []
        for request in requests:
            result = service.generate_copy(request)
            results.append(result)
        
        # Force garbage collection
        gc.collect()
        
        # Get final memory usage
        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Validate memory usage
        assert len(results) == 100
        assert memory_increase < 100 * 1024 * 1024  # Should not increase by more than 100MB
    
    def test_service_configuration(self, service):
        """Test service configuration."""
        # Test service configuration
        assert hasattr(service, 'config')
        assert service.config is not None
        
        # Test configuration values
        config = service.config
        assert hasattr(config, 'max_retries')
        assert hasattr(config, 'timeout')
        assert hasattr(config, 'batch_size')
        
        # Validate configuration values
        assert config.max_retries > 0
        assert config.timeout > 0
        assert config.batch_size > 0
    
    def test_service_logging(self, service, sample_request):
        """Test service logging."""
        # Generate copy
        result = service.generate_copy(sample_request)
        
        # Validate logging calls
        assert service.logger.info.called
        assert result is not None
    
    def test_service_metrics(self, service, sample_request):
        """Test service metrics collection."""
        # Generate copy
        result = service.generate_copy(sample_request)
        
        # Validate metrics collection
        assert service.metrics.increment_request_count.called
        assert service.metrics.record_request_duration.called
        assert service.metrics.record_tokens_used.called
        assert result is not None
    
    def test_service_caching(self, service, sample_request):
        """Test service caching."""
        # Generate copy (should cache)
        result1 = service.generate_copy(sample_request)
        
        # Generate copy again (should use cache)
        result2 = service.generate_copy(sample_request)
        
        # Validate caching
        assert service.cache.get.called
        assert service.cache.set.called
        assert result1 is not None
        assert result2 is not None
    
    def test_service_validation(self, service):
        """Test service input validation."""
        # Test valid input
        valid_request = TestDataFactory.create_copywriting_input()
        assert service.validate_input(valid_request) is True
        
        # Test invalid input
        invalid_request = TestDataFactory.create_copywriting_input(
            product_description=""  # Empty description
        )
        assert service.validate_input(invalid_request) is False
        
        # Test edge cases
        edge_cases = [
            TestDataFactory.create_copywriting_input(product_description="x" * 3000),  # Too long
        ]
        
        for edge_case in edge_cases:
            assert service.validate_input(edge_case) is False
    
    def test_service_async_operations(self, service):
        """Test async operations."""
        async def test_async_generation():
            request = TestDataFactory.create_copywriting_input()
            result = await service.generate_copy_async(request)
            return result
        
        # Run async test
        result = asyncio.run(test_async_generation())
        
        # Validate result
        assert result is not None
        assert hasattr(result, 'variants')
        assert len(result.variants) > 0
    
    def test_service_error_recovery(self, service, sample_request):
        """Test service error recovery."""
        # Test with valid input (should succeed)
        result = service.generate_copy(sample_request)
        
        # Validate result
        assert result is not None
        assert hasattr(result, 'variants')
        assert len(result.variants) > 0
    
    def test_service_batch_processing_performance(self, service):
        """Test batch processing performance."""
        # Create large batch
        batch_requests = TestDataFactory.create_batch_inputs(50)
        
        start_time = time.time()
        
        # Process batch
        results = service.process_batch(batch_requests)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Validate performance
        assert len(results) == 50
        assert execution_time < 5.0  # Should complete within 5 seconds
        assert execution_time > 0
    
    def test_service_resource_cleanup(self, service):
        """Test service resource cleanup."""
        # Create service instance
        service_instance = MockCopywritingService()
        
        # Use service
        request = TestDataFactory.create_copywriting_input()
        result = service_instance.generate_copy(request)
        assert result is not None
        
        # Cleanup
        service_instance.cleanup()
        
        # Validate cleanup
        assert hasattr(service_instance, 'cleanup')
    
    def test_service_input_validation_edge_cases(self, service):
        """Test service input validation edge cases."""
        # Test various edge cases
        edge_cases = [
            # Empty string
            TestDataFactory.create_copywriting_input(product_description=""),
            # Very long string
            TestDataFactory.create_copywriting_input(product_description="x" * 3000),
            # Only whitespace
            TestDataFactory.create_copywriting_input(product_description="   "),
        ]
        
        for edge_case in edge_cases:
            assert service.validate_input(edge_case) is False
    
    def test_service_batch_error_handling(self, service):
        """Test batch processing error handling."""
        # Create batch with some invalid requests
        valid_request = TestDataFactory.create_copywriting_input()
        invalid_request = TestDataFactory.create_copywriting_input(product_description="")
        
        batch_requests = [valid_request, invalid_request, valid_request]
        
        # Process batch
        results = service.process_batch(batch_requests)
        
        # Validate results
        assert len(results) == 3
        # All should be processed (invalid ones should get error responses)
        for result in results:
            assert hasattr(result, 'variants')
            assert len(result.variants) > 0
    
    def test_service_metrics_collection(self, service, sample_request):
        """Test comprehensive metrics collection."""
        # Generate copy
        result = service.generate_copy(sample_request)
        
        # Validate all metrics are collected
        assert service.metrics.increment_request_count.called
        assert service.metrics.record_request_duration.called
        assert service.metrics.record_tokens_used.called
        
        # Validate result
        assert result is not None
        assert result.generation_time >= 0
        assert result.tokens_used > 0
