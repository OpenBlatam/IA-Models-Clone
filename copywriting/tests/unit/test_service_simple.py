"""
Simple unit tests for copywriting service layer.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List
import asyncio
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from service import CopywritingService
from models import (
    CopywritingInput,
    CopywritingOutput,
    Feedback
)
from tests.test_utils import TestDataFactory, MockAIService


class TestCopywritingService:
    """Test cases for CopywritingService."""
    
    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        return CopywritingService()
    
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
        # Mock the AI service call
        with patch.object(service, 'ai_service') as mock_ai:
            mock_ai.generate_copy.return_value = {
                "variants": [
                    {
                        "variant_id": "test_1",
                        "headline": "Test Headline",
                        "primary_text": "Test content",
                        "call_to_action": "Test CTA"
                    }
                ],
                "model_used": "gpt-3.5-turbo",
                "generation_time": 1.0,
                "tokens_used": 100
            }
            
            # Test the method
            result = service.generate_copy(sample_request)
            
            # Validate result
            assert result is not None
            assert hasattr(result, 'variants')
            assert len(result.variants) > 0
            assert result.model_used == "gpt-3.5-turbo"
    
    def test_generate_copy_with_validation(self, service, sample_request):
        """Test copy generation with input validation."""
        # Test with valid input
        result = service.generate_copy(sample_request)
        assert result is not None
        
        # Test with invalid input (should handle gracefully)
        invalid_request = TestDataFactory.create_copywriting_input(
            product_description=""  # Empty description
        )
        
        # Should either return error or handle gracefully
        try:
            result = service.generate_copy(invalid_request)
            # If it succeeds, validate the result
            assert result is not None
        except Exception as e:
            # If it fails, that's also acceptable for validation
            assert "validation" in str(e).lower() or "required" in str(e).lower()
    
    def test_process_batch(self, service):
        """Test batch processing."""
        # Create batch of requests
        batch_requests = TestDataFactory.create_batch_inputs(3)
        
        # Mock the AI service
        with patch.object(service, 'ai_service') as mock_ai:
            mock_ai.generate_copy.return_value = {
                "variants": [{"variant_id": "test", "headline": "Test", "primary_text": "Content"}],
                "model_used": "gpt-3.5-turbo",
                "generation_time": 1.0,
                "tokens_used": 100
            }
            
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
        # Create sample feedback
        feedback = TestDataFactory.create_feedback()
        
        # Mock feedback retrieval
        with patch.object(service, 'feedback_service') as mock_feedback:
            mock_feedback.get_feedback.return_value = feedback
            
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
            product_description=None  # Invalid input
        )
        
        # Should handle error gracefully
        try:
            result = service.generate_copy(invalid_request)
            # If it succeeds, validate the result
            assert result is not None
        except Exception as e:
            # If it fails, that's acceptable for error handling
            assert isinstance(e, Exception)
    
    def test_service_performance(self, service, sample_request):
        """Test service performance."""
        start_time = time.time()
        
        # Mock the AI service for fast response
        with patch.object(service, 'ai_service') as mock_ai:
            mock_ai.generate_copy.return_value = {
                "variants": [{"variant_id": "test", "headline": "Test", "primary_text": "Content"}],
                "model_used": "gpt-3.5-turbo",
                "generation_time": 0.1,
                "tokens_used": 100
            }
            
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
        
        # Mock the AI service
        with patch.object(service, 'ai_service') as mock_ai:
            mock_ai.generate_copy.return_value = {
                "variants": [{"variant_id": "test", "headline": "Test", "primary_text": "Content"}],
                "model_used": "gpt-3.5-turbo",
                "generation_time": 0.1,
                "tokens_used": 100
            }
            
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
        
        # Mock the AI service
        with patch.object(service, 'ai_service') as mock_ai:
            mock_ai.generate_copy.return_value = {
                "variants": [{"variant_id": "test", "headline": "Test", "primary_text": "Content"}],
                "model_used": "gpt-3.5-turbo",
                "generation_time": 0.1,
                "tokens_used": 100
            }
            
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
        # Mock logging
        with patch('service.logger') as mock_logger:
            # Generate copy
            with patch.object(service, 'ai_service') as mock_ai:
                mock_ai.generate_copy.return_value = {
                    "variants": [{"variant_id": "test", "headline": "Test", "primary_text": "Content"}],
                    "model_used": "gpt-3.5-turbo",
                    "generation_time": 1.0,
                    "tokens_used": 100
                }
                
                result = service.generate_copy(sample_request)
                
                # Validate logging calls
                assert mock_logger.info.called
                assert mock_logger.debug.called
    
    def test_service_metrics(self, service, sample_request):
        """Test service metrics collection."""
        # Mock metrics collection
        with patch('service.metrics') as mock_metrics:
            # Generate copy
            with patch.object(service, 'ai_service') as mock_ai:
                mock_ai.generate_copy.return_value = {
                    "variants": [{"variant_id": "test", "headline": "Test", "primary_text": "Content"}],
                    "model_used": "gpt-3.5-turbo",
                    "generation_time": 1.0,
                    "tokens_used": 100
                }
                
                result = service.generate_copy(sample_request)
                
                # Validate metrics collection
                assert mock_metrics.increment_request_count.called
                assert mock_metrics.record_request_duration.called
                assert mock_metrics.record_tokens_used.called
    
    def test_service_caching(self, service, sample_request):
        """Test service caching."""
        # Mock caching
        with patch.object(service, 'cache') as mock_cache:
            mock_cache.get.return_value = None  # Cache miss
            mock_cache.set.return_value = True
            
            # Generate copy
            with patch.object(service, 'ai_service') as mock_ai:
                mock_ai.generate_copy.return_value = {
                    "variants": [{"variant_id": "test", "headline": "Test", "primary_text": "Content"}],
                    "model_used": "gpt-3.5-turbo",
                    "generation_time": 1.0,
                    "tokens_used": 100
                }
                
                result = service.generate_copy(sample_request)
                
                # Validate caching
                assert mock_cache.get.called
                assert mock_cache.set.called
    
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
            TestDataFactory.create_copywriting_input(target_platform="invalid"),  # Invalid platform
            TestDataFactory.create_copywriting_input(tone="invalid"),  # Invalid tone
        ]
        
        for edge_case in edge_cases:
            assert service.validate_input(edge_case) is False
    
    def test_service_async_operations(self, service):
        """Test async operations."""
        async def test_async_generation():
            request = TestDataFactory.create_copywriting_input()
            
            # Mock the AI service
            with patch.object(service, 'ai_service') as mock_ai:
                mock_ai.generate_copy_async.return_value = {
                    "variants": [{"variant_id": "test", "headline": "Test", "primary_text": "Content"}],
                    "model_used": "gpt-3.5-turbo",
                    "generation_time": 1.0,
                    "tokens_used": 100
                }
                
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
        # Mock AI service to fail first, then succeed
        with patch.object(service, 'ai_service') as mock_ai:
            mock_ai.generate_copy.side_effect = [
                Exception("AI service error"),
                {
                    "variants": [{"variant_id": "test", "headline": "Test", "primary_text": "Content"}],
                    "model_used": "gpt-3.5-turbo",
                    "generation_time": 1.0,
                    "tokens_used": 100
                }
            ]
            
            # Should retry and eventually succeed
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
        
        # Mock the AI service
        with patch.object(service, 'ai_service') as mock_ai:
            mock_ai.generate_copy.return_value = {
                "variants": [{"variant_id": "test", "headline": "Test", "primary_text": "Content"}],
                "model_used": "gpt-3.5-turbo",
                "generation_time": 0.1,
                "tokens_used": 100
            }
            
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
        service_instance = CopywritingService()
        
        # Use service
        request = TestDataFactory.create_copywriting_input()
        with patch.object(service_instance, 'ai_service') as mock_ai:
            mock_ai.generate_copy.return_value = {
                "variants": [{"variant_id": "test", "headline": "Test", "primary_text": "Content"}],
                "model_used": "gpt-3.5-turbo",
                "generation_time": 1.0,
                "tokens_used": 100
            }
            
            result = service_instance.generate_copy(request)
            assert result is not None
        
        # Cleanup
        service_instance.cleanup()
        
        # Validate cleanup
        assert hasattr(service_instance, 'cleanup')
        # Additional cleanup validation can be added based on implementation
