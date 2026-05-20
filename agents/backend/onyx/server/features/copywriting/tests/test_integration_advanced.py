"""
Advanced integration tests for copywriting service.
"""
import pytest
import asyncio
from typing import Dict, Any, List
from unittest.mock import patch, Mock, AsyncMock
import time

from tests.test_utils import TestDataFactory, MockAIService, TestAssertions
from agents.backend.onyx.server.features.copywriting.service import CopywritingService
from agents.backend.onyx.server.features.copywriting.models import (
    CopywritingRequest,
    CopywritingResponse,
    BatchCopywritingRequest,
    FeedbackRequest
)


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    @pytest.fixture
    def service(self):
        """Create service instance for testing."""
        return CopywritingService()
    
    @pytest.mark.asyncio
    async def test_complete_copywriting_workflow(self, service):
        """Test complete copywriting workflow from request to feedback."""
        # Step 1: Create copywriting request
        request = TestDataFactory.create_sample_request(
            product_description="Smartphone de última generación",
            target_platform="Instagram",
            tone="inspirational",
            target_audience="Tech enthusiasts",
            key_points=["Innovación", "Rendimiento", "Diseño"],
            instructions="Enfatiza la tecnología avanzada",
            restrictions=["No mencionar precio"],
            creativity_level=0.9,
            language="es"
        )
        
        # Step 2: Generate copywriting
        with patch.object(service, '_call_ai_model') as mock_call:
            mock_call.return_value = {
                "variants": [
                    {
                        "headline": "¡Revoluciona tu Experiencia! 📱",
                        "primary_text": "Descubre el smartphone que cambiará tu forma de ver el mundo. Tecnología de vanguardia, rendimiento excepcional y un diseño que inspira.",
                        "call_to_action": "Descubre más",
                        "hashtags": ["#tecnología", "#innovación", "#smartphone"]
                    },
                    {
                        "headline": "El Futuro en tus Manos 🚀",
                        "primary_text": "Experimenta la próxima generación de smartphones. Potencia, elegancia y funcionalidad en un solo dispositivo.",
                        "call_to_action": "Conoce más",
                        "hashtags": ["#futuro", "#tecnología", "#innovación"]
                    }
                ],
                "model_used": "gpt-3.5-turbo",
                "generation_time": 2.5,
                "extra_metadata": {"tokens_used": 200}
            }
            
            response = await service.generate_copywriting(request, "gpt-3.5-turbo")
        
        # Step 3: Validate response
        TestAssertions.assert_valid_copywriting_response(response)
        assert len(response.variants) == 2
        assert response.model_used == "gpt-3.5-turbo"
        assert response.generation_time == 2.5
        
        # Step 4: Validate content quality
        for variant in response.variants:
            assert "smartphone" in variant["headline"].lower() or "smartphone" in variant["primary_text"].lower()
            assert len(variant["hashtags"]) >= 2
            assert variant["call_to_action"] is not None
        
        # Step 5: Simulate user feedback
        feedback = FeedbackRequest(
            variant_id="variant_1",
            feedback={
                "type": "human",
                "score": 0.9,
                "comments": "Excelente copy, muy inspirador",
                "user_id": "user123",
                "timestamp": "2024-06-01T12:00:00Z"
            }
        )
        
        assert feedback.variant_id == "variant_1"
        assert feedback.feedback["score"] == 0.9
        assert "inspirador" in feedback.feedback["comments"]
    
    @pytest.mark.asyncio
    async def test_batch_processing_workflow(self, service):
        """Test complete batch processing workflow."""
        # Step 1: Create batch request
        requests = [
            TestDataFactory.create_sample_request(
                product_description=f"Producto {i}",
                target_platform=["Instagram", "Facebook", "Twitter"][i % 3],
                tone=["inspirational", "informative", "playful"][i % 3]
            )
            for i in range(5)
        ]
        
        batch_request = BatchCopywritingRequest(requests=requests)
        
        # Step 2: Process batch
        with patch.object(service, 'generate_copywriting') as mock_generate:
            mock_generate.side_effect = [
                TestDataFactory.create_sample_response(
                    variants=[{"headline": f"Headline {i}", "primary_text": f"Content {i}"}]
                )
                for i in range(5)
            ]
            
            batch_response = await service.batch_generate_copywriting(batch_request)
        
        # Step 3: Validate batch response
        TestAssertions.assert_valid_batch_response(batch_response, 5)
        
        # Step 4: Validate individual results
        for i, result in enumerate(batch_response.results):
            assert result.variants[0]["headline"] == f"Headline {i}"
            assert result.variants[0]["primary_text"] == f"Content {i}"
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, service):
        """Test error recovery and fallback workflows."""
        request = TestDataFactory.create_sample_request()
        
        # Test AI service failure with retry
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


class TestServiceIntegration:
    """Test service integration with external dependencies."""
    
    @pytest.fixture
    def service(self):
        """Create service instance for testing."""
        return CopywritingService()
    
    @pytest.mark.asyncio
    async def test_ai_model_integration(self, service):
        """Test integration with AI models."""
        request = TestDataFactory.create_sample_request()
        
        # Test different AI models
        models = ["gpt-3.5-turbo", "gpt-4", "claude-3"]
        
        for model in models:
            with patch.object(service, '_call_ai_model') as mock_call:
                mock_call.return_value = {
                    "variants": [{"headline": f"Test {model}", "primary_text": "Content"}],
                    "model_used": model,
                    "generation_time": 1.0,
                    "extra_metadata": {}
                }
                
                response = await service.generate_copywriting(request, model)
                
                assert response.model_used == model
                assert response.variants[0]["headline"] == f"Test {model}"
    
    @pytest.mark.asyncio
    async def test_database_integration(self, service):
        """Test integration with database operations."""
        # This would test actual database operations
        # For now, we test that the service can handle database-like operations
        
        request = TestDataFactory.create_sample_request()
        
        # Mock database operations
        with patch('agents.backend.onyx.server.features.copywriting.service.database') as mock_db:
            mock_db.save_request.return_value = "request_id_123"
            mock_db.save_response.return_value = "response_id_456"
            
            with patch.object(service, '_call_ai_model') as mock_call:
                mock_call.return_value = {
                    "variants": [{"headline": "Test", "primary_text": "Content"}],
                    "model_used": "gpt-3.5-turbo",
                    "generation_time": 1.0,
                    "extra_metadata": {}
                }
                
                response = await service.generate_copywriting(request, "gpt-3.5-turbo")
                
                # Verify database operations were called
                mock_db.save_request.assert_called_once()
                mock_db.save_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_integration(self, service):
        """Test integration with caching system."""
        request = TestDataFactory.create_sample_request()
        
        # Mock cache operations
        with patch('agents.backend.onyx.server.features.copywriting.service.cache') as mock_cache:
            mock_cache.get.return_value = None  # Cache miss
            mock_cache.set.return_value = True
            
            with patch.object(service, '_call_ai_model') as mock_call:
                mock_call.return_value = {
                    "variants": [{"headline": "Test", "primary_text": "Content"}],
                    "model_used": "gpt-3.5-turbo",
                    "generation_time": 1.0,
                    "extra_metadata": {}
                }
                
                response = await service.generate_copywriting(request, "gpt-3.5-turbo")
                
                # Verify cache operations
                mock_cache.get.assert_called_once()
                mock_cache.set.assert_called_once()


class TestConcurrentProcessing:
    """Test concurrent processing capabilities."""
    
    @pytest.fixture
    def service(self):
        """Create service instance for testing."""
        return CopywritingService()
    
    @pytest.mark.asyncio
    async def test_concurrent_single_requests(self, service):
        """Test concurrent single request processing."""
        requests = [
            TestDataFactory.create_sample_request(
                product_description=f"Concurrent product {i}"
            )
            for i in range(10)
        ]
        
        async def process_request(request):
            with patch.object(service, '_call_ai_model') as mock_call:
                mock_call.return_value = {
                    "variants": [{"headline": f"Concurrent {request.product_description}", "primary_text": "Content"}],
                    "model_used": "gpt-3.5-turbo",
                    "generation_time": 0.1,
                    "extra_metadata": {}
                }
                return await service.generate_copywriting(request, "gpt-3.5-turbo")
        
        # Process all requests concurrently
        start_time = time.time()
        results = await asyncio.gather(*[process_request(req) for req in requests])
        end_time = time.time()
        
        # Validate results
        assert len(results) == 10
        assert all(result is not None for result in results)
        assert end_time - start_time < 5.0  # Should complete quickly
    
    @pytest.mark.asyncio
    async def test_concurrent_batch_requests(self, service):
        """Test concurrent batch request processing."""
        batch_requests = [
            BatchCopywritingRequest(requests=[
                TestDataFactory.create_sample_request(
                    product_description=f"Batch {i} item {j}"
                )
                for j in range(3)
            ])
            for i in range(5)
        ]
        
        async def process_batch(batch_request):
            with patch.object(service, 'generate_copywriting') as mock_generate:
                mock_generate.return_value = TestDataFactory.create_sample_response()
                return await service.batch_generate_copywriting(batch_request)
        
        # Process all batches concurrently
        start_time = time.time()
        results = await asyncio.gather(*[process_batch(batch) for batch in batch_requests])
        end_time = time.time()
        
        # Validate results
        assert len(results) == 5
        assert all(len(result.results) == 3 for result in results)
        assert end_time - start_time < 10.0  # Should complete quickly
    
    @pytest.mark.asyncio
    async def test_mixed_concurrent_processing(self, service):
        """Test mixed concurrent processing (single + batch requests)."""
        # Mix of single and batch requests
        single_requests = [
            TestDataFactory.create_sample_request(
                product_description=f"Single {i}"
            )
            for i in range(5)
        ]
        
        batch_requests = [
            BatchCopywritingRequest(requests=[
                TestDataFactory.create_sample_request(
                    product_description=f"Batch {i} item {j}"
                )
                for j in range(2)
            ])
            for i in range(3)
        ]
        
        async def process_single(request):
            with patch.object(service, '_call_ai_model') as mock_call:
                mock_call.return_value = {
                    "variants": [{"headline": f"Single {request.product_description}", "primary_text": "Content"}],
                    "model_used": "gpt-3.5-turbo",
                    "generation_time": 0.1,
                    "extra_metadata": {}
                }
                return await service.generate_copywriting(request, "gpt-3.5-turbo")
        
        async def process_batch(batch_request):
            with patch.object(service, 'generate_copywriting') as mock_generate:
                mock_generate.return_value = TestDataFactory.create_sample_response()
                return await service.batch_generate_copywriting(batch_request)
        
        # Process all requests concurrently
        start_time = time.time()
        single_results = await asyncio.gather(*[process_single(req) for req in single_requests])
        batch_results = await asyncio.gather(*[process_batch(batch) for batch in batch_requests])
        end_time = time.time()
        
        # Validate results
        assert len(single_results) == 5
        assert len(batch_results) == 3
        assert all(result is not None for result in single_results)
        assert all(len(result.results) == 2 for result in batch_results)
        assert end_time - start_time < 8.0  # Should complete quickly


class TestDataFlowIntegration:
    """Test data flow through the system."""
    
    @pytest.fixture
    def service(self):
        """Create service instance for testing."""
        return CopywritingService()
    
    @pytest.mark.asyncio
    async def test_request_data_flow(self, service):
        """Test data flow from request to response."""
        # Create request with specific data
        request = TestDataFactory.create_sample_request(
            product_description="Test product",
            target_platform="Instagram",
            tone="inspirational",
            key_points=["Point1", "Point2"],
            instructions="Test instructions"
        )
        
        # Mock AI call to verify data is passed correctly
        with patch.object(service, '_call_ai_model') as mock_call:
            mock_call.return_value = {
                "variants": [{"headline": "Test", "primary_text": "Content"}],
                "model_used": "gpt-3.5-turbo",
                "generation_time": 1.0,
                "extra_metadata": {}
            }
            
            response = await service.generate_copywriting(request, "gpt-3.5-turbo")
            
            # Verify AI call received correct data
            call_args = mock_call.call_args[0]
            assert call_args[0] == request
            assert call_args[1] == "gpt-3.5-turbo"
    
    @pytest.mark.asyncio
    async def test_response_data_flow(self, service):
        """Test data flow in response processing."""
        request = TestDataFactory.create_sample_request()
        
        # Mock AI response with specific data
        mock_ai_response = {
            "variants": [
                {
                    "headline": "Test Headline",
                    "primary_text": "Test Content",
                    "call_to_action": "Test CTA",
                    "hashtags": ["#test", "#example"]
                }
            ],
            "model_used": "gpt-3.5-turbo",
            "generation_time": 2.5,
            "extra_metadata": {"tokens_used": 150, "cost": 0.01}
        }
        
        with patch.object(service, '_call_ai_model') as mock_call:
            mock_call.return_value = mock_ai_response
            
            response = await service.generate_copywriting(request, "gpt-3.5-turbo")
            
            # Verify response data is correctly processed
            assert response.model_used == "gpt-3.5-turbo"
            assert response.generation_time == 2.5
            assert response.extra_metadata["tokens_used"] == 150
            assert response.extra_metadata["cost"] == 0.01
            
            variant = response.variants[0]
            assert variant["headline"] == "Test Headline"
            assert variant["primary_text"] == "Test Content"
            assert variant["call_to_action"] == "Test CTA"
            assert variant["hashtags"] == ["#test", "#example"]
    
    @pytest.mark.asyncio
    async def test_error_data_flow(self, service):
        """Test data flow in error scenarios."""
        request = TestDataFactory.create_sample_request()
        
        # Test AI service error
        with patch.object(service, '_call_ai_model') as mock_call:
            mock_call.side_effect = Exception("AI service error")
            
            with pytest.raises(Exception, match="AI service error"):
                await service.generate_copywriting(request, "gpt-3.5-turbo")
        
        # Test invalid model
        with pytest.raises(ValueError, match="Modelo no soportado"):
            await service.generate_copywriting(request, "invalid_model")


class TestPerformanceIntegration:
    """Test performance integration scenarios."""
    
    @pytest.fixture
    def service(self):
        """Create service instance for testing."""
        return CopywritingService()
    
    @pytest.mark.asyncio
    async def test_high_throughput_integration(self, service):
        """Test high throughput integration scenario."""
        # Create many requests
        requests = [
            TestDataFactory.create_sample_request(
                product_description=f"High throughput product {i}"
            )
            for i in range(50)
        ]
        
        # Process with realistic delays
        async def process_request(request):
            with patch.object(service, '_call_ai_model') as mock_call:
                # Simulate realistic AI call delay
                await asyncio.sleep(0.01)
                mock_call.return_value = {
                    "variants": [{"headline": f"HT {request.product_description}", "primary_text": "Content"}],
                    "model_used": "gpt-3.5-turbo",
                    "generation_time": 0.01,
                    "extra_metadata": {}
                }
                return await service.generate_copywriting(request, "gpt-3.5-turbo")
        
        start_time = time.time()
        results = await asyncio.gather(*[process_request(req) for req in requests])
        end_time = time.time()
        
        # Validate performance
        total_time = end_time - start_time
        throughput = len(requests) / total_time
        
        assert len(results) == 50
        assert throughput >= 10  # At least 10 requests per second
        assert total_time < 10.0  # Should complete within 10 seconds
    
    @pytest.mark.asyncio
    async def test_memory_integration(self, service):
        """Test memory usage in integration scenarios."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process many requests
        for i in range(100):
            request = TestDataFactory.create_sample_request(
                product_description=f"Memory test product {i}"
            )
            
            with patch.object(service, '_call_ai_model') as mock_call:
                mock_call.return_value = {
                    "variants": [{"headline": f"Memory {i}", "primary_text": "Content"}],
                    "model_used": "gpt-3.5-turbo",
                    "generation_time": 0.1,
                    "extra_metadata": {}
                }
                
                await service.generate_copywriting(request, "gpt-3.5-turbo")
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Validate memory usage
        assert memory_increase < 200.0  # Should not use more than 200MB
        assert memory_increase >= 0  # Memory should not decrease
