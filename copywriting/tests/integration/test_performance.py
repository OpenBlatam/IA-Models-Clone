"""
Performance and load tests for copywriting service.
"""
import pytest
import asyncio
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import statistics

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import (
    CopywritingInput,
    CopywritingOutput
)
from tests.test_utils import TestDataFactory, MockAIService


class TestPerformance:
    """Performance tests for copywriting service."""
    
    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        return CopywritingService()
    
    @pytest.fixture
    def sample_requests(self):
        """Create multiple sample requests for load testing."""
        return [
            CopywritingRequest(
                product_description=f"Producto deportivo {i}",
                target_platform="Instagram",
                tone="inspirational",
                target_audience="Jóvenes activos",
                key_points=["Calidad", "Diseño", "Precio"],
                instructions="Enfatiza la innovación",
                restrictions=["no mencionar precio"],
                creativity_level=0.8,
                language="es"
            )
            for i in range(10)
        ]
    
    @pytest.mark.asyncio
    async def test_single_request_performance(self, service, sample_requests):
        """Test performance of single request generation."""
        request = sample_requests[0]
        
        start_time = time.time()
        
        with pytest.MonkeyPatch().context() as m:
            m.setattr(service, '_call_ai_model', self._mock_ai_call)
            result = await service.generate_copywriting(request, "gpt-3.5-turbo")
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        assert generation_time < 5.0  # Should complete within 5 seconds
        assert result.generation_time > 0
        assert len(result.variants) > 0
    
    @pytest.mark.asyncio
    async def test_batch_request_performance(self, service, sample_requests):
        """Test performance of batch request generation."""
        batch_request = BatchCopywritingRequest(requests=sample_requests[:5])
        
        start_time = time.time()
        
        with pytest.MonkeyPatch().context() as m:
            m.setattr(service, '_call_ai_model', self._mock_ai_call)
            result = await service.batch_generate_copywriting(batch_request)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        assert total_time < 10.0  # Should complete within 10 seconds
        assert len(result.results) == 5
        
        # Check individual generation times
        for individual_result in result.results:
            assert individual_result.generation_time > 0
            assert len(individual_result.variants) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_performance(self, service, sample_requests):
        """Test performance with concurrent requests."""
        async def generate_single(request):
            with pytest.MonkeyPatch().context() as m:
                m.setattr(service, '_call_ai_model', self._mock_ai_call)
                return await service.generate_copywriting(request, "gpt-3.5-turbo")
        
        start_time = time.time()
        
        # Run 5 concurrent requests
        tasks = [generate_single(req) for req in sample_requests[:5]]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        assert total_time < 8.0  # Should complete within 8 seconds
        assert len(results) == 5
        
        for result in results:
            assert len(result.variants) > 0
            assert result.generation_time > 0
    
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, service, sample_requests):
        """Test memory usage stability with multiple requests."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate multiple requests
        for i in range(20):
            request = sample_requests[i % len(sample_requests)]
            
            with pytest.MonkeyPatch().context() as m:
                m.setattr(service, '_call_ai_model', self._mock_ai_call)
                await service.generate_copywriting(request, "gpt-3.5-turbo")
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100.0
    
    @pytest.mark.asyncio
    async def test_response_time_consistency(self, service, sample_requests):
        """Test that response times are consistent."""
        response_times = []
        
        for i in range(10):
            request = sample_requests[i % len(sample_requests)]
            
            start_time = time.time()
            
            with pytest.MonkeyPatch().context() as m:
                m.setattr(service, '_call_ai_model', self._mock_ai_call)
                await service.generate_copywriting(request, "gpt-3.5-turbo")
            
            end_time = time.time()
            response_times.append(end_time - start_time)
        
        # Calculate statistics
        mean_time = statistics.mean(response_times)
        std_dev = statistics.stdev(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        
        # Response times should be consistent
        assert std_dev < mean_time * 0.5  # Standard deviation should be less than 50% of mean
        assert max_time < mean_time * 2.0  # Max time should be less than 2x mean
        assert min_time > 0.1  # Minimum time should be reasonable
    
    @pytest.mark.asyncio
    async def test_error_handling_performance(self, service, sample_requests):
        """Test that error handling doesn't significantly impact performance."""
        request = sample_requests[0]
        
        # Test with AI model error
        start_time = time.time()
        
        with pytest.MonkeyPatch().context() as m:
            m.setattr(service, '_call_ai_model', self._mock_ai_error)
            
            with pytest.raises(Exception):
                await service.generate_copywriting(request, "gpt-3.5-turbo")
        
        end_time = time.time()
        error_handling_time = end_time - start_time
        
        # Error handling should be fast (less than 1 second)
        assert error_handling_time < 1.0
    
    @pytest.mark.asyncio
    async def test_large_batch_performance(self, service):
        """Test performance with large batch (maximum allowed size)."""
        requests = [
            CopywritingRequest(
                product_description=f"Producto {i}",
                target_platform="Instagram",
                tone="inspirational",
                language="es"
            )
            for i in range(20)  # Maximum batch size
        ]
        
        batch_request = BatchCopywritingRequest(requests=requests)
        
        start_time = time.time()
        
        with pytest.MonkeyPatch().context() as m:
            m.setattr(service, '_call_ai_model', self._mock_ai_call)
            result = await service.batch_generate_copywriting(batch_request)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        assert total_time < 30.0  # Should complete within 30 seconds
        assert len(result.results) == 20
        
        # Check that all results are valid
        for individual_result in result.results:
            assert len(individual_result.variants) > 0
            assert individual_result.generation_time > 0
    
    @pytest.mark.asyncio
    async def test_different_model_performance(self, service, sample_requests):
        """Test performance with different AI models."""
        request = sample_requests[0]
        models = ["gpt-3.5-turbo", "gpt-4", "claude-3"]
        
        for model in models:
            start_time = time.time()
            
            with pytest.MonkeyPatch().context() as m:
                m.setattr(service, '_call_ai_model', self._mock_ai_call)
                result = await service.generate_copywriting(request, model)
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            assert generation_time < 5.0  # All models should be fast
            assert result.model_used == model
            assert len(result.variants) > 0
    
    def _mock_ai_call(self, request, model):
        """Mock AI model call for testing."""
        # Simulate some processing time
        time.sleep(0.1)
        
        return {
            "variants": [
                {
                    "headline": f"Test Headline for {request.product_description}",
                    "primary_text": f"Test content for {request.target_platform}",
                    "call_to_action": "Compra ahora",
                    "hashtags": ["#test", "#producto"]
                }
            ],
            "model_used": model,
            "generation_time": 0.1,
            "extra_metadata": {"tokens_used": 50}
        }
    
    def _mock_ai_error(self, request, model):
        """Mock AI model call that raises an error."""
        raise Exception("Simulated AI model error")


class TestLoadTesting:
    """Load testing for copywriting service."""
    
    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        return CopywritingService()
    
    @pytest.mark.asyncio
    async def test_high_concurrency_load(self, service):
        """Test service under high concurrency load."""
        async def worker(worker_id):
            """Worker function for concurrent testing."""
            request = CopywritingRequest(
                product_description=f"Producto {worker_id}",
                target_platform="Instagram",
                tone="inspirational",
                language="es"
            )
            
            try:
                with pytest.MonkeyPatch().context() as m:
                    m.setattr(service, '_call_ai_model', self._mock_ai_call)
                    result = await service.generate_copywriting(request, "gpt-3.5-turbo")
                    return {"success": True, "worker_id": worker_id, "result": result}
            except Exception as e:
                return {"success": False, "worker_id": worker_id, "error": str(e)}
        
        # Run 50 concurrent workers
        num_workers = 50
        tasks = [worker(i) for i in range(num_workers)]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Count successful and failed requests
        successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        failed = num_workers - successful
        
        # At least 80% should succeed under load
        success_rate = successful / num_workers
        assert success_rate >= 0.8, f"Success rate {success_rate} is too low"
        
        # Should complete within reasonable time
        assert total_time < 60.0, f"Load test took too long: {total_time}s"
        
        print(f"Load test results: {successful}/{num_workers} successful ({success_rate:.2%}) in {total_time:.2f}s")
    
    def _mock_ai_call(self, request, model):
        """Mock AI model call for load testing."""
        # Simulate variable processing time
        import random
        time.sleep(random.uniform(0.05, 0.2))
        
        return {
            "variants": [
                {
                    "headline": f"Load Test Headline {request.product_description}",
                    "primary_text": f"Load test content for {request.target_platform}",
                    "call_to_action": "Compra ahora",
                    "hashtags": ["#loadtest", "#producto"]
                }
            ],
            "model_used": model,
            "generation_time": 0.1,
            "extra_metadata": {"tokens_used": 50}
        }





