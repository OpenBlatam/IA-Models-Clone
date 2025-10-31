"""
Simple performance and load tests for copywriting service.
"""
import pytest
import asyncio
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import statistics
import psutil
import gc
from unittest.mock import Mock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import (
    CopywritingInput,
    CopywritingOutput
)
from tests.test_utils import TestDataFactory, MockAIService


class MockCopywritingService:
    """Mock copywriting service for performance testing."""
    
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
        # Simulate processing time
        time.sleep(0.01)  # 10ms processing time
        
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
        
        return response
    
    def process_batch(self, requests: List[CopywritingInput]) -> List[CopywritingOutput]:
        """Process a batch of requests."""
        results = []
        for request in requests:
            try:
                result = self.generate_copy(request)
                results.append(result)
            except Exception as e:
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
    
    async def generate_copy_async(self, request: CopywritingInput) -> CopywritingOutput:
        """Generate copy asynchronously."""
        await asyncio.sleep(0.01)  # Simulate async processing
        return self.generate_copy(request)


class TestPerformance:
    """Performance tests for copywriting service."""
    
    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        return MockCopywritingService()
    
    @pytest.fixture
    def sample_requests(self):
        """Create multiple sample requests for load testing."""
        return TestDataFactory.create_batch_inputs(10)
    
    def test_single_request_performance(self, service):
        """Test performance of single request processing."""
        request = TestDataFactory.create_copywriting_input()
        
        start_time = time.time()
        result = service.generate_copy(request)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Validate performance
        assert result is not None
        assert execution_time < 0.1  # Should complete within 100ms
        assert execution_time > 0
    
    def test_batch_processing_performance(self, service, sample_requests):
        """Test performance of batch processing."""
        start_time = time.time()
        results = service.process_batch(sample_requests)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Validate performance
        assert len(results) == len(sample_requests)
        assert execution_time < 0.5  # Should complete within 500ms
        assert execution_time > 0
        
        # Validate all results
        for result in results:
            assert hasattr(result, 'variants')
            assert len(result.variants) > 0
    
    def test_concurrent_processing_performance(self, service):
        """Test performance of concurrent processing."""
        requests = TestDataFactory.create_batch_inputs(5)
        
        start_time = time.time()
        
        # Process requests concurrently
        async def process_concurrent():
            tasks = [service.generate_copy_async(req) for req in requests]
            return await asyncio.gather(*tasks)
        
        results = asyncio.run(process_concurrent())
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Validate performance
        assert len(results) == len(requests)
        assert execution_time < 0.2  # Should complete within 200ms
        assert execution_time > 0
        
        # Validate all results
        for result in results:
            assert hasattr(result, 'variants')
            assert len(result.variants) > 0
    
    def test_memory_usage_performance(self, service):
        """Test memory usage during processing."""
        # Get initial memory usage
        initial_memory = psutil.Process().memory_info().rss
        
        # Create and process multiple requests
        requests = TestDataFactory.create_batch_inputs(100)
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
        assert memory_increase < 50 * 1024 * 1024  # Should not increase by more than 50MB
        assert memory_increase > 0
    
    def test_large_batch_performance(self, service):
        """Test performance with large batch sizes."""
        # Create large batch
        large_batch = TestDataFactory.create_batch_inputs(50)
        
        start_time = time.time()
        results = service.process_batch(large_batch)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Validate performance
        assert len(results) == 50
        assert execution_time < 2.0  # Should complete within 2 seconds
        assert execution_time > 0
        
        # Validate all results
        for result in results:
            assert hasattr(result, 'variants')
            assert len(result.variants) > 0
    
    def test_error_handling_performance(self, service):
        """Test performance of error handling."""
        # Create requests with some invalid data
        valid_requests = TestDataFactory.create_batch_inputs(5)
        invalid_requests = [
            TestDataFactory.create_copywriting_input(product_description=""),  # Invalid
            TestDataFactory.create_copywriting_input(product_description="x" * 3000),  # Too long
        ]
        
        all_requests = valid_requests + invalid_requests
        
        start_time = time.time()
        results = service.process_batch(all_requests)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Validate performance
        assert len(results) == len(all_requests)
        assert execution_time < 0.3  # Should complete within 300ms
        assert execution_time > 0
        
        # Validate results (some should be error responses)
        for result in results:
            assert hasattr(result, 'variants')
            assert len(result.variants) > 0
    
    def test_throughput_performance(self, service):
        """Test throughput performance."""
        # Create batch of requests
        requests = TestDataFactory.create_batch_inputs(20)
        
        start_time = time.time()
        results = service.process_batch(requests)
        end_time = time.time()
        
        execution_time = end_time - start_time
        throughput = len(requests) / execution_time if execution_time > 0 else 0
        
        # Validate performance
        assert len(results) == 20
        assert throughput > 10  # Should process at least 10 requests per second
        assert execution_time > 0
    
    def test_response_time_consistency(self, service):
        """Test consistency of response times."""
        request = TestDataFactory.create_copywriting_input()
        response_times = []
        
        # Run multiple iterations
        for _ in range(10):
            start_time = time.time()
            result = service.generate_copy(request)
            end_time = time.time()
            
            response_times.append(end_time - start_time)
            assert result is not None
        
        # Calculate statistics
        mean_time = statistics.mean(response_times)
        std_dev = statistics.stdev(response_times) if len(response_times) > 1 else 0
        
        # Validate consistency
        assert mean_time < 0.1  # Mean should be under 100ms
        assert std_dev < 0.05  # Standard deviation should be low
        assert all(t > 0 for t in response_times)  # All times should be positive
    
    def test_scalability_performance(self, service):
        """Test scalability with increasing load."""
        batch_sizes = [5, 10, 20, 50]
        execution_times = []
        
        for batch_size in batch_sizes:
            requests = TestDataFactory.create_batch_inputs(batch_size)
            
            start_time = time.time()
            results = service.process_batch(requests)
            end_time = time.time()
            
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            
            # Validate results
            assert len(results) == batch_size
            for result in results:
                assert hasattr(result, 'variants')
                assert len(result.variants) > 0
        
        # Validate scalability (execution time should increase reasonably)
        assert all(t > 0 for t in execution_times)
        assert execution_times[-1] < 5.0  # Largest batch should complete within 5 seconds
    
    def test_resource_cleanup_performance(self, service):
        """Test performance of resource cleanup."""
        # Create and process requests
        requests = TestDataFactory.create_batch_inputs(100)
        results = service.process_batch(requests)
        
        # Get memory before cleanup
        memory_before = psutil.Process().memory_info().rss
        
        # Force garbage collection
        gc.collect()
        
        # Get memory after cleanup
        memory_after = psutil.Process().memory_info().rss
        memory_freed = memory_before - memory_after
        
        # Validate cleanup
        assert len(results) == 100
        assert memory_freed >= 0  # Memory should not increase after cleanup
    
    def test_concurrent_batch_performance(self, service):
        """Test performance of concurrent batch processing."""
        # Create multiple batches
        batches = [TestDataFactory.create_batch_inputs(5) for _ in range(3)]
        
        start_time = time.time()
        
        # Process batches concurrently
        async def process_batches():
            tasks = [service.process_batch_async(batch) for batch in batches]
            return await asyncio.gather(*tasks)
        
        # Mock async batch processing
        async def process_batch_async(batch):
            await asyncio.sleep(0.01)
            return service.process_batch(batch)
        
        # Replace the method temporarily
        service.process_batch_async = process_batch_async
        
        results = asyncio.run(process_batches())
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Validate performance
        assert len(results) == 3
        assert execution_time < 0.3  # Should complete within 300ms
        assert execution_time > 0
        
        # Validate all results
        for batch_results in results:
            assert len(batch_results) == 5
            for result in batch_results:
                assert hasattr(result, 'variants')
                assert len(result.variants) > 0
