"""
Simple performance benchmarks for copywriting service.
"""
import pytest
import time
import asyncio
import statistics
from typing import List, Dict, Any
from unittest.mock import patch, Mock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_utils import TestDataFactory, MockAIService, TestAssertions, TestConfig
from models import CopywritingInput, CopywritingOutput


class TestPerformanceBenchmarks:
    """Performance benchmarks for copywriting service."""
    
    @pytest.fixture
    def sample_requests(self):
        """Create sample requests for benchmarking."""
        return TestDataFactory.create_batch_inputs(10)
    
    @pytest.mark.benchmark
    def test_model_creation_performance(self, sample_requests):
        """Benchmark model creation performance."""
        start_time = time.time()
        
        # Test CopywritingInput creation
        for i in range(100):
            TestDataFactory.create_copywriting_input(
                product_description=f"Test product {i}",
                target_platform="instagram",
                tone="inspirational"
            )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should create 100 models in less than 1 second
        TestAssertions.assert_performance_threshold(execution_time, 1.0)
        print(f"Created 100 CopywritingInput models in {execution_time:.3f}s")
    
    @pytest.mark.benchmark
    def test_model_serialization_performance(self, sample_requests):
        """Benchmark model serialization performance."""
        request = sample_requests[0]
        
        start_time = time.time()
        
        # Test JSON serialization
        for i in range(100):
            json_str = request.model_dump_json()
            assert isinstance(json_str, str)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should serialize 100 models in less than 0.5 seconds
        TestAssertions.assert_performance_threshold(execution_time, 0.5)
        print(f"Serialized 100 models in {execution_time:.3f}s")
    
    @pytest.mark.benchmark
    def test_model_validation_performance(self, sample_requests):
        """Benchmark model validation performance."""
        start_time = time.time()
        
        # Test validation with various inputs
        for i in range(100):
            try:
                TestDataFactory.create_copywriting_input(
                    product_description=f"Test product {i}",
                    target_platform="instagram",
                    tone="inspirational",
                    use_case="product_launch",
                    content_type="social_post"
                )
            except Exception as e:
                pytest.fail(f"Validation failed: {e}")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should validate 100 models in less than 0.3 seconds
        TestAssertions.assert_performance_threshold(execution_time, 0.3)
        print(f"Validated 100 models in {execution_time:.3f}s")
    
    @pytest.mark.benchmark
    def test_batch_processing_performance(self, sample_requests):
        """Benchmark batch processing performance."""
        start_time = time.time()
        
        # Process batch of requests
        results = []
        for request in sample_requests:
            # Simulate processing
            result = {
                "input_id": request.tracking_id,
                "processed_at": time.time(),
                "status": "success"
            }
            results.append(result)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should process 10 requests in less than 0.1 seconds
        TestAssertions.assert_performance_threshold(execution_time, 0.1)
        assert len(results) == len(sample_requests)
        print(f"Processed {len(sample_requests)} requests in {execution_time:.3f}s")
    
    @pytest.mark.benchmark
    def test_memory_usage_performance(self, sample_requests):
        """Benchmark memory usage performance."""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create many objects
        objects = []
        for i in range(1000):
            obj = TestDataFactory.create_copywriting_input(
                product_description=f"Test product {i}",
                target_platform="instagram",
                tone="inspirational"
            )
            objects.append(obj)
        
        # Get memory usage after creation
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = peak_memory - initial_memory
        
        # Clean up
        del objects
        gc.collect()
        
        # Should use less than 50MB for 1000 objects
        assert memory_used < 50, f"Memory usage {memory_used:.1f}MB exceeds threshold 50MB"
        print(f"Used {memory_used:.1f}MB for 1000 objects")
    
    @pytest.mark.benchmark
    def test_concurrent_processing_performance(self, sample_requests):
        """Benchmark concurrent processing performance."""
        async def process_request(request):
            """Process a single request asynchronously."""
            await asyncio.sleep(0.001)  # Simulate processing time
            return {
                "input_id": request.tracking_id,
                "processed_at": time.time(),
                "status": "success"
            }
        
        async def run_concurrent_test():
            """Run concurrent processing test."""
            start_time = time.time()
            
            # Process all requests concurrently
            tasks = [process_request(req) for req in sample_requests]
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            return results, execution_time
        
        # Run the async test
        results, execution_time = asyncio.run(run_concurrent_test())
        
        # Should process 10 requests concurrently in less than 0.1 seconds
        TestAssertions.assert_performance_threshold(execution_time, 0.1)
        assert len(results) == len(sample_requests)
        print(f"Processed {len(sample_requests)} requests concurrently in {execution_time:.3f}s")
    
    @pytest.mark.benchmark
    def test_error_handling_performance(self):
        """Benchmark error handling performance."""
        start_time = time.time()
        
        # Test error handling with invalid inputs
        error_count = 0
        for i in range(100):
            try:
                # This should fail validation
                TestDataFactory.create_copywriting_input(
                    product_description="",  # Invalid empty description
                    target_platform="invalid_platform",  # Invalid platform
                    tone="invalid_tone",  # Invalid tone
                    use_case="invalid_use_case",  # Invalid use case
                    content_type="invalid_type"  # Invalid content type
                )
            except Exception:
                error_count += 1
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should handle 100 errors in less than 0.5 seconds
        TestAssertions.assert_performance_threshold(execution_time, 0.5)
        assert error_count == 100  # All should fail
        print(f"Handled {error_count} errors in {execution_time:.3f}s")
    
    @pytest.mark.benchmark
    def test_large_data_processing_performance(self):
        """Benchmark large data processing performance."""
        # Create large input data (within model limits)
        large_key_points = [f"Key point {i}" for i in range(15)]  # Max 15 key points
        large_instructions = "This is a very long instruction. " * 20  # Within max length
        
        start_time = time.time()
        
        # Create request with large data
        request = TestDataFactory.create_copywriting_input(
            product_description="Test product with large data",
            target_platform="instagram",
            tone="inspirational",
            key_points=large_key_points,
            instructions=large_instructions
        )
        
        # Serialize and deserialize
        json_str = request.model_dump_json()
        parsed_data = request.model_validate_json(json_str)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should handle large data in less than 0.2 seconds
        TestAssertions.assert_performance_threshold(execution_time, 0.2)
        assert len(parsed_data.key_points) == 15
        print(f"Processed large data in {execution_time:.3f}s")


class TestRegressionBenchmarks:
    """Regression benchmarks to detect performance degradation."""
    
    def test_model_creation_regression(self):
        """Test that model creation performance hasn't regressed."""
        start_time = time.time()
        
        # Create 1000 models
        for i in range(1000):
            TestDataFactory.create_copywriting_input(
                product_description=f"Test product {i}",
                target_platform="instagram",
                tone="inspirational"
            )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Regression threshold: should not take more than 2 seconds
        assert execution_time < 2.0, f"Model creation regression: {execution_time:.3f}s > 2.0s"
        print(f"Model creation regression test passed: {execution_time:.3f}s")
    
    def test_serialization_regression(self):
        """Test that serialization performance hasn't regressed."""
        request = TestDataFactory.create_copywriting_input(
            product_description="Test product",
            target_platform="instagram",
            tone="inspirational"
        )
        
        start_time = time.time()
        
        # Serialize 1000 times
        for i in range(1000):
            json_str = request.model_dump_json()
            assert isinstance(json_str, str)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Regression threshold: should not take more than 1 second
        assert execution_time < 1.0, f"Serialization regression: {execution_time:.3f}s > 1.0s"
        print(f"Serialization regression test passed: {execution_time:.3f}s")
    
    def test_validation_regression(self):
        """Test that validation performance hasn't regressed."""
        start_time = time.time()
        
        # Validate 1000 models
        for i in range(1000):
            try:
                TestDataFactory.create_copywriting_input(
                    product_description=f"Test product {i}",
                    target_platform="instagram",
                    tone="inspirational"
                )
            except Exception as e:
                pytest.fail(f"Validation regression: {e}")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Regression threshold: should not take more than 0.5 seconds
        assert execution_time < 0.5, f"Validation regression: {execution_time:.3f}s > 0.5s"
        print(f"Validation regression test passed: {execution_time:.3f}s")
