"""
Refactored performance and load tests using base classes.
"""
import pytest
import asyncio
import time
import statistics
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import psutil
import os

from tests.base import BaseTestClass, MockAIService, TestAssertions, TestConfig, PerformanceMixin
from agents.backend.onyx.server.features.copywriting.service import CopywritingService
from agents.backend.onyx.server.features.copywriting.models import CopywritingRequest


class TestPerformanceRefactored(BaseTestClass, PerformanceMixin):
    """Refactored performance tests for copywriting service."""
    
    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        return CopywritingService()
    
    @pytest.fixture
    def sample_requests(self):
        """Create multiple sample requests for load testing."""
        return [
            self.create_request(
                product_description=f"Producto deportivo {i}",
                target_platform=["Instagram", "Facebook", "Twitter"][i % 3],
                tone=["inspirational", "informative", "playful"][i % 3]
            )
            for i in range(20)
        ]
    
    @pytest.mark.asyncio
    async def test_single_request_performance(self, service, sample_requests):
        """Test performance of single request generation."""
        request = sample_requests[0]
        mock_ai = MockAIService(delay=0.1)
        
        with patch.object(service, '_call_ai_model', mock_ai.mock_call):
            result, execution_time = await self.measure_async_execution_time(
                service.generate_copywriting(request, "gpt-3.5-turbo")
            )
            
            self.assert_performance_threshold(execution_time, TestConfig.SINGLE_REQUEST_MAX_TIME)
            TestAssertions.assert_valid_copywriting_response(result)
    
    @pytest.mark.asyncio
    async def test_batch_request_performance(self, service, sample_requests):
        """Test performance of batch request generation."""
        batch_request = self.create_batch_request(count=5)
        mock_ai = MockAIService(delay=0.1)
        
        with patch.object(service, 'generate_copywriting') as mock_generate:
            mock_generate.return_value = self.create_response()
            
            result, execution_time = await self.measure_async_execution_time(
                service.batch_generate_copywriting(batch_request)
            )
            
            self.assert_performance_threshold(execution_time, TestConfig.BATCH_REQUEST_MAX_TIME)
            TestAssertions.assert_valid_batch_response(result, 5)
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_performance(self, service, sample_requests):
        """Test performance with concurrent requests."""
        async def generate_single(request):
            mock_ai = MockAIService(delay=0.1)
            with patch.object(service, '_call_ai_model', mock_ai.mock_call):
                return await service.generate_copywriting(request, "gpt-3.5-turbo")
        
        # Run 5 concurrent requests
        tasks = [generate_single(req) for req in sample_requests[:5]]
        
        result, execution_time = await self.measure_async_execution_time(
            asyncio.gather(*tasks)
        )
        
        self.assert_performance_threshold(execution_time, TestConfig.CONCURRENT_REQUEST_MAX_TIME)
        assert len(result) == 5
        
        for response in result:
            TestAssertions.assert_valid_copywriting_response(response)
    
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, service, sample_requests):
        """Test memory usage stability with multiple requests."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate multiple requests
        for i in range(20):
            request = sample_requests[i % len(sample_requests)]
            mock_ai = MockAIService(delay=0.01)
            
            with patch.object(service, '_call_ai_model', mock_ai.mock_call):
                await service.generate_copywriting(request, "gpt-3.5-turbo")
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < TestConfig.MAX_MEMORY_INCREASE_MB
        assert memory_increase >= 0
    
    @pytest.mark.asyncio
    async def test_response_time_consistency(self, service, sample_requests):
        """Test that response times are consistent."""
        response_times = []
        
        for i in range(10):
            request = sample_requests[i % len(sample_requests)]
            mock_ai = MockAIService(delay=0.1)
            
            with patch.object(service, '_call_ai_model', mock_ai.mock_call):
                _, execution_time = await self.measure_async_execution_time(
                    service.generate_copywriting(request, "gpt-3.5-turbo")
                )
                response_times.append(execution_time)
        
        # Calculate statistics
        mean_time = statistics.mean(response_times)
        std_dev = statistics.stdev(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        
        # Response times should be consistent
        assert std_dev < mean_time * 0.5  # Standard deviation should be less than 50% of mean
        assert max_time < mean_time * 2.0  # Max time should be less than 2x mean
        assert min_time > 0.01  # Minimum time should be reasonable
    
    @pytest.mark.asyncio
    async def test_error_handling_performance(self, service, sample_requests):
        """Test that error handling doesn't significantly impact performance."""
        request = sample_requests[0]
        mock_ai = MockAIService(should_fail=True)
        
        with patch.object(service, '_call_ai_model', mock_ai.mock_call):
            start_time = time.time()
            
            with pytest.raises(Exception):
                await service.generate_copywriting(request, "gpt-3.5-turbo")
            
            end_time = time.time()
            error_handling_time = end_time - start_time
            
            # Error handling should be fast
            assert error_handling_time < 1.0
    
    @pytest.mark.asyncio
    async def test_large_batch_performance(self, service):
        """Test performance with large batch (maximum allowed size)."""
        batch_request = self.create_batch_request(count=TestConfig.MAX_BATCH_SIZE)
        mock_ai = MockAIService(delay=0.1)
        
        with patch.object(service, 'generate_copywriting') as mock_generate:
            mock_generate.return_value = self.create_response()
            
            result, execution_time = await self.measure_async_execution_time(
                service.batch_generate_copywriting(batch_request)
            )
            
            self.assert_performance_threshold(execution_time, 30.0)  # 30 seconds for large batch
            TestAssertions.assert_valid_batch_response(result, TestConfig.MAX_BATCH_SIZE)
    
    @pytest.mark.asyncio
    async def test_different_model_performance(self, service, sample_requests):
        """Test performance with different AI models."""
        request = sample_requests[0]
        models = ["gpt-3.5-turbo", "gpt-4", "claude-3"]
        
        for model in models:
            mock_ai = MockAIService(delay=0.1)
            
            with patch.object(service, '_call_ai_model', mock_ai.mock_call):
                result, execution_time = await self.measure_async_execution_time(
                    service.generate_copywriting(request, model)
                )
                
                self.assert_performance_threshold(execution_time, TestConfig.SINGLE_REQUEST_MAX_TIME)
                assert result.model_used == model
                TestAssertions.assert_valid_copywriting_response(result)


class TestLoadTestingRefactored(BaseTestClass, PerformanceMixin):
    """Refactored load testing for copywriting service."""
    
    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        return CopywritingService()
    
    @pytest.mark.asyncio
    async def test_high_concurrency_load(self, service):
        """Test service under high concurrency load."""
        async def worker(worker_id):
            """Worker function for concurrent testing."""
            request = self.create_request(
                product_description=f"Load test product {worker_id}"
            )
            mock_ai = MockAIService(delay=0.1)
            
            try:
                with patch.object(service, '_call_ai_model', mock_ai.mock_call):
                    result = await service.generate_copywriting(request, "gpt-3.5-turbo")
                    return {"success": True, "worker_id": worker_id, "result": result}
            except Exception as e:
                return {"success": False, "worker_id": worker_id, "error": str(e)}
        
        # Run 50 concurrent workers
        num_workers = 50
        tasks = [worker(i) for i in range(num_workers)]
        
        result, execution_time = await self.measure_async_execution_time(
            asyncio.gather(*tasks)
        )
        
        # Count successful and failed requests
        successful = sum(1 for r in result if r.get("success"))
        failed = num_workers - successful
        
        # At least 80% should succeed under load
        success_rate = successful / num_workers
        assert success_rate >= 0.8, f"Success rate {success_rate} is too low"
        
        # Should complete within reasonable time
        self.assert_performance_threshold(execution_time, TestConfig.LOAD_TEST_MAX_TIME)
        
        print(f"Load test results: {successful}/{num_workers} successful ({success_rate:.2%}) in {execution_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_sustained_load(self, service):
        """Test service under sustained load over time."""
        async def sustained_worker():
            """Worker for sustained load testing."""
            for i in range(10):  # 10 requests per worker
                request = self.create_request(
                    product_description=f"Sustained load product {i}"
                )
                mock_ai = MockAIService(delay=0.1)
                
                with patch.object(service, '_call_ai_model', mock_ai.mock_call):
                    await service.generate_copywriting(request, "gpt-3.5-turbo")
        
        # Run 20 workers for sustained load
        num_workers = 20
        tasks = [sustained_worker() for _ in range(num_workers)]
        
        result, execution_time = await self.measure_async_execution_time(
            asyncio.gather(*tasks)
        )
        
        total_requests = num_workers * 10  # 200 total requests
        requests_per_second = total_requests / execution_time
        
        # Sustained load assertions
        self.assert_performance_threshold(execution_time, 120.0)  # 2 minutes
        assert requests_per_second >= 1.0, f"Throughput {requests_per_second} req/s too low"
        assert total_requests == 200, f"Expected 200 requests, got {total_requests}"


class TestScalabilityRefactored(BaseTestClass, PerformanceMixin):
    """Refactored scalability testing for copywriting service."""
    
    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        return CopywritingService()
    
    @pytest.mark.asyncio
    async def test_batch_size_scalability(self, service):
        """Test scalability with increasing batch sizes."""
        batch_sizes = [1, 5, 10, 15, TestConfig.MAX_BATCH_SIZE]
        execution_times = []
        
        for batch_size in batch_sizes:
            batch_request = self.create_batch_request(count=batch_size)
            
            with patch.object(service, 'generate_copywriting') as mock_generate:
                mock_generate.return_value = self.create_response()
                
                _, execution_time = await self.measure_async_execution_time(
                    service.batch_generate_copywriting(batch_request)
                )
                
                execution_times.append(execution_time)
        
        # Scalability assertions
        for i, (batch_size, exec_time) in enumerate(zip(batch_sizes, execution_times)):
            # Execution time should not grow exponentially
            if i > 0:
                time_per_request = exec_time / batch_size
                prev_time_per_request = execution_times[i-1] / batch_sizes[i-1]
                assert time_per_request <= prev_time_per_request * 1.5, \
                    f"Time per request increased too much for batch size {batch_size}"
    
    @pytest.mark.asyncio
    async def test_concurrent_worker_scalability(self, service):
        """Test scalability with increasing concurrent workers."""
        worker_counts = [1, 5, 10, 25, 50]
        throughputs = []
        
        async def worker():
            """Worker function for scalability testing."""
            request = self.create_request()
            mock_ai = MockAIService(delay=0.1)
            
            with patch.object(service, '_call_ai_model', mock_ai.mock_call):
                await service.generate_copywriting(request, "gpt-3.5-turbo")
        
        for worker_count in worker_counts:
            tasks = [worker() for _ in range(worker_count)]
            
            _, execution_time = await self.measure_async_execution_time(
                asyncio.gather(*tasks)
            )
            
            throughput = worker_count / execution_time
            throughputs.append(throughput)
        
        # Scalability assertions
        for i, (worker_count, throughput) in enumerate(zip(worker_counts, throughputs)):
            # Throughput should generally increase with more workers
            if i > 0:
                assert throughput >= throughputs[i-1] * 0.8, \
                    f"Throughput decreased too much for {worker_count} workers"


class TestPerformanceRegressionRefactored(BaseTestClass, PerformanceMixin):
    """Refactored performance regression testing."""
    
    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        return CopywritingService()
    
    @pytest.fixture
    def baseline_metrics(self):
        """Baseline performance metrics for regression testing."""
        return {
            "single_request_max_time": TestConfig.SINGLE_REQUEST_MAX_TIME,
            "batch_request_max_time": TestConfig.BATCH_REQUEST_MAX_TIME,
            "concurrent_max_time": TestConfig.CONCURRENT_REQUEST_MAX_TIME,
            "memory_max_increase": TestConfig.MAX_MEMORY_INCREASE_MB,
            "response_time_std_dev_ratio": 0.5,
            "min_success_rate": 0.95
        }
    
    @pytest.mark.asyncio
    async def test_single_request_regression(self, service, baseline_metrics):
        """Test single request performance regression."""
        request = self.create_request()
        mock_ai = MockAIService(delay=0.1)
        
        with patch.object(service, '_call_ai_model', mock_ai.mock_call):
            _, execution_time = await self.measure_async_execution_time(
                service.generate_copywriting(request, "gpt-3.5-turbo")
            )
            
            # Regression check
            assert execution_time <= baseline_metrics["single_request_max_time"], \
                f"Single request time {execution_time}s exceeds baseline {baseline_metrics['single_request_max_time']}s"
    
    @pytest.mark.asyncio
    async def test_batch_request_regression(self, service, baseline_metrics):
        """Test batch request performance regression."""
        batch_request = self.create_batch_request(count=5)
        
        with patch.object(service, 'generate_copywriting') as mock_generate:
            mock_generate.return_value = self.create_response()
            
            _, execution_time = await self.measure_async_execution_time(
                service.batch_generate_copywriting(batch_request)
            )
            
            # Regression check
            assert execution_time <= baseline_metrics["batch_request_max_time"], \
                f"Batch request time {execution_time}s exceeds baseline {baseline_metrics['batch_request_max_time']}s"
    
    @pytest.mark.asyncio
    async def test_memory_usage_regression(self, service, baseline_metrics):
        """Test memory usage regression."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple requests
        for i in range(100):
            request = self.create_request(
                product_description=f"Memory test product {i}"
            )
            mock_ai = MockAIService(delay=0.01)
            
            with patch.object(service, '_call_ai_model', mock_ai.mock_call):
                await service.generate_copywriting(request, "gpt-3.5-turbo")
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Regression check
        assert memory_increase <= baseline_metrics["memory_max_increase"], \
            f"Memory increase {memory_increase}MB exceeds baseline {baseline_metrics['memory_max_increase']}MB"
