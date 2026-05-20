"""
Performance benchmarks and regression testing for copywriting service.
"""
import pytest
import time
import asyncio
import statistics
from typing import List, Dict, Any
from unittest.mock import patch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_utils import TestDataFactory, MockAIService
from service import CopywritingService
from models import (
    CopywritingInput,
    CopywritingOutput
)


class TestPerformanceBenchmarks:
    """Performance benchmarks for copywriting service."""
    
    @pytest.fixture
    def service(self):
        """Create service instance for benchmarking."""
        return CopywritingService()
    
    @pytest.fixture
    def sample_requests(self):
        """Create sample requests for benchmarking."""
        return [
            TestDataFactory.create_copywriting_input(
                product_description=f"Producto de prueba {i}",
                target_platform=["instagram", "facebook", "twitter"][i % 3],
                tone=["inspirational", "informative", "playful"][i % 3]
            )
            for i in range(20)
        ]
    
    @pytest.mark.benchmark
    def test_single_request_benchmark(self, service, sample_requests):
        """Benchmark single request performance."""
        request = sample_requests[0]
        
        with patch.object(service, '_call_ai_model') as mock_call:
            mock_call.return_value = {
                "variants": [{"headline": "Test", "primary_text": "Content"}],
                "model_used": "gpt-3.5-turbo",
                "generation_time": 0.1,
                "extra_metadata": {}
            }
            
            # Measure execution time
            start_time = time.time()
            result = asyncio.run(service.generate_copywriting(request, "gpt-3.5-turbo"))
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Performance assertions
            assert execution_time < 1.0  # Should complete within 1 second
            assert result is not None
            assert len(result.variants) > 0
    
    @pytest.mark.benchmark
    def test_batch_request_benchmark(self, service, sample_requests):
        """Benchmark batch request performance."""
        batch_request = BatchCopywritingRequest(requests=sample_requests[:5])
        
        with patch.object(service, 'generate_copywriting') as mock_generate:
            mock_generate.return_value = TestDataFactory.create_sample_response()
            
            # Measure execution time
            start_time = time.time()
            result = asyncio.run(service.batch_generate_copywriting(batch_request))
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Performance assertions
            assert execution_time < 5.0  # Should complete within 5 seconds
            assert len(result.results) == 5
            assert mock_generate.call_count == 5
    
    @pytest.mark.benchmark
    def test_concurrent_requests_benchmark(self, service, sample_requests):
        """Benchmark concurrent request handling."""
        async def process_request(request):
            with patch.object(service, '_call_ai_model') as mock_call:
                mock_call.return_value = {
                    "variants": [{"headline": "Test", "primary_text": "Content"}],
                    "model_used": "gpt-3.5-turbo",
                    "generation_time": 0.1,
                    "extra_metadata": {}
                }
                return await service.generate_copywriting(request, "gpt-3.5-turbo")
        
        # Process 10 requests concurrently
        requests_to_process = sample_requests[:10]
        
        start_time = time.time()
        results = asyncio.run(asyncio.gather(*[process_request(req) for req in requests_to_process]))
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Performance assertions
        assert execution_time < 10.0  # Should complete within 10 seconds
        assert len(results) == 10
        assert all(result is not None for result in results)
    
    @pytest.mark.benchmark
    def test_memory_usage_benchmark(self, service, sample_requests):
        """Benchmark memory usage under load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple requests
        with patch.object(service, '_call_ai_model') as mock_call:
            mock_call.return_value = {
                "variants": [{"headline": "Test", "primary_text": "Content"}],
                "model_used": "gpt-3.5-turbo",
                "generation_time": 0.1,
                "extra_metadata": {}
            }
            
            for request in sample_requests[:50]:
                asyncio.run(service.generate_copywriting(request, "gpt-3.5-turbo"))
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory usage assertions
        assert memory_increase < 100.0  # Should not use more than 100MB
        assert memory_increase >= 0  # Memory should not decrease
    
    @pytest.mark.benchmark
    def test_response_time_consistency_benchmark(self, service, sample_requests):
        """Benchmark response time consistency."""
        response_times = []
        
        with patch.object(service, '_call_ai_model') as mock_call:
            mock_call.return_value = {
                "variants": [{"headline": "Test", "primary_text": "Content"}],
                "model_used": "gpt-3.5-turbo",
                "generation_time": 0.1,
                "extra_metadata": {}
            }
            
            # Measure response times for multiple requests
            for request in sample_requests[:20]:
                start_time = time.time()
                asyncio.run(service.generate_copywriting(request, "gpt-3.5-turbo"))
                end_time = time.time()
                response_times.append(end_time - start_time)
        
        # Calculate statistics
        mean_time = statistics.mean(response_times)
        std_dev = statistics.stdev(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        
        # Consistency assertions
        assert std_dev < mean_time * 0.5  # Standard deviation should be less than 50% of mean
        assert max_time < mean_time * 2.0  # Max time should be less than 2x mean
        assert min_time > 0.01  # Minimum time should be reasonable
        assert mean_time < 1.0  # Average time should be fast


class TestRegressionBenchmarks:
    """Regression testing benchmarks to detect performance degradation."""
    
    @pytest.fixture
    def baseline_metrics(self):
        """Baseline performance metrics for regression testing."""
        return {
            "single_request_max_time": 1.0,  # seconds
            "batch_request_max_time": 5.0,   # seconds
            "concurrent_max_time": 10.0,     # seconds
            "memory_max_increase": 100.0,    # MB
            "response_time_std_dev_ratio": 0.5,  # std_dev / mean
            "min_success_rate": 0.95  # 95% success rate
        }
    
    @pytest.mark.regression
    def test_single_request_regression(self, baseline_metrics):
        """Test single request performance regression."""
        service = CopywritingService()
        request = TestDataFactory.create_sample_request()
        
        with patch.object(service, '_call_ai_model') as mock_call:
            mock_call.return_value = {
                "variants": [{"headline": "Test", "primary_text": "Content"}],
                "model_used": "gpt-3.5-turbo",
                "generation_time": 0.1,
                "extra_metadata": {}
            }
            
            start_time = time.time()
            result = asyncio.run(service.generate_copywriting(request, "gpt-3.5-turbo"))
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Regression check
            assert execution_time <= baseline_metrics["single_request_max_time"], \
                f"Single request time {execution_time}s exceeds baseline {baseline_metrics['single_request_max_time']}s"
    
    @pytest.mark.regression
    def test_batch_request_regression(self, baseline_metrics):
        """Test batch request performance regression."""
        service = CopywritingService()
        requests = [TestDataFactory.create_sample_request() for _ in range(5)]
        batch_request = BatchCopywritingRequest(requests=requests)
        
        with patch.object(service, 'generate_copywriting') as mock_generate:
            mock_generate.return_value = TestDataFactory.create_sample_response()
            
            start_time = time.time()
            result = asyncio.run(service.batch_generate_copywriting(batch_request))
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Regression check
            assert execution_time <= baseline_metrics["batch_request_max_time"], \
                f"Batch request time {execution_time}s exceeds baseline {baseline_metrics['batch_request_max_time']}s"
    
    @pytest.mark.regression
    def test_memory_usage_regression(self, baseline_metrics):
        """Test memory usage regression."""
        import psutil
        import os
        
        service = CopywritingService()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with patch.object(service, '_call_ai_model') as mock_call:
            mock_call.return_value = {
                "variants": [{"headline": "Test", "primary_text": "Content"}],
                "model_used": "gpt-3.5-turbo",
                "generation_time": 0.1,
                "extra_metadata": {}
            }
            
            # Process multiple requests
            for i in range(100):
                request = TestDataFactory.create_sample_request(
                    product_description=f"Producto {i}"
                )
                asyncio.run(service.generate_copywriting(request, "gpt-3.5-turbo"))
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Regression check
        assert memory_increase <= baseline_metrics["memory_max_increase"], \
            f"Memory increase {memory_increase}MB exceeds baseline {baseline_metrics['memory_max_increase']}MB"
    
    @pytest.mark.regression
    def test_response_time_consistency_regression(self, baseline_metrics):
        """Test response time consistency regression."""
        service = CopywritingService()
        response_times = []
        
        with patch.object(service, '_call_ai_model') as mock_call:
            mock_call.return_value = {
                "variants": [{"headline": "Test", "primary_text": "Content"}],
                "model_used": "gpt-3.5-turbo",
                "generation_time": 0.1,
                "extra_metadata": {}
            }
            
            # Measure response times
            for i in range(30):
                request = TestDataFactory.create_sample_request(
                    product_description=f"Producto {i}"
                )
                start_time = time.time()
                asyncio.run(service.generate_copywriting(request, "gpt-3.5-turbo"))
                end_time = time.time()
                response_times.append(end_time - start_time)
        
        # Calculate consistency metrics
        mean_time = statistics.mean(response_times)
        std_dev = statistics.stdev(response_times)
        consistency_ratio = std_dev / mean_time if mean_time > 0 else 0
        
        # Regression check
        assert consistency_ratio <= baseline_metrics["response_time_std_dev_ratio"], \
            f"Response time consistency {consistency_ratio} exceeds baseline {baseline_metrics['response_time_std_dev_ratio']}"


class TestLoadBenchmarks:
    """Load testing benchmarks for high-traffic scenarios."""
    
    @pytest.mark.load
    def test_high_concurrency_load(self):
        """Test service under high concurrency load."""
        service = CopywritingService()
        
        async def worker(worker_id):
            """Worker function for load testing."""
            request = TestDataFactory.create_sample_request(
                product_description=f"Load test product {worker_id}"
            )
            
            try:
                with patch.object(service, '_call_ai_model') as mock_call:
                    mock_call.return_value = {
                        "variants": [{"headline": "Test", "primary_text": "Content"}],
                        "model_used": "gpt-3.5-turbo",
                        "generation_time": 0.1,
                        "extra_metadata": {}
                    }
                    result = await service.generate_copywriting(request, "gpt-3.5-turbo")
                    return {"success": True, "worker_id": worker_id, "result": result}
            except Exception as e:
                return {"success": False, "worker_id": worker_id, "error": str(e)}
        
        # Run 100 concurrent workers
        num_workers = 100
        start_time = time.time()
        results = asyncio.run(asyncio.gather(*[worker(i) for i in range(num_workers)]))
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Count successful and failed requests
        successful = sum(1 for r in results if r.get("success"))
        failed = num_workers - successful
        success_rate = successful / num_workers
        
        # Load test assertions
        assert success_rate >= 0.95, f"Success rate {success_rate} below 95%"
        assert total_time < 60.0, f"Load test took too long: {total_time}s"
        assert successful >= 95, f"Only {successful} out of {num_workers} requests succeeded"
    
    @pytest.mark.load
    def test_sustained_load(self):
        """Test service under sustained load over time."""
        service = CopywritingService()
        
        async def sustained_worker():
            """Worker for sustained load testing."""
            for i in range(10):  # 10 requests per worker
                request = TestDataFactory.create_sample_request(
                    product_description=f"Sustained load product {i}"
                )
                
                with patch.object(service, '_call_ai_model') as mock_call:
                    mock_call.return_value = {
                        "variants": [{"headline": "Test", "primary_text": "Content"}],
                        "model_used": "gpt-3.5-turbo",
                        "generation_time": 0.1,
                        "extra_metadata": {}
                    }
                    await service.generate_copywriting(request, "gpt-3.5-turbo")
        
        # Run 20 workers for sustained load
        num_workers = 20
        start_time = time.time()
        asyncio.run(asyncio.gather(*[sustained_worker() for _ in range(num_workers)]))
        end_time = time.time()
        
        total_time = end_time - start_time
        total_requests = num_workers * 10  # 200 total requests
        requests_per_second = total_requests / total_time
        
        # Sustained load assertions
        assert total_time < 120.0, f"Sustained load test took too long: {total_time}s"
        assert requests_per_second >= 1.0, f"Throughput {requests_per_second} req/s too low"
        assert total_requests == 200, f"Expected 200 requests, got {total_requests}"


class TestScalabilityBenchmarks:
    """Scalability testing benchmarks."""
    
    @pytest.mark.scalability
    def test_batch_size_scalability(self):
        """Test scalability with increasing batch sizes."""
        service = CopywritingService()
        batch_sizes = [1, 5, 10, 15, 20]  # Maximum allowed batch size
        execution_times = []
        
        for batch_size in batch_sizes:
            requests = [TestDataFactory.create_sample_request() for _ in range(batch_size)]
            batch_request = BatchCopywritingRequest(requests=requests)
            
            with patch.object(service, 'generate_copywriting') as mock_generate:
                mock_generate.return_value = TestDataFactory.create_sample_response()
                
                start_time = time.time()
                asyncio.run(service.batch_generate_copywriting(batch_request))
                end_time = time.time()
                
                execution_times.append(end_time - start_time)
        
        # Scalability assertions
        for i, (batch_size, exec_time) in enumerate(zip(batch_sizes, execution_times)):
            # Execution time should not grow exponentially
            if i > 0:
                time_per_request = exec_time / batch_size
                prev_time_per_request = execution_times[i-1] / batch_sizes[i-1]
                assert time_per_request <= prev_time_per_request * 1.5, \
                    f"Time per request increased too much for batch size {batch_size}"
    
    @pytest.mark.scalability
    def test_concurrent_worker_scalability(self):
        """Test scalability with increasing concurrent workers."""
        service = CopywritingService()
        worker_counts = [1, 5, 10, 25, 50]
        throughputs = []
        
        async def worker():
            """Worker function for scalability testing."""
            request = TestDataFactory.create_sample_request()
            
            with patch.object(service, '_call_ai_model') as mock_call:
                mock_call.return_value = {
                    "variants": [{"headline": "Test", "primary_text": "Content"}],
                    "model_used": "gpt-3.5-turbo",
                    "generation_time": 0.1,
                    "extra_metadata": {}
                }
                await service.generate_copywriting(request, "gpt-3.5-turbo")
        
        for worker_count in worker_counts:
            start_time = time.time()
            asyncio.run(asyncio.gather(*[worker() for _ in range(worker_count)]))
            end_time = time.time()
            
            total_time = end_time - start_time
            throughput = worker_count / total_time
            throughputs.append(throughput)
        
        # Scalability assertions
        for i, (worker_count, throughput) in enumerate(zip(worker_counts, throughputs)):
            # Throughput should generally increase with more workers
            if i > 0:
                assert throughput >= throughputs[i-1] * 0.8, \
                    f"Throughput decreased too much for {worker_count} workers"
