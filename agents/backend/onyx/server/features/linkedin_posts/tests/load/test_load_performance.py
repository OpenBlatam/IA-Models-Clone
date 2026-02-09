"""
Load and Performance Tests for LinkedIn Posts
============================================

Load tests to verify system performance under high load,
stress tests to find breaking points, and performance benchmarks.
"""

import pytest
import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# Import components for load testing
from ...services.post_service import PostService, PostRepository, AIService, CacheService
from ...core.entities import (
    LinkedInPost, PostContent, PostGenerationRequest, PostGenerationResponse,
    PostOptimizationResult, PostValidationResult, PostType, PostTone, PostStatus,
    EngagementMetrics, ContentAnalysisResult
)


class TestLoadPerformance:
    """Load and performance tests for LinkedIn Posts system."""

    @pytest.fixture
    def mock_services(self):
        """Create mocked services for load testing."""
        mock_repository = AsyncMock(spec=PostRepository)
        mock_ai_service = AsyncMock(spec=AIService)
        mock_cache_service = AsyncMock(spec=CacheService)
        
        # Configure realistic response times
        async def delayed_response(delay: float = 0.1):
            await asyncio.sleep(delay)
            return True
        
        mock_repository.createPost.side_effect = lambda post: delayed_response(0.05)
        mock_repository.getPost.side_effect = lambda post_id: delayed_response(0.02)
        mock_repository.listPosts.side_effect = lambda user_id, filters=None: delayed_response(0.03)
        mock_ai_service.generatePost.side_effect = lambda request: delayed_response(0.2)
        mock_ai_service.optimizePost.side_effect = lambda post: delayed_response(0.3)
        mock_cache_service.get.side_effect = lambda key: delayed_response(0.01)
        
        return PostService(mock_repository, mock_ai_service, mock_cache_service)

    @pytest.fixture
    def sample_request(self) -> PostGenerationRequest:
        """Sample request for load testing."""
        return PostGenerationRequest(
            topic="Load Test Post",
            keyPoints=["Point 1", "Point 2"],
            targetAudience="Test audience",
            industry="Technology",
            tone=PostTone.PROFESSIONAL,
            postType=PostType.TEXT,
            keywords=["load", "test", "performance"]
        )

    @pytest.mark.asyncio
    async def test_concurrent_post_creation_load(self, mock_services: PostService, sample_request: PostGenerationRequest):
        """Test concurrent post creation under load."""
        num_concurrent_requests = 50
        start_time = time.time()
        
        # Create concurrent tasks
        tasks = [
            mock_services.createPost(sample_request)
            for _ in range(num_concurrent_requests)
        ]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        successful_requests = sum(1 for r in results if not isinstance(r, Exception))
        failed_requests = len(results) - successful_requests
        requests_per_second = len(results) / total_time
        
        # Assertions
        assert successful_requests > 0
        assert requests_per_second > 10  # Should handle at least 10 RPS
        assert total_time < 30  # Should complete within 30 seconds
        
        print(f"Load Test Results:")
        print(f"  Total requests: {len(results)}")
        print(f"  Successful: {successful_requests}")
        print(f"  Failed: {failed_requests}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Requests per second: {requests_per_second:.2f}")

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, mock_services: PostService, sample_request: PostGenerationRequest):
        """Test memory usage under load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create many requests
        num_requests = 100
        tasks = [
            mock_services.createPost(sample_request)
            for _ in range(num_requests)
        ]
        
        # Execute requests
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory Usage Test:")
        print(f"  Initial memory: {initial_memory:.2f} MB")
        print(f"  Final memory: {final_memory:.2f} MB")
        print(f"  Memory increase: {memory_increase:.2f} MB")
        
        # Memory increase should be reasonable
        assert memory_increase < 100  # Should not increase by more than 100MB

    @pytest.mark.asyncio
    async def test_response_time_distribution(self, mock_services: PostService, sample_request: PostGenerationRequest):
        """Test response time distribution under load."""
        num_requests = 100
        response_times = []
        
        for _ in range(num_requests):
            start_time = time.time()
            await mock_services.createPost(sample_request)
            end_time = time.time()
            response_times.append(end_time - start_time)
        
        # Calculate statistics
        avg_response_time = statistics.mean(response_times)
        median_response_time = statistics.median(response_times)
        p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        p99_response_time = statistics.quantiles(response_times, n=100)[98]  # 99th percentile
        
        print(f"Response Time Distribution:")
        print(f"  Average: {avg_response_time:.3f}s")
        print(f"  Median: {median_response_time:.3f}s")
        print(f"  95th percentile: {p95_response_time:.3f}s")
        print(f"  99th percentile: {p99_response_time:.3f}s")
        
        # Assertions
        assert avg_response_time < 1.0  # Average should be under 1 second
        assert p95_response_time < 2.0  # 95% should be under 2 seconds
        assert p99_response_time < 5.0  # 99% should be under 5 seconds

    @pytest.mark.asyncio
    async def test_cache_performance_under_load(self, mock_services: PostService, sample_request: PostGenerationRequest):
        """Test cache performance under load."""
        # First request (cache miss)
        start_time = time.time()
        await mock_services.createPost(sample_request)
        cache_miss_time = time.time() - start_time
        
        # Second request (cache hit)
        start_time = time.time()
        await mock_services.createPost(sample_request)
        cache_hit_time = time.time() - start_time
        
        # Cache hit should be significantly faster
        performance_improvement = cache_miss_time / cache_hit_time
        
        print(f"Cache Performance Test:")
        print(f"  Cache miss time: {cache_miss_time:.3f}s")
        print(f"  Cache hit time: {cache_hit_time:.3f}s")
        print(f"  Performance improvement: {performance_improvement:.2f}x")
        
        assert cache_hit_time < cache_miss_time
        assert performance_improvement > 2.0  # Should be at least 2x faster

    @pytest.mark.asyncio
    async def test_database_connection_pool_under_load(self, mock_services: PostService, sample_request: PostGenerationRequest):
        """Test database connection pool under load."""
        num_concurrent_requests = 100
        
        # Simulate database connection pool exhaustion
        async def create_post_with_delay():
            await asyncio.sleep(0.01)  # Simulate DB connection time
            return await mock_services.createPost(sample_request)
        
        start_time = time.time()
        tasks = [create_post_with_delay() for _ in range(num_concurrent_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        successful_requests = sum(1 for r in results if not isinstance(r, Exception))
        total_time = end_time - start_time
        
        print(f"Database Connection Pool Test:")
        print(f"  Total requests: {num_concurrent_requests}")
        print(f"  Successful: {successful_requests}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Requests per second: {num_concurrent_requests / total_time:.2f}")
        
        assert successful_requests > 0
        assert total_time < 60  # Should complete within 60 seconds

    @pytest.mark.asyncio
    async def test_ai_service_bottleneck_simulation(self, mock_services: PostService, sample_request: PostGenerationRequest):
        """Test system behavior when AI service is the bottleneck."""
        # Simulate slow AI service
        original_generate_post = mock_services.ai_service.generatePost
        
        async def slow_generate_post(request):
            await asyncio.sleep(0.5)  # Simulate slow AI processing
            return await original_generate_post(request)
        
        mock_services.ai_service.generatePost = slow_generate_post
        
        num_requests = 20
        start_time = time.time()
        
        tasks = [
            mock_services.createPost(sample_request)
            for _ in range(num_requests)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        successful_requests = sum(1 for r in results if not isinstance(r, Exception))
        total_time = end_time - start_time
        
        print(f"AI Service Bottleneck Test:")
        print(f"  Total requests: {num_requests}")
        print(f"  Successful: {successful_requests}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average time per request: {total_time / num_requests:.2f}s")
        
        assert successful_requests > 0
        assert total_time > 5  # Should take longer due to AI bottleneck

    @pytest.mark.asyncio
    async def test_error_rate_under_load(self, mock_services: PostService, sample_request: PostGenerationRequest):
        """Test error rate under high load."""
        # Simulate intermittent failures
        call_count = 0
        
        async def create_post_with_failures(request):
            nonlocal call_count
            call_count += 1
            
            # Simulate 10% failure rate
            if call_count % 10 == 0:
                raise Exception("Simulated failure")
            
            return await mock_services.createPost(request)
        
        num_requests = 100
        results = []
        
        for _ in range(num_requests):
            try:
                result = await create_post_with_failures(sample_request)
                results.append(("success", result))
            except Exception as e:
                results.append(("error", str(e)))
        
        successful_requests = sum(1 for r in results if r[0] == "success")
        failed_requests = sum(1 for r in results if r[0] == "error")
        error_rate = failed_requests / num_requests
        
        print(f"Error Rate Test:")
        print(f"  Total requests: {num_requests}")
        print(f"  Successful: {successful_requests}")
        print(f"  Failed: {failed_requests}")
        print(f"  Error rate: {error_rate:.2%}")
        
        assert error_rate > 0  # Should have some failures
        assert error_rate < 0.2  # Should not exceed 20% error rate

    @pytest.mark.asyncio
    async def test_throughput_scalability(self, mock_services: PostService, sample_request: PostGenerationRequest):
        """Test throughput scalability with different load levels."""
        load_levels = [10, 25, 50, 100]
        throughput_results = []
        
        for load in load_levels:
            start_time = time.time()
            
            tasks = [
                mock_services.createPost(sample_request)
                for _ in range(load)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            successful_requests = sum(1 for r in results if not isinstance(r, Exception))
            total_time = end_time - start_time
            throughput = successful_requests / total_time
            
            throughput_results.append({
                'load': load,
                'throughput': throughput,
                'total_time': total_time,
                'successful': successful_requests
            })
        
        print(f"Throughput Scalability Test:")
        for result in throughput_results:
            print(f"  Load {result['load']}: {result['throughput']:.2f} RPS, "
                  f"{result['successful']}/{result['load']} successful, "
                  f"{result['total_time']:.2f}s")
        
        # Throughput should generally increase with load (up to a point)
        assert len(throughput_results) == len(load_levels)
        assert all(r['successful'] > 0 for r in throughput_results)

    @pytest.mark.asyncio
    async def test_resource_utilization_monitoring(self, mock_services: PostService, sample_request: PostGenerationRequest):
        """Test resource utilization monitoring under load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Monitor resources during load test
        num_requests = 50
        cpu_usage = []
        memory_usage = []
        
        for i in range(num_requests):
            # Record resource usage before request
            cpu_percent = process.cpu_percent()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            cpu_usage.append(cpu_percent)
            memory_usage.append(memory_mb)
            
            # Make request
            await mock_services.createPost(sample_request)
            
            # Small delay to allow resource monitoring
            await asyncio.sleep(0.01)
        
        avg_cpu = statistics.mean(cpu_usage)
        max_cpu = max(cpu_usage)
        avg_memory = statistics.mean(memory_usage)
        max_memory = max(memory_usage)
        
        print(f"Resource Utilization Test:")
        print(f"  Average CPU: {avg_cpu:.1f}%")
        print(f"  Max CPU: {max_cpu:.1f}%")
        print(f"  Average Memory: {avg_memory:.1f} MB")
        print(f"  Max Memory: {max_memory:.1f} MB")
        
        # Resource usage should be reasonable
        assert avg_cpu < 80  # Average CPU should be under 80%
        assert max_cpu < 95  # Max CPU should be under 95%
        assert avg_memory < 500  # Average memory should be under 500MB

    @pytest.mark.asyncio
    async def test_stress_test_breaking_point(self, mock_services: PostService, sample_request: PostGenerationRequest):
        """Test system behavior at breaking point."""
        # Gradually increase load until system breaks
        load_levels = [10, 25, 50, 100, 200, 500, 1000]
        breaking_point = None
        
        for load in load_levels:
            try:
                start_time = time.time()
                
                tasks = [
                    mock_services.createPost(sample_request)
                    for _ in range(load)
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True, timeout=30)
                end_time = time.time()
                
                successful_requests = sum(1 for r in results if not isinstance(r, Exception))
                success_rate = successful_requests / load
                
                print(f"Load {load}: {success_rate:.2%} success rate, {end_time - start_time:.2f}s")
                
                if success_rate < 0.5:  # Less than 50% success rate
                    breaking_point = load
                    break
                    
            except asyncio.TimeoutError:
                breaking_point = load
                print(f"Load {load}: Timeout occurred")
                break
            except Exception as e:
                breaking_point = load
                print(f"Load {load}: Exception occurred: {e}")
                break
        
        print(f"Breaking Point Test:")
        print(f"  Breaking point: {breaking_point}")
        
        assert breaking_point is not None
        assert breaking_point > 50  # Should handle at least 50 concurrent requests

    @pytest.mark.asyncio
    async def test_recovery_after_load(self, mock_services: PostService, sample_request: PostGenerationRequest):
        """Test system recovery after high load."""
        # Apply high load
        high_load_tasks = [
            mock_services.createPost(sample_request)
            for _ in range(100)
        ]
        
        await asyncio.gather(*high_load_tasks, return_exceptions=True)
        
        # Wait for system to stabilize
        await asyncio.sleep(1)
        
        # Test normal operation after load
        start_time = time.time()
        normal_result = await mock_services.createPost(sample_request)
        recovery_time = time.time() - start_time
        
        print(f"Recovery Test:")
        print(f"  Recovery time: {recovery_time:.3f}s")
        
        assert recovery_time < 1.0  # Should recover quickly
        assert normal_result is not None  # Should work normally after load

    @pytest.mark.asyncio
    async def test_concurrent_optimization_load(self, mock_services: PostService, sample_request: PostGenerationRequest):
        """Test concurrent optimization under load."""
        # Create a post first
        post = await mock_services.createPost(sample_request)
        
        # Run multiple optimizations concurrently
        num_optimizations = 20
        start_time = time.time()
        
        tasks = [
            mock_services.optimizePost(post.id)
            for _ in range(num_optimizations)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        successful_optimizations = sum(1 for r in results if not isinstance(r, Exception))
        total_time = end_time - start_time
        
        print(f"Concurrent Optimization Load Test:")
        print(f"  Total optimizations: {num_optimizations}")
        print(f"  Successful: {successful_optimizations}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Optimizations per second: {successful_optimizations / total_time:.2f}")
        
        assert successful_optimizations > 0
        assert total_time < 30  # Should complete within 30 seconds

    @pytest.mark.asyncio
    async def test_mixed_operations_load(self, mock_services: PostService, sample_request: PostGenerationRequest):
        """Test mixed operations under load."""
        # Create posts first
        posts = []
        for _ in range(10):
            post = await mock_services.createPost(sample_request)
            posts.append(post)
        
        # Mix different operations
        operations = []
        for i in range(50):
            if i % 4 == 0:
                # Create new post
                operations.append(mock_services.createPost(sample_request))
            elif i % 4 == 1:
                # Get existing post
                post_id = posts[i % len(posts)].id
                operations.append(mock_services.getPost(post_id))
            elif i % 4 == 2:
                # Optimize post
                post_id = posts[i % len(posts)].id
                operations.append(mock_services.optimizePost(post_id))
            else:
                # List posts
                operations.append(mock_services.listPosts("user-123"))
        
        start_time = time.time()
        results = await asyncio.gather(*operations, return_exceptions=True)
        end_time = time.time()
        
        successful_operations = sum(1 for r in results if not isinstance(r, Exception))
        total_time = end_time - start_time
        
        print(f"Mixed Operations Load Test:")
        print(f"  Total operations: {len(operations)}")
        print(f"  Successful: {successful_operations}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Operations per second: {successful_operations / total_time:.2f}")
        
        assert successful_operations > 0
        assert total_time < 60  # Should complete within 60 seconds
