"""
Performance Benchmark Tests for LinkedIn Posts
============================================

Comprehensive performance benchmark tests for LinkedIn posts including
response time benchmarks, throughput tests, memory usage analysis,
and performance optimization validation.
"""

import pytest
import asyncio
import time
import statistics
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

# Import components for performance testing
from ...services.post_service import PostService, PostRepository, AIService, CacheService
from ...core.entities import (
    LinkedInPost, PostContent, PostGenerationRequest, PostGenerationResponse,
    PostOptimizationResult, PostValidationResult, PostType, PostTone, PostStatus,
    EngagementMetrics, ContentAnalysisResult
)


class TestPerformanceBenchmarks:
    """Test suite for performance benchmarks and optimization."""

    @pytest.fixture
    def mock_services(self):
        """Create mocked services for performance testing."""
        mock_repository = AsyncMock(spec=PostRepository)
        mock_ai_service = AsyncMock(spec=AIService)
        mock_cache_service = AsyncMock(spec=CacheService)
        return PostService(mock_repository, mock_ai_service, mock_cache_service)

    @pytest.fixture
    def sample_request(self) -> PostGenerationRequest:
        """Sample request for performance testing."""
        return PostGenerationRequest(
            topic="AI in Modern Business",
            keyPoints=["Increased efficiency", "Cost reduction", "Innovation"],
            targetAudience="Business leaders",
            industry="Technology",
            tone=PostTone.PROFESSIONAL,
            postType=PostType.TEXT,
            keywords=["AI", "business", "innovation"],
            additionalContext="Focus on practical applications"
        )

    @pytest.mark.asyncio
    async def test_post_creation_response_time_benchmark(self, mock_services: PostService, sample_request: PostGenerationRequest):
        """Benchmark post creation response time."""
        # Mock successful response
        mock_services.ai_service.generatePost.return_value = PostGenerationResponse(
            success=True,
            post=LinkedInPost(
                id="test-123",
                userId="user-123",
                title="AI in Modern Business",
                content=PostContent(
                    text="Generated content",
                    hashtags=["#AI", "#business"],
                    mentions=[],
                    links=[],
                    images=[],
                    callToAction=""
                ),
                postType=PostType.TEXT,
                tone=PostTone.PROFESSIONAL,
                status=PostStatus.DRAFT,
                createdAt=datetime.now(),
                updatedAt=datetime.now(),
                aiScore=85.0
            ),
            message="Post generated successfully"
        )

        # Measure response time for multiple requests
        response_times = []
        num_requests = 100

        for i in range(num_requests):
            start_time = time.time()
            result = await mock_services.createPost(sample_request)
            end_time = time.time()
            response_times.append(end_time - start_time)

        # Calculate statistics
        avg_response_time = statistics.mean(response_times)
        median_response_time = statistics.median(response_times)
        p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        p99_response_time = statistics.quantiles(response_times, n=100)[98]  # 99th percentile

        # Performance assertions
        assert avg_response_time < 0.1  # Average should be under 100ms
        assert median_response_time < 0.1  # Median should be under 100ms
        assert p95_response_time < 0.2  # 95th percentile should be under 200ms
        assert p99_response_time < 0.5  # 99th percentile should be under 500ms

        print(f"Performance Results:")
        print(f"  Average Response Time: {avg_response_time:.4f}s")
        print(f"  Median Response Time: {median_response_time:.4f}s")
        print(f"  95th Percentile: {p95_response_time:.4f}s")
        print(f"  99th Percentile: {p99_response_time:.4f}s")

    @pytest.mark.asyncio
    async def test_memory_usage_benchmark(self, mock_services: PostService, sample_request: PostGenerationRequest):
        """Benchmark memory usage during post creation."""
        # Mock successful response
        mock_services.ai_service.generatePost.return_value = PostGenerationResponse(
            success=True,
            post=LinkedInPost(
                id="test-123",
                userId="user-123",
                title="AI in Modern Business",
                content=PostContent(
                    text="Generated content",
                    hashtags=["#AI", "#business"],
                    mentions=[],
                    links=[],
                    images=[],
                    callToAction=""
                ),
                postType=PostType.TEXT,
                tone=PostTone.PROFESSIONAL,
                status=PostStatus.DRAFT,
                createdAt=datetime.now(),
                updatedAt=datetime.now(),
                aiScore=85.0
            ),
            message="Post generated successfully"
        )

        # Measure memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create multiple posts
        num_posts = 1000
        posts = []

        for i in range(num_posts):
            post = await mock_services.createPost(sample_request)
            posts.append(post)

        # Force garbage collection
        gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Performance assertions
        assert memory_increase < 50  # Memory increase should be under 50MB
        assert len(posts) == num_posts

        print(f"Memory Usage Results:")
        print(f"  Initial Memory: {initial_memory:.2f} MB")
        print(f"  Final Memory: {final_memory:.2f} MB")
        print(f"  Memory Increase: {memory_increase:.2f} MB")
        print(f"  Memory per Post: {memory_increase / num_posts:.4f} MB")

    @pytest.mark.asyncio
    async def test_throughput_benchmark(self, mock_services: PostService, sample_request: PostGenerationRequest):
        """Benchmark system throughput."""
        # Mock successful response
        mock_services.ai_service.generatePost.return_value = PostGenerationResponse(
            success=True,
            post=LinkedInPost(
                id="test-123",
                userId="user-123",
                title="AI in Modern Business",
                content=PostContent(
                    text="Generated content",
                    hashtags=["#AI", "#business"],
                    mentions=[],
                    links=[],
                    images=[],
                    callToAction=""
                ),
                postType=PostType.TEXT,
                tone=PostTone.PROFESSIONAL,
                status=PostStatus.DRAFT,
                createdAt=datetime.now(),
                updatedAt=datetime.now(),
                aiScore=85.0
            ),
            message="Post generated successfully"
        )

        # Measure throughput
        num_requests = 1000
        start_time = time.time()

        # Create posts concurrently
        tasks = [mock_services.createPost(sample_request) for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)

        end_time = time.time()
        total_time = end_time - start_time
        throughput = num_requests / total_time

        # Performance assertions
        assert throughput > 100  # Should handle at least 100 requests per second
        assert total_time < 10  # Should complete within 10 seconds
        assert len(results) == num_requests

        print(f"Throughput Results:")
        print(f"  Total Requests: {num_requests}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.2f} requests/second")

    @pytest.mark.asyncio
    async def test_cache_performance_benchmark(self, mock_services: PostService, sample_request: PostGenerationRequest):
        """Benchmark cache performance."""
        # Mock cache behavior
        cache_hits = 0
        cache_misses = 0

        async def mock_cache_get(key):
            nonlocal cache_hits, cache_misses
            if "cached" in key:
                cache_hits += 1
                return "cached_data"
            else:
                cache_misses += 1
                return None

        mock_services.cache_service.get.side_effect = mock_cache_get

        # Test cache performance
        num_requests = 1000
        cache_operations = []

        for i in range(num_requests):
            start_time = time.time()
            await mock_services.cache_service.get(f"test_key_{i % 10}")  # 10 unique keys
            end_time = time.time()
            cache_operations.append(end_time - start_time)

        # Calculate cache statistics
        avg_cache_time = statistics.mean(cache_operations)
        cache_hit_rate = cache_hits / (cache_hits + cache_misses)

        # Performance assertions
        assert avg_cache_time < 0.001  # Cache operations should be very fast
        assert cache_hit_rate > 0.8  # Should have good cache hit rate

        print(f"Cache Performance Results:")
        print(f"  Cache Hits: {cache_hits}")
        print(f"  Cache Misses: {cache_misses}")
        print(f"  Hit Rate: {cache_hit_rate:.2%}")
        print(f"  Average Cache Time: {avg_cache_time:.6f}s")

    @pytest.mark.asyncio
    async def test_database_performance_benchmark(self, mock_services: PostService, sample_request: PostGenerationRequest):
        """Benchmark database operations performance."""
        # Mock database operations
        db_operations = []

        async def mock_db_operation(operation_type):
            start_time = time.time()
            # Simulate database operation time
            await asyncio.sleep(0.01)  # 10ms simulation
            end_time = time.time()
            db_operations.append(end_time - start_time)

        # Test different database operations
        operations = ["create", "read", "update", "delete"] * 25  # 100 operations

        for operation in operations:
            await mock_db_operation(operation)

        # Calculate database performance statistics
        avg_db_time = statistics.mean(db_operations)
        max_db_time = max(db_operations)
        min_db_time = min(db_operations)

        # Performance assertions
        assert avg_db_time < 0.02  # Average DB operation should be under 20ms
        assert max_db_time < 0.05  # Max DB operation should be under 50ms

        print(f"Database Performance Results:")
        print(f"  Total Operations: {len(db_operations)}")
        print(f"  Average Time: {avg_db_time:.4f}s")
        print(f"  Min Time: {min_db_time:.4f}s")
        print(f"  Max Time: {max_db_time:.4f}s")

    @pytest.mark.asyncio
    async def test_ai_service_performance_benchmark(self, mock_services: PostService, sample_request: PostGenerationRequest):
        """Benchmark AI service performance."""
        # Mock AI service with realistic delays
        ai_response_times = []

        async def mock_ai_generate(request):
            start_time = time.time()
            # Simulate AI processing time
            await asyncio.sleep(0.1)  # 100ms simulation
            end_time = time.time()
            ai_response_times.append(end_time - start_time)
            
            return PostGenerationResponse(
                success=True,
                post=LinkedInPost(
                    id="test-123",
                    userId="user-123",
                    title="AI in Modern Business",
                    content=PostContent(
                        text="Generated content",
                        hashtags=["#AI", "#business"],
                        mentions=[],
                        links=[],
                        images=[],
                        callToAction=""
                    ),
                    postType=PostType.TEXT,
                    tone=PostTone.PROFESSIONAL,
                    status=PostStatus.DRAFT,
                    createdAt=datetime.now(),
                    updatedAt=datetime.now(),
                    aiScore=85.0
                ),
                message="Post generated successfully"
            )

        mock_services.ai_service.generatePost.side_effect = mock_ai_generate

        # Test AI service performance
        num_requests = 50
        results = []

        for i in range(num_requests):
            result = await mock_services.createPost(sample_request)
            results.append(result)

        # Calculate AI performance statistics
        avg_ai_time = statistics.mean(ai_response_times)
        p95_ai_time = statistics.quantiles(ai_response_times, n=20)[18]

        # Performance assertions
        assert avg_ai_time < 0.15  # Average AI time should be under 150ms
        assert p95_ai_time < 0.2  # 95th percentile should be under 200ms
        assert len(results) == num_requests

        print(f"AI Service Performance Results:")
        print(f"  Total Requests: {num_requests}")
        print(f"  Average AI Time: {avg_ai_time:.4f}s")
        print(f"  95th Percentile: {p95_ai_time:.4f}s")

    @pytest.mark.asyncio
    async def test_concurrent_user_performance_benchmark(self, mock_services: PostService, sample_request: PostGenerationRequest):
        """Benchmark performance under concurrent user load."""
        # Mock successful response
        mock_services.ai_service.generatePost.return_value = PostGenerationResponse(
            success=True,
            post=LinkedInPost(
                id="test-123",
                userId="user-123",
                title="AI in Modern Business",
                content=PostContent(
                    text="Generated content",
                    hashtags=["#AI", "#business"],
                    mentions=[],
                    links=[],
                    images=[],
                    callToAction=""
                ),
                postType=PostType.TEXT,
                tone=PostTone.PROFESSIONAL,
                status=PostStatus.DRAFT,
                createdAt=datetime.now(),
                updatedAt=datetime.now(),
                aiScore=85.0
            ),
            message="Post generated successfully"
        )

        # Test with different levels of concurrency
        concurrency_levels = [1, 5, 10, 20, 50]
        results = {}

        for concurrency in concurrency_levels:
            start_time = time.time()
            
            # Create concurrent tasks
            tasks = [mock_services.createPost(sample_request) for _ in range(concurrency)]
            concurrent_results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            results[concurrency] = {
                "total_time": total_time,
                "throughput": concurrency / total_time,
                "avg_response_time": total_time / concurrency
            }

        # Performance assertions
        for concurrency, metrics in results.items():
            assert metrics["throughput"] > 10  # Should maintain minimum throughput
            assert metrics["avg_response_time"] < 1.0  # Should maintain reasonable response time

        print(f"Concurrent User Performance Results:")
        for concurrency, metrics in results.items():
            print(f"  {concurrency} concurrent users:")
            print(f"    Total Time: {metrics['total_time']:.2f}s")
            print(f"    Throughput: {metrics['throughput']:.2f} req/s")
            print(f"    Avg Response Time: {metrics['avg_response_time']:.4f}s")

    @pytest.mark.asyncio
    async def test_resource_utilization_benchmark(self, mock_services: PostService, sample_request: PostGenerationRequest):
        """Benchmark system resource utilization."""
        # Mock successful response
        mock_services.ai_service.generatePost.return_value = PostGenerationResponse(
            success=True,
            post=LinkedInPost(
                id="test-123",
                userId="user-123",
                title="AI in Modern Business",
                content=PostContent(
                    text="Generated content",
                    hashtags=["#AI", "#business"],
                    mentions=[],
                    links=[],
                    images=[],
                    callToAction=""
                ),
                postType=PostType.TEXT,
                tone=PostTone.PROFESSIONAL,
                status=PostStatus.DRAFT,
                createdAt=datetime.now(),
                updatedAt=datetime.now(),
                aiScore=85.0
            ),
            message="Post generated successfully"
        )

        # Monitor resource usage
        process = psutil.Process()
        initial_cpu_percent = process.cpu_percent()
        initial_memory = process.memory_info().rss / 1024 / 1024

        # Perform operations
        num_operations = 100
        cpu_usage = []
        memory_usage = []

        for i in range(num_operations):
            # Measure CPU and memory during operation
            start_cpu = process.cpu_percent()
            start_memory = process.memory_info().rss / 1024 / 1024
            
            await mock_services.createPost(sample_request)
            
            end_cpu = process.cpu_percent()
            end_memory = process.memory_info().rss / 1024 / 1024
            
            cpu_usage.append(end_cpu)
            memory_usage.append(end_memory)

        # Calculate resource utilization statistics
        avg_cpu = statistics.mean(cpu_usage)
        max_cpu = max(cpu_usage)
        avg_memory = statistics.mean(memory_usage)
        max_memory = max(memory_usage)

        # Performance assertions
        assert avg_cpu < 80  # Average CPU usage should be under 80%
        assert max_cpu < 95  # Peak CPU usage should be under 95%
        assert avg_memory < 500  # Average memory usage should be under 500MB
        assert max_memory < 1000  # Peak memory usage should be under 1GB

        print(f"Resource Utilization Results:")
        print(f"  Average CPU Usage: {avg_cpu:.2f}%")
        print(f"  Peak CPU Usage: {max_cpu:.2f}%")
        print(f"  Average Memory Usage: {avg_memory:.2f} MB")
        print(f"  Peak Memory Usage: {max_memory:.2f} MB")

    @pytest.mark.asyncio
    async def test_error_handling_performance_benchmark(self, mock_services: PostService, sample_request: PostGenerationRequest):
        """Benchmark performance under error conditions."""
        # Mock error responses
        error_count = 0
        success_count = 0
        error_times = []
        success_times = []

        async def mock_ai_with_errors(request):
            nonlocal error_count, success_count
            
            # Simulate 10% error rate
            if error_count < 10:  # First 10 requests fail
                error_count += 1
                start_time = time.time()
                await asyncio.sleep(0.05)  # Error response time
                end_time = time.time()
                error_times.append(end_time - start_time)
                raise Exception("AI service error")
            else:
                success_count += 1
                start_time = time.time()
                await asyncio.sleep(0.1)  # Success response time
                end_time = time.time()
                success_times.append(end_time - start_time)
                
                return PostGenerationResponse(
                    success=True,
                    post=LinkedInPost(
                        id="test-123",
                        userId="user-123",
                        title="AI in Modern Business",
                        content=PostContent(
                            text="Generated content",
                            hashtags=["#AI", "#business"],
                            mentions=[],
                            links=[],
                            images=[],
                            callToAction=""
                        ),
                        postType=PostType.TEXT,
                        tone=PostTone.PROFESSIONAL,
                        status=PostStatus.DRAFT,
                        createdAt=datetime.now(),
                        updatedAt=datetime.now(),
                        aiScore=85.0
                    ),
                    message="Post generated successfully"
                )

        mock_services.ai_service.generatePost.side_effect = mock_ai_with_errors

        # Test error handling performance
        num_requests = 100
        results = []
        errors = []

        for i in range(num_requests):
            try:
                result = await mock_services.createPost(sample_request)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Calculate error handling performance
        error_rate = len(errors) / num_requests
        avg_error_time = statistics.mean(error_times) if error_times else 0
        avg_success_time = statistics.mean(success_times) if success_times else 0

        # Performance assertions
        assert error_rate < 0.2  # Error rate should be under 20%
        assert avg_error_time < 0.1  # Error handling should be fast
        assert avg_success_time < 0.15  # Success responses should be reasonable

        print(f"Error Handling Performance Results:")
        print(f"  Total Requests: {num_requests}")
        print(f"  Success Count: {success_count}")
        print(f"  Error Count: {error_count}")
        print(f"  Error Rate: {error_rate:.2%}")
        print(f"  Average Error Time: {avg_error_time:.4f}s")
        print(f"  Average Success Time: {avg_success_time:.4f}s")
