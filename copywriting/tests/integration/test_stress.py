"""
Stress tests for high load scenarios in copywriting service.
"""
import pytest
import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
import random

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import CopywritingInput, CopywritingOutput, Feedback
from tests.test_utils import TestDataFactory, MockAIService, TestAssertions


class TestStressScenarios:
    """Stress tests for high load scenarios."""
    
    @pytest.fixture(scope="class")
    def stress_data(self):
        """Create stress test data."""
        return {
            "products": [
                {
                    "name": f"Product {i}",
                    "description": f"Amazing product {i} with advanced features and cutting-edge technology",
                    "features": [f"Feature {j}" for j in range(5)],
                    "target_audience": f"Target audience {i}"
                }
                for i in range(100)
            ],
            "platforms": ["instagram", "facebook", "twitter", "linkedin", "tiktok"],
            "content_types": ["social_post", "ad_copy", "tweet", "article", "email"],
            "tones": ["inspirational", "persuasive", "casual", "professional", "urgent"],
            "use_cases": ["product_launch", "promotion", "engagement", "education", "entertainment"]
        }
    
    @pytest.fixture(scope="class")
    def mock_stress_services(self):
        """Create mock services for stress testing."""
        services = {
            "ai_service": MockAIService(),
            "copywriting_service": Mock(),
            "database": Mock(),
            "cache": Mock(),
            "load_balancer": Mock()
        }
        
        # Configure mock services for stress testing
        services["copywriting_service"].generate_copy = Mock()
        services["copywriting_service"].process_batch = Mock()
        services["database"].query = Mock()
        services["database"].insert = Mock()
        services["cache"].get = Mock(return_value=None)
        services["cache"].set = Mock()
        services["load_balancer"].get_available_instance = Mock()
        
        return services
    
    def test_high_concurrent_requests(self, stress_data, mock_stress_services):
        """Test high concurrent request handling."""
        # Prepare concurrent requests
        num_requests = 50
        requests = []
        
        for i in range(num_requests):
            product = random.choice(stress_data["products"])
            platform = random.choice(stress_data["platforms"])
            content_type = random.choice(stress_data["content_types"])
            tone = random.choice(stress_data["tones"])
            use_case = random.choice(stress_data["use_cases"])
            
            request = CopywritingInput(
                product_description=product["description"],
                target_platform=platform,
                content_type=content_type,
                tone=tone,
                use_case=use_case,
                key_points=product["features"],
                target_audience=product["target_audience"]
            )
            requests.append(request)
        
        # Mock responses
        def mock_generate_copy(request):
            time.sleep(random.uniform(0.1, 0.5))  # Simulate processing time
            return CopywritingOutput(
                variants=[{
                    "variant_id": f"stress_{hash(str(request.model_dump()))}",
                    "headline": f"Stress Test Headline {hash(str(request.model_dump()))}",
                    "primary_text": f"Stress test content for {request.target_platform}",
                    "call_to_action": "Learn More"
                }],
                model_used="gpt-3.5-turbo",
                generation_time=random.uniform(0.5, 2.0)
            )
        
        mock_stress_services["copywriting_service"].generate_copy.side_effect = mock_generate_copy
        
        # Execute concurrent requests
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(mock_stress_services["copywriting_service"].generate_copy, req) for req in requests]
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=10)
                    results.append(result)
                except Exception as e:
                    pytest.fail(f"Request failed: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify results
        assert len(results) == num_requests
        assert total_time < 30.0  # Should complete within 30 seconds
        assert all(result is not None for result in results)
        
        # Verify performance metrics
        avg_generation_time = sum(result.generation_time for result in results) / len(results)
        assert avg_generation_time < 2.0  # Average should be reasonable
        
        print(f"Processed {num_requests} concurrent requests in {total_time:.2f} seconds")
        print(f"Average generation time: {avg_generation_time:.2f} seconds")
    
    def test_memory_usage_under_load(self, stress_data, mock_stress_services):
        """Test memory usage under high load."""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate large number of requests
        num_requests = 200
        requests = []
        
        for i in range(num_requests):
            product = random.choice(stress_data["products"])
            request = CopywritingInput(
                product_description=product["description"] * 10,  # Large description
                target_platform=random.choice(stress_data["platforms"]),
                content_type=random.choice(stress_data["content_types"]),
                tone=random.choice(stress_data["tones"]),
                use_case=random.choice(stress_data["use_cases"]),
                key_points=product["features"] * 5,  # Large key points
                target_audience=product["target_audience"]
            )
            requests.append(request)
        
        # Mock memory-intensive responses
        def mock_memory_intensive_copy(request):
            # Create large response
            large_variants = []
            for i in range(10):  # Multiple variants
                variant = {
                    "variant_id": f"memory_{hash(str(request.model_dump()))}_{i}",
                    "headline": f"Memory Test Headline {i} " * 10,
                    "primary_text": f"Memory test content {i} " * 50,
                    "call_to_action": "Learn More",
                    "hashtags": [f"#hashtag{j}" for j in range(20)]
                }
                large_variants.append(variant)
            
            return CopywritingOutput(
                variants=large_variants,
                model_used="gpt-3.5-turbo",
                generation_time=1.0
            )
        
        mock_stress_services["copywriting_service"].generate_copy.side_effect = mock_memory_intensive_copy
        
        # Process requests in batches
        batch_size = 20
        results = []
        
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]
            
            # Process batch
            batch_results = []
            for request in batch:
                result = mock_stress_services["copywriting_service"].generate_copy(request)
                batch_results.append(result)
            
            results.extend(batch_results)
            
            # Force garbage collection
            gc.collect()
            
            # Check memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            
            # Memory should not increase excessively
            assert memory_increase < 500  # Should not exceed 500MB increase
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_memory_increase = final_memory - initial_memory
        
        print(f"Initial memory: {initial_memory:.2f} MB")
        print(f"Final memory: {final_memory:.2f} MB")
        print(f"Memory increase: {total_memory_increase:.2f} MB")
        print(f"Processed {len(results)} requests")
        
        # Verify results
        assert len(results) == num_requests
        assert total_memory_increase < 500  # Should not exceed 500MB increase
    
    def test_database_connection_pool_stress(self, stress_data, mock_stress_services):
        """Test database connection pool under stress."""
        # Simulate database connection pool
        max_connections = 10
        active_connections = 0
        connection_lock = threading.Lock()
        
        def get_connection():
            nonlocal active_connections
            with connection_lock:
                if active_connections >= max_connections:
                    raise Exception("Connection pool exhausted")
                active_connections += 1
                return f"connection_{active_connections}"
        
        def release_connection(conn):
            nonlocal active_connections
            with connection_lock:
                active_connections -= 1
        
        # Mock database operations
        def mock_database_operation(data):
            conn = get_connection()
            try:
                time.sleep(random.uniform(0.01, 0.1))  # Simulate DB operation
                return {"stored": True, "connection": conn}
            finally:
                release_connection(conn)
        
        mock_stress_services["database"].insert.side_effect = mock_database_operation
        
        # Generate many database operations
        num_operations = 100
        operations = []
        
        for i in range(num_operations):
            data = {
                "id": i,
                "content": f"Database operation {i}",
                "timestamp": time.time()
            }
            operations.append(data)
        
        # Execute operations concurrently
        results = []
        errors = []
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(mock_stress_services["database"].insert, op) for op in operations]
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=5)
                    results.append(result)
                except Exception as e:
                    errors.append(str(e))
        
        # Verify results
        success_rate = len(results) / num_operations
        print(f"Database operations: {len(results)} successful, {len(errors)} failed")
        print(f"Success rate: {success_rate:.2%}")
        
        # Should have high success rate
        assert success_rate > 0.8  # At least 80% success rate
        
        # Verify connection pool management
        assert active_connections == 0  # All connections should be released
    
    def test_cache_performance_under_load(self, stress_data, mock_stress_services):
        """Test cache performance under high load."""
        # Simulate cache with size limit
        cache_size_limit = 1000
        cache = {}
        cache_hits = 0
        cache_misses = 0
        
        def mock_cache_get(key):
            nonlocal cache_hits, cache_misses
            if key in cache:
                cache_hits += 1
                return cache[key]
            else:
                cache_misses += 1
                return None
        
        def mock_cache_set(key, value):
            if len(cache) >= cache_size_limit:
                # Remove oldest entry
                oldest_key = next(iter(cache))
                del cache[oldest_key]
            cache[key] = value
        
        mock_stress_services["cache"].get.side_effect = mock_cache_get
        mock_stress_services["cache"].set.side_effect = mock_cache_set
        
        # Generate cache operations
        num_operations = 2000
        operations = []
        
        for i in range(num_operations):
            key = f"cache_key_{i % 500}"  # Some keys will be repeated
            value = f"cache_value_{i}"
            operations.append((key, value))
        
        # Execute cache operations
        start_time = time.time()
        
        for key, value in operations:
            # Try to get from cache
            cached_value = mock_stress_services["cache"].get(key)
            if cached_value is None:
                # Cache miss, set value
                mock_stress_services["cache"].set(key, value)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify cache performance
        hit_rate = cache_hits / (cache_hits + cache_misses)
        print(f"Cache operations: {num_operations}")
        print(f"Cache hits: {cache_hits}")
        print(f"Cache misses: {cache_misses}")
        print(f"Hit rate: {hit_rate:.2%}")
        print(f"Total time: {total_time:.2f} seconds")
        
        # Should have reasonable hit rate due to key repetition
        assert hit_rate > 0.3  # At least 30% hit rate
        assert total_time < 5.0  # Should be fast
    
    def test_api_rate_limiting_stress(self, stress_data, mock_stress_services):
        """Test API rate limiting under stress."""
        # Simulate rate limiting
        rate_limit = 100  # requests per minute
        request_times = []
        rate_limit_lock = threading.Lock()
        
        def check_rate_limit():
            nonlocal request_times
            current_time = time.time()
            
            with rate_limit_lock:
                # Remove requests older than 1 minute
                request_times = [t for t in request_times if current_time - t < 60]
                
                if len(request_times) >= rate_limit:
                    return False  # Rate limit exceeded
                
                request_times.append(current_time)
                return True
        
        def mock_rate_limited_request(request):
            if not check_rate_limit():
                raise Exception("Rate limit exceeded")
            
            time.sleep(random.uniform(0.01, 0.05))  # Simulate processing
            return CopywritingOutput(
                variants=[{
                    "variant_id": f"rate_limited_{hash(str(request.model_dump()))}",
                    "headline": "Rate Limited Response",
                    "primary_text": "This response was rate limited",
                    "call_to_action": "Try Again Later"
                }],
                model_used="gpt-3.5-turbo",
                generation_time=0.1
            )
        
        mock_stress_services["copywriting_service"].generate_copy.side_effect = mock_rate_limited_request
        
        # Generate requests that exceed rate limit
        num_requests = 150  # More than rate limit
        requests = []
        
        for i in range(num_requests):
            request = CopywritingInput(
                product_description=f"Rate limit test {i}",
                target_platform="instagram",
                content_type="social_post",
                tone="inspirational",
                use_case="product_launch"
            )
            requests.append(request)
        
        # Execute requests
        results = []
        rate_limit_errors = []
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(mock_stress_services["copywriting_service"].generate_copy, req) for req in requests]
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=10)
                    results.append(result)
                except Exception as e:
                    if "Rate limit exceeded" in str(e):
                        rate_limit_errors.append(e)
                    else:
                        pytest.fail(f"Unexpected error: {e}")
        
        # Verify rate limiting
        print(f"Successful requests: {len(results)}")
        print(f"Rate limited requests: {len(rate_limit_errors)}")
        print(f"Rate limit effectiveness: {len(rate_limit_errors) / num_requests:.2%}")
        
        # Should have some rate limiting
        assert len(rate_limit_errors) > 0
        assert len(results) <= rate_limit  # Should not exceed rate limit
    
    def test_error_recovery_under_stress(self, stress_data, mock_stress_services):
        """Test error recovery under stress."""
        # Simulate intermittent failures
        failure_rate = 0.3  # 30% failure rate
        request_count = 0
        
        def mock_unreliable_service(request):
            nonlocal request_count
            request_count += 1
            
            if random.random() < failure_rate:
                raise Exception(f"Service failure {request_count}")
            
            time.sleep(random.uniform(0.01, 0.1))
            return CopywritingOutput(
                variants=[{
                    "variant_id": f"recovery_{request_count}",
                    "headline": "Recovery Success",
                    "primary_text": "Service recovered successfully",
                    "call_to_action": "Continue"
                }],
                model_used="gpt-3.5-turbo",
                generation_time=0.1
            )
        
        mock_stress_services["copywriting_service"].generate_copy.side_effect = mock_unreliable_service
        
        # Generate requests
        num_requests = 100
        requests = []
        
        for i in range(num_requests):
            request = CopywritingInput(
                product_description=f"Error recovery test {i}",
                target_platform="instagram",
                content_type="social_post",
                tone="inspirational",
                use_case="product_launch"
            )
            requests.append(request)
        
        # Execute requests with retry logic
        results = []
        errors = []
        max_retries = 3
        
        def retry_request(request, max_retries=3):
            for attempt in range(max_retries):
                try:
                    result = mock_stress_services["copywriting_service"].generate_copy(request)
                    return result, None
                except Exception as e:
                    if attempt == max_retries - 1:
                        return None, e
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
            return None, Exception("Max retries exceeded")
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(retry_request, req, max_retries) for req in requests]
            
            for future in as_completed(futures):
                result, error = future.result(timeout=30)
                if result:
                    results.append(result)
                else:
                    errors.append(error)
        
        # Verify error recovery
        success_rate = len(results) / num_requests
        print(f"Successful requests: {len(results)}")
        print(f"Failed requests: {len(errors)}")
        print(f"Success rate: {success_rate:.2%}")
        
        # Should have high success rate with retry logic
        assert success_rate > 0.8  # At least 80% success rate
        assert len(errors) < num_requests * 0.2  # Less than 20% failures
    
    def test_memory_leak_detection(self, stress_data, mock_stress_services):
        """Test for memory leaks under stress."""
        import psutil
        import gc
        
        process = psutil.Process()
        
        # Get initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple cycles to detect leaks
        num_cycles = 10
        cycle_memory = []
        
        for cycle in range(num_cycles):
            # Generate requests for this cycle
            num_requests = 50
            requests = []
            
            for i in range(num_requests):
                request = CopywritingInput(
                    product_description=f"Memory leak test cycle {cycle} request {i}",
                    target_platform="instagram",
                    content_type="social_post",
                    tone="inspirational",
                    use_case="product_launch"
                )
                requests.append(request)
            
            # Mock response
            def mock_response(request):
                return CopywritingOutput(
                    variants=[{
                        "variant_id": f"leak_test_{cycle}_{hash(str(request.model_dump()))}",
                        "headline": f"Memory leak test {cycle}",
                        "primary_text": f"Cycle {cycle} content",
                        "call_to_action": "Test"
                    }],
                    model_used="gpt-3.5-turbo",
                    generation_time=0.1
                )
            
            mock_stress_services["copywriting_service"].generate_copy.side_effect = mock_response
            
            # Process requests
            results = []
            for request in requests:
                result = mock_stress_services["copywriting_service"].generate_copy(request)
                results.append(result)
            
            # Force garbage collection
            gc.collect()
            
            # Check memory after cycle
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            cycle_memory.append(current_memory)
            
            print(f"Cycle {cycle + 1}: Memory = {current_memory:.2f} MB")
        
        # Analyze memory trend
        memory_increase = cycle_memory[-1] - initial_memory
        memory_trend = []
        
        for i in range(1, len(cycle_memory)):
            trend = cycle_memory[i] - cycle_memory[i-1]
            memory_trend.append(trend)
        
        avg_memory_increase = sum(memory_trend) / len(memory_trend)
        
        print(f"Initial memory: {initial_memory:.2f} MB")
        print(f"Final memory: {cycle_memory[-1]:.2f} MB")
        print(f"Total memory increase: {memory_increase:.2f} MB")
        print(f"Average memory increase per cycle: {avg_memory_increase:.2f} MB")
        
        # Check for memory leaks
        assert memory_increase < 100  # Should not increase by more than 100MB
        assert avg_memory_increase < 10  # Average increase per cycle should be small
        
        # Memory should stabilize (not continuously increase)
        if len(memory_trend) > 3:
            recent_trend = memory_trend[-3:]
            assert max(recent_trend) - min(recent_trend) < 20  # Should stabilize
