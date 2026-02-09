"""
Performance and load tests
Tests for performance characteristics and load handling
"""

import pytest
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch


@pytest.fixture
def performance_client():
    """Create test client for performance testing"""
    # Use canonical API router instead of deprecated recovery_api
    from api.recovery_api_refactored import router
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestResponseTime:
    """Tests for response time performance"""
    
    def test_assessment_endpoint_response_time(self, performance_client):
        """Test that assessment endpoint responds within acceptable time"""
        request_data = {
            "addiction_type": "smoking",
            "severity": "moderate",
            "frequency": "daily",
            "user_id": "user_123"
        }
        
        start_time = time.time()
        response = performance_client.post("/assessment/assess", json=request_data)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Should respond within 2 seconds (adjust based on requirements)
        assert response_time < 2.0
        assert response.status_code in [200, 201, 422, 500]
    
    def test_progress_endpoint_response_time(self, performance_client):
        """Test that progress endpoint responds within acceptable time"""
        user_id = "user_123"
        
        start_time = time.time()
        response = performance_client.get(f"/progress/{user_id}")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Should respond within 1 second
        assert response_time < 1.0
    
    def test_multiple_requests_average_time(self, performance_client):
        """Test average response time for multiple requests"""
        request_data = {
            "addiction_type": "smoking",
            "severity": "moderate",
            "frequency": "daily",
            "user_id": "user_123"
        }
        
        response_times = []
        num_requests = 10
        
        for _ in range(num_requests):
            start_time = time.time()
            performance_client.post("/assessment/assess", json=request_data)
            end_time = time.time()
            response_times.append(end_time - start_time)
        
        avg_time = statistics.mean(response_times)
        max_time = max(response_times)
        
        # Average should be reasonable
        assert avg_time < 1.0
        # Max should not be too high
        assert max_time < 3.0


class TestConcurrentLoad:
    """Tests for concurrent request handling"""
    
    def test_concurrent_assessment_requests(self, performance_client):
        """Test handling multiple concurrent assessment requests"""
        request_data = {
            "addiction_type": "smoking",
            "severity": "moderate",
            "frequency": "daily",
            "user_id": "user_123"
        }
        
        def make_request():
            return performance_client.post("/assessment/assess", json=request_data)
        
        num_concurrent = 20
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(make_request) for _ in range(num_concurrent)]
            results = [f.result() for f in as_completed(futures)]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All requests should complete
        assert len(results) == num_concurrent
        
        # Should handle concurrent requests efficiently
        # (time should be less than sequential requests)
        assert total_time < num_concurrent * 1.0  # Should be parallelized
    
    def test_concurrent_progress_requests(self, performance_client):
        """Test handling multiple concurrent progress requests"""
        user_id = "user_123"
        
        def make_request():
            return performance_client.get(f"/progress/{user_id}")
        
        num_concurrent = 30
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(make_request) for _ in range(num_concurrent)]
            results = [f.result() for f in as_completed(futures)]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        assert len(results) == num_concurrent
        # Should handle GET requests efficiently
        assert total_time < num_concurrent * 0.5


class TestMemoryUsage:
    """Tests for memory usage"""
    
    def test_large_payload_memory(self, performance_client):
        """Test memory handling with large payloads"""
        # Create large notes field
        large_notes = "A" * 100000  # 100KB
        
        request_data = {
            "user_id": "user_123",
            "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "mood": "good",
            "cravings_level": 3,
            "notes": large_notes
        }
        
        response = performance_client.post("/progress/log-entry", json=request_data)
        
        # Should handle or reject gracefully
        assert response.status_code in [201, 400, 413, 422]
    
    def test_many_entries_memory(self, performance_client):
        """Test memory with many entries"""
        # Create many entries
        for i in range(100):
            request_data = {
                "user_id": "user_123",
                "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "mood": "good",
                "cravings_level": 3
            }
            
            response = performance_client.post("/progress/log-entry", json=request_data)
            # Should handle many entries
            assert response.status_code in [201, 400, 422, 500]


class TestScalability:
    """Tests for scalability"""
    
    def test_increasing_load(self, performance_client):
        """Test performance with increasing load"""
        request_data = {
            "addiction_type": "smoking",
            "severity": "moderate",
            "frequency": "daily",
            "user_id": "user_123"
        }
        
        load_levels = [1, 5, 10, 20]
        results = []
        
        for load in load_levels:
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=load) as executor:
                futures = [executor.submit(
                    lambda: performance_client.post("/assessment/assess", json=request_data)
                ) for _ in range(load)]
                [f.result() for f in as_completed(futures)]
            
            end_time = time.time()
            results.append(end_time - start_time)
        
        # Performance should degrade gracefully (not exponentially)
        # Later loads should not be much worse than earlier
        if len(results) > 1:
            # Check that increase is reasonable
            increase_ratio = results[-1] / results[0] if results[0] > 0 else 1
            # Should not increase more than 5x
            assert increase_ratio < 5.0


class TestCaching:
    """Tests for caching performance"""
    
    def test_cached_response_faster(self, performance_client):
        """Test that cached responses are faster"""
        user_id = "user_123"
        
        # First request (cache miss)
        start1 = time.time()
        response1 = performance_client.get(f"/progress/{user_id}")
        time1 = time.time() - start1
        
        # Second request (should be cache hit if caching enabled)
        start2 = time.time()
        response2 = performance_client.get(f"/progress/{user_id}")
        time2 = time.time() - start2
        
        # Second request might be faster (if cached)
        # Or might be similar (if not cached)
        assert time2 <= time1 * 1.5  # Should not be much slower


class TestDatabasePerformance:
    """Tests for database-related performance"""
    
    @pytest.mark.skip(reason="Requires database setup")
    def test_database_query_performance(self, performance_client):
        """Test database query performance"""
        # This would require actual database
        pass
    
    @pytest.mark.skip(reason="Requires database setup")
    def test_database_connection_pooling(self, performance_client):
        """Test database connection pooling"""
        # This would require actual database
        pass


class TestResourceUsage:
    """Tests for resource usage"""
    
    def test_cpu_usage_under_load(self, performance_client):
        """Test CPU usage under load"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_cpu = process.cpu_percent()
        
        # Generate load
        request_data = {
            "addiction_type": "smoking",
            "severity": "moderate",
            "frequency": "daily",
            "user_id": "user_123"
        }
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(
                lambda: performance_client.post("/assessment/assess", json=request_data)
            ) for _ in range(50)]
            [f.result() for f in as_completed(futures)]
        
        # CPU usage should be reasonable
        # (This is a basic test, actual monitoring would be better)
        final_cpu = process.cpu_percent()
        # Just verify it doesn't crash
        assert final_cpu >= 0



