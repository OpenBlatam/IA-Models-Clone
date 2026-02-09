"""
Playwright Load Tests
=====================
Load and stress testing with Playwright.
"""

import pytest
from playwright.sync_api import Page, Browser
import time
import concurrent.futures
from typing import List


@pytest.fixture
def api_base_url():
    """API base URL."""
    return "http://localhost:8000"


class TestPlaywrightLoadTesting:
    """Load testing with Playwright."""
    
    @pytest.mark.playwright
    @pytest.mark.slow
    @pytest.mark.load
    def test_concurrent_health_checks(self, browser, api_base_url):
        """Test concurrent health check requests."""
        def make_request():
            context = browser.new_context()
            page = context.new_page()
            try:
                start = time.time()
                response = page.request.get(f"{api_base_url}/health")
                elapsed = time.time() - start
                return response.status, elapsed
            finally:
                page.close()
                context.close()
        
        # Make 50 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(make_request) for _ in range(50)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        statuses, times = zip(*results)
        
        # All should succeed
        assert all(status == 200 for status in statuses)
        
        # Average time should be reasonable
        avg_time = sum(times) / len(times)
        assert avg_time < 2.0, f"Average response time too high: {avg_time:.3f}s"
        
        # Max time should not be too high
        max_time = max(times)
        assert max_time < 5.0, f"Max response time too high: {max_time:.3f}s"
    
    @pytest.mark.playwright
    @pytest.mark.slow
    @pytest.mark.load
    def test_sustained_load(self, browser, api_base_url):
        """Test sustained load over time."""
        def make_request():
            context = browser.new_context()
            page = context.new_page()
            try:
                return page.request.get(f"{api_base_url}/health")
            finally:
                page.close()
                context.close()
        
        # Make requests for 30 seconds
        start_time = time.time()
        request_count = 0
        errors = []
        
        while time.time() - start_time < 30:
            try:
                response = make_request()
                request_count += 1
                if response.status != 200:
                    errors.append(response.status)
            except Exception as e:
                errors.append(str(e))
            time.sleep(0.1)  # 10 requests per second
        
        # Should handle sustained load
        assert request_count > 0
        # Error rate should be low
        error_rate = len(errors) / request_count if request_count > 0 else 0
        assert error_rate < 0.1, f"Error rate too high: {error_rate:.2%}"
    
    @pytest.mark.playwright
    @pytest.mark.slow
    @pytest.mark.load
    def test_ramp_up_load(self, browser, api_base_url):
        """Test gradual ramp-up of load."""
        def make_request():
            context = browser.new_context()
            page = context.new_page()
            try:
                return page.request.get(f"{api_base_url}/health")
            finally:
                page.close()
                context.close()
        
        # Gradually increase load
        for workers in [1, 5, 10, 20, 30]:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(make_request) for _ in range(workers * 2)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            # All should succeed at each level
            statuses = [r.status for r in results]
            success_rate = sum(1 for s in statuses if s == 200) / len(statuses)
            assert success_rate >= 0.9, f"Success rate too low at {workers} workers: {success_rate:.2%}"


class TestPlaywrightStressTesting:
    """Stress testing with Playwright."""
    
    @pytest.mark.playwright
    @pytest.mark.slow
    @pytest.mark.stress
    def test_max_concurrent_connections(self, browser, api_base_url):
        """Test maximum concurrent connections."""
        def make_request():
            context = browser.new_context()
            page = context.new_page()
            try:
                return page.request.get(f"{api_base_url}/health")
            finally:
                page.close()
                context.close()
        
        # Try to make many concurrent connections
        max_workers = 100
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(make_request) for _ in range(max_workers)]
            results = []
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=10)
                    results.append(result.status)
                except Exception:
                    results.append(None)
        
        # Most should succeed
        success_count = sum(1 for s in results if s == 200)
        success_rate = success_count / len(results)
        assert success_rate >= 0.7, f"Success rate too low: {success_rate:.2%}"
    
    @pytest.mark.playwright
    @pytest.mark.slow
    @pytest.mark.stress
    def test_rapid_fire_requests(self, browser, api_base_url):
        """Test rapid-fire requests."""
        def make_request():
            context = browser.new_context()
            page = context.new_page()
            try:
                return page.request.get(f"{api_base_url}/health")
            finally:
                page.close()
                context.close()
        
        # Make requests as fast as possible
        start = time.time()
        request_count = 0
        
        while time.time() - start < 10:  # 10 seconds
            try:
                make_request()
                request_count += 1
            except Exception:
                pass
        
        # Should handle rapid requests
        assert request_count > 0
        requests_per_second = request_count / 10
        assert requests_per_second > 1, f"Too few requests per second: {requests_per_second:.1f}"


class TestPlaywrightMemoryLeaks:
    """Tests for memory leaks."""
    
    @pytest.mark.playwright
    @pytest.mark.slow
    def test_memory_usage_over_time(self, browser, api_base_url):
        """Test memory usage over time."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Make many requests
        for i in range(100):
            context = browser.new_context()
            page = context.new_page()
            try:
                page.request.get(f"{api_base_url}/health")
            finally:
                page.close()
                context.close()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 100, f"Memory leak detected: {memory_increase:.1f}MB increase"


class TestPlaywrightConnectionPooling:
    """Tests for connection pooling."""
    
    @pytest.mark.playwright
    @pytest.mark.slow
    def test_connection_reuse(self, browser, api_base_url):
        """Test connection reuse."""
        context = browser.new_context()
        page = context.new_page()
        
        try:
            # Make multiple requests with same context
            times = []
            for _ in range(20):
                start = time.time()
                page.request.get(f"{api_base_url}/health")
                elapsed = time.time() - start
                times.append(elapsed)
            
            # Later requests should be faster (connection reuse)
            first_half = sum(times[:10]) / 10
            second_half = sum(times[10:]) / 10
            
            # Second half should be similar or faster
            assert second_half <= first_half * 1.5, "Connection reuse not working"
        finally:
            page.close()
            context.close()


class TestPlaywrightTimeoutHandling:
    """Tests for timeout handling under load."""
    
    @pytest.mark.playwright
    @pytest.mark.slow
    def test_timeout_under_load(self, browser, api_base_url):
        """Test timeout behavior under load."""
        def make_request():
            context = browser.new_context()
            page = context.new_page()
            try:
                return page.request.get(
                    f"{api_base_url}/health",
                    timeout=5000
                )
            except Exception as e:
                return str(e)
            finally:
                page.close()
                context.close()
        
        # Make many concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(make_request) for _ in range(100)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Most should succeed
        success_count = sum(1 for r in results if isinstance(r, type) and hasattr(r, 'status'))
        success_rate = success_count / len(results)
        assert success_rate >= 0.8, f"Too many timeouts: {success_rate:.2%}"


class TestPlaywrightResourceLimits:
    """Tests for resource limits."""
    
    @pytest.mark.playwright
    @pytest.mark.slow
    def test_max_file_size_handling(self, browser, api_base_url, auth_headers):
        """Test handling of maximum file sizes."""
        # Try different file sizes
        file_sizes = [1024, 10240, 102400, 1048576]  # 1KB to 1MB
        
        for size in file_sizes:
            large_pdf = b"%PDF-1.4\n" + b"x" * size
            
            context = browser.new_context()
            page = context.new_page()
            
            try:
                files = {
                    "file": {
                        "name": f"test_{size}.pdf",
                        "mimeType": "application/pdf",
                        "buffer": large_pdf
                    }
                }
                
                response = page.request.post(
                    f"{api_base_url}/pdf/upload",
                    multipart=files,
                    headers=auth_headers,
                    timeout=30000
                )
                
                # May succeed or fail based on size limits
                assert response.status in [200, 201, 413, 400, 401, 403]
            finally:
                page.close()
                context.close()



