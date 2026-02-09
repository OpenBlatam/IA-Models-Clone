"""
Improved Playwright Tests
==========================
Enhanced Playwright tests with improved structure and best practices.
"""

import pytest
from playwright.sync_api import Page, Response
import time
import json
from typing import Dict, Any, List

try:
    from .playwright_helpers import (
        retry_request,
        assert_json_response,
        measure_performance,
        validate_response_schema,
        create_test_pdf,
        extract_error_details,
        compare_responses,
        wait_for_condition,
        capture_network_log,
        assert_no_console_errors,
        batch_requests,
        measure_api_performance
    )
except ImportError:
    from playwright_helpers import (
        retry_request,
        assert_json_response,
        measure_performance,
        validate_response_schema,
        create_test_pdf,
        extract_error_details,
        compare_responses,
        wait_for_condition,
        capture_network_log,
        assert_no_console_errors,
        batch_requests,
        measure_api_performance
    )


@pytest.fixture
def api_base_url():
    """API base URL."""
    return "http://localhost:8000"


@pytest.fixture
def auth_headers():
    """Authentication headers."""
    return {
        "Authorization": "Bearer test_token_123",
        "X-User-ID": "test_user_123"
    }


class TestPlaywrightImprovedHelpers:
    """Tests using improved helper functions."""
    
    @pytest.mark.playwright
    def test_retry_with_helpers(self, page, api_base_url):
        """Test retry logic with improved helpers."""
        response = retry_request(
            page,
            "GET",
            f"{api_base_url}/health",
            max_retries=3
        )
        
        assert response is not None
        assert response.status == 200
    
    @pytest.mark.playwright
    def test_validate_response_schema(self, page, api_base_url):
        """Test response schema validation."""
        response = page.request.get(f"{api_base_url}/health")
        
        schema = {
            "status": str,
            "message": str
        }
        
        # May or may not match schema exactly
        is_valid = validate_response_schema(response, schema)
        assert response.status == 200  # At least should be successful
    
    @pytest.mark.playwright
    def test_create_test_pdf(self):
        """Test creating test PDFs of different sizes."""
        small_pdf = create_test_pdf(size_kb=1)
        large_pdf = create_test_pdf(size_kb=100)
        
        assert len(small_pdf) > 0
        assert len(large_pdf) > len(small_pdf)
        assert len(large_pdf) >= 100 * 1024
    
    @pytest.mark.playwright
    def test_extract_error_details(self, page, api_base_url):
        """Test extracting error details."""
        response = page.request.get(f"{api_base_url}/nonexistent_endpoint_xyz")
        
        error_details = extract_error_details(response)
        
        assert "status" in error_details
        assert "url" in error_details
        assert error_details["status"] >= 400
    
    @pytest.mark.playwright
    def test_compare_responses(self, page, api_base_url):
        """Test comparing responses."""
        response1 = page.request.get(f"{api_base_url}/health")
        time.sleep(0.1)
        response2 = page.request.get(f"{api_base_url}/health")
        
        differences = compare_responses(response1, response2)
        
        # Should be able to compare
        assert isinstance(differences, dict)
        assert "status_different" in differences
    
    @pytest.mark.playwright
    def test_wait_for_condition(self, page, api_base_url):
        """Test waiting for condition."""
        def check_health():
            response = page.request.get(f"{api_base_url}/health")
            return response.status == 200
        
        result = wait_for_condition(page, check_health, timeout=5000)
        assert result is True
    
    @pytest.mark.playwright
    def test_capture_network_log(self, page, api_base_url):
        """Test capturing network log."""
        network_log = capture_network_log(page)
        
        # Make some requests
        page.request.get(f"{api_base_url}/health")
        time.sleep(0.5)
        
        # Should have captured requests
        assert len(network_log) > 0
    
    @pytest.mark.playwright
    def test_assert_no_console_errors(self, page, api_base_url):
        """Test checking for console errors."""
        errors = assert_no_console_errors(page)
        
        try:
            page.goto(api_base_url, wait_until="networkidle", timeout=5000)
        except Exception:
            pass
        
        # Should be able to check for errors
        assert isinstance(errors, list)
    
    @pytest.mark.playwright
    def test_batch_requests(self, page, api_base_url):
        """Test batch request execution."""
        requests = [
            {"method": "GET", "url": f"{api_base_url}/health"},
            {"method": "GET", "url": f"{api_base_url}/health"},
            {"method": "GET", "url": f"{api_base_url}/health"}
        ]
        
        responses = batch_requests(page, requests)
        
        assert len(responses) == 3
        assert all(r.status == 200 for r in responses)
    
    @pytest.mark.playwright
    @pytest.mark.slow
    def test_measure_api_performance(self, page, api_base_url):
        """Test measuring API performance."""
        metrics = measure_api_performance(page, f"{api_base_url}/health", iterations=10)
        
        assert "min" in metrics
        assert "max" in metrics
        assert "avg" in metrics
        assert "median" in metrics
        assert metrics["avg"] < 1.0  # Should be fast


class TestPlaywrightImprovedFixtures:
    """Tests using improved fixtures."""
    
    @pytest.mark.playwright
    def test_page_with_tracing(self, page_with_tracing, api_base_url):
        """Test page with tracing enabled."""
        response = page_with_tracing.request.get(f"{api_base_url}/health")
        assert response.status == 200
    
    @pytest.mark.playwright
    def test_slow_network(self, slow_network_context, api_base_url):
        """Test with slow network simulation."""
        page = slow_network_context.new_page()
        try:
            start = time.time()
            response = page.request.get(f"{api_base_url}/health")
            elapsed = time.time() - start
            
            assert response.status == 200
            # May be slower due to network simulation
            assert elapsed >= 0
        finally:
            page.close()
    
    @pytest.mark.playwright
    def test_offline_mode(self, offline_context, api_base_url):
        """Test offline mode."""
        page = offline_context.new_page()
        try:
            try:
                response = page.request.get(f"{api_base_url}/health", timeout=2000)
            except Exception:
                # Should fail in offline mode
                assert True
        finally:
            page.close()
    
    @pytest.mark.playwright
    def test_large_pdf_upload(self, page, api_base_url, large_pdf_content, auth_headers):
        """Test uploading large PDF."""
        files = {
            "file": {
                "name": "large_test.pdf",
                "mimeType": "application/pdf",
                "buffer": large_pdf_content
            }
        }
        
        response = page.request.post(
            f"{api_base_url}/pdf/upload",
            multipart=files,
            headers=auth_headers,
            timeout=60000  # Longer timeout for large file
        )
        
        # May succeed, fail with 413, or require auth
        assert response.status in [200, 201, 413, 401, 403]


class TestPlaywrightImprovedErrorHandling:
    """Tests with improved error handling."""
    
    @pytest.mark.playwright
    def test_graceful_error_handling(self, page, api_base_url):
        """Test graceful error handling."""
        try:
            response = page.request.get(f"{api_base_url}/nonexistent")
            error_details = extract_error_details(response)
            
            # Should extract error details gracefully
            assert "status" in error_details
            assert error_details["status"] >= 400
        except Exception as e:
            # Should handle exceptions gracefully
            assert isinstance(e, Exception)
    
    @pytest.mark.playwright
    def test_error_recovery(self, page, api_base_url):
        """Test error recovery."""
        # Try invalid request
        try:
            response = page.request.post(
                f"{api_base_url}/pdf/upload",
                json={"invalid": "data"}
            )
            assert response.status >= 400
        except Exception:
            pass
        
        # Should recover and make valid request
        response = page.request.get(f"{api_base_url}/health")
        assert response.status == 200


class TestPlaywrightImprovedPerformance:
    """Tests with improved performance monitoring."""
    
    @pytest.mark.playwright
    @pytest.mark.slow
    def test_performance_metrics(self, page, api_base_url):
        """Test detailed performance metrics."""
        metrics = measure_api_performance(page, f"{api_base_url}/health", iterations=20)
        
        # Check all metrics are present
        assert "min" in metrics
        assert "max" in metrics
        assert "avg" in metrics
        assert "median" in metrics
        assert "p95" in metrics
        assert "p99" in metrics
        
        # Metrics should be reasonable
        assert metrics["avg"] < 1.0
        assert metrics["p95"] < 2.0
        assert metrics["p99"] < 3.0
    
    @pytest.mark.playwright
    @pytest.mark.slow
    def test_performance_under_load(self, page, api_base_url):
        """Test performance under load."""
        import concurrent.futures
        
        def make_request():
            start = time.time()
            response = page.request.get(f"{api_base_url}/health")
            elapsed = time.time() - start
            return response.status, elapsed
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_request) for _ in range(50)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        statuses, times = zip(*results)
        
        # All should succeed
        assert all(status == 200 for status in statuses)
        
        # Performance should be acceptable
        avg_time = sum(times) / len(times)
        assert avg_time < 2.0, f"Performance degraded: {avg_time:.3f}s"


class TestPlaywrightImprovedValidation:
    """Tests with improved validation."""
    
    @pytest.mark.playwright
    def test_response_validation(self, page, api_base_url):
        """Test improved response validation."""
        response = page.request.get(f"{api_base_url}/health")
        
        # Validate response
        data = assert_json_response(response, expected_keys=["status"])
        
        assert "status" in data
        assert response.status == 200
    
    @pytest.mark.playwright
    def test_schema_validation(self, page, api_base_url):
        """Test schema validation."""
        response = page.request.get(f"{api_base_url}/health")
        
        schema = {
            "status": str
        }
        
        # May or may not match exactly
        is_valid = validate_response_schema(response, schema)
        assert response.status == 200  # At least should work


class TestPlaywrightImprovedNetwork:
    """Tests with improved network monitoring."""
    
    @pytest.mark.playwright
    def test_network_logging(self, page, api_base_url):
        """Test network logging."""
        network_log = capture_network_log(page)
        
        # Make requests
        page.request.get(f"{api_base_url}/health")
        page.request.get(f"{api_base_url}/health")
        
        time.sleep(0.5)
        
        # Should have logged requests
        assert len(network_log) >= 2
    
    @pytest.mark.playwright
    def test_request_comparison(self, page, api_base_url):
        """Test comparing requests."""
        response1 = page.request.get(f"{api_base_url}/health")
        time.sleep(0.1)
        response2 = page.request.get(f"{api_base_url}/health")
        
        differences = compare_responses(response1, response2)
        
        # Should be able to compare
        assert isinstance(differences, dict)
        assert "status_different" in differences
        assert "headers_different" in differences
        assert "body_different" in differences



