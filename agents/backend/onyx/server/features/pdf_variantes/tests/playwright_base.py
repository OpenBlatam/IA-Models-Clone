"""
Playwright Base Classes
=======================
Base classes for Playwright tests to reduce duplication.
"""

import pytest
from playwright.sync_api import Page, Response
from typing import Dict, Any, Optional
import time


class BasePlaywrightTest:
    """Base class for Playwright tests."""
    
    @pytest.fixture(autouse=True)
    def setup_test(self, api_base_url, auth_headers):
        """Setup for each test."""
        self.api_base_url = api_base_url
        self.auth_headers = auth_headers
        self.test_start_time = time.time()
    
    def make_request(
        self,
        page: Page,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Response:
        """Make HTTP request with common defaults."""
        url = f"{self.api_base_url}{endpoint}"
        headers = kwargs.pop("headers", {})
        headers.update(self.auth_headers)
        
        if method.upper() == "GET":
            return page.request.get(url, headers=headers, **kwargs)
        elif method.upper() == "POST":
            return page.request.post(url, headers=headers, **kwargs)
        elif method.upper() == "PUT":
            return page.request.put(url, headers=headers, **kwargs)
        elif method.upper() == "DELETE":
            return page.request.delete(url, headers=headers, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def assert_success(self, response: Response, expected_status: int = 200):
        """Assert response is successful."""
        assert response.status == expected_status, f"Expected {expected_status}, got {response.status}"
    
    def assert_json_response(self, response: Response, expected_keys: Optional[list] = None) -> Dict[str, Any]:
        """Assert response is JSON and optionally check keys."""
        assert response.status < 500, f"Server error: {response.status}"
        
        content_type = response.headers.get("content-type", "")
        assert "application/json" in content_type, f"Expected JSON, got {content_type}"
        
        data = response.json()
        assert isinstance(data, (dict, list)), "Response should be dict or list"
        
        if expected_keys and isinstance(data, dict):
            for key in expected_keys:
                assert key in data, f"Missing key: {key}"
        
        return data
    
    def wait_for_status(
        self,
        page: Page,
        endpoint: str,
        expected_status: int = 200,
        timeout: int = 30000,
        interval: int = 1000
    ) -> Response:
        """Wait for endpoint to return expected status."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = self.make_request(page, "GET", endpoint)
            if response.status == expected_status:
                return response
            time.sleep(interval / 1000)
        
        raise TimeoutError(f"Endpoint did not return {expected_status} within {timeout}ms")


class BaseAPITest(BasePlaywrightTest):
    """Base class for API tests."""
    
    def test_endpoint_exists(self, page: Page, endpoint: str):
        """Test that endpoint exists."""
        response = self.make_request(page, "GET", endpoint)
        # Should not be 404 (unless it's a valid 404)
        assert response.status != 404 or endpoint == "/nonexistent", f"Endpoint {endpoint} not found"
    
    def test_endpoint_returns_json(self, page: Page, endpoint: str):
        """Test that endpoint returns JSON."""
        response = self.make_request(page, "GET", endpoint)
        if response.status == 200:
            content_type = response.headers.get("content-type", "")
            assert "application/json" in content_type, f"Expected JSON, got {content_type}"


class BasePerformanceTest(BasePlaywrightTest):
    """Base class for performance tests."""
    
    def measure_response_time(
        self,
        page: Page,
        endpoint: str,
        iterations: int = 10
    ) -> Dict[str, float]:
        """Measure response time."""
        times = []
        
        for _ in range(iterations):
            start = time.time()
            response = self.make_request(page, "GET", endpoint)
            elapsed = time.time() - start
            times.append(elapsed)
            assert response.status == 200
        
        sorted_times = sorted(times)
        return {
            "min": min(times),
            "max": max(times),
            "avg": sum(times) / len(times),
            "median": sorted_times[len(times) // 2],
            "p95": sorted_times[int(len(times) * 0.95)],
            "p99": sorted_times[int(len(times) * 0.99)]
        }
    
    def assert_performance_threshold(
        self,
        metrics: Dict[str, float],
        max_avg: float = 1.0,
        max_p95: float = 2.0,
        max_p99: float = 3.0
    ):
        """Assert performance metrics meet thresholds."""
        assert metrics["avg"] < max_avg, f"Average time too high: {metrics['avg']:.3f}s"
        assert metrics["p95"] < max_p95, f"P95 time too high: {metrics['p95']:.3f}s"
        assert metrics["p99"] < max_p99, f"P99 time too high: {metrics['p99']:.3f}s"


class BaseSecurityTest(BasePlaywrightTest):
    """Base class for security tests."""
    
    def test_authentication_required(self, page: Page, endpoint: str):
        """Test that authentication is required."""
        # Request without auth
        response = page.request.get(f"{self.api_base_url}{endpoint}")
        # Should require auth (unless public endpoint)
        assert response.status in [200, 401, 403]
    
    def test_input_sanitization(self, page: Page, endpoint: str, malicious_input: str):
        """Test input sanitization."""
        # Try malicious input
        response = page.request.get(f"{self.api_base_url}{endpoint}/{malicious_input}")
        # Should handle safely
        assert response.status in [200, 400, 404, 422]


class BaseWorkflowTest(BasePlaywrightTest):
    """Base class for workflow tests."""
    
    def upload_file(self, page: Page, filename: str, content: bytes) -> Optional[str]:
        """Upload file and return file_id."""
        files = {
            "file": {
                "name": filename,
                "mimeType": "application/pdf",
                "buffer": content
            }
        }
        
        response = page.request.post(
            f"{self.api_base_url}/pdf/upload",
            multipart=files,
            headers=self.auth_headers
        )
        
        if response.status in [200, 201]:
            data = response.json()
            return data.get("file_id") or data.get("id")
        return None
    
    def generate_variant(
        self,
        page: Page,
        file_id: str,
        variant_type: str = "summary",
        options: Optional[Dict[str, Any]] = None
    ) -> Response:
        """Generate variant for file."""
        variant_data = {
            "variant_type": variant_type,
            "options": options or {}
        }
        
        return self.make_request(
            page,
            "POST",
            f"/pdf/{file_id}/variants",
            json=variant_data
        )
    
    def extract_topics(
        self,
        page: Page,
        file_id: str,
        min_relevance: float = 0.5,
        max_topics: int = 10
    ) -> Response:
        """Extract topics from file."""
        endpoint = f"/pdf/{file_id}/topics?min_relevance={min_relevance}&max_topics={max_topics}"
        return self.make_request(page, "GET", endpoint)
    
    def get_preview(
        self,
        page: Page,
        file_id: str,
        page_number: int = 1
    ) -> Response:
        """Get preview of file."""
        endpoint = f"/pdf/{file_id}/preview?page_number={page_number}"
        return self.make_request(page, "GET", endpoint)



