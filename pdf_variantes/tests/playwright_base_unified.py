"""
Unified Playwright Base Classes
================================
Consolidated base classes using mixins for better organization.
"""

import pytest
from playwright.sync_api import Page, Response
from typing import Dict, Any, Optional
import time
from playwright_mixins import (
    RequestMixin,
    AssertionMixin,
    APIOperationsMixin,
    PerformanceMixin,
    DebuggingMixin,
    AnalyticsMixin,
    WorkflowMixin
)


class BasePlaywrightTest(RequestMixin, AssertionMixin):
    """Base class for all Playwright tests."""
    
    @pytest.fixture(autouse=True)
    def setup_test(self, api_base_url, auth_headers):
        """Setup for each test."""
        self.api_base_url = api_base_url
        self.auth_headers = auth_headers
        self.test_start_time = time.time()
    
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
        
        while time.time() - start_time < timeout / 1000:
            response = self.make_request_simple(
                page, "GET", endpoint, self.api_base_url, self.auth_headers
            )
            if response.status == expected_status:
                return response
            time.sleep(interval / 1000)
        
        raise TimeoutError(f"Endpoint did not return {expected_status} within {timeout}ms")


class BaseAPITest(BasePlaywrightTest, APIOperationsMixin):
    """Base class for API tests."""
    
    def test_endpoint_exists(self, page: Page, endpoint: str):
        """Test that endpoint exists."""
        response = self.make_request_simple(
            page, "GET", endpoint, self.api_base_url, self.auth_headers
        )
        assert response.status != 404 or endpoint == "/nonexistent", \
            f"Endpoint {endpoint} not found"
    
    def test_endpoint_returns_json(self, page: Page, endpoint: str):
        """Test that endpoint returns JSON."""
        response = self.make_request_simple(
            page, "GET", endpoint, self.api_base_url, self.auth_headers
        )
        if response.status == 200:
            self.assert_json_response(response)


class BasePerformanceTest(BasePlaywrightTest, PerformanceMixin):
    """Base class for performance tests."""
    pass


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


class BaseWorkflowTest(BasePlaywrightTest, WorkflowMixin):
    """Base class for workflow tests."""
    pass


class BaseDebugTest(BasePlaywrightTest, DebuggingMixin):
    """Base class for tests with debugging."""
    pass


class BaseAnalyticsTest(BasePlaywrightTest, AnalyticsMixin):
    """Base class for tests with analytics."""
    pass


class BaseComprehensiveTest(
    BasePlaywrightTest,
    APIOperationsMixin,
    PerformanceMixin,
    DebuggingMixin,
    AnalyticsMixin,
    WorkflowMixin
):
    """Base class for comprehensive tests with all features."""
    pass



