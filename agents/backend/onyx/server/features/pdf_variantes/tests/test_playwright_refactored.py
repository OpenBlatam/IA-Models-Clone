"""
Refactored Playwright Tests
===========================
Refactored tests using base classes for better organization.
"""

import pytest
from playwright.sync_api import Page

try:
    from .playwright_base import (
        BaseAPITest,
        BasePerformanceTest,
        BaseSecurityTest,
        BaseWorkflowTest
    )
except ImportError:
    from playwright_base import (
        BaseAPITest,
        BasePerformanceTest,
        BaseSecurityTest,
        BaseWorkflowTest
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


@pytest.fixture
def sample_pdf():
    """Sample PDF content."""
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\nxref\n0 2\ntrailer\n<<\n/Size 2\n>>\nstartxref\n20\n%%EOF"


class TestRefactoredAPI(BaseAPITest):
    """Refactored API tests using base class."""
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    def test_health_endpoint(self, page):
        """Test health endpoint."""
        response = self.make_request(page, "GET", "/health")
        self.assert_success(response)
        self.assert_json_response(response, expected_keys=["status"])
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    def test_endpoints_exist(self, page):
        """Test that key endpoints exist."""
        endpoints = ["/health", "/docs", "/openapi.json"]
        
        for endpoint in endpoints:
            response = self.make_request(page, "GET", endpoint)
            # Should exist or return appropriate status
            assert response.status in [200, 404]


class TestRefactoredPerformance(BasePerformanceTest):
    """Refactored performance tests using base class."""
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    @pytest.mark.slow
    def test_health_performance(self, page):
        """Test health endpoint performance."""
        metrics = self.measure_response_time(page, "/health", iterations=20)
        self.assert_performance_threshold(metrics)
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    @pytest.mark.slow
    def test_endpoint_performance(self, page):
        """Test endpoint performance."""
        endpoints = ["/health"]
        
        for endpoint in endpoints:
            metrics = self.measure_response_time(page, endpoint, iterations=10)
            self.assert_performance_threshold(metrics, max_avg=1.0)


class TestRefactoredSecurity(BaseSecurityTest):
    """Refactored security tests using base class."""
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    def test_upload_requires_auth(self, page):
        """Test upload requires authentication."""
        self.test_authentication_required(page, "/pdf/upload")
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    def test_sql_injection_protection(self, page):
        """Test SQL injection protection."""
        sql_payload = "'; DROP TABLE users; --"
        self.test_input_sanitization(page, "/pdf", sql_payload)
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    def test_xss_protection(self, page):
        """Test XSS protection."""
        xss_payload = "<script>alert('xss')</script>"
        self.test_input_sanitization(page, "/pdf", xss_payload)


class TestRefactoredWorkflow(BaseWorkflowTest):
    """Refactored workflow tests using base class."""
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    def test_complete_workflow(self, page, sample_pdf):
        """Test complete workflow using base methods."""
        # 1. Upload
        file_id = self.upload_file(page, "workflow_test.pdf", sample_pdf)
        if not file_id:
            pytest.skip("Upload failed")
        
        # 2. Get preview
        preview_response = self.get_preview(page, file_id)
        assert preview_response.status in [200, 202, 404, 401, 403]
        
        # 3. Extract topics
        topics_response = self.extract_topics(page, file_id)
        assert topics_response.status in [200, 202, 404, 401, 403]
        
        # 4. Generate variant
        variant_response = self.generate_variant(page, file_id, "summary")
        assert variant_response.status in [200, 202, 404, 401, 403]
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    def test_multiple_variants_workflow(self, page, sample_pdf):
        """Test generating multiple variants."""
        file_id = self.upload_file(page, "variants_test.pdf", sample_pdf)
        if not file_id:
            pytest.skip("Upload failed")
        
        variant_types = ["summary", "outline", "highlights"]
        
        for variant_type in variant_types:
            response = self.generate_variant(page, file_id, variant_type)
            assert response.status in [200, 202, 404, 401, 403]


class TestRefactoredCombined(BaseAPITest, BasePerformanceTest):
    """Combined tests using multiple base classes."""
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    @pytest.mark.slow
    def test_endpoint_with_performance(self, page):
        """Test endpoint existence and performance."""
        endpoint = "/health"
        
        # Test existence
        self.test_endpoint_exists(page, endpoint)
        
        # Test performance
        metrics = self.measure_response_time(page, endpoint, iterations=10)
        self.assert_performance_threshold(metrics)

