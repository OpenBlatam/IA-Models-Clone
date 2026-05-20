"""
Refactored Playwright Tests Using Decorators
============================================
Example tests using decorators for common functionality.
"""

import pytest
from playwright.sync_api import Page
from playwright_decorators import (
    retry_on_failure,
    measure_performance,
    capture_screenshot_on_failure,
    validate_response_time,
    skip_if_api_unavailable,
    log_test_execution,
    require_auth
)
from playwright_utils import create_request_builder, validate_response
from fixtures_common import api_base_url, auth_headers, sample_pdf


class TestRefactoredWithDecorators:
    """Tests using decorators."""
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    @skip_if_api_unavailable
    @log_test_execution
    @measure_performance
    def test_health_with_decorators(self, page, api_base_url):
        """Test health with multiple decorators."""
        response = (
            create_request_builder(page, api_base_url)
            .get("/health")
            .execute()
        )
        
        validate_response(response).assert_status(200)
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    @retry_on_failure(max_retries=3, delay=1.0)
    @capture_screenshot_on_failure
    def test_upload_with_retry(self, page, api_base_url, auth_headers, sample_pdf):
        """Test upload with retry and screenshot on failure."""
        from playwright_utils import create_test_data
        
        factory = create_test_data()
        files = factory.create_upload_files("test.pdf", sample_pdf)
        
        response = (
            create_request_builder(page, api_base_url)
            .post("/pdf/upload")
            .with_headers(auth_headers)
            .with_multipart(files)
            .execute()
        )
        
        validate_response(response).assert_status_range(200, 201)
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    @validate_response_time(max_time=2.0)
    @require_auth
    def test_variant_with_timing(self, page, api_base_url, auth_headers):
        """Test variant generation with timing validation."""
        from playwright_utils import create_test_data
        
        factory = create_test_data()
        variant_request = factory.create_variant_request("summary")
        
        response = (
            create_request_builder(page, api_base_url)
            .post("/pdf/test_file_123/variants")
            .with_headers(auth_headers)
            .with_json(variant_request)
            .execute()
        )
        
        validate_response(response).assert_status_range(200, 202)
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    @measure_performance
    @log_test_execution
    def test_performance_monitored(self, page, api_base_url):
        """Test with performance monitoring."""
        response = (
            create_request_builder(page, api_base_url)
            .get("/health")
            .execute()
        )
        
        validate_response(response).assert_status(200)
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    @skip_if_api_unavailable
    @retry_on_failure(max_retries=2)
    def test_with_all_decorators(self, page, api_base_url, auth_headers):
        """Test using all decorators."""
        response = (
            create_request_builder(page, api_base_url)
            .get("/health")
            .with_headers(auth_headers)
            .execute()
        )
        
        validate_response(response).assert_status(200)



