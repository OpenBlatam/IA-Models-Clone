"""
Refactored Playwright Tests V2
================================
Tests using the new mixin-based system.
"""

import pytest
from playwright.sync_api import Page
from playwright_base_unified import (
    BasePlaywrightTest,
    BaseAPITest,
    BasePerformanceTest,
    BaseWorkflowTest,
    BaseDebugTest,
    BaseComprehensiveTest
)
from fixtures_common import api_base_url, auth_headers, sample_pdf


class TestRefactoredV2Simple(BasePlaywrightTest):
    """Simple tests using base class."""
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    def test_health_simple(self, page: Page, api_base_url, auth_headers):
        """Test health using simple base class."""
        response = self.make_request_simple(
            page, "GET", "/health", api_base_url, auth_headers
        )
        self.assert_success(response)
        data = self.assert_json_response(response)
        assert "status" in data


class TestRefactoredV2API(BaseAPITest):
    """API tests using API base class."""
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    def test_upload_refactored(self, page: Page, api_base_url, auth_headers, sample_pdf):
        """Test upload using API base class."""
        result = self.upload_pdf(
            page, api_base_url, "test.pdf", sample_pdf, auth_headers
        )
        file_id = self.get_file_id_from_response(result)
        assert file_id is not None
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    def test_variant_refactored(self, page: Page, api_base_url, auth_headers, sample_pdf):
        """Test variant generation using API base class."""
        # Upload first
        upload_result = self.upload_pdf(
            page, api_base_url, "variant_test.pdf", sample_pdf, auth_headers
        )
        file_id = self.get_file_id_from_response(upload_result)
        if not file_id:
            pytest.skip("Upload failed")
        
        # Generate variant
        variant_result = self.generate_variant(
            page, api_base_url, file_id, "summary", auth_headers
        )
        assert variant_result is not None
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    def test_topics_refactored(self, page: Page, api_base_url, auth_headers, sample_pdf):
        """Test topic extraction using API base class."""
        # Upload first
        upload_result = self.upload_pdf(
            page, api_base_url, "topics_test.pdf", sample_pdf, auth_headers
        )
        file_id = self.get_file_id_from_response(upload_result)
        if not file_id:
            pytest.skip("Upload failed")
        
        # Extract topics
        topics_result = self.extract_topics(page, api_base_url, file_id, auth_headers)
        assert topics_result is not None


class TestRefactoredV2Performance(BasePerformanceTest):
    """Performance tests using performance base class."""
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    @pytest.mark.performance
    def test_response_time_refactored(self, page: Page, api_base_url, auth_headers):
        """Test response time using performance base class."""
        metrics = self.measure_response_time(
            page, api_base_url, "/health", auth_headers, iterations=5
        )
        
        assert "avg" in metrics
        assert "p95" in metrics
        assert metrics["avg"] < 1.0  # Should be fast
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    @pytest.mark.performance
    def test_performance_threshold_refactored(self, page: Page, api_base_url, auth_headers):
        """Test performance threshold using performance base class."""
        metrics = self.measure_response_time(
            page, api_base_url, "/health", auth_headers, iterations=10
        )
        
        self.assert_performance_threshold(
            metrics,
            max_avg=1.0,
            max_p95=2.0,
            max_p99=3.0
        )


class TestRefactoredV2Workflow(BaseWorkflowTest):
    """Workflow tests using workflow base class."""
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    def test_complete_workflow_refactored(self, page: Page, api_base_url, auth_headers, sample_pdf):
        """Test complete workflow using workflow base class."""
        result = self.complete_workflow(
            page, api_base_url, "workflow_test.pdf", sample_pdf, auth_headers
        )
        
        assert "file_id" in result
        assert "upload" in result
        assert "variant" in result
        assert "topics" in result
        assert "preview" in result


class TestRefactoredV2Debug(BaseDebugTest):
    """Debug tests using debug base class."""
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    @pytest.mark.debug
    def test_with_debug_refactored(self, page: Page, api_base_url):
        """Test with debugging using debug base class."""
        debugger = self.setup_debugger(page)
        page.goto(api_base_url)
        
        debug_file = self.capture_debug_info(page, "test_debug_refactored")
        from pathlib import Path
        assert Path(debug_file).exists()


class TestRefactoredV2Comprehensive(BaseComprehensiveTest):
    """Comprehensive tests using all features."""
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    def test_comprehensive_workflow(self, page: Page, api_base_url, auth_headers, sample_pdf):
        """Test using all features from comprehensive base class."""
        # Setup analytics
        analytics = self.setup_analytics()
        
        # Complete workflow
        result = self.complete_workflow(
            page, api_base_url, "comprehensive_test.pdf", sample_pdf, auth_headers
        )
        
        # Measure performance
        metrics = self.measure_response_time(
            page, api_base_url, "/health", auth_headers, iterations=5
        )
        
        # Record metrics
        import time
        duration = time.time() - self.test_start_time
        self.record_test_metrics(
            analytics,
            "test_comprehensive_workflow",
            duration,
            "passed",
            [{"duration": metrics["avg"], "status": 200}]
        )
        
        assert "file_id" in result
        assert metrics["avg"] < 1.0



