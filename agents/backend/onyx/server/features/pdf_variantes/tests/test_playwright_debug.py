"""
Playwright Debug Tests
======================
Tests demonstrating debugging utilities.
"""

import pytest
from playwright.sync_api import Page
from playwright_debug import (
    create_debugger,
    troubleshoot_timeout,
    troubleshoot_performance
)
from fixtures_common import api_base_url, auth_headers, sample_pdf


class TestPlaywrightDebug:
    """Tests for debugging utilities."""
    
    @pytest.mark.playwright
    @pytest.mark.debug
    def test_debugger_capture_screenshot(self, page: Page, api_base_url):
        """Test debugger screenshot capture."""
        debugger = create_debugger(page)
        page.goto(api_base_url)
        screenshot_path = debugger.capture_screenshot("test_screenshot.png")
        assert screenshot_path.exists()
    
    @pytest.mark.playwright
    @pytest.mark.debug
    def test_debugger_network_log(self, page: Page, api_base_url):
        """Test debugger network log capture."""
        debugger = create_debugger(page)
        
        # Make some requests
        page.goto(api_base_url)
        page.request.get(f"{api_base_url}/health")
        
        network_log = debugger.capture_network_log()
        assert "total_requests" in network_log
        assert network_log["total_requests"] > 0
    
    @pytest.mark.playwright
    @pytest.mark.debug
    def test_debugger_console_log(self, page: Page, api_base_url):
        """Test debugger console log capture."""
        debugger = create_debugger(page)
        
        # Navigate and trigger console logs
        page.goto(api_base_url)
        page.evaluate("console.log('Test log'); console.error('Test error')")
        
        console_log = debugger.capture_console_log()
        assert "total_logs" in console_log
        assert console_log["total_logs"] > 0
    
    @pytest.mark.playwright
    @pytest.mark.debug
    def test_debugger_save_debug_info(self, page: Page, api_base_url):
        """Test saving complete debug info."""
        debugger = create_debugger(page)
        
        page.goto(api_base_url)
        page.request.get(f"{api_base_url}/health")
        
        debug_file = debugger.save_debug_info("test_debug")
        assert debug_file.exists()
        
        import json
        debug_data = json.loads(debug_file.read_text())
        assert "test_name" in debug_data
        assert "network" in debug_data
        assert "console" in debug_data
    
    @pytest.mark.playwright
    @pytest.mark.debug
    def test_debugger_performance_analysis(self, page: Page, api_base_url):
        """Test performance analysis."""
        debugger = create_debugger(page)
        
        page.goto(api_base_url)
        
        performance = debugger.analyze_performance()
        assert "metrics" in performance
        assert "network_requests" in performance
    
    @pytest.mark.playwright
    @pytest.mark.debug
    def test_troubleshoot_timeout(self, page: Page, api_base_url):
        """Test timeout troubleshooting."""
        page.goto(api_base_url)
        
        diagnosis = troubleshoot_timeout(page)
        assert "has_issues" in diagnosis
        assert "issues" in diagnosis
    
    @pytest.mark.playwright
    @pytest.mark.debug
    def test_troubleshoot_performance(self, page: Page, api_base_url):
        """Test performance troubleshooting."""
        page.goto(api_base_url)
        
        diagnosis = troubleshoot_performance(page)
        assert "has_issues" in diagnosis
        assert "performance_data" in diagnosis
    
    @pytest.mark.playwright
    @pytest.mark.debug
    def test_debugger_with_api_request(self, page: Page, api_base_url, auth_headers):
        """Test debugger with API request."""
        debugger = create_debugger(page)
        
        # Make API request
        response = page.request.get(
            f"{api_base_url}/health",
            headers=auth_headers
        )
        
        # Capture debug info
        network_log = debugger.capture_network_log()
        assert network_log["total_requests"] > 0
        
        # Save debug info
        debug_file = debugger.save_debug_info("api_request_debug")
        assert debug_file.exists()
    
    @pytest.mark.playwright
    @pytest.mark.debug
    def test_debugger_wait_and_debug(self, page: Page, api_base_url):
        """Test wait and debug functionality."""
        debugger = create_debugger(page)
        
        page.goto(api_base_url)
        
        # Wait for condition that will fail
        result = debugger.wait_and_debug(
            lambda: False,  # Condition that never succeeds
            timeout=1000
        )
        
        assert result is False
        # Screenshot should be captured
        assert (debugger.output_dir / "wait_failed.png").exists()



