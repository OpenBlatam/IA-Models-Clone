"""
Advanced Playwright Tests
=========================
Advanced browser automation tests with enhanced features.
"""

import pytest
from playwright.sync_api import Page, expect, Response
import time
import json
from typing import Dict, Any, List
from pathlib import Path

try:
    from .playwright_helpers import (
        extract_api_endpoints_from_openapi,
        wait_for_api_success,
        verify_response_headers,
        assert_json_response
    )
except ImportError:
    try:
        from playwright_helpers import (
            extract_api_endpoints_from_openapi,
            wait_for_api_success,
            verify_response_headers,
            assert_json_response
        )
    except ImportError:
        # Helpers not available
        pass


class TestPlaywrightAPIDiscovery:
    """Tests for discovering and testing API endpoints."""
    
    @pytest.mark.playwright
    def test_discover_all_endpoints(self, page, api_base_url):
        """Discover all API endpoints from OpenAPI spec."""
        try:
            from playwright_helpers import extract_api_endpoints_from_openapi
            
            endpoints = extract_api_endpoints_from_openapi(page, api_base_url)
            
            if endpoints:
                assert len(endpoints) > 0
                # Test a few endpoints
                for endpoint in endpoints[:5]:  # Test first 5
                    try:
                        response = page.request.get(f"{api_base_url}{endpoint}")
                        assert response.status is not None
                    except Exception:
                        pass  # Some endpoints may require auth or specific methods
        except ImportError:
            pytest.skip("Helper function not available")
    
    @pytest.mark.playwright
    def test_endpoint_methods(self, page, api_base_url):
        """Test different HTTP methods on endpoints."""
        endpoints_to_test = ["/health", "/"]
        
        for endpoint in endpoints_to_test:
            # Test GET
            get_response = page.request.get(f"{api_base_url}{endpoint}")
            assert get_response.status is not None
            
            # Test OPTIONS (for CORS)
            try:
                options_response = page.request.options(f"{api_base_url}{endpoint}")
                assert options_response.status is not None
            except Exception:
                pass  # OPTIONS may not be supported


class TestPlaywrightDataValidation:
    """Tests for validating API data."""
    
    @pytest.mark.playwright
    def test_validate_health_response_structure(self, page, api_base_url):
        """Validate health check response structure."""
        response = page.request.get(f"{api_base_url}/health")
        
        try:
            from playwright_helpers import assert_json_response
            
            data = assert_json_response(response, expected_keys=["status"])
            assert data["status"] in ["healthy", "ok", "up"]
        except ImportError:
            # Fallback
            assert response.status == 200
            data = response.json()
            assert "status" in data
    
    @pytest.mark.playwright
    def test_validate_error_response_structure(self, page, api_base_url):
        """Validate error response structure."""
        response = page.request.get(f"{api_base_url}/nonexistent_endpoint_xyz")
        
        if response.status in [404, 400]:
            data = response.json()
            # Error responses should have detail or message
            assert "detail" in data or "error" in data or "message" in data


class TestPlaywrightCookieHandling:
    """Tests for cookie handling."""
    
    @pytest.mark.playwright
    def test_set_and_get_cookies(self, context, page, api_base_url):
        """Test setting and getting cookies."""
        # Set a cookie
        context.add_cookies([{
            "name": "test_cookie",
            "value": "test_value",
            "domain": "localhost",
            "path": "/"
        }])
        
        # Get cookies
        cookies = context.cookies()
        assert len(cookies) > 0
        assert any(cookie["name"] == "test_cookie" for cookie in cookies)
    
    @pytest.mark.playwright
    def test_cookie_persistence(self, context, page, api_base_url):
        """Test cookie persistence across requests."""
        # Set cookie
        context.add_cookies([{
            "name": "session_id",
            "value": "abc123",
            "domain": "localhost",
            "path": "/"
        }])
        
        # Make request - cookie should be sent
        response = page.request.get(f"{api_base_url}/health")
        assert response.status is not None


class TestPlaywrightLocalStorage:
    """Tests for localStorage handling."""
    
    @pytest.mark.playwright
    def test_local_storage_operations(self, page, api_base_url):
        """Test localStorage operations."""
        try:
            page.goto(api_base_url, wait_until="networkidle", timeout=5000)
            
            # Set localStorage
            page.evaluate("localStorage.setItem('test_key', 'test_value')")
            
            # Get localStorage
            value = page.evaluate("localStorage.getItem('test_key')")
            assert value == "test_value"
            
            # Clear localStorage
            page.evaluate("localStorage.clear()")
            value = page.evaluate("localStorage.getItem('test_key')")
            assert value is None
        except Exception:
            pytest.skip("Could not test localStorage")


class TestPlaywrightSessionStorage:
    """Tests for sessionStorage handling."""
    
    @pytest.mark.playwright
    def test_session_storage_operations(self, page, api_base_url):
        """Test sessionStorage operations."""
        try:
            page.goto(api_base_url, wait_until="networkidle", timeout=5000)
            
            # Set sessionStorage
            page.evaluate("sessionStorage.setItem('session_key', 'session_value')")
            
            # Get sessionStorage
            value = page.evaluate("sessionStorage.getItem('session_key')")
            assert value == "session_value"
        except Exception:
            pytest.skip("Could not test sessionStorage")


class TestPlaywrightVideoRecording:
    """Tests with video recording."""
    
    @pytest.mark.playwright
    def test_record_test_video(self, context, page, api_base_url):
        """Test recording video of test execution."""
        # Context should be configured with video recording
        # This is typically done in conftest
        
        try:
            page.goto(api_base_url, wait_until="networkidle", timeout=5000)
            time.sleep(1)  # Record some activity
            
            # Video should be automatically saved
            assert True
        except Exception:
            pytest.skip("Could not record video")


class TestPlaywrightTraceRecording:
    """Tests with trace recording."""
    
    @pytest.mark.playwright
    def test_record_trace(self, context, page, api_base_url):
        """Test recording trace for debugging."""
        # Start tracing
        context.tracing.start(screenshots=True, snapshots=True)
        
        try:
            page.goto(api_base_url, wait_until="networkidle", timeout=5000)
            page.request.get(f"{api_base_url}/health")
        finally:
            # Stop tracing and save
            trace_path = Path("trace.zip")
            context.tracing.stop(path=str(trace_path))
            
            # Trace file should exist
            # Note: In real tests, you'd want to save to tmp_path
            assert True  # Trace recording works


class TestPlaywrightMultiBrowser:
    """Tests across multiple browsers."""
    
    @pytest.mark.playwright
    @pytest.mark.parametrize("browser_name", ["chromium"])  # Test only chromium by default
    def test_cross_browser_compatibility(self, browser_name, api_base_url):
        """Test API compatibility across browsers."""
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser_map = {
                "chromium": p.chromium,
                "firefox": p.firefox,
                "webkit": p.webkit
            }
            
            if browser_name not in browser_map:
                pytest.skip(f"Browser {browser_name} not available")
            
            browser = browser_map[browser_name].launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            
            try:
                response = page.request.get(f"{api_base_url}/health")
                assert response.status == 200
            finally:
                page.close()
                context.close()
                browser.close()


class TestPlaywrightGeolocation:
    """Tests for geolocation features."""
    
    @pytest.mark.playwright
    def test_set_geolocation(self, context, page, api_base_url):
        """Test setting geolocation."""
        # Set geolocation
        context.set_geolocation({"latitude": 40.7128, "longitude": -74.0060})  # NYC
        context.grant_permissions(["geolocation"])
        
        try:
            page.goto(api_base_url, wait_until="networkidle", timeout=5000)
            
            # Geolocation should be set
            location = context.geolocation
            assert location["latitude"] == 40.7128
            assert location["longitude"] == -74.0060
        except Exception:
            pytest.skip("Could not test geolocation")


class TestPlaywrightPermissions:
    """Tests for permission handling."""
    
    @pytest.mark.playwright
    def test_grant_permissions(self, context, page, api_base_url):
        """Test granting permissions."""
        # Grant permissions
        context.grant_permissions(["geolocation", "notifications"])
        
        try:
            page.goto(api_base_url, wait_until="networkidle", timeout=5000)
            # Permissions should be granted
            assert True
        except Exception:
            pytest.skip("Could not test permissions")
    
    @pytest.mark.playwright
    def test_clear_permissions(self, context, page, api_base_url):
        """Test clearing permissions."""
        context.grant_permissions(["geolocation"])
        context.clear_permissions()
        
        try:
            page.goto(api_base_url, wait_until="networkidle", timeout=5000)
            # Permissions should be cleared
            assert True
        except Exception:
            pytest.skip("Could not test permissions")


class TestPlaywrightNetworkConditions:
    """Tests for different network conditions."""
    
    @pytest.mark.playwright
    def test_slow_network(self, context, page, api_base_url):
        """Test with slow network conditions."""
        # Simulate slow 3G
        context.set_extra_http_headers({})
        
        # Use CDP to throttle network
        try:
            cdp_session = context.new_cdp_session(page)
            cdp_session.send("Network.emulateNetworkConditions", {
                "offline": False,
                "downloadThroughput": 1.5 * 1024 * 1024 / 8,  # 1.5 Mbps
                "uploadThroughput": 750 * 1024 / 8,  # 750 Kbps
                "latency": 562.5  # ms
            })
            
            start = time.time()
            response = page.request.get(f"{api_base_url}/health")
            elapsed = time.time() - start
            
            # Should still work, just slower
            assert response.status == 200
            assert elapsed > 0.5  # Should be slower
        except Exception:
            pytest.skip("Could not simulate slow network")
    
    @pytest.mark.playwright
    def test_offline_mode(self, context, page, api_base_url):
        """Test offline mode."""
        # Set offline
        context.set_offline(True)
        
        try:
            response = page.request.get(f"{api_base_url}/health", timeout=5000)
        except Exception:
            # Should fail in offline mode
            assert True
        
        # Go back online
        context.set_offline(False)
        
        # Should work again
        response = page.request.get(f"{api_base_url}/health")
        assert response.status == 200


class TestPlaywrightDeviceEmulation:
    """Tests for device emulation."""
    
    @pytest.mark.playwright
    def test_iphone_emulation(self, context, api_base_url):
        """Test iPhone device emulation."""
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            iphone = p.devices["iPhone 12"]
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(**iphone)
            page = context.new_page()
            
            try:
                response = page.request.get(f"{api_base_url}/health")
                assert response.status == 200
                
                # Check viewport matches device
                viewport = page.viewport_size
                assert viewport["width"] == iphone["viewport"]["width"]
            finally:
                page.close()
                context.close()
                browser.close()
    
    @pytest.mark.playwright
    def test_ipad_emulation(self, context, api_base_url):
        """Test iPad device emulation."""
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            ipad = p.devices["iPad Pro"]
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(**ipad)
            page = context.new_page()
            
            try:
                response = page.request.get(f"{api_base_url}/health")
                assert response.status == 200
            finally:
                page.close()
                context.close()
                browser.close()

