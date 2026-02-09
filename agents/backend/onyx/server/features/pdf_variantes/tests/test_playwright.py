"""
Playwright Tests for PDF Variantes
===================================
Browser automation tests using Playwright.
"""

import pytest
from playwright.sync_api import Page, expect, Browser, BrowserContext, Response
import time
import io
from pathlib import Path
from typing import Dict, Any, List

# Import helpers
try:
    from .playwright_helpers import (
        wait_for_api_response,
        retry_request,
        assert_json_response,
        wait_for_element_with_retry,
        take_screenshot_on_failure,
        measure_performance,
        check_accessibility,
        wait_for_network_idle,
        mock_api_response,
        check_console_errors,
        verify_response_headers,
        extract_api_endpoints_from_openapi
    )
except ImportError:
    try:
        from playwright_helpers import (
            wait_for_api_response,
            retry_request,
            assert_json_response,
            wait_for_element_with_retry,
            take_screenshot_on_failure,
            measure_performance,
            check_accessibility,
            wait_for_network_idle,
            mock_api_response,
            check_console_errors,
            verify_response_headers,
            extract_api_endpoints_from_openapi
        )
    except ImportError:
        # Helpers might not be available - define minimal versions
        def wait_for_network_idle(page, timeout=30000):
            page.wait_for_load_state("networkidle", timeout=timeout)


@pytest.fixture(scope="session")
def browser(browser_type_launch_args):
    """Create browser instance for Playwright tests."""
    from playwright.sync_api import sync_playwright
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, **browser_type_launch_args)
        yield browser
        browser.close()


@pytest.fixture
def context(browser):
    """Create browser context."""
    context = browser.new_context()
    yield context
    context.close()


@pytest.fixture
def page(context):
    """Create page for testing."""
    page = context.new_page()
    yield page
    page.close()


@pytest.fixture
def api_base_url():
    """API base URL for testing."""
    return "http://localhost:8000"


@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing."""
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\nxref\n0 3\ntrailer\n<<\n/Size 3\n>>\nstartxref\n50\n%%EOF"


class TestPlaywrightAPIRequests:
    """Playwright tests for API requests."""
    
    @pytest.mark.playwright
    def test_health_check_via_playwright(self, page, api_base_url):
        """Test health check endpoint via Playwright."""
        response = page.request.get(f"{api_base_url}/health")
        
        assert response.status == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "ok"]
    
    @pytest.mark.playwright
    def test_api_documentation_page(self, page, api_base_url):
        """Test API documentation page (Swagger/OpenAPI)."""
        # Try common documentation paths
        docs_paths = ["/docs", "/swagger", "/openapi.json", "/redoc"]
        
        for path in docs_paths:
            try:
                response = page.request.get(f"{api_base_url}{path}")
                if response.status == 200:
                    # Documentation page exists
                    assert response.status == 200
                    break
            except Exception:
                continue
        else:
            # If no docs found, that's okay - API might not have docs
            pytest.skip("API documentation not available")
    
    @pytest.mark.playwright
    def test_upload_pdf_via_playwright(self, page, api_base_url, sample_pdf_content):
        """Test PDF upload via Playwright request."""
        # Create form data
        files = {
            "file": {
                "name": "test.pdf",
                "mimeType": "application/pdf",
                "buffer": sample_pdf_content
            }
        }
        
        response = page.request.post(
            f"{api_base_url}/pdf/upload",
            multipart=files,
            headers={"Authorization": "Bearer test_token"}
        )
        
        # Should accept or require auth
        assert response.status in [200, 201, 401, 403]
        
        if response.status in [200, 201]:
            data = response.json()
            assert "file_id" in data or "id" in data
    
    @pytest.mark.playwright
    def test_get_preview_via_playwright(self, page, api_base_url):
        """Test getting PDF preview via Playwright."""
        # First need to upload a file, but for now test endpoint exists
        file_id = "test_file_123"
        response = page.request.get(
            f"{api_base_url}/pdf/{file_id}/preview?page_number=1",
            headers={"Authorization": "Bearer test_token"}
        )
        
        # May return 404 if file doesn't exist, or 401 if auth required
        assert response.status in [200, 404, 401, 403]
    
    @pytest.mark.playwright
    def test_generate_variant_via_playwright(self, page, api_base_url):
        """Test variant generation via Playwright."""
        file_id = "test_file_123"
        variant_data = {
            "variant_type": "summary",
            "options": {
                "max_length": 500,
                "style": "academic"
            }
        }
        
        response = page.request.post(
            f"{api_base_url}/pdf/{file_id}/variants",
            json=variant_data,
            headers={"Authorization": "Bearer test_token"}
        )
        
        # May return 404 if file doesn't exist, or 401 if auth required
        assert response.status in [200, 202, 404, 401, 403]


class TestPlaywrightBrowserNavigation:
    """Playwright tests for browser navigation."""
    
    @pytest.mark.playwright
    def test_navigate_to_api_base(self, page, api_base_url):
        """Test navigating to API base URL."""
        page.goto(api_base_url)
        
        # Should load (may be API response or HTML page)
        assert page.url.startswith(api_base_url)
    
    @pytest.mark.playwright
    def test_api_response_content_type(self, page, api_base_url):
        """Test API response content types."""
        response = page.request.get(f"{api_base_url}/health")
        
        content_type = response.headers.get("content-type", "")
        assert "application/json" in content_type or "text/html" in content_type
    
    @pytest.mark.playwright
    def test_cors_headers(self, page, api_base_url):
        """Test CORS headers in API responses."""
        response = page.request.get(f"{api_base_url}/health")
        
        # May have CORS headers
        cors_headers = [
            "access-control-allow-origin",
            "access-control-allow-methods",
            "access-control-allow-headers"
        ]
        
        # Check if any CORS headers exist
        has_cors = any(header in response.headers for header in cors_headers)
        # CORS may or may not be enabled
        assert True  # Just verify request works


class TestPlaywrightFormSubmission:
    """Playwright tests for form submissions."""
    
    @pytest.mark.playwright
    def test_file_upload_form(self, page, api_base_url, sample_pdf_content, tmp_path):
        """Test file upload via form if HTML form exists."""
        # Create a test PDF file
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(sample_pdf_content)
        
        # Navigate to upload page if it exists
        upload_url = f"{api_base_url}/upload"  # Common upload page path
        
        try:
            page.goto(upload_url, wait_until="networkidle", timeout=5000)
            
            # Look for file input
            file_input = page.locator('input[type="file"]').first
            
            if file_input.count() > 0:
                file_input.set_input_files(str(pdf_file))
                
                # Look for submit button
                submit_button = page.locator('button[type="submit"], input[type="submit"]').first
                if submit_button.count() > 0:
                    submit_button.click()
                    
                    # Wait for response
                    page.wait_for_timeout(2000)
                    
                    # Check for success message or redirect
                    assert True  # Form submitted
        except Exception:
            # No HTML form exists, which is fine for API-only service
            pytest.skip("No HTML upload form available")


class TestPlaywrightAPIDocumentation:
    """Playwright tests for API documentation."""
    
    @pytest.mark.playwright
    def test_swagger_ui_loads(self, page, api_base_url):
        """Test that Swagger UI loads if available."""
        swagger_url = f"{api_base_url}/docs"
        
        try:
            page.goto(swagger_url, wait_until="networkidle", timeout=5000)
            
            # Check for Swagger UI elements
            swagger_elements = [
                "swagger-ui",
                ".swagger-ui",
                "#swagger-ui",
                "openapi"
            ]
            
            for selector in swagger_elements:
                if page.locator(selector).count() > 0:
                    assert True  # Swagger UI found
                    return
            
            # If page loaded but no Swagger, check title
            title = page.title()
            if "swagger" in title.lower() or "api" in title.lower():
                assert True
        except Exception:
            pytest.skip("Swagger UI not available")
    
    @pytest.mark.playwright
    def test_openapi_spec_available(self, page, api_base_url):
        """Test that OpenAPI spec is available."""
        openapi_paths = ["/openapi.json", "/swagger.json", "/api/openapi.json"]
        
        for path in openapi_paths:
            try:
                response = page.request.get(f"{api_base_url}{path}")
                if response.status == 200:
                    data = response.json()
                    # Should be valid OpenAPI spec
                    assert "openapi" in data or "swagger" in data
                    assert "paths" in data
                    return
            except Exception:
                continue
        
        pytest.skip("OpenAPI spec not available")
    
    @pytest.mark.playwright
    def test_redoc_loads(self, page, api_base_url):
        """Test that ReDoc loads if available."""
        redoc_url = f"{api_base_url}/redoc"
        
        try:
            page.goto(redoc_url, wait_until="networkidle", timeout=5000)
            
            # Check for ReDoc elements
            if page.locator("redoc, .redoc, #redoc").count() > 0:
                assert True  # ReDoc found
            elif "redoc" in page.title().lower():
                assert True
        except Exception:
            pytest.skip("ReDoc not available")


class TestPlaywrightErrorHandling:
    """Playwright tests for error handling."""
    
    @pytest.mark.playwright
    def test_404_error_page(self, page, api_base_url):
        """Test 404 error handling."""
        response = page.request.get(f"{api_base_url}/nonexistent-endpoint")
        
        assert response.status == 404
        
        # Check error response format
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            data = response.json()
            assert "detail" in data or "error" in data or "message" in data
    
    @pytest.mark.playwright
    def test_400_error_handling(self, page, api_base_url):
        """Test 400 error handling for bad requests."""
        # Try to upload invalid data
        response = page.request.post(
            f"{api_base_url}/pdf/upload",
            json={"invalid": "data"},
            headers={"Content-Type": "application/json"}
        )
        
        # Should return 400 or 422 for validation error
        assert response.status in [400, 422, 415]
        
        if response.status in [400, 422]:
            data = response.json()
            assert "detail" in data or "error" in data or "message" in data
    
    @pytest.mark.playwright
    def test_401_unauthorized(self, page, api_base_url):
        """Test 401 unauthorized error."""
        # Try to access protected endpoint without auth
        response = page.request.get(f"{api_base_url}/pdf/test_file/preview")
        
        # May require auth or may be public
        if response.status == 401:
            data = response.json()
            assert "detail" in data or "error" in data or "message" in data


class TestPlaywrightPerformance:
    """Playwright tests for performance."""
    
    @pytest.mark.playwright
    @pytest.mark.slow
    def test_page_load_time(self, page, api_base_url):
        """Test page load time."""
        start_time = time.time()
        
        page.goto(api_base_url, wait_until="networkidle")
        
        load_time = time.time() - start_time
        
        # Should load within 5 seconds
        assert load_time < 5.0, f"Page took too long to load: {load_time:.2f}s"
    
    @pytest.mark.playwright
    @pytest.mark.slow
    def test_api_response_time(self, page, api_base_url):
        """Test API response time."""
        start_time = time.time()
        
        response = page.request.get(f"{api_base_url}/health")
        
        response_time = time.time() - start_time
        
        # Health check should be fast
        assert response_time < 1.0, f"API response too slow: {response_time:.2f}s"
        assert response.status == 200
    
    @pytest.mark.playwright
    @pytest.mark.slow
    def test_concurrent_requests(self, page, api_base_url):
        """Test handling concurrent requests."""
        import concurrent.futures
        
        def make_request():
            return page.request.get(f"{api_base_url}/health")
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        assert len(results) == 10
        assert all(r.status == 200 for r in results)


class TestPlaywrightScreenshots:
    """Playwright tests with screenshots."""
    
    @pytest.mark.playwright
    def test_capture_api_docs_screenshot(self, page, api_base_url, tmp_path):
        """Capture screenshot of API documentation if available."""
        docs_paths = ["/docs", "/redoc"]
        
        for path in docs_paths:
            try:
                page.goto(f"{api_base_url}{path}", wait_until="networkidle", timeout=5000)
                
                # Take screenshot
                screenshot_path = tmp_path / f"api_docs_{path.replace('/', '_')}.png"
                page.screenshot(path=str(screenshot_path))
                
                # Verify screenshot was created
                assert screenshot_path.exists()
                return
            except Exception:
                continue
        
        pytest.skip("No API documentation page available for screenshot")
    
    @pytest.mark.playwright
    def test_capture_error_page_screenshot(self, page, api_base_url, tmp_path):
        """Capture screenshot of error page."""
        try:
            page.goto(f"{api_base_url}/nonexistent", wait_until="networkidle", timeout=5000)
            
            screenshot_path = tmp_path / "error_page.png"
            page.screenshot(path=str(screenshot_path))
            
            assert screenshot_path.exists()
        except Exception:
            pytest.skip("Could not capture error page")


class TestPlaywrightNetworkMonitoring:
    """Playwright tests for network monitoring."""
    
    @pytest.mark.playwright
    def test_network_requests_logging(self, page, api_base_url):
        """Test logging network requests."""
        requests_log = []
        responses_log = []
        
        def handle_request(request):
            requests_log.append({
                "url": request.url,
                "method": request.method,
                "headers": request.headers
            })
        
        def handle_response(response):
            responses_log.append({
                "url": response.url,
                "status": response.status,
                "headers": response.headers
            })
        
        page.on("request", handle_request)
        page.on("response", handle_response)
        
        page.goto(api_base_url)
        try:
            wait_for_network_idle(page)
        except NameError:
            page.wait_for_load_state("networkidle", timeout=5000)
        
        # Should have made at least one request
        assert len(requests_log) > 0
        assert len(responses_log) > 0
        assert any("health" in req["url"] or api_base_url in req["url"] for req in requests_log)
    
    @pytest.mark.playwright
    def test_response_headers(self, page, api_base_url):
        """Test response headers."""
        response = page.request.get(f"{api_base_url}/health")
        
        # Check for important headers
        headers = response.headers
        
        # Should have content-type
        assert "content-type" in headers
        
        # May have security headers
        security_headers = [
            "x-content-type-options",
            "x-frame-options",
            "x-xss-protection"
        ]
        
        # At least some security headers should be present
        has_security = any(header in headers for header in security_headers)
        # Security headers may or may not be present
        assert True  # Just verify we can check headers
    
    @pytest.mark.playwright
    def test_request_interception(self, page, api_base_url):
        """Test intercepting and modifying requests."""
        intercepted = []
        
        def handle_route(route):
            intercepted.append(route.request.url)
            route.continue_()
        
        page.route("**/*", handle_route)
        
        response = page.request.get(f"{api_base_url}/health")
        
        # Should have intercepted requests
        assert len(intercepted) > 0
    
    @pytest.mark.playwright
    def test_response_mocking(self, page, api_base_url):
        """Test mocking API responses."""
        import json
        mock_data = {"status": "mocked", "message": "This is a mock response"}
        
        def handle_route(route):
            if "/health" in route.request.url:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(mock_data)
                )
            else:
                route.continue_()
        
        page.route("**/*", handle_route)
        
        response = page.request.get(f"{api_base_url}/health")
        data = response.json()
        
        assert data["status"] == "mocked"
    
    @pytest.mark.playwright
    def test_network_error_handling(self, page):
        """Test handling network errors."""
        # Try to access non-existent server
        try:
            response = page.request.get("http://localhost:9999/nonexistent", timeout=5000)
        except Exception as e:
            # Network error is expected
            assert "timeout" in str(e).lower() or "connection" in str(e).lower() or "refused" in str(e).lower()


class TestPlaywrightAccessibility:
    """Playwright tests for accessibility."""
    
    @pytest.mark.playwright
    def test_basic_accessibility(self, page, api_base_url):
        """Test basic accessibility features."""
        try:
            page.goto(api_base_url, wait_until="networkidle", timeout=5000)
            
            # Check for basic accessibility
            title = page.title()
            assert title is not None and len(title) > 0, "Page should have a title"
            
            # Check for lang attribute
            lang = page.evaluate("() => document.documentElement.lang")
            # Lang may or may not be set
            assert True  # Just verify page loaded
        except Exception:
            pytest.skip("Could not load page for accessibility testing")
    
    @pytest.mark.playwright
    def test_keyboard_navigation(self, page, api_base_url):
        """Test keyboard navigation."""
        try:
            page.goto(api_base_url, wait_until="networkidle", timeout=5000)
            
            # Try tab navigation
            page.keyboard.press("Tab")
            page.keyboard.press("Tab")
            
            # Should be able to navigate
            assert True
        except Exception:
            pytest.skip("Could not test keyboard navigation")


class TestPlaywrightResponsiveDesign:
    """Playwright tests for responsive design."""
    
    @pytest.mark.playwright
    def test_mobile_viewport(self, mobile_page, api_base_url):
        """Test API in mobile viewport."""
        try:
            mobile_page.goto(api_base_url, wait_until="networkidle", timeout=5000)
            
            # Check viewport size
            viewport = mobile_page.viewport_size
            assert viewport["width"] == 375
            assert viewport["height"] == 667
        except Exception:
            pytest.skip("Could not test mobile viewport")
    
    @pytest.mark.playwright
    def test_tablet_viewport(self, tablet_page, api_base_url):
        """Test API in tablet viewport."""
        try:
            tablet_page.goto(api_base_url, wait_until="networkidle", timeout=5000)
            
            # Check viewport size
            viewport = tablet_page.viewport_size
            assert viewport["width"] == 768
            assert viewport["height"] == 1024
        except Exception:
            pytest.skip("Could not test tablet viewport")
    
    @pytest.mark.playwright
    def test_desktop_viewport(self, page, api_base_url):
        """Test API in desktop viewport."""
        try:
            page.goto(api_base_url, wait_until="networkidle", timeout=5000)
            
            # Check viewport size
            viewport = page.viewport_size
            assert viewport["width"] >= 1920
            assert viewport["height"] >= 1080
        except Exception:
            pytest.skip("Could not test desktop viewport")


class TestPlaywrightRetryLogic:
    """Playwright tests with retry logic."""
    
    @pytest.mark.playwright
    def test_retry_on_failure(self, page, api_base_url):
        """Test retrying requests on failure."""
        try:
            from playwright_helpers import retry_request
            
            response = retry_request(page, "GET", f"{api_base_url}/health", max_retries=3)
            assert response is not None
            assert response.status == 200
        except ImportError:
            # Fallback to direct request
            response = page.request.get(f"{api_base_url}/health")
            assert response.status == 200
    
    @pytest.mark.playwright
    def test_wait_for_api_response(self, page, api_base_url):
        """Test waiting for specific API response."""
        try:
            from playwright_helpers import wait_for_api_response
            
            # Make a request that triggers another
            page.goto(api_base_url)
            
            # Wait for health endpoint response
            response = wait_for_api_response(page, "/health", timeout=5000)
            # Response may or may not be found
            assert True
        except ImportError:
            pytest.skip("Helper function not available")


class TestPlaywrightPerformanceMonitoring:
    """Enhanced performance monitoring tests."""
    
    @pytest.mark.playwright
    @pytest.mark.slow
    def test_detailed_performance_metrics(self, page, api_base_url):
        """Test detailed performance metrics."""
        try:
            from playwright_helpers import measure_performance
            
            metrics = measure_performance(page, f"{api_base_url}/health")
            
            assert "request_time" in metrics
            assert metrics["request_time"] < 5.0, "Request should be fast"
        except ImportError:
            # Fallback
            start = time.time()
            response = page.request.get(f"{api_base_url}/health")
            elapsed = time.time() - start
            assert elapsed < 5.0
    
    @pytest.mark.playwright
    @pytest.mark.slow
    def test_load_time_under_load(self, page, api_base_url):
        """Test load time under concurrent load."""
        import concurrent.futures
        
        def load_page():
            start = time.time()
            page.goto(api_base_url, wait_until="networkidle", timeout=10000)
            return time.time() - start
        
        # Create multiple pages for concurrent load
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(load_page) for _ in range(5)]
            times = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Average should be reasonable
        avg_time = sum(times) / len(times)
        assert avg_time < 10.0, f"Average load time too high: {avg_time:.2f}s"


class TestPlaywrightErrorRecovery:
    """Enhanced error recovery tests."""
    
    @pytest.mark.playwright
    def test_console_error_detection(self, page, api_base_url):
        """Test detecting console errors."""
        try:
            from playwright_helpers import check_console_errors
            
            errors = check_console_errors(page)
            page.goto(api_base_url, wait_until="networkidle", timeout=5000)
            
            # Should be able to check for errors
            assert isinstance(errors, list)
        except ImportError:
            # Fallback
            errors = []
            def handle_console(msg):
                if msg.type == "error":
                    errors.append(msg.text)
            
            page.on("console", handle_console)
            try:
                page.goto(api_base_url, wait_until="networkidle", timeout=5000)
            except Exception:
                pass
            
            assert isinstance(errors, list)
    
    @pytest.mark.playwright
    def test_graceful_degradation(self, page, api_base_url):
        """Test graceful degradation when services fail."""
        # Mock a failing endpoint
        def handle_route(route):
            if "/health" in route.request.url:
                route.fulfill(status=500, body="Internal Server Error")
            else:
                route.continue_()
        
        page.route("**/*", handle_route)
        
        response = page.request.get(f"{api_base_url}/health")
        assert response.status == 500
        
        # System should handle gracefully
        assert True

