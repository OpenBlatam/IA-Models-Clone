"""
Base Test Classes for Playwright
=================================
Base classes to reduce code duplication.
"""

import pytest
from playwright.sync_api import Page, Response
from typing import Dict, Any, Optional


class BasePlaywrightTest:
    """Base class for Playwright tests."""
    
    @staticmethod
    def assert_response_success(response: Response, expected_status: int = 200) -> None:
        """Assert response is successful."""
        assert response.status == expected_status, f"Expected {expected_status}, got {response.status}"
    
    @staticmethod
    def assert_response_error(response: Response, min_status: int = 400, max_status: int = 499) -> None:
        """Assert response is an error."""
        assert min_status <= response.status <= max_status, \
            f"Expected error status {min_status}-{max_status}, got {response.status}"
    
    @staticmethod
    def assert_json_response(response: Response) -> Dict[str, Any]:
        """Assert response is JSON and return data."""
        content_type = response.headers.get("content-type", "")
        assert "application/json" in content_type, f"Expected JSON, got {content_type}"
        return response.json()
    
    @staticmethod
    def create_pdf_file(name: str = "test.pdf", size_kb: int = 1) -> Dict[str, Any]:
        """Create a test PDF file structure."""
        base_pdf = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\nxref\n0 2\ntrailer\n<<\n/Size 2\n>>\nstartxref\n20\n%%EOF"
        
        if size_kb > 1:
            padding = b" " * ((size_kb * 1024) - len(base_pdf))
            content = base_pdf + padding
        else:
            content = base_pdf
        
        return {
            "name": name,
            "mimeType": "application/pdf",
            "buffer": content
        }
    
    @staticmethod
    def make_request_with_retry(page: Page, method: str, url: str, max_retries: int = 3, **kwargs) -> Optional[Response]:
        """Make request with retry logic."""
        import time
        
        for attempt in range(max_retries):
            try:
                if method.upper() == "GET":
                    response = page.request.get(url, **kwargs)
                elif method.upper() == "POST":
                    response = page.request.post(url, **kwargs)
                elif method.upper() == "PUT":
                    response = page.request.put(url, **kwargs)
                elif method.upper() == "DELETE":
                    response = page.request.delete(url, **kwargs)
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                if response.status < 500:
                    return response
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)
        
        return None


class BaseAPITest(BasePlaywrightTest):
    """Base class for API tests."""
    
    def upload_pdf(self, page: Page, api_base_url: str, auth_headers: Dict[str, str], 
                   sample_pdf: bytes, filename: str = "test.pdf") -> Optional[Response]:
        """Helper to upload PDF."""
        files = {
            "file": {
                "name": filename,
                "mimeType": "application/pdf",
                "buffer": sample_pdf
            }
        }
        
        return page.request.post(
            f"{api_base_url}/pdf/upload",
            multipart=files,
            headers=auth_headers
        )
    
    def get_file_id_from_response(self, response: Response) -> Optional[str]:
        """Extract file_id from upload response."""
        if response.status in [200, 201]:
            data = response.json()
            return data.get("file_id") or data.get("id")
        return None
    
    def generate_variant(self, page: Page, api_base_url: str, file_id: str, 
                        variant_type: str, options: Dict[str, Any] = None,
                        auth_headers: Dict[str, str] = None) -> Response:
        """Helper to generate variant."""
        variant_data = {
            "variant_type": variant_type,
            "options": options or {}
        }
        
        return page.request.post(
            f"{api_base_url}/pdf/{file_id}/variants",
            json=variant_data,
            headers=auth_headers or {}
        )
    
    def get_topics(self, page: Page, api_base_url: str, file_id: str,
                   auth_headers: Dict[str, str] = None, **query_params) -> Response:
        """Helper to get topics."""
        query_string = "&".join([f"{k}={v}" for k, v in query_params.items()])
        url = f"{api_base_url}/pdf/{file_id}/topics"
        if query_string:
            url += f"?{query_string}"
        
        return page.request.get(url, headers=auth_headers or {})
    
    def get_preview(self, page: Page, api_base_url: str, file_id: str,
                    page_number: int = 1, auth_headers: Dict[str, str] = None) -> Response:
        """Helper to get preview."""
        return page.request.get(
            f"{api_base_url}/pdf/{file_id}/preview?page_number={page_number}",
            headers=auth_headers or {}
        )


class BaseUITest(BasePlaywrightTest):
    """Base class for UI tests."""
    
    def navigate_and_wait(self, page: Page, url: str, timeout: int = 5000) -> bool:
        """Navigate and wait for page load."""
        try:
            page.goto(url, wait_until="networkidle", timeout=timeout)
            return True
        except Exception:
            return False
    
    def find_and_click(self, page: Page, selector: str, timeout: int = 5000) -> bool:
        """Find element and click."""
        try:
            element = page.locator(selector).first
            if element.count() > 0:
                element.click(timeout=timeout)
                return True
            return False
        except Exception:
            return False
    
    def fill_form_field(self, page: Page, selector: str, value: str, timeout: int = 5000) -> bool:
        """Fill form field."""
        try:
            element = page.locator(selector).first
            if element.count() > 0:
                element.fill(value, timeout=timeout)
                return True
            return False
        except Exception:
            return False


class BaseLoadTest(BasePlaywrightTest):
    """Base class for load tests."""
    
    def make_concurrent_requests(self, browser, api_base_url: str, num_requests: int = 10,
                                 max_workers: int = 5) -> list:
        """Make concurrent requests."""
        import concurrent.futures
        
        def make_request():
            context = browser.new_context()
            page = context.new_page()
            try:
                return page.request.get(f"{api_base_url}/health")
            finally:
                page.close()
                context.close()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            return [future.result() for future in concurrent.futures.as_completed(futures)]


class BaseSecurityTest(BasePlaywrightTest):
    """Base class for security tests."""
    
    def test_injection_attempt(self, page: Page, api_base_url: str, 
                               payload: str, endpoint: str) -> Response:
        """Test injection attempt."""
        return page.request.get(f"{api_base_url}{endpoint}/{payload}")
    
    def test_authentication(self, page: Page, api_base_url: str,
                           endpoint: str, headers: Dict[str, str] = None) -> Response:
        """Test authentication."""
        return page.request.get(
            f"{api_base_url}{endpoint}",
            headers=headers or {}
        )



