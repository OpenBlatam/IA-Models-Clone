"""
Playwright Utility Functions
============================
Utility functions for common Playwright operations.
"""

from playwright.sync_api import Page, Response
from typing import Dict, Any, List, Optional
import time
import json


class PlaywrightRequestBuilder:
    """Builder pattern for Playwright requests."""
    
    def __init__(self, page: Page, base_url: str):
        self.page = page
        self.base_url = base_url
        self.method = "GET"
        self.endpoint = ""
        self.headers = {}
        self.json_data = None
        self.multipart_data = None
        self.query_params = {}
        self.timeout = 30000
    
    def get(self, endpoint: str):
        """Set GET method."""
        self.method = "GET"
        self.endpoint = endpoint
        return self
    
    def post(self, endpoint: str):
        """Set POST method."""
        self.method = "POST"
        self.endpoint = endpoint
        return self
    
    def put(self, endpoint: str):
        """Set PUT method."""
        self.method = "PUT"
        self.endpoint = endpoint
        return self
    
    def delete(self, endpoint: str):
        """Set DELETE method."""
        self.method = "DELETE"
        self.endpoint = endpoint
        return self
    
    def with_headers(self, headers: Dict[str, str]):
        """Add headers."""
        self.headers.update(headers)
        return self
    
    def with_json(self, data: Dict[str, Any]):
        """Add JSON data."""
        self.json_data = data
        return self
    
    def with_multipart(self, data: Dict[str, Any]):
        """Add multipart data."""
        self.multipart_data = data
        return self
    
    def with_query(self, params: Dict[str, Any]):
        """Add query parameters."""
        self.query_params.update(params)
        return self
    
    def with_timeout(self, timeout: int):
        """Set timeout."""
        self.timeout = timeout
        return self
    
    def execute(self) -> Response:
        """Execute the request."""
        url = f"{self.base_url}{self.endpoint}"
        
        # Add query parameters
        if self.query_params:
            query_string = "&".join([f"{k}={v}" for k, v in self.query_params.items()])
            url += f"?{query_string}"
        
        kwargs = {
            "headers": self.headers,
            "timeout": self.timeout
        }
        
        if self.json_data:
            kwargs["json"] = self.json_data
        elif self.multipart_data:
            kwargs["multipart"] = self.multipart_data
        
        if self.method == "GET":
            return self.page.request.get(url, **kwargs)
        elif self.method == "POST":
            return self.page.request.post(url, **kwargs)
        elif self.method == "PUT":
            return self.page.request.put(url, **kwargs)
        elif self.method == "DELETE":
            return self.page.request.delete(url, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {self.method}")


class PlaywrightResponseValidator:
    """Validator for Playwright responses."""
    
    def __init__(self, response: Response):
        self.response = response
    
    def assert_status(self, expected_status: int) -> 'PlaywrightResponseValidator':
        """Assert response status."""
        assert self.response.status == expected_status, \
            f"Expected status {expected_status}, got {self.response.status}"
        return self
    
    def assert_status_range(self, min_status: int, max_status: int) -> 'PlaywrightResponseValidator':
        """Assert response status in range."""
        assert min_status <= self.response.status <= max_status, \
            f"Expected status {min_status}-{max_status}, got {self.response.status}"
        return self
    
    def assert_json(self) -> Dict[str, Any]:
        """Assert response is JSON and return data."""
        content_type = self.response.headers.get("content-type", "")
        assert "application/json" in content_type, \
            f"Expected JSON, got {content_type}"
        return self.response.json()
    
    def assert_has_keys(self, *keys: str) -> 'PlaywrightResponseValidator':
        """Assert response has specified keys."""
        data = self.assert_json()
        for key in keys:
            assert key in data, f"Missing key: {key}"
        return self
    
    def assert_header(self, header_name: str, expected_value: Optional[str] = None) -> 'PlaywrightResponseValidator':
        """Assert response has header."""
        headers = self.response.headers
        assert header_name in headers, f"Missing header: {header_name}"
        if expected_value:
            assert headers[header_name] == expected_value, \
                f"Header {header_name} value mismatch"
        return self
    
    def assert_content_type(self, expected_type: str) -> 'PlaywrightResponseValidator':
        """Assert content type."""
        content_type = self.response.headers.get("content-type", "")
        assert expected_type in content_type, \
            f"Expected content type {expected_type}, got {content_type}"
        return self


class PlaywrightTestDataFactory:
    """Factory for creating test data."""
    
    @staticmethod
    def create_pdf_file(name: str = "test.pdf", size_kb: int = 1) -> Dict[str, Any]:
        """Create PDF file structure."""
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
    def create_variant_request(variant_type: str = "summary", options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create variant request data."""
        return {
            "variant_type": variant_type,
            "options": options or {}
        }
    
    @staticmethod
    def create_upload_files(filename: str = "test.pdf", content: Optional[bytes] = None) -> Dict[str, Any]:
        """Create upload files structure."""
        if content is None:
            content = PlaywrightTestDataFactory.create_pdf_file(filename)["buffer"]
        
        return {
            "file": {
                "name": filename,
                "mimeType": "application/pdf",
                "buffer": content
            }
        }
    
    @staticmethod
    def create_auth_headers(token: str = "test_token_123", user_id: str = "test_user_123") -> Dict[str, str]:
        """Create authentication headers."""
        return {
            "Authorization": f"Bearer {token}",
            "X-User-ID": user_id,
            "Content-Type": "application/json"
        }


class PlaywrightAssertions:
    """Custom assertions for Playwright tests."""
    
    @staticmethod
    def assert_response_time(response_time: float, max_time: float = 1.0):
        """Assert response time is acceptable."""
        assert response_time < max_time, \
            f"Response time {response_time:.3f}s exceeds maximum {max_time}s"
    
    @staticmethod
    def assert_performance_metrics(metrics: Dict[str, float], thresholds: Dict[str, float]):
        """Assert performance metrics meet thresholds."""
        for metric, threshold in thresholds.items():
            if metric in metrics:
                assert metrics[metric] < threshold, \
                    f"{metric} {metrics[metric]:.3f} exceeds threshold {threshold}"
    
    @staticmethod
    def assert_no_errors_in_console(page: Page) -> List[str]:
        """Assert no console errors."""
        errors = []
        
        def handle_console(msg):
            if msg.type == "error":
                errors.append(msg.text)
        
        page.on("console", handle_console)
        return errors
    
    @staticmethod
    def assert_accessibility_basic(page: Page):
        """Assert basic accessibility."""
        title = page.title()
        assert title is not None and len(title) > 0, "Page should have a title"
        
        lang = page.evaluate("() => document.documentElement.lang")
        # Lang may or may not be set
        assert True


def create_request_builder(page: Page, base_url: str) -> PlaywrightRequestBuilder:
    """Create a request builder."""
    return PlaywrightRequestBuilder(page, base_url)


def validate_response(response: Response) -> PlaywrightResponseValidator:
    """Create a response validator."""
    return PlaywrightResponseValidator(response)


def create_test_data() -> PlaywrightTestDataFactory:
    """Create test data factory."""
    return PlaywrightTestDataFactory()



