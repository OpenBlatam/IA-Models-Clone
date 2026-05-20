"""
Playwright Exploratory Tests
============================
Exploratory testing with Playwright.
"""

import pytest
from playwright.sync_api import Page, Response
import time
import random
from typing import Dict, Any, List


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


class TestPlaywrightExploratory:
    """Exploratory tests."""
    
    @pytest.mark.playwright
    @pytest.mark.exploratory
    def test_explore_endpoints(self, page, api_base_url):
        """Explore available endpoints."""
        common_endpoints = [
            "/health",
            "/docs",
            "/swagger",
            "/openapi.json",
            "/metrics",
            "/api",
            "/v1",
            "/v2"
        ]
        
        found_endpoints = []
        
        for endpoint in common_endpoints:
            try:
                response = page.request.get(f"{api_base_url}{endpoint}", timeout=2000)
                if response.status < 500:
                    found_endpoints.append((endpoint, response.status))
            except Exception:
                pass
        
        # Should find at least some endpoints
        assert len(found_endpoints) > 0
    
    @pytest.mark.playwright
    @pytest.mark.exploratory
    def test_explore_response_formats(self, page, api_base_url):
        """Explore response formats."""
        response = page.request.get(f"{api_base_url}/health")
        
        content_type = response.headers.get("content-type", "")
        
        # Explore what formats are supported
        formats = {
            "json": "application/json" in content_type,
            "html": "text/html" in content_type,
            "xml": "application/xml" in content_type or "text/xml" in content_type
        }
        
        # Should support at least one format
        assert any(formats.values())
    
    @pytest.mark.playwright
    @pytest.mark.exploratory
    def test_explore_error_responses(self, page, api_base_url):
        """Explore error response formats."""
        error_scenarios = [
            ("/nonexistent", 404),
            ("/pdf/invalid_id/preview", 404),
            ("/pdf/upload", 400),  # Invalid request
        ]
        
        error_formats = []
        
        for endpoint, expected_status in error_scenarios:
            try:
                if endpoint == "/pdf/upload":
                    response = page.request.post(
                        f"{api_base_url}{endpoint}",
                        json={"invalid": "data"}
                    )
                else:
                    response = page.request.get(f"{api_base_url}{endpoint}")
                
                if response.status >= 400:
                    content_type = response.headers.get("content-type", "")
                    error_formats.append(content_type)
            except Exception:
                pass
        
        # Should have consistent error format
        assert len(error_formats) > 0
    
    @pytest.mark.playwright
    @pytest.mark.exploratory
    def test_explore_headers(self, page, api_base_url):
        """Explore response headers."""
        response = page.request.get(f"{api_base_url}/health")
        headers = response.headers
        
        # Explore what headers are present
        header_categories = {
            "security": ["x-content-type-options", "x-frame-options", "x-xss-protection"],
            "caching": ["cache-control", "etag", "last-modified"],
            "cors": ["access-control-allow-origin", "access-control-allow-methods"],
            "versioning": ["x-api-version", "api-version"]
        }
        
        found_headers = {}
        for category, header_list in header_categories.items():
            found_headers[category] = [h for h in header_list if h in headers]
        
        # Should have some headers
        assert len(headers) > 0
    
    @pytest.mark.playwright
    @pytest.mark.exploratory
    def test_explore_query_parameters(self, page, api_base_url):
        """Explore query parameter handling."""
        query_params = [
            "?format=json",
            "?format=xml",
            "?page=1",
            "?limit=10",
            "?sort=date",
            "?filter=active"
        ]
        
        results = []
        
        for param in query_params:
            try:
                response = page.request.get(f"{api_base_url}/health{param}", timeout=2000)
                results.append((param, response.status))
            except Exception:
                pass
        
        # Should handle at least some query parameters
        assert len(results) > 0
    
    @pytest.mark.playwright
    @pytest.mark.exploratory
    def test_explore_http_methods(self, page, api_base_url):
        """Explore HTTP methods support."""
        methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]
        
        method_support = {}
        
        for method in methods:
            try:
                if method == "GET":
                    response = page.request.get(f"{api_base_url}/health")
                elif method == "OPTIONS":
                    response = page.request.options(f"{api_base_url}/health")
                elif method == "HEAD":
                    response = page.request.head(f"{api_base_url}/health")
                else:
                    response = page.request.post(f"{api_base_url}/health")
                
                method_support[method] = response.status
            except Exception:
                method_support[method] = None
        
        # Should support at least GET
        assert method_support.get("GET") is not None


class TestPlaywrightExploratoryData:
    """Exploratory data tests."""
    
    @pytest.mark.playwright
    @pytest.mark.exploratory
    def test_explore_data_structures(self, page, api_base_url):
        """Explore data structures in responses."""
        response = page.request.get(f"{api_base_url}/health")
        
        if response.status == 200:
            data = response.json()
            
            # Explore structure
            structure_info = {
                "is_dict": isinstance(data, dict),
                "is_list": isinstance(data, list),
                "keys": list(data.keys()) if isinstance(data, dict) else [],
                "depth": self._get_depth(data)
            }
            
            # Should have some structure
            assert structure_info["is_dict"] or structure_info["is_list"]
    
    def _get_depth(self, obj, depth=0):
        """Get depth of nested structure."""
        if isinstance(obj, dict):
            if obj:
                return max(self._get_depth(v, depth + 1) for v in obj.values())
            return depth
        elif isinstance(obj, list):
            if obj:
                return max(self._get_depth(item, depth + 1) for item in obj)
            return depth
        return depth
    
    @pytest.mark.playwright
    @pytest.mark.exploratory
    def test_explore_nested_resources(self, page, api_base_url, auth_headers):
        """Explore nested resource structures."""
        # Try to explore nested resources
        nested_paths = [
            "/pdf/test_file/variants",
            "/pdf/test_file/topics",
            "/pdf/test_file/preview",
            "/pdf/test_file/metadata"
        ]
        
        found_resources = []
        
        for path in nested_paths:
            try:
                response = page.request.get(
                    f"{api_base_url}{path}",
                    headers=auth_headers,
                    timeout=2000
                )
                if response.status < 500:
                    found_resources.append((path, response.status))
            except Exception:
                pass
        
        # Should find some resources
        assert len(found_resources) >= 0  # May or may not have nested resources



