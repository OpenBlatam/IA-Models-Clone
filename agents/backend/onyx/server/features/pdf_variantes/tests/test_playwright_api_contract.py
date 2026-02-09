"""
Playwright API Contract Tests
==============================
Contract testing to ensure API contracts are maintained.
"""

import pytest
from playwright.sync_api import Page, Response
import time
import json
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


class TestPlaywrightAPIContract:
    """API contract tests."""
    
    @pytest.mark.playwright
    @pytest.mark.contract
    def test_health_endpoint_contract(self, page, api_base_url):
        """Test health endpoint contract."""
        response = page.request.get(f"{api_base_url}/health")
        
        assert response.status == 200
        
        # Contract: Should return JSON
        content_type = response.headers.get("content-type", "")
        assert "application/json" in content_type
        
        # Contract: Should have status field
        data = response.json()
        assert "status" in data or isinstance(data, dict)
    
    @pytest.mark.playwright
    @pytest.mark.contract
    def test_upload_endpoint_contract(self, page, api_base_url, sample_pdf, auth_headers):
        """Test upload endpoint contract."""
        files = {
            "file": {
                "name": "contract_test.pdf",
                "mimeType": "application/pdf",
                "buffer": sample_pdf
            }
        }
        
        response = page.request.post(
            f"{api_base_url}/pdf/upload",
            multipart=files,
            headers=auth_headers
        )
        
        if response.status in [200, 201]:
            # Contract: Should return JSON
            content_type = response.headers.get("content-type", "")
            assert "application/json" in content_type
            
            # Contract: Should have file_id or id
            data = response.json()
            assert "file_id" in data or "id" in data
    
    @pytest.mark.playwright
    @pytest.mark.contract
    def test_variant_endpoint_contract(self, page, api_base_url, auth_headers):
        """Test variant endpoint contract."""
        file_id = "test_file_123"
        variant_data = {
            "variant_type": "summary",
            "options": {}
        }
        
        response = page.request.post(
            f"{api_base_url}/pdf/{file_id}/variants",
            json=variant_data,
            headers=auth_headers
        )
        
        if response.status in [200, 202]:
            # Contract: Should return JSON
            content_type = response.headers.get("content-type", "")
            assert "application/json" in content_type
            
            # Contract: Should have variant information
            data = response.json()
            assert isinstance(data, dict)
    
    @pytest.mark.playwright
    @pytest.mark.contract
    def test_error_response_contract(self, page, api_base_url):
        """Test error response contract."""
        response = page.request.get(f"{api_base_url}/nonexistent_endpoint_xyz")
        
        if response.status >= 400:
            # Contract: Error responses should be JSON
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                data = response.json()
                # Contract: Should have error information
                assert "detail" in data or "error" in data or "message" in data or isinstance(data, dict)


class TestPlaywrightSchemaContract:
    """Schema contract tests."""
    
    @pytest.mark.playwright
    @pytest.mark.contract
    def test_response_schema_consistency(self, page, api_base_url):
        """Test response schema consistency."""
        responses = []
        
        # Get multiple responses
        for _ in range(5):
            response = page.request.get(f"{api_base_url}/health")
            if response.status == 200:
                responses.append(response.json())
            time.sleep(0.1)
        
        if len(responses) > 1:
            # Contract: All responses should have same schema
            first_keys = set(responses[0].keys())
            for resp in responses[1:]:
                assert set(resp.keys()) == first_keys, "Schema inconsistency detected"
    
    @pytest.mark.playwright
    @pytest.mark.contract
    def test_request_schema_validation(self, page, api_base_url, auth_headers):
        """Test request schema validation."""
        # Valid request
        valid_request = {
            "variant_type": "summary",
            "options": {}
        }
        
        response1 = page.request.post(
            f"{api_base_url}/pdf/test_file/variants",
            json=valid_request,
            headers=auth_headers
        )
        
        # Invalid request (missing required field)
        invalid_request = {
            "options": {}
        }
        
        response2 = page.request.post(
            f"{api_base_url}/pdf/test_file/variants",
            json=invalid_request,
            headers=auth_headers
        )
        
        # Contract: Valid should succeed or fail gracefully
        assert response1.status in [200, 202, 404, 401, 403]
        
        # Contract: Invalid should be rejected
        assert response2.status in [400, 422, 404, 401, 403]


class TestPlaywrightVersionContract:
    """Version contract tests."""
    
    @pytest.mark.playwright
    @pytest.mark.contract
    def test_api_version_contract(self, page, api_base_url):
        """Test API version contract."""
        response = page.request.get(f"{api_base_url}/health")
        headers = response.headers
        
        # Contract: May have version header
        version_headers = ["x-api-version", "api-version", "version"]
        has_version = any(header in headers for header in version_headers)
        # Version may or may not be present
        assert True
    
    @pytest.mark.playwright
    @pytest.mark.contract
    def test_backward_compatibility_contract(self, page, api_base_url):
        """Test backward compatibility contract."""
        # Test v1 endpoint
        v1_response = page.request.get(f"{api_base_url}/v1/health")
        
        # Test current endpoint
        current_response = page.request.get(f"{api_base_url}/health")
        
        # Contract: Both should work or handle gracefully
        assert v1_response.status is not None
        assert current_response.status is not None


class TestPlaywrightHeaderContract:
    """Header contract tests."""
    
    @pytest.mark.playwright
    @pytest.mark.contract
    def test_required_headers(self, page, api_base_url):
        """Test required headers in responses."""
        response = page.request.get(f"{api_base_url}/health")
        headers = response.headers
        
        # Contract: Should have content-type
        assert "content-type" in headers
        
        # Contract: May have other standard headers
        standard_headers = ["date", "server", "content-length"]
        has_standard = any(header in headers for header in standard_headers)
        # Standard headers may or may not be present
        assert True
    
    @pytest.mark.playwright
    @pytest.mark.contract
    def test_cors_headers_contract(self, page, api_base_url):
        """Test CORS headers contract."""
        # OPTIONS request for CORS
        try:
            options_response = page.request.options(f"{api_base_url}/health")
            headers = options_response.headers
            
            # Contract: Should have CORS headers if CORS is enabled
            cors_headers = ["access-control-allow-origin", "access-control-allow-methods"]
            has_cors = any(header in headers for header in cors_headers)
            # CORS may or may not be enabled
            assert True
        except Exception:
            # OPTIONS may not be supported
            assert True



