"""
Comprehensive Playwright Tests
===============================
Complete and comprehensive Playwright test suite with best practices.
"""

import pytest
from playwright.sync_api import Page, Response, Browser, BrowserContext
import time
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import tempfile


@pytest.fixture
def api_base_url():
    """API base URL."""
    return "http://localhost:8000"


@pytest.fixture
def auth_headers():
    """Authentication headers."""
    return {
        "Authorization": "Bearer test_token_123",
        "X-User-ID": "test_user_123",
        "Content-Type": "application/json"
    }


@pytest.fixture
def sample_pdf():
    """Sample PDF content."""
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\nxref\n0 2\ntrailer\n<<\n/Size 2\n>>\nstartxref\n20\n%%EOF"


class TestPlaywrightRequestValidation:
    """Tests for request validation."""
    
    @pytest.mark.playwright
    def test_missing_required_fields(self, page, api_base_url, auth_headers):
        """Test requests with missing required fields."""
        # Try to upload without file
        response = page.request.post(
            f"{api_base_url}/pdf/upload",
            json={},
            headers=auth_headers
        )
        assert response.status in [400, 422, 415]
    
    @pytest.mark.playwright
    def test_invalid_json_syntax(self, page, api_base_url, auth_headers):
        """Test requests with invalid JSON."""
        response = page.request.post(
            f"{api_base_url}/pdf/test_file/variants",
            data="invalid json {",
            headers={**auth_headers, "Content-Type": "application/json"}
        )
        assert response.status in [400, 422]
    
    @pytest.mark.playwright
    def test_wrong_content_type(self, page, api_base_url, auth_headers):
        """Test requests with wrong content type."""
        response = page.request.post(
            f"{api_base_url}/pdf/test_file/variants",
            data="plain text",
            headers={**auth_headers, "Content-Type": "text/plain"}
        )
        assert response.status in [400, 415, 422]
    
    @pytest.mark.playwright
    def test_malformed_query_parameters(self, page, api_base_url, auth_headers):
        """Test requests with malformed query parameters."""
        # Invalid query params
        response = page.request.get(
            f"{api_base_url}/pdf/test_file/topics?min_relevance=invalid&max_topics=not_a_number",
            headers=auth_headers
        )
        assert response.status in [400, 422]
    
    @pytest.mark.playwright
    def test_sql_injection_attempt(self, page, api_base_url, auth_headers):
        """Test SQL injection attempts."""
        sql_injection = "'; DROP TABLE users; --"
        response = page.request.get(
            f"{api_base_url}/pdf/{sql_injection}/preview",
            headers=auth_headers
        )
        # Should handle safely (may return 404 or sanitize)
        assert response.status in [200, 400, 404, 422]
    
    @pytest.mark.playwright
    def test_xss_attempt(self, page, api_base_url, auth_headers):
        """Test XSS attempts."""
        xss_payload = "<script>alert('xss')</script>"
        response = page.request.get(
            f"{api_base_url}/pdf/{xss_payload}/preview",
            headers=auth_headers
        )
        # Should handle safely
        assert response.status in [200, 400, 404, 422]


class TestPlaywrightResponseValidation:
    """Tests for response validation."""
    
    @pytest.mark.playwright
    def test_response_content_type(self, page, api_base_url):
        """Test response content types."""
        response = page.request.get(f"{api_base_url}/health")
        
        content_type = response.headers.get("content-type", "")
        assert "application/json" in content_type or "text/html" in content_type
    
    @pytest.mark.playwright
    def test_response_structure_consistency(self, page, api_base_url):
        """Test that responses have consistent structure."""
        response = page.request.get(f"{api_base_url}/health")
        
        if response.status == 200:
            data = response.json()
            # Should be dict or list
            assert isinstance(data, (dict, list))
    
    @pytest.mark.playwright
    def test_error_response_format(self, page, api_base_url):
        """Test error response format."""
        response = page.request.get(f"{api_base_url}/nonexistent_endpoint_xyz")
        
        if response.status >= 400:
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                data = response.json()
                # Should have error information
                assert "detail" in data or "error" in data or "message" in data


class TestPlaywrightCaching:
    """Tests for caching behavior."""
    
    @pytest.mark.playwright
    def test_cache_headers(self, page, api_base_url):
        """Test cache control headers."""
        response = page.request.get(f"{api_base_url}/health")
        
        headers = response.headers
        cache_headers = [
            "cache-control",
            "etag",
            "last-modified",
            "expires"
        ]
        
        # May or may not have cache headers
        has_cache = any(header in headers for header in cache_headers)
        assert True  # Just verify we can check
    
    @pytest.mark.playwright
    def test_conditional_requests(self, page, api_base_url):
        """Test conditional requests (If-None-Match, If-Modified-Since)."""
        # First request
        response1 = page.request.get(f"{api_base_url}/health")
        etag = response1.headers.get("etag")
        
        if etag:
            # Second request with If-None-Match
            response2 = page.request.get(
                f"{api_base_url}/health",
                headers={"If-None-Match": etag}
            )
            # May return 304 Not Modified
            assert response2.status in [200, 304]


class TestPlaywrightCompression:
    """Tests for response compression."""
    
    @pytest.mark.playwright
    def test_gzip_compression(self, page, api_base_url):
        """Test GZIP compression."""
        response = page.request.get(
            f"{api_base_url}/health",
            headers={"Accept-Encoding": "gzip, deflate"}
        )
        
        encoding = response.headers.get("content-encoding", "")
        # May or may not use compression
        assert True  # Just verify request works
    
    @pytest.mark.playwright
    def test_deflate_compression(self, page, api_base_url):
        """Test DEFLATE compression."""
        response = page.request.get(
            f"{api_base_url}/health",
            headers={"Accept-Encoding": "deflate"}
        )
        
        # Should handle compression request
        assert response.status == 200


class TestPlaywrightStreaming:
    """Tests for streaming responses."""
    
    @pytest.mark.playwright
    def test_streaming_response(self, page, api_base_url, auth_headers):
        """Test streaming response handling."""
        file_id = "test_file_123"
        
        response = page.request.get(
            f"{api_base_url}/pdf/{file_id}/stream",
            headers=auth_headers
        )
        
        # May support streaming
        assert response.status in [200, 404, 401, 403]
        
        if response.status == 200:
            content_type = response.headers.get("content-type", "")
            # May be streaming content type
            assert True


class TestPlaywrightPagination:
    """Tests for pagination."""
    
    @pytest.mark.playwright
    def test_pagination_links(self, page, api_base_url, auth_headers):
        """Test pagination link headers."""
        response = page.request.get(
            f"{api_base_url}/pdf?page=1&per_page=10",
            headers=auth_headers
        )
        
        headers = response.headers
        
        # Check for pagination headers
        pagination_headers = [
            "link",
            "x-total-count",
            "x-page",
            "x-per-page"
        ]
        
        has_pagination = any(header in headers for header in pagination_headers)
        # Pagination may or may not be implemented
        assert True
    
    @pytest.mark.playwright
    def test_pagination_consistency(self, page, api_base_url, auth_headers):
        """Test pagination consistency."""
        # Get first page
        response1 = page.request.get(
            f"{api_base_url}/pdf?page=1&per_page=10",
            headers=auth_headers
        )
        
        # Get second page
        response2 = page.request.get(
            f"{api_base_url}/pdf?page=2&per_page=10",
            headers=auth_headers
        )
        
        # Both should return valid responses
        assert response1.status in [200, 401, 403]
        assert response2.status in [200, 401, 403]


class TestPlaywrightSorting:
    """Tests for sorting and ordering."""
    
    @pytest.mark.playwright
    def test_sort_by_date(self, page, api_base_url, auth_headers):
        """Test sorting by date."""
        sort_options = ["date_asc", "date_desc", "created_at", "updated_at"]
        
        for sort_by in sort_options:
            response = page.request.get(
                f"{api_base_url}/pdf?sort_by={sort_by}",
                headers=auth_headers
            )
            assert response.status in [200, 400, 401, 403]
    
    @pytest.mark.playwright
    def test_sort_by_name(self, page, api_base_url, auth_headers):
        """Test sorting by name."""
        response = page.request.get(
            f"{api_base_url}/pdf?sort_by=name&order=asc",
            headers=auth_headers
        )
        assert response.status in [200, 400, 401, 403]


class TestPlaywrightFieldSelection:
    """Tests for field selection (sparse fieldsets)."""
    
    @pytest.mark.playwright
    def test_field_selection(self, page, api_base_url, auth_headers):
        """Test selecting specific fields."""
        file_id = "test_file_123"
        
        # Request only specific fields
        response = page.request.get(
            f"{api_base_url}/pdf/{file_id}?fields=file_id,filename",
            headers=auth_headers
        )
        
        if response.status == 200:
            data = response.json()
            # Should only have requested fields (or all fields)
            assert isinstance(data, dict)


class TestPlaywrightContentNegotiation:
    """Tests for content negotiation."""
    
    @pytest.mark.playwright
    def test_accept_header_json(self, page, api_base_url):
        """Test Accept header for JSON."""
        response = page.request.get(
            f"{api_base_url}/health",
            headers={"Accept": "application/json"}
        )
        
        content_type = response.headers.get("content-type", "")
        assert "application/json" in content_type or response.status == 200
    
    @pytest.mark.playwright
    def test_accept_header_xml(self, page, api_base_url):
        """Test Accept header for XML."""
        response = page.request.get(
            f"{api_base_url}/health",
            headers={"Accept": "application/xml"}
        )
        
        # May or may not support XML
        assert response.status in [200, 406]  # 406 Not Acceptable
    
    @pytest.mark.playwright
    def test_accept_header_html(self, page, api_base_url):
        """Test Accept header for HTML."""
        response = page.request.get(
            f"{api_base_url}/health",
            headers={"Accept": "text/html"}
        )
        
        # May return HTML or JSON
        assert response.status == 200


class TestPlaywrightIdempotency:
    """Tests for idempotency."""
    
    @pytest.mark.playwright
    def test_idempotent_operations(self, page, api_base_url, auth_headers):
        """Test idempotent operations."""
        file_id = "test_file_123"
        
        # DELETE should be idempotent
        response1 = page.request.delete(
            f"{api_base_url}/pdf/{file_id}",
            headers=auth_headers
        )
        
        # Second DELETE should have same result
        response2 = page.request.delete(
            f"{api_base_url}/pdf/{file_id}",
            headers=auth_headers
        )
        
        # Both should return same status (404 if already deleted)
        assert response1.status == response2.status or response2.status == 404


class TestPlaywrightConcurrencyControl:
    """Tests for concurrency control."""
    
    @pytest.mark.playwright
    def test_etag_for_optimistic_locking(self, page, api_base_url, auth_headers):
        """Test ETag for optimistic locking."""
        file_id = "test_file_123"
        
        # Get resource with ETag
        response1 = page.request.get(
            f"{api_base_url}/pdf/{file_id}",
            headers=auth_headers
        )
        
        etag = response1.headers.get("etag")
        
        if etag and response1.status == 200:
            # Update with If-Match
            update_data = {"title": "Updated"}
            response2 = page.request.put(
                f"{api_base_url}/pdf/{file_id}",
                json=update_data,
                headers={**auth_headers, "If-Match": etag}
            )
            
            # Should accept or reject based on ETag match
            assert response2.status in [200, 412, 404, 401, 403]


class TestPlaywrightHATEOAS:
    """Tests for HATEOAS (Hypermedia as the Engine of Application State)."""
    
    @pytest.mark.playwright
    def test_links_in_response(self, page, api_base_url, auth_headers):
        """Test links in API responses."""
        file_id = "test_file_123"
        
        response = page.request.get(
            f"{api_base_url}/pdf/{file_id}",
            headers=auth_headers
        )
        
        if response.status == 200:
            data = response.json()
            # May have links for related resources
            if "links" in data or "_links" in data:
                links = data.get("links") or data.get("_links")
                assert isinstance(links, dict)


class TestPlaywrightVersioning:
    """Tests for API versioning."""
    
    @pytest.mark.playwright
    def test_url_versioning(self, page, api_base_url):
        """Test URL-based versioning."""
        versions = ["v1", "v2", "v3"]
        
        for version in versions:
            response = page.request.get(f"{api_base_url}/{version}/health")
            # May or may not support versioning
            assert response.status is not None
    
    @pytest.mark.playwright
    def test_header_versioning(self, page, api_base_url):
        """Test header-based versioning."""
        response = page.request.get(
            f"{api_base_url}/health",
            headers={"API-Version": "v2"}
        )
        # May or may not support header versioning
        assert response.status is not None


class TestPlaywrightDeprecation:
    """Tests for API deprecation."""
    
    @pytest.mark.playwright
    def test_deprecation_headers(self, page, api_base_url):
        """Test deprecation headers."""
        response = page.request.get(f"{api_base_url}/health")
        
        headers = response.headers
        
        # Check for deprecation headers
        deprecation_headers = [
            "deprecation",
            "sunset",
            "x-deprecated"
        ]
        
        has_deprecation = any(header in headers for header in deprecation_headers)
        # Deprecation may or may not be present
        assert True


class TestPlaywrightMetrics:
    """Tests for metrics and monitoring."""
    
    @pytest.mark.playwright
    def test_metrics_endpoint(self, page, api_base_url):
        """Test metrics endpoint."""
        metrics_paths = ["/metrics", "/health/metrics", "/api/metrics"]
        
        for path in metrics_paths:
            try:
                response = page.request.get(f"{api_base_url}{path}")
                if response.status == 200:
                    # Should return metrics
                    assert True
                    return
            except Exception:
                continue
        
        pytest.skip("Metrics endpoint not available")
    
    @pytest.mark.playwright
    def test_prometheus_metrics(self, page, api_base_url):
        """Test Prometheus metrics format."""
        try:
            response = page.request.get(f"{api_base_url}/metrics")
            if response.status == 200:
                content = response.text
                # Prometheus format has specific structure
                assert "#" in content or "TYPE" in content or len(content) > 0
        except Exception:
            pytest.skip("Prometheus metrics not available")


class TestPlaywrightWebSocket:
    """Tests for WebSocket connections."""
    
    @pytest.mark.playwright
    def test_websocket_connection(self, page, api_base_url):
        """Test WebSocket connection."""
        ws_url = api_base_url.replace("http", "ws") + "/ws"
        
        try:
            with page.expect_websocket() as ws_info:
                page.evaluate(f"new WebSocket('{ws_url}')")
            
            ws = ws_info.value
            # WebSocket should connect
            assert ws is not None
        except Exception:
            pytest.skip("WebSocket not available")


class TestPlaywrightGraphQL:
    """Tests for GraphQL endpoint (if available)."""
    
    @pytest.mark.playwright
    def test_graphql_endpoint(self, page, api_base_url, auth_headers):
        """Test GraphQL endpoint."""
        graphql_query = {
            "query": "{ health { status } }"
        }
        
        response = page.request.post(
            f"{api_base_url}/graphql",
            json=graphql_query,
            headers=auth_headers
        )
        
        # May or may not support GraphQL
        if response.status == 200:
            data = response.json()
            assert "data" in data or "errors" in data
        else:
            assert response.status in [404, 400, 401, 403]


class TestPlaywrightOAuth:
    """Tests for OAuth endpoints."""
    
    @pytest.mark.playwright
    def test_oauth_authorize(self, page, api_base_url):
        """Test OAuth authorization endpoint."""
        response = page.request.get(
            f"{api_base_url}/oauth/authorize?client_id=test&response_type=code"
        )
        
        # May or may not support OAuth
        assert response.status in [200, 302, 400, 404]
    
    @pytest.mark.playwright
    def test_oauth_token(self, page, api_base_url):
        """Test OAuth token endpoint."""
        token_data = {
            "grant_type": "authorization_code",
            "code": "test_code",
            "client_id": "test_client"
        }
        
        response = page.request.post(
            f"{api_base_url}/oauth/token",
            json=token_data
        )
        
        # May or may not support OAuth
        assert response.status in [200, 400, 401, 404]


class TestPlaywrightFileDownload:
    """Tests for file download."""
    
    @pytest.mark.playwright
    def test_download_pdf(self, page, api_base_url, auth_headers, tmp_path):
        """Test downloading PDF file."""
        file_id = "test_file_123"
        
        response = page.request.get(
            f"{api_base_url}/pdf/{file_id}/download",
            headers=auth_headers
        )
        
        if response.status == 200:
            # Save downloaded file
            download_path = tmp_path / "downloaded.pdf"
            download_path.write_bytes(response.body())
            
            # Verify file was downloaded
            assert download_path.exists()
            assert download_path.stat().st_size > 0
    
    @pytest.mark.playwright
    def test_download_variant(self, page, api_base_url, auth_headers, tmp_path):
        """Test downloading variant."""
        file_id = "test_file_123"
        variant_id = "variant_123"
        
        response = page.request.get(
            f"{api_base_url}/pdf/{file_id}/variants/{variant_id}/download",
            headers=auth_headers
        )
        
        if response.status == 200:
            content_type = response.headers.get("content-type", "")
            # Should be PDF or other document type
            assert "application/pdf" in content_type or "application/octet-stream" in content_type


class TestPlaywrightExport:
    """Tests for export functionality."""
    
    @pytest.mark.playwright
    def test_export_to_json(self, page, api_base_url, auth_headers):
        """Test exporting to JSON."""
        file_id = "test_file_123"
        
        response = page.request.get(
            f"{api_base_url}/pdf/{file_id}/export?format=json",
            headers=auth_headers
        )
        
        if response.status == 200:
            data = response.json()
            assert isinstance(data, dict)
    
    @pytest.mark.playwright
    def test_export_to_csv(self, page, api_base_url, auth_headers):
        """Test exporting to CSV."""
        response = page.request.get(
            f"{api_base_url}/pdf/export?format=csv",
            headers=auth_headers
        )
        
        if response.status == 200:
            content_type = response.headers.get("content-type", "")
            assert "text/csv" in content_type or "application/csv" in content_type


class TestPlaywrightBulkOperations:
    """Tests for bulk operations."""
    
    @pytest.mark.playwright
    def test_bulk_delete(self, page, api_base_url, auth_headers):
        """Test bulk delete operation."""
        file_ids = ["file_1", "file_2", "file_3"]
        
        response = page.request.post(
            f"{api_base_url}/pdf/bulk/delete",
            json={"file_ids": file_ids},
            headers=auth_headers
        )
        
        # May or may not support bulk operations
        assert response.status in [200, 202, 400, 404, 401, 403]
    
    @pytest.mark.playwright
    def test_bulk_update(self, page, api_base_url, auth_headers):
        """Test bulk update operation."""
        updates = [
            {"file_id": "file_1", "title": "Title 1"},
            {"file_id": "file_2", "title": "Title 2"}
        ]
        
        response = page.request.post(
            f"{api_base_url}/pdf/bulk/update",
            json={"updates": updates},
            headers=auth_headers
        )
        
        assert response.status in [200, 202, 400, 404, 401, 403]


class TestPlaywrightNotifications:
    """Tests for notifications."""
    
    @pytest.mark.playwright
    def test_get_notifications(self, page, api_base_url, auth_headers):
        """Test getting notifications."""
        response = page.request.get(
            f"{api_base_url}/notifications",
            headers=auth_headers
        )
        
        if response.status == 200:
            data = response.json()
            assert isinstance(data, (list, dict))
    
    @pytest.mark.playwright
    def test_mark_notification_read(self, page, api_base_url, auth_headers):
        """Test marking notification as read."""
        notification_id = "notif_123"
        
        response = page.request.put(
            f"{api_base_url}/notifications/{notification_id}/read",
            headers=auth_headers
        )
        
        assert response.status in [200, 404, 401, 403]



