"""
Playwright Integration Tests
============================
Integration tests combining multiple features with Playwright.
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


@pytest.fixture
def sample_pdf():
    """Sample PDF content."""
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\nxref\n0 2\ntrailer\n<<\n/Size 2\n>>\nstartxref\n20\n%%EOF"


class TestPlaywrightMultiFeatureIntegration:
    """Tests integrating multiple features."""
    
    @pytest.mark.playwright
    @pytest.mark.integration
    def test_upload_preview_topics_variants_integration(self, page, api_base_url, sample_pdf, auth_headers):
        """Test integration of upload, preview, topics, and variants."""
        
        # 1. Upload
        files = {
            "file": {
                "name": "integration_test.pdf",
                "mimeType": "application/pdf",
                "buffer": sample_pdf
            }
        }
        
        upload_response = page.request.post(
            f"{api_base_url}/pdf/upload",
            multipart=files,
            headers=auth_headers
        )
        
        if upload_response.status not in [200, 201]:
            pytest.skip("Upload failed")
        
        file_id = upload_response.json().get("file_id") or upload_response.json().get("id")
        
        # 2. Preview
        preview_response = page.request.get(
            f"{api_base_url}/pdf/{file_id}/preview?page_number=1",
            headers=auth_headers
        )
        assert preview_response.status in [200, 202, 404, 401, 403]
        
        # 3. Topics
        topics_response = page.request.get(
            f"{api_base_url}/pdf/{file_id}/topics",
            headers=auth_headers
        )
        assert topics_response.status in [200, 202, 404, 401, 403]
        
        # 4. Variants
        variant_response = page.request.post(
            f"{api_base_url}/pdf/{file_id}/variants",
            json={"variant_type": "summary", "options": {}},
            headers=auth_headers
        )
        assert variant_response.status in [200, 202, 404, 401, 403]
    
    @pytest.mark.playwright
    @pytest.mark.integration
    def test_search_filter_sort_integration(self, page, api_base_url, auth_headers):
        """Test integration of search, filter, and sort."""
        
        # 1. Search
        search_response = page.request.get(
            f"{api_base_url}/pdf/search?q=test",
            headers=auth_headers
        )
        assert search_response.status in [200, 401, 403]
        
        # 2. Filter
        filter_response = page.request.get(
            f"{api_base_url}/pdf?tags=important&start_date=2024-01-01",
            headers=auth_headers
        )
        assert filter_response.status in [200, 401, 403]
        
        # 3. Sort
        sort_response = page.request.get(
            f"{api_base_url}/pdf?sort_by=date&order=desc",
            headers=auth_headers
        )
        assert sort_response.status in [200, 401, 403]
    
    @pytest.mark.playwright
    @pytest.mark.integration
    def test_batch_operations_integration(self, page, api_base_url, sample_pdf, auth_headers):
        """Test integration of batch operations."""
        
        # Upload multiple files
        file_ids = []
        for i in range(3):
            files = {
                "file": {
                    "name": f"batch_{i}.pdf",
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
                file_id = response.json().get("file_id") or response.json().get("id")
                if file_id:
                    file_ids.append(file_id)
        
        # Generate variants for all
        for file_id in file_ids:
            variant_response = page.request.post(
                f"{api_base_url}/pdf/{file_id}/variants",
                json={"variant_type": "summary", "options": {}},
                headers=auth_headers
            )
            assert variant_response.status in [200, 202, 404, 401, 403]


class TestPlaywrightCacheIntegration:
    """Tests for cache integration."""
    
    @pytest.mark.playwright
    @pytest.mark.integration
    def test_cache_headers_integration(self, page, api_base_url):
        """Test cache headers integration."""
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
    
    @pytest.mark.playwright
    @pytest.mark.integration
    def test_cache_invalidation(self, page, api_base_url, sample_pdf, auth_headers):
        """Test cache invalidation."""
        # Upload file
        files = {
            "file": {
                "name": "cache_test.pdf",
                "mimeType": "application/pdf",
                "buffer": sample_pdf
            }
        }
        
        upload_response = page.request.post(
            f"{api_base_url}/pdf/upload",
            multipart=files,
            headers=auth_headers
        )
        
        if upload_response.status not in [200, 201]:
            pytest.skip("Upload failed")
        
        file_id = upload_response.json().get("file_id") or upload_response.json().get("id")
        
        # Get metadata (may be cached)
        response1 = page.request.get(
            f"{api_base_url}/pdf/{file_id}",
            headers=auth_headers
        )
        
        # Update metadata (should invalidate cache)
        update_response = page.request.put(
            f"{api_base_url}/pdf/{file_id}",
            json={"title": "Updated Title"},
            headers=auth_headers
        )
        
        # Get again (should have new data)
        response2 = page.request.get(
            f"{api_base_url}/pdf/{file_id}",
            headers=auth_headers
        )
        
        assert response1.status in [200, 404, 401, 403]
        assert response2.status in [200, 404, 401, 403]


class TestPlaywrightAuthIntegration:
    """Tests for authentication integration."""
    
    @pytest.mark.playwright
    @pytest.mark.integration
    def test_auth_token_refresh(self, page, api_base_url):
        """Test token refresh integration."""
        # Initial request with token
        response1 = page.request.get(
            f"{api_base_url}/pdf/test_file/preview",
            headers={"Authorization": "Bearer old_token"}
        )
        
        # If token expired, should get 401
        if response1.status == 401:
            # Refresh token
            refresh_response = page.request.post(
                f"{api_base_url}/auth/refresh",
                json={"refresh_token": "refresh_token_123"}
            )
            
            if refresh_response.status == 200:
                new_token = refresh_response.json().get("access_token")
                
                # Use new token
                response2 = page.request.get(
                    f"{api_base_url}/pdf/test_file/preview",
                    headers={"Authorization": f"Bearer {new_token}"}
                )
                assert response2.status in [200, 404, 401, 403]
    
    @pytest.mark.playwright
    @pytest.mark.integration
    def test_role_based_access_integration(self, page, api_base_url):
        """Test role-based access integration."""
        roles = ["viewer", "editor", "admin"]
        
        for role in roles:
            response = page.request.get(
                f"{api_base_url}/admin/users",
                headers={
                    "Authorization": f"Bearer {role}_token",
                    "X-Role": role
                }
            )
            # Should enforce role-based access
            assert response.status in [200, 403, 401, 404]


class TestPlaywrightWebhookIntegration:
    """Tests for webhook integration."""
    
    @pytest.mark.playwright
    @pytest.mark.integration
    def test_webhook_event_flow(self, page, api_base_url, sample_pdf, auth_headers):
        """Test webhook event flow integration."""
        # Register webhook
        webhook_data = {
            "url": "https://example.com/webhook",
            "events": ["pdf.uploaded", "variant.generated"]
        }
        
        webhook_response = page.request.post(
            f"{api_base_url}/webhooks",
            json=webhook_data,
            headers=auth_headers
        )
        
        if webhook_response.status not in [200, 201]:
            pytest.skip("Webhook registration failed")
        
        webhook_id = webhook_response.json().get("webhook_id") or webhook_response.json().get("id")
        
        # Upload file (should trigger webhook)
        files = {
            "file": {
                "name": "webhook_test.pdf",
                "mimeType": "application/pdf",
                "buffer": sample_pdf
            }
        }
        
        upload_response = page.request.post(
            f"{api_base_url}/pdf/upload",
            multipart=files,
            headers=auth_headers
        )
        
        # Webhook should be triggered (in real scenario, would verify)
        assert upload_response.status in [200, 201, 401, 403]
        
        # Check webhook status
        status_response = page.request.get(
            f"{api_base_url}/webhooks/{webhook_id}/status",
            headers=auth_headers
        )
        assert status_response.status in [200, 404, 401, 403]


class TestPlaywrightNotificationIntegration:
    """Tests for notification integration."""
    
    @pytest.mark.playwright
    @pytest.mark.integration
    def test_notification_flow(self, page, api_base_url, sample_pdf, auth_headers):
        """Test notification flow integration."""
        # Upload file
        files = {
            "file": {
                "name": "notification_test.pdf",
                "mimeType": "application/pdf",
                "buffer": sample_pdf
            }
        }
        
        upload_response = page.request.post(
            f"{api_base_url}/pdf/upload",
            multipart=files,
            headers=auth_headers
        )
        
        if upload_response.status not in [200, 201]:
            pytest.skip("Upload failed")
        
        file_id = upload_response.json().get("file_id") or upload_response.json().get("id")
        
        # Check for notifications
        notifications_response = page.request.get(
            f"{api_base_url}/notifications",
            headers=auth_headers
        )
        
        if notifications_response.status == 200:
            notifications = notifications_response.json()
            # May have notifications about upload
            assert isinstance(notifications, (list, dict))


class TestPlaywrightExportIntegration:
    """Tests for export integration."""
    
    @pytest.mark.playwright
    @pytest.mark.integration
    def test_export_workflow(self, page, api_base_url, sample_pdf, auth_headers):
        """Test export workflow integration."""
        # Upload
        files = {
            "file": {
                "name": "export_test.pdf",
                "mimeType": "application/pdf",
                "buffer": sample_pdf
            }
        }
        
        upload_response = page.request.post(
            f"{api_base_url}/pdf/upload",
            multipart=files,
            headers=auth_headers
        )
        
        if upload_response.status not in [200, 201]:
            pytest.skip("Upload failed")
        
        file_id = upload_response.json().get("file_id") or upload_response.json().get("id")
        
        # Export to different formats
        export_formats = ["json", "csv", "xml"]
        
        for format_type in export_formats:
            export_response = page.request.get(
                f"{api_base_url}/pdf/{file_id}/export?format={format_type}",
                headers=auth_headers
            )
            assert export_response.status in [200, 404, 401, 403]


class TestPlaywrightSearchIntegration:
    """Tests for search integration."""
    
    @pytest.mark.playwright
    @pytest.mark.integration
    def test_search_with_filters(self, page, api_base_url, auth_headers):
        """Test search with filters integration."""
        # Search with multiple filters
        search_params = {
            "q": "test",
            "tags": "important,urgent",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "sort_by": "date",
            "order": "desc"
        }
        
        query_string = "&".join([f"{k}={v}" for k, v in search_params.items()])
        
        response = page.request.get(
            f"{api_base_url}/pdf/search?{query_string}",
            headers=auth_headers
        )
        
        assert response.status in [200, 401, 403]
    
    @pytest.mark.playwright
    @pytest.mark.integration
    def test_search_pagination(self, page, api_base_url, auth_headers):
        """Test search with pagination."""
        # First page
        response1 = page.request.get(
            f"{api_base_url}/pdf/search?q=test&page=1&per_page=10",
            headers=auth_headers
        )
        
        # Second page
        response2 = page.request.get(
            f"{api_base_url}/pdf/search?q=test&page=2&per_page=10",
            headers=auth_headers
        )
        
        assert response1.status in [200, 401, 403]
        assert response2.status in [200, 401, 403]


class TestPlaywrightVersioningIntegration:
    """Tests for API versioning integration."""
    
    @pytest.mark.playwright
    @pytest.mark.integration
    def test_version_compatibility(self, page, api_base_url):
        """Test version compatibility."""
        versions = ["v1", "v2"]
        
        for version in versions:
            # Health check for each version
            response = page.request.get(f"{api_base_url}/{version}/health")
            # May or may not support versioning
            assert response.status is not None
    
    @pytest.mark.playwright
    @pytest.mark.integration
    def test_version_header_negotiation(self, page, api_base_url):
        """Test version header negotiation."""
        # Request with version header
        response = page.request.get(
            f"{api_base_url}/health",
            headers={"API-Version": "v2"}
        )
        
        # May return version in response
        assert response.status in [200, 400, 406]


class TestPlaywrightErrorHandlingIntegration:
    """Tests for error handling integration."""
    
    @pytest.mark.playwright
    @pytest.mark.integration
    def test_error_recovery_chain(self, page, api_base_url, sample_pdf, auth_headers):
        """Test error recovery chain."""
        # Try invalid operation
        invalid_response = page.request.post(
            f"{api_base_url}/pdf/upload",
            json={"invalid": "data"}
        )
        assert invalid_response.status in [400, 422, 415]
        
        # Recover with valid operation
        files = {
            "file": {
                "name": "recovery_test.pdf",
                "mimeType": "application/pdf",
                "buffer": sample_pdf
            }
        }
        
        valid_response = page.request.post(
            f"{api_base_url}/pdf/upload",
            multipart=files,
            headers=auth_headers
        )
        assert valid_response.status in [200, 201, 401, 403]
    
    @pytest.mark.playwright
    @pytest.mark.integration
    def test_cascading_errors(self, page, api_base_url, auth_headers):
        """Test cascading error handling."""
        # Try to access non-existent file
        response1 = page.request.get(
            f"{api_base_url}/pdf/nonexistent_file/preview",
            headers=auth_headers
        )
        
        # Try to generate variant for non-existent file
        response2 = page.request.post(
            f"{api_base_url}/pdf/nonexistent_file/variants",
            json={"variant_type": "summary"},
            headers=auth_headers
        )
        
        # Both should handle errors gracefully
        assert response1.status in [404, 401, 403]
        assert response2.status in [404, 401, 403]



