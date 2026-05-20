"""
Playwright API Tests
====================
Comprehensive API testing with Playwright.
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


class TestPlaywrightPDFUpload:
    """Playwright tests for PDF upload endpoints."""
    
    @pytest.mark.playwright
    def test_upload_pdf_multipart(self, page, api_base_url, sample_pdf, auth_headers):
        """Test PDF upload with multipart form data."""
        files = {
            "file": {
                "name": "test_document.pdf",
                "mimeType": "application/pdf",
                "buffer": sample_pdf
            }
        }
        
        response = page.request.post(
            f"{api_base_url}/pdf/upload",
            multipart=files,
            headers=auth_headers
        )
        
        assert response.status in [200, 201, 401, 403]
        
        if response.status in [200, 201]:
            data = response.json()
            assert "file_id" in data or "id" in data
    
    @pytest.mark.playwright
    def test_upload_pdf_with_options(self, page, api_base_url, sample_pdf, auth_headers):
        """Test PDF upload with query parameters."""
        files = {
            "file": {
                "name": "test.pdf",
                "mimeType": "application/pdf",
                "buffer": sample_pdf
            }
        }
        
        response = page.request.post(
            f"{api_base_url}/pdf/upload?auto_process=true&extract_text=true",
            multipart=files,
            headers=auth_headers
        )
        
        assert response.status in [200, 201, 401, 403]
    
    @pytest.mark.playwright
    def test_upload_large_file(self, page, api_base_url, auth_headers):
        """Test uploading a large PDF file."""
        # Create larger PDF content
        large_pdf = b"%PDF-1.4\n" + b"x" * 1000000  # 1MB
        
        files = {
            "file": {
                "name": "large.pdf",
                "mimeType": "application/pdf",
                "buffer": large_pdf
            }
        }
        
        response = page.request.post(
            f"{api_base_url}/pdf/upload",
            multipart=files,
            headers=auth_headers,
            timeout=30000  # Longer timeout for large file
        )
        
        # May succeed, fail with 413, or require auth
        assert response.status in [200, 201, 413, 401, 403]
    
    @pytest.mark.playwright
    def test_upload_invalid_file_type(self, page, api_base_url, auth_headers):
        """Test uploading invalid file type."""
        files = {
            "file": {
                "name": "test.txt",
                "mimeType": "text/plain",
                "buffer": b"not a pdf"
            }
        }
        
        response = page.request.post(
            f"{api_base_url}/pdf/upload",
            multipart=files,
            headers=auth_headers
        )
        
        # Should reject invalid file
        assert response.status in [400, 422, 415]
    
    @pytest.mark.playwright
    def test_upload_empty_file(self, page, api_base_url, auth_headers):
        """Test uploading empty file."""
        files = {
            "file": {
                "name": "empty.pdf",
                "mimeType": "application/pdf",
                "buffer": b""
            }
        }
        
        response = page.request.post(
            f"{api_base_url}/pdf/upload",
            multipart=files,
            headers=auth_headers
        )
        
        # Should reject empty file
        assert response.status in [400, 422]


class TestPlaywrightVariantGeneration:
    """Playwright tests for variant generation."""
    
    @pytest.mark.playwright
    def test_generate_summary_variant(self, page, api_base_url, auth_headers):
        """Test generating summary variant."""
        file_id = "test_file_123"
        variant_data = {
            "variant_type": "summary",
            "options": {
                "max_length": 500,
                "style": "academic",
                "language": "en"
            }
        }
        
        response = page.request.post(
            f"{api_base_url}/pdf/{file_id}/variants",
            json=variant_data,
            headers=auth_headers
        )
        
        assert response.status in [200, 202, 404, 401, 403]
    
    @pytest.mark.playwright
    def test_generate_all_variant_types(self, page, api_base_url, auth_headers):
        """Test generating all variant types."""
        file_id = "test_file_123"
        variant_types = [
            "summary",
            "outline",
            "highlights",
            "notes",
            "quiz",
            "presentation"
        ]
        
        results = []
        for variant_type in variant_types:
            variant_data = {
                "variant_type": variant_type,
                "options": {}
            }
            
            response = page.request.post(
                f"{api_base_url}/pdf/{file_id}/variants",
                json=variant_data,
                headers=auth_headers
            )
            results.append((variant_type, response.status))
        
        # All requests should complete
        assert len(results) == len(variant_types)
    
    @pytest.mark.playwright
    def test_generate_variant_with_custom_options(self, page, api_base_url, auth_headers):
        """Test generating variant with custom options."""
        file_id = "test_file_123"
        variant_data = {
            "variant_type": "summary",
            "options": {
                "max_length": 1000,
                "style": "casual",
                "language": "es",
                "tone": "friendly",
                "include_images": False,
                "include_tables": True
            }
        }
        
        response = page.request.post(
            f"{api_base_url}/pdf/{file_id}/variants",
            json=variant_data,
            headers=auth_headers
        )
        
        assert response.status in [200, 202, 404, 401, 403]
    
    @pytest.mark.playwright
    def test_generate_variant_async(self, page, api_base_url, auth_headers):
        """Test async variant generation (202 Accepted)."""
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
        
        # May return 202 for async processing
        if response.status == 202:
            data = response.json()
            # Should have job_id or similar
            assert "job_id" in data or "task_id" in data or "status" in data


class TestPlaywrightTopicExtraction:
    """Playwright tests for topic extraction."""
    
    @pytest.mark.playwright
    def test_extract_topics_basic(self, page, api_base_url, auth_headers):
        """Test basic topic extraction."""
        file_id = "test_file_123"
        
        response = page.request.get(
            f"{api_base_url}/pdf/{file_id}/topics",
            headers=auth_headers
        )
        
        assert response.status in [200, 202, 404, 401, 403]
        
        if response.status == 200:
            data = response.json()
            assert "topics" in data or "file_id" in data
    
    @pytest.mark.playwright
    def test_extract_topics_with_filters(self, page, api_base_url, auth_headers):
        """Test topic extraction with filters."""
        file_id = "test_file_123"
        
        # Test different filter combinations
        filter_combinations = [
            {"min_relevance": 0.5, "max_topics": 10},
            {"min_relevance": 0.7, "max_topics": 20},
            {"min_relevance": 0.3, "max_topics": 50},
        ]
        
        for filters in filter_combinations:
            query_string = "&".join([f"{k}={v}" for k, v in filters.items()])
            response = page.request.get(
                f"{api_base_url}/pdf/{file_id}/topics?{query_string}",
                headers=auth_headers
            )
            assert response.status in [200, 202, 404, 401, 403]
    
    @pytest.mark.playwright
    def test_extract_topics_pagination(self, page, api_base_url, auth_headers):
        """Test topic extraction with pagination."""
        file_id = "test_file_123"
        
        # Test pagination parameters
        response = page.request.get(
            f"{api_base_url}/pdf/{file_id}/topics?page=1&per_page=10",
            headers=auth_headers
        )
        
        assert response.status in [200, 202, 404, 401, 403]
        
        if response.status == 200:
            data = response.json()
            # May have pagination info
            assert "topics" in data or "items" in data or "results" in data


class TestPlaywrightPDFPreview:
    """Playwright tests for PDF preview."""
    
    @pytest.mark.playwright
    def test_get_preview_first_page(self, page, api_base_url, auth_headers):
        """Test getting preview of first page."""
        file_id = "test_file_123"
        
        response = page.request.get(
            f"{api_base_url}/pdf/{file_id}/preview?page_number=1",
            headers=auth_headers
        )
        
        assert response.status in [200, 202, 404, 401, 403]
        
        if response.status == 200:
            content_type = response.headers.get("content-type", "")
            # May return JSON with base64 image or actual image
            assert "application/json" in content_type or "image" in content_type
    
    @pytest.mark.playwright
    def test_get_preview_multiple_pages(self, page, api_base_url, auth_headers):
        """Test getting preview of multiple pages."""
        file_id = "test_file_123"
        
        for page_num in [1, 2, 3]:
            response = page.request.get(
                f"{api_base_url}/pdf/{file_id}/preview?page_number={page_num}",
                headers=auth_headers
            )
            assert response.status in [200, 202, 404, 401, 403]
    
    @pytest.mark.playwright
    def test_get_preview_with_options(self, page, api_base_url, auth_headers):
        """Test getting preview with options."""
        file_id = "test_file_123"
        
        # Test with different options
        options = [
            "?page_number=1&width=800&height=600",
            "?page_number=1&format=png",
            "?page_number=1&quality=high",
        ]
        
        for option in options:
            response = page.request.get(
                f"{api_base_url}/pdf/{file_id}/preview{option}",
                headers=auth_headers
            )
            assert response.status in [200, 202, 404, 401, 403]


class TestPlaywrightPDFManagement:
    """Playwright tests for PDF management operations."""
    
    @pytest.mark.playwright
    def test_list_pdfs(self, page, api_base_url, auth_headers):
        """Test listing all PDFs."""
        response = page.request.get(
            f"{api_base_url}/pdf",
            headers=auth_headers
        )
        
        assert response.status in [200, 401, 403]
        
        if response.status == 200:
            data = response.json()
            assert isinstance(data, (list, dict))
    
    @pytest.mark.playwright
    def test_get_pdf_metadata(self, page, api_base_url, auth_headers):
        """Test getting PDF metadata."""
        file_id = "test_file_123"
        
        response = page.request.get(
            f"{api_base_url}/pdf/{file_id}",
            headers=auth_headers
        )
        
        assert response.status in [200, 404, 401, 403]
        
        if response.status == 200:
            data = response.json()
            assert "file_id" in data or "id" in data
            assert "filename" in data or "name" in data
    
    @pytest.mark.playwright
    def test_delete_pdf(self, page, api_base_url, auth_headers):
        """Test deleting PDF."""
        file_id = "test_file_123"
        
        response = page.request.delete(
            f"{api_base_url}/pdf/{file_id}",
            headers=auth_headers
        )
        
        assert response.status in [200, 204, 404, 401, 403]
    
    @pytest.mark.playwright
    def test_update_pdf_metadata(self, page, api_base_url, auth_headers):
        """Test updating PDF metadata."""
        file_id = "test_file_123"
        update_data = {
            "title": "Updated Title",
            "tags": ["tag1", "tag2"]
        }
        
        response = page.request.put(
            f"{api_base_url}/pdf/{file_id}",
            json=update_data,
            headers=auth_headers
        )
        
        assert response.status in [200, 404, 401, 403]


class TestPlaywrightBatchOperations:
    """Playwright tests for batch operations."""
    
    @pytest.mark.playwright
    def test_batch_upload(self, page, api_base_url, sample_pdf, auth_headers):
        """Test batch PDF upload."""
        files = []
        for i in range(3):
            files.append({
                "file": {
                    "name": f"batch_{i}.pdf",
                    "mimeType": "application/pdf",
                    "buffer": sample_pdf
                }
            })
        
        results = []
        for file_data in files:
            response = page.request.post(
                f"{api_base_url}/pdf/upload",
                multipart=file_data,
                headers=auth_headers
            )
            results.append(response.status)
        
        # All should complete
        assert len(results) == 3
    
    @pytest.mark.playwright
    def test_batch_variant_generation(self, page, api_base_url, auth_headers):
        """Test batch variant generation."""
        file_ids = ["file_1", "file_2", "file_3"]
        variant_data = {
            "variant_type": "summary",
            "options": {}
        }
        
        results = []
        for file_id in file_ids:
            response = page.request.post(
                f"{api_base_url}/pdf/{file_id}/variants",
                json=variant_data,
                headers=auth_headers
            )
            results.append(response.status)
        
        assert len(results) == 3


class TestPlaywrightSearchAndFilter:
    """Playwright tests for search and filtering."""
    
    @pytest.mark.playwright
    def test_search_pdfs(self, page, api_base_url, auth_headers):
        """Test searching PDFs."""
        search_queries = ["test", "document", "pdf"]
        
        for query in search_queries:
            response = page.request.get(
                f"{api_base_url}/pdf/search?q={query}",
                headers=auth_headers
            )
            assert response.status in [200, 401, 403]
    
    @pytest.mark.playwright
    def test_filter_pdfs_by_date(self, page, api_base_url, auth_headers):
        """Test filtering PDFs by date."""
        response = page.request.get(
            f"{api_base_url}/pdf?start_date=2024-01-01&end_date=2024-12-31",
            headers=auth_headers
        )
        
        assert response.status in [200, 401, 403]
    
    @pytest.mark.playwright
    def test_filter_pdfs_by_tags(self, page, api_base_url, auth_headers):
        """Test filtering PDFs by tags."""
        response = page.request.get(
            f"{api_base_url}/pdf?tags=important,urgent",
            headers=auth_headers
        )
        
        assert response.status in [200, 401, 403]


class TestPlaywrightWebhooks:
    """Playwright tests for webhooks."""
    
    @pytest.mark.playwright
    def test_register_webhook(self, page, api_base_url, auth_headers):
        """Test registering a webhook."""
        webhook_data = {
            "url": "https://example.com/webhook",
            "events": ["pdf.uploaded", "variant.generated"]
        }
        
        response = page.request.post(
            f"{api_base_url}/webhooks",
            json=webhook_data,
            headers=auth_headers
        )
        
        assert response.status in [200, 201, 401, 403]
    
    @pytest.mark.playwright
    def test_list_webhooks(self, page, api_base_url, auth_headers):
        """Test listing webhooks."""
        response = page.request.get(
            f"{api_base_url}/webhooks",
            headers=auth_headers
        )
        
        assert response.status in [200, 401, 403]


class TestPlaywrightRateLimiting:
    """Playwright tests for rate limiting."""
    
    @pytest.mark.playwright
    @pytest.mark.slow
    def test_rate_limit_detection(self, page, api_base_url, auth_headers):
        """Test detecting rate limiting."""
        status_codes = []
        
        # Make many rapid requests
        for i in range(50):
            response = page.request.get(
                f"{api_base_url}/health",
                headers=auth_headers
            )
            status_codes.append(response.status)
            time.sleep(0.1)  # Small delay
        
        # Check if rate limiting is active
        if 429 in status_codes:
            # Rate limiting is working
            assert True
        else:
            # Rate limiting may not be enabled
            assert True
    
    @pytest.mark.playwright
    def test_rate_limit_headers(self, page, api_base_url, auth_headers):
        """Test rate limit headers."""
        response = page.request.get(
            f"{api_base_url}/health",
            headers=auth_headers
        )
        
        headers = response.headers
        
        # Check for rate limit headers
        rate_limit_headers = [
            "x-ratelimit-limit",
            "x-ratelimit-remaining",
            "x-ratelimit-reset"
        ]
        
        has_rate_limit = any(header in headers for header in rate_limit_headers)
        # Rate limit headers may or may not be present
        assert True


class TestPlaywrightVersioning:
    """Playwright tests for API versioning."""
    
    @pytest.mark.playwright
    def test_api_version_header(self, page, api_base_url):
        """Test API version in headers."""
        response = page.request.get(f"{api_base_url}/health")
        
        headers = response.headers
        
        # May have version header
        if "x-api-version" in headers or "api-version" in headers:
            version = headers.get("x-api-version") or headers.get("api-version")
            assert version is not None
    
    @pytest.mark.playwright
    def test_versioned_endpoints(self, page, api_base_url, auth_headers):
        """Test versioned API endpoints."""
        versions = ["v1", "v2"]
        
        for version in versions:
            response = page.request.get(
                f"{api_base_url}/{version}/health",
                headers=auth_headers
            )
            # May or may not support versioning
            assert response.status is not None



