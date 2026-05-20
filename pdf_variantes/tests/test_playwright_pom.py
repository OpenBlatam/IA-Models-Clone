"""
Playwright Tests Using Page Object Model
========================================
Tests using Page Object Model pattern.
"""

import pytest
from playwright.sync_api import Page
from playwright_pages import APIPage
from fixtures_common import api_base_url, auth_headers, sample_pdf


class TestPlaywrightPOM:
    """Tests using Page Object Model."""
    
    @pytest.mark.playwright
    @pytest.mark.pom
    def test_health_with_pom(self, page, api_base_url):
        """Test health check using POM."""
        api_page = APIPage(page, api_base_url)
        status = api_page.health.get_status()
        assert status in ["healthy", "ok", "up"]
    
    @pytest.mark.playwright
    @pytest.mark.pom
    def test_upload_with_pom(self, page, api_base_url, auth_headers, sample_pdf):
        """Test upload using POM."""
        api_page = APIPage(page, api_base_url)
        file_id = api_page.upload.get_file_id("test.pdf", sample_pdf, auth_headers)
        assert file_id is not None
    
    @pytest.mark.playwright
    @pytest.mark.pom
    def test_variant_with_pom(self, page, api_base_url, auth_headers, sample_pdf):
        """Test variant generation using POM."""
        api_page = APIPage(page, api_base_url)
        
        # Upload first
        file_id = api_page.upload.get_file_id("variant_test.pdf", sample_pdf, auth_headers)
        if not file_id:
            pytest.skip("Upload failed")
        
        # Generate variant
        variant_result = api_page.variant.generate_variant(file_id, "summary", auth_headers)
        assert variant_result is not None
    
    @pytest.mark.playwright
    @pytest.mark.pom
    def test_topics_with_pom(self, page, api_base_url, auth_headers, sample_pdf):
        """Test topic extraction using POM."""
        api_page = APIPage(page, api_base_url)
        
        # Upload first
        file_id = api_page.upload.get_file_id("topics_test.pdf", sample_pdf, auth_headers)
        if not file_id:
            pytest.skip("Upload failed")
        
        # Extract topics
        topics_result = api_page.topic.extract_topics(file_id, auth_headers)
        assert topics_result is not None
    
    @pytest.mark.playwright
    @pytest.mark.pom
    def test_preview_with_pom(self, page, api_base_url, auth_headers, sample_pdf):
        """Test preview using POM."""
        api_page = APIPage(page, api_base_url)
        
        # Upload first
        file_id = api_page.upload.get_file_id("preview_test.pdf", sample_pdf, auth_headers)
        if not file_id:
            pytest.skip("Upload failed")
        
        # Get preview
        preview_result = api_page.preview.get_preview(file_id, 1, auth_headers)
        assert preview_result is not None
    
    @pytest.mark.playwright
    @pytest.mark.pom
    def test_complete_workflow_with_pom(self, page, api_base_url, auth_headers, sample_pdf):
        """Test complete workflow using POM."""
        api_page = APIPage(page, api_base_url)
        
        result = api_page.complete_workflow("workflow_test.pdf", sample_pdf, auth_headers)
        
        assert "file_id" in result
        assert "upload" in result
        assert "variant" in result
        assert "topics" in result
        assert "preview" in result
    
    @pytest.mark.playwright
    @pytest.mark.pom
    def test_multiple_variants_with_pom(self, page, api_base_url, auth_headers, sample_pdf):
        """Test generating multiple variants using POM."""
        api_page = APIPage(page, api_base_url)
        
        # Upload
        file_id = api_page.upload.get_file_id("multi_variant_test.pdf", sample_pdf, auth_headers)
        if not file_id:
            pytest.skip("Upload failed")
        
        # Generate multiple variants
        variant_types = ["summary", "outline", "highlights"]
        results = api_page.variant.generate_multiple_variants(file_id, variant_types, auth_headers)
        
        assert len(results) == len(variant_types)
    
    @pytest.mark.playwright
    @pytest.mark.pom
    def test_search_with_pom(self, page, api_base_url, auth_headers):
        """Test search using POM."""
        api_page = APIPage(page, api_base_url)
        
        # Search
        results = api_page.search.search("test", auth_headers)
        assert isinstance(results, (list, dict))
        
        # Search by tags
        tag_results = api_page.search.search_by_tags(["important"], auth_headers)
        assert isinstance(tag_results, (list, dict))
    
    @pytest.mark.playwright
    @pytest.mark.pom
    def test_management_with_pom(self, page, api_base_url, auth_headers, sample_pdf):
        """Test PDF management using POM."""
        api_page = APIPage(page, api_base_url)
        
        # Upload
        file_id = api_page.upload.get_file_id("management_test.pdf", sample_pdf, auth_headers)
        if not file_id:
            pytest.skip("Upload failed")
        
        # Get metadata
        metadata = api_page.management.get_metadata(file_id, auth_headers)
        assert metadata is not None
        
        # Update metadata
        updated = api_page.management.update_metadata(
            file_id,
            {"title": "Updated Title"},
            auth_headers
        )
        assert updated is not None
        
        # Delete
        deleted = api_page.management.delete_pdf(file_id, auth_headers)
        assert deleted is True



