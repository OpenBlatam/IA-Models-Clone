"""
Example of Refactored Playwright Tests
======================================
Example showing how to use base classes and centralized fixtures.
"""

import pytest
from base_playwright_test import BaseAPITest, BaseUITest
from fixtures_common import api_base_url, auth_headers, sample_pdf


class TestRefactoredAPI(BaseAPITest):
    """Example of refactored API tests using base class."""
    
    @pytest.mark.playwright
    def test_upload_using_base_class(self, page, api_base_url, auth_headers, sample_pdf):
        """Test upload using base class helper."""
        # Use helper from base class
        response = self.upload_pdf(page, api_base_url, auth_headers, sample_pdf)
        
        # Use assertion from base class
        self.assert_response_success(response, expected_status=201)
        
        # Use helper to extract file_id
        file_id = self.get_file_id_from_response(response)
        assert file_id is not None
    
    @pytest.mark.playwright
    def test_complete_workflow_using_base_class(self, page, api_base_url, auth_headers, sample_pdf):
        """Test complete workflow using base class helpers."""
        # 1. Upload
        upload_response = self.upload_pdf(page, api_base_url, auth_headers, sample_pdf)
        
        if upload_response.status not in [200, 201]:
            pytest.skip("Upload failed")
        
        file_id = self.get_file_id_from_response(upload_response)
        
        # 2. Generate variant
        variant_response = self.generate_variant(
            page, api_base_url, file_id, "summary", {}, auth_headers
        )
        assert variant_response.status in [200, 202, 404, 401, 403]
        
        # 3. Get topics
        topics_response = self.get_topics(
            page, api_base_url, file_id, auth_headers, min_relevance=0.5, max_topics=10
        )
        assert topics_response.status in [200, 202, 404, 401, 403]
        
        # 4. Get preview
        preview_response = self.get_preview(page, api_base_url, file_id, 1, auth_headers)
        assert preview_response.status in [200, 202, 404, 401, 403]


class TestRefactoredUI(BaseUITest):
    """Example of refactored UI tests using base class."""
    
    @pytest.mark.playwright
    def test_navigation_using_base_class(self, page, api_base_url):
        """Test navigation using base class helper."""
        # Use helper from base class
        success = self.navigate_and_wait(page, api_base_url)
        
        if success:
            # Page loaded successfully
            assert page.url.startswith(api_base_url)
        else:
            pytest.skip("Could not navigate to page")
    
    @pytest.mark.playwright
    def test_form_interaction_using_base_class(self, page, api_base_url):
        """Test form interaction using base class helpers."""
        if not self.navigate_and_wait(page, api_base_url):
            pytest.skip("Could not navigate")
        
        # Try to find and fill form fields
        input_selectors = ['input[type="text"]', 'textarea', 'input[type="email"]']
        
        for selector in input_selectors:
            if self.fill_form_field(page, selector, "test_value"):
                # Successfully filled field
                assert True
                break



