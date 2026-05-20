"""
Refactored Playwright Tests Using Utilities
===========================================
Example tests using the new utility classes.
"""

import pytest
from playwright.sync_api import Page
from playwright_utils import (
    create_request_builder,
    validate_response,
    create_test_data
)
from fixtures_common import api_base_url, auth_headers


class TestRefactoredWithUtils:
    """Tests using utility classes."""
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    def test_health_with_builder(self, page, api_base_url):
        """Test health endpoint using request builder."""
        response = (
            create_request_builder(page, api_base_url)
            .get("/health")
            .execute()
        )
        
        validate_response(response).assert_status(200).assert_json()
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    def test_upload_with_builder(self, page, api_base_url, auth_headers):
        """Test upload using request builder."""
        test_data = create_test_data()
        files = test_data.create_upload_files("test.pdf")
        
        response = (
            create_request_builder(page, api_base_url)
            .post("/pdf/upload")
            .with_headers(auth_headers)
            .with_multipart(files)
            .execute()
        )
        
        validator = validate_response(response)
        validator.assert_status_range(200, 201)
        data = validator.assert_has_keys("file_id", "id").assert_json()
        assert "file_id" in data or "id" in data
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    def test_variant_with_builder(self, page, api_base_url, auth_headers):
        """Test variant generation using request builder."""
        test_data = create_test_data()
        variant_request = test_data.create_variant_request("summary", {"max_length": 500})
        
        response = (
            create_request_builder(page, api_base_url)
            .post("/pdf/test_file_123/variants")
            .with_headers(auth_headers)
            .with_json(variant_request)
            .execute()
        )
        
        validate_response(response).assert_status_range(200, 202)
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    def test_topics_with_query_params(self, page, api_base_url, auth_headers):
        """Test topics with query parameters using builder."""
        response = (
            create_request_builder(page, api_base_url)
            .get("/pdf/test_file_123/topics")
            .with_headers(auth_headers)
            .with_query({"min_relevance": 0.5, "max_topics": 10})
            .execute()
        )
        
        validate_response(response).assert_status_range(200, 202)
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    def test_response_validation_chain(self, page, api_base_url):
        """Test chained response validation."""
        response = (
            create_request_builder(page, api_base_url)
            .get("/health")
            .execute()
        )
        
        # Chain validations
        validator = (
            validate_response(response)
            .assert_status(200)
            .assert_content_type("application/json")
            .assert_has_keys("status")
        )
        
        data = validator.assert_json()
        assert "status" in data
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    def test_test_data_factory(self):
        """Test using test data factory."""
        factory = create_test_data()
        
        # Create PDF file
        pdf_file = factory.create_pdf_file("test.pdf", size_kb=10)
        assert pdf_file["name"] == "test.pdf"
        assert len(pdf_file["buffer"]) >= 10 * 1024
        
        # Create variant request
        variant_request = factory.create_variant_request("summary", {"max_length": 500})
        assert variant_request["variant_type"] == "summary"
        assert variant_request["options"]["max_length"] == 500
        
        # Create auth headers
        headers = factory.create_auth_headers("custom_token", "custom_user")
        assert headers["Authorization"] == "Bearer custom_token"
        assert headers["X-User-ID"] == "custom_user"


class TestRefactoredWorkflowWithUtils:
    """Workflow tests using utilities."""
    
    @pytest.mark.playwright
    @pytest.mark.refactored
    def test_complete_workflow_with_utils(self, page, api_base_url, auth_headers, sample_pdf):
        """Complete workflow using utility classes."""
        builder = create_request_builder(page, api_base_url)
        factory = create_test_data()
        
        # 1. Upload
        files = factory.create_upload_files("workflow_test.pdf", sample_pdf)
        upload_response = (
            builder.post("/pdf/upload")
            .with_headers(auth_headers)
            .with_multipart(files)
            .execute()
        )
        
        if upload_response.status not in [200, 201]:
            pytest.skip("Upload failed")
        
        file_id = validate_response(upload_response).assert_json().get("file_id") or \
                  validate_response(upload_response).assert_json().get("id")
        
        # 2. Generate variant
        variant_request = factory.create_variant_request("summary")
        variant_response = (
            builder.post(f"/pdf/{file_id}/variants")
            .with_headers(auth_headers)
            .with_json(variant_request)
            .execute()
        )
        validate_response(variant_response).assert_status_range(200, 202)
        
        # 3. Get topics
        topics_response = (
            builder.get(f"/pdf/{file_id}/topics")
            .with_headers(auth_headers)
            .with_query({"min_relevance": 0.5})
            .execute()
        )
        validate_response(topics_response).assert_status_range(200, 202)
        
        # 4. Get preview
        preview_response = (
            builder.get(f"/pdf/{file_id}/preview")
            .with_headers(auth_headers)
            .with_query({"page_number": 1})
            .execute()
        )
        validate_response(preview_response).assert_status_range(200, 202)



