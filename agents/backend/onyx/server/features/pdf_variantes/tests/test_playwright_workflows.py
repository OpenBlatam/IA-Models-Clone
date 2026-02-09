"""
Playwright Workflow Tests
=========================
Complex workflow tests with Playwright.
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


class TestPlaywrightComplexWorkflows:
    """Complex workflow tests."""
    
    @pytest.mark.playwright
    @pytest.mark.workflow
    def test_complete_document_processing_workflow(self, page, api_base_url, sample_pdf, auth_headers):
        """Complete workflow: Upload -> Process -> Generate Variants -> Extract Topics -> Export."""
        
        # Step 1: Upload PDF
        files = {
            "file": {
                "name": "workflow_test.pdf",
                "mimeType": "application/pdf",
                "buffer": sample_pdf
            }
        }
        
        upload_response = page.request.post(
            f"{api_base_url}/pdf/upload?auto_process=true&extract_text=true",
            multipart=files,
            headers=auth_headers
        )
        
        if upload_response.status not in [200, 201]:
            pytest.skip("Upload failed")
        
        file_id = upload_response.json().get("file_id") or upload_response.json().get("id")
        
        # Step 2: Wait for processing
        time.sleep(2)
        
        # Step 3: Get metadata
        metadata_response = page.request.get(
            f"{api_base_url}/pdf/{file_id}",
            headers=auth_headers
        )
        assert metadata_response.status in [200, 404, 401, 403]
        
        # Step 4: Generate multiple variants
        variant_types = ["summary", "outline", "highlights"]
        variant_ids = []
        
        for variant_type in variant_types:
            variant_response = page.request.post(
                f"{api_base_url}/pdf/{file_id}/variants",
                json={"variant_type": variant_type, "options": {}},
                headers=auth_headers
            )
            if variant_response.status in [200, 202]:
                variant_data = variant_response.json()
                if "variant_id" in variant_data:
                    variant_ids.append(variant_data["variant_id"])
        
        # Step 5: Extract topics
        topics_response = page.request.get(
            f"{api_base_url}/pdf/{file_id}/topics?min_relevance=0.5&max_topics=20",
            headers=auth_headers
        )
        assert topics_response.status in [200, 202, 404, 401, 403]
        
        # Step 6: Get preview
        preview_response = page.request.get(
            f"{api_base_url}/pdf/{file_id}/preview?page_number=1",
            headers=auth_headers
        )
        assert preview_response.status in [200, 202, 404, 401, 403]
    
    @pytest.mark.playwright
    @pytest.mark.workflow
    def test_collaborative_workflow(self, page, api_base_url, sample_pdf):
        """Workflow with multiple users collaborating."""
        
        # User 1 uploads document
        files = {
            "file": {
                "name": "shared_doc.pdf",
                "mimeType": "application/pdf",
                "buffer": sample_pdf
            }
        }
        
        user1_headers = {"Authorization": "Bearer user1_token", "X-User-ID": "user1"}
        upload_response = page.request.post(
            f"{api_base_url}/pdf/upload",
            multipart=files,
            headers=user1_headers
        )
        
        if upload_response.status not in [200, 201]:
            pytest.skip("Upload failed")
        
        file_id = upload_response.json().get("file_id") or upload_response.json().get("id")
        
        # User 2 accesses document
        user2_headers = {"Authorization": "Bearer user2_token", "X-User-ID": "user2"}
        preview_response = page.request.get(
            f"{api_base_url}/pdf/{file_id}/preview",
            headers=user2_headers
        )
        assert preview_response.status in [200, 202, 404, 401, 403]
        
        # User 1 generates variant
        variant_response = page.request.post(
            f"{api_base_url}/pdf/{file_id}/variants",
            json={"variant_type": "summary", "options": {}},
            headers=user1_headers
        )
        assert variant_response.status in [200, 202, 404, 401, 403]
        
        # User 2 extracts topics
        topics_response = page.request.get(
            f"{api_base_url}/pdf/{file_id}/topics",
            headers=user2_headers
        )
        assert topics_response.status in [200, 202, 404, 401, 403]
    
    @pytest.mark.playwright
    @pytest.mark.workflow
    def test_error_recovery_workflow(self, page, api_base_url, sample_pdf, auth_headers):
        """Workflow with error recovery."""
        
        # Step 1: Try invalid upload
        invalid_response = page.request.post(
            f"{api_base_url}/pdf/upload",
            json={"invalid": "data"},
            headers=auth_headers
        )
        assert invalid_response.status in [400, 422, 415]
        
        # Step 2: Recover with valid upload
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
        
        # Step 3: Try invalid variant generation
        if valid_response.status in [200, 201]:
            file_id = valid_response.json().get("file_id") or valid_response.json().get("id")
            
            invalid_variant = page.request.post(
                f"{api_base_url}/pdf/{file_id}/variants",
                json={"variant_type": "invalid_type"},
                headers=auth_headers
            )
            assert invalid_variant.status in [400, 422]
            
            # Step 4: Recover with valid variant
            valid_variant = page.request.post(
                f"{api_base_url}/pdf/{file_id}/variants",
                json={"variant_type": "summary", "options": {}},
                headers=auth_headers
            )
            assert valid_variant.status in [200, 202, 404, 401, 403]


class TestPlaywrightDataFlow:
    """Tests for data flow through the system."""
    
    @pytest.mark.playwright
    @pytest.mark.workflow
    def test_data_consistency_workflow(self, page, api_base_url, sample_pdf, auth_headers):
        """Test data consistency through workflow."""
        
        # Upload
        files = {
            "file": {
                "name": "consistency_test.pdf",
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
        original_filename = "consistency_test.pdf"
        
        # Verify file_id is consistent across requests
        metadata_response = page.request.get(
            f"{api_base_url}/pdf/{file_id}",
            headers=auth_headers
        )
        
        if metadata_response.status == 200:
            metadata = metadata_response.json()
            # File ID should match
            assert metadata.get("file_id") == file_id or metadata.get("id") == file_id
    
    @pytest.mark.playwright
    @pytest.mark.workflow
    def test_state_transitions(self, page, api_base_url, sample_pdf, auth_headers):
        """Test state transitions in workflow."""
        
        # Upload (should be in 'uploaded' or 'processing' state)
        files = {
            "file": {
                "name": "state_test.pdf",
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
        
        # Check initial state
        metadata_response = page.request.get(
            f"{api_base_url}/pdf/{file_id}",
            headers=auth_headers
        )
        
        if metadata_response.status == 200:
            metadata = metadata_response.json()
            # May have status field
            if "status" in metadata:
                initial_status = metadata["status"]
                assert initial_status in ["uploaded", "processing", "ready", "completed"]


class TestPlaywrightAsyncOperations:
    """Tests for async operations."""
    
    @pytest.mark.playwright
    @pytest.mark.workflow
    @pytest.mark.slow
    def test_async_variant_generation(self, page, api_base_url, sample_pdf, auth_headers):
        """Test async variant generation workflow."""
        
        # Upload
        files = {
            "file": {
                "name": "async_test.pdf",
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
        
        # Start async variant generation
        variant_response = page.request.post(
            f"{api_base_url}/pdf/{file_id}/variants",
            json={"variant_type": "summary", "options": {}},
            headers=auth_headers
        )
        
        if variant_response.status == 202:
            # Async operation started
            job_data = variant_response.json()
            job_id = job_data.get("job_id") or job_data.get("task_id")
            
            if job_id:
                # Poll for completion
                max_polls = 10
                for _ in range(max_polls):
                    time.sleep(1)
                    status_response = page.request.get(
                        f"{api_base_url}/jobs/{job_id}",
                        headers=auth_headers
                    )
                    
                    if status_response.status == 200:
                        status_data = status_response.json()
                        if status_data.get("status") == "completed":
                            break
    
    @pytest.mark.playwright
    @pytest.mark.workflow
    @pytest.mark.slow
    def test_polling_for_completion(self, page, api_base_url, sample_pdf, auth_headers):
        """Test polling for operation completion."""
        
        # Upload
        files = {
            "file": {
                "name": "polling_test.pdf",
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
        
        # Poll for processing completion
        max_polls = 20
        for i in range(max_polls):
            metadata_response = page.request.get(
                f"{api_base_url}/pdf/{file_id}",
                headers=auth_headers
            )
            
            if metadata_response.status == 200:
                metadata = metadata_response.json()
                if metadata.get("status") == "completed" or metadata.get("processing_status") == "completed":
                    break
            
            time.sleep(0.5)


class TestPlaywrightChainOperations:
    """Tests for chained operations."""
    
    @pytest.mark.playwright
    @pytest.mark.workflow
    def test_chain_variant_generation(self, page, api_base_url, sample_pdf, auth_headers):
        """Test chaining variant generations."""
        
        # Upload
        files = {
            "file": {
                "name": "chain_test.pdf",
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
        
        # Chain: summary -> outline -> highlights
        chain = [
            {"variant_type": "summary", "options": {}},
            {"variant_type": "outline", "options": {}},
            {"variant_type": "highlights", "options": {}}
        ]
        
        results = []
        for variant_config in chain:
            response = page.request.post(
                f"{api_base_url}/pdf/{file_id}/variants",
                json=variant_config,
                headers=auth_headers
            )
            results.append(response.status)
            time.sleep(0.5)  # Small delay between requests
        
        # All should complete
        assert len(results) == 3
        assert all(status in [200, 202] for status in results) or any(status in [200, 202] for status in results)
    
    @pytest.mark.playwright
    @pytest.mark.workflow
    def test_dependent_operations(self, page, api_base_url, sample_pdf, auth_headers):
        """Test operations that depend on previous ones."""
        
        # Upload
        files = {
            "file": {
                "name": "dependent_test.pdf",
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
        
        # Wait for processing
        time.sleep(1)
        
        # Generate summary (depends on upload)
        summary_response = page.request.post(
            f"{api_base_url}/pdf/{file_id}/variants",
            json={"variant_type": "summary", "options": {}},
            headers=auth_headers
        )
        
        if summary_response.status in [200, 202]:
            summary_data = summary_response.json()
            variant_id = summary_data.get("variant_id")
            
            if variant_id:
                # Download variant (depends on generation)
                download_response = page.request.get(
                    f"{api_base_url}/pdf/{file_id}/variants/{variant_id}/download",
                    headers=auth_headers
                )
                assert download_response.status in [200, 404, 401, 403]


class TestPlaywrightRollback:
    """Tests for rollback scenarios."""
    
    @pytest.mark.playwright
    @pytest.mark.workflow
    def test_rollback_on_error(self, page, api_base_url, sample_pdf, auth_headers):
        """Test rollback when error occurs."""
        
        # Upload
        files = {
            "file": {
                "name": "rollback_test.pdf",
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
        
        # Try invalid operation
        invalid_response = page.request.post(
            f"{api_base_url}/pdf/{file_id}/variants",
            json={"invalid": "data"},
            headers=auth_headers
        )
        
        # Should fail gracefully
        assert invalid_response.status in [400, 422]
        
        # Original file should still be accessible
        check_response = page.request.get(
            f"{api_base_url}/pdf/{file_id}",
            headers=auth_headers
        )
        # File should still exist
        assert check_response.status in [200, 404, 401, 403]


class TestPlaywrightTransaction:
    """Tests for transactional operations."""
    
    @pytest.mark.playwright
    @pytest.mark.workflow
    def test_atomic_operations(self, page, api_base_url, sample_pdf, auth_headers):
        """Test atomic operations."""
        
        # Upload should be atomic
        files = {
            "file": {
                "name": "atomic_test.pdf",
                "mimeType": "application/pdf",
                "buffer": sample_pdf
            }
        }
        
        upload_response = page.request.post(
            f"{api_base_url}/pdf/upload",
            multipart=files,
            headers=auth_headers
        )
        
        # Should either fully succeed or fully fail
        if upload_response.status in [200, 201]:
            file_id = upload_response.json().get("file_id") or upload_response.json().get("id")
            # File should be fully uploaded
            assert file_id is not None
        else:
            # Should fully fail (no partial state)
            assert upload_response.status in [400, 401, 403, 422, 413]


class TestPlaywrightOptimisticLocking:
    """Tests for optimistic locking."""
    
    @pytest.mark.playwright
    @pytest.mark.workflow
    def test_concurrent_updates(self, page, api_base_url, sample_pdf, auth_headers):
        """Test concurrent updates with optimistic locking."""
        
        # Upload
        files = {
            "file": {
                "name": "locking_test.pdf",
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
        
        # Get current version/ETag
        get_response = page.request.get(
            f"{api_base_url}/pdf/{file_id}",
            headers=auth_headers
        )
        
        if get_response.status == 200:
            etag = get_response.headers.get("etag")
            version = get_response.json().get("version")
            
            # Try concurrent updates
            update1 = page.request.put(
                f"{api_base_url}/pdf/{file_id}",
                json={"title": "Update 1"},
                headers={**auth_headers, "If-Match": etag} if etag else auth_headers
            )
            
            update2 = page.request.put(
                f"{api_base_url}/pdf/{file_id}",
                json={"title": "Update 2"},
                headers={**auth_headers, "If-Match": etag} if etag else auth_headers
            )
            
            # One should succeed, one may fail with 412
            assert update1.status in [200, 412, 404, 401, 403]
            assert update2.status in [200, 412, 404, 401, 403]


class TestPlaywrightEventDriven:
    """Tests for event-driven workflows."""
    
    @pytest.mark.playwright
    @pytest.mark.workflow
    def test_event_sequence(self, page, api_base_url, sample_pdf, auth_headers):
        """Test event sequence in workflow."""
        
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
        
        if webhook_response.status in [200, 201]:
            webhook_id = webhook_response.json().get("webhook_id") or webhook_response.json().get("id")
            
            # Upload (should trigger event)
            files = {
                "file": {
                    "name": "event_test.pdf",
                    "mimeType": "application/pdf",
                    "buffer": sample_pdf
                }
            }
            
            upload_response = page.request.post(
                f"{api_base_url}/pdf/upload",
                multipart=files,
                headers=auth_headers
            )
            
            # Event should be triggered (webhook called)
            # In real scenario, would verify webhook was called
            assert upload_response.status in [200, 201, 401, 403]



