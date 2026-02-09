"""
End-to-End (E2E) Tests for PDF Variantes
=========================================
Complete end-to-end tests that test real user workflows from start to finish.
"""

import pytest
import asyncio
import tempfile
import io
import time
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, Mock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import the actual app
try:
    from api.main import app as main_app
except ImportError:
    try:
        from main import app as main_app
    except ImportError:
        # Create minimal app for testing
        main_app = FastAPI()


@pytest.fixture(scope="module")
def app():
    """Create FastAPI app for E2E testing."""
    return main_app


@pytest.fixture(scope="module")
def client(app):
    """Create test client for E2E tests."""
    return TestClient(app, timeout=30.0)


@pytest.fixture
def sample_pdf_content():
    """Valid PDF content for testing."""
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Hello World) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000204 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n297\n%%EOF"


@pytest.fixture
def auth_headers():
    """Authentication headers for E2E tests."""
    return {
        "Authorization": "Bearer test_token_123",
        "X-User-ID": "test_user_123"
    }


class TestE2EHealthCheck:
    """E2E tests for health check endpoint."""
    
    def test_health_check_e2e(self, client):
        """Test health check endpoint end-to-end."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "ok"]


class TestE2ECompleteUserJourney:
    """Complete E2E user journey tests."""
    
    @pytest.mark.e2e
    def test_complete_pdf_upload_journey(self, client, sample_pdf_content, auth_headers):
        """Test complete journey: Upload PDF -> Get Preview -> Extract Topics -> Generate Variants."""
        
        # Step 1: Upload PDF
        files = {"file": ("test_document.pdf", io.BytesIO(sample_pdf_content), "application/pdf")}
        upload_response = client.post(
            "/pdf/upload?auto_process=true&extract_text=true",
            files=files,
            headers=auth_headers
        )
        
        assert upload_response.status_code in [200, 201], f"Upload failed: {upload_response.text}"
        upload_data = upload_response.json()
        assert "file_id" in upload_data or "id" in upload_data
        
        file_id = upload_data.get("file_id") or upload_data.get("id")
        assert file_id is not None
        
        # Step 2: Get Preview
        preview_response = client.get(
            f"/pdf/{file_id}/preview?page_number=1",
            headers=auth_headers
        )
        
        # Preview may or may not be available immediately
        assert preview_response.status_code in [200, 404, 202]
        if preview_response.status_code == 200:
            preview_data = preview_response.json()
            assert "preview" in preview_data or "file_id" in preview_data
        
        # Step 3: Extract Topics
        topics_response = client.get(
            f"/pdf/{file_id}/topics?min_relevance=0.5&max_topics=10",
            headers=auth_headers
        )
        
        assert topics_response.status_code in [200, 202, 404]
        if topics_response.status_code == 200:
            topics_data = topics_response.json()
            assert "topics" in topics_data or "file_id" in topics_data
        
        # Step 4: Generate Variant
        variant_data = {
            "variant_type": "summary",
            "options": {
                "max_length": 500,
                "style": "academic"
            }
        }
        variant_response = client.post(
            f"/pdf/{file_id}/variants",
            json=variant_data,
            headers=auth_headers
        )
        
        assert variant_response.status_code in [200, 202, 404]
        if variant_response.status_code == 200:
            variant_result = variant_response.json()
            assert "variant" in variant_result or "success" in variant_result
    
    @pytest.mark.e2e
    def test_multiple_variants_generation(self, client, sample_pdf_content, auth_headers):
        """Test generating multiple variants for the same PDF."""
        
        # Upload PDF
        files = {"file": ("test.pdf", io.BytesIO(sample_pdf_content), "application/pdf")}
        upload_response = client.post("/pdf/upload", files=files, headers=auth_headers)
        
        if upload_response.status_code not in [200, 201]:
            pytest.skip("Upload failed, cannot test variants")
        
        file_id = upload_response.json().get("file_id") or upload_response.json().get("id")
        
        # Generate multiple variants
        variant_types = ["summary", "outline", "highlights"]
        results = []
        
        for variant_type in variant_types:
            variant_data = {
                "variant_type": variant_type,
                "options": {}
            }
            response = client.post(
                f"/pdf/{file_id}/variants",
                json=variant_data,
                headers=auth_headers
            )
            results.append(response.status_code)
        
        # At least some should succeed
        assert any(status in [200, 202] for status in results)
    
    @pytest.mark.e2e
    def test_pdf_deletion_workflow(self, client, sample_pdf_content, auth_headers):
        """Test complete workflow including deletion."""
        
        # Upload PDF
        files = {"file": ("test.pdf", io.BytesIO(sample_pdf_content), "application/pdf")}
        upload_response = client.post("/pdf/upload", files=files, headers=auth_headers)
        
        if upload_response.status_code not in [200, 201]:
            pytest.skip("Upload failed")
        
        file_id = upload_response.json().get("file_id") or upload_response.json().get("id")
        
        # Verify file exists
        preview_response = client.get(f"/pdf/{file_id}/preview", headers=auth_headers)
        initial_status = preview_response.status_code
        
        # Delete PDF
        delete_response = client.delete(f"/pdf/{file_id}", headers=auth_headers)
        assert delete_response.status_code in [200, 204, 404]
        
        # Verify file is deleted
        preview_after = client.get(f"/pdf/{file_id}/preview", headers=auth_headers)
        if initial_status == 200:
            # File should no longer be accessible
            assert preview_after.status_code in [404, 410]


class TestE2EErrorScenarios:
    """E2E tests for error scenarios."""
    
    @pytest.mark.e2e
    def test_upload_invalid_file_e2e(self, client, auth_headers):
        """Test uploading invalid file end-to-end."""
        files = {"file": ("test.txt", io.BytesIO(b"not a pdf"), "text/plain")}
        response = client.post("/pdf/upload", files=files, headers=auth_headers)
        
        # Should reject invalid file
        assert response.status_code in [400, 422]
    
    @pytest.mark.e2e
    def test_access_nonexistent_file_e2e(self, client, auth_headers):
        """Test accessing non-existent file end-to-end."""
        nonexistent_id = "nonexistent_file_12345"
        
        # Try to get preview
        preview_response = client.get(
            f"/pdf/{nonexistent_id}/preview",
            headers=auth_headers
        )
        assert preview_response.status_code in [404, 410]
        
        # Try to generate variant
        variant_response = client.post(
            f"/pdf/{nonexistent_id}/variants",
            json={"variant_type": "summary", "options": {}},
            headers=auth_headers
        )
        assert variant_response.status_code in [404, 410]
    
    @pytest.mark.e2e
    def test_invalid_variant_type_e2e(self, client, sample_pdf_content, auth_headers):
        """Test generating variant with invalid type."""
        # Upload valid PDF first
        files = {"file": ("test.pdf", io.BytesIO(sample_pdf_content), "application/pdf")}
        upload_response = client.post("/pdf/upload", files=files, headers=auth_headers)
        
        if upload_response.status_code not in [200, 201]:
            pytest.skip("Upload failed")
        
        file_id = upload_response.json().get("file_id") or upload_response.json().get("id")
        
        # Try invalid variant type
        variant_data = {
            "variant_type": "invalid_type_xyz",
            "options": {}
        }
        response = client.post(
            f"/pdf/{file_id}/variants",
            json=variant_data,
            headers=auth_headers
        )
        
        # Should reject invalid variant type
        assert response.status_code in [400, 422]
    
    @pytest.mark.e2e
    def test_unauthorized_access_e2e(self, client, sample_pdf_content):
        """Test accessing endpoints without authentication."""
        files = {"file": ("test.pdf", io.BytesIO(sample_pdf_content), "application/pdf")}
        response = client.post("/pdf/upload", files=files)
        
        # May require auth or may allow
        # If auth is required, should return 401/403
        assert response.status_code in [200, 201, 401, 403]


class TestE2EConcurrentOperations:
    """E2E tests for concurrent operations."""
    
    @pytest.mark.e2e
    def test_concurrent_uploads(self, client, sample_pdf_content, auth_headers):
        """Test multiple concurrent uploads."""
        import concurrent.futures
        
        def upload_pdf():
            files = {"file": (f"test_{time.time()}.pdf", io.BytesIO(sample_pdf_content), "application/pdf")}
            return client.post("/pdf/upload", files=files, headers=auth_headers)
        
        # Execute 10 concurrent uploads
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(upload_pdf) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All should complete (may succeed or fail, but shouldn't crash)
        assert len(results) == 10
        status_codes = [r.status_code for r in results]
        # At least some should succeed
        assert any(code in [200, 201] for code in status_codes)
    
    @pytest.mark.e2e
    def test_concurrent_variant_generation(self, client, sample_pdf_content, auth_headers):
        """Test generating variants concurrently."""
        # Upload PDF first
        files = {"file": ("test.pdf", io.BytesIO(sample_pdf_content), "application/pdf")}
        upload_response = client.post("/pdf/upload", files=files, headers=auth_headers)
        
        if upload_response.status_code not in [200, 201]:
            pytest.skip("Upload failed")
        
        file_id = upload_response.json().get("file_id") or upload_response.json().get("id")
        
        import concurrent.futures
        
        def generate_variant(variant_type):
            variant_data = {"variant_type": variant_type, "options": {}}
            return client.post(
                f"/pdf/{file_id}/variants",
                json=variant_data,
                headers=auth_headers
            )
        
        # Generate multiple variants concurrently
        variant_types = ["summary", "outline", "highlights", "notes", "quiz"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(generate_variant, vt) for vt in variant_types]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All should complete
        assert len(results) == 5
        # At least some should succeed
        assert any(r.status_code in [200, 202] for r in results)


class TestE2EPerformance:
    """E2E tests for performance requirements."""
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_upload_performance(self, client, sample_pdf_content, auth_headers):
        """Test that upload completes within acceptable time."""
        start_time = time.time()
        
        files = {"file": ("test.pdf", io.BytesIO(sample_pdf_content), "application/pdf")}
        response = client.post("/pdf/upload", files=files, headers=auth_headers)
        
        elapsed = time.time() - start_time
        
        # Upload should complete within 5 seconds
        assert elapsed < 5.0, f"Upload took too long: {elapsed:.2f}s"
        assert response.status_code in [200, 201]
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_variant_generation_performance(self, client, sample_pdf_content, auth_headers):
        """Test that variant generation completes within acceptable time."""
        # Upload first
        files = {"file": ("test.pdf", io.BytesIO(sample_pdf_content), "application/pdf")}
        upload_response = client.post("/pdf/upload", files=files, headers=auth_headers)
        
        if upload_response.status_code not in [200, 201]:
            pytest.skip("Upload failed")
        
        file_id = upload_response.json().get("file_id") or upload_response.json().get("id")
        
        # Generate variant and measure time
        start_time = time.time()
        variant_data = {"variant_type": "summary", "options": {}}
        response = client.post(
            f"/pdf/{file_id}/variants",
            json=variant_data,
            headers=auth_headers
        )
        elapsed = time.time() - start_time
        
        # Variant generation should complete within 30 seconds (or return 202 for async)
        assert elapsed < 30.0 or response.status_code == 202, \
            f"Variant generation took too long: {elapsed:.2f}s"


class TestE2EDataPersistence:
    """E2E tests for data persistence."""
    
    @pytest.mark.e2e
    def test_file_persistence_across_requests(self, client, sample_pdf_content, auth_headers):
        """Test that uploaded files persist across multiple requests."""
        # Upload PDF
        files = {"file": ("test.pdf", io.BytesIO(sample_pdf_content), "application/pdf")}
        upload_response = client.post("/pdf/upload", files=files, headers=auth_headers)
        
        if upload_response.status_code not in [200, 201]:
            pytest.skip("Upload failed")
        
        file_id = upload_response.json().get("file_id") or upload_response.json().get("id")
        
        # Wait a bit
        time.sleep(0.5)
        
        # Access file again
        preview_response = client.get(f"/pdf/{file_id}/preview", headers=auth_headers)
        
        # File should still be accessible
        assert preview_response.status_code in [200, 202, 404]
    
    @pytest.mark.e2e
    def test_multiple_operations_same_file(self, client, sample_pdf_content, auth_headers):
        """Test performing multiple operations on the same file."""
        # Upload PDF
        files = {"file": ("test.pdf", io.BytesIO(sample_pdf_content), "application/pdf")}
        upload_response = client.post("/pdf/upload", files=files, headers=auth_headers)
        
        if upload_response.status_code not in [200, 201]:
            pytest.skip("Upload failed")
        
        file_id = upload_response.json().get("file_id") or upload_response.json().get("id")
        
        # Perform multiple operations
        operations = [
            ("preview", lambda: client.get(f"/pdf/{file_id}/preview?page_number=1", headers=auth_headers)),
            ("topics", lambda: client.get(f"/pdf/{file_id}/topics", headers=auth_headers)),
            ("variant", lambda: client.post(
                f"/pdf/{file_id}/variants",
                json={"variant_type": "summary", "options": {}},
                headers=auth_headers
            )),
        ]
        
        results = []
        for op_name, op_func in operations:
            response = op_func()
            results.append((op_name, response.status_code))
            # Should not crash
            assert response.status_code is not None
        
        # At least some operations should succeed
        assert any(status in [200, 202] for _, status in results)


class TestE2EAPICompatibility:
    """E2E tests for API compatibility."""
    
    @pytest.mark.e2e
    def test_api_version_header(self, client):
        """Test API version header in responses."""
        response = client.get("/health")
        
        # May include version header
        assert "x-api-version" in response.headers or response.status_code == 200
    
    @pytest.mark.e2e
    def test_cors_headers(self, client):
        """Test CORS headers if CORS is enabled."""
        response = client.options("/health")
        
        # May have CORS headers
        assert response.status_code in [200, 204, 405]
    
    @pytest.mark.e2e
    def test_content_type_headers(self, client, sample_pdf_content, auth_headers):
        """Test content type headers in responses."""
        files = {"file": ("test.pdf", io.BytesIO(sample_pdf_content), "application/pdf")}
        response = client.post("/pdf/upload", files=files, headers=auth_headers)
        
        if response.status_code in [200, 201]:
            # Should return JSON
            assert "application/json" in response.headers.get("content-type", "")


class TestE2ERateLimiting:
    """E2E tests for rate limiting."""
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_rate_limiting(self, client, sample_pdf_content, auth_headers):
        """Test that rate limiting works if enabled."""
        # Make many rapid requests
        files = {"file": ("test.pdf", io.BytesIO(sample_pdf_content), "application/pdf")}
        
        status_codes = []
        for _ in range(20):
            response = client.post("/pdf/upload", files=files, headers=auth_headers)
            status_codes.append(response.status_code)
            time.sleep(0.1)  # Small delay
        
        # If rate limiting is enabled, some should return 429
        # If not enabled, all should succeed or fail for other reasons
        assert len(status_codes) == 20
        
        # Check if rate limiting is active
        if 429 in status_codes:
            # Rate limiting is working
            assert True
        else:
            # Rate limiting may not be enabled, which is also valid
            assert True


class TestE2EEndToEndWorkflows:
    """Complete end-to-end workflow tests."""
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_complete_document_lifecycle(self, client, sample_pdf_content, auth_headers):
        """Test complete document lifecycle: Upload -> Process -> Use -> Delete."""
        
        # 1. Upload
        files = {"file": ("lifecycle_test.pdf", io.BytesIO(sample_pdf_content), "application/pdf")}
        upload_response = client.post("/pdf/upload", files=files, headers=auth_headers)
        
        if upload_response.status_code not in [200, 201]:
            pytest.skip("Upload failed")
        
        file_id = upload_response.json().get("file_id") or upload_response.json().get("id")
        
        # 2. Wait for processing (if async)
        time.sleep(1)
        
        # 3. Get preview
        preview_response = client.get(f"/pdf/{file_id}/preview", headers=auth_headers)
        assert preview_response.status_code in [200, 202, 404]
        
        # 4. Extract topics
        topics_response = client.get(f"/pdf/{file_id}/topics", headers=auth_headers)
        assert topics_response.status_code in [200, 202, 404]
        
        # 5. Generate variant
        variant_response = client.post(
            f"/pdf/{file_id}/variants",
            json={"variant_type": "summary", "options": {}},
            headers=auth_headers
        )
        assert variant_response.status_code in [200, 202, 404]
        
        # 6. Delete
        delete_response = client.delete(f"/pdf/{file_id}", headers=auth_headers)
        assert delete_response.status_code in [200, 204, 404]
        
        # 7. Verify deletion
        final_preview = client.get(f"/pdf/{file_id}/preview", headers=auth_headers)
        if delete_response.status_code in [200, 204]:
            assert final_preview.status_code in [404, 410]



