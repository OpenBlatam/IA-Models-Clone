"""
Playwright Smoke Tests
======================
Quick smoke tests to verify basic functionality.
"""

import pytest
from playwright.sync_api import Page, Response
import time


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


class TestPlaywrightSmokeBasic:
    """Basic smoke tests."""
    
    @pytest.mark.playwright
    @pytest.mark.smoke
    def test_health_check_smoke(self, page, api_base_url):
        """Smoke test: Health check."""
        response = page.request.get(f"{api_base_url}/health")
        assert response.status == 200
    
    @pytest.mark.playwright
    @pytest.mark.smoke
    def test_api_accessible_smoke(self, page, api_base_url):
        """Smoke test: API is accessible."""
        response = page.request.get(f"{api_base_url}/health")
        assert response.status in [200, 503]
    
    @pytest.mark.playwright
    @pytest.mark.smoke
    def test_response_time_smoke(self, page, api_base_url):
        """Smoke test: Response time is acceptable."""
        start = time.time()
        response = page.request.get(f"{api_base_url}/health")
        elapsed = time.time() - start
        
        assert response.status == 200
        assert elapsed < 5.0, f"Response too slow: {elapsed:.3f}s"


class TestPlaywrightSmokeEndpoints:
    """Endpoint smoke tests."""
    
    @pytest.mark.playwright
    @pytest.mark.smoke
    def test_upload_endpoint_smoke(self, page, api_base_url, sample_pdf, auth_headers):
        """Smoke test: Upload endpoint works."""
        files = {
            "file": {
                "name": "smoke_test.pdf",
                "mimeType": "application/pdf",
                "buffer": sample_pdf
            }
        }
        
        response = page.request.post(
            f"{api_base_url}/pdf/upload",
            multipart=files,
            headers=auth_headers
        )
        
        # Should work or require different auth
        assert response.status in [200, 201, 401, 403]
    
    @pytest.mark.playwright
    @pytest.mark.smoke
    def test_preview_endpoint_smoke(self, page, api_base_url, auth_headers):
        """Smoke test: Preview endpoint works."""
        file_id = "test_file_123"
        
        response = page.request.get(
            f"{api_base_url}/pdf/{file_id}/preview",
            headers=auth_headers
        )
        
        # Should work or return appropriate error
        assert response.status in [200, 202, 404, 401, 403]
    
    @pytest.mark.playwright
    @pytest.mark.smoke
    def test_variants_endpoint_smoke(self, page, api_base_url, auth_headers):
        """Smoke test: Variants endpoint works."""
        file_id = "test_file_123"
        
        response = page.request.post(
            f"{api_base_url}/pdf/{file_id}/variants",
            json={"variant_type": "summary", "options": {}},
            headers=auth_headers
        )
        
        # Should work or return appropriate error
        assert response.status in [200, 202, 404, 401, 403]
    
    @pytest.mark.playwright
    @pytest.mark.smoke
    def test_topics_endpoint_smoke(self, page, api_base_url, auth_headers):
        """Smoke test: Topics endpoint works."""
        file_id = "test_file_123"
        
        response = page.request.get(
            f"{api_base_url}/pdf/{file_id}/topics",
            headers=auth_headers
        )
        
        # Should work or return appropriate error
        assert response.status in [200, 202, 404, 401, 403]


class TestPlaywrightSmokeWorkflow:
    """Workflow smoke tests."""
    
    @pytest.mark.playwright
    @pytest.mark.smoke
    def test_basic_workflow_smoke(self, page, api_base_url, sample_pdf, auth_headers):
        """Smoke test: Basic workflow works."""
        # 1. Health check
        health_response = page.request.get(f"{api_base_url}/health")
        assert health_response.status == 200
        
        # 2. Upload
        files = {
            "file": {
                "name": "workflow_smoke_test.pdf",
                "mimeType": "application/pdf",
                "buffer": sample_pdf
            }
        }
        
        upload_response = page.request.post(
            f"{api_base_url}/pdf/upload",
            multipart=files,
            headers=auth_headers
        )
        
        # Should work or require different auth
        assert upload_response.status in [200, 201, 401, 403]


class TestPlaywrightSmokeErrorHandling:
    """Error handling smoke tests."""
    
    @pytest.mark.playwright
    @pytest.mark.smoke
    def test_404_handling_smoke(self, page, api_base_url):
        """Smoke test: 404 errors are handled."""
        response = page.request.get(f"{api_base_url}/nonexistent_endpoint_xyz")
        assert response.status in [404, 400]
    
    @pytest.mark.playwright
    @pytest.mark.smoke
    def test_400_handling_smoke(self, page, api_base_url):
        """Smoke test: 400 errors are handled."""
        response = page.request.post(
            f"{api_base_url}/pdf/upload",
            json={"invalid": "data"}
        )
        assert response.status in [400, 422, 415]


class TestPlaywrightSmokePerformance:
    """Performance smoke tests."""
    
    @pytest.mark.playwright
    @pytest.mark.smoke
    def test_response_time_smoke(self, page, api_base_url):
        """Smoke test: Response time is acceptable."""
        times = []
        
        for _ in range(5):
            start = time.time()
            response = page.request.get(f"{api_base_url}/health")
            elapsed = time.time() - start
            times.append(elapsed)
            assert response.status == 200
        
        avg_time = sum(times) / len(times)
        assert avg_time < 2.0, f"Average response time too high: {avg_time:.3f}s"
    
    @pytest.mark.playwright
    @pytest.mark.smoke
    def test_concurrent_requests_smoke(self, page, api_base_url):
        """Smoke test: Concurrent requests work."""
        import concurrent.futures
        
        def make_request():
            return page.request.get(f"{api_base_url}/health")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        assert len(results) == 10
        assert all(r.status == 200 for r in results)



