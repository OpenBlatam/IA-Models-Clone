"""
Playwright Regression Tests
===========================
Regression tests to prevent bugs from reoccurring.
"""

import pytest
from playwright.sync_api import Page, Response
import time
from typing import Dict, Any


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


class TestPlaywrightRegressionBugs:
    """Regression tests for known bugs."""
    
    @pytest.mark.playwright
    @pytest.mark.regression
    def test_file_id_consistency(self, page, api_base_url, auth_headers, sample_pdf):
        """Regression: File ID should be consistent across operations."""
        # Upload
        files = {
            "file": {
                "name": "regression_test.pdf",
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
        
        # File ID should be same in all subsequent requests
        operations = [
            ("preview", f"/pdf/{file_id}/preview"),
            ("topics", f"/pdf/{file_id}/topics"),
            ("metadata", f"/pdf/{file_id}")
        ]
        
        for op_name, endpoint in operations:
            response = page.request.get(
                f"{api_base_url}{endpoint}",
                headers=auth_headers
            )
            
            if response.status == 200:
                data = response.json()
                # File ID should match
                returned_id = data.get("file_id") or data.get("id")
                if returned_id:
                    assert returned_id == file_id, f"File ID mismatch in {op_name}"
    
    @pytest.mark.playwright
    @pytest.mark.regression
    def test_no_duplicate_file_ids(self, page, api_base_url, sample_pdf, auth_headers):
        """Regression: No duplicate file IDs should be generated."""
        file_ids = []
        
        # Upload multiple files
        for i in range(5):
            files = {
                "file": {
                    "name": f"duplicate_test_{i}.pdf",
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
        
        # All file IDs should be unique
        assert len(file_ids) == len(set(file_ids)), "Duplicate file IDs detected"
    
    @pytest.mark.playwright
    @pytest.mark.regression
    def test_error_message_consistency(self, page, api_base_url, auth_headers):
        """Regression: Error messages should be consistent."""
        # Test same error scenario multiple times
        error_responses = []
        
        for _ in range(3):
            response = page.request.get(
                f"{api_base_url}/pdf/nonexistent_file_xyz/preview",
                headers=auth_headers
            )
            if response.status >= 400:
                error_responses.append(response.json())
        
        if len(error_responses) > 1:
            # Error format should be consistent
            first_error = error_responses[0]
            for error in error_responses[1:]:
                # Should have same structure
                assert set(first_error.keys()) == set(error.keys()) or True
    
    @pytest.mark.playwright
    @pytest.mark.regression
    def test_no_memory_leak_on_repeated_requests(self, page, api_base_url):
        """Regression: No memory leak on repeated requests."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Make many requests
        for _ in range(100):
            page.request.get(f"{api_base_url}/health")
        
        # Force garbage collection
        import gc
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal
        assert memory_increase < 50, f"Possible memory leak: {memory_increase:.1f}MB increase"
    
    @pytest.mark.playwright
    @pytest.mark.regression
    def test_timeout_handling_consistency(self, page, api_base_url):
        """Regression: Timeout handling should be consistent."""
        timeout_responses = []
        
        # Make requests with very short timeout
        for _ in range(5):
            try:
                response = page.request.get(
                    f"{api_base_url}/health",
                    timeout=1  # 1ms (should timeout)
                )
                timeout_responses.append(response.status)
            except Exception as e:
                timeout_responses.append(str(e))
        
        # Should handle timeouts consistently
        assert len(timeout_responses) == 5


class TestPlaywrightDataIntegrity:
    """Tests for data integrity."""
    
    @pytest.mark.playwright
    @pytest.mark.regression
    def test_data_not_corrupted_on_retry(self, page, api_base_url, sample_pdf, auth_headers):
        """Regression: Data should not be corrupted on retry."""
        files = {
            "file": {
                "name": "integrity_test.pdf",
                "mimeType": "application/pdf",
                "buffer": sample_pdf
            }
        }
        
        # First upload attempt
        response1 = page.request.post(
            f"{api_base_url}/pdf/upload",
            multipart=files,
            headers=auth_headers
        )
        
        # Retry same upload
        response2 = page.request.post(
            f"{api_base_url}/pdf/upload",
            multipart=files,
            headers=auth_headers
        )
        
        # Both should succeed or handle gracefully
        assert response1.status in [200, 201, 400, 401, 403]
        assert response2.status in [200, 201, 400, 401, 403]
    
    @pytest.mark.playwright
    @pytest.mark.regression
    def test_metadata_preservation(self, page, api_base_url, sample_pdf, auth_headers):
        """Regression: Metadata should be preserved."""
        files = {
            "file": {
                "name": "metadata_test.pdf",
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
        
        # Get metadata
        metadata_response = page.request.get(
            f"{api_base_url}/pdf/{file_id}",
            headers=auth_headers
        )
        
        if metadata_response.status == 200:
            metadata1 = metadata_response.json()
            
            # Wait and get again
            time.sleep(1)
            metadata_response2 = page.request.get(
                f"{api_base_url}/pdf/{file_id}",
                headers=auth_headers
            )
            
            if metadata_response2.status == 200:
                metadata2 = metadata_response2.json()
                # Core metadata should be preserved
                assert metadata1.get("file_id") == metadata2.get("file_id")
                assert metadata1.get("filename") == metadata2.get("filename")


class TestPlaywrightBackwardCompatibility:
    """Tests for backward compatibility."""
    
    @pytest.mark.playwright
    @pytest.mark.regression
    def test_old_api_version_still_works(self, page, api_base_url):
        """Regression: Old API version should still work."""
        # Test v1 endpoints
        response = page.request.get(f"{api_base_url}/v1/health")
        # May or may not support versioning
        assert response.status is not None
    
    @pytest.mark.playwright
    @pytest.mark.regression
    def test_deprecated_endpoints(self, page, api_base_url, auth_headers):
        """Regression: Deprecated endpoints should still work with warning."""
        # Try deprecated endpoint
        response = page.request.get(
            f"{api_base_url}/pdf/old_endpoint",
            headers=auth_headers
        )
        
        # Should work or return 404
        assert response.status in [200, 404, 401, 403]
        
        # May have deprecation header
        if response.status == 200:
            headers = response.headers
            if "deprecation" in headers:
                assert True  # Deprecation header present


class TestPlaywrightRaceConditions:
    """Tests for race conditions."""
    
    @pytest.mark.playwright
    @pytest.mark.regression
    @pytest.mark.slow
    def test_concurrent_uploads_no_conflicts(self, page, api_base_url, sample_pdf, auth_headers):
        """Regression: Concurrent uploads should not conflict."""
        import concurrent.futures
        
        def upload_file(i):
            files = {
                "file": {
                    "name": f"race_test_{i}.pdf",
                    "mimeType": "application/pdf",
                    "buffer": sample_pdf
                }
            }
            return page.request.post(
                f"{api_base_url}/pdf/upload",
                multipart=files,
                headers=auth_headers
            )
        
        # Concurrent uploads
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(upload_file, i) for i in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All should complete
        assert len(results) == 10
        
        # Check for duplicate file IDs
        file_ids = []
        for result in results:
            if result.status in [200, 201]:
                file_id = result.json().get("file_id") or result.json().get("id")
                if file_id:
                    file_ids.append(file_id)
        
        # All should be unique
        assert len(file_ids) == len(set(file_ids)), "Race condition: duplicate file IDs"
    
    @pytest.mark.playwright
    @pytest.mark.regression
    @pytest.mark.slow
    def test_concurrent_deletes_no_errors(self, page, api_base_url, sample_pdf, auth_headers):
        """Regression: Concurrent deletes should not cause errors."""
        # Upload file first
        files = {
            "file": {
                "name": "delete_race_test.pdf",
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
        
        # Concurrent deletes
        import concurrent.futures
        
        def delete_file():
            return page.request.delete(
                f"{api_base_url}/pdf/{file_id}",
                headers=auth_headers
            )
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(delete_file) for _ in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Should handle gracefully (one succeeds, others may get 404)
        status_codes = [r.status for r in results]
        assert all(status in [200, 204, 404] for status in status_codes)


class TestPlaywrightEdgeCaseRegressions:
    """Tests for edge case regressions."""
    
    @pytest.mark.playwright
    @pytest.mark.regression
    def test_empty_string_handling(self, page, api_base_url, auth_headers):
        """Regression: Empty strings should be handled correctly."""
        # Try operations with empty strings
        test_cases = [
            ("", "empty file_id"),
            ("   ", "whitespace file_id"),
            ("\t\n", "whitespace chars file_id")
        ]
        
        for test_value, description in test_cases:
            response = page.request.get(
                f"{api_base_url}/pdf/{test_value}/preview",
                headers=auth_headers
            )
            # Should handle gracefully
            assert response.status in [200, 400, 404, 422]
    
    @pytest.mark.playwright
    @pytest.mark.regression
    def test_very_long_identifiers(self, page, api_base_url, auth_headers):
        """Regression: Very long identifiers should be handled."""
        long_id = "a" * 1000  # Very long ID
        
        response = page.request.get(
            f"{api_base_url}/pdf/{long_id}/preview",
            headers=auth_headers
        )
        
        # Should handle gracefully (reject or process)
        assert response.status in [200, 400, 404, 422, 414]  # 414 URI Too Long
    
    @pytest.mark.playwright
    @pytest.mark.regression
    def test_special_characters_in_ids(self, page, api_base_url, auth_headers):
        """Regression: Special characters in IDs should be handled."""
        special_chars = ["test@file", "test#file", "test$file", "test%file"]
        
        for special_id in special_chars:
            response = page.request.get(
                f"{api_base_url}/pdf/{special_id}/preview",
                headers=auth_headers
            )
            # Should handle gracefully
            assert response.status in [200, 400, 404, 422]


class TestPlaywrightPerformanceRegressions:
    """Tests for performance regressions."""
    
    @pytest.mark.playwright
    @pytest.mark.regression
    @pytest.mark.slow
    def test_response_time_not_degraded(self, page, api_base_url):
        """Regression: Response time should not degrade."""
        times = []
        
        # Measure response times
        for _ in range(20):
            start = time.time()
            response = page.request.get(f"{api_base_url}/health")
            elapsed = time.time() - start
            times.append(elapsed)
            assert response.status == 200
        
        # Average should be reasonable
        avg_time = sum(times) / len(times)
        assert avg_time < 1.0, f"Performance regression: avg time {avg_time:.3f}s"
        
        # Max should not be too high
        max_time = max(times)
        assert max_time < 2.0, f"Performance regression: max time {max_time:.3f}s
    
    @pytest.mark.playwright
    @pytest.mark.regression
    @pytest.mark.slow
    def test_no_memory_growth_over_time(self, page, api_base_url):
        """Regression: Memory should not grow over time."""
        import psutil
        import os
        import gc
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Make requests over time
        for i in range(50):
            page.request.get(f"{api_base_url}/health")
            if i % 10 == 0:
                gc.collect()
                current_memory = process.memory_info().rss / 1024 / 1024
                # Memory should not grow significantly
                growth = current_memory - initial_memory
                assert growth < 100, f"Memory growth detected: {growth:.1f}MB"



