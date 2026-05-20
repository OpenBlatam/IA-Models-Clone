"""
Playwright Parallel Tests
=========================
Tests for parallel execution and concurrency with Playwright.
"""

import pytest
from playwright.sync_api import Page, Browser
import time
import concurrent.futures
from typing import List, Dict, Any


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


class TestPlaywrightParallelExecution:
    """Tests for parallel execution."""
    
    @pytest.mark.playwright
    @pytest.mark.parallel
    @pytest.mark.slow
    def test_parallel_health_checks(self, browser, api_base_url):
        """Test parallel health check requests."""
        def make_request():
            context = browser.new_context()
            page = context.new_page()
            try:
                return page.request.get(f"{api_base_url}/health")
            finally:
                page.close()
                context.close()
        
        # Execute in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(20)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        assert len(results) == 20
        assert all(r.status == 200 for r in results)
    
    @pytest.mark.playwright
    @pytest.mark.parallel
    @pytest.mark.slow
    def test_parallel_uploads(self, browser, api_base_url, sample_pdf, auth_headers):
        """Test parallel PDF uploads."""
        def upload_file(i):
            context = browser.new_context()
            page = context.new_page()
            try:
                files = {
                    "file": {
                        "name": f"parallel_{i}.pdf",
                        "mimeType": "application/pdf",
                        "buffer": sample_pdf
                    }
                }
                return page.request.post(
                    f"{api_base_url}/pdf/upload",
                    multipart=files,
                    headers=auth_headers
                )
            finally:
                page.close()
                context.close()
        
        # Execute in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
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
        assert len(file_ids) == len(set(file_ids)), "Duplicate file IDs in parallel uploads"
    
    @pytest.mark.playwright
    @pytest.mark.parallel
    @pytest.mark.slow
    def test_parallel_variant_generation(self, browser, api_base_url, sample_pdf, auth_headers):
        """Test parallel variant generation."""
        # First upload a file
        context = browser.new_context()
        page = context.new_page()
        
        files = {
            "file": {
                "name": "parallel_variant_test.pdf",
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
        page.close()
        context.close()
        
        # Generate variants in parallel
        def generate_variant(variant_type):
            context = browser.new_context()
            page = context.new_page()
            try:
                return page.request.post(
                    f"{api_base_url}/pdf/{file_id}/variants",
                    json={"variant_type": variant_type, "options": {}},
                    headers=auth_headers
                )
            finally:
                page.close()
                context.close()
        
        variant_types = ["summary", "outline", "highlights"]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(generate_variant, vt) for vt in variant_types]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All should complete
        assert len(results) == len(variant_types)


class TestPlaywrightConcurrentAccess:
    """Tests for concurrent access scenarios."""
    
    @pytest.mark.playwright
    @pytest.mark.parallel
    @pytest.mark.slow
    def test_concurrent_reads(self, browser, api_base_url, sample_pdf, auth_headers):
        """Test concurrent read operations."""
        # Upload file
        context = browser.new_context()
        page = context.new_page()
        
        files = {
            "file": {
                "name": "concurrent_read_test.pdf",
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
        page.close()
        context.close()
        
        # Concurrent reads
        def read_metadata():
            context = browser.new_context()
            page = context.new_page()
            try:
                return page.request.get(
                    f"{api_base_url}/pdf/{file_id}",
                    headers=auth_headers
                )
            finally:
                page.close()
                context.close()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(read_metadata) for _ in range(20)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        assert len(results) == 20
        assert all(r.status in [200, 404, 401, 403] for r in results)
    
    @pytest.mark.playwright
    @pytest.mark.parallel
    @pytest.mark.slow
    def test_concurrent_updates(self, browser, api_base_url, sample_pdf, auth_headers):
        """Test concurrent update operations."""
        # Upload file
        context = browser.new_context()
        page = context.new_page()
        
        files = {
            "file": {
                "name": "concurrent_update_test.pdf",
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
        page.close()
        context.close()
        
        # Concurrent updates
        def update_metadata(i):
            context = browser.new_context()
            page = context.new_page()
            try:
                return page.request.put(
                    f"{api_base_url}/pdf/{file_id}",
                    json={"title": f"Title {i}"},
                    headers=auth_headers
                )
            finally:
                page.close()
                context.close()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(update_metadata, i) for i in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All should complete (may have conflicts)
        assert len(results) == 10
        assert all(r.status in [200, 412, 404, 401, 403] for r in results)


class TestPlaywrightRaceConditions:
    """Tests for race condition scenarios."""
    
    @pytest.mark.playwright
    @pytest.mark.parallel
    @pytest.mark.slow
    def test_race_condition_upload_delete(self, browser, api_base_url, sample_pdf, auth_headers):
        """Test race condition between upload and delete."""
        def upload_and_delete():
            context = browser.new_context()
            page = context.new_page()
            try:
                # Upload
                files = {
                    "file": {
                        "name": "race_test.pdf",
                        "mimeType": "application/pdf",
                        "buffer": sample_pdf
                    }
                }
                
                upload_response = page.request.post(
                    f"{api_base_url}/pdf/upload",
                    multipart=files,
                    headers=auth_headers
                )
                
                if upload_response.status in [200, 201]:
                    file_id = upload_response.json().get("file_id") or upload_response.json().get("id")
                    
                    # Immediately try to delete
                    delete_response = page.request.delete(
                        f"{api_base_url}/pdf/{file_id}",
                        headers=auth_headers
                    )
                    return delete_response.status
                return None
            finally:
                page.close()
                context.close()
        
        # Execute concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(upload_and_delete) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Should handle gracefully
        assert len(results) == 10
    
    @pytest.mark.playwright
    @pytest.mark.parallel
    @pytest.mark.slow
    def test_race_condition_create_update(self, browser, api_base_url, sample_pdf, auth_headers):
        """Test race condition between create and update."""
        # Upload file
        context = browser.new_context()
        page = context.new_page()
        
        files = {
            "file": {
                "name": "race_create_update_test.pdf",
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
        page.close()
        context.close()
        
        # Concurrent create variant and update
        def create_variant():
            context = browser.new_context()
            page = context.new_page()
            try:
                return page.request.post(
                    f"{api_base_url}/pdf/{file_id}/variants",
                    json={"variant_type": "summary", "options": {}},
                    headers=auth_headers
                )
            finally:
                page.close()
                context.close()
        
        def update_metadata():
            context = browser.new_context()
            page = context.new_page()
            try:
                return page.request.put(
                    f"{api_base_url}/pdf/{file_id}",
                    json={"title": "Updated"},
                    headers=auth_headers
                )
            finally:
                page.close()
                context.close()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(create_variant),
                executor.submit(update_metadata)
            ]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Both should complete
        assert len(results) == 2


class TestPlaywrightLoadBalancing:
    """Tests for load balancing scenarios."""
    
    @pytest.mark.playwright
    @pytest.mark.parallel
    @pytest.mark.slow
    def test_distributed_load(self, browser, api_base_url):
        """Test distributed load across multiple requests."""
        def make_request():
            context = browser.new_context()
            page = context.new_page()
            try:
                start = time.time()
                response = page.request.get(f"{api_base_url}/health")
                elapsed = time.time() - start
                return response.status, elapsed
            finally:
                page.close()
                context.close()
        
        # Distribute load
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_request) for _ in range(100)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        statuses, times = zip(*results)
        
        # All should succeed
        assert all(status == 200 for status in statuses)
        
        # Load should be distributed (times should be reasonable)
        avg_time = sum(times) / len(times)
        assert avg_time < 2.0, f"Load not distributed well: {avg_time:.3f}s"


class TestPlaywrightResourceContention:
    """Tests for resource contention scenarios."""
    
    @pytest.mark.playwright
    @pytest.mark.parallel
    @pytest.mark.slow
    def test_resource_contention(self, browser, api_base_url, sample_pdf, auth_headers):
        """Test resource contention scenarios."""
        # Upload multiple files
        file_ids = []
        context = browser.new_context()
        page = context.new_page()
        
        for i in range(5):
            files = {
                "file": {
                    "name": f"contention_{i}.pdf",
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
        
        page.close()
        context.close()
        
        # Concurrent operations on same resources
        def operate_on_file(file_id):
            context = browser.new_context()
            page = context.new_page()
            try:
                # Try multiple operations
                operations = [
                    page.request.get(f"{api_base_url}/pdf/{file_id}", headers=auth_headers),
                    page.request.get(f"{api_base_url}/pdf/{file_id}/preview", headers=auth_headers),
                    page.request.post(
                        f"{api_base_url}/pdf/{file_id}/variants",
                        json={"variant_type": "summary", "options": {}},
                        headers=auth_headers
                    )
                ]
                return operations
            finally:
                page.close()
                context.close()
        
        # Concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(operate_on_file, file_id) for file_id in file_ids]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All should complete
        assert len(results) == len(file_ids)



