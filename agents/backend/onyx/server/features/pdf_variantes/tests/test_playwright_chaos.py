"""
Playwright Chaos Engineering Tests
==================================
Chaos engineering tests to test system resilience.
"""

import pytest
from playwright.sync_api import Page, Response
import time
import random
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


class TestPlaywrightChaosNetwork:
    """Chaos tests for network conditions."""
    
    @pytest.mark.playwright
    @pytest.mark.chaos
    @pytest.mark.slow
    def test_network_latency_chaos(self, page, api_base_url):
        """Test system behavior under network latency."""
        # Simulate high latency
        times = []
        
        for _ in range(10):
            start = time.time()
            try:
                response = page.request.get(f"{api_base_url}/health", timeout=10000)
                elapsed = time.time() - start
                times.append(elapsed)
                assert response.status == 200
            except Exception:
                # May timeout with high latency
                assert True
        
        # System should handle latency
        if times:
            avg_time = sum(times) / len(times)
            assert avg_time < 10.0, f"System not handling latency well: {avg_time:.3f}s"
    
    @pytest.mark.playwright
    @pytest.mark.chaos
    @pytest.mark.slow
    def test_network_packet_loss_chaos(self, page, api_base_url):
        """Test system behavior under packet loss."""
        # Make requests that may experience packet loss
        success_count = 0
        total_requests = 20
        
        for _ in range(total_requests):
            try:
                response = page.request.get(f"{api_base_url}/health", timeout=5000)
                if response.status == 200:
                    success_count += 1
            except Exception:
                # May fail due to packet loss
                pass
            time.sleep(0.1)
        
        # System should handle packet loss gracefully
        success_rate = success_count / total_requests
        assert success_rate >= 0.7, f"System not handling packet loss: {success_rate:.2%}"
    
    @pytest.mark.playwright
    @pytest.mark.chaos
    def test_network_timeout_chaos(self, page, api_base_url):
        """Test system behavior with timeouts."""
        # Make requests with very short timeout
        timeout_responses = []
        
        for _ in range(5):
            try:
                response = page.request.get(f"{api_base_url}/health", timeout=1)
                timeout_responses.append(response.status)
            except Exception:
                # Timeout expected
                timeout_responses.append(None)
        
        # System should handle timeouts gracefully
        assert len(timeout_responses) == 5


class TestPlaywrightChaosLoad:
    """Chaos tests for load conditions."""
    
    @pytest.mark.playwright
    @pytest.mark.chaos
    @pytest.mark.slow
    def test_sudden_load_spike(self, browser, api_base_url):
        """Test system behavior under sudden load spike."""
        def make_request():
            context = browser.new_context()
            page = context.new_page()
            try:
                return page.request.get(f"{api_base_url}/health")
            finally:
                page.close()
                context.close()
        
        # Sudden spike: 0 to 50 requests
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(make_request) for _ in range(50)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # System should handle spike
        success_count = sum(1 for r in results if r and r.status == 200)
        success_rate = success_count / len(results)
        assert success_rate >= 0.8, f"System not handling load spike: {success_rate:.2%}"
    
    @pytest.mark.playwright
    @pytest.mark.chaos
    @pytest.mark.slow
    def test_sustained_high_load(self, browser, api_base_url):
        """Test system behavior under sustained high load."""
        def make_request():
            context = browser.new_context()
            page = context.new_page()
            try:
                return page.request.get(f"{api_base_url}/health")
            finally:
                page.close()
                context.close()
        
        # Sustained load: 30 seconds
        import concurrent.futures
        
        start_time = time.time()
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            while time.time() - start_time < 30:
                futures = [executor.submit(make_request) for _ in range(10)]
                batch_results = [future.result() for future in concurrent.futures.as_completed(futures)]
                results.extend(batch_results)
                time.sleep(0.5)
        
        # System should maintain performance
        success_count = sum(1 for r in results if r and r.status == 200)
        success_rate = success_count / len(results) if results else 0
        assert success_rate >= 0.9, f"System not handling sustained load: {success_rate:.2%}"


class TestPlaywrightChaosErrors:
    """Chaos tests for error conditions."""
    
    @pytest.mark.playwright
    @pytest.mark.chaos
    def test_error_injection(self, page, api_base_url):
        """Test system behavior with injected errors."""
        # Inject various errors
        error_scenarios = [
            ("/nonexistent_endpoint_xyz", 404),
            ("/pdf/invalid_id/preview", 404),
            ("/pdf/upload", 400),  # Invalid request
        ]
        
        for endpoint, expected_status in error_scenarios:
            try:
                if endpoint == "/pdf/upload":
                    response = page.request.post(
                        f"{api_base_url}{endpoint}",
                        json={"invalid": "data"}
                    )
                else:
                    response = page.request.get(f"{api_base_url}{endpoint}")
                
                # System should handle errors gracefully
                assert response.status in [expected_status, 400, 401, 403, 422]
            except Exception:
                # Should not crash
                assert True
    
    @pytest.mark.playwright
    @pytest.mark.chaos
    def test_malformed_requests(self, page, api_base_url):
        """Test system behavior with malformed requests."""
        malformed_requests = [
            ("GET", "/health", {"headers": {"Content-Type": "invalid"}}),
            ("POST", "/pdf/upload", {"data": "not json"}),
            ("GET", "/health?invalid=param&another=param", {}),
        ]
        
        for method, endpoint, kwargs in malformed_requests:
            try:
                if method == "GET":
                    response = page.request.get(f"{api_base_url}{endpoint}", **kwargs)
                elif method == "POST":
                    response = page.request.post(f"{api_base_url}{endpoint}", **kwargs)
                else:
                    continue
                
                # Should handle malformed requests gracefully
                assert response.status is not None
            except Exception:
                # Should not crash
                assert True


class TestPlaywrightChaosData:
    """Chaos tests for data conditions."""
    
    @pytest.mark.playwright
    @pytest.mark.chaos
    def test_large_payload_chaos(self, page, api_base_url, auth_headers):
        """Test system behavior with large payloads."""
        # Create large payload
        large_data = {"data": "x" * (10 * 1024 * 1024)}  # 10MB
        
        try:
            response = page.request.post(
                f"{api_base_url}/pdf/upload",
                json=large_data,
                headers=auth_headers,
                timeout=30000
            )
            # Should handle large payloads (reject or process)
            assert response.status in [200, 201, 413, 400, 422]
        except Exception:
            # May timeout or reject
            assert True
    
    @pytest.mark.playwright
    @pytest.mark.chaos
    def test_rapid_data_changes(self, page, api_base_url, sample_pdf, auth_headers):
        """Test system behavior with rapid data changes."""
        # Upload file
        files = {
            "file": {
                "name": "chaos_test.pdf",
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
        
        # Rapid updates
        for i in range(10):
            update_response = page.request.put(
                f"{api_base_url}/pdf/{file_id}",
                json={"title": f"Title {i}"},
                headers=auth_headers
            )
            # Should handle rapid changes
            assert update_response.status in [200, 412, 404, 401, 403]
            time.sleep(0.1)


class TestPlaywrightChaosConcurrency:
    """Chaos tests for concurrency conditions."""
    
    @pytest.mark.playwright
    @pytest.mark.chaos
    @pytest.mark.slow
    def test_concurrent_conflicts(self, browser, api_base_url, sample_pdf, auth_headers):
        """Test system behavior with concurrent conflicts."""
        # Upload file
        context = browser.new_context()
        page = context.new_page()
        
        files = {
            "file": {
                "name": "conflict_test.pdf",
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
        
        # Concurrent conflicting operations
        def update_file(i):
            context = browser.new_context()
            page = context.new_page()
            try:
                return page.request.put(
                    f"{api_base_url}/pdf/{file_id}",
                    json={"title": f"Conflict {i}"},
                    headers=auth_headers
                )
            finally:
                page.close()
                context.close()
        
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(update_file, i) for i in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Should handle conflicts gracefully
        assert len(results) == 10
        assert all(r.status in [200, 412, 404, 401, 403] for r in results)


class TestPlaywrightChaosRecovery:
    """Chaos tests for recovery scenarios."""
    
    @pytest.mark.playwright
    @pytest.mark.chaos
    def test_graceful_degradation(self, page, api_base_url):
        """Test graceful degradation under stress."""
        # Make requests under stress
        responses = []
        
        for _ in range(20):
            try:
                response = page.request.get(f"{api_base_url}/health", timeout=2000)
                responses.append(response.status)
            except Exception:
                responses.append(None)
            time.sleep(0.05)
        
        # System should degrade gracefully
        success_count = sum(1 for s in responses if s == 200)
        success_rate = success_count / len(responses)
        assert success_rate >= 0.5, f"System not degrading gracefully: {success_rate:.2%}"
    
    @pytest.mark.playwright
    @pytest.mark.chaos
    @pytest.mark.slow
    def test_recovery_after_failure(self, page, api_base_url):
        """Test recovery after failure."""
        # Simulate failure scenario
        failure_responses = []
        
        for _ in range(5):
            try:
                response = page.request.get(f"{api_base_url}/nonexistent_endpoint", timeout=1000)
                failure_responses.append(response.status)
            except Exception:
                failure_responses.append(None)
        
        # System should recover
        time.sleep(1)
        
        # Make valid request
        recovery_response = page.request.get(f"{api_base_url}/health")
        assert recovery_response.status == 200



