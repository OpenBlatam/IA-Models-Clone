from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import pytest
import asyncio
import time
from httpx import AsyncClient
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
import json
from api_examples import app, ScanService, VulnerabilityService, SecurityService
    import sys
    import subprocess
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
FastAPI Security Tooling API Testing Examples
Comprehensive testing examples for API-driven security tooling.
"""


# Import the API app

# ============================================================================
# TEST CONFIGURATION
# ============================================================================

@pytest.fixture
def test_client():
    """Create test client."""
    return TestClient(app)

@pytest.fixture
async def async_client():
    """Create async test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def mock_token():
    """Create mock JWT token."""
    return "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyMTIzIiwiaWF0IjoxNjE2MjM5MDIyfQ.test_signature"

@pytest.fixture
def valid_scan_request():
    """Create valid scan request."""
    return {
        "targets": ["127.0.0.1", "192.168.1.1"],
        "ports": [80, 443, 22],
        "scan_type": "tcp",
        "timeout": 5.0,
        "banner_grab": True,
        "ssl_check": True
    }

@pytest.fixture
def invalid_scan_request():
    """Create invalid scan request."""
    return {
        "targets": ["invalid-target"],
        "ports": [99999],  # Invalid port
        "scan_type": "invalid",
        "timeout": -1.0
    }

# ============================================================================
# UNIT TESTS
# ============================================================================

class TestScanService:
    """Test ScanService class."""
    
    @pytest.mark.asyncio
    async def test_create_scan_success(self) -> Any:
        """Test successful scan creation."""
        service = ScanService()
        scan_request = {
            "targets": ["127.0.0.1"],
            "ports": [80, 443],
            "scan_type": "tcp",
            "timeout": 5.0,
            "banner_grab": True,
            "ssl_check": True
        }
        
        scan_id = await service.create_scan(scan_request, "user123")
        
        assert scan_id is not None
        assert len(scan_id) > 0
        
        scan_data = service.get_scan(scan_id)
        assert scan_data is not None
        assert scan_data["user_id"] == "user123"
        assert scan_data["status"] == "pending"
        assert scan_data["progress"] == 0.0
    
    @pytest.mark.asyncio
    async async def test_create_scan_invalid_request(self) -> Any:
        """Test scan creation with invalid request."""
        service = ScanService()
        
        with pytest.raises(Exception):
            await service.create_scan(None, "user123")
    
    @pytest.mark.asyncio
    async def test_get_scan_not_found(self) -> Optional[Dict[str, Any]]:
        """Test getting non-existent scan."""
        service = ScanService()
        
        scan_data = service.get_scan("non-existent")
        assert scan_data is None
    
    @pytest.mark.asyncio
    async def test_get_user_scans(self) -> Optional[Dict[str, Any]]:
        """Test getting user scans."""
        service = ScanService()
        
        # Create multiple scans for user
        scan_request = {
            "targets": ["127.0.0.1"],
            "ports": [80],
            "scan_type": "tcp",
            "timeout": 5.0
        }
        
        await service.create_scan(scan_request, "user123")
        await service.create_scan(scan_request, "user123")
        await service.create_scan(scan_request, "user456")  # Different user
        
        user_scans = service.get_user_scans("user123")
        assert len(user_scans) == 2
        
        user_scans = service.get_user_scans("user456")
        assert len(user_scans) == 1
    
    @pytest.mark.asyncio
    async def test_scan_execution(self) -> Any:
        """Test scan execution."""
        service = ScanService()
        scan_request = {
            "targets": ["127.0.0.1"],
            "ports": [80, 443],
            "scan_type": "tcp",
            "timeout": 5.0
        }
        
        scan_id = await service.create_scan(scan_request, "user123")
        
        # Start scan
        await service.start_scan(scan_id)
        
        # Wait for completion
        for _ in range(10):  # Wait up to 10 seconds
            scan_data = service.get_scan(scan_id)
            if scan_data["status"] == "completed":
                break
            await asyncio.sleep(1)
        
        scan_data = service.get_scan(scan_id)
        assert scan_data["status"] == "completed"
        assert scan_data["progress"] == 100.0
        assert scan_data["results"] is not None

class TestVulnerabilityService:
    """Test VulnerabilityService class."""
    
    def test_analyze_scan_results(self) -> Any:
        """Test vulnerability analysis."""
        service = VulnerabilityService()
        
        scan_results = {
            "127.0.0.1": [
                {
                    "target": "127.0.0.1",
                    "port": 22,
                    "is_open": True,
                    "service_name": "ssh",
                    "banner": "SSH-2.0-OpenSSH_7.0",
                    "response_time": 0.1,
                    "protocol": "tcp"
                },
                {
                    "target": "127.0.0.1",
                    "port": 23,
                    "is_open": True,
                    "service_name": "telnet",
                    "banner": "Telnet service",
                    "response_time": 0.2,
                    "protocol": "tcp"
                }
            ]
        }
        
        report = service.analyze_scan_results(scan_results)
        
        assert report.scan_id is not None
        assert len(report.vulnerabilities) > 0
        assert report.risk_score > 0
        assert report.summary is not None
        assert len(report.recommendations) > 0
    
    def test_analyze_scan_results_no_vulnerabilities(self) -> Any:
        """Test analysis with no vulnerabilities."""
        service = VulnerabilityService()
        
        scan_results = {
            "127.0.0.1": [
                {
                    "target": "127.0.0.1",
                    "port": 80,
                    "is_open": False,
                    "service_name": None,
                    "banner": None,
                    "response_time": 0.1,
                    "protocol": "tcp"
                }
            ]
        }
        
        report = service.analyze_scan_results(scan_results)
        
        assert report.risk_score == 0.0
        assert len(report.vulnerabilities) == 0

class TestSecurityService:
    """Test SecurityService class."""
    
    def test_verify_token_valid(self) -> Any:
        """Test token verification with valid token."""
        service = SecurityService()
        
        # Mock JWT decode
        with patch('jwt.decode') as mock_decode:
            mock_decode.return_value = {"sub": "user123"}
            
            payload = service.verify_token("valid_token")
            assert payload["sub"] == "user123"
    
    def test_verify_token_invalid(self) -> Any:
        """Test token verification with invalid token."""
        service = SecurityService()
        
        with patch('jwt.decode') as mock_decode:
            mock_decode.side_effect = Exception("Invalid token")
            
            with pytest.raises(HTTPException) as exc_info:
                service.verify_token("invalid_token")
            
            assert exc_info.value.status_code == 401

# ============================================================================
# API INTEGRATION TESTS
# ============================================================================

class TestScanAPI:
    """Test scan API endpoints."""
    
    def test_create_scan_success(self, test_client, mock_token, valid_scan_request) -> Any:
        """Test successful scan creation via API."""
        headers = {"Authorization": f"Bearer {mock_token}"}
        
        with patch('api_examples.SecurityService.verify_token') as mock_verify:
            mock_verify.return_value = {"sub": "user123"}
            
            response = test_client.post(
                "/api/v1/scans",
                json=valid_scan_request,
                headers=headers
            )
            
            assert response.status_code == 201
            data = response.json()
            assert "scan_id" in data
            assert data["status"] == "pending"
            assert data["progress"] == 0.0
    
    async def test_create_scan_invalid_request(self, test_client, mock_token, invalid_scan_request) -> Any:
        """Test scan creation with invalid request via API."""
        headers = {"Authorization": f"Bearer {mock_token}"}
        
        with patch('api_examples.SecurityService.verify_token') as mock_verify:
            mock_verify.return_value = {"sub": "user123"}
            
            response = test_client.post(
                "/api/v1/scans",
                json=invalid_scan_request,
                headers=headers
            )
            
            assert response.status_code == 422
    
    def test_create_scan_unauthorized(self, test_client, valid_scan_request) -> Any:
        """Test scan creation without authentication."""
        response = test_client.post("/api/v1/scans", json=valid_scan_request)
        
        assert response.status_code == 401
    
    def test_get_scan_success(self, test_client, mock_token) -> Optional[Dict[str, Any]]:
        """Test getting scan via API."""
        headers = {"Authorization": f"Bearer {mock_token}"}
        
        with patch('api_examples.SecurityService.verify_token') as mock_verify:
            mock_verify.return_value = {"sub": "user123"}
            
            # First create a scan
            scan_request = {
                "targets": ["127.0.0.1"],
                "ports": [80],
                "scan_type": "tcp",
                "timeout": 5.0
            }
            
            create_response = test_client.post(
                "/api/v1/scans",
                json=scan_request,
                headers=headers
            )
            
            scan_id = create_response.json()["scan_id"]
            
            # Then get the scan
            response = test_client.get(f"/api/v1/scans/{scan_id}", headers=headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["scan_id"] == scan_id
            assert data["status"] == "pending"
    
    def test_get_scan_not_found(self, test_client, mock_token) -> Optional[Dict[str, Any]]:
        """Test getting non-existent scan via API."""
        headers = {"Authorization": f"Bearer {mock_token}"}
        
        with patch('api_examples.SecurityService.verify_token') as mock_verify:
            mock_verify.return_value = {"sub": "user123"}
            
            response = test_client.get("/api/v1/scans/non-existent", headers=headers)
            
            assert response.status_code == 404
    
    def test_list_scans(self, test_client, mock_token) -> List[Any]:
        """Test listing scans via API."""
        headers = {"Authorization": f"Bearer {mock_token}"}
        
        with patch('api_examples.SecurityService.verify_token') as mock_verify:
            mock_verify.return_value = {"sub": "user123"}
            
            # Create some scans
            scan_request = {
                "targets": ["127.0.0.1"],
                "ports": [80],
                "scan_type": "tcp",
                "timeout": 5.0
            }
            
            for _ in range(3):
                test_client.post("/api/v1/scans", json=scan_request, headers=headers)
            
            # List scans
            response = test_client.get("/api/v1/scans", headers=headers)
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 3
    
    def test_cancel_scan(self, test_client, mock_token) -> Any:
        """Test cancelling scan via API."""
        headers = {"Authorization": f"Bearer {mock_token}"}
        
        with patch('api_examples.SecurityService.verify_token') as mock_verify:
            mock_verify.return_value = {"sub": "user123"}
            
            # Create a scan
            scan_request = {
                "targets": ["127.0.0.1"],
                "ports": [80],
                "scan_type": "tcp",
                "timeout": 5.0
            }
            
            create_response = test_client.post(
                "/api/v1/scans",
                json=scan_request,
                headers=headers
            )
            
            scan_id = create_response.json()["scan_id"]
            
            # Cancel the scan
            response = test_client.delete(f"/api/v1/scans/{scan_id}", headers=headers)
            
            assert response.status_code == 200
            assert response.json()["message"] == "Scan cancelled successfully"

class TestVulnerabilityAPI:
    """Test vulnerability API endpoints."""
    
    def test_generate_vulnerability_report(self, test_client, mock_token) -> Any:
        """Test generating vulnerability report via API."""
        headers = {"Authorization": f"Bearer {mock_token}"}
        
        with patch('api_examples.SecurityService.verify_token') as mock_verify:
            mock_verify.return_value = {"sub": "user123"}
            
            # Create and complete a scan
            scan_request = {
                "targets": ["127.0.0.1"],
                "ports": [22, 23],  # SSH and Telnet for vulnerabilities
                "scan_type": "tcp",
                "timeout": 5.0
            }
            
            create_response = test_client.post(
                "/api/v1/scans",
                json=scan_request,
                headers=headers
            )
            
            scan_id = create_response.json()["scan_id"]
            
            # Wait for scan completion (simulate)
            # In real test, you'd wait for the scan to complete
            
            # Generate report
            response = test_client.post(
                f"/api/v1/scans/{scan_id}/vulnerability-report",
                headers=headers
            )
            
            # Should fail because scan is not completed
            assert response.status_code == 400

class TestHealthAPI:
    """Test health check endpoints."""
    
    def test_health_check(self, test_client) -> Any:
        """Test health check endpoint."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "1.0.0"
    
    def test_readiness_check(self, test_client) -> Any:
        """Test readiness check endpoint."""
        response = test_client.get("/ready")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert "services" in data

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Test API performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_concurrent_scan_creation(self, async_client, mock_token, valid_scan_request) -> Any:
        """Test concurrent scan creation."""
        headers = {"Authorization": f"Bearer {mock_token}"}
        
        with patch('api_examples.SecurityService.verify_token') as mock_verify:
            mock_verify.return_value = {"sub": "user123"}
            
            start_time = time.time()
            
            # Create 10 concurrent scans
            tasks = [
                async_client.post("/api/v1/scans", json=valid_scan_request, headers=headers)
                for _ in range(10)
            ]
            
            responses = await asyncio.gather(*tasks)
            
            duration = time.time() - start_time
            
            # All should succeed
            assert all(r.status_code == 201 for r in responses)
            
            # Should complete within reasonable time
            assert duration < 5.0
    
    @pytest.mark.asyncio
    async async def test_large_scan_request(self, async_client, mock_token) -> Any:
        """Test large scan request handling."""
        headers = {"Authorization": f"Bearer {mock_token}"}
        
        large_scan_request = {
            "targets": [f"192.168.1.{i}" for i in range(1, 51)],  # 50 targets
            "ports": list(range(1, 101)),  # 100 ports
            "scan_type": "tcp",
            "timeout": 10.0
        }
        
        with patch('api_examples.SecurityService.verify_token') as mock_verify:
            mock_verify.return_value = {"sub": "user123"}
            
            start_time = time.time()
            
            response = await async_client.post(
                "/api/v1/scans",
                json=large_scan_request,
                headers=headers
            )
            
            duration = time.time() - start_time
            
            assert response.status_code == 201
            assert duration < 10.0  # Should handle large requests efficiently

# ============================================================================
# SECURITY TESTS
# ============================================================================

class TestSecurity:
    """Test security aspects of the API."""
    
    def test_authentication_required(self, test_client, valid_scan_request) -> Any:
        """Test that authentication is required for protected endpoints."""
        # Test without token
        response = test_client.post("/api/v1/scans", json=valid_scan_request)
        assert response.status_code == 401
        
        response = test_client.get("/api/v1/scans")
        assert response.status_code == 401
        
        response = test_client.get("/api/v1/scans/test-id")
        assert response.status_code == 401
    
    def test_invalid_token(self, test_client, valid_scan_request) -> Any:
        """Test API with invalid token."""
        headers = {"Authorization": "Bearer invalid_token"}
        
        with patch('api_examples.SecurityService.verify_token') as mock_verify:
            mock_verify.side_effect = HTTPException(status_code=401, detail="Invalid token")
            
            response = test_client.post("/api/v1/scans", json=valid_scan_request, headers=headers)
            assert response.status_code == 401
    
    def test_rate_limiting(self, test_client, mock_token, valid_scan_request) -> Any:
        """Test rate limiting functionality."""
        headers = {"Authorization": f"Bearer {mock_token}"}
        
        with patch('api_examples.SecurityService.verify_token') as mock_verify:
            mock_verify.return_value = {"sub": "user123"}
            
            # Make many requests quickly
            responses = []
            for _ in range(15):  # Exceed rate limit
                response = test_client.post("/api/v1/scans", json=valid_scan_request, headers=headers)
                responses.append(response)
            
            # Some should be rate limited
            rate_limited = [r for r in responses if r.status_code == 429]
            assert len(rate_limited) > 0

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling in the API."""
    
    def test_validation_error(self, test_client, mock_token) -> Any:
        """Test validation error handling."""
        headers = {"Authorization": f"Bearer {mock_token}"}
        
        with patch('api_examples.SecurityService.verify_token') as mock_verify:
            mock_verify.return_value = {"sub": "user123"}
            
            invalid_request = {
                "targets": [],  # Empty targets
                "ports": [99999],  # Invalid port
                "scan_type": "invalid_type",
                "timeout": -1.0
            }
            
            response = test_client.post("/api/v1/scans", json=invalid_request, headers=headers)
            
            assert response.status_code == 422
            data = response.json()
            assert "detail" in data
    
    def test_internal_server_error(self, test_client, mock_token, valid_scan_request) -> Any:
        """Test internal server error handling."""
        headers = {"Authorization": f"Bearer {mock_token}"}
        
        with patch('api_examples.SecurityService.verify_token') as mock_verify:
            mock_verify.return_value = {"sub": "user123"}
            
            with patch('api_examples.ScanService.create_scan') as mock_create:
                mock_create.side_effect = Exception("Database error")
                
                response = test_client.post("/api/v1/scans", json=valid_scan_request, headers=headers)
                
                assert response.status_code == 500
                data = response.json()
                assert "error" in data
                assert data["error"] == "Internal server error"

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_scan_workflow(self, async_client, mock_token) -> Any:
        """Test complete scan workflow from creation to results."""
        headers = {"Authorization": f"Bearer {mock_token}"}
        
        with patch('api_examples.SecurityService.verify_token') as mock_verify:
            mock_verify.return_value = {"sub": "user123"}
            
            # 1. Create scan
            scan_request = {
                "targets": ["127.0.0.1"],
                "ports": [80, 443, 22],
                "scan_type": "tcp",
                "timeout": 5.0
            }
            
            create_response = await async_client.post(
                "/api/v1/scans",
                json=scan_request,
                headers=headers
            )
            
            assert create_response.status_code == 201
            scan_id = create_response.json()["scan_id"]
            
            # 2. Check scan status
            status_response = await async_client.get(f"/api/v1/scans/{scan_id}", headers=headers)
            assert status_response.status_code == 200
            
            # 3. Wait for completion (in real scenario)
            # For testing, we'll just verify the scan was created properly
            scan_data = status_response.json()
            assert scan_data["scan_id"] == scan_id
            assert scan_data["status"] in ["pending", "running", "completed"]
    
    @pytest.mark.asyncio
    async def test_user_isolation(self, async_client, mock_token) -> Any:
        """Test that users can only access their own scans."""
        headers_user1 = {"Authorization": f"Bearer {mock_token}"}
        headers_user2 = {"Authorization": f"Bearer {mock_token}"}
        
        with patch('api_examples.SecurityService.verify_token') as mock_verify:
            # Create scan for user1
            mock_verify.return_value = {"sub": "user1"}
            
            scan_request = {
                "targets": ["127.0.0.1"],
                "ports": [80],
                "scan_type": "tcp",
                "timeout": 5.0
            }
            
            create_response = await async_client.post(
                "/api/v1/scans",
                json=scan_request,
                headers=headers_user1
            )
            
            scan_id = create_response.json()["scan_id"]
            
            # Try to access scan with user2
            mock_verify.return_value = {"sub": "user2"}
            
            access_response = await async_client.get(f"/api/v1/scans/{scan_id}", headers=headers_user2)
            
            assert access_response.status_code == 403

# ============================================================================
# BENCHMARK TESTS
# ============================================================================

class TestBenchmarks:
    """Benchmark tests for API performance."""
    
    @pytest.mark.asyncio
    async def test_scan_creation_benchmark(self, async_client, mock_token, valid_scan_request) -> Any:
        """Benchmark scan creation performance."""
        headers = {"Authorization": f"Bearer {mock_token}"}
        
        with patch('api_examples.SecurityService.verify_token') as mock_verify:
            mock_verify.return_value = {"sub": "user123"}
            
            times = []
            
            for _ in range(100):
                start_time = time.time()
                
                response = await async_client.post(
                    "/api/v1/scans",
                    json=valid_scan_request,
                    headers=headers
                )
                
                end_time = time.time()
                times.append(end_time - start_time)
                
                assert response.status_code == 201
            
            avg_time = sum(times) / len(times)
            max_time = max(times)
            
            # Performance assertions
            assert avg_time < 0.1  # Average should be under 100ms
            assert max_time < 0.5  # Max should be under 500ms
    
    @pytest.mark.asyncio
    async async def test_concurrent_requests_benchmark(self, async_client, mock_token, valid_scan_request) -> Any:
        """Benchmark concurrent request handling."""
        headers = {"Authorization": f"Bearer {mock_token}"}
        
        with patch('api_examples.SecurityService.verify_token') as mock_verify:
            mock_verify.return_value = {"sub": "user123"}
            
            start_time = time.time()
            
            # Create 50 concurrent requests
            tasks = [
                async_client.post("/api/v1/scans", json=valid_scan_request, headers=headers)
                for _ in range(50)
            ]
            
            responses = await asyncio.gather(*tasks)
            
            total_time = time.time() - start_time
            
            # All should succeed
            assert all(r.status_code == 201 for r in responses)
            
            # Performance assertions
            assert total_time < 10.0  # Should complete within 10 seconds
            assert total_time / 50 < 0.2  # Average per request should be under 200ms

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    # Run tests with pytest
    
    # Run specific test categories
    test_categories = [
        "test_scan_service",
        "test_vulnerability_service", 
        "test_security_service",
        "test_scan_api",
        "test_vulnerability_api",
        "test_health_api",
        "test_performance",
        "test_security",
        "test_error_handling",
        "test_integration",
        "test_benchmarks"
    ]
    
    for category in test_categories:
        print(f"\nRunning {category} tests...")
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            f"test_api_testing_examples.py::{category}",
            "-v", "--tb=short"
        ])
        
        if result.returncode != 0:
            print(f"❌ {category} tests failed")
        else:
            print(f"✅ {category} tests passed")
    
    print("\n🎉 All tests completed!") 