"""
Comprehensive Tests for Health Checks
Tests for all health check endpoints
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from main import create_application
from api.health import (
    health_check,
    readiness_check,
    liveness_check,
    detailed_health_check
)
from tests.test_base import BaseAPITest
from tests.test_helpers import AssertionHelpers


class TestHealthCheck(BaseAPITest):
    """Tests for basic health check"""
    
    @pytest.mark.asyncio
    async def test_health_check_endpoint(self, client):
        """Test basic health check endpoint"""
        response = client.get("/health")
        
        self.assert_status_code(response, 200)
        data = self.assert_json_response(response, ["status", "timestamp", "service"])
        assert data["status"] == "healthy"
        assert data["service"] == "dermatology_ai"
    
    @pytest.mark.asyncio
    async def test_health_check_function(self):
        """Test health check function directly"""
        result = await health_check()
        
        assert result["status"] == "healthy"
        assert "timestamp" in result
        assert result["service"] == "dermatology_ai"


class TestReadinessCheck(BaseAPITest):
    """Tests for readiness check"""
    
    @pytest.mark.asyncio
    async def test_readiness_check_ready(self):
        """Test readiness check when service is ready"""
        with patch('api.health.get_composition_root') as mock_composition:
            mock_root = Mock()
            mock_root._initialized = True
            mock_root._database_adapter = Mock()
            mock_root.service_factory.create = AsyncMock(return_value=Mock())
            mock_composition.return_value = mock_root
            
            result = await readiness_check()
            
            assert result["status"] in ["ready", "not_ready"]
            assert "checks" in result
            assert "ready" in result
    
    @pytest.mark.asyncio
    async def test_readiness_check_not_ready(self):
        """Test readiness check when service is not ready"""
        with patch('api.health.get_composition_root') as mock_composition:
            mock_root = Mock()
            mock_root._initialized = False
            mock_composition.return_value = mock_root
            
            with pytest.raises(Exception):  # HTTPException
                await readiness_check()
    
    @pytest.mark.asyncio
    async def test_readiness_check_endpoint(self, client):
        """Test readiness check endpoint"""
        response = client.get("/health/ready")
        
        # May return 200 or 503 depending on initialization
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "checks" in data


class TestLivenessCheck(BaseAPITest):
    """Tests for liveness check"""
    
    @pytest.mark.asyncio
    async def test_liveness_check(self):
        """Test liveness check function"""
        result = await liveness_check()
        
        assert result["status"] == "alive"
        assert "timestamp" in result
    
    @pytest.mark.asyncio
    async def test_liveness_check_endpoint(self, client):
        """Test liveness check endpoint"""
        response = client.get("/health/live")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"


class TestDetailedHealthCheck(BaseAPITest):
    """Tests for detailed health check"""
    
    @pytest.mark.asyncio
    async def test_detailed_health_check_initialized(self):
        """Test detailed health check when initialized"""
        with patch('api.health.get_composition_root') as mock_composition:
            mock_root = Mock()
            mock_root._initialized = True
            mock_root._use_case_cache = {"use_case1": Mock(), "use_case2": Mock()}
            mock_root.service_factory = Mock()
            mock_root.service_factory.registrations = {"service1": Mock()}
            mock_root.service_factory.singletons = {"singleton1": Mock()}
            mock_composition.return_value = mock_root
            
            result = await detailed_health_check()
            
            assert result["status"] in ["healthy", "degraded"]
            assert "components" in result
            assert "composition_root" in result["components"]
    
    @pytest.mark.asyncio
    async def test_detailed_health_check_not_initialized(self):
        """Test detailed health check when not initialized"""
        with patch('api.health.get_composition_root') as mock_composition:
            mock_root = Mock()
            mock_root._initialized = False
            mock_composition.return_value = mock_root
            
            result = await detailed_health_check()
            
            assert result["status"] == "degraded"
            assert "components" in result
    
    @pytest.mark.asyncio
    async def test_detailed_health_check_endpoint(self, client):
        """Test detailed health check endpoint"""
        response = client.get("/health/detailed")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data


class TestHealthCheckIntegration(BaseAPITest):
    """Integration tests for health checks"""
    
    def test_health_check_chain(self, client):
        """Test health check chain (health -> ready -> live)"""
        # Basic health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # Liveness (should always work)
        live_response = client.get("/health/live")
        assert live_response.status_code == 200
        
        # Readiness (may vary)
        ready_response = client.get("/health/ready")
        assert ready_response.status_code in [200, 503]
    
    def test_health_check_metrics(self, client):
        """Test health metrics endpoint"""
        response = client.get("/health/metrics")
        
        # May return 200 or 503 depending on Prometheus availability
        assert response.status_code in [200, 503]

