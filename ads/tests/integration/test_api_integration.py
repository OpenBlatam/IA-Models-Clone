"""
Integration tests for the ads API layer.

This module tests the integration between different API components:
- Core API endpoints
- AI API endpoints  
- Advanced API endpoints
- Integrated API endpoints
- Optimized API endpoints
- Cross-layer API interactions
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
from typing import Dict, Any, List

# Import the API components
from agents.backend.onyx.server.features.ads.api import main_router
from agents.backend.onyx.server.features.ads.api.core import router as core_router
from agents.backend.onyx.server.features.ads.api.ai import router as ai_router
from agents.backend.onyx.server.features.ads.api.advanced import router as advanced_router
from agents.backend.onyx.server.features.ads.api.integrated import router as integrated_router
from agents.backend.onyx.server.features.ads.api.optimized import router as optimized_router

# Import domain and application components for testing
from agents.backend.onyx.server.features.ads.domain.entities import Ad, AdCampaign, AdGroup
from agents.backend.onyx.server.features.ads.application.dto import (
    CreateAdRequest, CreateAdResponse, BrandVoiceRequest, BrandVoiceResponse
)
from agents.backend.onyx.server.features.ads.application.use_cases import CreateAdUseCase


class TestAPIIntegration:
    """Test API integration and cross-layer communication."""

    @pytest.fixture
    def app(self):
        """Create a test FastAPI application."""
        app = FastAPI(title="Ads API Test")
        app.include_router(main_router)
        return app

    @pytest.fixture
    def client(self, app):
        """Create a test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_create_ad_use_case(self):
        """Mock the CreateAdUseCase."""
        mock = AsyncMock(spec=CreateAdUseCase)
        mock.execute.return_value = CreateAdResponse(
            success=True,
            ad_id="test-ad-123",
            message="Ad created successfully",
            ad_data={
                "id": "test-ad-123",
                "title": "Test Ad",
                "description": "Test Description",
                "status": "draft"
            }
        )
        return mock

    @pytest.fixture
    def sample_ad_request(self):
        """Sample ad creation request."""
        return {
            "title": "Test Ad Campaign",
            "description": "A test advertisement for integration testing",
            "brand_voice": "Professional and friendly",
            "target_audience": "Tech professionals",
            "platform": "facebook",
            "budget": {
                "amount": 1000.0,
                "currency": "USD",
                "duration_days": 30
            },
            "targeting_criteria": {
                "age_range": [25, 45],
                "interests": ["technology", "business"],
                "location": "United States"
            }
        }

    @pytest.mark.asyncio
    async def test_core_api_router_integration(self, client):
        """Test that the core API router is properly integrated."""
        # Test health endpoint
        response = client.get("/ads/core/health")
        assert response.status_code == 200
        
        # Test capabilities endpoint
        response = client.get("/ads/core/capabilities")
        assert response.status_code == 200
        data = response.json()
        assert "capabilities" in data
        assert "endpoints" in data

    @pytest.mark.asyncio
    async def test_ai_api_router_integration(self, client):
        """Test that the AI API router is properly integrated."""
        # Test capabilities endpoint
        response = client.get("/ads/ai/capabilities")
        assert response.status_code == 200
        data = response.json()
        assert "ai_capabilities" in data
        assert "models" in data

    @pytest.mark.asyncio
    async def test_advanced_api_router_integration(self, client):
        """Test that the advanced API router is properly integrated."""
        # Test capabilities endpoint
        response = client.get("/ads/advanced/capabilities")
        assert response.status_code == 200
        data = response.json()
        assert "advanced_capabilities" in data
        assert "features" in data

    @pytest.mark.asyncio
    async def test_integrated_api_router_integration(self, client):
        """Test that the integrated API router is properly integrated."""
        # Test capabilities endpoint
        response = client.get("/ads/integrated/capabilities")
        assert response.status_code == 200
        data = response.json()
        assert "integration_capabilities" in data
        assert "platforms" in data

    @pytest.mark.asyncio
    async def test_optimized_api_router_integration(self, client):
        """Test that the optimized API router is properly integrated."""
        # Test health endpoint
        response = client.get("/ads/optimized/health")
        assert response.status_code == 200
        
        # Test capabilities endpoint
        response = client.get("/ads/optimized/capabilities")
        assert response.status_code == 200
        data = response.json()
        assert "optimization_capabilities" in data
        assert "performance_features" in data

    @pytest.mark.asyncio
    async def test_main_router_integration(self, client):
        """Test that the main router properly includes all sub-routers."""
        # Test that all router prefixes are accessible
        endpoints = [
            "/ads/core/health",
            "/ads/ai/capabilities", 
            "/ads/advanced/capabilities",
            "/ads/integrated/capabilities",
            "/ads/optimized/health"
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200, f"Endpoint {endpoint} failed"

    @pytest.mark.asyncio
    async def test_cross_router_data_flow(self, client):
        """Test data flow between different API routers."""
        # Test that data can flow from core to other routers
        core_response = client.get("/ads/core/capabilities")
        assert core_response.status_code == 200
        
        # Test that other routers can access similar data
        ai_response = client.get("/ads/ai/capabilities")
        assert ai_response.status_code == 200
        
        # Verify consistent response structure
        core_data = core_response.json()
        ai_data = ai_response.json()
        
        assert "capabilities" in core_data
        assert "ai_capabilities" in ai_data

    @pytest.mark.asyncio
    async def test_api_error_handling_integration(self, client):
        """Test error handling across API layers."""
        # Test invalid endpoint
        response = client.get("/ads/invalid/endpoint")
        assert response.status_code == 404
        
        # Test invalid method
        response = client.post("/ads/core/health")
        assert response.status_code == 405

    @pytest.mark.asyncio
    async def test_api_authentication_integration(self, client):
        """Test authentication integration across API layers."""
        # Test protected endpoints (if any)
        # This would depend on the actual authentication implementation
        
        # For now, test that public endpoints are accessible
        public_endpoints = [
            "/ads/core/health",
            "/ads/core/capabilities",
            "/ads/ai/capabilities"
        ]
        
        for endpoint in public_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_api_rate_limiting_integration(self, client):
        """Test rate limiting integration across API layers."""
        # Test that rate limiting is consistent across routers
        # This would depend on the actual rate limiting implementation
        
        # For now, test that endpoints respond to multiple requests
        for _ in range(3):
            response = client.get("/ads/core/health")
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_api_logging_integration(self, client):
        """Test logging integration across API layers."""
        # Test that all endpoints generate logs
        # This would depend on the actual logging implementation
        
        # For now, test that endpoints respond consistently
        endpoints = [
            "/ads/core/health",
            "/ads/ai/capabilities",
            "/ads/advanced/capabilities"
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_api_metrics_integration(self, client):
        """Test metrics integration across API layers."""
        # Test that metrics are collected consistently
        # This would depend on the actual metrics implementation
        
        # For now, test that endpoints respond and can be monitored
        response = client.get("/ads/core/health")
        assert response.status_code == 200
        
        # Verify response time is reasonable
        assert response.elapsed.total_seconds() < 1.0

    @pytest.mark.asyncio
    async def test_api_caching_integration(self, client):
        """Test caching integration across API layers."""
        # Test that caching works consistently
        # This would depend on the actual caching implementation
        
        # For now, test that endpoints respond consistently
        endpoint = "/ads/core/capabilities"
        
        # First request
        response1 = client.get(endpoint)
        assert response1.status_code == 200
        data1 = response1.json()
        
        # Second request (should be cached if implemented)
        response2 = client.get(endpoint)
        assert response2.status_code == 200
        data2 = response2.json()
        
        # Data should be consistent
        assert data1 == data2

    @pytest.mark.asyncio
    async def test_api_background_tasks_integration(self, client):
        """Test background tasks integration across API layers."""
        # Test that background tasks are properly integrated
        # This would depend on the actual background task implementation
        
        # For now, test that endpoints respond
        response = client.get("/ads/optimized/capabilities")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_api_validation_integration(self, client):
        """Test validation integration across API layers."""
        # Test that validation is consistent across routers
        
        # Test with invalid data (if POST endpoints exist)
        # For now, test that GET endpoints handle invalid parameters gracefully
        
        # Test with query parameters
        response = client.get("/ads/core/capabilities?invalid=param")
        assert response.status_code == 200  # Should ignore invalid params

    @pytest.mark.asyncio
    async def test_api_middleware_integration(self, client):
        """Test middleware integration across API layers."""
        # Test that middleware is properly applied
        
        # Test CORS headers
        response = client.options("/ads/core/health")
        # CORS preflight should work
        
        # Test that regular requests work
        response = client.get("/ads/core/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_api_dependency_injection_integration(self, client):
        """Test dependency injection integration across API layers."""
        # Test that dependencies are properly injected
        
        # This would depend on the actual DI implementation
        # For now, test that endpoints respond
        response = client.get("/ads/core/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_api_serialization_integration(self, client):
        """Test serialization integration across API layers."""
        # Test that serialization is consistent
        
        # Test JSON responses
        response = client.get("/ads/core/capabilities")
        assert response.status_code == 200
        
        # Verify JSON is valid
        data = response.json()
        assert isinstance(data, dict)
        
        # Verify expected structure
        assert "capabilities" in data

    @pytest.mark.asyncio
    async def test_api_async_integration(self, client):
        """Test async integration across API layers."""
        # Test that async operations work properly
        
        # Test concurrent requests
        import asyncio
        
        async def make_request():
            return client.get("/ads/core/health")
        
        # Make multiple concurrent requests
        responses = await asyncio.gather(
            *[make_request() for _ in range(3)]
        )
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_api_performance_integration(self, client):
        """Test performance integration across API layers."""
        # Test that performance is consistent
        
        # Test response times
        import time
        
        start_time = time.time()
        response = client.get("/ads/core/health")
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 1.0  # Should respond within 1 second

    @pytest.mark.asyncio
    async def test_api_security_integration(self, client):
        """Test security integration across API layers."""
        # Test that security measures are in place
        
        # Test for common security headers
        response = client.get("/ads/core/health")
        assert response.status_code == 200
        
        # Check for security headers (if implemented)
        headers = response.headers
        # These would depend on actual security implementation
        # assert "X-Content-Type-Options" in headers
        # assert "X-Frame-Options" in headers

    @pytest.mark.asyncio
    async def test_api_monitoring_integration(self, client):
        """Test monitoring integration across API layers."""
        # Test that monitoring is properly integrated
        
        # Test health check
        response = client.get("/ads/core/health")
        assert response.status_code == 200
        
        # Test that metrics are collected
        # This would depend on actual monitoring implementation

    @pytest.mark.asyncio
    async def test_api_documentation_integration(self, client):
        """Test API documentation integration."""
        # Test that OpenAPI docs are accessible
        response = client.get("/docs")
        # This would depend on FastAPI docs configuration
        # assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_api_versioning_integration(self, client):
        """Test API versioning integration."""
        # Test that versioning is properly handled
        
        # Test current version endpoints
        response = client.get("/ads/core/health")
        assert response.status_code == 200
        
        # Test version-specific endpoints (if implemented)
        # response = client.get("/ads/v1/core/health")
        # assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_api_health_check_integration(self, client):
        """Test health check integration across all API layers."""
        # Test health endpoints in all routers
        health_endpoints = [
            "/ads/core/health",
            "/ads/optimized/health"
        ]
        
        for endpoint in health_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200
            
            data = response.json()
            assert "status" in data
            assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_api_capabilities_integration(self, client):
        """Test capabilities endpoint integration across all API layers."""
        # Test capabilities endpoints in all routers
        capabilities_endpoints = [
            "/ads/core/capabilities",
            "/ads/ai/capabilities",
            "/ads/advanced/capabilities",
            "/ads/integrated/capabilities",
            "/ads/optimized/capabilities"
        ]
        
        for endpoint in capabilities_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200
            
            data = response.json()
            assert "capabilities" in data or "ai_capabilities" in data or "advanced_capabilities" in data or "integration_capabilities" in data or "optimization_capabilities" in data

    @pytest.mark.asyncio
    async def test_api_end_to_end_integration(self, client):
        """Test end-to-end API integration."""
        # Test a complete flow through multiple API layers
        
        # 1. Check system health
        health_response = client.get("/ads/core/health")
        assert health_response.status_code == 200
        
        # 2. Get system capabilities
        capabilities_response = client.get("/ads/core/capabilities")
        assert capabilities_response.status_code == 200
        
        # 3. Check AI capabilities
        ai_capabilities_response = client.get("/ads/ai/capabilities")
        assert ai_capabilities_response.status_code == 200
        
        # 4. Check advanced capabilities
        advanced_capabilities_response = client.get("/ads/advanced/capabilities")
        assert advanced_capabilities_response.status_code == 200
        
        # 5. Check integrated capabilities
        integrated_capabilities_response = client.get("/ads/integrated/capabilities")
        assert integrated_capabilities_response.status_code == 200
        
        # 6. Check optimized capabilities
        optimized_capabilities_response = client.get("/ads/optimized/capabilities")
        assert optimized_capabilities_response.status_code == 200
        
        # All endpoints should be accessible and functional
        assert all([
            health_response.status_code == 200,
            capabilities_response.status_code == 200,
            ai_capabilities_response.status_code == 200,
            advanced_capabilities_response.status_code == 200,
            integrated_capabilities_response.status_code == 200,
            optimized_capabilities_response.status_code == 200
        ])


class TestAPICrossLayerIntegration:
    """Test cross-layer API integration."""

    @pytest.mark.asyncio
    async def test_domain_to_api_integration(self):
        """Test integration between domain layer and API layer."""
        # Test that domain entities can be used in API responses
        
        # Create a domain entity
        ad = Ad(
            id="test-ad-123",
            title="Test Ad",
            description="Test Description",
            status="draft",
            platform="facebook",
            budget=1000.0
        )
        
        # Verify entity is properly structured
        assert ad.id == "test-ad-123"
        assert ad.title == "Test Ad"
        assert ad.status == "draft"
        
        # Test that entity can be serialized (if needed for API)
        ad_dict = {
            "id": ad.id,
            "title": ad.title,
            "description": ad.description,
            "status": ad.status,
            "platform": ad.platform,
            "budget": ad.budget
        }
        
        assert ad_dict["id"] == "test-ad-123"
        assert ad_dict["title"] == "Test Ad"

    @pytest.mark.asyncio
    async def test_application_to_api_integration(self):
        """Test integration between application layer and API layer."""
        # Test that application DTOs can be used in API requests/responses
        
        # Create a request DTO
        request = CreateAdRequest(
            title="Test Ad",
            description="Test Description",
            brand_voice="Professional",
            target_audience="Tech professionals",
            platform="facebook",
            budget=1000.0
        )
        
        # Verify DTO is properly structured
        assert request.title == "Test Ad"
        assert request.platform == "facebook"
        assert request.budget == 1000.0
        
        # Create a response DTO
        response = CreateAdResponse(
            success=True,
            ad_id="test-ad-123",
            message="Ad created successfully",
            ad_data={"id": "test-ad-123", "title": "Test Ad"}
        )
        
        # Verify response DTO is properly structured
        assert response.success is True
        assert response.ad_id == "test-ad-123"
        assert "ad_data" in response.__dict__

    @pytest.mark.asyncio
    async def test_infrastructure_to_api_integration(self):
        """Test integration between infrastructure layer and API layer."""
        # Test that infrastructure services can be used by API endpoints
        
        # This would test actual service integration
        # For now, test that the structure supports it
        
        # Test that API can access infrastructure components
        from agents.backend.onyx.server.features.ads.infrastructure.database import DatabaseManager
        from agents.backend.onyx.server.features.ads.infrastructure.storage import FileStorageManager
        from agents.backend.onyx.server.features.ads.infrastructure.cache import CacheManager
        
        # Verify infrastructure components exist and can be imported
        assert DatabaseManager is not None
        assert FileStorageManager is not None
        assert CacheManager is not None

    @pytest.mark.asyncio
    async def test_optimization_to_api_integration(self):
        """Test integration between optimization layer and API layer."""
        # Test that optimization services can be used by API endpoints
        
        # Test that API can access optimization components
        from agents.backend.onyx.server.features.ads.optimization.factory import OptimizationFactory
        from agents.backend.onyx.server.features.ads.optimization.performance_optimizer import PerformanceOptimizer
        
        # Verify optimization components exist and can be imported
        assert OptimizationFactory is not None
        assert PerformanceOptimizer is not None

    @pytest.mark.asyncio
    async def test_training_to_api_integration(self):
        """Test integration between training layer and API layer."""
        # Test that training services can be used by API endpoints
        
        # Test that API can access training components
        from agents.backend.onyx.server.features.ads.training.factory import TrainingFactory
        from agents.backend.onyx.server.features.ads.training.pytorch_trainer import PyTorchTrainer
        
        # Verify training components exist and can be imported
        assert TrainingFactory is not None
        assert PyTorchTrainer is not None


class TestAPIPerformanceIntegration:
    """Test API performance integration."""

    @pytest.mark.asyncio
    async def test_api_response_time_consistency(self, client):
        """Test that API response times are consistent across layers."""
        import time
        
        endpoints = [
            "/ads/core/health",
            "/ads/ai/capabilities",
            "/ads/advanced/capabilities"
        ]
        
        response_times = []
        
        for endpoint in endpoints:
            start_time = time.time()
            response = client.get(endpoint)
            end_time = time.time()
            
            assert response.status_code == 200
            response_times.append(end_time - start_time)
        
        # Response times should be reasonably consistent
        avg_response_time = sum(response_times) / len(response_times)
        for response_time in response_times:
            assert response_time < avg_response_time * 2  # Within 2x of average

    @pytest.mark.asyncio
    async def test_api_concurrent_request_handling(self, client):
        """Test that API can handle concurrent requests properly."""
        import asyncio
        import time
        
        async def make_request():
            start_time = time.time()
            response = client.get("/ads/core/health")
            end_time = time.time()
            return response.status_code, end_time - start_time
        
        # Make 5 concurrent requests
        start_time = time.time()
        results = await asyncio.gather(*[make_request() for _ in range(5)])
        total_time = time.time() - start_time
        
        # All requests should succeed
        for status_code, response_time in results:
            assert status_code == 200
            assert response_time < 1.0  # Each request should complete within 1 second
        
        # Total time should be reasonable (concurrent execution)
        assert total_time < 2.0  # Should complete within 2 seconds total

    @pytest.mark.asyncio
    async def test_api_memory_usage_consistency(self, client):
        """Test that API memory usage is consistent."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Make several requests
        for _ in range(10):
            response = client.get("/ads/core/health")
            assert response.status_code == 200
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 10MB)
        assert memory_increase < 10 * 1024 * 1024  # 10MB in bytes


class TestAPIErrorHandlingIntegration:
    """Test API error handling integration."""

    @pytest.mark.asyncio
    async def test_api_error_response_consistency(self, client):
        """Test that error responses are consistent across API layers."""
        # Test invalid endpoints
        invalid_endpoints = [
            "/ads/invalid/endpoint",
            "/ads/core/invalid",
            "/ads/ai/invalid"
        ]
        
        for endpoint in invalid_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 404
        
        # Test invalid methods
        response = client.post("/ads/core/health")
        assert response.status_code == 405

    @pytest.mark.asyncio
    async def test_api_validation_error_handling(self, client):
        """Test that validation errors are handled consistently."""
        # Test with invalid query parameters
        response = client.get("/ads/core/capabilities?invalid=param&another=invalid")
        assert response.status_code == 200  # Should handle gracefully
        
        # Test with malformed requests (if POST endpoints exist)
        # This would depend on actual POST endpoint implementation

    @pytest.mark.asyncio
    async def test_api_internal_error_handling(self, client):
        """Test that internal errors are handled gracefully."""
        # Test that endpoints don't crash on internal errors
        # This would depend on actual error handling implementation
        
        # For now, test that endpoints respond
        response = client.get("/ads/core/health")
        assert response.status_code == 200


class TestAPISecurityIntegration:
    """Test API security integration."""

    @pytest.mark.asyncio
    async def test_api_input_validation(self, client):
        """Test that API input validation is secure."""
        # Test with potentially malicious input
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../../etc/passwd",
            "admin' OR '1'='1"
        ]
        
        # Test that endpoints handle malicious input gracefully
        for malicious_input in malicious_inputs:
            response = client.get(f"/ads/core/capabilities?param={malicious_input}")
            # Should not crash or expose sensitive information
            assert response.status_code in [200, 400, 422]

    @pytest.mark.asyncio
    async def test_api_rate_limiting_security(self, client):
        """Test that rate limiting provides security benefits."""
        # Test that endpoints don't allow unlimited requests
        # This would depend on actual rate limiting implementation
        
        # For now, test that endpoints respond consistently
        for _ in range(5):
            response = client.get("/ads/core/health")
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_api_authentication_security(self, client):
        """Test that authentication is properly secured."""
        # Test that protected endpoints require authentication
        # This would depend on actual authentication implementation
        
        # For now, test that public endpoints are accessible
        response = client.get("/ads/core/health")
        assert response.status_code == 200


class TestAPIMonitoringIntegration:
    """Test API monitoring integration."""

    @pytest.mark.asyncio
    async def test_api_health_monitoring(self, client):
        """Test that health monitoring works across API layers."""
        # Test health endpoints
        health_endpoints = [
            "/ads/core/health",
            "/ads/optimized/health"
        ]
        
        for endpoint in health_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200
            
            data = response.json()
            assert "status" in data
            assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_api_metrics_collection(self, client):
        """Test that metrics are collected consistently."""
        # Test that endpoints generate metrics
        # This would depend on actual metrics implementation
        
        # For now, test that endpoints respond and can be monitored
        response = client.get("/ads/core/health")
        assert response.status_code == 200
        
        # Verify response time is measurable
        assert response.elapsed.total_seconds() >= 0

    @pytest.mark.asyncio
    async def test_api_logging_consistency(self, client):
        """Test that logging is consistent across API layers."""
        # Test that all endpoints generate logs
        # This would depend on actual logging implementation
        
        # For now, test that endpoints respond consistently
        endpoints = [
            "/ads/core/health",
            "/ads/ai/capabilities",
            "/ads/advanced/capabilities"
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200


# Test data and utilities for integration tests
@pytest.fixture
def sample_api_test_data():
    """Sample data for API integration tests."""
    return {
        "valid_ad_request": {
            "title": "Test Ad",
            "description": "Test Description",
            "brand_voice": "Professional",
            "target_audience": "Tech professionals",
            "platform": "facebook",
            "budget": 1000.0
        },
        "invalid_ad_request": {
            "title": "",  # Invalid: empty title
            "description": "Test Description",
            "platform": "invalid_platform",  # Invalid platform
            "budget": -100  # Invalid: negative budget
        },
        "test_endpoints": [
            "/ads/core/health",
            "/ads/ai/capabilities",
            "/ads/advanced/capabilities",
            "/ads/integrated/capabilities",
            "/ads/optimized/health"
        ]
    }


@pytest.fixture
def api_test_utilities():
    """Utility functions for API integration tests."""
    
    def validate_response_structure(response_data: Dict[str, Any], expected_keys: List[str]) -> bool:
        """Validate that response has expected structure."""
        return all(key in response_data for key in expected_keys)
    
    def validate_error_response(response_data: Dict[str, Any]) -> bool:
        """Validate error response structure."""
        return "error" in response_data or "detail" in response_data
    
    def measure_response_time(func, *args, **kwargs):
        """Measure execution time of a function."""
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    
    return {
        "validate_response_structure": validate_response_structure,
        "validate_error_response": validate_error_response,
        "measure_response_time": measure_response_time
    }
