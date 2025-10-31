"""
Integration Tests
================

End-to-end integration tests for the copywriting service.
"""

import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient

from ..app import app
from ..services import get_copywriting_service, cleanup_copywriting_service
from ..schemas import CopywritingRequest, CopywritingTone, CopywritingStyle, CopywritingPurpose


@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Async test client fixture"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture(autouse=True)
async def setup_teardown():
    """Setup and teardown for each test"""
    # Setup
    service = await get_copywriting_service()
    yield
    # Teardown
    await cleanup_copywriting_service()


class TestCopywritingIntegration:
    """Integration tests for copywriting endpoints"""
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/api/v2/copywriting/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "uptime_seconds" in data
        assert "dependencies" in data
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Copywriting Service API"
        assert data["version"] == "2.0.0"
        assert data["status"] == "running"
    
    def test_generate_copywriting_sync(self, client):
        """Test copywriting generation (synchronous)"""
        request_data = {
            "topic": "AI-Powered Marketing",
            "target_audience": "Marketing professionals",
            "key_points": ["Automation", "Personalization", "ROI"],
            "tone": "professional",
            "style": "direct_response",
            "purpose": "sales",
            "word_count": 500,
            "include_cta": True,
            "variants_count": 2
        }
        
        response = client.post("/api/v2/copywriting/generate", json=request_data)
        assert response.status_code == 201
        
        data = response.json()
        assert "request_id" in data
        assert "variants" in data
        assert "processing_time_ms" in data
        assert len(data["variants"]) == 2
        
        # Check variant structure
        variant = data["variants"][0]
        assert "id" in variant
        assert "title" in variant
        assert "content" in variant
        assert "word_count" in variant
        assert "confidence_score" in variant
    
    @pytest.mark.asyncio
    async def test_generate_copywriting_async(self, async_client):
        """Test copywriting generation (asynchronous)"""
        request_data = {
            "topic": "Async Test Topic",
            "target_audience": "Async test audience",
            "tone": "casual",
            "style": "storytelling",
            "purpose": "engagement"
        }
        
        response = await async_client.post("/api/v2/copywriting/generate", json=request_data)
        assert response.status_code == 201
        
        data = response.json()
        assert "request_id" in data
        assert "variants" in data
        assert len(data["variants"]) == 3  # Default variants_count
    
    def test_batch_generation(self, client):
        """Test batch copywriting generation"""
        request_data = {
            "requests": [
                {
                    "topic": "Batch Topic 1",
                    "target_audience": "Audience 1"
                },
                {
                    "topic": "Batch Topic 2",
                    "target_audience": "Audience 2"
                }
            ]
        }
        
        response = client.post("/api/v2/copywriting/generate/batch", json=request_data)
        assert response.status_code == 201
        
        data = response.json()
        assert "batch_id" in data
        assert "results" in data
        assert "failed_requests" in data
        assert "total_processing_time_ms" in data
        assert len(data["results"]) == 2
    
    def test_feedback_submission(self, client):
        """Test feedback submission"""
        # First generate some content to get a variant ID
        request_data = {
            "topic": "Feedback Test",
            "target_audience": "Test audience"
        }
        
        response = client.post("/api/v2/copywriting/generate", json=request_data)
        assert response.status_code == 201
        
        data = response.json()
        variant_id = data["variants"][0]["id"]
        
        # Submit feedback
        feedback_data = {
            "variant_id": variant_id,
            "rating": 4,
            "feedback_text": "Great content!",
            "improvements": ["More examples"],
            "is_helpful": True
        }
        
        response = client.post("/api/v2/copywriting/feedback", json=feedback_data)
        assert response.status_code == 201
        
        data = response.json()
        assert "feedback_id" in data
        assert data["variant_id"] == variant_id
        assert data["status"] == "received"
    
    def test_validation_errors(self, client):
        """Test validation error handling"""
        # Test empty topic
        request_data = {
            "topic": "",
            "target_audience": "Test audience"
        }
        
        response = client.post("/api/v2/copywriting/generate", json=request_data)
        assert response.status_code == 422
        
        # Test invalid word count
        request_data = {
            "topic": "Valid topic",
            "target_audience": "Test audience",
            "word_count": 10  # Too low
        }
        
        response = client.post("/api/v2/copywriting/generate", json=request_data)
        assert response.status_code == 422
    
    def test_invalid_feedback(self, client):
        """Test invalid feedback submission"""
        feedback_data = {
            "variant_id": "invalid-uuid",
            "rating": 6,  # Too high
            "is_helpful": True
        }
        
        response = client.post("/api/v2/copywriting/feedback", json=feedback_data)
        assert response.status_code == 422
    
    def test_stats_endpoint(self, client):
        """Test stats endpoint"""
        response = client.get("/api/v2/copywriting/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_requests" in data
        assert "total_variants_generated" in data
        assert "average_processing_time_ms" in data
        assert "success_rate" in data
    
    def test_variant_not_found(self, client):
        """Test variant not found error"""
        import uuid
        variant_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v2/copywriting/variants/{variant_id}")
        assert response.status_code == 404


class TestServiceIntegration:
    """Integration tests for service layer"""
    
    @pytest.mark.asyncio
    async def test_service_startup_shutdown(self):
        """Test service startup and shutdown"""
        service = await get_copywriting_service()
        assert service is not None
        
        # Test health check
        health = await service.get_health_status()
        assert health.status in ["healthy", "degraded", "unhealthy"]
        assert health.uptime_seconds >= 0
    
    @pytest.mark.asyncio
    async def test_copywriting_generation_service(self):
        """Test copywriting generation through service"""
        service = await get_copywriting_service()
        
        request = CopywritingRequest(
            topic="Service Test Topic",
            target_audience="Service test audience",
            tone=CopywritingTone.PROFESSIONAL,
            style=CopywritingStyle.DIRECT_RESPONSE,
            purpose=CopywritingPurpose.SALES
        )
        
        response = await service.generate_copywriting(request)
        
        assert response.request_id is not None
        assert len(response.variants) == 3
        assert response.processing_time_ms >= 0
        assert response.total_variants == 3
        assert response.best_variant is not None
    
    @pytest.mark.asyncio
    async def test_batch_generation_service(self):
        """Test batch generation through service"""
        service = await get_copywriting_service()
        
        from ..schemas import BatchCopywritingRequest
        
        requests = [
            CopywritingRequest(topic=f"Topic {i}", target_audience=f"Audience {i}")
            for i in range(3)
        ]
        
        batch_request = BatchCopywritingRequest(requests=requests)
        response = await service.generate_batch_copywriting(batch_request)
        
        assert response.batch_id is not None
        assert len(response.results) == 3
        assert len(response.failed_requests) == 0
        assert response.success_count == 3
        assert response.failure_count == 0


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_malformed_json(self, client):
        """Test malformed JSON handling"""
        response = client.post(
            "/api/v2/copywriting/generate",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_required_fields(self, client):
        """Test missing required fields"""
        request_data = {
            "topic": "Valid topic"
            # Missing target_audience
        }
        
        response = client.post("/api/v2/copywriting/generate", json=request_data)
        assert response.status_code == 422
    
    def test_invalid_enum_values(self, client):
        """Test invalid enum values"""
        request_data = {
            "topic": "Valid topic",
            "target_audience": "Valid audience",
            "tone": "invalid_tone"  # Invalid enum value
        }
        
        response = client.post("/api/v2/copywriting/generate", json=request_data)
        assert response.status_code == 422
    
    def test_large_request(self, client):
        """Test handling of large requests"""
        request_data = {
            "topic": "Large topic " * 100,  # Very long topic
            "target_audience": "Large audience " * 100,
            "key_points": [f"Point {i}" for i in range(20)]  # Many key points
        }
        
        response = client.post("/api/v2/copywriting/generate", json=request_data)
        # Should either succeed or fail gracefully
        assert response.status_code in [201, 422, 400]


@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test handling of concurrent requests"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Create multiple concurrent requests
        tasks = []
        for i in range(5):
            request_data = {
                "topic": f"Concurrent Topic {i}",
                "target_audience": f"Concurrent Audience {i}"
            }
            task = client.post("/api/v2/copywriting/generate", json=request_data)
            tasks.append(task)
        
        # Execute all requests concurrently
        responses = await asyncio.gather(*tasks)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 201
            data = response.json()
            assert "variants" in data
            assert len(data["variants"]) > 0






























