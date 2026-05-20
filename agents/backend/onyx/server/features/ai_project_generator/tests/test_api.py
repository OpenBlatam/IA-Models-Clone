"""
Tests for Generator API endpoints
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import json

from ..api.generator_api import create_generator_app, ProjectRequest


class TestGeneratorAPI:
    """Test suite for Generator API"""

    @pytest.fixture
    def client(self, temp_dir):
        """Create test client"""
        app = create_generator_app(
            base_output_dir=str(temp_dir / "generated_projects"),
            enable_continuous=True
        )
        return TestClient(app)

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_health_detailed(self, client):
        """Test detailed health check"""
        response = client.get("/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_api_version(self, client):
        """Test API version endpoint"""
        response = client.get("/api/version")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data

    def test_status_endpoint(self, client):
        """Test status endpoint"""
        response = client.get("/api/v1/status")
        assert response.status_code == 200
        data = response.json()
        assert "is_running" in data

    def test_generate_project(self, client):
        """Test project generation endpoint"""
        with patch('ai_project_generator.api.generator_api.continuous_generator') as mock_gen:
            mock_gen.add_project.return_value = "test-project-123"
            mock_gen.get_project_status.return_value = {
                "id": "test-project-123",
                "status": "pending",
                "description": "A test AI project"
            }
            
            response = client.post(
                "/api/v1/generate",
                json={
                    "description": "A test AI project that does something useful with machine learning",
                    "project_name": "test_project",
                    "author": "Test Author"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "project_id" in data
            assert data["status"] == "queued"

    def test_generate_project_invalid_description(self, client):
        """Test project generation with invalid description"""
        response = client.post(
            "/api/v1/generate",
            json={
                "description": "short",  # Too short
                "project_name": "test_project"
            }
        )
        assert response.status_code == 422  # Validation error

    def test_generate_project_spam_detection(self, client):
        """Test spam detection in description"""
        response = client.post(
            "/api/v1/generate",
            json={
                "description": "test test test test test",  # Repeated words
                "project_name": "test_project"
            }
        )
        # Should fail validation due to spam detection
        assert response.status_code == 422

    def test_get_project_status(self, client):
        """Test getting project status"""
        with patch('ai_project_generator.api.generator_api.continuous_generator') as mock_gen:
            mock_gen.get_project_status.return_value = {
                "id": "test-123",
                "status": "processing",
                "description": "Test project"
            }
            
            response = client.get("/api/v1/project/test-123")
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "test-123"
            assert data["status"] == "processing"

    def test_get_project_status_not_found(self, client):
        """Test getting status of non-existent project"""
        with patch('ai_project_generator.api.generator_api.continuous_generator') as mock_gen:
            mock_gen.get_project_status.return_value = None
            
            response = client.get("/api/v1/project/non-existent")
            assert response.status_code == 404

    def test_get_queue(self, client):
        """Test getting queue status"""
        with patch('ai_project_generator.api.generator_api.continuous_generator') as mock_gen:
            mock_gen.get_queue.return_value = {
                "total": 2,
                "pending": 2,
                "projects": [
                    {"id": "proj-1", "status": "pending"},
                    {"id": "proj-2", "status": "pending"}
                ]
            }
            
            response = client.get("/api/v1/queue")
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 2

    def test_start_generator(self, client):
        """Test starting the generator"""
        with patch('ai_project_generator.api.generator_api.continuous_generator') as mock_gen:
            mock_gen.is_running = False
            mock_gen.start = Mock()
            
            response = client.post("/api/v1/start")
            assert response.status_code == 200
            mock_gen.start.assert_called_once()

    def test_stop_generator(self, client):
        """Test stopping the generator"""
        with patch('ai_project_generator.api.generator_api.continuous_generator') as mock_gen:
            mock_gen.is_running = True
            mock_gen.stop = AsyncMock()
            
            response = client.post("/api/v1/stop")
            assert response.status_code == 200

    def test_delete_project(self, client):
        """Test deleting a project from queue"""
        with patch('ai_project_generator.api.generator_api.continuous_generator') as mock_gen:
            mock_gen.remove_project.return_value = True
            
            response = client.delete("/api/v1/project/test-123")
            assert response.status_code == 200
            mock_gen.remove_project.assert_called_once_with("test-123")

    def test_get_stats(self, client):
        """Test getting generator statistics"""
        with patch('ai_project_generator.api.generator_api.continuous_generator') as mock_gen:
            mock_gen.get_stats.return_value = {
                "queue_size": 5,
                "processed_count": 10,
                "is_running": True
            }
            
            response = client.get("/api/v1/stats")
            assert response.status_code == 200
            data = response.json()
            assert data["queue_size"] == 5

    def test_validate_project(self, client, temp_dir):
        """Test project validation endpoint"""
        project_path = str(temp_dir / "test_project")
        
        with patch('ai_project_generator.api.generator_api.ProjectValidator') as mock_validator:
            mock_validator.return_value.validate_project = AsyncMock(return_value={
                "valid": True,
                "errors": []
            })
            
            response = client.post(
                "/api/v1/validate",
                json={"project_path": project_path}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True

    def test_export_zip(self, client, temp_dir):
        """Test exporting project to ZIP"""
        project_path = str(temp_dir / "test_project")
        
        with patch('ai_project_generator.api.generator_api.ExportGenerator') as mock_export:
            mock_export.return_value.export_to_zip = AsyncMock(return_value={
                "zip_path": str(temp_dir / "test_project.zip"),
                "size": 1024
            })
            
            response = client.post(
                "/api/v1/export/zip",
                json={"project_path": project_path}
            )
            assert response.status_code == 200
            data = response.json()
            assert "zip_path" in data

    def test_batch_generate(self, client):
        """Test batch project generation"""
        with patch('ai_project_generator.api.generator_api.continuous_generator') as mock_gen:
            mock_gen.add_project.return_value = "test-project-123"
            
            response = client.post(
                "/api/v1/generate/batch",
                json={
                    "projects": [
                        {
                            "description": "First test AI project with machine learning capabilities",
                            "project_name": "project1"
                        },
                        {
                            "description": "Second test AI project with deep learning features",
                            "project_name": "project2"
                        }
                    ],
                    "parallel": True
                }
            )
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert len(data["results"]) == 2

    def test_rate_limit_info(self, client):
        """Test rate limit information endpoint"""
        response = client.get("/api/v1/rate-limit")
        assert response.status_code == 200

    def test_cache_stats(self, client):
        """Test cache statistics endpoint"""
        with patch('ai_project_generator.api.generator_api.cache_manager') as mock_cache:
            mock_cache.get_stats.return_value = {
                "hits": 10,
                "misses": 5,
                "size": 15
            }
            
            response = client.get("/api/v1/cache/stats")
            assert response.status_code == 200
            data = response.json()
            assert "hits" in data

    def test_clear_cache(self, client):
        """Test clearing cache"""
        with patch('ai_project_generator.api.generator_api.cache_manager') as mock_cache:
            mock_cache.clear_cache = AsyncMock()
            
            response = client.post("/api/v1/cache/clear")
            assert response.status_code == 200

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200

    def test_search_projects(self, client):
        """Test project search endpoint"""
        with patch('ai_project_generator.api.generator_api.search_engine') as mock_search:
            mock_search.search.return_value = {
                "results": [
                    {"id": "proj-1", "name": "test1"},
                    {"id": "proj-2", "name": "test2"}
                ],
                "total": 2
            }
            
            response = client.get("/api/v1/search?query=test")
            assert response.status_code == 200
            data = response.json()
            assert "results" in data

    def test_templates_list(self, client):
        """Test listing templates"""
        with patch('ai_project_generator.api.generator_api.template_manager') as mock_template:
            mock_template.list_templates.return_value = [
                {"name": "template1", "description": "Test template"}
            ]
            
            response = client.get("/api/v1/templates/list")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)

    def test_webhooks_list(self, client):
        """Test listing webhooks"""
        with patch('ai_project_generator.api.generator_api.webhook_manager') as mock_webhook:
            mock_webhook.list_webhooks.return_value = [
                {"id": "wh-1", "url": "http://example.com/webhook"}
            ]
            
            response = client.get("/api/v1/webhooks")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)

    def test_system_info(self, client):
        """Test system info endpoint"""
        response = client.get("/api/v1/system/info")
        assert response.status_code == 200
        data = response.json()
        assert "platform" in data or "system" in data or "version" in data

