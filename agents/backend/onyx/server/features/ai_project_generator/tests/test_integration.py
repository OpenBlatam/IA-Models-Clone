"""
Integration tests for AI Project Generator
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock

from ..core.project_generator import ProjectGenerator
from ..core.continuous_generator import ContinuousGenerator
from ..api.generator_api import create_generator_app
from fastapi.testclient import TestClient


class TestIntegration:
    """Integration tests for the complete system"""

    @pytest.mark.asyncio
    async def test_full_project_generation_flow(self, temp_dir):
        """Test complete project generation flow"""
        generator = ProjectGenerator(base_output_dir=str(temp_dir / "projects"))
        
        description = "A complete AI chat system with authentication and database"
        
        with patch.object(generator.cache_manager, 'get_cached_project', 
                         new_callable=AsyncMock, return_value=None), \
             patch.object(generator.cache_manager, 'cache_project', 
                         new_callable=AsyncMock):
            
            result = await generator.generate_project(
                description=description,
                project_name="integration_test",
                author="Test Author"
            )
            
            assert "project_id" in result
            assert result["name"] == "integration_test"
            assert result["description"] == description

    @pytest.mark.asyncio
    async def test_continuous_generator_flow(self, temp_dir):
        """Test continuous generator with queue processing"""
        generator = ContinuousGenerator(base_output_dir=str(temp_dir / "projects"))
        
        # Add project to queue
        project_id = generator.add_project(
            description="A test AI project for integration testing",
            project_name="queue_test",
            priority=5
        )
        
        assert project_id is not None
        assert len(generator.queue) == 1
        
        # Get queue status
        queue_info = generator.get_queue()
        assert queue_info["total"] == 1
        assert queue_info["pending"] == 1
        
        # Get project status
        status = generator.get_project_status(project_id)
        assert status is not None
        assert status["status"] == "pending"

    def test_api_health_check(self, temp_dir):
        """Test API health check endpoint"""
        app = create_generator_app(
            base_output_dir=str(temp_dir / "generated_projects"),
            enable_continuous=True
        )
        client = TestClient(app)
        
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_api_generate_and_status(self, temp_dir):
        """Test API project generation and status check"""
        app = create_generator_app(
            base_output_dir=str(temp_dir / "generated_projects"),
            enable_continuous=True
        )
        client = TestClient(app)
        
        with patch('ai_project_generator.api.generator_api.continuous_generator') as mock_gen:
            mock_gen.add_project.return_value = "test-api-123"
            mock_gen.get_project_status.return_value = {
                "id": "test-api-123",
                "status": "queued",
                "description": "Test API project"
            }
            
            # Generate project
            response = client.post(
                "/api/v1/generate",
                json={
                    "description": "A comprehensive AI project for API integration testing with multiple features",
                    "project_name": "api_test"
                }
            )
            assert response.status_code == 200
            project_id = response.json()["project_id"]
            
            # Check status
            status_response = client.get(f"/api/v1/project/{project_id}")
            assert status_response.status_code == 200
            assert status_response.json()["id"] == project_id

    @pytest.mark.asyncio
    async def test_cache_integration(self, temp_dir):
        """Test cache integration with project generator"""
        from ..utils.cache_manager import CacheManager
        
        cache_manager = CacheManager(cache_dir=temp_dir / "cache")
        generator = ProjectGenerator(base_output_dir=str(temp_dir / "projects"))
        generator.cache_manager = cache_manager
        
        description = "A cached project for testing"
        config = {"framework": "fastapi"}
        project_info = {"project_id": "cache-test", "name": "cached_project"}
        
        # Cache project
        await cache_manager.cache_project(description, config, project_info)
        
        # Retrieve from cache
        cached = await cache_manager.get_cached_project(description, config)
        assert cached is not None
        assert cached["project_id"] == "cache-test"

    def test_validator_integration(self, temp_dir):
        """Test validator integration"""
        from ..utils.validator import ProjectValidator
        
        validator = ProjectValidator()
        
        # Create valid project structure
        project_dir = temp_dir / "valid_integration"
        (project_dir / "backend" / "app" / "api").mkdir(parents=True)
        (project_dir / "backend" / "app" / "core").mkdir(parents=True)
        (project_dir / "frontend" / "src").mkdir(parents=True)
        (project_dir / "backend" / "main.py").write_text("# Main")
        (project_dir / "backend" / "requirements.txt").write_text("fastapi")
        (project_dir / "frontend" / "package.json").write_text('{"name": "test"}')
        (project_dir / "README.md").write_text("# Test")
        
        project_info = {"name": "test", "description": "Test"}
        
        # Validate (async)
        import asyncio
        result = asyncio.run(validator.validate_project(project_dir, project_info))
        assert result["valid"] is True

