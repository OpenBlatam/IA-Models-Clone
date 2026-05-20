"""
Regression tests for AI Project Generator
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock

from ..core.project_generator import ProjectGenerator
from ..core.continuous_generator import ContinuousGenerator


class TestRegression:
    """Test suite for regression testing"""

    @pytest.mark.asyncio
    async def test_project_generation_consistency(self, project_generator, temp_dir):
        """Test that project generation is consistent across runs"""
        description = "A consistent test project"
        
        with patch.object(project_generator.backend_generator, 'generate', new_callable=AsyncMock) as mock_backend, \
             patch.object(project_generator.frontend_generator, 'generate', new_callable=AsyncMock) as mock_frontend, \
             patch.object(project_generator.test_generator, 'generate_backend_tests', new_callable=AsyncMock), \
             patch.object(project_generator.test_generator, 'generate_frontend_tests', new_callable=AsyncMock), \
             patch.object(project_generator.cicd_generator, 'generate_github_actions', new_callable=AsyncMock), \
             patch.object(project_generator.cache_manager, 'get_cached_project', new_callable=AsyncMock, return_value=None), \
             patch.object(project_generator.cache_manager, 'cache_project', new_callable=AsyncMock), \
             patch.object(project_generator.validator, 'validate_project', new_callable=AsyncMock, return_value={"valid": True}):
            
            mock_backend.return_value = {"framework": "fastapi"}
            mock_frontend.return_value = {"framework": "react"}
            
            # Generate project twice
            result1 = await project_generator.generate_project(
                description=description,
                project_name="consistent_test"
            )
            
            result2 = await project_generator.generate_project(
                description=description,
                project_name="consistent_test"
            )
            
            # Keywords should be consistent
            assert result1["keywords"]["ai_type"] == result2["keywords"]["ai_type"]
            assert result1["keywords"]["complexity"] == result2["keywords"]["complexity"]

    def test_keyword_extraction_consistency(self, project_generator):
        """Test that keyword extraction is consistent"""
        description = "A chat AI system with authentication"
        
        keywords1 = project_generator._extract_keywords(description)
        keywords2 = project_generator._extract_keywords(description)
        
        # Should produce same results
        assert keywords1["ai_type"] == keywords2["ai_type"]
        assert keywords1["requires_auth"] == keywords2["requires_auth"]
        assert keywords1["features"] == keywords2["features"]

    def test_sanitize_name_consistency(self, project_generator):
        """Test that name sanitization is consistent"""
        test_names = [
            "Test Project",
            "Test-Project",
            "Test@Project#123"
        ]
        
        for name in test_names:
            result1 = project_generator._sanitize_name(name)
            result2 = project_generator._sanitize_name(name)
            
            # Should be consistent
            assert result1 == result2

    @pytest.mark.asyncio
    async def test_cache_consistency(self, temp_dir):
        """Test that cache produces consistent results"""
        from ..utils.cache_manager import CacheManager
        
        manager = CacheManager(cache_dir=temp_dir / "cache")
        
        description = "Consistent cache test"
        config = {"framework": "fastapi"}
        project_info = {"project_id": "cache-test-123"}
        
        # Cache project
        await manager.cache_project(description, config, project_info)
        
        # Retrieve multiple times
        cached1 = await manager.get_cached_project(description, config)
        cached2 = await manager.get_cached_project(description, config)
        
        # Should be consistent
        assert cached1 == cached2
        assert cached1["project_id"] == "cache-test-123"

    def test_continuous_generator_queue_persistence(self, temp_dir):
        """Test that queue persists correctly"""
        from ..core.continuous_generator import ContinuousGenerator
        import json
        
        queue_file = temp_dir / "persist_queue.json"
        
        # Create generator and add projects
        generator1 = ContinuousGenerator(
            base_output_dir=str(temp_dir / "projects"),
            queue_file=str(queue_file)
        )
        
        generator1.add_project("Project 1", priority=5)
        generator1.add_project("Project 2", priority=3)
        generator1._save_queue()
        
        # Create new generator instance
        generator2 = ContinuousGenerator(
            base_output_dir=str(temp_dir / "projects"),
            queue_file=str(queue_file)
        )
        
        # Should load queue
        assert len(generator2.queue) == 2

    @pytest.mark.asyncio
    async def test_project_structure_consistency(self, temp_dir):
        """Test that generated project structure is consistent"""
        from ..core.backend_generator import BackendGenerator
        from ..core.frontend_generator import FrontendGenerator
        
        backend_gen = BackendGenerator()
        frontend_gen = FrontendGenerator()
        
        keywords = {
            "ai_type": "chat",
            "requires_websocket": True,
            "requires_auth": False,
            "requires_database": False
        }
        
        project_info = {
            "name": "structure_test",
            "version": "1.0.0",
            "author": "Test"
        }
        
        # Generate backend
        backend_dir = temp_dir / "backend_test"
        await backend_gen.generate(backend_dir, "Test", keywords, project_info)
        
        # Generate frontend
        frontend_dir = temp_dir / "frontend_test"
        await frontend_gen.generate(frontend_dir, "Test", keywords, project_info)
        
        # Structure should be consistent
        assert (backend_dir / "app" / "api").exists()
        assert (backend_dir / "app" / "core").exists()
        assert (frontend_dir / "src" / "components").exists()
        assert (frontend_dir / "src" / "pages").exists()

    def test_rate_limiter_consistency(self):
        """Test that rate limiter behaves consistently"""
        from ..utils.rate_limiter import RateLimiter
        
        limiter = RateLimiter()
        limiter.limits["default"] = {"requests": 10, "window": 3600}
        
        # Make requests
        results = []
        for i in range(10):
            allowed, info = limiter.is_allowed("test_client")
            results.append(allowed)
        
        # All should be allowed
        assert all(results)
        
        # Next should be blocked
        allowed, _ = limiter.is_allowed("test_client")
        assert allowed is False

    @pytest.mark.asyncio
    async def test_validator_consistency(self, temp_dir):
        """Test that validator produces consistent results"""
        from ..utils.validator import ProjectValidator
        
        validator = ProjectValidator()
        
        # Create valid project
        project_dir = temp_dir / "valid_regression"
        (project_dir / "backend" / "app" / "api").mkdir(parents=True)
        (project_dir / "backend" / "app" / "core").mkdir(parents=True)
        (project_dir / "frontend" / "src").mkdir(parents=True)
        (project_dir / "backend" / "main.py").write_text("# Main")
        (project_dir / "backend" / "requirements.txt").write_text("fastapi")
        (project_dir / "frontend" / "package.json").write_text('{"name": "test"}')
        (project_dir / "README.md").write_text("# Test")
        
        project_info = {"name": "test"}
        
        # Validate multiple times
        result1 = await validator.validate_project(project_dir, project_info)
        result2 = await validator.validate_project(project_dir, project_info)
        
        # Should be consistent
        assert result1["valid"] == result2["valid"]
        assert len(result1["errors"]) == len(result2["errors"])

