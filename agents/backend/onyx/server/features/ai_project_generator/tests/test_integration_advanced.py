"""
Advanced integration tests
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch

from ..core.project_generator import ProjectGenerator
from ..core.backend_generator import BackendGenerator
from ..core.frontend_generator import FrontendGenerator
from ..utils.cache_manager import CacheManager
from ..utils.validator import ProjectValidator
from ..utils.rate_limiter import RateLimiter


class TestAdvancedIntegration:
    """Advanced integration test suite"""

    @pytest.mark.asyncio
    async def test_full_workflow_with_cache(self, temp_dir):
        """Test full workflow with caching"""
        generator = ProjectGenerator(output_dir=temp_dir)
        cache = CacheManager(cache_dir=temp_dir / "cache")
        validator = ProjectValidator()
        
        description = "A chat AI with authentication"
        
        # First generation
        project1 = await generator.generate_project(description)
        assert project1 is not None
        
        # Cache the project
        await cache.cache_project(description, {}, {"project_id": project1["project_id"]})
        
        # Second generation should use cache
        cached = await cache.get_cached_project(description, {})
        assert cached is not None

    @pytest.mark.asyncio
    async def test_full_workflow_with_rate_limiting(self, temp_dir):
        """Test full workflow with rate limiting"""
        generator = ProjectGenerator(output_dir=temp_dir)
        limiter = RateLimiter()
        
        description = "A vision AI system"
        
        # Multiple requests
        for i in range(5):
            allowed, info = limiter.is_allowed(f"client-{i}", "generate")
            if allowed:
                project = await generator.generate_project(description)
                assert project is not None

    @pytest.mark.asyncio
    async def test_full_workflow_with_validation(self, temp_dir):
        """Test full workflow with validation"""
        generator = ProjectGenerator(output_dir=temp_dir)
        validator = ProjectValidator()
        
        description = "A complete AI project with all features"
        
        project = await generator.generate_project(description)
        assert project is not None
        
        project_path = Path(project["project_path"])
        project_info = {"name": project["name"]}
        
        # Validate generated project
        validation = await validator.validate_project(project_path, project_info)
        assert validation["valid"] is True

    @pytest.mark.asyncio
    async def test_concurrent_generation(self, temp_dir):
        """Test concurrent project generation"""
        generator = ProjectGenerator(output_dir=temp_dir)
        
        descriptions = [
            "A chat AI",
            "A vision AI",
            "An audio AI",
        ]
        
        # Generate concurrently
        tasks = [generator.generate_project(desc) for desc in descriptions]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert all(r is not None for r in results)

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, temp_dir):
        """Test error recovery in workflow"""
        generator = ProjectGenerator(output_dir=temp_dir)
        
        # Try with invalid input
        try:
            project = await generator.generate_project("")
            # Should handle gracefully
        except Exception:
            # Expected to fail, but should not crash system
            pass
        
        # Should still work after error
        project = await generator.generate_project("A valid project")
        assert project is not None

    @pytest.mark.asyncio
    async def test_backend_frontend_integration(self, temp_dir):
        """Test backend and frontend integration"""
        backend_gen = BackendGenerator(output_dir=temp_dir / "backend")
        frontend_gen = FrontendGenerator(output_dir=temp_dir / "frontend")
        
        # Generate both
        backend = await backend_gen.generate_backend("fastapi", ["auth", "database"])
        frontend = await frontend_gen.generate_frontend("react", ["auth"])
        
        assert backend is not None
        assert frontend is not None
        
        # Both should exist
        assert Path(backend["project_path"]).exists()
        assert Path(frontend["project_path"]).exists()

