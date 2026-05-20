"""
Tests for edge cases and boundary conditions
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock

from ..core.project_generator import ProjectGenerator
from ..core.continuous_generator import ContinuousGenerator


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions"""

    def test_sanitize_name_empty(self, project_generator):
        """Test sanitizing empty name"""
        result = project_generator._sanitize_name("")
        assert result == ""

    def test_sanitize_name_special_chars_only(self, project_generator):
        """Test sanitizing name with only special characters"""
        result = project_generator._sanitize_name("!!!@@@###")
        assert result == ""

    def test_sanitize_name_very_long(self, project_generator):
        """Test sanitizing very long name"""
        long_name = "a" * 200
        result = project_generator._sanitize_name(long_name)
        assert len(result) == 50  # Should be limited

    def test_extract_keywords_empty_description(self, project_generator):
        """Test extracting keywords from empty description"""
        keywords = project_generator._extract_keywords("")
        assert keywords["ai_type"] == "general"
        assert keywords["complexity"] == "medium"

    def test_extract_keywords_very_long_description(self, project_generator):
        """Test extracting keywords from very long description"""
        long_desc = "chat " * 1000
        keywords = project_generator._extract_keywords(long_desc)
        assert keywords["ai_type"] == "chat"

    def test_extract_keywords_multiple_ai_types(self, project_generator):
        """Test extracting keywords when multiple AI types are mentioned"""
        desc = "A chat system with vision capabilities and audio processing"
        keywords = project_generator._extract_keywords(desc)
        # Should detect the first or most prominent
        assert keywords["ai_type"] in ["chat", "vision", "audio"]

    @pytest.mark.asyncio
    async def test_generate_project_invalid_name(self, project_generator, temp_dir):
        """Test generating project with invalid name"""
        with patch.object(project_generator.cache_manager, 'get_cached_project', 
                         new_callable=AsyncMock, return_value=None):
            # Name with invalid characters should be sanitized
            result = await project_generator.generate_project(
                description="A test project with many features and capabilities",
                project_name="!!!invalid!!!name!!!"
            )
            assert result["name"] != "!!!invalid!!!name!!!"
            assert "_" not in result["name"] or result["name"].replace("_", "").isalnum()

    @pytest.mark.asyncio
    async def test_generate_project_concurrent_requests(self, project_generator, temp_dir):
        """Test handling concurrent project generation requests"""
        description = "A concurrent test project"
        
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
            
            # Generate multiple projects concurrently
            tasks = [
                project_generator.generate_project(description=description, project_name=f"concurrent_{i}")
                for i in range(5)
            ]
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 5
            assert all("project_id" in r for r in results)

    def test_continuous_generator_empty_queue(self, continuous_generator):
        """Test continuous generator with empty queue"""
        queue_info = continuous_generator.get_queue()
        assert queue_info["total"] == 0
        assert queue_info["pending"] == 0

    def test_continuous_generator_high_priority(self, continuous_generator):
        """Test adding projects with very high priority"""
        id1 = continuous_generator.add_project("Low", priority=-10)
        id2 = continuous_generator.add_project("High", priority=100)
        id3 = continuous_generator.add_project("Medium", priority=0)
        
        # Should be sorted by priority (descending)
        assert continuous_generator.queue[0]["priority"] == 100
        assert continuous_generator.queue[-1]["priority"] == -10

    def test_continuous_generator_duplicate_names(self, continuous_generator):
        """Test adding projects with duplicate names"""
        id1 = continuous_generator.add_project("Test", project_name="duplicate")
        id2 = continuous_generator.add_project("Test 2", project_name="duplicate")
        
        assert len(continuous_generator.queue) == 2
        assert id1 != id2

    @pytest.mark.asyncio
    async def test_cache_manager_concurrent_access(self, temp_dir):
        """Test cache manager with concurrent access"""
        from ..utils.cache_manager import CacheManager
        
        manager = CacheManager(cache_dir=temp_dir / "cache")
        
        description = "Concurrent cache test"
        config = {"framework": "fastapi"}
        project_info = {"project_id": "concurrent-123"}
        
        # Cache multiple projects concurrently
        tasks = [
            manager.cache_project(description, config, {**project_info, "id": i})
            for i in range(10)
        ]
        await asyncio.gather(*tasks)
        
        # All should be cached
        stats = await manager.get_stats()
        assert stats["total_cached"] == 1  # Same description/config = same cache key

    def test_rate_limiter_concurrent_requests(self):
        """Test rate limiter with concurrent requests"""
        from ..utils.rate_limiter import RateLimiter
        
        limiter = RateLimiter()
        limiter.limits["default"] = {"requests": 100, "window": 3600}
        
        # Make many concurrent requests
        results = []
        for i in range(50):
            allowed, info = limiter.is_allowed(f"client_{i % 10}")
            results.append(allowed)
        
        # All should be allowed (different clients)
        assert all(results)

    def test_validator_empty_project(self, temp_dir):
        """Test validating empty project directory"""
        from ..utils.validator import ProjectValidator
        
        validator = ProjectValidator()
        project_dir = temp_dir / "empty_project"
        project_dir.mkdir()
        
        result = asyncio.run(validator.validate_project(project_dir, {"name": "test"}))
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_validator_malformed_files(self, temp_dir):
        """Test validating project with malformed files"""
        from ..utils.validator import ProjectValidator
        
        validator = ProjectValidator()
        project_dir = temp_dir / "malformed"
        (project_dir / "backend" / "app" / "api").mkdir(parents=True)
        (project_dir / "frontend" / "src").mkdir(parents=True)
        (project_dir / "backend" / "main.py").write_text("invalid python syntax !!!")
        (project_dir / "frontend" / "package.json").write_text("invalid json {")
        
        result = asyncio.run(validator.validate_project(project_dir, {"name": "test"}))
        # Should detect issues but not crash
        assert isinstance(result, dict)

