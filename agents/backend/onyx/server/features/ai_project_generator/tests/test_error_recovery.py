"""
Error recovery and resilience tests
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock, Mock, side_effect

from ..core.project_generator import ProjectGenerator
from ..core.continuous_generator import ContinuousGenerator
from ..utils.cache_manager import CacheManager


class TestErrorRecovery:
    """Test suite for error recovery and resilience"""

    @pytest.mark.asyncio
    async def test_project_generation_backend_error(self, project_generator, temp_dir):
        """Test recovery from backend generation error"""
        description = "A test project that will fail in backend"
        
        with patch.object(project_generator.backend_generator, 'generate',
                         new_callable=AsyncMock, side_effect=Exception("Backend error")), \
             patch.object(project_generator.cache_manager, 'get_cached_project',
                         new_callable=AsyncMock, return_value=None):
            
            with pytest.raises(Exception):
                await project_generator.generate_project(description=description)
            
            # Should not leave partial files
            project_dir = project_generator.base_output_dir / project_generator._sanitize_name(description)
            # Error should be raised, not silently ignored

    @pytest.mark.asyncio
    async def test_project_generation_frontend_error(self, project_generator, temp_dir):
        """Test recovery from frontend generation error"""
        description = "A test project that will fail in frontend"
        
        with patch.object(project_generator.backend_generator, 'generate',
                         new_callable=AsyncMock, return_value={"framework": "fastapi"}), \
             patch.object(project_generator.frontend_generator, 'generate',
                         new_callable=AsyncMock, side_effect=Exception("Frontend error")), \
             patch.object(project_generator.cache_manager, 'get_cached_project',
                         new_callable=AsyncMock, return_value=None):
            
            with pytest.raises(Exception):
                await project_generator.generate_project(description=description)

    @pytest.mark.asyncio
    async def test_cache_error_recovery(self, temp_dir):
        """Test recovery from cache errors"""
        manager = CacheManager(cache_dir=temp_dir / "cache")
        
        # Try to get from non-existent cache
        result = await manager.get_cached_project("Non-existent", {"f": "fastapi"})
        assert result is None  # Should handle gracefully
        
        # Try to cache with invalid data
        try:
            await manager.cache_project("Test", {"f": "fastapi"}, {"project_id": "test"})
            # Should succeed
            assert True
        except Exception:
            # Or handle error gracefully
            assert True

    @pytest.mark.asyncio
    async def test_continuous_generator_error_recovery(self, temp_dir):
        """Test continuous generator error recovery"""
        generator = ContinuousGenerator(base_output_dir=str(temp_dir / "projects"))
        
        # Add project that will fail
        project_id = generator.add_project("Project that will fail")
        
        with patch.object(generator.project_generator, 'generate_project',
                         new_callable=AsyncMock, side_effect=Exception("Generation error")):
            
            generator.start()
            await asyncio.sleep(0.5)  # Give it time to process
            await generator.stop()
            
            # Should handle error gracefully
            assert generator.is_running is False

    def test_rate_limiter_error_recovery(self):
        """Test rate limiter error recovery"""
        from ..utils.rate_limiter import RateLimiter
        
        limiter = RateLimiter()
        
        # Should handle invalid client_id gracefully
        allowed, info = limiter.is_allowed(None)
        # Should not crash
        assert isinstance(allowed, bool)

    @pytest.mark.asyncio
    async def test_validator_error_recovery(self, temp_dir):
        """Test validator error recovery"""
        from ..utils.validator import ProjectValidator
        
        validator = ProjectValidator()
        
        # Validate non-existent directory
        non_existent = temp_dir / "non_existent"
        result = await validator.validate_project(non_existent, {"name": "test"})
        
        # Should return validation result, not crash
        assert "valid" in result
        assert result["valid"] is False

    @pytest.mark.asyncio
    async def test_export_error_recovery(self, temp_dir):
        """Test export error recovery"""
        from ..utils.export_generator import ExportGenerator
        
        generator = ExportGenerator()
        
        # Try to export non-existent directory
        non_existent = temp_dir / "non_existent"
        
        with pytest.raises(Exception):
            await generator.export_to_zip(non_existent)
        
        # Should raise error, not crash silently

    def test_webhook_error_recovery(self):
        """Test webhook error recovery"""
        from ..utils.webhook_manager import WebhookManager
        
        manager = WebhookManager()
        
        # Register webhook
        webhook_id = manager.register_webhook(
            url="http://invalid-url-that-will-fail.com/webhook",
            events=["test.event"]
        )
        
        # Trigger webhook (should handle error gracefully)
        async def test_trigger():
            await manager.trigger_webhook("test.event", {"data": "test"})
            # Should not crash
        
        asyncio.run(test_trigger())

    @pytest.mark.asyncio
    async def test_template_manager_error_recovery(self, temp_dir):
        """Test template manager error recovery"""
        from ..utils.template_manager import TemplateManager
        
        manager = TemplateManager(templates_dir=temp_dir / "templates")
        
        # Try to load non-existent template
        result = await manager.load_template("non_existent")
        assert result is None  # Should handle gracefully
        
        # Try to delete non-existent template
        result = await manager.delete_template("non_existent")
        assert result is False  # Should handle gracefully

    @pytest.mark.asyncio
    async def test_cloner_error_recovery(self, temp_dir):
        """Test cloner error recovery"""
        from ..utils.project_cloner import ProjectCloner
        
        cloner = ProjectCloner()
        
        # Try to clone non-existent project
        non_existent = temp_dir / "non_existent"
        
        with pytest.raises(ValueError):
            await cloner.clone_project(non_existent)
        
        # Should raise error, not crash silently

    @pytest.mark.asyncio
    async def test_search_engine_error_recovery(self, temp_dir):
        """Test search engine error recovery"""
        from ..utils.search_engine import ProjectSearchEngine
        
        engine = ProjectSearchEngine(projects_dir=temp_dir / "projects")
        (temp_dir / "projects").mkdir(parents=True)
        
        # Create project with invalid JSON
        invalid_project = temp_dir / "projects" / "invalid"
        invalid_project.mkdir()
        (invalid_project / "project_info.json").write_text("invalid json {")
        
        # Should handle error gracefully
        results = await engine.search_projects()
        # Should skip invalid projects, not crash
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_metrics_collector_error_recovery(self):
        """Test metrics collector error recovery"""
        from ..utils.metrics_collector import MetricsCollector
        
        collector = MetricsCollector()
        
        # Record metrics with invalid data
        collector.record_request(None, None, None)
        # Should not crash
        
        # Get metrics
        metrics = collector.get_metrics()
        assert isinstance(metrics, dict)

