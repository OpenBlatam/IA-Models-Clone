"""
Compatibility tests for AI Project Generator
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock

from ..core.project_generator import ProjectGenerator
from ..core.backend_generator import BackendGenerator
from ..core.frontend_generator import FrontendGenerator


class TestCompatibility:
    """Test suite for compatibility"""

    def test_project_generator_backward_compatibility(self, temp_dir):
        """Test that ProjectGenerator maintains backward compatibility"""
        # Old initialization style
        generator = ProjectGenerator(base_output_dir=str(temp_dir / "projects"))
        assert generator.base_output_dir.exists()
        assert generator.backend_framework == "fastapi"
        assert generator.frontend_framework == "react"

    def test_keyword_extraction_compatibility(self, project_generator):
        """Test that keyword extraction handles old and new formats"""
        # Old format descriptions
        old_descriptions = [
            "A simple chat bot",
            "An image recognition system",
            "A speech to text converter",
        ]
        
        for desc in old_descriptions:
            keywords = project_generator._extract_keywords(desc)
            # Should always return valid keywords structure
            assert "ai_type" in keywords
            assert "features" in keywords
            assert "complexity" in keywords

    @pytest.mark.asyncio
    async def test_backend_generator_framework_compatibility(self, temp_dir, sample_keywords, sample_project_info):
        """Test backend generator with different framework versions"""
        generator = BackendGenerator(framework="fastapi")
        
        project_dir = temp_dir / "compat_backend"
        
        result = await generator.generate(
            project_dir=project_dir,
            description="Test project",
            keywords=sample_keywords,
            project_info=sample_project_info
        )
        
        # Should generate valid FastAPI structure
        assert result["framework"] == "fastapi"
        assert (project_dir / "app").exists()

    @pytest.mark.asyncio
    async def test_frontend_generator_framework_compatibility(self, temp_dir, sample_keywords, sample_project_info):
        """Test frontend generator with different framework versions"""
        generator = FrontendGenerator(framework="react")
        
        project_dir = temp_dir / "compat_frontend"
        
        result = await generator.generate(
            project_dir=project_dir,
            description="Test project",
            keywords=sample_keywords,
            project_info=sample_project_info
        )
        
        # Should generate valid React structure
        assert result["framework"] == "react"
        assert (project_dir / "src").exists()

    def test_cache_format_compatibility(self, temp_dir):
        """Test cache format compatibility"""
        from ..utils.cache_manager import CacheManager
        import json
        from datetime import datetime
        
        manager = CacheManager(cache_dir=temp_dir / "cache")
        
        # Create cache in old format
        cache_key = manager._generate_cache_key("Test", {"f": "fastapi"})
        cache_file = manager.cache_dir / f"{cache_key}.json"
        
        old_format = {
            "project_info": {"project_id": "old-123"},
            "created_at": datetime.now().isoformat()
        }
        cache_file.write_text(json.dumps(old_format))
        
        # Should still be readable
        cached = asyncio.run(manager.get_cached_project("Test", {"f": "fastapi"}))
        assert cached is not None

    def test_project_info_format_compatibility(self, temp_dir):
        """Test project info format compatibility"""
        project_dir = temp_dir / "compat_project"
        project_dir.mkdir()
        
        # Old format project_info
        old_info = {
            "name": "test_project",
            "description": "Test",
            "created": "2024-01-01"  # Old date format
        }
        
        import json
        (project_dir / "project_info.json").write_text(json.dumps(old_info))
        
        # Should be readable
        info = json.loads((project_dir / "project_info.json").read_text())
        assert info["name"] == "test_project"

    @pytest.mark.asyncio
    async def test_continuous_generator_queue_format(self, temp_dir):
        """Test continuous generator queue format compatibility"""
        from ..core.continuous_generator import ContinuousGenerator
        import json
        
        queue_file = temp_dir / "compat_queue.json"
        
        # Old queue format
        old_queue = {
            "queue": [
                {"id": "old-1", "description": "Old project", "status": "pending"}
            ]
        }
        queue_file.write_text(json.dumps(old_queue))
        
        generator = ContinuousGenerator(
            base_output_dir=str(temp_dir / "projects"),
            queue_file=str(queue_file)
        )
        
        # Should load old format
        assert len(generator.queue) == 1

    def test_api_request_compatibility(self):
        """Test API request format compatibility"""
        from ..api.generator_api import ProjectRequest
        
        # Minimal request (backward compatible)
        minimal = ProjectRequest(
            description="A comprehensive AI project with machine learning capabilities"
        )
        assert minimal.description is not None
        assert minimal.author == "Blatam Academy"  # Default
        
        # Full request (new format)
        full = ProjectRequest(
            description="A comprehensive AI project with machine learning capabilities",
            project_name="test_project",
            author="Custom Author",
            priority=5,
            generate_tests=True,
            include_docker=True
        )
        assert full.project_name == "test_project"
        assert full.author == "Custom Author"
        assert full.priority == 5

    def test_keywords_structure_compatibility(self, project_generator):
        """Test that keywords structure is backward compatible"""
        # Test with minimal description
        keywords = project_generator._extract_keywords("A test project")
        
        # Should have all required fields
        required_fields = [
            "ai_type", "features", "requires_auth", "requires_database",
            "requires_api", "complexity"
        ]
        for field in required_fields:
            assert field in keywords

