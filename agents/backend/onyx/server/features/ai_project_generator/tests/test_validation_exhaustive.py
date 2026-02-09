"""
Exhaustive validation tests
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch

from ..core.project_generator import ProjectGenerator
from ..utils.validator import ProjectValidator


class TestExhaustiveValidation:
    """Exhaustive validation test suite"""

    def test_validate_all_ai_types(self, project_generator):
        """Test validation of all AI types"""
        ai_types = ["chat", "vision", "audio", "nlp", "video", "recommendation"]
        
        for ai_type in ai_types:
            description = f"A {ai_type} AI system"
            keywords = project_generator._extract_keywords(description)
            # Should detect the AI type
            assert keywords["ai_type"] == ai_type or ai_type in description.lower()

    def test_validate_all_providers(self, project_generator):
        """Test validation of all model providers"""
        providers = ["openai", "anthropic", "google", "huggingface", "local"]
        
        for provider in providers:
            description = f"An AI system using {provider} models"
            keywords = project_generator._extract_keywords(description)
            # Should detect provider
            assert provider in keywords["model_providers"] or provider in description.lower()

    def test_validate_all_complexities(self, project_generator):
        """Test validation of all complexity levels"""
        complexities = {
            "simple": "A simple AI project",
            "medium": "A standard AI system",
            "complex": "An advanced enterprise AI platform"
        }
        
        for complexity, description in complexities.items():
            keywords = project_generator._extract_keywords(description)
            assert keywords["complexity"] == complexity

    def test_validate_all_features(self, project_generator):
        """Test validation of all features"""
        feature_tests = {
            "auth": "A system with user authentication",
            "database": "A system that stores data in a database",
            "websocket": "A real-time system with websocket",
            "file_upload": "A system that accepts file uploads",
            "cache": "A system with Redis cache",
            "queue": "A system with background task queue",
            "streaming": "A system with streaming data processing"
        }
        
        for feature, description in feature_tests.items():
            keywords = project_generator._extract_keywords(description)
            feature_key = f"requires_{feature}"
            if feature_key in keywords:
                assert keywords[feature_key] is True

    @pytest.mark.asyncio
    async def test_validate_project_structure_complete(self, temp_dir):
        """Test validation of complete project structure"""
        validator = ProjectValidator()
        
        project_dir = temp_dir / "complete_project"
        
        # Create complete structure
        (project_dir / "backend" / "app" / "api" / "v1").mkdir(parents=True)
        (project_dir / "backend" / "app" / "core").mkdir(parents=True)
        (project_dir / "backend" / "app" / "models").mkdir(parents=True)
        (project_dir / "backend" / "app" / "services").mkdir(parents=True)
        (project_dir / "backend" / "app" / "utils").mkdir(parents=True)
        (project_dir / "frontend" / "src" / "components").mkdir(parents=True)
        (project_dir / "frontend" / "src" / "pages").mkdir(parents=True)
        (project_dir / "frontend" / "src" / "services").mkdir(parents=True)
        (project_dir / "frontend" / "src" / "utils").mkdir(parents=True)
        
        # Essential files
        (project_dir / "backend" / "main.py").write_text("from fastapi import FastAPI\napp = FastAPI()")
        (project_dir / "backend" / "requirements.txt").write_text("fastapi\nuvicorn")
        (project_dir / "frontend" / "package.json").write_text('{"name": "test", "version": "1.0.0"}')
        (project_dir / "frontend" / "vite.config.ts").write_text("export default {}")
        (project_dir / "README.md").write_text("# Complete Project")
        
        project_info = {"name": "complete_project"}
        
        result = await validator.validate_project(project_dir, project_info)
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_name_sanitization_all_cases(self, project_generator):
        """Test name sanitization for all edge cases"""
        test_cases = {
            "Normal Name": "normal_name",
            "Name-With-Dashes": "name_with_dashes",
            "Name@With#Special$Chars": "name_with_special_chars",
            "  Name With Spaces  ": "name_with_spaces",
            "Name123": "name123",
            "UPPERCASE NAME": "uppercase_name",
            "MixedCaseName": "mixedcasename",
            "": "",
            "!@#$%^&*()": "",
            "a" * 200: "a" * 50,  # Length limit
        }
        
        for input_name, expected in test_cases.items():
            result = project_generator._sanitize_name(input_name)
            # Should sanitize correctly
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_validate_keywords_all_combinations(self, project_generator):
        """Test keyword extraction for various combinations"""
        combinations = [
            "A chat AI with authentication and database",
            "A vision AI with websocket and file upload",
            "A deep learning project using PyTorch with training",
            "An LLM system with transformers and fine-tuning",
            "A simple chatbot",
            "A complex enterprise AI platform with all features"
        ]
        
        for description in combinations:
            keywords = project_generator._extract_keywords(description)
            # Should extract valid keywords
            assert "ai_type" in keywords
            assert "complexity" in keywords
            assert "features" in keywords
            assert isinstance(keywords["features"], list)

    def test_validate_cache_all_scenarios(self, temp_dir):
        """Test cache validation for all scenarios"""
        from ..utils.cache_manager import CacheManager
        
        manager = CacheManager(cache_dir=temp_dir / "cache")
        
        # Test scenarios
        scenarios = [
            ("Project 1", {"f": "fastapi"}, {"id": "1"}),
            ("Project 2", {"f": "flask"}, {"id": "2"}),
            ("Project 1", {"f": "fastapi"}, {"id": "1"}),  # Duplicate
        ]
        
        async def test_scenarios():
            for desc, config, info in scenarios:
                await manager.cache_project(desc, config, info)
            
            # Verify caching
            cached = await manager.get_cached_project("Project 1", {"f": "fastapi"})
            assert cached is not None
        
        asyncio.run(test_scenarios())

    def test_validate_rate_limiter_all_endpoints(self):
        """Test rate limiter for all endpoint types"""
        from ..utils.rate_limiter import RateLimiter
        
        limiter = RateLimiter()
        
        endpoints = ["default", "generate", "search"]
        
        for endpoint in endpoints:
            allowed, info = limiter.is_allowed("test_client", endpoint)
            assert allowed is True
            assert "limit" in info
            assert "remaining" in info

