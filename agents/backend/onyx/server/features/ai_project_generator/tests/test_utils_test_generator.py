"""
Tests for TestGenerator utility
"""

import pytest
import asyncio
from pathlib import Path

from ..utils.test_generator import TestGenerator


class TestTestGenerator:
    """Test suite for TestGenerator"""

    def test_init(self):
        """Test TestGenerator initialization"""
        generator = TestGenerator()
        assert generator is not None

    @pytest.mark.asyncio
    async def test_generate_backend_tests(self, temp_dir, sample_keywords, sample_project_info):
        """Test generating backend tests"""
        generator = TestGenerator()
        project_dir = temp_dir / "backend_tests"
        (project_dir / "main.py").write_text("from fastapi import FastAPI\napp = FastAPI()")
        
        await generator.generate_backend_tests(project_dir, sample_keywords, sample_project_info)
        
        tests_dir = project_dir / "tests"
        assert tests_dir.exists()
        assert (tests_dir / "test_main.py").exists()
        
        # Verify test content
        test_content = (tests_dir / "test_main.py").read_text()
        assert "test_root" in test_content
        assert "test_health" in test_content
        assert "TestClient" in test_content

    @pytest.mark.asyncio
    async def test_generate_backend_tests_with_ai(self, temp_dir, sample_keywords, sample_project_info):
        """Test generating backend tests with AI endpoints"""
        generator = TestGenerator()
        project_dir = temp_dir / "backend_ai_tests"
        (project_dir / "main.py").write_text("from fastapi import FastAPI\napp = FastAPI()")
        
        sample_keywords["ai_type"] = "chat"
        
        await generator.generate_backend_tests(project_dir, sample_keywords, sample_project_info)
        
        tests_dir = project_dir / "tests"
        assert (tests_dir / "test_ai_endpoints.py").exists()
        
        test_content = (tests_dir / "test_ai_endpoints.py").read_text()
        assert "test_ai_status" in test_content
        assert "test_ai_process" in test_content

    @pytest.mark.asyncio
    async def test_generate_frontend_tests(self, temp_dir, sample_keywords, sample_project_info):
        """Test generating frontend tests"""
        generator = TestGenerator()
        project_dir = temp_dir / "frontend_tests"
        (project_dir / "src").mkdir(parents=True)
        (project_dir / "package.json").write_text('{"name": "test"}')
        
        await generator.generate_frontend_tests(project_dir, sample_keywords, sample_project_info)
        
        # Check for test files
        src_dir = project_dir / "src"
        test_files = list(src_dir.rglob("*.test.tsx")) + list(src_dir.rglob("*.test.ts"))
        assert len(test_files) > 0 or (project_dir / "src" / "__tests__").exists()

    @pytest.mark.asyncio
    async def test_generate_backend_tests_creates_directory(self, temp_dir, sample_keywords, sample_project_info):
        """Test that test directory is created if it doesn't exist"""
        generator = TestGenerator()
        project_dir = temp_dir / "new_backend"
        project_dir.mkdir()
        
        await generator.generate_backend_tests(project_dir, sample_keywords, sample_project_info)
        
        assert (project_dir / "tests").exists()

    @pytest.mark.asyncio
    async def test_generate_backend_tests_with_websocket(self, temp_dir, sample_keywords, sample_project_info):
        """Test generating tests for WebSocket endpoints"""
        generator = TestGenerator()
        project_dir = temp_dir / "websocket_tests"
        (project_dir / "main.py").write_text("from fastapi import FastAPI\napp = FastAPI()")
        
        sample_keywords["requires_websocket"] = True
        
        await generator.generate_backend_tests(project_dir, sample_keywords, sample_project_info)
        
        tests_dir = project_dir / "tests"
        # Should have WebSocket tests
        test_files = list(tests_dir.glob("*.py"))
        assert len(test_files) > 0

