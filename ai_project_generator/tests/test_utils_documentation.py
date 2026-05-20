"""
Tests for AutoDocumentation utility
"""

import pytest
from pathlib import Path

from ..utils.auto_documentation import AutoDocumentation


class TestAutoDocumentation:
    """Test suite for AutoDocumentation"""

    def test_init(self):
        """Test AutoDocumentation initialization"""
        doc_gen = AutoDocumentation()
        assert doc_gen is not None

    def test_generate_readme(self, temp_dir):
        """Test generating README"""
        doc_gen = AutoDocumentation()
        
        project_info = {
            "name": "Test Project",
            "description": "A test AI project",
            "ai_type": "chat",
            "backend_framework": "fastapi",
            "frontend_framework": "react",
            "author": "Test Author",
            "version": "1.0.0",
            "features": ["auth", "database"]
        }
        
        output_path = temp_dir / "README.md"
        readme = doc_gen.generate_readme(project_info, output_path)
        
        assert isinstance(readme, str)
        assert "Test Project" in readme
        assert "chat" in readme
        assert "fastapi" in readme
        assert "react" in readme
        assert output_path.exists()

    def test_generate_readme_minimal(self, temp_dir):
        """Test generating README with minimal info"""
        doc_gen = AutoDocumentation()
        
        project_info = {
            "name": "Minimal Project"
        }
        
        output_path = temp_dir / "README.md"
        readme = doc_gen.generate_readme(project_info, output_path)
        
        assert isinstance(readme, str)
        assert "Minimal Project" in readme

    def test_generate_api_docs(self, temp_dir):
        """Test generating API documentation"""
        doc_gen = AutoDocumentation()
        
        api_endpoints = [
            {"path": "/api/v1/generate", "method": "POST", "description": "Generate project"},
            {"path": "/api/v1/status", "method": "GET", "description": "Get status"},
        ]
        
        output_path = temp_dir / "API.md"
        api_docs = doc_gen.generate_api_docs(api_endpoints, output_path)
        
        assert isinstance(api_docs, str)
        assert "/api/v1/generate" in api_docs
        assert output_path.exists()

    def test_generate_architecture_docs(self, temp_dir):
        """Test generating architecture documentation"""
        doc_gen = AutoDocumentation()
        
        architecture_info = {
            "backend": "FastAPI",
            "frontend": "React",
            "database": "PostgreSQL",
            "cache": "Redis"
        }
        
        output_path = temp_dir / "ARCHITECTURE.md"
        arch_docs = doc_gen.generate_architecture_docs(architecture_info, output_path)
        
        assert isinstance(arch_docs, str)
        assert "FastAPI" in arch_docs
        assert output_path.exists()

