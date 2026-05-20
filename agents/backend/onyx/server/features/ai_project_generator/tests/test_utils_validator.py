"""
Tests for ProjectValidator utility
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch

from ..utils.validator import ProjectValidator


class TestProjectValidator:
    """Test suite for ProjectValidator"""

    def test_init(self):
        """Test ProjectValidator initialization"""
        validator = ProjectValidator()
        assert validator is not None

    @pytest.mark.asyncio
    async def test_validate_project_valid(self, temp_dir):
        """Test validation of a valid project"""
        validator = ProjectValidator()
        
        # Create valid project structure
        project_dir = temp_dir / "valid_project"
        (project_dir / "backend" / "app" / "api").mkdir(parents=True)
        (project_dir / "backend" / "app" / "core").mkdir(parents=True)
        (project_dir / "frontend" / "src").mkdir(parents=True)
        (project_dir / "backend" / "main.py").write_text("# Main file")
        (project_dir / "backend" / "requirements.txt").write_text("fastapi")
        (project_dir / "frontend" / "package.json").write_text('{"name": "test"}')
        (project_dir / "README.md").write_text("# Test Project")
        
        project_info = {
            "name": "test_project",
            "description": "Test project"
        }
        
        result = await validator.validate_project(project_dir, project_info)
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    @pytest.mark.asyncio
    async def test_validate_project_missing_structure(self, temp_dir):
        """Test validation with missing directory structure"""
        validator = ProjectValidator()
        
        project_dir = temp_dir / "invalid_project"
        project_dir.mkdir()
        # Missing backend/frontend directories
        
        project_info = {"name": "test"}
        
        result = await validator.validate_project(project_dir, project_info)
        
        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert any("backend" in error.lower() or "frontend" in error.lower() 
                  for error in result["errors"])

    @pytest.mark.asyncio
    async def test_validate_project_missing_files(self, temp_dir):
        """Test validation with missing essential files"""
        validator = ProjectValidator()
        
        project_dir = temp_dir / "missing_files_project"
        (project_dir / "backend" / "app" / "api").mkdir(parents=True)
        (project_dir / "backend" / "app" / "core").mkdir(parents=True)
        (project_dir / "frontend" / "src").mkdir(parents=True)
        # Missing main.py, requirements.txt, etc.
        
        project_info = {"name": "test"}
        
        result = await validator.validate_project(project_dir, project_info)
        
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_validate_structure(self, temp_dir):
        """Test structure validation"""
        validator = ProjectValidator()
        
        project_dir = temp_dir / "structure_test"
        (project_dir / "backend" / "app" / "api").mkdir(parents=True)
        (project_dir / "backend" / "app" / "core").mkdir(parents=True)
        (project_dir / "frontend" / "src").mkdir(parents=True)
        
        errors = validator._validate_structure(project_dir)
        assert len(errors) == 0

    def test_validate_structure_missing(self, temp_dir):
        """Test structure validation with missing directories"""
        validator = ProjectValidator()
        
        project_dir = temp_dir / "missing_structure"
        project_dir.mkdir()
        # Missing required directories
        
        errors = validator._validate_structure(project_dir)
        assert len(errors) > 0

    def test_validate_files(self, temp_dir):
        """Test file validation"""
        validator = ProjectValidator()
        
        project_dir = temp_dir / "files_test"
        (project_dir / "backend").mkdir(parents=True)
        (project_dir / "frontend").mkdir()
        (project_dir / "backend" / "main.py").write_text("# Main")
        (project_dir / "backend" / "requirements.txt").write_text("fastapi")
        (project_dir / "frontend" / "package.json").write_text('{"name": "test"}')
        (project_dir / "README.md").write_text("# Test")
        
        errors, warnings = validator._validate_files(project_dir)
        assert len(errors) == 0

    def test_validate_files_missing(self, temp_dir):
        """Test file validation with missing files"""
        validator = ProjectValidator()
        
        project_dir = temp_dir / "missing_files"
        (project_dir / "backend").mkdir(parents=True)
        (project_dir / "frontend").mkdir()
        # Missing essential files
        
        errors, warnings = validator._validate_files(project_dir)
        assert len(errors) > 0

    def test_validate_config(self, temp_dir):
        """Test configuration validation"""
        validator = ProjectValidator()
        
        project_dir = temp_dir / "config_test"
        project_dir.mkdir()
        
        project_info = {
            "name": "test_project",
            "description": "Test",
            "author": "Test Author",
            "version": "1.0.0"
        }
        
        errors = validator._validate_config(project_dir, project_info)
        # Should pass basic config validation
        assert isinstance(errors, list)

    @pytest.mark.asyncio
    async def test_validate_code(self, temp_dir):
        """Test code validation"""
        validator = ProjectValidator()
        
        project_dir = temp_dir / "code_test"
        (project_dir / "backend").mkdir(parents=True)
        (project_dir / "backend" / "main.py").write_text("""
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello"}
""")
        
        errors = await validator._validate_code(project_dir)
        # Should validate Python syntax
        assert isinstance(errors, list)

    @pytest.mark.asyncio
    async def test_validate_code_syntax_error(self, temp_dir):
        """Test code validation with syntax errors"""
        validator = ProjectValidator()
        
        project_dir = temp_dir / "syntax_error"
        (project_dir / "backend").mkdir(parents=True)
        (project_dir / "backend" / "main.py").write_text("""
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root(
    return {"message": "Hello"}  # Missing closing parenthesis
""")
        
        errors = await validator._validate_code(project_dir)
        # Should detect syntax errors
        assert isinstance(errors, list)

