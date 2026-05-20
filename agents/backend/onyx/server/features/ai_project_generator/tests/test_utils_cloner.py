"""
Tests for ProjectCloner utility
"""

import pytest
import asyncio
import json
from pathlib import Path

from ..utils.project_cloner import ProjectCloner


class TestProjectCloner:
    """Test suite for ProjectCloner"""

    def test_init(self):
        """Test ProjectCloner initialization"""
        cloner = ProjectCloner()
        assert cloner is not None

    @pytest.mark.asyncio
    async def test_clone_project(self, temp_dir):
        """Test cloning a project"""
        cloner = ProjectCloner()
        
        # Create source project
        source_dir = temp_dir / "source_project"
        (source_dir / "backend" / "app").mkdir(parents=True)
        (source_dir / "frontend" / "src").mkdir(parents=True)
        (source_dir / "backend" / "main.py").write_text("# Main file")
        (source_dir / "README.md").write_text("# Source Project")
        
        # Create project_info.json
        project_info = {
            "name": "source_project",
            "description": "Source project",
            "author": "Test Author"
        }
        (source_dir / "project_info.json").write_text(
            json.dumps(project_info), encoding="utf-8"
        )
        
        # Clone project
        result = await cloner.clone_project(source_dir)
        
        assert result["success"] is True
        assert "target_path" in result
        target_path = Path(result["target_path"])
        assert target_path.exists()
        
        # Verify cloned files
        assert (target_path / "backend" / "main.py").exists()
        assert (target_path / "README.md").exists()
        
        # Verify project_info updated
        cloned_info = json.loads((target_path / "project_info.json").read_text())
        assert cloned_info["is_clone"] is True
        assert "cloned_from" in cloned_info
        assert "cloned_at" in cloned_info

    @pytest.mark.asyncio
    async def test_clone_project_with_new_name(self, temp_dir):
        """Test cloning project with new name"""
        cloner = ProjectCloner()
        
        source_dir = temp_dir / "original"
        source_dir.mkdir()
        (source_dir / "test.txt").write_text("test")
        
        result = await cloner.clone_project(source_dir, new_name="cloned_project")
        
        assert result["success"] is True
        target_path = Path(result["target_path"])
        assert target_path.name == "cloned_project"
        assert target_path.exists()

    @pytest.mark.asyncio
    async def test_clone_project_excludes_files(self, temp_dir):
        """Test that cloning excludes unnecessary files"""
        cloner = ProjectCloner()
        
        source_dir = temp_dir / "source_with_excludes"
        source_dir.mkdir()
        (source_dir / "test.py").write_text("# Test")
        (source_dir / "__pycache__").mkdir()
        (source_dir / "__pycache__" / "test.pyc").write_text("compiled")
        (source_dir / ".git").mkdir()
        (source_dir / ".git" / "config").write_text("git config")
        (source_dir / "node_modules").mkdir()
        (source_dir / "node_modules" / "package.json").write_text('{}')
        
        result = await cloner.clone_project(source_dir)
        target_path = Path(result["target_path"])
        
        # Should have test.py
        assert (target_path / "test.py").exists()
        # Should exclude these
        assert not (target_path / "__pycache__").exists()
        assert not (target_path / ".git").exists()
        assert not (target_path / "node_modules").exists()

    @pytest.mark.asyncio
    async def test_clone_project_not_found(self, temp_dir):
        """Test cloning non-existent project"""
        cloner = ProjectCloner()
        
        non_existent = temp_dir / "non_existent"
        
        with pytest.raises(ValueError, match="no existe"):
            await cloner.clone_project(non_existent)

    @pytest.mark.asyncio
    async def test_clone_project_update_config(self, temp_dir):
        """Test cloning with config update"""
        cloner = ProjectCloner()
        
        source_dir = temp_dir / "config_source"
        source_dir.mkdir()
        (source_dir / "package.json").write_text('{"name": "original"}')
        
        result = await cloner.clone_project(
            source_dir,
            new_name="updated",
            update_config=True
        )
        
        target_path = Path(result["target_path"])
        # Config should be updated
        assert target_path.exists()

    @pytest.mark.asyncio
    async def test_clone_project_no_update_config(self, temp_dir):
        """Test cloning without config update"""
        cloner = ProjectCloner()
        
        source_dir = temp_dir / "no_update_source"
        source_dir.mkdir()
        (source_dir / "config.json").write_text('{"name": "original"}')
        
        result = await cloner.clone_project(
            source_dir,
            update_config=False
        )
        
        assert result["success"] is True

