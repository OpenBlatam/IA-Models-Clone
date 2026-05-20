"""
Tests for comparing different implementations and versions
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List


class TestComparison:
    """Tests for comparing implementations"""
    
    def test_compare_project_structures(self, temp_dir):
        """Test comparing two project structures"""
        from .test_utilities import TestUtilities
        
        # Create two similar projects
        project1 = temp_dir / "project1"
        project1.mkdir()
        (project1 / "README.md").write_text("# Project 1")
        (project1 / "backend").mkdir()
        (project1 / "backend" / "main.py").write_text("app1")
        
        project2 = temp_dir / "project2"
        project2.mkdir()
        (project2 / "README.md").write_text("# Project 2")
        (project2 / "backend").mkdir()
        (project2 / "backend" / "main.py").write_text("app2")
        
        # Compare files
        comparison = TestUtilities.compare_files(
            project1 / "README.md",
            project2 / "README.md"
        )
        
        assert comparison["identical"] is False
        assert len(comparison["differences"]) > 0
    
    def test_compare_identical_files(self, temp_dir):
        """Test comparing identical files"""
        from .test_utilities import TestUtilities
        
        file1 = temp_dir / "file1.txt"
        file2 = temp_dir / "file2.txt"
        
        content = "Same content"
        file1.write_text(content)
        file2.write_text(content)
        
        comparison = TestUtilities.compare_files(file1, file2)
        
        assert comparison["identical"] is True
        assert len(comparison["differences"]) == 0
    
    def test_compare_project_versions(self, temp_dir):
        """Test comparing different versions of a project"""
        from ..utils.project_versioning import ProjectVersioning
        
        versioning = ProjectVersioning(versions_dir=temp_dir / "versions")
        
        # Create project
        project_path = temp_dir / "project"
        project_path.mkdir()
        (project_path / "file.txt").write_text("version 1")
        
        # Create versions
        versioning.create_version("test-proj", project_path, "1.0.0")
        
        # Modify project
        (project_path / "file.txt").write_text("version 2")
        versioning.create_version("test-proj", project_path, "2.0.0")
        
        # Compare versions
        comparison = versioning.compare_versions("test-proj", "1.0.0", "2.0.0")
        
        assert comparison is not None
        assert comparison["version1"] == "1.0.0"
        assert comparison["version2"] == "2.0.0"
        assert comparison["version1_hash"] != comparison["version2_hash"]

