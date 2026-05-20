"""
Tests for ProjectVersioning utility
"""

import pytest
import json
import shutil
from pathlib import Path
from datetime import datetime

from ..utils.project_versioning import ProjectVersioning


class TestProjectVersioning:
    """Test suite for ProjectVersioning"""

    def test_init(self, temp_dir):
        """Test ProjectVersioning initialization"""
        versions_dir = temp_dir / "versions"
        versioning = ProjectVersioning(versions_dir=versions_dir)
        assert versioning.versions_dir == versions_dir
        assert versions_dir.exists()

    def test_init_default_dir(self):
        """Test ProjectVersioning with default directory"""
        versioning = ProjectVersioning()
        assert versioning.versions_dir.exists()

    def test_create_version(self, temp_dir):
        """Test creating a project version"""
        versioning = ProjectVersioning(versions_dir=temp_dir / "versions")
        
        # Create test project
        project_path = temp_dir / "test_project"
        project_path.mkdir()
        (project_path / "README.md").write_text("# Test Project")
        (project_path / "main.py").write_text("# Main file")
        
        version_info = versioning.create_version(
            project_id="test-123",
            project_path=project_path,
            version="1.0.0",
            description="Initial version"
        )
        
        assert version_info["project_id"] == "test-123"
        assert version_info["version"] == "1.0.0"
        assert version_info["description"] == "Initial version"
        assert "project_hash" in version_info
        
        # Verify version directory exists
        version_dir = versioning.versions_dir / "test-123" / "1.0.0"
        assert version_dir.exists()
        assert (version_dir / "project" / "README.md").exists()

    def test_create_version_with_metadata(self, temp_dir):
        """Test creating version with metadata"""
        versioning = ProjectVersioning(versions_dir=temp_dir / "versions")
        
        project_path = temp_dir / "project"
        project_path.mkdir()
        
        metadata = {"author": "Test Author", "tags": ["ai", "ml"]}
        version_info = versioning.create_version(
            project_id="test-456",
            project_path=project_path,
            version="1.0.0",
            metadata=metadata
        )
        
        assert version_info["metadata"] == metadata

    def test_list_versions(self, temp_dir):
        """Test listing project versions"""
        versioning = ProjectVersioning(versions_dir=temp_dir / "versions")
        
        project_path = temp_dir / "project"
        project_path.mkdir()
        
        # Create multiple versions
        versioning.create_version("test-789", project_path, "1.0.0")
        versioning.create_version("test-789", project_path, "1.1.0")
        versioning.create_version("test-789", project_path, "2.0.0")
        
        versions = versioning.list_versions("test-789")
        
        assert len(versions) == 3
        version_numbers = [v["version"] for v in versions]
        assert "1.0.0" in version_numbers
        assert "1.1.0" in version_numbers
        assert "2.0.0" in version_numbers

    def test_get_version(self, temp_dir):
        """Test getting specific version"""
        versioning = ProjectVersioning(versions_dir=temp_dir / "versions")
        
        project_path = temp_dir / "project"
        project_path.mkdir()
        (project_path / "file.txt").write_text("content")
        
        versioning.create_version("test-get", project_path, "1.0.0", "Test version")
        
        version_info = versioning.get_version("test-get", "1.0.0")
        
        assert version_info is not None
        assert version_info["version"] == "1.0.0"
        assert version_info["description"] == "Test version"

    def test_get_version_not_found(self, temp_dir):
        """Test getting non-existent version"""
        versioning = ProjectVersioning(versions_dir=temp_dir / "versions")
        
        version_info = versioning.get_version("non-existent", "1.0.0")
        
        assert version_info is None

    def test_restore_version(self, temp_dir):
        """Test restoring a version"""
        versioning = ProjectVersioning(versions_dir=temp_dir / "versions")
        
        # Create project and version
        project_path = temp_dir / "original"
        project_path.mkdir()
        (project_path / "original.txt").write_text("original content")
        
        versioning.create_version("restore-test", project_path, "1.0.0")
        
        # Restore to new location
        restore_path = temp_dir / "restored"
        result = versioning.restore_version("restore-test", "1.0.0", restore_path)
        
        assert result["success"] is True
        assert (restore_path / "original.txt").exists()
        assert (restore_path / "original.txt").read_text() == "original content"

    def test_calculate_hash(self, temp_dir):
        """Test hash calculation"""
        versioning = ProjectVersioning(versions_dir=temp_dir / "versions")
        
        project_path = temp_dir / "hash_test"
        project_path.mkdir()
        (project_path / "file1.txt").write_text("content1")
        (project_path / "file2.txt").write_text("content2")
        
        hash1 = versioning._calculate_hash(project_path)
        hash2 = versioning._calculate_hash(project_path)
        
        # Same project should have same hash
        assert hash1 == hash2
        
        # Modify project
        (project_path / "file3.txt").write_text("content3")
        hash3 = versioning._calculate_hash(project_path)
        
        # Different content should have different hash
        assert hash1 != hash3

    def test_compare_versions(self, temp_dir):
        """Test comparing two versions"""
        versioning = ProjectVersioning(versions_dir=temp_dir / "versions")
        
        # Create two versions
        project_v1 = temp_dir / "v1"
        project_v1.mkdir()
        (project_v1 / "file.txt").write_text("version 1")
        
        project_v2 = temp_dir / "v2"
        project_v2.mkdir()
        (project_v2 / "file.txt").write_text("version 2")
        
        versioning.create_version("compare-test", project_v1, "1.0.0")
        versioning.create_version("compare-test", project_v2, "2.0.0")
        
        comparison = versioning.compare_versions("compare-test", "1.0.0", "2.0.0")
        
        assert comparison is not None
        assert comparison["version1"] == "1.0.0"
        assert comparison["version2"] == "2.0.0"
        # Hashes should be different
        assert comparison["version1_hash"] != comparison["version2_hash"]

    def test_delete_version(self, temp_dir):
        """Test deleting a version"""
        versioning = ProjectVersioning(versions_dir=temp_dir / "versions")
        
        project_path = temp_dir / "delete_test"
        project_path.mkdir()
        
        versioning.create_version("delete-test", project_path, "1.0.0")
        version_dir = versioning.versions_dir / "delete-test" / "1.0.0"
        assert version_dir.exists()
        
        result = versioning.delete_version("delete-test", "1.0.0")
        
        assert result is True
        assert not version_dir.exists()

    def test_delete_version_not_found(self, temp_dir):
        """Test deleting non-existent version"""
        versioning = ProjectVersioning(versions_dir=temp_dir / "versions")
        
        result = versioning.delete_version("non-existent", "1.0.0")
        
        assert result is False

