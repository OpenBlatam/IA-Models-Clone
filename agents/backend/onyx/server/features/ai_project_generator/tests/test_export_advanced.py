"""
Advanced export tests
"""

import pytest
from pathlib import Path
from typing import Dict, Any
import json
import zipfile
import tarfile


class TestExportAdvanced:
    """Advanced export tests"""
    
    def test_zip_export(self, temp_dir):
        """Test ZIP export"""
        # Create project files
        project_dir = temp_dir / "project"
        project_dir.mkdir()
        (project_dir / "file1.txt").write_text("content1")
        (project_dir / "file2.txt").write_text("content2")
        
        # Create ZIP
        zip_path = temp_dir / "export.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in project_dir.rglob("*"):
                if file.is_file():
                    zipf.write(file, file.relative_to(project_dir.parent))
        
        # Verify ZIP
        assert zip_path.exists()
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            files = zipf.namelist()
            assert len(files) >= 2
    
    def test_tar_export(self, temp_dir):
        """Test TAR export"""
        # Create project files
        project_dir = temp_dir / "project"
        project_dir.mkdir()
        (project_dir / "file.txt").write_text("content")
        
        # Create TAR
        tar_path = temp_dir / "export.tar.gz"
        with tarfile.open(tar_path, 'w:gz') as tarf:
            tarf.add(project_dir, arcname="project")
        
        # Verify TAR
        assert tar_path.exists()
        with tarfile.open(tar_path, 'r:gz') as tarf:
            members = tarf.getnames()
            assert len(members) >= 1
    
    def test_json_export(self, temp_dir):
        """Test JSON export"""
        data = {
            "name": "test-project",
            "version": "1.0.0",
            "files": ["file1.py", "file2.py"]
        }
        
        json_path = temp_dir / "export.json"
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        
        # Verify JSON
        assert json_path.exists()
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert loaded["name"] == "test-project"
        assert len(loaded["files"]) == 2
    
    def test_multiple_format_export(self, temp_dir):
        """Test exporting in multiple formats"""
        project_dir = temp_dir / "project"
        project_dir.mkdir()
        (project_dir / "file.txt").write_text("content")
        
        # Export as ZIP
        zip_path = temp_dir / "export.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in project_dir.rglob("*"):
                if file.is_file():
                    zipf.write(file, file.relative_to(project_dir.parent))
        
        # Export as JSON
        data = {"files": [f.name for f in project_dir.iterdir() if f.is_file()]}
        json_path = temp_dir / "export.json"
        json_path.write_text(json.dumps(data), encoding="utf-8")
        
        # Both should exist
        assert zip_path.exists()
        assert json_path.exists()
    
    def test_export_metadata(self, temp_dir):
        """Test export with metadata"""
        metadata = {
            "export_date": "2024-01-01",
            "version": "1.0.0",
            "format": "zip"
        }
        
        # Create export with metadata
        zip_path = temp_dir / "export.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Add metadata file
            metadata_str = json.dumps(metadata)
            zipf.writestr("metadata.json", metadata_str)
        
        # Verify metadata
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            metadata_content = zipf.read("metadata.json").decode("utf-8")
            loaded_metadata = json.loads(metadata_content)
            assert loaded_metadata["version"] == "1.0.0"

