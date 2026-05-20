"""
Tests for ExportGenerator utility
"""

import pytest
import asyncio
import zipfile
import tarfile
from pathlib import Path

from ..utils.export_generator import ExportGenerator


class TestExportGenerator:
    """Test suite for ExportGenerator"""

    def test_init(self):
        """Test ExportGenerator initialization"""
        generator = ExportGenerator()
        assert generator is not None

    @pytest.mark.asyncio
    async def test_export_to_zip(self, temp_dir):
        """Test exporting project to ZIP"""
        generator = ExportGenerator()
        
        # Create test project structure
        project_dir = temp_dir / "test_project"
        (project_dir / "backend" / "app").mkdir(parents=True)
        (project_dir / "frontend" / "src").mkdir(parents=True)
        (project_dir / "backend" / "main.py").write_text("# Main file")
        (project_dir / "README.md").write_text("# Test Project")
        
        output_path = temp_dir / "test_project.zip"
        result = await generator.export_to_zip(project_dir, output_path)
        
        assert result == output_path
        assert output_path.exists()
        
        # Verify ZIP contents
        with zipfile.ZipFile(output_path, 'r') as zipf:
            files = zipf.namelist()
            assert "backend/main.py" in files
            assert "README.md" in files

    @pytest.mark.asyncio
    async def test_export_to_zip_auto_path(self, temp_dir):
        """Test exporting to ZIP with auto-generated path"""
        generator = ExportGenerator()
        
        project_dir = temp_dir / "auto_project"
        project_dir.mkdir()
        (project_dir / "test.txt").write_text("test")
        
        result = await generator.export_to_zip(project_dir)
        
        assert result.exists()
        assert result.suffix == ".zip"
        assert result.name == "auto_project.zip"

    @pytest.mark.asyncio
    async def test_export_to_zip_excludes_files(self, temp_dir):
        """Test that ZIP excludes unnecessary files"""
        generator = ExportGenerator()
        
        project_dir = temp_dir / "exclude_test"
        project_dir.mkdir()
        (project_dir / "test.py").write_text("# Test")
        (project_dir / "__pycache__" / "test.pyc").mkdir(parents=True)
        (project_dir / "__pycache__" / "test.pyc").write_text("compiled")
        (project_dir / ".git" / "config").mkdir(parents=True)
        (project_dir / ".git" / "config").write_text("git config")
        
        output_path = temp_dir / "exclude_test.zip"
        await generator.export_to_zip(project_dir, output_path)
        
        with zipfile.ZipFile(output_path, 'r') as zipf:
            files = zipf.namelist()
            assert "test.py" in files
            assert not any("__pycache__" in f for f in files)
            assert not any(".git" in f for f in files)

    @pytest.mark.asyncio
    async def test_export_to_tar(self, temp_dir):
        """Test exporting project to TAR"""
        generator = ExportGenerator()
        
        project_dir = temp_dir / "tar_project"
        project_dir.mkdir()
        (project_dir / "test.txt").write_text("test")
        
        output_path = temp_dir / "tar_project.tar.gz"
        result = await generator.export_to_tar(project_dir, output_path, compression="gz")
        
        assert result == output_path
        assert output_path.exists()

    @pytest.mark.asyncio
    async def test_export_to_tar_different_compressions(self, temp_dir):
        """Test exporting to TAR with different compressions"""
        generator = ExportGenerator()
        
        project_dir = temp_dir / "compression_test"
        project_dir.mkdir()
        (project_dir / "test.txt").write_text("test")
        
        # Test gz compression
        result_gz = await generator.export_to_tar(project_dir, None, compression="gz")
        assert result_gz.suffixes == [".tar", ".gz"]
        
        # Test bz2 compression
        result_bz2 = await generator.export_to_tar(project_dir, None, compression="bz2")
        assert result_bz2.suffixes == [".tar", ".bz2"]
        
        # Test no compression
        result_none = await generator.export_to_tar(project_dir, None, compression=None)
        assert result_none.suffix == ".tar"

    @pytest.mark.asyncio
    async def test_export_error_handling(self, temp_dir):
        """Test error handling during export"""
        generator = ExportGenerator()
        
        # Try to export non-existent directory
        non_existent = temp_dir / "non_existent"
        
        with pytest.raises(Exception):
            await generator.export_to_zip(non_existent)

