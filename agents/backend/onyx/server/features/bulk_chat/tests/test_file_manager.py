"""
Tests for File Manager
=======================
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from ..core.file_manager import FileManager


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def file_manager(temp_dir):
    """Create file manager for testing."""
    return FileManager(storage_path=str(temp_dir))


@pytest.mark.asyncio
async def test_upload_file(file_manager):
    """Test uploading a file."""
    file_content = b"Test file content"
    
    file_id = await file_manager.upload_file(
        file_id="test_file.txt",
        content=file_content,
        metadata={"category": "test"}
    )
    
    assert file_id == "test_file.txt"
    assert file_id in file_manager.files


@pytest.mark.asyncio
async def test_download_file(file_manager):
    """Test downloading a file."""
    file_content = b"Test file content"
    
    await file_manager.upload_file("test_file.txt", file_content)
    
    downloaded = await file_manager.download_file("test_file.txt")
    
    assert downloaded == file_content


@pytest.mark.asyncio
async def test_delete_file(file_manager):
    """Test deleting a file."""
    await file_manager.upload_file("test_file.txt", b"content")
    
    assert "test_file.txt" in file_manager.files
    
    await file_manager.delete_file("test_file.txt")
    
    assert "test_file.txt" not in file_manager.files


@pytest.mark.asyncio
async def test_get_file_metadata(file_manager):
    """Test getting file metadata."""
    await file_manager.upload_file(
        "test_file.txt",
        b"content",
        metadata={"category": "test", "size": 7}
    )
    
    metadata = file_manager.get_file_metadata("test_file.txt")
    
    assert metadata is not None
    assert "category" in metadata or "size" in metadata or "file_id" in metadata


@pytest.mark.asyncio
async def test_list_files(file_manager):
    """Test listing files."""
    await file_manager.upload_file("file1.txt", b"content1")
    await file_manager.upload_file("file2.txt", b"content2")
    
    files = file_manager.list_files()
    
    assert len(files) >= 2
    assert any(f.file_id == "file1.txt" for f in files)
    assert any(f.file_id == "file2.txt" for f in files)


@pytest.mark.asyncio
async def test_get_file_manager_summary(file_manager):
    """Test getting file manager summary."""
    await file_manager.upload_file("file1.txt", b"content1")
    await file_manager.upload_file("file2.txt", b"content2")
    
    summary = file_manager.get_file_manager_summary()
    
    assert summary is not None
    assert "total_files" in summary or "total_size" in summary


