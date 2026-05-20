"""
File Testing Helpers
Specialized helpers for file and storage testing
"""

from typing import Any, Dict, List, Optional
from unittest.mock import Mock, AsyncMock
import os
import tempfile
import shutil
from pathlib import Path
import hashlib


class FileTestHelpers:
    """Helpers for file testing"""
    
    @staticmethod
    def create_temp_directory(prefix: str = "test_") -> str:
        """Create temporary directory"""
        return tempfile.mkdtemp(prefix=prefix)
    
    @staticmethod
    def create_temp_file(
        content: bytes = b"test content",
        suffix: str = ".tmp",
        directory: Optional[str] = None
    ) -> str:
        """Create temporary file with content"""
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        fd, path = tempfile.mkstemp(suffix=suffix, dir=directory)
        try:
            os.write(fd, content)
            return path
        finally:
            os.close(fd)
    
    @staticmethod
    def cleanup_temp_path(path: str):
        """Cleanup temporary file or directory"""
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
    
    @staticmethod
    def assert_file_exists(file_path: str):
        """Assert file exists"""
        assert os.path.exists(file_path), f"File {file_path} does not exist"
        assert os.path.isfile(file_path), f"{file_path} is not a file"
    
    @staticmethod
    def assert_file_content(file_path: str, expected_content: bytes):
        """Assert file has expected content"""
        FileTestHelpers.assert_file_exists(file_path)
        with open(file_path, 'rb') as f:
            actual_content = f.read()
        assert actual_content == expected_content, \
            f"File content does not match expected"
    
    @staticmethod
    def assert_file_size(file_path: str, expected_size: int, tolerance: int = 0):
        """Assert file has expected size"""
        FileTestHelpers.assert_file_exists(file_path)
        actual_size = os.path.getsize(file_path)
        assert abs(actual_size - expected_size) <= tolerance, \
            f"File size {actual_size} does not match expected {expected_size} (tolerance: {tolerance})"
    
    @staticmethod
    def calculate_file_hash(file_path: str, algorithm: str = "md5") -> str:
        """Calculate file hash"""
        hash_obj = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    
    @staticmethod
    def assert_file_hash(file_path: str, expected_hash: str, algorithm: str = "md5"):
        """Assert file has expected hash"""
        actual_hash = FileTestHelpers.calculate_file_hash(file_path, algorithm)
        assert actual_hash == expected_hash, \
            f"File hash {actual_hash} does not match expected {expected_hash}"


class StorageHelpers:
    """Helpers for storage testing"""
    
    @staticmethod
    def create_mock_storage(
        files: Optional[Dict[str, bytes]] = None
    ) -> Mock:
        """Create mock storage service"""
        file_storage = files or {}
        storage = Mock()
        
        async def upload_side_effect(file_path: str, destination: str, content: bytes):
            file_storage[destination] = content
            return {"url": f"https://storage.example.com/{destination}", "path": destination}
        
        async def download_side_effect(file_path: str):
            return file_storage.get(file_path)
        
        async def delete_side_effect(file_path: str):
            if file_path in file_storage:
                del file_storage[file_path]
            return True
        
        async def exists_side_effect(file_path: str):
            return file_path in file_storage
        
        storage.upload = AsyncMock(side_effect=upload_side_effect)
        storage.download = AsyncMock(side_effect=download_side_effect)
        storage.delete = AsyncMock(side_effect=delete_side_effect)
        storage.exists = AsyncMock(side_effect=exists_side_effect)
        storage.files = file_storage
        return storage
    
    @staticmethod
    def assert_file_uploaded(
        storage: Mock,
        file_path: str,
        expected_content: Optional[bytes] = None
    ):
        """Assert file was uploaded"""
        assert storage.upload.called, f"File {file_path} was not uploaded"
        
        if hasattr(storage, "files") and expected_content:
            assert file_path in storage.files, f"File {file_path} not in storage"
            assert storage.files[file_path] == expected_content, \
                "Uploaded file content does not match expected"
    
    @staticmethod
    def assert_file_deleted(storage: Mock, file_path: str):
        """Assert file was deleted"""
        assert storage.delete.called, f"File {file_path} was not deleted"
        if hasattr(storage, "files"):
            assert file_path not in storage.files, f"File {file_path} still exists in storage"


class ImageFileHelpers:
    """Helpers for image file testing"""
    
    @staticmethod
    def create_test_image_file(
        width: int = 200,
        height: int = 200,
        format: str = "JPEG",
        color: str = "red"
    ) -> str:
        """Create test image file"""
        from PIL import Image
        import io
        
        img = Image.new('RGB', (width, height), color=color)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format=format)
        img_bytes.seek(0)
        
        return FileTestHelpers.create_temp_file(
            content=img_bytes.read(),
            suffix=f".{format.lower()}"
        )
    
    @staticmethod
    def assert_image_file_valid(file_path: str, expected_format: Optional[str] = None):
        """Assert image file is valid"""
        FileTestHelpers.assert_file_exists(file_path)
        
        from PIL import Image
        try:
            img = Image.open(file_path)
            img.verify()
            if expected_format:
                assert img.format == expected_format, \
                    f"Image format {img.format} does not match expected {expected_format}"
        except Exception as e:
            raise AssertionError(f"Invalid image file: {e}")


class FileUploadHelpers:
    """Helpers for file upload testing"""
    
    @staticmethod
    def create_mock_upload_file(
        filename: str = "test.jpg",
        content: bytes = b"fake image content",
        content_type: str = "image/jpeg"
    ) -> Mock:
        """Create mock upload file"""
        from fastapi import UploadFile
        import io
        
        file_obj = UploadFile(
            filename=filename,
            file=io.BytesIO(content),
            headers={"content-type": content_type}
        )
        return file_obj
    
    @staticmethod
    def assert_file_upload_valid(
        upload_file: Mock,
        max_size: Optional[int] = None,
        allowed_types: Optional[List[str]] = None
    ):
        """Assert file upload is valid"""
        assert upload_file.filename is not None, "Upload file missing filename"
        
        if max_size:
            content = upload_file.file.read()
            upload_file.file.seek(0)  # Reset for future reads
            assert len(content) <= max_size, \
                f"File size {len(content)} exceeds max {max_size}"
        
        if allowed_types:
            # Check file extension or content type
            ext = Path(upload_file.filename).suffix.lower()
            assert ext in [f".{t}" for t in allowed_types] or \
                   upload_file.content_type in allowed_types, \
                f"File type not allowed: {ext} or {upload_file.content_type}"


# Convenience exports
create_temp_directory = FileTestHelpers.create_temp_directory
create_temp_file = FileTestHelpers.create_temp_file
cleanup_temp_path = FileTestHelpers.cleanup_temp_path
assert_file_exists = FileTestHelpers.assert_file_exists
assert_file_content = FileTestHelpers.assert_file_content
assert_file_size = FileTestHelpers.assert_file_size
calculate_file_hash = FileTestHelpers.calculate_file_hash
assert_file_hash = FileTestHelpers.assert_file_hash

create_mock_storage = StorageHelpers.create_mock_storage
assert_file_uploaded = StorageHelpers.assert_file_uploaded
assert_file_deleted = StorageHelpers.assert_file_deleted

create_test_image_file = ImageFileHelpers.create_test_image_file
assert_image_file_valid = ImageFileHelpers.assert_image_file_valid

create_mock_upload_file = FileUploadHelpers.create_mock_upload_file
assert_file_upload_valid = FileUploadHelpers.assert_file_upload_valid



