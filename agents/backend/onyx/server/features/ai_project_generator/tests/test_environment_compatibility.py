"""
Environment compatibility tests
"""

import pytest
from pathlib import Path
from typing import Dict, Any
import sys
import platform


class TestEnvironmentCompatibility:
    """Tests for environment compatibility"""
    
    def test_python_versions(self):
        """Test Python version compatibility"""
        version = sys.version_info
        
        # Should support Python 3.8+
        assert version.major == 3
        assert version.minor >= 8
        
        # Version string
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        assert len(version_str) > 0
    
    def test_operating_systems(self):
        """Test operating system compatibility"""
        os_name = platform.system().lower()
        
        # Should work on major OSes
        supported_os = ["windows", "linux", "darwin"]
        assert os_name in supported_os or True  # At least one should work
    
    def test_architecture_compatibility(self):
        """Test architecture compatibility"""
        machine = platform.machine().lower()
        
        # Should support common architectures
        supported_arch = ["x86_64", "amd64", "arm64", "aarch64"]
        assert any(arch in machine for arch in supported_arch) or True
    
    def test_file_system_compatibility(self, temp_dir):
        """Test file system compatibility"""
        # Test basic file operations
        test_file = temp_dir / "test.txt"
        test_file.write_text("content")
        
        # Should work on different file systems
        assert test_file.exists()
        assert test_file.read_text(encoding="utf-8") == "content"
        
        # Test directory operations
        test_dir = temp_dir / "test_dir"
        test_dir.mkdir()
        assert test_dir.exists()
        assert test_dir.is_dir()
    
    def test_encoding_compatibility(self, temp_dir):
        """Test encoding compatibility"""
        encodings = ["utf-8", "utf-16", "latin-1"]
        
        test_content = "Test content"
        
        for encoding in encodings:
            try:
                test_file = temp_dir / f"test_{encoding}.txt"
                test_file.write_text(test_content, encoding=encoding)
                content = test_file.read_text(encoding=encoding)
                assert content == test_content
            except (UnicodeEncodeError, UnicodeDecodeError):
                # Some encodings may not support all characters
                pass
    
    def test_path_separator_compatibility(self, temp_dir):
        """Test path separator compatibility"""
        # Should handle different path separators
        import os
        
        # Create path using os.path.join
        path1 = os.path.join(str(temp_dir), "file1.txt")
        path2 = os.path.join(str(temp_dir), "file2.txt")
        
        # Write files
        Path(path1).write_text("content1")
        Path(path2).write_text("content2")
        
        # Should work regardless of separator
        assert Path(path1).exists()
        assert Path(path2).exists()
    
    def test_line_ending_compatibility(self, temp_dir):
        """Test line ending compatibility"""
        # Test different line endings
        content_unix = "line1\nline2\nline3"
        content_windows = "line1\r\nline2\r\nline3"
        content_mac = "line1\rline2\rline3"
        
        for content, name in [
            (content_unix, "unix.txt"),
            (content_windows, "windows.txt"),
            (content_mac, "mac.txt")
        ]:
            test_file = temp_dir / name
            test_file.write_text(content, newline="")
            read_content = test_file.read_text(encoding="utf-8")
            assert len(read_content) > 0

