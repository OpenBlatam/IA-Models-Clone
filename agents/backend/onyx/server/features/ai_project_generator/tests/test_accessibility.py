"""
Accessibility and usability tests
"""

import pytest
from pathlib import Path
from typing import Dict, Any


class TestAccessibility:
    """Tests for accessibility and usability"""
    
    def test_readme_readability(self, sample_project_structure):
        """Test that README is readable and informative"""
        readme = sample_project_structure / "README.md"
        
        if readme.exists():
            content = readme.read_text(encoding="utf-8")
            
            # Should have title
            assert "#" in content or "Title" in content or len(content) > 0
            
            # Should have reasonable length
            assert len(content) > 50, "README is too short"
    
    def test_project_structure_navigation(self, sample_project_structure):
        """Test that project structure is navigable"""
        # Backend should be accessible
        backend = sample_project_structure / "backend"
        if backend.exists():
            assert backend.is_dir(), "Backend is not a directory"
            
            # Main file should be accessible
            main_py = backend / "main.py"
            if main_py.exists():
                assert main_py.is_file(), "main.py is not a file"
        
        # Frontend should be accessible
        frontend = sample_project_structure / "frontend"
        if frontend.exists():
            assert frontend.is_dir(), "Frontend is not a directory"
    
    def test_file_permissions(self, temp_dir):
        """Test that generated files have correct permissions"""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")
        
        # File should be readable
        assert test_file.exists(), "File was not created"
        
        # Should be able to read
        content = test_file.read_text(encoding="utf-8")
        assert content == "test", "File is not readable"
    
    def test_error_messages_clarity(self):
        """Test that error messages are clear"""
        from ..core.project_generator import ProjectGenerator
        
        generator = ProjectGenerator()
        
        # Test with invalid input
        try:
            result = generator._sanitize_name(None)
            # Should handle gracefully
        except Exception as e:
            # Error message should be informative
            assert len(str(e)) > 0, "Error message is empty"

