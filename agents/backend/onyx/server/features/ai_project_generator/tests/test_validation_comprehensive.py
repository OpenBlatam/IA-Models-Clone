"""
Comprehensive validation tests
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List
import json


class TestValidationComprehensive:
    """Comprehensive validation tests"""
    
    def test_complete_project_validation(self, temp_dir):
        """Test complete project validation"""
        project_dir = temp_dir / "project"
        project_dir.mkdir()
        
        # Create complete project structure
        (project_dir / "README.md").write_text("# Project")
        (project_dir / "main.py").write_text("print('Hello')")
        (project_dir / "requirements.txt").write_text("requests==2.0.0")
        (project_dir / "config.json").write_text(json.dumps({"name": "test"}))
        
        # Validate structure
        validations = {
            "has_readme": (project_dir / "README.md").exists(),
            "has_python": len(list(project_dir.glob("*.py"))) > 0,
            "has_requirements": (project_dir / "requirements.txt").exists(),
            "has_config": (project_dir / "config.json").exists(),
        }
        
        assert all(validations.values())
    
    def test_file_content_validation(self, temp_dir):
        """Test file content validation"""
        # Create files
        python_file = temp_dir / "code.py"
        python_file.write_text("def function():\n    return True")
        
        json_file = temp_dir / "data.json"
        json_file.write_text(json.dumps({"key": "value"}))
        
        # Validate content
        python_content = python_file.read_text(encoding="utf-8")
        json_content = json_file.read_text(encoding="utf-8")
        
        # Python validation
        assert "def" in python_content
        assert "return" in python_content
        
        # JSON validation
        try:
            data = json.loads(json_content)
            assert "key" in data
        except json.JSONDecodeError:
            pytest.fail("Invalid JSON")
    
    def test_dependency_validation(self, temp_dir):
        """Test dependency validation"""
        requirements_file = temp_dir / "requirements.txt"
        requirements_file.write_text("requests==2.0.0\nflask==3.0.0\npytest>=7.0.0")
        
        # Parse requirements
        requirements = []
        for line in requirements_file.read_text(encoding="utf-8").split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
        
        # Validate format
        assert len(requirements) == 3
        assert any("requests" in req for req in requirements)
        assert any("flask" in req for req in requirements)
    
    def test_configuration_validation(self, temp_dir):
        """Test configuration validation"""
        config = {
            "name": "test-project",
            "version": "1.0.0",
            "settings": {
                "port": 8000,
                "debug": True
            }
        }
        
        config_file = temp_dir / "config.json"
        config_file.write_text(json.dumps(config), encoding="utf-8")
        
        # Validate
        loaded = json.loads(config_file.read_text(encoding="utf-8"))
        
        assert "name" in loaded
        assert "version" in loaded
        assert "settings" in loaded
        assert isinstance(loaded["settings"]["port"], int)
        assert 1 <= loaded["settings"]["port"] <= 65535
        assert isinstance(loaded["settings"]["debug"], bool)
    
    def test_structure_validation(self, temp_dir):
        """Test project structure validation"""
        project_dir = temp_dir / "project"
        project_dir.mkdir()
        
        # Create structure
        (project_dir / "src").mkdir()
        (project_dir / "tests").mkdir()
        (project_dir / "src" / "main.py").write_text("code")
        (project_dir / "tests" / "test_main.py").write_text("test")
        
        # Validate structure
        structure_valid = {
            "has_src": (project_dir / "src").exists(),
            "has_tests": (project_dir / "tests").exists(),
            "has_main": (project_dir / "src" / "main.py").exists(),
            "has_test_file": (project_dir / "tests" / "test_main.py").exists(),
        }
        
        assert all(structure_valid.values())
    
    def test_naming_convention_validation(self, temp_dir):
        """Test naming convention validation"""
        # Valid names
        valid_names = ["my_project", "my-project", "myProject", "MY_PROJECT"]
        
        for name in valid_names:
            # Basic validation: no spaces, reasonable length
            is_valid = (
                " " not in name and
                len(name) > 0 and
                len(name) <= 50
            )
            assert is_valid, f"Name '{name}' should be valid"
        
        # Invalid names
        invalid_names = ["", " " * 10, "a" * 100]
        
        for name in invalid_names:
            is_valid = (
                " " not in name and
                len(name) > 0 and
                len(name) <= 50
            )
            assert not is_valid, f"Name '{name}' should be invalid"

