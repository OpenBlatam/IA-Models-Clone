"""
Advanced quality tests
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List


class TestQualityAdvanced:
    """Advanced quality tests"""
    
    def test_code_quality_metrics(self, temp_dir):
        """Test code quality metrics"""
        code_file = temp_dir / "code.py"
        code_content = """
def good_function():
    '''Well documented function.'''
    return True

class GoodClass:
    '''Well documented class.'''
    def method(self):
        return True
"""
        code_file.write_text(code_content)
        
        # Calculate quality metrics
        content = code_file.read_text(encoding="utf-8")
        lines = content.split("\n")
        
        metrics = {
            "total_lines": len(lines),
            "code_lines": len([l for l in lines if l.strip() and not l.strip().startswith("#")]),
            "has_docstrings": '"""' in content or "'''" in content,
            "functions": content.count("def "),
            "classes": content.count("class "),
        }
        
        assert metrics["total_lines"] > 0
        assert metrics["has_docstrings"] is True
        assert metrics["functions"] >= 1
    
    def test_project_structure_quality(self, temp_dir):
        """Test project structure quality"""
        project_dir = temp_dir / "project"
        project_dir.mkdir()
        
        # Create good structure
        (project_dir / "README.md").write_text("# Project")
        (project_dir / "src").mkdir()
        (project_dir / "tests").mkdir()
        (project_dir / "requirements.txt").write_text("requests==2.0.0")
        
        # Quality checks
        quality = {
            "has_readme": (project_dir / "README.md").exists(),
            "has_src": (project_dir / "src").exists(),
            "has_tests": (project_dir / "tests").exists(),
            "has_requirements": (project_dir / "requirements.txt").exists(),
        }
        
        assert all(quality.values())
    
    def test_documentation_quality(self, temp_dir):
        """Test documentation quality"""
        readme = temp_dir / "README.md"
        readme.write_text("""
# Project Name

## Description
This is a well-documented project.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from project import main
main()
```
""")
        
        content = readme.read_text(encoding="utf-8")
        
        # Quality checks
        quality = {
            "has_title": "#" in content,
            "has_description": "Description" in content or "description" in content.lower(),
            "has_installation": "Installation" in content or "install" in content.lower(),
            "has_usage": "Usage" in content or "usage" in content.lower(),
        }
        
        assert sum(quality.values()) >= 2
    
    def test_naming_convention_quality(self):
        """Test naming convention quality"""
        # Good names
        good_names = [
            "my_project",
            "MyClass",
            "my_function",
            "MY_CONSTANT"
        ]
        
        # Bad names
        bad_names = [
            "my project",  # spaces
            "123project",  # starts with number
            "project-",  # trailing dash
        ]
        
        # Validate good names
        for name in good_names:
            is_valid = (
                " " not in name and
                len(name) > 0 and
                name.replace("_", "").replace("-", "").isalnum() or
                any(c.isalpha() for c in name)
            )
            assert is_valid
    
    def test_dependency_quality(self, temp_dir):
        """Test dependency quality"""
        requirements = temp_dir / "requirements.txt"
        requirements.write_text("""
requests==2.0.0
flask>=3.0.0
pytest>=7.0.0
""")
        
        content = requirements.read_text(encoding="utf-8")
        lines = [l.strip() for l in content.split("\n") if l.strip() and not l.startswith("#")]
        
        # Quality checks
        quality = {
            "has_dependencies": len(lines) > 0,
            "has_versions": any("==" in line or ">=" in line for line in lines),
            "all_valid": all(
                "==" in line or ">=" in line or "<=" in line or line
                for line in lines
            ),
        }
        
        assert quality["has_dependencies"] is True
        assert quality["has_versions"] is True

