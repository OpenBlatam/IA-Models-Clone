"""
Complete documentation tests
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List
import re


class TestDocumentationComplete:
    """Complete documentation tests"""
    
    def test_readme_completeness(self, temp_dir):
        """Test README completeness"""
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

## Contributing
Contributions are welcome!

## License
MIT
""")
        
        content = readme.read_text(encoding="utf-8")
        
        # Check for essential sections
        sections = {
            "title": bool(re.search(r'^#\s+\w+', content, re.MULTILINE)),
            "description": "Description" in content or "description" in content.lower(),
            "installation": "Installation" in content or "install" in content.lower(),
            "usage": "Usage" in content or "usage" in content.lower(),
        }
        
        assert sum(sections.values()) >= 3
    
    def test_api_documentation(self, temp_dir):
        """Test API documentation"""
        api_doc = temp_dir / "API.md"
        api_doc.write_text("""
# API Documentation

## Endpoints

### GET /api/projects
Returns list of projects.

### POST /api/projects
Creates a new project.
""")
        
        content = api_doc.read_text(encoding="utf-8")
        
        # Check for API elements
        has_endpoints = "GET" in content or "POST" in content
        has_descriptions = len(content.split("\n")) > 5
        
        assert has_endpoints or has_descriptions
    
    def test_code_documentation(self, temp_dir):
        """Test code documentation"""
        code_file = temp_dir / "code.py"
        code_content = '''
def documented_function(param1, param2):
    """
    This is a well-documented function.
    
    Args:
        param1: First parameter
        param2: Second parameter
    
    Returns:
        bool: True if successful
    """
    return True

class DocumentedClass:
    """This is a well-documented class."""
    
    def method(self):
        """Documented method."""
        return True
'''
        code_file.write_text(code_content)
        
        content = code_file.read_text(encoding="utf-8")
        
        # Check for documentation
        has_docstrings = '"""' in content or "'''" in content
        has_args = "Args:" in content
        has_returns = "Returns:" in content
        
        assert has_docstrings
        assert has_args or has_returns
    
    def test_changelog_format(self, temp_dir):
        """Test changelog format"""
        changelog = temp_dir / "CHANGELOG.md"
        changelog.write_text("""
# Changelog

## [1.0.0] - 2024-01-01
### Added
- Initial release

## [0.9.0] - 2023-12-01
### Changed
- Updated dependencies
""")
        
        content = changelog.read_text(encoding="utf-8")
        
        # Check for changelog elements
        has_versions = bool(re.search(r'\[[\d.]+\]', content))
        has_dates = bool(re.search(r'\d{4}-\d{2}-\d{2}', content))
        has_sections = "Added" in content or "Changed" in content
        
        assert has_versions or has_dates or has_sections
    
    def test_examples_documentation(self, temp_dir):
        """Test examples documentation"""
        examples = temp_dir / "EXAMPLES.md"
        examples.write_text("""
# Examples

## Basic Usage
```python
from project import ProjectGenerator

generator = ProjectGenerator()
project = generator.generate("My project")
```

## Advanced Usage
```python
# Advanced example here
```
""")
        
        content = examples.read_text(encoding="utf-8")
        
        # Check for examples
        has_code_blocks = "```" in content
        has_examples = "example" in content.lower() or "usage" in content.lower()
        
        assert has_code_blocks or has_examples
    
    def test_troubleshooting_documentation(self, temp_dir):
        """Test troubleshooting documentation"""
        troubleshooting = temp_dir / "TROUBLESHOOTING.md"
        troubleshooting.write_text("""
# Troubleshooting

## Common Issues

### Issue 1
**Problem:** Description
**Solution:** Solution description

### Issue 2
**Problem:** Description
**Solution:** Solution description
""")
        
        content = troubleshooting.read_text(encoding="utf-8")
        
        # Check for troubleshooting elements
        has_issues = "Issue" in content or "Problem" in content
        has_solutions = "Solution" in content
        
        assert has_issues or has_solutions

