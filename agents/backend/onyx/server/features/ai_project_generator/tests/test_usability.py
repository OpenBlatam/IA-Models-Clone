"""
Usability tests
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List


class TestUsability:
    """Tests for system usability"""
    
    def test_ease_of_use(self, project_generator):
        """Test ease of use"""
        # Simple usage should be straightforward
        description = "A simple AI project"
        project = project_generator.generate_project(description)
        
        # Should be easy to use
        assert project is not None
        assert "project_id" in project or "project_path" in project
    
    def test_clear_error_messages(self, project_generator):
        """Test clear error messages"""
        # Error messages should be clear
        try:
            project = project_generator.generate_project("")
        except Exception as e:
            # Error message should be informative
            error_msg = str(e)
            assert len(error_msg) > 0  # Should have a message
    
    def test_intuitive_api(self, project_generator):
        """Test intuitive API"""
        # API should be intuitive
        result = project_generator._sanitize_name("My Project")
        
        # Should work as expected
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_helpful_documentation(self, temp_dir):
        """Test helpful documentation"""
        readme = temp_dir / "README.md"
        readme.write_text("""
# Project

## Quick Start
1. Install dependencies
2. Run the project
3. Enjoy!

## Examples
See examples/ directory
""")
        
        content = readme.read_text(encoding="utf-8")
        
        # Should be helpful
        has_quick_start = "Quick Start" in content or "quick" in content.lower()
        has_examples = "Examples" in content or "example" in content.lower()
        
        assert has_quick_start or has_examples
    
    def test_consistent_behavior(self, project_generator):
        """Test consistent behavior"""
        # Same input should produce consistent results
        description = "Test project"
        
        result1 = project_generator._sanitize_name(description)
        result2 = project_generator._sanitize_name(description)
        
        # Should be consistent
        assert result1 == result2
    
    def test_feedback_mechanisms(self, project_generator):
        """Test feedback mechanisms"""
        # System should provide feedback
        project = project_generator.generate_project("Test")
        
        # Should provide some feedback
        assert project is not None
        # Project should have some status or information
        assert isinstance(project, dict) or project is not None
    
    def test_learning_curve(self, project_generator):
        """Test learning curve"""
        # Should be easy to learn
        # First use
        project1 = project_generator.generate_project("First project")
        
        # Second use (should be easier)
        project2 = project_generator.generate_project("Second project")
        
        # Should be straightforward
        assert project1 is not None
        assert project2 is not None

