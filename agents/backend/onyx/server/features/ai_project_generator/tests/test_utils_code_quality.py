"""
Tests for CodeQualityAnalyzer utility
"""

import pytest
from pathlib import Path

from ..utils.code_quality_analyzer import CodeQualityAnalyzer


class TestCodeQualityAnalyzer:
    """Test suite for CodeQualityAnalyzer"""

    def test_init(self):
        """Test CodeQualityAnalyzer initialization"""
        analyzer = CodeQualityAnalyzer()
        assert len(analyzer.quality_rules) > 0
        assert any(rule["id"] == "complexity" for rule in analyzer.quality_rules)

    def test_analyze_python_file(self, temp_dir):
        """Test analyzing a Python file"""
        analyzer = CodeQualityAnalyzer()
        
        # Create test Python file
        test_file = temp_dir / "test.py"
        test_file.write_text("""
def simple_function():
    return "hello"

class TestClass:
    def method(self):
        pass
""")
        
        result = analyzer.analyze_file(test_file)
        
        assert "file" in result
        assert result["file"] == str(test_file)
        assert "lines" in result
        assert "functions" in result
        assert "classes" in result

    def test_analyze_python_file_with_issues(self, temp_dir):
        """Test analyzing Python file with quality issues"""
        analyzer = CodeQualityAnalyzer()
        
        # Create file with long function
        test_file = temp_dir / "long_function.py"
        long_function = "def long_function():\n" + "    pass\n" * 60
        test_file.write_text(long_function)
        
        result = analyzer.analyze_file(test_file)
        
        assert "issues" in result
        # Should detect long function

    def test_analyze_python_file_syntax_error(self, temp_dir):
        """Test analyzing Python file with syntax error"""
        analyzer = CodeQualityAnalyzer()
        
        test_file = temp_dir / "syntax_error.py"
        test_file.write_text("def invalid syntax here")
        
        result = analyzer.analyze_file(test_file)
        
        assert "syntax_error" in result or "error" in result

    def test_analyze_generic_file(self, temp_dir):
        """Test analyzing non-Python file"""
        analyzer = CodeQualityAnalyzer()
        
        test_file = temp_dir / "test.txt"
        test_file.write_text("This is a text file")
        
        result = analyzer.analyze_file(test_file)
        
        assert "file" in result
        assert "lines" in result

    def test_analyze_file_not_found(self, temp_dir):
        """Test analyzing non-existent file"""
        analyzer = CodeQualityAnalyzer()
        
        result = analyzer.analyze_file(temp_dir / "nonexistent.py")
        
        assert "error" in result

    def test_analyze_project(self, temp_dir):
        """Test analyzing entire project"""
        analyzer = CodeQualityAnalyzer()
        
        # Create project structure
        project_dir = temp_dir / "project"
        project_dir.mkdir()
        (project_dir / "main.py").write_text("def main(): pass")
        (project_dir / "utils.py").write_text("def util(): pass")
        
        result = analyzer.analyze_project(project_dir)
        
        assert "files" in result
        assert "total_issues" in result
        assert len(result["files"]) == 2

    def test_get_quality_score(self, temp_dir):
        """Test getting quality score"""
        analyzer = CodeQualityAnalyzer()
        
        test_file = temp_dir / "good_code.py"
        test_file.write_text("""
def well_named_function():
    return True
""")
        
        result = analyzer.analyze_file(test_file)
        score = analyzer.get_quality_score(result)
        
        assert 0 <= score <= 100

