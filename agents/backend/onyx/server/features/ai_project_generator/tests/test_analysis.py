"""
Analysis and reporting tests
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime


class TestAnalysis:
    """Tests for analysis and reporting"""
    
    def test_project_statistics(self, temp_dir):
        """Test project statistics generation"""
        # Create sample project structure
        project_dir = temp_dir / "project"
        project_dir.mkdir()
        
        # Create files
        (project_dir / "file1.py").write_text("code")
        (project_dir / "file2.py").write_text("more code")
        (project_dir / "README.md").write_text("# Project")
        
        # Calculate statistics
        stats = {
            "total_files": len(list(project_dir.rglob("*"))),
            "python_files": len(list(project_dir.glob("*.py"))),
            "markdown_files": len(list(project_dir.glob("*.md"))),
        }
        
        assert stats["total_files"] >= 3
        assert stats["python_files"] == 2
        assert stats["markdown_files"] == 1
    
    def test_code_metrics(self, temp_dir):
        """Test code metrics calculation"""
        code_file = temp_dir / "code.py"
        code_content = """
def function1():
    return 1

def function2(x, y):
    result = x + y
    return result

class MyClass:
    def method(self):
        pass
"""
        code_file.write_text(code_content)
        
        # Basic metrics
        content = code_file.read_text(encoding="utf-8")
        lines = content.split("\n")
        
        metrics = {
            "total_lines": len(lines),
            "code_lines": len([l for l in lines if l.strip() and not l.strip().startswith("#")]),
            "functions": content.count("def "),
            "classes": content.count("class "),
        }
        
        assert metrics["total_lines"] > 0
        assert metrics["functions"] >= 2
        assert metrics["classes"] >= 1
    
    def test_trend_analysis(self):
        """Test trend analysis"""
        data_points = [
            {"date": "2024-01-01", "value": 10},
            {"date": "2024-01-02", "value": 15},
            {"date": "2024-01-03", "value": 20},
            {"date": "2024-01-04", "value": 18},
            {"date": "2024-01-05", "value": 25},
        ]
        
        # Calculate trend
        values = [d["value"] for d in data_points]
        trend = "increasing" if values[-1] > values[0] else "decreasing"
        avg_value = sum(values) / len(values)
        
        assert trend == "increasing"
        assert avg_value > 0
    
    def test_comparison_analysis(self, temp_dir):
        """Test comparison analysis"""
        # Create two projects
        project1_dir = temp_dir / "project1"
        project2_dir = temp_dir / "project2"
        project1_dir.mkdir()
        project2_dir.mkdir()
        
        (project1_dir / "file.py").write_text("code1")
        (project2_dir / "file.py").write_text("code2")
        
        # Compare
        files1 = set(f.name for f in project1_dir.iterdir())
        files2 = set(f.name for f in project2_dir.iterdir())
        
        common = files1 & files2
        unique1 = files1 - files2
        unique2 = files2 - files1
        
        assert len(common) >= 1
        assert len(unique1) == 0
        assert len(unique2) == 0
    
    def test_quality_metrics(self, temp_dir):
        """Test quality metrics calculation"""
        project_dir = temp_dir / "project"
        project_dir.mkdir()
        
        # Create files with different quality indicators
        (project_dir / "good.py").write_text("def good_function():\n    return True")
        (project_dir / "README.md").write_text("# Project\n\nDescription")
        
        # Calculate quality metrics
        metrics = {
            "has_readme": (project_dir / "README.md").exists(),
            "has_python_files": len(list(project_dir.glob("*.py"))) > 0,
            "total_files": len(list(project_dir.iterdir())),
        }
        
        assert metrics["has_readme"] is True
        assert metrics["has_python_files"] is True
        assert metrics["total_files"] >= 2

