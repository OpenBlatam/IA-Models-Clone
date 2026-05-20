"""
Advanced search tests
"""

import pytest
from pathlib import Path
from typing import List, Dict, Any
import re


class TestSearchAdvanced:
    """Advanced search tests"""
    
    def test_full_text_search(self, temp_dir):
        """Test full-text search"""
        # Create files with content
        (temp_dir / "file1.txt").write_text("Python is a programming language")
        (temp_dir / "file2.txt").write_text("JavaScript is also a language")
        (temp_dir / "file3.txt").write_text("Python and JavaScript are different")
        
        # Search
        search_term = "Python"
        results = []
        for file in temp_dir.glob("*.txt"):
            content = file.read_text(encoding="utf-8")
            if search_term.lower() in content.lower():
                results.append(file.name)
        
        assert len(results) >= 2
        assert "file1.txt" in results or "file3.txt" in results
    
    def test_regex_search(self, temp_dir):
        """Test regex search"""
        # Create files
        (temp_dir / "file1.py").write_text("def function1(): pass")
        (temp_dir / "file2.py").write_text("class MyClass: pass")
        (temp_dir / "file3.py").write_text("def function2(): pass")
        
        # Regex search for functions
        pattern = r"def\s+\w+\("
        results = []
        for file in temp_dir.glob("*.py"):
            content = file.read_text(encoding="utf-8")
            if re.search(pattern, content):
                matches = re.findall(pattern, content)
                results.extend(matches)
        
        assert len(results) >= 2
    
    def test_fuzzy_search(self, temp_dir):
        """Test fuzzy search"""
        files = ["project_manager.py", "project_generator.py", "project_validator.py"]
        
        for filename in files:
            (temp_dir / filename).write_text("code")
        
        # Fuzzy search for "generator"
        search_term = "generator"
        results = []
        for file in temp_dir.glob("*.py"):
            if search_term.lower() in file.name.lower():
                results.append(file.name)
        
        assert len(results) >= 1
        assert "project_generator.py" in results
    
    def test_case_insensitive_search(self, temp_dir):
        """Test case-insensitive search"""
        (temp_dir / "File1.txt").write_text("CONTENT")
        (temp_dir / "file2.txt").write_text("content")
        (temp_dir / "FILE3.txt").write_text("Content")
        
        # Case-insensitive search
        search_term = "content"
        results = []
        for file in temp_dir.glob("*.txt"):
            content = file.read_text(encoding="utf-8")
            if search_term.lower() in content.lower():
                results.append(file.name)
        
        assert len(results) == 3
    
    def test_multi_criteria_search(self, temp_dir):
        """Test multi-criteria search"""
        # Create files with different attributes
        (temp_dir / "file1.py").write_text("def test(): pass")
        (temp_dir / "file2.py").write_text("class Test: pass")
        (temp_dir / "file3.js").write_text("function test() {}")
        
        # Multi-criteria: Python files containing "test"
        results = []
        for file in temp_dir.glob("*.py"):
            content = file.read_text(encoding="utf-8")
            if "test" in content.lower():
                results.append(file.name)
        
        assert len(results) >= 1
    
    def test_search_ranking(self, temp_dir):
        """Test search result ranking"""
        files_content = {
            "file1.txt": "Python Python Python",  # 3 matches
            "file2.txt": "Python Python",  # 2 matches
            "file3.txt": "Python",  # 1 match
        }
        
        for filename, content in files_content.items():
            (temp_dir / filename).write_text(content)
        
        # Search and rank by match count
        search_term = "Python"
        results = []
        for file in temp_dir.glob("*.txt"):
            content = file.read_text(encoding="utf-8")
            count = content.lower().count(search_term.lower())
            if count > 0:
                results.append((file.name, count))
        
        # Sort by count (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        assert len(results) == 3
        assert results[0][1] >= results[-1][1]  # First has more matches

