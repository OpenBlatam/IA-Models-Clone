"""
Custom matchers for more expressive assertions
"""

import pytest
from typing import Any, Dict, List, Optional
import re


class CustomMatchers:
    """Custom matchers for more readable assertions"""
    
    @staticmethod
    def match_project_structure(project_path, expected_structure: Dict[str, Any]) -> bool:
        """Match project structure against expected structure"""
        from pathlib import Path
        path = Path(project_path)
        
        for key, value in expected_structure.items():
            item_path = path / key
            
            if isinstance(value, dict):
                # It's a directory
                if not item_path.exists() or not item_path.is_dir():
                    return False
                # Recursively check sub-structure
                if not CustomMatchers.match_project_structure(item_path, value):
                    return False
            elif isinstance(value, str):
                # It's a file with expected content
                if not item_path.exists() or not item_path.is_file():
                    return False
                if value:  # If content specified
                    actual_content = item_path.read_text(encoding="utf-8")
                    if value not in actual_content:
                        return False
            elif value is True:
                # Just check existence
                if not item_path.exists():
                    return False
        
        return True
    
    @staticmethod
    def match_json_structure(data: Dict, pattern: Dict[str, Any]) -> bool:
        """Match JSON structure against pattern"""
        for key, expected in pattern.items():
            if key not in data:
                return False
            
            if isinstance(expected, type):
                if not isinstance(data[key], expected):
                    return False
            elif isinstance(expected, dict):
                if not isinstance(data[key], dict):
                    return False
                if not CustomMatchers.match_json_structure(data[key], expected):
                    return False
            elif callable(expected):
                if not expected(data[key]):
                    return False
            else:
                if data[key] != expected:
                    return False
        
        return True
    
    @staticmethod
    def match_regex(content: str, pattern: str) -> bool:
        """Match content against regex pattern"""
        return bool(re.search(pattern, content))
    
    @staticmethod
    def match_file_pattern(file_path, pattern: str) -> bool:
        """Match file path against pattern"""
        from pathlib import Path
        return bool(Path(file_path).match(pattern))


@pytest.fixture
def match():
    """Fixture for custom matchers"""
    return CustomMatchers

