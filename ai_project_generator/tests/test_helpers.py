"""
Test helpers and utilities for AI Project Generator tests
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import ast
import re


class TestHelpers:
    """Helper class with utility methods for tests"""
    
    @staticmethod
    def assert_project_structure(project_path: Path, required_files: List[str] = None):
        """Assert that project structure exists"""
        if required_files is None:
            required_files = ["README.md"]
        
        for file_path in required_files:
            full_path = project_path / file_path
            assert full_path.exists(), f"Required file {file_path} not found in {project_path}"
    
    @staticmethod
    def assert_valid_json(file_path: Path):
        """Assert that a file contains valid JSON"""
        assert file_path.exists(), f"File {file_path} does not exist"
        try:
            json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            pytest.fail(f"File {file_path} contains invalid JSON: {e}")
    
    @staticmethod
    def assert_valid_python(file_path: Path):
        """Assert that a file contains valid Python syntax"""
        assert file_path.exists(), f"File {file_path} does not exist"
        try:
            compile(file_path.read_text(encoding="utf-8"), str(file_path), "exec")
        except SyntaxError as e:
            pytest.fail(f"File {file_path} contains invalid Python: {e}")
    
    @staticmethod
    def assert_file_contains(file_path: Path, expected_content: str):
        """Assert that a file contains expected content"""
        assert file_path.exists(), f"File {file_path} does not exist"
        content = file_path.read_text(encoding="utf-8")
        assert expected_content in content, f"Expected content '{expected_content}' not found in {file_path}"
    
    @staticmethod
    def assert_file_not_contains(file_path: Path, unexpected_content: str):
        """Assert that a file does not contain unexpected content"""
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            assert unexpected_content not in content, f"Unexpected content '{unexpected_content}' found in {file_path}"
    
    @staticmethod
    def count_files(directory: Path, pattern: str = "*") -> int:
        """Count files in a directory matching a pattern"""
        if not directory.exists():
            return 0
        return len(list(directory.glob(pattern)))
    
    @staticmethod
    def count_lines(file_path: Path) -> int:
        """Count lines in a file"""
        if not file_path.exists():
            return 0
        return len(file_path.read_text(encoding="utf-8").splitlines())
    
    @staticmethod
    def extract_imports(file_path: Path) -> List[str]:
        """Extract import statements from a Python file"""
        if not file_path.exists():
            return []
        
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"))
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            return imports
        except SyntaxError:
            return []
    
    @staticmethod
    def assert_imports_present(file_path: Path, required_imports: List[str]):
        """Assert that required imports are present in a Python file"""
        imports = TestHelpers.extract_imports(file_path)
        for required in required_imports:
            assert any(required in imp for imp in imports), f"Required import '{required}' not found in {file_path}"
    
    @staticmethod
    def create_test_project_structure(base_dir: Path, structure: Dict[str, Any]):
        """Create a test project structure from a dictionary"""
        for key, value in structure.items():
            path = base_dir / key
            if isinstance(value, dict):
                path.mkdir(parents=True, exist_ok=True)
                TestHelpers.create_test_project_structure(path, value)
            elif isinstance(value, str):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value, encoding="utf-8")
    
    @staticmethod
    def assert_response_success(response, expected_status: int = 200):
        """Assert that an API response is successful"""
        assert response.status_code == expected_status, f"Expected status {expected_status}, got {response.status_code}"
        if response.status_code == 200:
            assert response.json() is not None, "Response JSON is None"
    
    @staticmethod
    def assert_response_error(response, expected_status: int = 400):
        """Assert that an API response is an error"""
        assert response.status_code == expected_status, f"Expected error status {expected_status}, got {response.status_code}"
        data = response.json()
        assert "error" in data or "detail" in data, "Error response missing error/detail field"
    
    @staticmethod
    def generate_large_description(size: int = 1000) -> str:
        """Generate a large description for testing"""
        words = ["AI", "system", "project", "application", "service", "platform"]
        return " ".join([words[i % len(words)] for i in range(size)])
    
    @staticmethod
    def generate_special_chars_string() -> str:
        """Generate a string with special characters"""
        return "!@#$%^&*()_+-=[]{}|;':\",./<>?`~"
    
    @staticmethod
    def generate_unicode_string() -> str:
        """Generate a string with unicode characters"""
        return "测试项目 🚀 プロジェクト тест"
    
    @staticmethod
    def assert_dict_contains(dictionary: Dict, required_keys: List[str]):
        """Assert that a dictionary contains required keys"""
        for key in required_keys:
            assert key in dictionary, f"Required key '{key}' not found in dictionary"
    
    @staticmethod
    def assert_dict_structure(dictionary: Dict, structure: Dict[str, type]):
        """Assert that a dictionary has the expected structure with types"""
        for key, expected_type in structure.items():
            assert key in dictionary, f"Key '{key}' not found in dictionary"
            assert isinstance(dictionary[key], expected_type), \
                f"Key '{key}' has type {type(dictionary[key])}, expected {expected_type}"

