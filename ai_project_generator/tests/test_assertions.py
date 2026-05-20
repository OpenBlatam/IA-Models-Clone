"""
Custom assertions for better test readability
"""

import pytest
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import ast


class CustomAssertions:
    """Custom assertion methods for clearer test failures"""
    
    @staticmethod
    def assert_project_exists(project_path: Path, project_name: str = None):
        """Assert that a project exists and is valid"""
        assert project_path.exists(), f"Project path {project_path} does not exist"
        assert project_path.is_dir(), f"{project_path} exists but is not a directory"
        
        if project_name:
            assert project_path.name == project_name, \
                f"Project name mismatch: expected {project_name}, got {project_path.name}"
    
    @staticmethod
    def assert_backend_structure(project_path: Path):
        """Assert that backend structure is correct"""
        backend_path = project_path / "backend"
        assert backend_path.exists(), "Backend directory not found"
        
        main_py = backend_path / "main.py"
        assert main_py.exists(), "backend/main.py not found"
        
        requirements_txt = backend_path / "requirements.txt"
        assert requirements_txt.exists(), "backend/requirements.txt not found"
    
    @staticmethod
    def assert_frontend_structure(project_path: Path):
        """Assert that frontend structure is correct"""
        frontend_path = project_path / "frontend"
        assert frontend_path.exists(), "Frontend directory not found"
        
        package_json = frontend_path / "package.json"
        assert package_json.exists(), "frontend/package.json not found"
        
        # Validate package.json is valid JSON
        try:
            json.loads(package_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            pytest.fail(f"package.json is not valid JSON: {e}")
    
    @staticmethod
    def assert_file_content_contains(file_path: Path, expected_content: str, case_sensitive: bool = True):
        """Assert that file contains expected content"""
        assert file_path.exists(), f"File {file_path} does not exist"
        
        content = file_path.read_text(encoding="utf-8")
        if not case_sensitive:
            content = content.lower()
            expected_content = expected_content.lower()
        
        assert expected_content in content, \
            f"Expected content '{expected_content}' not found in {file_path}"
    
    @staticmethod
    def assert_file_content_not_contains(file_path: Path, unexpected_content: str):
        """Assert that file does not contain unexpected content"""
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            assert unexpected_content not in content, \
                f"Unexpected content '{unexpected_content}' found in {file_path}"
    
    @staticmethod
    def assert_python_imports(file_path: Path, required_imports: List[str]):
        """Assert that Python file contains required imports"""
        assert file_path.exists(), f"File {file_path} does not exist"
        
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
            
            for required in required_imports:
                assert any(required in imp or imp.endswith(required) for imp in imports), \
                    f"Required import '{required}' not found in {file_path}"
        except SyntaxError as e:
            pytest.fail(f"File {file_path} has syntax errors: {e}")
    
    @staticmethod
    def assert_dict_structure(data: Dict, structure: Dict[str, type], allow_extra: bool = True):
        """Assert that dictionary has expected structure with types"""
        for key, expected_type in structure.items():
            assert key in data, f"Required key '{key}' not found in dictionary"
            assert isinstance(data[key], expected_type), \
                f"Key '{key}' has type {type(data[key])}, expected {expected_type}"
        
        if not allow_extra:
            extra_keys = set(data.keys()) - set(structure.keys())
            assert len(extra_keys) == 0, f"Unexpected keys in dictionary: {extra_keys}"
    
    @staticmethod
    def assert_list_not_empty(items: List, item_name: str = "items"):
        """Assert that a list is not empty"""
        assert len(items) > 0, f"Expected {item_name} to not be empty"
    
    @staticmethod
    def assert_list_contains(items: List, expected_item: Any, item_name: str = "items"):
        """Assert that a list contains an expected item"""
        assert expected_item in items, \
            f"Expected {item_name} to contain {expected_item}, but got {items}"
    
    @staticmethod
    def assert_response_success(response, expected_status: int = 200):
        """Assert that API response is successful"""
        assert response.status_code == expected_status, \
            f"Expected status {expected_status}, got {response.status_code}. " \
            f"Response: {response.text[:200]}"
        
        if response.status_code == 200:
            try:
                data = response.json()
                assert data is not None, "Response JSON is None"
            except json.JSONDecodeError:
                pytest.fail(f"Response is not valid JSON: {response.text[:200]}")
    
    @staticmethod
    def assert_response_error(response, expected_status: int = 400):
        """Assert that API response is an error"""
        assert response.status_code == expected_status, \
            f"Expected error status {expected_status}, got {response.status_code}"
        
        data = response.json()
        assert "error" in data or "detail" in data, \
            f"Error response missing error/detail field. Response: {data}"


# Make assertions available as pytest fixtures
@pytest.fixture
def assert_project():
    """Fixture providing project assertions"""
    return CustomAssertions


@pytest.fixture
def assert_file():
    """Fixture providing file assertions"""
    return CustomAssertions


@pytest.fixture
def assert_api():
    """Fixture providing API assertions"""
    return CustomAssertions

