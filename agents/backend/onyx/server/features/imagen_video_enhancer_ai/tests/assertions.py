"""
Advanced Assertions
==================

Advanced assertion helpers for tests.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import json


class AssertionHelpers:
    """Advanced assertion helpers."""
    
    @staticmethod
    def assert_response_success(response: Dict[str, Any], expected_data: Optional[Dict[str, Any]] = None):
        """
        Assert API response is successful.
        
        Args:
            response: API response dictionary
            expected_data: Optional expected data
        """
        assert "success" in response, "Response missing 'success' field"
        assert response["success"] is True, f"Response not successful: {response}"
        
        if expected_data:
            assert "data" in response, "Response missing 'data' field"
            AssertionHelpers.assert_dict_contains(response["data"], expected_data)
    
    @staticmethod
    def assert_response_error(response: Dict[str, Any], expected_message: Optional[str] = None):
        """
        Assert API response is an error.
        
        Args:
            response: API response dictionary
            expected_message: Optional expected error message
        """
        assert "success" in response, "Response missing 'success' field"
        assert response["success"] is False, f"Response should be error: {response}"
        
        if expected_message:
            assert "message" in response, "Response missing 'message' field"
            assert expected_message in response["message"], \
                f"Error message mismatch: {response.get('message')} != {expected_message}"
    
    @staticmethod
    def assert_dict_contains(actual: Dict[str, Any], expected: Dict[str, Any]):
        """
        Assert dictionary contains expected keys and values.
        
        Args:
            actual: Actual dictionary
            expected: Expected dictionary
        """
        for key, value in expected.items():
            assert key in actual, f"Key '{key}' not found in actual dict"
            if isinstance(value, dict):
                AssertionHelpers.assert_dict_contains(actual[key], value)
            elif isinstance(value, list):
                assert isinstance(actual[key], list), f"Key '{key}' is not a list"
                assert len(actual[key]) == len(value), \
                    f"List length mismatch for '{key}': {len(actual[key])} != {len(value)}"
            else:
                assert actual[key] == value, \
                    f"Value mismatch for '{key}': {actual[key]} != {value}"
    
    @staticmethod
    def assert_list_contains(items: List[Any], item: Any):
        """
        Assert list contains item.
        
        Args:
            items: List to check
            item: Item to find
        """
        assert item in items, f"Item {item} not found in list"
    
    @staticmethod
    def assert_list_length(items: List[Any], expected_length: int):
        """
        Assert list has expected length.
        
        Args:
            items: List to check
            expected_length: Expected length
        """
        assert len(items) == expected_length, \
            f"List length mismatch: {len(items)} != {expected_length}"
    
    @staticmethod
    def assert_file_exists(file_path: Path, expected_size: Optional[int] = None):
        """
        Assert file exists and optionally check size.
        
        Args:
            file_path: File path
            expected_size: Optional expected file size
        """
        assert file_path.exists(), f"File {file_path} does not exist"
        assert file_path.is_file(), f"{file_path} is not a file"
        
        if expected_size is not None:
            actual_size = file_path.stat().st_size
            assert actual_size == expected_size, \
                f"File size mismatch: {actual_size} != {expected_size}"
    
    @staticmethod
    def assert_directory_exists(dir_path: Path, expected_files: Optional[List[str]] = None):
        """
        Assert directory exists and optionally check files.
        
        Args:
            dir_path: Directory path
            expected_files: Optional list of expected file names
        """
        assert dir_path.exists(), f"Directory {dir_path} does not exist"
        assert dir_path.is_dir(), f"{dir_path} is not a directory"
        
        if expected_files:
            actual_files = {f.name for f in dir_path.iterdir() if f.is_file()}
            expected_set = set(expected_files)
            assert expected_set.issubset(actual_files), \
                f"Missing files: {expected_set - actual_files}"
    
    @staticmethod
    def assert_json_file(file_path: Path, expected_data: Optional[Dict[str, Any]] = None):
        """
        Assert JSON file exists and optionally validate content.
        
        Args:
            file_path: JSON file path
            expected_data: Optional expected JSON data
        """
        AssertionHelpers.assert_file_exists(file_path)
        
        if expected_data:
            with open(file_path, 'r') as f:
                actual_data = json.load(f)
            AssertionHelpers.assert_dict_contains(actual_data, expected_data)
    
    @staticmethod
    def assert_task_status(task: Dict[str, Any], expected_status: str):
        """
        Assert task has expected status.
        
        Args:
            task: Task dictionary
            expected_status: Expected status
        """
        assert "status" in task, "Task missing 'status' field"
        assert task["status"] == expected_status, \
            f"Task status mismatch: {task['status']} != {expected_status}"
    
    @staticmethod
    def assert_task_has_result(task: Dict[str, Any]):
        """
        Assert task has result.
        
        Args:
            task: Task dictionary
        """
        assert "result" in task, "Task missing 'result' field"
        assert task["result"] is not None, "Task result is None"




