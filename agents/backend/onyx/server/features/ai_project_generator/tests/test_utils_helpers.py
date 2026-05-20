"""
Additional test utilities and helpers
"""

import pytest
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from contextlib import contextmanager
from unittest.mock import Mock, MagicMock, AsyncMock
import json
import tempfile
import shutil


class AsyncTestHelpers:
    """Helpers for async testing"""
    
    @staticmethod
    async def wait_for_condition(
        condition: Callable[[], bool],
        timeout: float = 5.0,
        interval: float = 0.1,
        error_message: str = "Condition not met within timeout"
    ):
        """Wait for a condition to be true"""
        start_time = time.time()
        while not condition():
            if time.time() - start_time > timeout:
                pytest.fail(error_message)
            await asyncio.sleep(interval)
    
    @staticmethod
    async def retry_async(
        func: Callable,
        max_attempts: int = 3,
        delay: float = 0.5,
        exceptions: tuple = (Exception,)
    ):
        """Retry an async function on failure"""
        last_exception = None
        for attempt in range(max_attempts):
            try:
                return await func()
            except exceptions as e:
                last_exception = e
                if attempt < max_attempts - 1:
                    await asyncio.sleep(delay)
                else:
                    raise last_exception
    
    @staticmethod
    async def timeout_async(
        func: Callable,
        timeout: float = 5.0,
        error_message: str = "Operation timed out"
    ):
        """Run async function with timeout"""
        try:
            return await asyncio.wait_for(func(), timeout=timeout)
        except asyncio.TimeoutError:
            pytest.fail(error_message)


class FileTestHelpers:
    """Helpers for file operations in tests"""
    
    @staticmethod
    def create_temp_file(content: str = "", suffix: str = ".txt") -> Path:
        """Create a temporary file with content"""
        fd, path = tempfile.mkstemp(suffix=suffix)
        with open(fd, 'w', encoding='utf-8') as f:
            f.write(content)
        return Path(path)
    
    @staticmethod
    def create_temp_directory(prefix: str = "test_") -> Path:
        """Create a temporary directory"""
        return Path(tempfile.mkdtemp(prefix=prefix))
    
    @staticmethod
    def cleanup_path(path: Path):
        """Cleanup a path (file or directory)"""
        try:
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass
    
    @staticmethod
    def assert_file_size(file_path: Path, min_size: int = 0, max_size: Optional[int] = None):
        """Assert file size is within range"""
        assert file_path.exists(), f"File {file_path} does not exist"
        size = file_path.stat().st_size
        assert size >= min_size, f"File {file_path} size {size} is less than minimum {min_size}"
        if max_size is not None:
            assert size <= max_size, f"File {file_path} size {size} exceeds maximum {max_size}"
    
    @staticmethod
    def assert_directory_not_empty(directory: Path):
        """Assert directory is not empty"""
        assert directory.exists(), f"Directory {directory} does not exist"
        assert directory.is_dir(), f"{directory} is not a directory"
        files = list(directory.iterdir())
        assert len(files) > 0, f"Directory {directory} is empty"


class MockHelpers:
    """Helpers for creating mocks"""
    
    @staticmethod
    def create_async_mock(return_value: Any = None, side_effect: Any = None):
        """Create an async mock"""
        mock = AsyncMock()
        if return_value is not None:
            mock.return_value = return_value
        if side_effect is not None:
            mock.side_effect = side_effect
        return mock
    
    @staticmethod
    def create_mock_response(status_code: int = 200, json_data: Dict = None, text: str = ""):
        """Create a mock HTTP response"""
        mock = MagicMock()
        mock.status_code = status_code
        mock.json.return_value = json_data or {}
        mock.text = text
        return mock
    
    @staticmethod
    def create_mock_project_data(project_id: str = "test-123", **kwargs) -> Dict[str, Any]:
        """Create mock project data"""
        default_data = {
            "project_id": project_id,
            "name": "test_project",
            "description": "A test project",
            "status": "completed",
            "created_at": "2024-01-01T00:00:00",
            "ai_type": "chat",
            "backend_framework": "fastapi",
            "frontend_framework": "react"
        }
        default_data.update(kwargs)
        return default_data


class PerformanceTestHelpers:
    """Helpers for performance testing"""
    
    @staticmethod
    @contextmanager
    def measure_time():
        """Context manager to measure execution time"""
        start = time.time()
        yield
        elapsed = time.time() - start
        return elapsed
    
    @staticmethod
    def assert_performance(
        elapsed_time: float,
        max_time: float,
        operation_name: str = "Operation"
    ):
        """Assert that operation completed within time limit"""
        assert elapsed_time <= max_time, \
            f"{operation_name} took {elapsed_time:.2f}s, exceeded limit of {max_time:.2f}s"
    
    @staticmethod
    def benchmark_function(func: Callable, iterations: int = 10) -> Dict[str, float]:
        """Benchmark a function"""
        times = []
        for _ in range(iterations):
            start = time.time()
            func()
            times.append(time.time() - start)
        
        return {
            "min": min(times),
            "max": max(times),
            "avg": sum(times) / len(times),
            "total": sum(times)
        }


class ValidationHelpers:
    """Helpers for validation in tests"""
    
    @staticmethod
    def validate_project_structure(project_path: Path, required_dirs: List[str] = None, 
                                   required_files: List[str] = None):
        """Validate project structure"""
        if required_dirs:
            for dir_name in required_dirs:
                dir_path = project_path / dir_name
                assert dir_path.exists(), f"Required directory {dir_name} not found"
                assert dir_path.is_dir(), f"{dir_name} exists but is not a directory"
        
        if required_files:
            for file_name in required_files:
                file_path = project_path / file_name
                assert file_path.exists(), f"Required file {file_name} not found"
                assert file_path.is_file(), f"{file_name} exists but is not a file"
    
    @staticmethod
    def validate_json_structure(data: Dict, required_keys: List[str], 
                               key_types: Dict[str, type] = None):
        """Validate JSON structure"""
        for key in required_keys:
            assert key in data, f"Required key '{key}' not found in JSON"
            if key_types and key in key_types:
                assert isinstance(data[key], key_types[key]), \
                    f"Key '{key}' has wrong type. Expected {key_types[key]}, got {type(data[key])}"
    
    @staticmethod
    def validate_api_response(response, expected_status: int = 200, 
                              required_fields: List[str] = None):
        """Validate API response"""
        assert response.status_code == expected_status, \
            f"Expected status {expected_status}, got {response.status_code}"
        
        if response.status_code == 200:
            data = response.json()
            if required_fields:
                for field in required_fields:
                    assert field in data, f"Required field '{field}' not in response"


class DataGenerators:
    """Helpers for generating test data"""
    
    @staticmethod
    def generate_project_name(prefix: str = "test", suffix: str = "") -> str:
        """Generate a unique project name"""
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        return f"{prefix}_{unique_id}{suffix}"
    
    @staticmethod
    def generate_large_string(size: int = 1000, char: str = "a") -> str:
        """Generate a large string"""
        return char * size
    
    @staticmethod
    def generate_project_description(ai_type: str = "chat", features: List[str] = None) -> str:
        """Generate a project description"""
        features = features or []
        desc = f"A {ai_type} AI system"
        if features:
            desc += f" with {', '.join(features)}"
        return desc
    
    @staticmethod
    def generate_random_dict(keys: List[str], value_generator: Callable = lambda: "value") -> Dict:
        """Generate a random dictionary"""
        return {key: value_generator() for key in keys}


# Export all helpers
__all__ = [
    'AsyncTestHelpers',
    'FileTestHelpers',
    'MockHelpers',
    'PerformanceTestHelpers',
    'ValidationHelpers',
    'DataGenerators'
]

