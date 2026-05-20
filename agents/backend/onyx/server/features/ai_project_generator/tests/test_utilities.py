"""
Utility functions for tests
"""

import pytest
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import functools
import time


def skip_if_not_installed(package_name: str):
    """Decorator to skip test if package is not installed"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                __import__(package_name)
            except ImportError:
                pytest.skip(f"Package {package_name} is not installed")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def skip_if_env_not_set(env_var: str):
    """Decorator to skip test if environment variable is not set"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not os.getenv(env_var):
                pytest.skip(f"Environment variable {env_var} is not set")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
    """Decorator to retry test on failure"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
                    else:
                        raise last_exception
            return None
        return wrapper
    return decorator


def timeout(seconds: float):
    """Decorator to timeout test after specified seconds"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Test {func.__name__} timed out after {seconds} seconds")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))
            
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            
            return result
        return wrapper
    return decorator


class TestUtilities:
    """Utility class for common test operations"""
    
    @staticmethod
    def get_test_data_path(relative_path: str) -> Path:
        """Get path to test data file"""
        test_dir = Path(__file__).parent
        return test_dir / "test_data" / relative_path
    
    @staticmethod
    def load_test_data(filename: str) -> Dict[str, Any]:
        """Load test data from JSON file"""
        import json
        data_path = TestUtilities.get_test_data_path(filename)
        if data_path.exists():
            return json.loads(data_path.read_text(encoding="utf-8"))
        return {}
    
    @staticmethod
    def save_test_data(filename: str, data: Dict[str, Any]):
        """Save test data to JSON file"""
        import json
        data_path = TestUtilities.get_test_data_path(filename)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(
            json.dumps(data, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8"
        )
    
    @staticmethod
    def compare_files(file1: Path, file2: Path) -> Dict[str, Any]:
        """Compare two files and return differences"""
        if not file1.exists():
            return {"error": f"File {file1} does not exist"}
        if not file2.exists():
            return {"error": f"File {file2} does not exist"}
        
        content1 = file1.read_text(encoding="utf-8")
        content2 = file2.read_text(encoding="utf-8")
        
        lines1 = content1.splitlines()
        lines2 = content2.splitlines()
        
        differences = []
        max_lines = max(len(lines1), len(lines2))
        
        for i in range(max_lines):
            line1 = lines1[i] if i < len(lines1) else None
            line2 = lines2[i] if i < len(lines2) else None
            
            if line1 != line2:
                differences.append({
                    "line": i + 1,
                    "file1": line1,
                    "file2": line2
                })
        
        return {
            "identical": len(differences) == 0,
            "differences": differences,
            "file1_lines": len(lines1),
            "file2_lines": len(lines2)
        }
    
    @staticmethod
    def get_file_stats(file_path: Path) -> Dict[str, Any]:
        """Get statistics about a file"""
        if not file_path.exists():
            return {"error": "File does not exist"}
        
        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        
        return {
            "size_bytes": file_path.stat().st_size,
            "size_kb": file_path.stat().st_size / 1024,
            "line_count": len(lines),
            "non_empty_lines": len([l for l in lines if l.strip()]),
            "empty_lines": len([l for l in lines if not l.strip()]),
            "char_count": len(content),
            "word_count": len(content.split())
        }


@pytest.fixture
def test_utilities():
    """Fixture for test utilities"""
    return TestUtilities

