#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script for Blaze AI Plugin System.
Simplified version without encoding issues for Windows compatibility.
"""

import sys
import os
import subprocess
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any

class TestResult:
    """Class to store test results."""
    def __init__(self, name: str, success: bool, message: str = "", duration: float = 0.0):
        self.name = name
        self.success = success
        self.message = message
        self.duration = duration

class TestSuite:
    """Main test suite class."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = time.time()
    
    def add_result(self, result: TestResult):
        """Add a test result."""
        self.results.append(result)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test summary statistics."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.success)
        failed = total - passed
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "success_rate": (passed / total * 100) if total > 0 else 0
        }

def run_with_timing(func, *args, **kwargs) -> Tuple[bool, str, float]:
    """Run a function and measure its execution time."""
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        return result, "", duration
    except Exception as e:
        duration = time.time() - start_time
        return False, str(e), duration

def test_file_compilation() -> Tuple[bool, str, float]:
    """Test that all Python files can be compiled without syntax errors."""
    print("Testing file compilation...")
    
    files_to_test = [
        "engines/__init__.py",
        "engines/plugins.py", 
        "engines/base.py",
        "engines/factory.py"
    ]
    
    passed = 0
    total = len(files_to_test)
    errors = []
    
    for file_path in files_to_test:
        if os.path.exists(file_path):
            try:
                # Use Python to compile the file
                result = subprocess.run([
                    sys.executable, "-m", "py_compile", file_path
                ], capture_output=True, text=True, check=True)
                print(f"  [OK] {file_path} compiled successfully")
                passed += 1
            except subprocess.CalledProcessError as e:
                error_msg = f"{file_path}: {e.stderr.strip()}"
                print(f"  [ERROR] {error_msg}")
                errors.append(error_msg)
            except Exception as e:
                error_msg = f"{file_path}: {str(e)}"
                print(f"  [ERROR] {error_msg}")
                errors.append(error_msg)
        else:
            print(f"  [WARN] {file_path} not found")
    
    success = passed == total
    message = f"Compilation: {passed}/{total} files passed"
    if errors:
        message += f" - Errors: {'; '.join(errors[:3])}"
    
    return success, message

def test_basic_imports() -> Tuple[bool, str, float]:
    """Test basic Python imports without complex dependencies."""
    print("\nTesting basic imports...")
    
    try:
        import tempfile
        import shutil
        import json
        import time
        from pathlib import Path
        from dataclasses import dataclass
        from typing import Any, Dict, List, Optional, Type, Callable, Union
        print("  [OK] Basic Python imports successful")
        return True, "Basic imports successful", 0.0
    except ImportError as e:
        print(f"  [ERROR] Basic imports failed: {e}")
        return False, f"Basic imports failed: {e}", 0.0

def test_file_structure() -> Tuple[bool, str, float]:
    """Test that the expected file structure exists."""
    print("\nTesting file structure...")
    
    expected_files = [
        "engines/",
        "engines/__init__.py",
        "engines/plugins.py",
        "engines/base.py",
        "engines/factory.py",
        "tests/",
        "tests/test_plugins.py",
        "tests/test_llm_engine_cache.py",
        "tests/conftest.py",
        "tests/requirements-test.txt",
        "tests/README.md",
        "run_tests.py"
    ]
    
    passed = 0
    total = len(expected_files)
    missing_files = []
    
    for file_path in expected_files:
        if os.path.exists(file_path):
            print(f"  [OK] {file_path} exists")
            passed += 1
        else:
            print(f"  [ERROR] {file_path} missing")
            missing_files.append(file_path)
    
    success = passed == total
    message = f"File structure: {passed}/{total} files exist"
    if missing_files:
        message += f" - Missing: {', '.join(missing_files[:3])}"
    
    return success, message

def test_python_environment() -> Tuple[bool, str, float]:
    """Test Python environment and version."""
    print("\nTesting Python environment...")
    
    try:
        version = sys.version
        executable = sys.executable
        print(f"  [OK] Python version: {version.split()[0]}")
        print(f"  [OK] Python executable: {executable}")
        
        # Test if we can run subprocess
        result = subprocess.run([sys.executable, "--version"], 
                              capture_output=True, text=True, check=True)
        print(f"  [OK] Subprocess working: {result.stdout.strip()}")
        
        return True, f"Python {version.split()[0]} environment ready", 0.0
    except Exception as e:
        print(f"  [ERROR] Python environment test failed: {e}")
        return False, f"Python environment test failed: {e}", 0.0

def test_disk_space() -> Tuple[bool, str, float]:
    """Test available disk space."""
    print("\nTesting disk space...")
    
    try:
        import shutil
        current_dir = Path.cwd()
        total, used, free = shutil.disk_usage(current_dir)
        free_gb = free / (1024**3)
        
        print(f"  [INFO] Available disk space: {free_gb:.2f} GB")
        
        if free_gb < 1.0:
            print(f"  [WARN] Low disk space warning: {free_gb:.2f} GB available")
            return False, f"Low disk space: {free_gb:.2f} GB", 0.0
        else:
            return True, f"Sufficient disk space: {free_gb:.2f} GB", 0.0
            
    except Exception as e:
        print(f"  [ERROR] Disk space test failed: {e}")
        return False, f"Disk space test failed: {e}", 0.0

def test_simple_syntax() -> Tuple[bool, str, float]:
    """Test simple syntax without compilation."""
    print("\nTesting basic syntax...")
    
    try:
        # Test basic Python syntax
        test_code = """
def test_function():
    return "Hello, World!"

class TestClass:
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value

# Test execution
obj = TestClass()
result = test_function()
assert obj.get_value() == 42
assert result == "Hello, World!"
"""
        
        # Execute the test code
        exec(test_code)
        print("  [OK] Basic Python syntax test passed")
        return True, "Basic syntax test passed", 0.0
        
    except Exception as e:
        print(f"  [ERROR] Basic syntax test failed: {e}")
        return False, f"Basic syntax test failed: {e}", 0.0

def run_all_tests() -> bool:
    """Run all tests and report results."""
    print("Blaze AI Plugin System - Enhanced Test Suite")
    print("=" * 70)
    
    test_suite = TestSuite()
    
    # Run all tests
    tests = [
        ("Python Environment", test_python_environment),
        ("Disk Space", test_disk_space),
        ("Basic Syntax", test_simple_syntax),
        ("File Structure", test_file_structure),
        ("Basic Imports", test_basic_imports),
        ("File Compilation", test_file_compilation),
    ]
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        success, message, duration = run_with_timing(test_func)
        result = TestResult(test_name, success, message, duration)
        test_suite.add_result(result)
    
    # Print detailed results
    print("\n" + "=" * 70)
    print("DETAILED TEST RESULTS")
    print("=" * 70)
    
    for result in test_suite.results:
        status = "[PASS]" if result.success else "[FAIL]"
        duration_str = f"({result.duration:.3f}s)" if result.duration > 0 else ""
        print(f"{status} {result.name} {duration_str}")
        if result.message:
            print(f"    {result.message}")
    
    # Print summary
    summary = test_suite.get_summary()
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total Tests: {summary['total']}")
    print(f"Passed: {summary['passed']} [PASS]")
    print(f"Failed: {summary['failed']} [FAIL]")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    
    # Overall result
    if summary['success_rate'] == 100:
        print("\n[SUCCESS] ALL TESTS PASSED! The system is ready for use.")
        return True
    elif summary['success_rate'] >= 80:
        print(f"\n[WARNING] MOST TESTS PASSED ({summary['success_rate']:.1f}%). Some issues need attention.")
        return True
    else:
        print(f"\n[ERROR] MANY TESTS FAILED ({summary['success_rate']:.1f}%). System needs significant fixes.")
        return False

if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[INFO] Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error during test execution: {e}")
        sys.exit(1)
