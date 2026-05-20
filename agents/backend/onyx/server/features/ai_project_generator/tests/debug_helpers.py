"""
Debugging helpers for tests
"""

import pytest
import traceback
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import pprint


class DebugHelpers:
    """Helpers for debugging tests"""
    
    @staticmethod
    def print_test_info(test_name: str, **kwargs):
        """Print test information for debugging"""
        print(f"\n{'='*70}")
        print(f"Test: {test_name}")
        print(f"{'='*70}")
        for key, value in kwargs.items():
            print(f"{key}: {value}")
        print(f"{'='*70}\n")
    
    @staticmethod
    def print_project_structure(project_path: Path, max_depth: int = 3):
        """Print project structure for debugging"""
        print(f"\nProject Structure: {project_path}")
        print("-" * 70)
        
        def print_tree(path: Path, prefix: str = "", depth: int = 0):
            if depth > max_depth:
                return
            
            if path.is_dir():
                print(f"{prefix}📁 {path.name}/")
                try:
                    items = sorted(path.iterdir())
                    for i, item in enumerate(items):
                        is_last = i == len(items) - 1
                        new_prefix = prefix + ("└── " if is_last else "├── ")
                        print_tree(item, new_prefix, depth + 1)
                except PermissionError:
                    print(f"{prefix}   [Permission Denied]")
            else:
                size = path.stat().st_size if path.exists() else 0
                print(f"{prefix}📄 {path.name} ({size} bytes)")
        
        print_tree(project_path)
        print("-" * 70 + "\n")
    
    @staticmethod
    def print_file_content(file_path: Path, max_lines: int = 50):
        """Print file content for debugging"""
        if not file_path.exists():
            print(f"File {file_path} does not exist")
            return
        
        print(f"\nFile Content: {file_path}")
        print("-" * 70)
        
        lines = file_path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines[:max_lines], 1):
            print(f"{i:4d} | {line}")
        
        if len(lines) > max_lines:
            print(f"... ({len(lines) - max_lines} more lines)")
        
        print("-" * 70 + "\n")
    
    @staticmethod
    def print_dict_pretty(data: Dict, title: str = "Data"):
        """Print dictionary in a pretty format"""
        print(f"\n{title}:")
        print("-" * 70)
        pprint.pprint(data, indent=2, width=70)
        print("-" * 70 + "\n")
    
    @staticmethod
    def save_debug_info(file_path: Path, data: Dict):
        """Save debug information to file"""
        debug_file = file_path.parent / f"{file_path.stem}_debug.json"
        debug_file.write_text(
            json.dumps(data, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"Debug info saved to: {debug_file}")
    
    @staticmethod
    def capture_exception_info(exception: Exception) -> Dict[str, Any]:
        """Capture exception information"""
        return {
            "type": type(exception).__name__,
            "message": str(exception),
            "traceback": traceback.format_exc(),
            "args": exception.args
        }
    
    @staticmethod
    def print_exception_info(exception: Exception):
        """Print exception information for debugging"""
        print(f"\n{'='*70}")
        print(f"Exception: {type(exception).__name__}")
        print(f"{'='*70}")
        print(f"Message: {str(exception)}")
        print(f"\nTraceback:")
        print(traceback.format_exc())
        print(f"{'='*70}\n")


@pytest.fixture
def debug():
    """Fixture for debug helpers"""
    return DebugHelpers


@pytest.fixture(autouse=True)
def debug_on_failure(request):
    """Automatically debug on test failure"""
    yield
    
    if hasattr(request.node, 'rep_call') and request.node.rep_call.failed:
        # Test failed, print debug info
        print("\n" + "="*70)
        print("TEST FAILED - Debug Information")
        print("="*70)
        
        # Print test name
        print(f"Test: {request.node.name}")
        
        # Print captured output if available
        if hasattr(request.node.rep_call, 'longrepr'):
            print(f"\nError:\n{request.node.rep_call.longrepr}")

