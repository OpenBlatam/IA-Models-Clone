#!/usr/bin/env python3
"""
Script to fix import paths in all test files.
Replaces the long import paths with relative imports.
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """Fix import paths in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix the long import paths
        old_pattern = r'from agents\.backend\.onyx\.server\.features\.heygen_ai\.(\w+) import'
        new_pattern = r'from \1 import'
        content = re.sub(old_pattern, new_pattern, content)
        
        # Fix import statements without 'from'
        old_pattern2 = r'import agents\.backend\.onyx\.server\.features\.heygen_ai\.(\w+) as'
        new_pattern2 = r'import \1 as'
        content = re.sub(old_pattern2, new_pattern2, content)
        
        # Fix import statements without 'as'
        old_pattern3 = r'import agents\.backend\.onyx\.server\.features\.heygen_ai\.(\w+)'
        new_pattern3 = r'import \1'
        content = re.sub(old_pattern3, new_pattern3, content)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {file_path}")
            return True
        else:
            print(f"No changes needed: {file_path}")
            return False
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def find_and_fix_test_files():
    """Find all test files and fix their imports."""
    tests_dir = Path(__file__).parent / "tests"
    
    if not tests_dir.exists():
        print(f"Tests directory not found: {tests_dir}")
        return
    
    # Find all Python test files
    test_files = []
    for pattern in ["**/*.py"]:
        test_files.extend(tests_dir.glob(pattern))
    
    print(f"Found {len(test_files)} test files")
    
    fixed_count = 0
    for test_file in test_files:
        if test_file.name.startswith('__'):
            continue
        
        if fix_imports_in_file(test_file):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} out of {len(test_files)} test files")

if __name__ == "__main__":
    find_and_fix_test_files()
