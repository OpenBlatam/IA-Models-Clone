#!/usr/bin/env python3
"""
Verification Script - FileStorage Refactoring
Verifies that all refactoring requirements are met
"""

import sys
import ast
import inspect
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.file_storage import FileStorage


def check_context_managers(source_code: str) -> tuple[bool, list[str]]:
    """Check if context managers are used for file operations"""
    issues = []
    has_with_write = 'with open' in source_code and 'def write' in source_code
    has_with_read = 'with open' in source_code and 'def read' in source_code
    
    if not has_with_write:
        issues.append("write() method does not use 'with open' context manager")
    if not has_with_read:
        issues.append("read() method does not use 'with open' context manager")
    
    # Check for old pattern (f = open(...))
    if 'f = open(' in source_code:
        issues.append("Found old pattern 'f = open(' - should use context manager")
    
    return len(issues) == 0, issues


def check_indentation(source_code: str) -> tuple[bool, list[str]]:
    """Check for indentation issues"""
    issues = []
    lines = source_code.split('\n')
    
    # Check for common indentation problems
    for i, line in enumerate(lines, 1):
        if 'def read(' in line or 'def update(' in line:
            # Check next few lines for proper indentation
            for j in range(i, min(i + 10, len(lines))):
                next_line = lines[j]
                if next_line.strip() and not next_line.startswith(' ') and not next_line.startswith('\t'):
                    if 'def ' not in next_line and 'class ' not in next_line:
                        if j > i + 1:  # Allow docstring
                            issues.append(f"Possible indentation issue at line {j+1}")
                            break
    
    return len(issues) == 0, issues


def check_update_function(source_code: str) -> tuple[bool, list[str]]:
    """Check if update() function properly writes back to file"""
    issues = []
    
    # Check if update calls write
    has_update_write = 'def update' in source_code and 'self.write(' in source_code
    
    if not has_update_write:
        issues.append("update() method does not call self.write() to save changes")
    
    # Check if update uses .get() for safe access
    if 'record[\'id\']' in source_code and 'record.get(\'id\')' not in source_code:
        issues.append("update() should use record.get('id') for safe dictionary access")
    
    return len(issues) == 0, issues


def check_error_handling(source_code: str) -> tuple[bool, list[str]]:
    """Check for appropriate error handling"""
    issues = []
    
    # Check for try-except blocks
    if 'def write(' in source_code:
        if 'try:' not in source_code or 'except' not in source_code:
            issues.append("write() method should have try-except error handling")
    
    if 'def read(' in source_code:
        if 'try:' not in source_code or 'except' not in source_code:
            issues.append("read() method should have try-except error handling")
    
    if 'def update(' in source_code:
        if 'try:' not in source_code or 'except' not in source_code:
            issues.append("update() method should have try-except error handling")
    
    # Check for input validation
    if 'def write(' in source_code:
        if 'isinstance(data' not in source_code:
            issues.append("write() should validate input type")
    
    if 'def update(' in source_code:
        if 'isinstance(record_id' not in source_code:
            issues.append("update() should validate record_id type")
    
    return len(issues) == 0, issues


def check_type_hints(source_code: str) -> tuple[bool, list[str]]:
    """Check for type hints"""
    issues = []
    
    methods = ['write', 'read', 'update']
    for method in methods:
        if f'def {method}(' in source_code:
            # Check if return type hint exists
            if '->' not in source_code.split(f'def {method}(')[1].split('\n')[0]:
                issues.append(f"{method}() method should have return type hint")
    
    return len(issues) == 0, issues


def run_functional_tests() -> tuple[bool, list[str]]:
    """Run functional tests to verify behavior"""
    issues = []
    import tempfile
    import os
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        storage = FileStorage(temp_path)
        
        # Test 1: Write and read
        test_data = [{"id": "1", "name": "Test"}]
        storage.write(test_data)
        read_data = storage.read()
        if read_data != test_data:
            issues.append("write()/read() not working correctly")
        
        # Test 2: Update
        success = storage.update("1", {"name": "Updated"})
        if not success:
            issues.append("update() returned False when it should return True")
        
        updated_data = storage.read()
        if updated_data[0]["name"] != "Updated":
            issues.append("update() did not properly save changes to file")
        
        # Test 3: Error handling
        try:
            storage.write("not a list")
            issues.append("write() should raise TypeError for invalid input")
        except TypeError:
            pass  # Expected
        
        try:
            storage.update(123, {"key": "value"})
            issues.append("update() should raise TypeError for invalid record_id")
        except TypeError:
            pass  # Expected
        
        # Cleanup
        os.unlink(temp_path)
        
    except Exception as e:
        issues.append(f"Functional test failed: {str(e)}")
    
    return len(issues) == 0, issues


def main():
    """Run all verification checks"""
    print("=" * 70)
    print("FileStorage Refactoring Verification")
    print("=" * 70)
    
    # Get source code
    source_file = Path(__file__).parent.parent / "utils" / "file_storage.py"
    with open(source_file, 'r', encoding='utf-8') as f:
        source_code = f.read()
    
    all_passed = True
    results = []
    
    print("\n1. Checking Context Managers...")
    passed, issues = check_context_managers(source_code)
    if passed:
        print("   ✓ Context managers are used correctly")
    else:
        print("   ✗ Issues found:")
        for issue in issues:
            print(f"     - {issue}")
        all_passed = False
    results.append(("Context Managers", passed, issues))
    
    print("\n2. Checking Indentation...")
    passed, issues = check_indentation(source_code)
    if passed:
        print("   ✓ Indentation is correct")
    else:
        print("   ✗ Issues found:")
        for issue in issues:
            print(f"     - {issue}")
        all_passed = False
    results.append(("Indentation", passed, issues))
    
    print("\n3. Checking update() Function...")
    passed, issues = check_update_function(source_code)
    if passed:
        print("   ✓ update() function is correct")
    else:
        print("   ✗ Issues found:")
        for issue in issues:
            print(f"     - {issue}")
        all_passed = False
    results.append(("update() Function", passed, issues))
    
    print("\n4. Checking Error Handling...")
    passed, issues = check_error_handling(source_code)
    if passed:
        print("   ✓ Error handling is appropriate")
    else:
        print("   ✗ Issues found:")
        for issue in issues:
            print(f"     - {issue}")
        all_passed = False
    results.append(("Error Handling", passed, issues))
    
    print("\n5. Checking Type Hints...")
    passed, issues = check_type_hints(source_code)
    if passed:
        print("   ✓ Type hints are present")
    else:
        print("   ⚠ Minor issues (not critical):")
        for issue in issues:
            print(f"     - {issue}")
    results.append(("Type Hints", passed, issues))
    
    print("\n6. Running Functional Tests...")
    passed, issues = run_functional_tests()
    if passed:
        print("   ✓ All functional tests passed")
    else:
        print("   ✗ Test failures:")
        for issue in issues:
            print(f"     - {issue}")
        all_passed = False
    results.append(("Functional Tests", passed, issues))
    
    # Summary
    print("\n" + "=" * 70)
    print("Verification Summary")
    print("=" * 70)
    
    for name, passed, issues in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {name}")
        if issues:
            for issue in issues:
                print(f"      {issue}")
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Refactoring is complete!")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Please review the issues above")
        return 1


if __name__ == "__main__":
    sys.exit(main())


