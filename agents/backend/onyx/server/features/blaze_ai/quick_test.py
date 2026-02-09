#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test script for Blaze AI Plugin System.
Very simple version for basic verification.
"""

import sys
import os

def test_basic_functionality():
    """Test basic Python functionality."""
    print("Testing basic functionality...")
    
    try:
        # Test basic imports
        import tempfile
        import shutil
        print("  [OK] Basic imports successful")
        
        # Test file operations
        test_file = "test_temp.txt"
        with open(test_file, 'w') as f:
            f.write("test")
        
        if os.path.exists(test_file):
            print("  [OK] File write successful")
            os.remove(test_file)
            print("  [OK] File delete successful")
        else:
            print("  [ERROR] File write failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"  [ERROR] Basic functionality test failed: {e}")
        return False

def test_file_structure():
    """Test that key files exist."""
    print("\nTesting file structure...")
    
    key_files = [
        "engines/__init__.py",
        "engines/plugins.py",
        "engines/base.py",
        "engines/factory.py"
    ]
    
    passed = 0
    total = len(key_files)
    
    for file_path in key_files:
        if os.path.exists(file_path):
            print(f"  [OK] {file_path} exists")
            passed += 1
        else:
            print(f"  [ERROR] {file_path} missing")
    
    print(f"  [INFO] {passed}/{total} key files found")
    return passed == total

def test_python_compilation():
    """Test that Python files can be compiled."""
    print("\nTesting Python compilation...")
    
    try:
        import py_compile
        
        files_to_test = [
            "engines/__init__.py",
            "engines/plugins.py"
        ]
        
        passed = 0
        total = len(files_to_test)
        
        for file_path in files_to_test:
            if os.path.exists(file_path):
                try:
                    py_compile.compile(file_path, doraise=True)
                    print(f"  [OK] {file_path} compiled successfully")
                    passed += 1
                except Exception as e:
                    print(f"  [ERROR] {file_path} compilation failed: {e}")
            else:
                print(f"  [WARN] {file_path} not found")
        
        print(f"  [INFO] {passed}/{total} files compiled successfully")
        return passed > 0
        
    except ImportError:
        print("  [WARN] py_compile module not available")
        return True  # Skip this test if module not available
    except Exception as e:
        print(f"  [ERROR] Compilation test failed: {e}")
        return False

def main():
    """Main test function."""
    print("Blaze AI Quick Test")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("File Structure", test_file_structure),
        ("Python Compilation", test_python_compilation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"  [SUCCESS] {test_name} passed")
            else:
                print(f"  [FAILURE] {test_name} failed")
        except Exception as e:
            print(f"  [ERROR] {test_name} crashed: {e}")
    
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        return True
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[INFO] Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        sys.exit(1)
