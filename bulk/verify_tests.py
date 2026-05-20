"""
Quick verification script to check if all test files are syntactically correct
and can be imported without errors.
"""

import sys
import ast
import os
from pathlib import Path

def check_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def check_imports(file_path):
    """Check if file can be imported (basic check)."""
    try:
        # Just check if we can parse it as a module
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Check for common import issues
        if 'import websockets' in code and 'WEBSOCKETS_AVAILABLE' not in code:
            return False, "websockets imported without availability check"
        
        return True, None
    except Exception as e:
        return False, f"Import check error: {str(e)}"

def main():
    """Verify all test files."""
    print("=" * 70)
    print("Test Files Verification")
    print("=" * 70)
    print()
    
    test_files = [
        "test_api_responses.py",
        "test_api_advanced.py",
        "test_security.py",
    ]
    
    all_ok = True
    results = {}
    
    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"❌ {test_file}: NOT FOUND")
            all_ok = False
            results[test_file] = "NOT FOUND"
            continue
        
        print(f"Checking {test_file}...")
        
        # Check syntax
        syntax_ok, syntax_error = check_syntax(test_file)
        if not syntax_ok:
            print(f"  ❌ Syntax Error: {syntax_error}")
            all_ok = False
            results[test_file] = f"Syntax Error: {syntax_error}"
            continue
        
        # Check imports
        import_ok, import_error = check_imports(test_file)
        if not import_ok:
            print(f"  ⚠️  Import Warning: {import_error}")
            results[test_file] = f"Warning: {import_error}"
        else:
            print(f"  ✅ OK")
            results[test_file] = "OK"
    
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    
    for test_file, status in results.items():
        if status == "OK":
            print(f"✅ {test_file}")
        else:
            print(f"❌ {test_file}: {status}")
    
    print()
    
    if all_ok:
        print("✅ All test files are syntactically correct!")
        print()
        print("To run tests:")
        print("  1. Start server: python api_frontend_ready.py")
        print("  2. Run tests: python test_api_responses.py")
        print("  3. Or use: run_all_tests.bat")
        return 0
    else:
        print("❌ Some test files have issues. Please review above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())









