"""
Script to validate test suite integrity
"""

import sys
from pathlib import Path
import ast
import importlib.util


def validate_test_files():
    """Validate that all test files are syntactically correct"""
    tests_dir = Path(__file__).parent
    errors = []
    
    for test_file in tests_dir.glob("test_*.py"):
        try:
            # Try to parse the file
            content = test_file.read_text(encoding="utf-8")
            ast.parse(content)
            print(f"✓ {test_file.name}")
        except SyntaxError as e:
            errors.append(f"✗ {test_file.name}: {e}")
            print(f"✗ {test_file.name}: Syntax error at line {e.lineno}")
        except Exception as e:
            errors.append(f"✗ {test_file.name}: {e}")
            print(f"✗ {test_file.name}: {e}")
    
    if errors:
        print(f"\n{len(errors)} error(s) found:")
        for error in errors:
            print(f"  {error}")
        return False
    
    print(f"\n✓ All test files are valid!")
    return True


def check_test_coverage():
    """Check that test files cover expected modules"""
    tests_dir = Path(__file__).parent
    test_files = list(tests_dir.glob("test_*.py"))
    
    print(f"\nFound {len(test_files)} test files")
    print(f"Expected: 60+ test files")
    
    if len(test_files) >= 60:
        print("✓ Test coverage is good!")
        return True
    else:
        print(f"⚠ Only {len(test_files)} test files found")
        return False


if __name__ == "__main__":
    print("Validating test suite...")
    print("=" * 70)
    
    syntax_ok = validate_test_files()
    coverage_ok = check_test_coverage()
    
    print("=" * 70)
    
    if syntax_ok and coverage_ok:
        print("✓ Test suite validation passed!")
        sys.exit(0)
    else:
        print("✗ Test suite validation failed!")
        sys.exit(1)

