#!/usr/bin/env python3
"""
Gamma App - Test Script
Script to run tests and validate the system
"""

import os
import sys
import asyncio
import pytest
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_unit_tests():
    """Run unit tests"""
    print("🧪 Running unit tests...")
    
    test_dirs = [
        "tests/unit",
        "tests/integration",
        "tests/api"
    ]
    
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            print(f"📁 Running tests in {test_dir}")
            result = pytest.main([test_dir, "-v", "--tb=short"])
            if result != 0:
                print(f"❌ Tests failed in {test_dir}")
                return False
    
    print("✅ All unit tests passed!")
    return True

def run_integration_tests():
    """Run integration tests"""
    print("🔗 Running integration tests...")
    
    # Check if integration test directory exists
    if not Path("tests/integration").exists():
        print("⚠️  No integration tests found")
        return True
    
    result = pytest.main(["tests/integration", "-v", "--tb=short"])
    if result != 0:
        print("❌ Integration tests failed")
        return False
    
    print("✅ All integration tests passed!")
    return True

def run_api_tests():
    """Run API tests"""
    print("🌐 Running API tests...")
    
    # Check if API test directory exists
    if not Path("tests/api").exists():
        print("⚠️  No API tests found")
        return True
    
    result = pytest.main(["tests/api", "-v", "--tb=short"])
    if result != 0:
        print("❌ API tests failed")
        return False
    
    print("✅ All API tests passed!")
    return True

def run_coverage():
    """Run tests with coverage"""
    print("📊 Running tests with coverage...")
    
    result = pytest.main([
        "tests/",
        "--cov=gamma_app",
        "--cov-report=html",
        "--cov-report=term",
        "--cov-fail-under=80"
    ])
    
    if result != 0:
        print("❌ Coverage tests failed")
        return False
    
    print("✅ Coverage tests passed!")
    return True

def run_linting():
    """Run code linting"""
    print("🔍 Running code linting...")
    
    # Run flake8
    result = os.system("flake8 gamma_app/ --max-line-length=100 --ignore=E203,W503")
    if result != 0:
        print("❌ Linting failed")
        return False
    
    # Run black check
    result = os.system("black --check gamma_app/")
    if result != 0:
        print("❌ Black formatting check failed")
        return False
    
    # Run isort check
    result = os.system("isort --check-only gamma_app/")
    if result != 0:
        print("❌ Import sorting check failed")
        return False
    
    print("✅ All linting checks passed!")
    return True

def run_type_checking():
    """Run type checking"""
    print("🔍 Running type checking...")
    
    result = os.system("mypy gamma_app/ --ignore-missing-imports")
    if result != 0:
        print("❌ Type checking failed")
        return False
    
    print("✅ Type checking passed!")
    return True

def main():
    """Main test function"""
    print("🧪 Running Gamma App Tests...")
    print("=" * 50)
    
    tests = [
        ("Unit Tests", run_unit_tests),
        ("Integration Tests", run_integration_tests),
        ("API Tests", run_api_tests),
        ("Linting", run_linting),
        ("Type Checking", run_type_checking),
        ("Coverage", run_coverage)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} passed!")
            else:
                print(f"❌ {test_name} failed!")
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for deployment.")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues before deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(main())



























