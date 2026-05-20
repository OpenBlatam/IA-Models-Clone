#!/usr/bin/env python3
"""
Test runner for Blaze AI system.

This script provides an easy way to run all tests or specific test categories.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        if e.stdout:
            print("Stdout:", e.stdout)
        if e.stderr:
            print("Stderr:", e.stderr)
        return False

def check_dependencies():
    """Check if required testing dependencies are installed."""
    print("🔍 Checking testing dependencies...")
    
    try:
        import pytest
        print(f"✅ pytest {pytest.__version__} is available")
    except ImportError:
        print("❌ pytest is not installed")
        print("Please install it with: pip install -r tests/requirements-test.txt")
        return False
    
    try:
        import pytest_asyncio
        print(f"✅ pytest-asyncio is available")
    except ImportError:
        print("❌ pytest-asyncio is not installed")
        print("Please install it with: pip install -r tests/requirements-test.txt")
        return False
    
    return True

def run_unit_tests():
    """Run unit tests."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "-m", "unit",
        "--tb=short"
    ]
    return run_command(cmd, "Unit tests")

def run_integration_tests():
    """Run integration tests."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "-m", "integration",
        "--tb=short"
    ]
    return run_command(cmd, "Integration tests")

def run_all_tests():
    """Run all tests."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short"
    ]
    return run_command(cmd, "All tests")

def run_specific_test(test_file):
    """Run a specific test file."""
    cmd = [
        sys.executable, "-m", "pytest",
        str(test_file),
        "-v",
        "--tb=short"
    ]
    return run_command(cmd, f"Test file: {test_file}")

def run_tests_with_coverage():
    """Run tests with coverage reporting."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "--cov=engines",
        "--cov-report=html",
        "--cov-report=term-missing",
        "-v"
    ]
    return run_command(cmd, "Tests with coverage")

def run_tests_parallel():
    """Run tests in parallel for faster execution."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-n", "auto",
        "-v",
        "--tb=short"
    ]
    return run_command(cmd, "Parallel tests")

def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Run Blaze AI tests")
    parser.add_argument(
        "--unit", action="store_true",
        help="Run only unit tests"
    )
    parser.add_argument(
        "--integration", action="store_true",
        help="Run only integration tests"
    )
    parser.add_argument(
        "--coverage", action="store_true",
        help="Run tests with coverage reporting"
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--file", type=str,
        help="Run a specific test file"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all tests (default)"
    )
    
    args = parser.parse_args()
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("🚀 Blaze AI Test Runner")
    print("=" * 50)
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    success = True
    
    # Run tests based on arguments
    if args.file:
        success = run_specific_test(args.file)
    elif args.unit:
        success = run_unit_tests()
    elif args.integration:
        success = run_integration_tests()
    elif args.coverage:
        success = run_tests_with_coverage()
    elif args.parallel:
        success = run_tests_parallel()
    else:
        # Default: run all tests
        success = run_all_tests()
    
    # Print summary
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
