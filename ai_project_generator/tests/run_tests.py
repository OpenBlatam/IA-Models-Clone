#!/usr/bin/env python3
"""
Enhanced test runner script with multiple options
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(
    pattern: str = None,
    marker: str = None,
    verbose: bool = False,
    coverage: bool = False,
    parallel: bool = False,
    slow: bool = True,
    output: str = None
):
    """Run tests with various options"""
    
    cmd = ["python", "-m", "pytest"]
    
    # Add pattern if specified
    if pattern:
        cmd.append(pattern)
    else:
        cmd.append("tests/")
    
    # Add marker filter
    if marker:
        cmd.extend(["-m", marker])
    elif not slow:
        cmd.extend(["-m", "not slow"])
    
    # Add verbose flag
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Add coverage
    if coverage:
        cmd.extend([
            "--cov=.",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Add parallel execution
    if parallel:
        try:
            import pytest_xdist
            cmd.extend(["-n", "auto"])
        except ImportError:
            print("Warning: pytest-xdist not installed, running sequentially")
    
    # Add output file
    if output:
        cmd.extend(["--junitxml", output])
    
    # Add additional options
    cmd.extend([
        "--tb=short",
        "--strict-markers",
        "--disable-warnings"
    ])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run AI Project Generator tests")
    
    parser.add_argument(
        "pattern",
        nargs="?",
        help="Test pattern to run (e.g., 'test_api.py' or 'tests/test_utils_*.py')"
    )
    
    parser.add_argument(
        "-m", "--marker",
        help="Run tests with specific marker (e.g., 'unit', 'integration', 'slow')"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Run with coverage report"
    )
    
    parser.add_argument(
        "-p", "--parallel",
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)"
    )
    
    parser.add_argument(
        "--no-slow",
        action="store_true",
        help="Exclude slow tests"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output file for JUnit XML report"
    )
    
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run only unit tests"
    )
    
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run only integration tests"
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run only fast tests (exclude slow and integration)"
    )
    
    args = parser.parse_args()
    
    # Determine marker based on flags
    marker = args.marker
    if args.unit:
        marker = "unit"
    elif args.integration:
        marker = "integration"
    elif args.fast:
        marker = "not slow and not integration"
    
    # Run tests
    return run_tests(
        pattern=args.pattern,
        marker=marker,
        verbose=args.verbose,
        coverage=args.coverage,
        parallel=args.parallel,
        slow=not args.no_slow,
        output=args.output
    )


if __name__ == "__main__":
    sys.exit(main())

