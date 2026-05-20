#!/usr/bin/env python3
"""
Advanced Test Runner Script
============================
Advanced script for running and managing Playwright tests.
"""

import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional
from playwright_test_runner import PlaywrightTestExecutor, PlaywrightTestRunner
from playwright_analytics import PlaywrightAnalytics, create_analytics


def run_tests(
    markers: Optional[List[str]] = None,
    test_files: Optional[List[str]] = None,
    parallel: bool = False,
    workers: int = 4,
    verbose: bool = True,
    coverage: bool = False,
    html_report: bool = True,
    analytics: bool = True
):
    """Run tests with advanced options."""
    executor = PlaywrightTestExecutor()
    analytics_instance = create_analytics() if analytics else None
    
    # Build pytest command
    cmd = ["pytest"]
    
    if test_files:
        cmd.extend(test_files)
    else:
        cmd.append("tests/test_playwright*.py")
    
    if markers:
        marker_expr = " and ".join([f"mark.{m}" for m in markers])
        cmd.extend(["-m", marker_expr])
    
    if parallel:
        cmd.extend(["-n", str(workers)])
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov", "--cov-report=html", "--cov-report=term"])
    
    # Run tests
    print(f"Running tests with command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Generate reports
    if html_report:
        print("Generating HTML report...")
        # Would integrate with actual test results here
    
    if analytics and analytics_instance:
        print("Generating analytics...")
        # Would integrate with actual test results here
        report_path = analytics_instance.generate_report(format="html")
        print(f"Analytics report saved to: {report_path}")
    
    return result.returncode


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Advanced Playwright test runner")
    
    parser.add_argument(
        "--markers",
        nargs="+",
        help="Test markers to filter (e.g., smoke api)"
    )
    
    parser.add_argument(
        "--files",
        nargs="+",
        help="Specific test files to run"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode (less verbose)"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "--no-html-report",
        action="store_true",
        help="Skip HTML report generation"
    )
    
    parser.add_argument(
        "--no-analytics",
        action="store_true",
        help="Skip analytics generation"
    )
    
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke tests only"
    )
    
    parser.add_argument(
        "--critical",
        action="store_true",
        help="Run critical tests only"
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run fast tests only"
    )
    
    args = parser.parse_args()
    
    # Determine markers
    markers = args.markers or []
    if args.smoke:
        markers.append("smoke")
    if args.critical:
        markers.append("critical")
    if args.fast:
        markers.append("fast")
    
    # Run tests
    exit_code = run_tests(
        markers=markers if markers else None,
        test_files=args.files,
        parallel=args.parallel,
        workers=args.workers,
        verbose=not args.quiet,
        coverage=args.coverage,
        html_report=not args.no_html_report,
        analytics=not args.no_analytics
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()



