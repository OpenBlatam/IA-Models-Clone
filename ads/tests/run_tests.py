#!/usr/bin/env python3
"""
Comprehensive test runner for the ads feature.

This script provides a unified interface for running all tests with various configurations:
- Unit tests only
- Integration tests only
- All tests
- Performance tests
- Coverage reports
- Test discovery
- Custom test selection
"""

import os
import sys
import argparse
import subprocess
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import time
import platform


class TestRunner:
    """Comprehensive test runner for the ads feature."""
    
    def __init__(self, project_root: str = None):
        """Initialize the test runner."""
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.tests_dir = self.project_root / "tests"
        self.results_dir = self.project_root / "test_results"
        self.coverage_dir = self.project_root / "coverage"
        
        # Ensure directories exist
        self.results_dir.mkdir(exist_ok=True)
        self.coverage_dir.mkdir(exist_ok=True)
        
        # Test configuration
        self.test_configs = {
            "unit": {
                "path": "tests/unit",
                "markers": ["unit"],
                "description": "Unit tests for individual components"
            },
            "integration": {
                "path": "tests/integration",
                "markers": ["integration"],
                "description": "Integration tests for component interactions"
            },
            "all": {
                "path": "tests",
                "markers": [],
                "description": "All tests (unit + integration)"
            },
            "performance": {
                "path": "tests",
                "markers": ["slow", "performance"],
                "description": "Performance and slow-running tests"
            },
            "coverage": {
                "path": "tests",
                "markers": [],
                "description": "Tests with coverage reporting"
            }
        }
    
    def run_command(self, command: List[str], capture_output: bool = True) -> Dict[str, Any]:
        """Run a shell command and return results."""
        try:
            start_time = time.time()
            result = subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                cwd=self.project_root,
                timeout=300  # 5 minute timeout
            )
            end_time = time.time()
            
            return {
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "stdout": result.stdout if capture_output else "",
                "stderr": result.stderr if capture_output else "",
                "execution_time": end_time - start_time,
                "command": " ".join(command)
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "return_code": -1,
                "stdout": "",
                "stderr": "Command timed out after 5 minutes",
                "execution_time": 300,
                "command": " ".join(command)
            }
        except Exception as e:
            return {
                "success": False,
                "return_code": -1,
                "stdout": "",
                "stderr": str(e),
                "execution_time": 0,
                "command": " ".join(command)
            }
    
    def discover_tests(self) -> Dict[str, List[str]]:
        """Discover available tests in the project."""
        tests = {}
        
        for test_type, config in self.test_configs.items():
            if test_type == "all":
                continue
                
            test_path = self.project_root / config["path"]
            if test_path.exists():
                test_files = []
                for file_path in test_path.rglob("test_*.py"):
                    if file_path.is_file():
                        test_files.append(str(file_path.relative_to(self.project_root)))
                tests[test_type] = test_files
        
        return tests
    
    def run_unit_tests(self, verbose: bool = False, parallel: bool = False) -> Dict[str, Any]:
        """Run unit tests only."""
        print("🔍 Running Unit Tests...")
        
        command = [
            sys.executable, "-m", "pytest",
            "tests/unit/",
            "-v" if verbose else "-q",
            "--tb=short",
            "--strict-markers",
            "--disable-warnings"
        ]
        
        if parallel:
            command.extend(["-n", "auto"])
        
        result = self.run_command(command)
        
        if result["success"]:
            print("✅ Unit tests completed successfully")
        else:
            print("❌ Unit tests failed")
            if result["stderr"]:
                print(f"Error: {result['stderr']}")
        
        return result
    
    def run_integration_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run integration tests only."""
        print("🔍 Running Integration Tests...")
        
        command = [
            sys.executable, "-m", "pytest",
            "tests/integration/",
            "-v" if verbose else "-q",
            "--tb=short",
            "--strict-markers",
            "--disable-warnings",
            "--integration"
        ]
        
        result = self.run_command(command)
        
        if result["success"]:
            print("✅ Integration tests completed successfully")
        else:
            print("❌ Integration tests failed")
            if result["stderr"]:
                print(f"Error: {result['stderr']}")
        
        return result
    
    def run_performance_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run performance tests only."""
        print("🔍 Running Performance Tests...")
        
        command = [
            sys.executable, "-m", "pytest",
            "tests/",
            "-v" if verbose else "-q",
            "--tb=short",
            "--strict-markers",
            "--disable-warnings",
            "-m", "slow or performance"
        ]
        
        result = self.run_command(command)
        
        if result["success"]:
            print("✅ Performance tests completed successfully")
        else:
            print("❌ Performance tests failed")
            if result["stderr"]:
                print(f"Error: {result['stderr']}")
        
        return result
    
    def run_coverage_tests(self, verbose: bool = False, html: bool = True) -> Dict[str, Any]:
        """Run tests with coverage reporting."""
        print("🔍 Running Tests with Coverage...")
        
        command = [
            sys.executable, "-m", "pytest",
            "tests/",
            "-v" if verbose else "-q",
            "--tb=short",
            "--strict-markers",
            "--disable-warnings",
            "--cov=agents.backend.onyx.server.features.ads",
            "--cov-report=term-missing",
            "--cov-report=json:coverage/coverage.json"
        ]
        
        if html:
            command.append("--cov-report=html:coverage/html")
        
        result = self.run_command(command)
        
        if result["success"]:
            print("✅ Coverage tests completed successfully")
            print("📊 Coverage report generated")
        else:
            print("❌ Coverage tests failed")
            if result["stderr"]:
                print(f"Error: {result['stderr']}")
        
        return result
    
    def run_all_tests(self, verbose: bool = False, parallel: bool = False) -> Dict[str, Any]:
        """Run all tests."""
        print("🔍 Running All Tests...")
        
        command = [
            sys.executable, "-m", "pytest",
            "tests/",
            "-v" if verbose else "-q",
            "--tb=short",
            "--strict-markers",
            "--disable-warnings"
        ]
        
        if parallel:
            command.extend(["-n", "auto"])
        
        result = self.run_command(command)
        
        if result["success"]:
            print("✅ All tests completed successfully")
        else:
            print("❌ Some tests failed")
            if result["stderr"]:
                print(f"Error: {result['stderr']}")
        
        return result
    
    def run_specific_tests(self, test_paths: List[str], verbose: bool = False) -> Dict[str, Any]:
        """Run specific test files or directories."""
        print(f"🔍 Running Specific Tests: {', '.join(test_paths)}")
        
        command = [
            sys.executable, "-m", "pytest",
            "-v" if verbose else "-q",
            "--tb=short",
            "--strict-markers",
            "--disable-warnings"
        ] + test_paths
        
        result = self.run_command(command)
        
        if result["success"]:
            print("✅ Specific tests completed successfully")
        else:
            print("❌ Specific tests failed")
            if result["stderr"]:
                print(f"Error: {result['stderr']}")
        
        return result
    
    def generate_test_report(self, results: Dict[str, Any], test_type: str) -> str:
        """Generate a test report."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        platform_info = platform.platform()
        python_version = sys.version
        
        report = f"""
# Test Report - {test_type.upper()}
Generated: {timestamp}
Platform: {platform_info}
Python: {python_version}

## Summary
- Test Type: {test_type}
- Success: {'✅ PASSED' if results['success'] else '❌ FAILED'}
- Return Code: {results['return_code']}
- Execution Time: {results['execution_time']:.2f} seconds

## Command Executed
```bash
{results['command']}
```

## Output
{results['stdout']}

## Errors (if any)
{results['stderr'] if results['stderr'] else 'No errors'}
"""
        
        return report
    
    def save_test_results(self, results: Dict[str, Any], test_type: str):
        """Save test results to file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{test_type}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Add metadata
        results["metadata"] = {
            "test_type": test_type,
            "timestamp": timestamp,
            "platform": platform.platform(),
            "python_version": sys.version,
            "project_root": str(self.project_root)
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📁 Test results saved to: {filepath}")
        
        # Also save as text report
        report_filename = f"test_report_{test_type}_{timestamp}.md"
        report_filepath = self.results_dir / report_filename
        
        report_content = self.generate_test_report(results, test_type)
        with open(report_filepath, 'w') as f:
            f.write(report_content)
        
        print(f"📁 Test report saved to: {report_filepath}")
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed."""
        required_packages = [
            "pytest",
            "pytest-asyncio",
            "pytest-cov",
            "pytest-xdist"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing required packages: {', '.join(missing_packages)}")
            print("Install them with: pip install " + " ".join(missing_packages))
            return False
        
        print("✅ All required dependencies are installed")
        return True
    
    def run_test_suite(self, test_type: str, **kwargs) -> Dict[str, Any]:
        """Run a specific test suite."""
        if test_type == "unit":
            return self.run_unit_tests(**kwargs)
        elif test_type == "integration":
            return self.run_integration_tests(**kwargs)
        elif test_type == "performance":
            return self.run_performance_tests(**kwargs)
        elif test_type == "coverage":
            return self.run_coverage_tests(**kwargs)
        elif test_type == "all":
            return self.run_all_tests(**kwargs)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
    
    def interactive_mode(self):
        """Run the test runner in interactive mode."""
        print("🚀 Ads Feature Test Runner - Interactive Mode")
        print("=" * 50)
        
        while True:
            print("\nAvailable options:")
            print("1. Run Unit Tests")
            print("2. Run Integration Tests")
            print("3. Run Performance Tests")
            print("4. Run Coverage Tests")
            print("5. Run All Tests")
            print("6. Discover Available Tests")
            print("7. Check Dependencies")
            print("8. Exit")
            
            choice = input("\nEnter your choice (1-8): ").strip()
            
            if choice == "1":
                verbose = input("Verbose output? (y/n): ").lower() == 'y'
                parallel = input("Run in parallel? (y/n): ").lower() == 'y'
                results = self.run_unit_tests(verbose=verbose, parallel=parallel)
                self.save_test_results(results, "unit")
                
            elif choice == "2":
                verbose = input("Verbose output? (y/n): ").lower() == 'y'
                results = self.run_integration_tests(verbose=verbose)
                self.save_test_results(results, "integration")
                
            elif choice == "3":
                verbose = input("Verbose output? (y/n): ").lower() == 'y'
                results = self.run_performance_tests(verbose=verbose)
                self.save_test_results(results, "performance")
                
            elif choice == "4":
                verbose = input("Verbose output? (y/n): ").lower() == 'y'
                html = input("Generate HTML coverage report? (y/n): ").lower() == 'y'
                results = self.run_coverage_tests(verbose=verbose, html=html)
                self.save_test_results(results, "coverage")
                
            elif choice == "5":
                verbose = input("Verbose output? (y/n): ").lower() == 'y'
                parallel = input("Run in parallel? (y/n): ").lower() == 'y'
                results = self.run_all_tests(verbose=verbose, parallel=parallel)
                self.save_test_results(results, "all")
                
            elif choice == "6":
                tests = self.discover_tests()
                print("\nAvailable Tests:")
                for test_type, test_files in tests.items():
                    print(f"\n{test_type.upper()}:")
                    for test_file in test_files:
                        print(f"  - {test_file}")
                        
            elif choice == "7":
                self.check_dependencies()
                
            elif choice == "8":
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter a number between 1-8.")


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for the ads feature",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python run_tests.py --all
  
  # Run only unit tests
  python run_tests.py --unit
  
  # Run tests with coverage
  python run_tests.py --coverage
  
  # Run specific test file
  python run_tests.py --specific tests/unit/test_domain.py
  
  # Interactive mode
  python run_tests.py --interactive
        """
    )
    
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Run all tests"
    )
    parser.add_argument(
        "--unit", 
        action="store_true", 
        help="Run unit tests only"
    )
    parser.add_argument(
        "--integration", 
        action="store_true", 
        help="Run integration tests only"
    )
    parser.add_argument(
        "--performance", 
        action="store_true", 
        help="Run performance tests only"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true", 
        help="Run tests with coverage reporting"
    )
    parser.add_argument(
        "--specific", 
        nargs="+", 
        help="Run specific test files or directories"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true", 
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--parallel", "-p", 
        action="store_true", 
        help="Run tests in parallel (where supported)"
    )
    parser.add_argument(
        "--project-root", 
        help="Path to project root directory"
    )
    parser.add_argument(
        "--save-results", 
        action="store_true", 
        help="Save test results to files"
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = TestRunner(project_root=args.project_root)
    
    # Check dependencies first
    if not runner.check_dependencies():
        sys.exit(1)
    
    # Determine what to run
    if args.interactive:
        runner.interactive_mode()
        return
    
    if args.specific:
        results = runner.run_specific_tests(args.specific, verbose=args.verbose)
        if args.save_results:
            runner.save_test_results(results, "specific")
        sys.exit(0 if results["success"] else 1)
    
    if args.unit:
        results = runner.run_unit_tests(verbose=args.verbose, parallel=args.parallel)
        if args.save_results:
            runner.save_test_results(results, "unit")
        sys.exit(0 if results["success"] else 1)
    
    if args.integration:
        results = runner.run_integration_tests(verbose=args.verbose)
        if args.save_results:
            runner.save_test_results(results, "integration")
        sys.exit(0 if results["success"] else 1)
    
    if args.performance:
        results = runner.run_performance_tests(verbose=args.verbose)
        if args.save_results:
            runner.save_test_results(results, "performance")
        sys.exit(0 if results["success"] else 1)
    
    if args.coverage:
        results = runner.run_coverage_tests(verbose=args.verbose)
        if args.save_results:
            runner.save_test_results(results, "coverage")
        sys.exit(0 if results["success"] else 1)
    
    if args.all:
        results = runner.run_all_tests(verbose=args.verbose, parallel=args.parallel)
        if args.save_results:
            runner.save_test_results(results, "all")
        sys.exit(0 if results["success"] else 1)
    
    # Default: run all tests
    print("🔍 No specific test type specified, running all tests...")
    results = runner.run_all_tests(verbose=args.verbose, parallel=args.parallel)
    if args.save_results:
        runner.save_test_results(results, "all")
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
