#!/usr/bin/env python3
"""
CI/CD Test Runner for HeyGen AI
==============================

Automated test runner designed for continuous integration and deployment.
Provides comprehensive testing with detailed reporting and exit codes.
"""

import sys
import os
import subprocess
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

class CITestRunner:
    """CI/CD test runner for HeyGen AI"""
    
    def __init__(self, verbose: bool = False, coverage: bool = False):
        self.base_dir = Path(__file__).parent
        self.test_dir = self.base_dir / "tests"
        self.verbose = verbose
        self.coverage = coverage
        self.results = {
            "start_time": None,
            "end_time": None,
            "duration": 0,
            "test_results": {},
            "coverage_results": {},
            "exit_code": 0
        }
    
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if self.verbose or level in ["ERROR", "WARNING"]:
            print(f"[{timestamp}] {level}: {message}")
    
    def check_environment(self) -> bool:
        """Check if the testing environment is ready"""
        self.log("Checking testing environment...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            self.log(f"Python {python_version.major}.{python_version.minor} is not supported. Minimum required: 3.8", "ERROR")
            return False
        
        self.log(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check if we're in the right directory
        if not self.test_dir.exists():
            self.log(f"Test directory not found: {self.test_dir}", "ERROR")
            return False
        
        self.log("Environment check passed")
        return True
    
    def install_dependencies(self) -> bool:
        """Install test dependencies"""
        self.log("Installing test dependencies...")
        
        try:
            # Install requirements-test.txt if it exists
            req_file = self.base_dir / "requirements-test.txt"
            if req_file.exists():
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(req_file)
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    self.log(f"Failed to install test dependencies: {result.stderr}", "ERROR")
                    return False
                
                self.log("Test dependencies installed successfully")
            else:
                self.log("No requirements-test.txt found, skipping dependency installation")
            
            return True
        except subprocess.TimeoutExpired:
            self.log("Dependency installation timed out", "ERROR")
            return False
        except Exception as e:
            self.log(f"Error installing dependencies: {e}", "ERROR")
            return False
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests"""
        self.log("Running unit tests...")
        
        cmd = [sys.executable, "-m", "pytest"]
        cmd.extend([
            str(self.test_dir),
            "-m", "unit",
            "--tb=short",
            "--strict-markers",
            "--disable-warnings"
        ])
        
        if self.verbose:
            cmd.append("-v")
        
        if self.coverage:
            cmd.extend(["--cov=core", "--cov-report=term-missing"])
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            end_time = time.time()
            
            return {
                "success": result.returncode == 0,
                "duration": end_time - start_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            self.log("Unit tests timed out", "ERROR")
            return {
                "success": False,
                "duration": 600,
                "stdout": "",
                "stderr": "Tests timed out after 10 minutes",
                "return_code": 124
            }
        except Exception as e:
            self.log(f"Error running unit tests: {e}", "ERROR")
            return {
                "success": False,
                "duration": 0,
                "stdout": "",
                "stderr": str(e),
                "return_code": 1
            }
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests"""
        self.log("Running integration tests...")
        
        cmd = [sys.executable, "-m", "pytest"]
        cmd.extend([
            str(self.test_dir),
            "-m", "integration",
            "--tb=short",
            "--strict-markers",
            "--disable-warnings"
        ])
        
        if self.verbose:
            cmd.append("-v")
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            end_time = time.time()
            
            return {
                "success": result.returncode == 0,
                "duration": end_time - start_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            self.log("Integration tests timed out", "ERROR")
            return {
                "success": False,
                "duration": 900,
                "stdout": "",
                "stderr": "Tests timed out after 15 minutes",
                "return_code": 124
            }
        except Exception as e:
            self.log(f"Error running integration tests: {e}", "ERROR")
            return {
                "success": False,
                "duration": 0,
                "stdout": "",
                "stderr": str(e),
                "return_code": 1
            }
    
    def run_enterprise_tests(self) -> Dict[str, Any]:
        """Run enterprise feature tests"""
        self.log("Running enterprise feature tests...")
        
        cmd = [sys.executable, "-m", "pytest"]
        cmd.extend([
            str(self.test_dir / "test_enterprise_features.py"),
            "--tb=short",
            "--strict-markers",
            "--disable-warnings"
        ])
        
        if self.verbose:
            cmd.append("-v")
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            end_time = time.time()
            
            return {
                "success": result.returncode == 0,
                "duration": end_time - start_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            self.log("Enterprise tests timed out", "ERROR")
            return {
                "success": False,
                "duration": 300,
                "stdout": "",
                "stderr": "Tests timed out after 5 minutes",
                "return_code": 124
            }
        except Exception as e:
            self.log(f"Error running enterprise tests: {e}", "ERROR")
            return {
                "success": False,
                "duration": 0,
                "stdout": "",
                "stderr": str(e),
                "return_code": 1
            }
    
    def run_import_validation(self) -> Dict[str, Any]:
        """Run import validation"""
        self.log("Running import validation...")
        
        try:
            start_time = time.time()
            result = subprocess.run([
                sys.executable, "validate_tests.py"
            ], capture_output=True, text=True, timeout=60)
            end_time = time.time()
            
            return {
                "success": result.returncode == 0,
                "duration": end_time - start_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            self.log("Import validation timed out", "ERROR")
            return {
                "success": False,
                "duration": 60,
                "stdout": "",
                "stderr": "Validation timed out after 1 minute",
                "return_code": 124
            }
        except Exception as e:
            self.log(f"Error running import validation: {e}", "ERROR")
            return {
                "success": False,
                "duration": 0,
                "stdout": "",
                "stderr": str(e),
                "return_code": 1
            }
    
    def generate_coverage_report(self) -> Dict[str, Any]:
        """Generate coverage report if coverage is enabled"""
        if not self.coverage:
            return {"enabled": False}
        
        self.log("Generating coverage report...")
        
        try:
            # Run tests with coverage
            cmd = [sys.executable, "-m", "pytest"]
            cmd.extend([
                str(self.test_dir),
                "--cov=core",
                "--cov-report=html",
                "--cov-report=json",
                "--cov-report=term-missing",
                "--tb=short",
                "--disable-warnings"
            ])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            # Check if coverage files were generated
            html_report = self.base_dir / "htmlcov" / "index.html"
            json_report = self.base_dir / "coverage.json"
            
            return {
                "enabled": True,
                "success": result.returncode == 0,
                "html_report_exists": html_report.exists(),
                "json_report_exists": json_report.exists(),
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except Exception as e:
            self.log(f"Error generating coverage report: {e}", "ERROR")
            return {
                "enabled": True,
                "success": False,
                "error": str(e)
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites"""
        self.log("Starting comprehensive test run...")
        
        start_time = time.time()
        self.results["start_time"] = start_time
        
        # Run import validation first
        import_results = self.run_import_validation()
        self.results["test_results"]["import_validation"] = import_results
        
        if not import_results["success"]:
            self.log("Import validation failed, skipping other tests", "ERROR")
            self.results["exit_code"] = 1
            return self.results
        
        # Run unit tests
        unit_results = self.run_unit_tests()
        self.results["test_results"]["unit_tests"] = unit_results
        
        # Run integration tests
        integration_results = self.run_integration_tests()
        self.results["test_results"]["integration_tests"] = integration_results
        
        # Run enterprise tests
        enterprise_results = self.run_enterprise_tests()
        self.results["test_results"]["enterprise_tests"] = enterprise_results
        
        # Generate coverage report if requested
        coverage_results = self.generate_coverage_report()
        self.results["coverage_results"] = coverage_results
        
        end_time = time.time()
        self.results["end_time"] = end_time
        self.results["duration"] = end_time - start_time
        
        # Determine overall success
        all_tests = [unit_results, integration_results, enterprise_results]
        overall_success = all(test["success"] for test in all_tests)
        
        if not overall_success:
            self.results["exit_code"] = 1
        
        return self.results
    
    def generate_summary_report(self) -> str:
        """Generate summary report"""
        report = []
        report.append("🧪 HeyGen AI CI/CD Test Summary")
        report.append("=" * 50)
        report.append(f"Start Time: {datetime.fromtimestamp(self.results['start_time']).strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"End Time: {datetime.fromtimestamp(self.results['end_time']).strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Duration: {self.results['duration']:.2f} seconds")
        report.append("")
        
        # Test results summary
        report.append("📊 Test Results:")
        for test_name, results in self.results["test_results"].items():
            status = "✅ PASSED" if results["success"] else "❌ FAILED"
            duration = f"{results['duration']:.2f}s"
            report.append(f"  {test_name.replace('_', ' ').title()}: {status} ({duration})")
        
        # Coverage summary
        if self.results["coverage_results"]["enabled"]:
            report.append("\n📈 Coverage:")
            if self.results["coverage_results"]["success"]:
                report.append("  ✅ Coverage report generated successfully")
            else:
                report.append("  ❌ Coverage report generation failed")
        
        # Overall status
        report.append(f"\n🎯 Overall Status: {'✅ SUCCESS' if self.results['exit_code'] == 0 else '❌ FAILURE'}")
        
        return "\n".join(report)
    
    def save_results(self, filename: str = "ci_test_results.json"):
        """Save test results to JSON file"""
        results_file = self.base_dir / filename
        
        # Convert datetime objects to strings for JSON serialization
        json_results = self.results.copy()
        if json_results["start_time"]:
            json_results["start_time"] = datetime.fromtimestamp(json_results["start_time"]).isoformat()
        if json_results["end_time"]:
            json_results["end_time"] = datetime.fromtimestamp(json_results["end_time"]).isoformat()
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        self.log(f"Test results saved to: {results_file}")

def main():
    """Main CI test runner function"""
    parser = argparse.ArgumentParser(description="CI/CD Test Runner for HeyGen AI")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", "-c", action="store_true", help="Generate coverage report")
    parser.add_argument("--install-deps", action="store_true", help="Install test dependencies")
    parser.add_argument("--test-type", choices=["unit", "integration", "enterprise", "all"], 
                       default="all", help="Type of tests to run")
    
    args = parser.parse_args()
    
    runner = CITestRunner(verbose=args.verbose, coverage=args.coverage)
    
    # Check environment
    if not runner.check_environment():
        return 1
    
    # Install dependencies if requested
    if args.install_deps:
        if not runner.install_dependencies():
            return 1
    
    # Run tests based on type
    if args.test_type == "all":
        results = runner.run_all_tests()
    elif args.test_type == "unit":
        results = {"test_results": {"unit_tests": runner.run_unit_tests()}}
    elif args.test_type == "integration":
        results = {"test_results": {"integration_tests": runner.run_integration_tests()}}
    elif args.test_type == "enterprise":
        results = {"test_results": {"enterprise_tests": runner.run_enterprise_tests()}}
    
    # Generate and display summary
    summary = runner.generate_summary_report()
    print(f"\n{summary}")
    
    # Save results
    runner.save_results()
    
    return runner.results.get("exit_code", 0)

if __name__ == "__main__":
    sys.exit(main())





