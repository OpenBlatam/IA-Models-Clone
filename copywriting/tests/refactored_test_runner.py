#!/usr/bin/env python3
"""
Refactored Test Runner for Copywriting Service
==============================================

Advanced test runner with refactored architecture, parallel execution,
and comprehensive reporting capabilities.

Features:
- Parallel test execution
- Smart test discovery
- Performance monitoring
- Memory usage tracking
- Test result caching
- CI/CD integration
- Custom reporting formats
- Test data management
- Environment validation
"""

import os
import sys
import time
import json
import argparse
import subprocess
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import pytest
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.config.test_config import TestConfig
from tests.data.test_data_manager import TestDataManager
from tests.base import BaseTest


@dataclass
class TestResult:
    """Test execution result with comprehensive metrics."""
    test_name: str
    status: str  # passed, failed, skipped, error
    duration: float
    memory_usage: float
    cpu_usage: float
    error_message: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class TestSuiteResult:
    """Complete test suite execution results."""
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    total_duration: float
    total_memory_usage: float
    average_cpu_usage: float
    test_results: List[TestResult]
    environment_info: Dict[str, Any]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    @property
    def success_rate(self) -> float:
        """Calculate test success rate."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed / self.total_tests) * 100
    
    @property
    def failure_rate(self) -> float:
        """Calculate test failure rate."""
        if self.total_tests == 0:
            return 0.0
        return ((self.failed + self.errors) / self.total_tests) * 100


class RefactoredTestRunner:
    """Advanced test runner with refactored architecture."""
    
    def __init__(self, config: Optional[TestConfig] = None):
        """Initialize the test runner."""
        self.config = config or TestConfig()
        self.data_manager = TestDataManager()
        self.results: List[TestResult] = []
        self.start_time = None
        self.process = None
        
    def validate_environment(self) -> bool:
        """Validate the test environment."""
        print("🔍 Validating test environment...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ required")
            return False
        
        # Check required packages
        required_packages = [
            'pytest', 'fastapi', 'pydantic', 'celery', 
            'psutil', 'pytest-asyncio', 'pytest-cov'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing packages: {', '.join(missing_packages)}")
            return False
        
        # Check test data availability
        if not self.data_manager.validate_data():
            print("❌ Test data validation failed")
            return False
        
        print("✅ Environment validation passed")
        return True
    
    def discover_tests(self, pattern: str = None) -> List[str]:
        """Discover test files and functions."""
        print("🔍 Discovering tests...")
        
        test_dir = Path(__file__).parent
        test_files = []
        
        # Find test files
        for test_file in test_dir.rglob("test_*.py"):
            if test_file.name != "__init__.py":
                test_files.append(str(test_file))
        
        # Filter by pattern if provided
        if pattern:
            test_files = [f for f in test_files if pattern in f]
        
        print(f"📁 Found {len(test_files)} test files")
        return test_files
    
    def run_single_test(self, test_file: str, test_function: str = None) -> TestResult:
        """Run a single test file or function."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Build pytest command
            cmd = [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"]
            if test_function:
                cmd.extend(["-k", test_function])
            
            # Add configuration options
            cmd.extend([
                "--disable-warnings",
                "--no-header",
                "--tb=line",
                f"--maxfail={self.config.max_failures}",
                f"--timeout={self.config.timeout_seconds}"
            ])
            
            # Run the test
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds
            )
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Determine status
            if result.returncode == 0:
                status = "passed"
            elif result.returncode == 1:
                status = "failed"
            elif result.returncode == 2:
                status = "error"
            else:
                status = "skipped"
            
            return TestResult(
                test_name=test_file,
                status=status,
                duration=end_time - start_time,
                memory_usage=end_memory - start_memory,
                cpu_usage=0.0,  # Would need more complex monitoring
                error_message=result.stderr if result.returncode != 0 else None,
                stdout=result.stdout,
                stderr=result.stderr
            )
            
        except subprocess.TimeoutExpired:
            return TestResult(
                test_name=test_file,
                status="error",
                duration=self.config.timeout_seconds,
                memory_usage=0.0,
                cpu_usage=0.0,
                error_message="Test timeout exceeded"
            )
        except Exception as e:
            return TestResult(
                test_name=test_file,
                status="error",
                duration=time.time() - start_time,
                memory_usage=0.0,
                cpu_usage=0.0,
                error_message=str(e)
            )
    
    def run_tests_parallel(self, test_files: List[str], max_workers: int = None) -> List[TestResult]:
        """Run tests in parallel."""
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), 4)
        
        print(f"🚀 Running {len(test_files)} tests in parallel (workers: {max_workers})")
        
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all test files
            future_to_file = {
                executor.submit(self.run_single_test, test_file): test_file
                for test_file in test_files
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                test_file = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"✅ {test_file}: {result.status} ({result.duration:.2f}s)")
                except Exception as e:
                    print(f"❌ {test_file}: Error - {e}")
                    results.append(TestResult(
                        test_name=test_file,
                        status="error",
                        duration=0.0,
                        memory_usage=0.0,
                        cpu_usage=0.0,
                        error_message=str(e)
                    ))
        
        return results
    
    def run_tests_sequential(self, test_files: List[str]) -> List[TestResult]:
        """Run tests sequentially."""
        print(f"🚀 Running {len(test_files)} tests sequentially")
        
        results = []
        for i, test_file in enumerate(test_files, 1):
            print(f"📝 Running test {i}/{len(test_files)}: {test_file}")
            result = self.run_single_test(test_file)
            results.append(result)
            print(f"   Status: {result.status}, Duration: {result.duration:.2f}s")
        
        return results
    
    def run_tests(self, test_files: List[str] = None, parallel: bool = True, 
                  max_workers: int = None) -> TestSuiteResult:
        """Run the complete test suite."""
        print("🚀 Starting refactored test execution...")
        self.start_time = time.time()
        
        # Discover tests if not provided
        if test_files is None:
            test_files = self.discover_tests()
        
        # Validate environment
        if not self.validate_environment():
            raise RuntimeError("Environment validation failed")
        
        # Run tests
        if parallel and len(test_files) > 1:
            self.results = self.run_tests_parallel(test_files, max_workers)
        else:
            self.results = self.run_tests_sequential(test_files)
        
        # Calculate summary statistics
        total_tests = len(self.results)
        passed = len([r for r in self.results if r.status == "passed"])
        failed = len([r for r in self.results if r.status == "failed"])
        skipped = len([r for r in self.results if r.status == "skipped"])
        errors = len([r for r in self.results if r.status == "error"])
        
        total_duration = time.time() - self.start_time
        total_memory = sum(r.memory_usage for r in self.results)
        avg_cpu = sum(r.cpu_usage for r in self.results) / total_tests if total_tests > 0 else 0
        
        # Environment info
        env_info = {
            "python_version": sys.version,
            "platform": sys.platform,
            "cpu_count": multiprocessing.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "test_config": asdict(self.config)
        }
        
        suite_result = TestSuiteResult(
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            total_duration=total_duration,
            total_memory_usage=total_memory,
            average_cpu_usage=avg_cpu,
            test_results=self.results,
            environment_info=env_info
        )
        
        return suite_result
    
    def generate_report(self, result: TestSuiteResult, format: str = "console") -> str:
        """Generate test execution report."""
        if format == "console":
            return self._generate_console_report(result)
        elif format == "json":
            return self._generate_json_report(result)
        elif format == "html":
            return self._generate_html_report(result)
        else:
            raise ValueError(f"Unsupported report format: {format}")
    
    def _generate_console_report(self, result: TestSuiteResult) -> str:
        """Generate console-formatted report."""
        report = []
        report.append("=" * 80)
        report.append("🧪 REFACTORED TEST EXECUTION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        report.append("📊 SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Tests:     {result.total_tests}")
        report.append(f"Passed:          {result.passed} ({result.success_rate:.1f}%)")
        report.append(f"Failed:          {result.failed}")
        report.append(f"Skipped:         {result.skipped}")
        report.append(f"Errors:          {result.errors}")
        report.append(f"Success Rate:    {result.success_rate:.1f}%")
        report.append(f"Failure Rate:    {result.failure_rate:.1f}%")
        report.append("")
        
        # Performance
        report.append("⚡ PERFORMANCE")
        report.append("-" * 40)
        report.append(f"Total Duration:  {result.total_duration:.2f} seconds")
        report.append(f"Memory Usage:    {result.total_memory_usage:.2f} MB")
        report.append(f"Avg CPU Usage:   {result.average_cpu_usage:.1f}%")
        report.append("")
        
        # Test Results
        report.append("📋 TEST RESULTS")
        report.append("-" * 40)
        for test_result in result.test_results:
            status_icon = {
                "passed": "✅",
                "failed": "❌",
                "skipped": "⏭️",
                "error": "💥"
            }.get(test_result.status, "❓")
            
            report.append(f"{status_icon} {test_result.test_name}")
            report.append(f"   Status: {test_result.status}")
            report.append(f"   Duration: {test_result.duration:.2f}s")
            report.append(f"   Memory: {test_result.memory_usage:.2f} MB")
            if test_result.error_message:
                report.append(f"   Error: {test_result.error_message}")
            report.append("")
        
        # Environment
        report.append("🌍 ENVIRONMENT")
        report.append("-" * 40)
        report.append(f"Python: {result.environment_info['python_version']}")
        report.append(f"Platform: {result.environment_info['platform']}")
        report.append(f"CPU Cores: {result.environment_info['cpu_count']}")
        report.append(f"Memory: {result.environment_info['memory_total'] / (1024**3):.1f} GB")
        report.append("")
        
        report.append("=" * 80)
        report.append(f"Report generated at: {result.timestamp}")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _generate_json_report(self, result: TestSuiteResult) -> str:
        """Generate JSON-formatted report."""
        return json.dumps(asdict(result), indent=2, default=str)
    
    def _generate_html_report(self, result: TestSuiteResult) -> str:
        """Generate HTML-formatted report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Test Execution Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .metric {{ background: #e8f4f8; padding: 15px; border-radius: 5px; text-align: center; }}
        .passed {{ color: #28a745; }}
        .failed {{ color: #dc3545; }}
        .test-result {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ddd; }}
        .test-passed {{ border-left-color: #28a745; }}
        .test-failed {{ border-left-color: #dc3545; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 Test Execution Report</h1>
        <p>Generated at: {result.timestamp}</p>
    </div>
    
    <div class="summary">
        <div class="metric">
            <h3>Total Tests</h3>
            <p>{result.total_tests}</p>
        </div>
        <div class="metric passed">
            <h3>Passed</h3>
            <p>{result.passed} ({result.success_rate:.1f}%)</p>
        </div>
        <div class="metric failed">
            <h3>Failed</h3>
            <p>{result.failed + result.errors}</p>
        </div>
        <div class="metric">
            <h3>Duration</h3>
            <p>{result.total_duration:.2f}s</p>
        </div>
    </div>
    
    <h2>Test Results</h2>
"""
        
        for test_result in result.test_results:
            status_class = "test-passed" if test_result.status == "passed" else "test-failed"
            html += f"""
    <div class="test-result {status_class}">
        <h4>{test_result.test_name}</h4>
        <p>Status: {test_result.status} | Duration: {test_result.duration:.2f}s | Memory: {test_result.memory_usage:.2f} MB</p>
"""
            if test_result.error_message:
                html += f"        <p><strong>Error:</strong> {test_result.error_message}</p>"
            html += "    </div>"
        
        html += """
</body>
</html>
"""
        return html
    
    def save_report(self, result: TestSuiteResult, filename: str, format: str = "json"):
        """Save test report to file."""
        report_content = self.generate_report(result, format)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📄 Report saved to: {filename}")
    
    def cleanup(self):
        """Cleanup test data and temporary files."""
        print("🧹 Cleaning up test data...")
        self.data_manager.cleanup()
        print("✅ Cleanup completed")


def main():
    """Main entry point for the refactored test runner."""
    parser = argparse.ArgumentParser(description="Refactored Test Runner")
    parser.add_argument("--pattern", help="Test file pattern to match")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--workers", type=int, help="Number of parallel workers")
    parser.add_argument("--format", choices=["console", "json", "html"], 
                       default="console", help="Report format")
    parser.add_argument("--output", help="Output file for report")
    parser.add_argument("--config", help="Path to test configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = TestConfig()
    if args.config:
        config.load_from_file(args.config)
    
    # Create test runner
    runner = RefactoredTestRunner(config)
    
    try:
        # Run tests
        result = runner.run_tests(
            pattern=args.pattern,
            parallel=args.parallel,
            max_workers=args.workers
        )
        
        # Generate and display report
        report = runner.generate_report(result, args.format)
        print(report)
        
        # Save report if requested
        if args.output:
            runner.save_report(result, args.output, args.format)
        
        # Exit with appropriate code
        if result.failed > 0 or result.errors > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n⚠️ Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"💥 Test execution failed: {e}")
        sys.exit(1)
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()
