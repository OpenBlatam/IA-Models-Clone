#!/usr/bin/env python3
"""
Refactored Test Runner
======================

Comprehensive test runner that integrates all refactored components:
- BaseTest architecture
- TestConfig management
- TestDataManager integration
- Performance monitoring
- Parallel execution
- Advanced reporting

Usage:
    python run_refactored_tests.py [options]

Examples:
    # Run all tests
    python run_refactored_tests.py

    # Run specific test categories
    python run_refactored_tests.py --category unit,integration

    # Run with parallel execution
    python run_refactored_tests.py --parallel --workers 4

    # Generate HTML report
    python run_refactored_tests.py --format html --output report.html

    # Run performance tests only
    python run_refactored_tests.py --category performance --benchmark
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
class TestExecutionResult:
    """Result of test execution with comprehensive metrics."""
    test_name: str
    category: str
    status: str  # passed, failed, skipped, error
    duration: float
    memory_usage: float
    cpu_usage: float
    error_message: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    timestamp: str = None
    performance_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.performance_metrics is None:
            self.performance_metrics = {}


@dataclass
class TestSuiteExecutionResult:
    """Complete test suite execution results."""
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    total_duration: float
    total_memory_usage: float
    average_cpu_usage: float
    test_results: List[TestExecutionResult]
    environment_info: Dict[str, Any]
    configuration: Dict[str, Any]
    timestamp: str = None
    success_rate: float = 0.0
    failure_rate: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        
        # Calculate rates
        if self.total_tests > 0:
            self.success_rate = (self.passed / self.total_tests) * 100
            self.failure_rate = ((self.failed + self.errors) / self.total_tests) * 100


class RefactoredTestRunner:
    """Advanced test runner with refactored architecture."""
    
    def __init__(self, config: Optional[TestConfig] = None):
        """Initialize the test runner."""
        self.config = config or TestConfig()
        self.data_manager = TestDataManager()
        self.results: List[TestExecutionResult] = []
        self.start_time = None
        self.process = None
        
        # Test categories
        self.categories = {
            "unit": "tests/unit/",
            "integration": "tests/integration/",
            "performance": "tests/performance/",
            "security": "tests/security/",
            "monitoring": "tests/monitoring/",
            "benchmarks": "tests/benchmarks/",
            "examples": "tests/examples/"
        }
    
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
        
        # Check refactored components
        try:
            from tests.base import BaseTest
            from tests.config.test_config import TestConfig
            from tests.data.test_data_manager import TestDataManager
        except ImportError as e:
            print(f"❌ Refactored components not available: {e}")
            return False
        
        print("✅ Environment validation passed")
        return True
    
    def discover_tests(self, categories: List[str] = None, pattern: str = None) -> List[Tuple[str, str]]:
        """Discover test files and categorize them."""
        print("🔍 Discovering tests...")
        
        test_dir = Path(__file__).parent
        test_files = []
        
        # Determine categories to search
        if categories is None:
            categories = list(self.categories.keys())
        
        for category in categories:
            if category in self.categories:
                category_dir = test_dir / self.categories[category]
                if category_dir.exists():
                    for test_file in category_dir.rglob("test_*.py"):
                        if test_file.name != "__init__.py":
                            test_files.append((str(test_file), category))
        
        # Also check root test directory
        for test_file in test_dir.glob("test_*.py"):
            if test_file.name != "__init__.py":
                test_files.append((str(test_file), "root"))
        
        # Filter by pattern if provided
        if pattern:
            test_files = [(f, c) for f, c in test_files if pattern in f]
        
        print(f"📁 Found {len(test_files)} test files")
        return test_files
    
    def run_single_test(self, test_file: str, category: str, test_function: str = None) -> TestExecutionResult:
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
                f"--timeout={self.config.timeout_seconds}",
                "--strict-markers",
                "--strict-config"
            ])
            
            # Add coverage if enabled
            if self.config.enable_coverage:
                cmd.extend(["--cov=copywriting", "--cov-report=term-missing"])
            
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
            
            # Extract performance metrics from output
            performance_metrics = self._extract_performance_metrics(result.stdout)
            
            return TestExecutionResult(
                test_name=test_file,
                category=category,
                status=status,
                duration=end_time - start_time,
                memory_usage=end_memory - start_memory,
                cpu_usage=0.0,  # Would need more complex monitoring
                error_message=result.stderr if result.returncode != 0 else None,
                stdout=result.stdout,
                stderr=result.stderr,
                performance_metrics=performance_metrics
            )
            
        except subprocess.TimeoutExpired:
            return TestExecutionResult(
                test_name=test_file,
                category=category,
                status="error",
                duration=self.config.timeout_seconds,
                memory_usage=0.0,
                cpu_usage=0.0,
                error_message="Test timeout exceeded"
            )
        except Exception as e:
            return TestExecutionResult(
                test_name=test_file,
                category=category,
                status="error",
                duration=time.time() - start_time,
                memory_usage=0.0,
                cpu_usage=0.0,
                error_message=str(e)
            )
    
    def _extract_performance_metrics(self, stdout: str) -> Dict[str, Any]:
        """Extract performance metrics from test output."""
        metrics = {}
        
        # Look for performance markers in output
        lines = stdout.split('\n')
        for line in lines:
            if "Performance:" in line:
                # Extract performance data
                try:
                    parts = line.split("Performance:")[1].strip()
                    # Parse performance metrics (customize based on your output format)
                    metrics["performance_note"] = parts
                except:
                    pass
        
        return metrics
    
    def run_tests_parallel(self, test_files: List[Tuple[str, str]], max_workers: int = None) -> List[TestExecutionResult]:
        """Run tests in parallel."""
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), 4)
        
        print(f"🚀 Running {len(test_files)} tests in parallel (workers: {max_workers})")
        
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all test files
            future_to_file = {
                executor.submit(self.run_single_test, test_file, category): (test_file, category)
                for test_file, category in test_files
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                test_file, category = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    status_icon = "✅" if result.status == "passed" else "❌"
                    print(f"{status_icon} {test_file}: {result.status} ({result.duration:.2f}s)")
                except Exception as e:
                    print(f"❌ {test_file}: Error - {e}")
                    results.append(TestExecutionResult(
                        test_name=test_file,
                        category=category,
                        status="error",
                        duration=0.0,
                        memory_usage=0.0,
                        cpu_usage=0.0,
                        error_message=str(e)
                    ))
        
        return results
    
    def run_tests_sequential(self, test_files: List[Tuple[str, str]]) -> List[TestExecutionResult]:
        """Run tests sequentially."""
        print(f"🚀 Running {len(test_files)} tests sequentially")
        
        results = []
        for i, (test_file, category) in enumerate(test_files, 1):
            print(f"📝 Running test {i}/{len(test_files)}: {test_file}")
            result = self.run_single_test(test_file, category)
            results.append(result)
            status_icon = "✅" if result.status == "passed" else "❌"
            print(f"   {status_icon} Status: {result.status}, Duration: {result.duration:.2f}s")
        
        return results
    
    def run_tests(self, categories: List[str] = None, pattern: str = None, 
                  parallel: bool = True, max_workers: int = None, 
                  benchmark: bool = False) -> TestSuiteExecutionResult:
        """Run the complete test suite."""
        print("🚀 Starting refactored test execution...")
        self.start_time = time.time()
        
        # Discover tests
        test_files = self.discover_tests(categories, pattern)
        
        # Filter for benchmark tests if requested
        if benchmark:
            test_files = [(f, c) for f, c in test_files if "benchmark" in f or "performance" in c]
        
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
        
        suite_result = TestSuiteExecutionResult(
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            total_duration=total_duration,
            total_memory_usage=total_memory,
            average_cpu_usage=avg_cpu,
            test_results=self.results,
            environment_info=env_info,
            configuration=asdict(self.config)
        )
        
        return suite_result
    
    def generate_report(self, result: TestSuiteExecutionResult, format: str = "console") -> str:
        """Generate test execution report."""
        if format == "console":
            return self._generate_console_report(result)
        elif format == "json":
            return self._generate_json_report(result)
        elif format == "html":
            return self._generate_html_report(result)
        else:
            raise ValueError(f"Unsupported report format: {format}")
    
    def _generate_console_report(self, result: TestSuiteExecutionResult) -> str:
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
        
        # Test Results by Category
        report.append("📋 TEST RESULTS BY CATEGORY")
        report.append("-" * 40)
        categories = {}
        for test_result in result.test_results:
            if test_result.category not in categories:
                categories[test_result.category] = {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0}
            categories[test_result.category]["total"] += 1
            categories[test_result.category][test_result.status] += 1
        
        for category, stats in categories.items():
            success_rate = (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            report.append(f"{category.upper()}:")
            report.append(f"  Total: {stats['total']}, Passed: {stats['passed']}, Failed: {stats['failed']}, Skipped: {stats['skipped']}, Errors: {stats['errors']}")
            report.append(f"  Success Rate: {success_rate:.1f}%")
            report.append("")
        
        # Individual Test Results
        report.append("📝 INDIVIDUAL TEST RESULTS")
        report.append("-" * 40)
        for test_result in result.test_results:
            status_icon = {
                "passed": "✅",
                "failed": "❌",
                "skipped": "⏭️",
                "error": "💥"
            }.get(test_result.status, "❓")
            
            report.append(f"{status_icon} {test_result.test_name} ({test_result.category})")
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
    
    def _generate_json_report(self, result: TestSuiteExecutionResult) -> str:
        """Generate JSON-formatted report."""
        return json.dumps(asdict(result), indent=2, default=str)
    
    def _generate_html_report(self, result: TestSuiteExecutionResult) -> str:
        """Generate HTML-formatted report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Refactored Test Execution Report</title>
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
        .category {{ margin: 20px 0; }}
        .category h3 {{ background: #f8f9fa; padding: 10px; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 Refactored Test Execution Report</h1>
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
    
    <h2>Test Results by Category</h2>
"""
        
        # Group by category
        categories = {}
        for test_result in result.test_results:
            if test_result.category not in categories:
                categories[test_result.category] = []
            categories[test_result.category].append(test_result)
        
        for category, tests in categories.items():
            passed = len([t for t in tests if t.status == "passed"])
            total = len(tests)
            success_rate = (passed / total) * 100 if total > 0 else 0
            
            html += f"""
    <div class="category">
        <h3>{category.upper()} ({passed}/{total} - {success_rate:.1f}%)</h3>
"""
            
            for test_result in tests:
                status_class = "test-passed" if test_result.status == "passed" else "test-failed"
                html += f"""
        <div class="test-result {status_class}">
            <h4>{test_result.test_name}</h4>
            <p>Status: {test_result.status} | Duration: {test_result.duration:.2f}s | Memory: {test_result.memory_usage:.2f} MB</p>
"""
                if test_result.error_message:
                    html += f"            <p><strong>Error:</strong> {test_result.error_message}</p>"
                html += "        </div>"
            
            html += "    </div>"
        
        html += """
</body>
</html>
"""
        return html
    
    def save_report(self, result: TestSuiteExecutionResult, filename: str, format: str = "json"):
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
    parser.add_argument("--categories", help="Comma-separated list of test categories")
    parser.add_argument("--pattern", help="Test file pattern to match")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--workers", type=int, help="Number of parallel workers")
    parser.add_argument("--format", choices=["console", "json", "html"], 
                       default="console", help="Report format")
    parser.add_argument("--output", help="Output file for report")
    parser.add_argument("--config", help="Path to test configuration file")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark tests only")
    
    args = parser.parse_args()
    
    # Load configuration
    config = TestConfig()
    if args.config:
        config.load_from_file(args.config)
    
    # Create test runner
    runner = RefactoredTestRunner(config)
    
    try:
        # Parse categories
        categories = None
        if args.categories:
            categories = [cat.strip() for cat in args.categories.split(",")]
        
        # Run tests
        result = runner.run_tests(
            categories=categories,
            pattern=args.pattern,
            parallel=args.parallel,
            max_workers=args.workers,
            benchmark=args.benchmark
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
