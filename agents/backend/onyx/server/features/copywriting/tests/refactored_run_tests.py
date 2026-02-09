"""
Refactored comprehensive test runner with advanced features.
"""
import os
import sys
import subprocess
import argparse
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))

from tests.base import BaseTestClass
from tests.config.test_config import test_config_manager, TestEnvironment, TestCategory
from tests.data.test_data_manager import test_data_manager, test_data_factory, test_data_cleanup


class RefactoredTestRunner:
    """Advanced test runner with refactored architecture."""
    
    def __init__(self):
        self.config = test_config_manager.get_config()
        self.start_time = None
        self.results = {}
        self.test_dir = Path(__file__).parent
        self.project_root = self.test_dir.parent.parent.parent.parent.parent
        
    def run_tests(self, 
                  test_types: List[str] = None,
                  environment: TestEnvironment = None,
                  category: TestCategory = None,
                  parallel: bool = True,
                  coverage: bool = True,
                  performance: bool = False,
                  security: bool = False,
                  monitoring: bool = False,
                  load: bool = False,
                  verbose: bool = False,
                  markers: List[str] = None) -> Dict[str, Any]:
        """Run tests with specified configuration."""
        
        self.start_time = time.time()
        
        # Set environment and category
        if environment:
            self.config.environment = environment
        if category:
            self.config.category = category
        
        # Build pytest command
        cmd = self._build_pytest_command(
            test_types=test_types,
            parallel=parallel,
            coverage=coverage,
            performance=performance,
            security=security,
            monitoring=monitoring,
            load=load,
            verbose=verbose,
            markers=markers
        )
        
        print(f"🚀 Running tests with command: {' '.join(cmd)}")
        print(f"📊 Environment: {self.config.environment.value}")
        print(f"📈 Category: {self.config.category.value}")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 80)
        
        # Run tests
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            # Parse results
            self.results = self._parse_results(result)
            
            # Generate reports
            self._generate_reports()
            
            # Cleanup test data
            self._cleanup_test_data()
            
            return self.results
            
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return {"error": str(e), "success": False}
    
    def _build_pytest_command(self, 
                             test_types: List[str] = None,
                             parallel: bool = True,
                             coverage: bool = True,
                             performance: bool = False,
                             security: bool = False,
                             monitoring: bool = False,
                             load: bool = False,
                             verbose: bool = False,
                             markers: List[str] = None) -> List[str]:
        """Build pytest command with specified options."""
        
        cmd = ["python", "-m", "pytest"]
        
        # Test directory
        test_path = str(self.test_dir)
        cmd.append(test_path)
        
        # Use refactored conftest
        cmd.extend(["-p", "no:warnings", "--confcutdir", str(self.test_dir)])
        
        # Markers
        if markers:
            cmd.extend(["-m", " or ".join(markers)])
        else:
            # Default markers based on environment
            default_markers = self._get_default_markers(
                test_types, performance, security, monitoring, load
            )
            if default_markers:
                cmd.extend(["-m", " or ".join(default_markers)])
        
        # Parallel execution
        if parallel and self.config.max_parallel_tests > 1:
            cmd.extend(["-n", str(self.config.max_parallel_tests)])
        
        # Coverage
        if coverage:
            cmd.extend([
                "--cov=agents.backend.onyx.server.features.copywriting",
                "--cov-report=html:reports/coverage_html",
                "--cov-report=xml:reports/coverage.xml",
                "--cov-report=json:reports/coverage.json",
                "--cov-report=term-missing"
            ])
        
        # Verbose output
        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")
        
        # Test discovery
        cmd.extend([
            "--tb=short",
            "--strict-markers",
            "--disable-warnings"
        ])
        
        # Performance testing
        if performance:
            cmd.extend(["--benchmark-only", "--benchmark-sort=mean"])
        
        # Load testing
        if load:
            cmd.extend(["--load-test", "--load-test-duration=60s"])
        
        # Security testing
        if security:
            cmd.extend(["--security-test", "--security-report"])
        
        # Monitoring testing
        if monitoring:
            cmd.extend(["--monitoring-test", "--monitoring-report"])
        
        # Output files
        cmd.extend([
            "--junitxml=reports/junit.xml",
            "--html=reports/test_report.html",
            "--self-contained-html"
        ])
        
        return cmd
    
    def _get_default_markers(self, 
                           test_types: List[str] = None,
                           performance: bool = False,
                           security: bool = False,
                           monitoring: bool = False,
                           load: bool = False) -> List[str]:
        """Get default markers based on configuration."""
        
        markers = []
        
        # Test types
        if test_types:
            markers.extend(test_types)
        else:
            # Default based on environment
            if self.config.environment == TestEnvironment.UNIT:
                markers.append("unit")
            elif self.config.environment == TestEnvironment.INTEGRATION:
                markers.append("integration")
            elif self.config.environment == TestEnvironment.PERFORMANCE:
                markers.extend(["performance", "benchmark"])
            elif self.config.environment == TestEnvironment.SECURITY:
                markers.append("security")
            elif self.config.environment == TestEnvironment.MONITORING:
                markers.append("monitoring")
            elif self.config.environment == TestEnvironment.LOAD:
                markers.append("load")
        
        # Additional markers
        if performance:
            markers.append("performance")
        if security:
            markers.append("security")
        if monitoring:
            markers.append("monitoring")
        if load:
            markers.append("load")
        
        # Category markers
        if self.config.category == TestCategory.FAST:
            markers.append("not slow")
        elif self.config.category == TestCategory.CRITICAL:
            markers.append("critical")
        elif self.config.category == TestCategory.OPTIONAL:
            markers.append("optional")
        
        return markers
    
    def _parse_results(self, result: subprocess.CompletedProcess) -> Dict[str, Any]:
        """Parse test results from subprocess output."""
        
        end_time = time.time()
        duration = end_time - (self.start_time or end_time)
        
        return {
            "success": result.returncode == 0,
            "return_code": result.returncode,
            "duration": duration,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timestamp": datetime.now().isoformat(),
            "environment": self.config.environment.value,
            "category": self.config.category.value
        }
    
    def _generate_reports(self):
        """Generate test reports."""
        
        # Create reports directory
        reports_dir = self.project_root / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        # Generate summary report
        self._generate_summary_report()
        
        # Generate performance report
        if self.config.environment == TestEnvironment.PERFORMANCE:
            self._generate_performance_report()
        
        # Generate security report
        if self.config.environment == TestEnvironment.SECURITY:
            self._generate_security_report()
        
        # Generate monitoring report
        if self.config.environment == TestEnvironment.MONITORING:
            self._generate_monitoring_report()
    
    def _generate_summary_report(self):
        """Generate summary report."""
        
        report_path = self.project_root / "reports" / "test_summary.json"
        
        summary = {
            "test_run": {
                "timestamp": self.results.get("timestamp"),
                "duration": self.results.get("duration"),
                "success": self.results.get("success"),
                "return_code": self.results.get("return_code"),
                "environment": self.results.get("environment"),
                "category": self.results.get("category")
            },
            "configuration": {
                "environment": self.config.environment.value,
                "category": self.config.category.value,
                "max_parallel_tests": self.config.max_parallel_tests,
                "performance_thresholds": {
                    "single_request_max_time": self.config.performance.single_request_max_time,
                    "batch_request_max_time": self.config.performance.batch_request_max_time,
                    "concurrent_request_max_time": self.config.performance.concurrent_request_max_time
                },
                "coverage_thresholds": {
                    "min_line_coverage": self.config.coverage.min_line_coverage,
                    "min_branch_coverage": self.config.coverage.min_branch_coverage,
                    "min_function_coverage": self.config.coverage.min_function_coverage
                }
            },
            "test_data_stats": test_data_manager.get_stats()
        }
        
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📊 Summary report generated: {report_path}")
    
    def _generate_performance_report(self):
        """Generate performance report."""
        
        report_path = self.project_root / "reports" / "performance_report.json"
        
        performance_data = {
            "timestamp": datetime.now().isoformat(),
            "environment": self.config.environment.value,
            "thresholds": {
                "single_request_max_time": self.config.performance.single_request_max_time,
                "batch_request_max_time": self.config.performance.batch_request_max_time,
                "concurrent_request_max_time": self.config.performance.concurrent_request_max_time,
                "load_test_max_time": self.config.performance.load_test_max_time,
                "memory_max_increase_mb": self.config.performance.memory_max_increase_mb
            },
            "test_data": test_data_manager.get_by_category("performance")
        }
        
        with open(report_path, 'w') as f:
            json.dump(performance_data, f, indent=2, default=str)
        
        print(f"📈 Performance report generated: {report_path}")
    
    def _generate_security_report(self):
        """Generate security report."""
        
        report_path = self.project_root / "reports" / "security_report.json"
        
        security_data = {
            "timestamp": datetime.now().isoformat(),
            "environment": self.config.environment.value,
            "thresholds": {
                "max_input_length": self.config.security.max_input_length,
                "max_malicious_inputs": self.config.security.max_malicious_inputs,
                "min_security_test_coverage": self.config.security.min_security_test_coverage
            },
            "test_data": test_data_manager.get_by_category("security")
        }
        
        with open(report_path, 'w') as f:
            json.dump(security_data, f, indent=2, default=str)
        
        print(f"🔒 Security report generated: {report_path}")
    
    def _generate_monitoring_report(self):
        """Generate monitoring report."""
        
        report_path = self.project_root / "reports" / "monitoring_report.json"
        
        monitoring_data = {
            "timestamp": datetime.now().isoformat(),
            "environment": self.config.environment.value,
            "test_data": test_data_manager.get_by_category("monitoring")
        }
        
        with open(report_path, 'w') as f:
            json.dump(monitoring_data, f, indent=2, default=str)
        
        print(f"📊 Monitoring report generated: {report_path}")
    
    def _cleanup_test_data(self):
        """Cleanup test data after test run."""
        
        # Cleanup expired data
        expired_count = test_data_cleanup.cleanup_expired()
        if expired_count > 0:
            print(f"🧹 Cleaned up {expired_count} expired test data entries")
        
        # Cleanup old data
        old_count = test_data_cleanup.cleanup_old_entries(days=1)
        if old_count > 0:
            print(f"🧹 Cleaned up {old_count} old test data entries")
        
        # Cleanup large data
        large_count = test_data_cleanup.cleanup_large_entries(size_threshold_mb=0.5)
        if large_count > 0:
            print(f"🧹 Cleaned up {large_count} large test data entries")
    
    def run_unit_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run unit tests."""
        return self.run_tests(
            test_types=["unit"],
            environment=TestEnvironment.UNIT,
            category=TestCategory.FAST,
            verbose=verbose
        )
    
    def run_integration_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run integration tests."""
        return self.run_tests(
            test_types=["integration"],
            environment=TestEnvironment.INTEGRATION,
            category=TestCategory.SLOW,
            verbose=verbose
        )
    
    def run_performance_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run performance tests."""
        return self.run_tests(
            test_types=["performance", "benchmark"],
            environment=TestEnvironment.PERFORMANCE,
            category=TestCategory.SLOW,
            performance=True,
            verbose=verbose
        )
    
    def run_security_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run security tests."""
        return self.run_tests(
            test_types=["security"],
            environment=TestEnvironment.SECURITY,
            category=TestCategory.CRITICAL,
            security=True,
            verbose=verbose
        )
    
    def run_monitoring_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run monitoring tests."""
        return self.run_tests(
            test_types=["monitoring"],
            environment=TestEnvironment.MONITORING,
            category=TestCategory.SLOW,
            monitoring=True,
            verbose=verbose
        )
    
    def run_load_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run load tests."""
        return self.run_tests(
            test_types=["load"],
            environment=TestEnvironment.LOAD,
            category=TestCategory.SLOW,
            load=True,
            verbose=verbose
        )
    
    def run_all_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run all tests."""
        return self.run_tests(
            test_types=["unit", "integration", "performance", "security", "monitoring"],
            environment=TestEnvironment.INTEGRATION,
            category=TestCategory.SLOW,
            performance=True,
            security=True,
            monitoring=True,
            verbose=verbose
        )


def main():
    """Main entry point for test runner."""
    
    parser = argparse.ArgumentParser(description="Refactored Test Runner for Copywriting Service")
    
    parser.add_argument("--test-type", choices=["unit", "integration", "performance", "security", "monitoring", "load", "all"], 
                       default="unit", help="Type of tests to run")
    parser.add_argument("--environment", choices=["unit", "integration", "performance", "security", "monitoring", "load"], 
                       help="Test environment")
    parser.add_argument("--category", choices=["fast", "slow", "critical", "optional"], 
                       help="Test category")
    parser.add_argument("--parallel", action="store_true", default=True, help="Run tests in parallel")
    parser.add_argument("--no-parallel", action="store_false", dest="parallel", help="Disable parallel execution")
    parser.add_argument("--coverage", action="store_true", default=True, help="Generate coverage report")
    parser.add_argument("--no-coverage", action="store_false", dest="coverage", help="Disable coverage report")
    parser.add_argument("--performance", action="store_true", help="Enable performance testing")
    parser.add_argument("--security", action="store_true", help="Enable security testing")
    parser.add_argument("--monitoring", action="store_true", help="Enable monitoring testing")
    parser.add_argument("--load", action="store_true", help="Enable load testing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--markers", nargs="+", help="Pytest markers to filter tests")
    
    args = parser.parse_args()
    
    # Create test runner
    runner = RefactoredTestRunner()
    
    # Map test types to methods
    test_methods = {
        "unit": runner.run_unit_tests,
        "integration": runner.run_integration_tests,
        "performance": runner.run_performance_tests,
        "security": runner.run_security_tests,
        "monitoring": runner.run_monitoring_tests,
        "load": runner.run_load_tests,
        "all": runner.run_all_tests
    }
    
    # Run tests
    if args.test_type in test_methods:
        result = test_methods[args.test_type](verbose=args.verbose)
    else:
        # Custom configuration
        result = runner.run_tests(
            test_types=[args.test_type] if args.test_type != "all" else None,
            environment=TestEnvironment(args.environment) if args.environment else None,
            category=TestCategory(args.category) if args.category else None,
            parallel=args.parallel,
            coverage=args.coverage,
            performance=args.performance,
            security=args.security,
            monitoring=args.monitoring,
            load=args.load,
            verbose=args.verbose,
            markers=args.markers
        )
    
    # Print results
    if result.get("success"):
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        print(f"Return code: {result.get('return_code')}")
        if result.get("stderr"):
            print(f"Error output: {result['stderr']}")
    
    print(f"⏱️  Duration: {result.get('duration', 0):.2f} seconds")
    
    return result.get("return_code", 1)


if __name__ == "__main__":
    sys.exit(main())
