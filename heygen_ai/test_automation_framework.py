#!/usr/bin/env python3
"""
Advanced Test Automation Framework
=================================

This module provides a comprehensive test automation framework with
intelligent test generation, execution orchestration, and result analysis.
"""

import sys
import json
import time
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import concurrent.futures
import threading
from queue import Queue, Empty

class TestStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"

class TestPriority(Enum):
    """Test priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class TestType(Enum):
    """Test types"""
    UNIT = "unit"
    INTEGRATION = "integration"
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    SECURITY = "security"
    REGRESSION = "regression"

@dataclass
class TestCase:
    """Test case definition"""
    id: str
    name: str
    description: str
    test_type: TestType
    priority: TestPriority
    file_path: str
    function_name: str
    dependencies: List[str]
    timeout: int = 300
    retry_count: int = 0
    tags: List[str] = None
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.parameters is None:
            self.parameters = {}

@dataclass
class TestResult:
    """Test execution result"""
    test_id: str
    status: TestStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration: float
    output: str
    error_message: Optional[str]
    retry_count: int
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class TestSuite:
    """Test suite definition"""
    id: str
    name: str
    description: str
    test_cases: List[TestCase]
    parallel_execution: bool = True
    max_workers: int = 4
    timeout: int = 1800
    retry_failed: bool = True
    stop_on_failure: bool = False

class TestAutomationFramework:
    """Advanced test automation framework"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_config(config_file)
        self.test_suites: Dict[str, TestSuite] = {}
        self.test_results: Dict[str, TestResult] = {}
        self.execution_queue = Queue()
        self.result_queue = Queue()
        self.running_tests: Dict[str, threading.Thread] = {}
        self.execution_stats = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "error_tests": 0,
            "start_time": None,
            "end_time": None
        }
        
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            "execution": {
                "max_workers": 4,
                "default_timeout": 300,
                "retry_failed": True,
                "retry_count": 2,
                "parallel_execution": True
            },
            "reporting": {
                "output_dir": "test_reports",
                "formats": ["json", "html", "xml"],
                "include_coverage": True,
                "include_performance": True
            },
            "notifications": {
                "enabled": False,
                "webhook_url": None,
                "email_recipients": []
            },
            "test_discovery": {
                "auto_discover": True,
                "test_patterns": ["test_*.py", "*_test.py"],
                "exclude_patterns": ["__pycache__", "*.pyc"]
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"⚠️  Warning: Could not load config file {config_file}: {e}")
        
        return default_config
    
    def discover_tests(self, base_path: str = ".", patterns: Optional[List[str]] = None) -> List[TestCase]:
        """Automatically discover test cases"""
        print("🔍 Discovering test cases...")
        
        if patterns is None:
            patterns = self.config["test_discovery"]["test_patterns"]
        
        base_path = Path(base_path)
        test_cases = []
        
        for pattern in patterns:
            for test_file in base_path.rglob(pattern):
                if self._should_exclude_file(test_file):
                    continue
                
                try:
                    file_test_cases = self._extract_tests_from_file(test_file)
                    test_cases.extend(file_test_cases)
                except Exception as e:
                    print(f"⚠️  Warning: Could not process {test_file}: {e}")
        
        print(f"✅ Discovered {len(test_cases)} test cases")
        return test_cases
    
    def _should_exclude_file(self, file_path: Path) -> bool:
        """Check if file should be excluded from test discovery"""
        exclude_patterns = self.config["test_discovery"]["exclude_patterns"]
        
        for pattern in exclude_patterns:
            if pattern in str(file_path):
                return True
        
        return False
    
    def _extract_tests_from_file(self, file_path: Path) -> List[TestCase]:
        """Extract test cases from a Python file"""
        test_cases = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple regex-based test extraction
            import re
            
            # Find test functions
            test_function_pattern = r'def\s+(test_\w+)\s*\([^)]*\):'
            test_functions = re.findall(test_function_pattern, content)
            
            # Find test classes
            test_class_pattern = r'class\s+(Test\w+)\s*\([^)]*\):'
            test_classes = re.findall(test_class_pattern, content)
            
            # Create test cases for functions
            for func_name in test_functions:
                test_case = TestCase(
                    id=f"{file_path.stem}_{func_name}",
                    name=func_name,
                    description=f"Test function {func_name} in {file_path.name}",
                    test_type=TestType.UNIT,
                    priority=TestPriority.MEDIUM,
                    file_path=str(file_path),
                    function_name=func_name,
                    dependencies=[],
                    timeout=self.config["execution"]["default_timeout"]
                )
                test_cases.append(test_case)
            
            # Create test cases for classes
            for class_name in test_classes:
                test_case = TestCase(
                    id=f"{file_path.stem}_{class_name}",
                    name=class_name,
                    description=f"Test class {class_name} in {file_path.name}",
                    test_type=TestType.UNIT,
                    priority=TestPriority.MEDIUM,
                    file_path=str(file_path),
                    function_name=class_name,
                    dependencies=[],
                    timeout=self.config["execution"]["default_timeout"]
                )
                test_cases.append(test_case)
        
        except Exception as e:
            print(f"⚠️  Error extracting tests from {file_path}: {e}")
        
        return test_cases
    
    def create_test_suite(self, suite_id: str, name: str, description: str, 
                         test_cases: List[TestCase], **kwargs) -> TestSuite:
        """Create a test suite"""
        test_suite = TestSuite(
            id=suite_id,
            name=name,
            description=description,
            test_cases=test_cases,
            parallel_execution=kwargs.get('parallel_execution', self.config["execution"]["parallel_execution"]),
            max_workers=kwargs.get('max_workers', self.config["execution"]["max_workers"]),
            timeout=kwargs.get('timeout', 1800),
            retry_failed=kwargs.get('retry_failed', self.config["execution"]["retry_failed"]),
            stop_on_failure=kwargs.get('stop_on_failure', False)
        )
        
        self.test_suites[suite_id] = test_suite
        return test_suite
    
    def execute_test_suite(self, suite_id: str) -> Dict[str, Any]:
        """Execute a test suite"""
        if suite_id not in self.test_suites:
            raise ValueError(f"Test suite '{suite_id}' not found")
        
        test_suite = self.test_suites[suite_id]
        print(f"🚀 Executing test suite: {test_suite.name}")
        
        # Initialize execution stats
        self.execution_stats = {
            "total_tests": len(test_suite.test_cases),
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "error_tests": 0,
            "start_time": datetime.now(),
            "end_time": None
        }
        
        # Execute tests
        if test_suite.parallel_execution:
            results = self._execute_tests_parallel(test_suite)
        else:
            results = self._execute_tests_sequential(test_suite)
        
        # Update final stats
        self.execution_stats["end_time"] = datetime.now()
        
        # Generate report
        report = self._generate_execution_report(test_suite, results)
        
        return report
    
    def _execute_tests_parallel(self, test_suite: TestSuite) -> List[TestResult]:
        """Execute tests in parallel"""
        print(f"⚡ Executing {len(test_suite.test_cases)} tests in parallel (max {test_suite.max_workers} workers)")
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=test_suite.max_workers) as executor:
            # Submit all tests
            future_to_test = {
                executor.submit(self._execute_single_test, test_case): test_case
                for test_case in test_suite.test_cases
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_test, timeout=test_suite.timeout):
                test_case = future_to_test[future]
                try:
                    result = future.result()
                    results.append(result)
                    self._update_execution_stats(result)
                except Exception as e:
                    error_result = TestResult(
                        test_id=test_case.id,
                        status=TestStatus.ERROR,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        duration=0.0,
                        output="",
                        error_message=str(e),
                        retry_count=0,
                        metadata={}
                    )
                    results.append(error_result)
                    self._update_execution_stats(error_result)
        
        return results
    
    def _execute_tests_sequential(self, test_suite: TestSuite) -> List[TestResult]:
        """Execute tests sequentially"""
        print(f"🔄 Executing {len(test_suite.test_cases)} tests sequentially")
        
        results = []
        for test_case in test_suite.test_cases:
            result = self._execute_single_test(test_case)
            results.append(result)
            self._update_execution_stats(result)
            
            # Stop on failure if configured
            if test_suite.stop_on_failure and result.status in [TestStatus.FAILED, TestStatus.ERROR]:
                print(f"🛑 Stopping execution due to failure in test: {test_case.id}")
                break
        
        return results
    
    def _execute_single_test(self, test_case: TestCase) -> TestResult:
        """Execute a single test case"""
        start_time = datetime.now()
        
        try:
            print(f"🧪 Running test: {test_case.name}")
            
            # Execute test using pytest
            cmd = [
                sys.executable, "-m", "pytest",
                f"{test_case.file_path}::{test_case.function_name}",
                "-v", "--tb=short", "--no-header"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=test_case.timeout
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Determine test status
            if result.returncode == 0:
                status = TestStatus.PASSED
                error_message = None
            elif result.returncode == 1:
                status = TestStatus.FAILED
                error_message = result.stderr
            elif result.returncode == 2:
                status = TestStatus.ERROR
                error_message = result.stderr
            else:
                status = TestStatus.ERROR
                error_message = f"Unexpected return code: {result.returncode}"
            
            test_result = TestResult(
                test_id=test_case.id,
                status=status,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                output=result.stdout,
                error_message=error_message,
                retry_count=0,
                metadata={
                    "return_code": result.returncode,
                    "command": " ".join(cmd)
                }
            )
            
            # Store result
            self.test_results[test_case.id] = test_result
            
            return test_result
            
        except subprocess.TimeoutExpired:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return TestResult(
                test_id=test_case.id,
                status=TestStatus.TIMEOUT,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                output="",
                error_message=f"Test timed out after {test_case.timeout} seconds",
                retry_count=0,
                metadata={}
            )
        
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return TestResult(
                test_id=test_case.id,
                status=TestStatus.ERROR,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                output="",
                error_message=str(e),
                retry_count=0,
                metadata={}
            )
    
    def _update_execution_stats(self, result: TestResult):
        """Update execution statistics"""
        if result.status == TestStatus.PASSED:
            self.execution_stats["passed_tests"] += 1
        elif result.status == TestStatus.FAILED:
            self.execution_stats["failed_tests"] += 1
        elif result.status == TestStatus.SKIPPED:
            self.execution_stats["skipped_tests"] += 1
        else:
            self.execution_stats["error_tests"] += 1
    
    def _generate_execution_report(self, test_suite: TestSuite, results: List[TestResult]) -> Dict[str, Any]:
        """Generate comprehensive execution report"""
        total_duration = (self.execution_stats["end_time"] - self.execution_stats["start_time"]).total_seconds()
        
        report = {
            "suite_info": {
                "id": test_suite.id,
                "name": test_suite.name,
                "description": test_suite.description,
                "total_tests": len(test_suite.test_cases)
            },
            "execution_summary": {
                "start_time": self.execution_stats["start_time"].isoformat(),
                "end_time": self.execution_stats["end_time"].isoformat(),
                "total_duration": total_duration,
                "passed_tests": self.execution_stats["passed_tests"],
                "failed_tests": self.execution_stats["failed_tests"],
                "skipped_tests": self.execution_stats["skipped_tests"],
                "error_tests": self.execution_stats["error_tests"],
                "success_rate": (self.execution_stats["passed_tests"] / self.execution_stats["total_tests"]) * 100 if self.execution_stats["total_tests"] > 0 else 0
            },
            "test_results": [asdict(result) for result in results],
            "performance_metrics": {
                "avg_test_duration": sum(r.duration for r in results) / len(results) if results else 0,
                "slowest_test": max(results, key=lambda r: r.duration).test_id if results else None,
                "fastest_test": min(results, key=lambda r: r.duration).test_id if results else None
            },
            "recommendations": self._generate_recommendations(results)
        }
        
        return report
    
    def _generate_recommendations(self, results: List[TestResult]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Failed tests
        failed_tests = [r for r in results if r.status == TestStatus.FAILED]
        if failed_tests:
            recommendations.append(f"🔧 {len(failed_tests)} tests failed - review and fix failing tests")
        
        # Slow tests
        slow_tests = [r for r in results if r.duration > 30]
        if slow_tests:
            recommendations.append(f"⏱️ {len(slow_tests)} tests are slow (>30s) - consider optimization")
        
        # Error tests
        error_tests = [r for r in results if r.status == TestStatus.ERROR]
        if error_tests:
            recommendations.append(f"🚨 {len(error_tests)} tests had errors - investigate test setup issues")
        
        # Timeout tests
        timeout_tests = [r for r in results if r.status == TestStatus.TIMEOUT]
        if timeout_tests:
            recommendations.append(f"⏰ {len(timeout_tests)} tests timed out - increase timeout or optimize tests")
        
        return recommendations
    
    def generate_html_report(self, report: Dict[str, Any], output_file: str = "test_report.html"):
        """Generate HTML test report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Execution Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
                .metric {{ background-color: #e8f4fd; padding: 15px; border-radius: 5px; text-align: center; }}
                .passed {{ color: #28a745; }}
                .failed {{ color: #dc3545; }}
                .skipped {{ color: #ffc107; }}
                .error {{ color: #dc3545; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .status-passed {{ color: #28a745; font-weight: bold; }}
                .status-failed {{ color: #dc3545; font-weight: bold; }}
                .status-skipped {{ color: #ffc107; font-weight: bold; }}
                .status-error {{ color: #dc3545; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧪 Test Execution Report</h1>
                <h2>{report['suite_info']['name']}</h2>
                <p>{report['suite_info']['description']}</p>
            </div>
            
            <div class="summary">
                <div class="metric">
                    <h3>Total Tests</h3>
                    <p style="font-size: 24px; font-weight: bold;">{report['execution_summary']['total_tests']}</p>
                </div>
                <div class="metric">
                    <h3 class="passed">Passed</h3>
                    <p style="font-size: 24px; font-weight: bold; color: #28a745;">{report['execution_summary']['passed_tests']}</p>
                </div>
                <div class="metric">
                    <h3 class="failed">Failed</h3>
                    <p style="font-size: 24px; font-weight: bold; color: #dc3545;">{report['execution_summary']['failed_tests']}</p>
                </div>
                <div class="metric">
                    <h3 class="skipped">Skipped</h3>
                    <p style="font-size: 24px; font-weight: bold; color: #ffc107;">{report['execution_summary']['skipped_tests']}</p>
                </div>
                <div class="metric">
                    <h3 class="error">Errors</h3>
                    <p style="font-size: 24px; font-weight: bold; color: #dc3545;">{report['execution_summary']['error_tests']}</p>
                </div>
            </div>
            
            <h3>📊 Execution Summary</h3>
            <ul>
                <li><strong>Start Time:</strong> {report['execution_summary']['start_time']}</li>
                <li><strong>End Time:</strong> {report['execution_summary']['end_time']}</li>
                <li><strong>Total Duration:</strong> {report['execution_summary']['total_duration']:.2f} seconds</li>
                <li><strong>Success Rate:</strong> {report['execution_summary']['success_rate']:.1f}%</li>
            </ul>
            
            <h3>🧪 Test Results</h3>
            <table>
                <tr>
                    <th>Test ID</th>
                    <th>Status</th>
                    <th>Duration (s)</th>
                    <th>Error Message</th>
                </tr>
        """
        
        for result in report['test_results']:
            status_class = f"status-{result['status'].value}"
            html_content += f"""
                <tr>
                    <td>{result['test_id']}</td>
                    <td class="{status_class}">{result['status'].value.upper()}</td>
                    <td>{result['duration']:.2f}</td>
                    <td>{result['error_message'] or ''}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h3>💡 Recommendations</h3>
            <ul>
        """
        
        for rec in report['recommendations']:
            html_content += f"<li>{rec}</li>"
        
        html_content += """
            </ul>
        </body>
        </html>
        """
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"✅ HTML report generated: {output_file}")
        except Exception as e:
            print(f"❌ Error generating HTML report: {e}")
    
    def save_report(self, report: Dict[str, Any], output_file: str = "test_report.json"):
        """Save test report to JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"✅ Test report saved: {output_file}")
        except Exception as e:
            print(f"❌ Error saving report: {e}")


def main():
    """Main function for test automation framework"""
    print("🤖 Advanced Test Automation Framework")
    print("=" * 50)
    
    # Initialize framework
    framework = TestAutomationFramework()
    
    # Discover tests
    print("🔍 Discovering tests...")
    test_cases = framework.discover_tests("tests")
    
    if not test_cases:
        print("⚠️  No tests discovered. Creating sample test suite...")
        
        # Create sample test cases
        sample_tests = [
            TestCase(
                id="sample_test_1",
                name="test_sample_functionality",
                description="Sample test case for demonstration",
                test_type=TestType.UNIT,
                priority=TestPriority.HIGH,
                file_path="tests/test_sample.py",
                function_name="test_sample_functionality",
                dependencies=[]
            ),
            TestCase(
                id="sample_test_2",
                name="test_integration",
                description="Sample integration test",
                test_type=TestType.INTEGRATION,
                priority=TestPriority.MEDIUM,
                file_path="tests/test_integration.py",
                function_name="test_integration",
                dependencies=["sample_test_1"]
            )
        ]
        test_cases = sample_tests
    
    # Create test suite
    test_suite = framework.create_test_suite(
        suite_id="comprehensive_tests",
        name="Comprehensive Test Suite",
        description="Complete test suite for the HeyGen AI system",
        test_cases=test_cases,
        parallel_execution=True,
        max_workers=4
    )
    
    print(f"📋 Created test suite with {len(test_cases)} test cases")
    
    # Execute test suite
    print("🚀 Executing test suite...")
    report = framework.execute_test_suite("comprehensive_tests")
    
    # Print summary
    print("\n📊 Execution Summary:")
    summary = report['execution_summary']
    print(f"  ✅ Passed: {summary['passed_tests']}")
    print(f"  ❌ Failed: {summary['failed_tests']}")
    print(f"  ⏭️  Skipped: {summary['skipped_tests']}")
    print(f"  🚨 Errors: {summary['error_tests']}")
    print(f"  📈 Success Rate: {summary['success_rate']:.1f}%")
    print(f"  ⏱️  Total Duration: {summary['total_duration']:.2f}s")
    
    # Print recommendations
    if report['recommendations']:
        print("\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  {rec}")
    
    # Save reports
    framework.save_report(report, "test_execution_report.json")
    framework.generate_html_report(report, "test_execution_report.html")
    
    print("\n🎉 Test automation framework execution completed!")
    print("📄 Check 'test_execution_report.json' and 'test_execution_report.html' for detailed results")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


