"""
Test Runner for Instagram Captions API v10.0

Comprehensive test execution and management system.
"""

import time
import sys
import traceback
from typing import Dict, Any, List, Optional, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

@dataclass
class TestResult:
    """Individual test result."""
    name: str
    status: TestStatus
    execution_time: float
    start_time: float
    end_time: float
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TestSuiteResult:
    """Test suite execution result."""
    name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    execution_time: float
    start_time: float
    end_time: float
    test_results: List[TestResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class TestRunner:
    """Advanced test runner with comprehensive reporting."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.test_suites: Dict[str, 'TestSuite'] = {}
        self.results: List[TestSuiteResult] = []
        self.current_suite: Optional[str] = None
        self.verbose = self.config.get('verbose', False)
        self.stop_on_failure = self.config.get('stop_on_failure', False)
        self.parallel_execution = self.config.get('parallel_execution', False)
        self.max_workers = self.config.get('max_workers', 4)
    
    def register_test_suite(self, name: str, test_suite: 'TestSuite'):
        """Register a test suite with the runner."""
        self.test_suites[name] = test_suite
        print(f"✅ Test suite registered: {name}")
    
    def run_test_suite(self, suite_name: str) -> TestSuiteResult:
        """Run a specific test suite."""
        if suite_name not in self.test_suites:
            raise ValueError(f"Test suite '{suite_name}' not found")
        
        test_suite = self.test_suites[suite_name]
        self.current_suite = suite_name
        
        print(f"\n🚀 Running test suite: {suite_name}")
        print("=" * 60)
        
        # Initialize suite result
        suite_result = TestSuiteResult(
            name=suite_name,
            total_tests=len(test_suite.tests),
            passed_tests=0,
            failed_tests=0,
            skipped_tests=0,
            error_tests=0,
            execution_time=0.0,
            start_time=time.time(),
            end_time=0.0
        )
        
        # Run tests
        for test_name, test_func in test_suite.tests.items():
            test_result = self._run_single_test(test_name, test_func)
            suite_result.test_results.append(test_result)
            
            # Update counters
            if test_result.status == TestStatus.PASSED:
                suite_result.passed_tests += 1
            elif test_result.status == TestStatus.FAILED:
                suite_result.failed_tests += 1
            elif test_result.status == TestStatus.SKIPPED:
                suite_result.skipped_tests += 1
            elif test_result.status == TestStatus.ERROR:
                suite_result.error_tests += 1
            
            # Stop on failure if configured
            if self.stop_on_failure and test_result.status in [TestStatus.FAILED, TestStatus.ERROR]:
                print(f"⛔ Stopping on failure: {test_name}")
                break
        
        # Finalize suite result
        suite_result.end_time = time.time()
        suite_result.execution_time = suite_result.end_time - suite_result.start_time
        
        # Print suite summary
        self._print_suite_summary(suite_result)
        
        # Store result
        self.results.append(suite_result)
        self.current_suite = None
        
        return suite_result
    
    def run_all_test_suites(self) -> List[TestSuiteResult]:
        """Run all registered test suites."""
        if not self.test_suites:
            print("⚠️ No test suites registered")
            return []
        
        print(f"\n🚀 Running all test suites ({len(self.test_suites)} suites)")
        print("=" * 80)
        
        all_results = []
        
        for suite_name in self.test_suites.keys():
            try:
                suite_result = self.run_test_suite(suite_name)
                all_results.append(suite_result)
            except Exception as e:
                print(f"❌ Error running test suite '{suite_name}': {e}")
                # Create error result
                error_result = TestSuiteResult(
                    name=suite_name,
                    total_tests=0,
                    passed_tests=0,
                    failed_tests=0,
                    skipped_tests=0,
                    error_tests=1,
                    execution_time=0.0,
                    start_time=time.time(),
                    end_time=time.time(),
                    metadata={'error': str(e)}
                )
                all_results.append(error_result)
        
        # Print overall summary
        self._print_overall_summary(all_results)
        
        return all_results
    
    def _run_single_test(self, test_name: str, test_func: Callable) -> TestResult:
        """Run a single test and return the result."""
        print(f"🧪 Running test: {test_name}")
        
        start_time = time.time()
        test_result = TestResult(
            name=test_name,
            status=TestStatus.RUNNING,
            execution_time=0.0,
            start_time=start_time,
            end_time=0.0
        )
        
        try:
            # Check if test should be skipped
            if hasattr(test_func, '__skip__') and test_func.__skip__:
                test_result.status = TestStatus.SKIPPED
                test_result.metadata['skip_reason'] = getattr(test_func, '__skip_reason__', 'No reason provided')
                print(f"⏭️ Test skipped: {test_name}")
                return test_result
            
            # Execute test
            if self.verbose:
                print(f"   Executing: {test_name}")
            
            # Run the test
            test_func()
            
            # Test passed
            test_result.status = TestStatus.PASSED
            print(f"✅ Test passed: {test_name}")
            
        except AssertionError as e:
            # Test failed
            test_result.status = TestStatus.FAILED
            test_result.error_message = str(e)
            test_result.error_traceback = traceback.format_exc()
            print(f"❌ Test failed: {test_name} - {e}")
            
        except Exception as e:
            # Test error
            test_result.status = TestStatus.ERROR
            test_result.error_message = str(e)
            test_result.error_traceback = traceback.format_exc()
            print(f"💥 Test error: {test_name} - {e}")
        
        finally:
            # Finalize test result
            test_result.end_time = time.time()
            test_result.execution_time = test_result.end_time - test_result.start_time
            
            if self.verbose and test_result.status == TestStatus.PASSED:
                print(f"   Execution time: {test_result.execution_time:.3f}s")
        
        return test_result
    
    def _print_suite_summary(self, suite_result: TestSuiteResult):
        """Print summary for a test suite."""
        print("\n" + "=" * 60)
        print(f"📊 Test Suite Summary: {suite_result.name}")
        print("=" * 60)
        
        # Statistics
        print(f"Total Tests: {suite_result.total_tests}")
        print(f"✅ Passed: {suite_result.passed_tests}")
        print(f"❌ Failed: {suite_result.failed_tests}")
        print(f"⏭️ Skipped: {suite_result.skipped_tests}")
        print(f"💥 Errors: {suite_result.error_tests}")
        print(f"⏱️ Execution Time: {suite_result.execution_time:.3f}s")
        
        # Success rate
        if suite_result.total_tests > 0:
            success_rate = (suite_result.passed_tests / suite_result.total_tests) * 100
            print(f"📈 Success Rate: {success_rate:.1f}%")
        
        # Failed tests details
        if suite_result.failed_tests > 0 or suite_result.error_tests > 0:
            print("\n❌ Failed/Error Tests:")
            for result in suite_result.test_results:
                if result.status in [TestStatus.FAILED, TestStatus.ERROR]:
                    print(f"   • {result.name}: {result.error_message}")
    
    def _print_overall_summary(self, all_results: List[TestSuiteResult]):
        """Print overall summary for all test suites."""
        print("\n" + "=" * 80)
        print("🏆 OVERALL TEST EXECUTION SUMMARY")
        print("=" * 80)
        
        total_suites = len(all_results)
        total_tests = sum(r.total_tests for r in all_results)
        total_passed = sum(r.passed_tests for r in all_results)
        total_failed = sum(r.failed_tests for r in all_results)
        total_skipped = sum(r.skipped_tests for r in all_results)
        total_errors = sum(r.error_tests for r in all_results)
        total_execution_time = sum(r.execution_time for r in all_results)
        
        print(f"Test Suites: {total_suites}")
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {total_passed}")
        print(f"❌ Failed: {total_failed}")
        print(f"⏭️ Skipped: {total_skipped}")
        print(f"💥 Errors: {total_errors}")
        print(f"⏱️ Total Execution Time: {total_execution_time:.3f}s")
        
        if total_tests > 0:
            overall_success_rate = (total_passed / total_tests) * 100
            print(f"📈 Overall Success Rate: {overall_success_rate:.1f}%")
        
        # Suite-by-suite breakdown
        print("\n📋 Suite-by-Suite Breakdown:")
        for result in all_results:
            status_icon = "✅" if result.passed_tests == result.total_tests else "⚠️"
            print(f"   {status_icon} {result.name}: {result.passed_tests}/{result.total_tests} passed")
    
    def get_test_statistics(self) -> Dict[str, Any]:
        """Get comprehensive test statistics."""
        if not self.results:
            return {"message": "No test results available"}
        
        stats = {
            'total_suites': len(self.results),
            'total_tests': sum(r.total_tests for r in self.results),
            'total_passed': sum(r.passed_tests for r in self.results),
            'total_failed': sum(r.failed_tests for r in self.results),
            'total_skipped': sum(r.skipped_tests for r in self.results),
            'total_errors': sum(r.error_tests for r in self.results),
            'total_execution_time': sum(r.execution_time for r in self_results),
            'success_rate': 0.0,
            'suite_results': []
        }
        
        if stats['total_tests'] > 0:
            stats['success_rate'] = (stats['total_passed'] / stats['total_tests']) * 100
        
        # Add suite results
        for result in self.results:
            suite_stats = {
                'name': result.name,
                'total_tests': result.total_tests,
                'passed_tests': result.passed_tests,
                'failed_tests': result.failed_tests,
                'skipped_tests': result.skipped_tests,
                'error_tests': result.error_tests,
                'execution_time': result.execution_time,
                'success_rate': (result.passed_tests / result.total_tests) * 100 if result.total_tests > 0 else 0
            }
            stats['suite_results'].append(suite_stats)
        
        return stats
    
    def export_results(self, format: str = "json", filename: Optional[str] = None) -> str:
        """Export test results in specified format."""
        if not self.results:
            return "No results to export"
        
        if format.lower() == "json":
            import json
            content = json.dumps(self.get_test_statistics(), indent=2, default=str)
            extension = "json"
        elif format.lower() == "yaml":
            import yaml
            content = yaml.dump(self.get_test_statistics(), default_flow_style=False, indent=2)
            extension = "yaml"
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.{extension}"
        
        # Write to file
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"📄 Test results exported to: {filename}")
            return filename
        except Exception as e:
            print(f"❌ Error exporting results: {e}")
            return str(e)
    
    def clear_results(self):
        """Clear all test results."""
        self.results.clear()
        print("🧹 Test results cleared")
    
    def get_failed_tests(self) -> List[TestResult]:
        """Get list of all failed tests."""
        failed_tests = []
        for suite_result in self.results:
            for test_result in suite_result.test_results:
                if test_result.status in [TestStatus.FAILED, TestStatus.ERROR]:
                    failed_tests.append(test_result)
        return failed_tests
    
    def rerun_failed_tests(self) -> List[TestResult]:
        """Rerun all failed tests."""
        failed_tests = self.get_failed_tests()
        if not failed_tests:
            print("✅ No failed tests to rerun")
            return []
        
        print(f"🔄 Rerunning {len(failed_tests)} failed tests")
        
        rerun_results = []
        for test_result in failed_tests:
            # Find the test function
            suite_name = None
            for suite_result in self.results:
                if test_result in suite_result.test_results:
                    suite_name = suite_result.name
                    break
            
            if suite_name and suite_name in self.test_suites:
                test_suite = self.test_suites[suite_name]
                if test_result.name in test_suite.tests:
                    test_func = test_suite.tests[test_result.name]
                    rerun_result = self._run_single_test(test_result.name, test_func)
                    rerun_results.append(rerun_result)
        
        return rerun_results






