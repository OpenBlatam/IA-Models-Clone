#!/usr/bin/env python3
"""
Main test runner for refactored HeyGen AI test suite.
Executes all refactored tests with comprehensive reporting.
"""

import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent
sys.path.insert(0, str(parent_dir))

# Import utilities directly
from utils.test_utilities import TestRunner, PerformanceProfiler, TestDataGenerator
from config.test_config import get_test_config

class RefactoredTestRunner:
    """Main test runner for refactored test suite."""
    
    def __init__(self):
        self.config = get_test_config()
        self.test_runner = TestRunner()
        self.performance_profiler = PerformanceProfiler()
        self.data_generator = TestDataGenerator(seed=42)
        self.results: List[Dict[str, Any]] = []
        self.start_time = None
        self.end_time = None
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all refactored tests."""
        print("🚀 Starting Refactored Test Suite Execution")
        print("=" * 60)
        
        self.start_time = time.time()
        
        # Test categories to run
        test_categories = [
            {
                "name": "Basic Functionality Tests",
                "file": "tests/test_refactored_simple.py::TestBasicFunctionality",
                "description": "Core Python functionality and imports"
            },
            {
                "name": "Performance Tests",
                "file": "tests/test_refactored_simple.py::TestPerformanceRefactored",
                "description": "Performance and benchmarking tests"
            },
            {
                "name": "Integration Tests",
                "file": "tests/test_refactored_simple.py::TestIntegrationRefactored",
                "description": "File system and subprocess integration"
            },
            {
                "name": "Unit Tests",
                "file": "tests/test_refactored_simple.py::TestUnitRefactored",
                "description": "Data structures and error handling"
            },
            {
                "name": "Data Generation Tests",
                "file": "tests/test_refactored_simple.py::TestDataGeneration",
                "description": "Test data generation functionality"
            },
            {
                "name": "Performance Profiling Tests",
                "file": "tests/test_refactored_simple.py::TestPerformanceProfiling",
                "description": "Performance profiling and measurement"
            },
            {
                "name": "Assertion Tests",
                "file": "tests/test_refactored_simple.py::TestAssertions",
                "description": "Custom assertion functionality"
            },
            {
                "name": "Test Runner Tests",
                "file": "tests/test_refactored_simple.py::TestRunnerFunctionality",
                "description": "Test runner functionality"
            }
        ]
        
        # Run each test category
        for category in test_categories:
            print(f"\n📋 Running {category['name']}...")
            print(f"   Description: {category['description']}")
            
            result = self._run_test_category(category)
            self.results.append(result)
            
            # Print category result
            status_icon = "✅" if result["status"] == "passed" else "❌"
            print(f"   {status_icon} {category['name']}: {result['status']} ({result['duration']:.2f}s)")
        
        self.end_time = time.time()
        
        # Generate comprehensive report
        report = self._generate_report()
        self._print_summary(report)
        self._save_report(report)
        
        return report
    
    def _run_test_category(self, category: Dict[str, str]) -> Dict[str, Any]:
        """Run a specific test category."""
        start_time = time.time()
        
        try:
            # Run pytest for the specific category
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                category["file"], 
                "-v", 
                "--tb=short",
                "--no-header"
            ], capture_output=True, text=True, timeout=self.config.test_timeout)
            
            duration = time.time() - start_time
            
            # Parse results
            output_lines = result.stdout.split('\n')
            test_count = 0
            passed_count = 0
            failed_count = 0
            
            for line in output_lines:
                if "::" in line and ("PASSED" in line or "FAILED" in line):
                    test_count += 1
                    if "PASSED" in line:
                        passed_count += 1
                    elif "FAILED" in line:
                        failed_count += 1
            
            status = "passed" if result.returncode == 0 else "failed"
            
            return {
                "name": category["name"],
                "description": category["description"],
                "status": status,
                "duration": duration,
                "test_count": test_count,
                "passed_count": passed_count,
                "failed_count": failed_count,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timestamp": datetime.now().isoformat()
            }
            
        except subprocess.TimeoutExpired:
            return {
                "name": category["name"],
                "description": category["description"],
                "status": "timeout",
                "duration": time.time() - start_time,
                "test_count": 0,
                "passed_count": 0,
                "failed_count": 0,
                "return_code": -1,
                "stdout": "",
                "stderr": f"Test category timed out after {self.config.test_timeout}s",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "name": category["name"],
                "description": category["description"],
                "status": "error",
                "duration": time.time() - start_time,
                "test_count": 0,
                "passed_count": 0,
                "failed_count": 0,
                "return_code": -1,
                "stdout": "",
                "stderr": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        # Calculate summary statistics
        total_tests = sum(r["test_count"] for r in self.results)
        total_passed = sum(r["passed_count"] for r in self.results)
        total_failed = sum(r["failed_count"] for r in self.results)
        total_categories = len(self.results)
        passed_categories = sum(1 for r in self.results if r["status"] == "passed")
        
        # Performance metrics
        performance_summary = self.performance_profiler.get_summary()
        
        # Test data generation summary
        data_summary = {
            "total_generated": len(self.data_generator.generated_data),
            "data_types": list(set(data.data_type.value for data in self.data_generator.generated_data))
        }
        
        return {
            "execution_summary": {
                "total_duration": total_duration,
                "start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
                "test_runner": "Refactored Test Suite v2.0"
            },
            "test_statistics": {
                "total_categories": total_categories,
                "passed_categories": passed_categories,
                "failed_categories": total_categories - passed_categories,
                "total_tests": total_tests,
                "passed_tests": total_passed,
                "failed_tests": total_failed,
                "success_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
                "category_success_rate": (passed_categories / total_categories * 100) if total_categories > 0 else 0
            },
            "performance_metrics": performance_summary,
            "data_generation": data_summary,
            "category_results": self.results,
            "configuration": {
                "environment": self.config.environment.value,
                "log_level": self.config.log_level.value,
                "debug_mode": self.config.debug_mode,
                "test_timeout": self.config.test_timeout,
                "parallel_workers": self.config.parallel_workers
            }
        }
    
    def _print_summary(self, report: Dict[str, Any]):
        """Print test execution summary."""
        print("\n" + "=" * 60)
        print("📊 REFACTORED TEST SUITE EXECUTION SUMMARY")
        print("=" * 60)
        
        # Execution info
        exec_summary = report["execution_summary"]
        print(f"⏱️  Total Duration: {exec_summary['total_duration']:.2f} seconds")
        print(f"🚀 Test Runner: {exec_summary['test_runner']}")
        
        # Test statistics
        stats = report["test_statistics"]
        print(f"\n📈 Test Statistics:")
        print(f"   Categories: {stats['passed_categories']}/{stats['total_categories']} passed ({stats['category_success_rate']:.1f}%)")
        print(f"   Tests: {stats['passed_tests']}/{stats['total_tests']} passed ({stats['success_rate']:.1f}%)")
        
        # Performance metrics
        perf = report["performance_metrics"]
        if perf:
            print(f"\n⚡ Performance Metrics:")
            print(f"   Total Operations: {perf.get('total_operations', 0)}")
            print(f"   Average Duration: {perf.get('average_duration', 0):.3f}s")
            print(f"   Max Duration: {perf.get('max_duration', 0):.3f}s")
            print(f"   Average Throughput: {perf.get('average_throughput', 0):.1f} ops/s")
        
        # Data generation
        data = report["data_generation"]
        print(f"\n📊 Data Generation:")
        print(f"   Total Generated: {data['total_generated']}")
        print(f"   Data Types: {', '.join(data['data_types'])}")
        
        # Category results
        print(f"\n📋 Category Results:")
        for result in report["category_results"]:
            status_icon = "✅" if result["status"] == "passed" else "❌"
            print(f"   {status_icon} {result['name']}: {result['status']} ({result['duration']:.2f}s)")
        
        # Overall result
        overall_success = stats["category_success_rate"] == 100 and stats["success_rate"] == 100
        result_icon = "🎉" if overall_success else "⚠️"
        result_text = "ALL TESTS PASSED" if overall_success else "SOME TESTS FAILED"
        
        print(f"\n{result_icon} OVERALL RESULT: {result_text}")
        print("=" * 60)
    
    def _save_report(self, report: Dict[str, Any]):
        """Save test report to file."""
        report_dir = Path("test_reports")
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"refactored_test_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Report saved to: {report_file}")

def main():
    """Main entry point."""
    print("🔄 HeyGen AI - Refactored Test Suite Runner")
    print("Version: 2.0 | Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print()
    
    try:
        runner = RefactoredTestRunner()
        report = runner.run_all_tests()
        
        # Exit with appropriate code
        overall_success = (report["test_statistics"]["category_success_rate"] == 100 and 
                          report["test_statistics"]["success_rate"] == 100)
        sys.exit(0 if overall_success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error running tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
