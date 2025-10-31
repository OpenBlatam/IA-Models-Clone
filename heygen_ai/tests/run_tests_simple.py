#!/usr/bin/env python3
"""
Simple test runner for refactored HeyGen AI test suite.
Executes all refactored tests with basic reporting.
"""

import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

def run_refactored_tests():
    """Run all refactored tests with comprehensive reporting."""
    print("🚀 Starting Refactored Test Suite Execution")
    print("=" * 60)
    
    start_time = time.time()
    
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
    
    results = []
    
    # Run each test category
    for category in test_categories:
        print(f"\n📋 Running {category['name']}...")
        print(f"   Description: {category['description']}")
        
        result = run_test_category(category)
        results.append(result)
        
        # Print category result
        status_icon = "✅" if result["status"] == "passed" else "❌"
        print(f"   {status_icon} {category['name']}: {result['status']} ({result['duration']:.2f}s)")
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Generate and print summary
    print_summary(results, total_duration)
    
    # Save report
    save_report(results, total_duration)
    
    return results

def run_test_category(category: Dict[str, str]) -> Dict[str, Any]:
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
        ], capture_output=True, text=True, timeout=300)
        
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
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def print_summary(results: List[Dict[str, Any]], total_duration: float):
    """Print test execution summary."""
    print("\n" + "=" * 60)
    print("📊 REFACTORED TEST SUITE EXECUTION SUMMARY")
    print("=" * 60)
    
    # Calculate summary statistics
    total_tests = sum(r["test_count"] for r in results)
    total_passed = sum(r["passed_count"] for r in results)
    total_failed = sum(r["failed_count"] for r in results)
    total_categories = len(results)
    passed_categories = sum(1 for r in results if r["status"] == "passed")
    
    # Execution info
    print(f"⏱️  Total Duration: {total_duration:.2f} seconds")
    print(f"🚀 Test Runner: Refactored Test Suite v2.0")
    
    # Test statistics
    print(f"\n📈 Test Statistics:")
    print(f"   Categories: {passed_categories}/{total_categories} passed ({passed_categories/total_categories*100:.1f}%)")
    print(f"   Tests: {total_passed}/{total_tests} passed ({total_passed/total_tests*100:.1f}%)")
    
    # Category results
    print(f"\n📋 Category Results:")
    for result in results:
        status_icon = "✅" if result["status"] == "passed" else "❌"
        print(f"   {status_icon} {result['name']}: {result['status']} ({result['duration']:.2f}s)")
    
    # Overall result
    overall_success = passed_categories == total_categories and total_passed == total_tests
    result_icon = "🎉" if overall_success else "⚠️"
    result_text = "ALL TESTS PASSED" if overall_success else "SOME TESTS FAILED"
    
    print(f"\n{result_icon} OVERALL RESULT: {result_text}")
    print("=" * 60)

def save_report(results: List[Dict[str, Any]], total_duration: float):
    """Save test report to file."""
    report_dir = Path("test_reports")
    report_dir.mkdir(exist_ok=True)
    
    # Calculate summary statistics
    total_tests = sum(r["test_count"] for r in results)
    total_passed = sum(r["passed_count"] for r in results)
    total_failed = sum(r["failed_count"] for r in results)
    total_categories = len(results)
    passed_categories = sum(1 for r in results if r["status"] == "passed")
    
    report = {
        "execution_summary": {
            "total_duration": total_duration,
            "timestamp": datetime.now().isoformat(),
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
        "category_results": results
    }
    
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
        results = run_refactored_tests()
        
        # Exit with appropriate code
        total_tests = sum(r["test_count"] for r in results)
        total_passed = sum(r["passed_count"] for r in results)
        overall_success = total_passed == total_tests
        
        sys.exit(0 if overall_success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error running tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
