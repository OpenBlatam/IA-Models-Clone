from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
import subprocess
from tests.conftest_optimized import (
            from tests.unit.test_optimized_unit import (
            from tests.integration.test_optimized_integration import (
            from tests.load.test_optimized_load import (
            from tests.debug.test_optimized_debug import (
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Optimized Test Runner
====================

Clean, fast, and efficient test runner with comprehensive reporting.
"""


# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our optimized fixtures and utilities
    OptimizedTestDataGenerator,
    OptimizedPerformanceMonitor,
    OptimizedTestUtils,
    OptimizedAsyncUtils
)


class OptimizedTestRunner:
    """Optimized test runner with comprehensive reporting."""
    
    def __init__(self) -> Any:
        self.test_data_generator = OptimizedTestDataGenerator()
        self.performance_monitor = OptimizedPerformanceMonitor()
        self.test_utils = OptimizedTestUtils()
        self.async_utils = OptimizedAsyncUtils()
        
        self.results = {
            "unit_tests": {},
            "integration_tests": {},
            "load_tests": {},
            "debug_tests": {},
            "performance_metrics": {},
            "summary": {}
        }
        
        self.start_time = time.time()
    
    async def run_unit_tests(self) -> Dict[str, Any]:
        """Run optimized unit tests."""
        print("🔬 Running Optimized Unit Tests...")
        
        self.performance_monitor.start_monitoring("unit_tests")
        
        try:
            # Import and run unit tests
                TestOptimizedDataGeneration,
                TestOptimizedPerformance,
                TestOptimizedAsyncOperations,
                TestOptimizedMocking,
                TestOptimizedPerformanceMonitoring,
                TestOptimizedFactoryBoy,
                TestOptimizedErrorHandling,
                TestOptimizedBenchmarks
            )
            
            # Run test classes
            test_classes = [
                TestOptimizedDataGeneration,
                TestOptimizedPerformance,
                TestOptimizedAsyncOperations,
                TestOptimizedMocking,
                TestOptimizedPerformanceMonitoring,
                TestOptimizedFactoryBoy,
                TestOptimizedErrorHandling,
                TestOptimizedBenchmarks
            ]
            
            results = {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "errors": [],
                "test_classes": len(test_classes),
                "execution_time": 0
            }
            
            for test_class in test_classes:
                try:
                    # Create test instance and run methods
                    test_instance = test_class()
                    
                    # Get all test methods
                    test_methods = [method for method in dir(test_instance) 
                                  if method.startswith('test_')]
                    
                    results["total_tests"] += len(test_methods)
                    
                    for method_name in test_methods:
                        try:
                            method = getattr(test_instance, method_name)
                            
                            # Run test method
                            if asyncio.iscoroutinefunction(method):
                                await method()
                            else:
                                method()
                            
                            results["passed"] += 1
                            print(f"  ✅ {method_name}")
                            
                        except Exception as e:
                            results["failed"] += 1
                            error_info = {
                                "test_class": test_class.__name__,
                                "test_method": method_name,
                                "error": str(e)
                            }
                            results["errors"].append(error_info)
                            print(f"  ❌ {method_name}: {e}")
                
                except Exception as e:
                    results["errors"].append({
                        "test_class": test_class.__name__,
                        "error": str(e)
                    })
            
            metrics = self.performance_monitor.stop_monitoring("unit_tests")
            results["execution_time"] = metrics["duration"]
            results["performance_metrics"] = metrics
            
            self.results["unit_tests"] = results
            
            print(f"  📊 Unit Tests: {results['passed']}/{results['total_tests']} passed")
            print(f"  ⏱️  Execution time: {results['execution_time']:.2f}s")
            
            return results
            
        except Exception as e:
            error_result = {
                "error": str(e),
                "total_tests": 0,
                "passed": 0,
                "failed": 1
            }
            self.results["unit_tests"] = error_result
            return error_result
    
    async def run_integration_tests(self) -> Dict[str, Any]:
        """Run optimized integration tests."""
        print("🔗 Running Optimized Integration Tests...")
        
        self.performance_monitor.start_monitoring("integration_tests")
        
        try:
            # Import and run integration tests
                TestOptimizedAPIIntegration,
                TestOptimizedCacheIntegration,
                TestOptimizedNLPIntegration,
                TestOptimizedRepositoryIntegration,
                TestOptimizedPerformanceIntegration,
                TestOptimizedErrorHandlingIntegration,
                TestOptimizedDataFlowIntegration
            )
            
            # Run test classes
            test_classes = [
                TestOptimizedAPIIntegration,
                TestOptimizedCacheIntegration,
                TestOptimizedNLPIntegration,
                TestOptimizedRepositoryIntegration,
                TestOptimizedPerformanceIntegration,
                TestOptimizedErrorHandlingIntegration,
                TestOptimizedDataFlowIntegration
            ]
            
            results = {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "errors": [],
                "test_classes": len(test_classes),
                "execution_time": 0
            }
            
            for test_class in test_classes:
                try:
                    # Create test instance and run methods
                    test_instance = test_class()
                    
                    # Get all test methods
                    test_methods = [method for method in dir(test_instance) 
                                  if method.startswith('test_')]
                    
                    results["total_tests"] += len(test_methods)
                    
                    for method_name in test_methods:
                        try:
                            method = getattr(test_instance, method_name)
                            
                            # Run test method
                            if asyncio.iscoroutinefunction(method):
                                await method()
                            else:
                                method()
                            
                            results["passed"] += 1
                            print(f"  ✅ {method_name}")
                            
                        except Exception as e:
                            results["failed"] += 1
                            error_info = {
                                "test_class": test_class.__name__,
                                "test_method": method_name,
                                "error": str(e)
                            }
                            results["errors"].append(error_info)
                            print(f"  ❌ {method_name}: {e}")
                
                except Exception as e:
                    results["errors"].append({
                        "test_class": test_class.__name__,
                        "error": str(e)
                    })
            
            metrics = self.performance_monitor.stop_monitoring("integration_tests")
            results["execution_time"] = metrics["duration"]
            results["performance_metrics"] = metrics
            
            self.results["integration_tests"] = results
            
            print(f"  📊 Integration Tests: {results['passed']}/{results['total_tests']} passed")
            print(f"  ⏱️  Execution time: {results['execution_time']:.2f}s")
            
            return results
            
        except Exception as e:
            error_result = {
                "error": str(e),
                "total_tests": 0,
                "passed": 0,
                "failed": 1
            }
            self.results["integration_tests"] = error_result
            return error_result
    
    async def run_load_tests(self) -> Dict[str, Any]:
        """Run optimized load tests."""
        print("⚡ Running Optimized Load Tests...")
        
        self.performance_monitor.start_monitoring("load_tests")
        
        try:
            # Import and run load tests
                OptimizedLoadTester,
                TestOptimizedLoadTesting,
                TestOptimizedStressTesting,
                TestOptimizedEnduranceTesting,
                TestOptimizedScalabilityTesting
            )
            
            # Create load tester
            load_tester = OptimizedLoadTester()
            
            # Run load test scenarios
            results = {
                "load_scenarios": {},
                "total_scenarios": 0,
                "passed": 0,
                "failed": 0,
                "errors": [],
                "execution_time": 0
            }
            
            # Test scenarios
            scenarios = [
                ("low_load", lambda: asyncio.sleep(0.01), 10.0, 10.0, 5),
                ("medium_load", lambda: asyncio.sleep(0.005), 15.0, 50.0, 20),
                ("high_load", lambda: asyncio.sleep(0.002), 20.0, 100.0, 50)
            ]
            
            for scenario_name, operation, duration, target_rps, max_concurrent in scenarios:
                try:
                    print(f"  🚀 Running {scenario_name} scenario...")
                    
                    scenario_result = await load_tester.run_single_load_test(
                        operation, duration, target_rps, max_concurrent
                    )
                    
                    results["load_scenarios"][scenario_name] = scenario_result
                    results["total_scenarios"] += 1
                    
                    # Check if scenario passed
                    if scenario_result["success_rate"] > 0.8:
                        results["passed"] += 1
                        print(f"    ✅ {scenario_name}: Success rate {scenario_result['success_rate']:.2%}")
                    else:
                        results["failed"] += 1
                        print(f"    ❌ {scenario_name}: Success rate {scenario_result['success_rate']:.2%}")
                
                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append({
                        "scenario": scenario_name,
                        "error": str(e)
                    })
                    print(f"    ❌ {scenario_name}: {e}")
            
            metrics = self.performance_monitor.stop_monitoring("load_tests")
            results["execution_time"] = metrics["duration"]
            results["performance_metrics"] = metrics
            
            self.results["load_tests"] = results
            
            print(f"  📊 Load Tests: {results['passed']}/{results['total_scenarios']} scenarios passed")
            print(f"  ⏱️  Execution time: {results['execution_time']:.2f}s")
            
            return results
            
        except Exception as e:
            error_result = {
                "error": str(e),
                "total_scenarios": 0,
                "passed": 0,
                "failed": 1
            }
            self.results["load_tests"] = error_result
            return error_result
    
    async def run_debug_tests(self) -> Dict[str, Any]:
        """Run optimized debug tests."""
        print("🐛 Running Optimized Debug Tests...")
        
        self.performance_monitor.start_monitoring("debug_tests")
        
        try:
            # Import and run debug tests
                OptimizedDebugger,
                OptimizedProfiler,
                OptimizedMemoryTracker,
                OptimizedErrorTracker,
                TestOptimizedDebugging,
                TestOptimizedProfiling,
                TestOptimizedMemoryTracking,
                TestOptimizedErrorTracking
            )
            
            # Run debug utilities tests
            results = {
                "debug_utilities": {},
                "total_utilities": 0,
                "passed": 0,
                "failed": 0,
                "errors": [],
                "execution_time": 0
            }
            
            # Test debug utilities
            utilities = [
                ("debugger", OptimizedDebugger),
                ("profiler", OptimizedProfiler),
                ("memory_tracker", OptimizedMemoryTracker),
                ("error_tracker", OptimizedErrorTracker)
            ]
            
            for utility_name, utility_class in utilities:
                try:
                    print(f"  🔧 Testing {utility_name}...")
                    
                    # Create utility instance
                    utility_instance = utility_class()
                    
                    # Test basic functionality
                    if utility_name == "debugger":
                        utility_instance.log_debug("Test message")
                        summary = utility_instance.get_debug_summary()
                        assert summary["total_logs"] > 0
                    
                    elif utility_name == "profiler":
                        with utility_instance.profile("test"):
                            time.sleep(0.01)
                        summary = utility_instance.get_profile_summary()
                        assert summary["total_profiles"] > 0
                    
                    elif utility_name == "memory_tracker":
                        utility_instance.take_snapshot("test")
                        summary = utility_instance.get_memory_summary()
                        assert summary["total_snapshots"] > 0
                    
                    elif utility_name == "error_tracker":
                        try:
                            raise ValueError("Test error")
                        except ValueError as e:
                            utility_instance.track_error(e)
                        summary = utility_instance.get_error_summary()
                        assert summary["total_errors"] > 0
                    
                    results["debug_utilities"][utility_name] = "passed"
                    results["total_utilities"] += 1
                    results["passed"] += 1
                    print(f"    ✅ {utility_name}")
                
                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append({
                        "utility": utility_name,
                        "error": str(e)
                    })
                    print(f"    ❌ {utility_name}: {e}")
            
            metrics = self.performance_monitor.stop_monitoring("debug_tests")
            results["execution_time"] = metrics["duration"]
            results["performance_metrics"] = metrics
            
            self.results["debug_tests"] = results
            
            print(f"  📊 Debug Tests: {results['passed']}/{results['total_utilities']} utilities passed")
            print(f"  ⏱️  Execution time: {results['execution_time']:.2f}s")
            
            return results
            
        except Exception as e:
            error_result = {
                "error": str(e),
                "total_utilities": 0,
                "passed": 0,
                "failed": 1
            }
            self.results["debug_tests"] = error_result
            return error_result
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        print("📈 Generating Performance Report...")
        
        report = {
            "test_suite_performance": {},
            "overall_metrics": {},
            "recommendations": []
        }
        
        # Calculate overall metrics
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_execution_time = 0
        
        for test_type, results in self.results.items():
            if isinstance(results, dict) and "total_tests" in results:
                total_tests += results.get("total_tests", 0)
                total_passed += results.get("passed", 0)
                total_failed += results.get("failed", 0)
                total_execution_time += results.get("execution_time", 0)
        
        # Overall metrics
        report["overall_metrics"] = {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "success_rate": total_passed / total_tests if total_tests > 0 else 0,
            "total_execution_time": total_execution_time,
            "tests_per_second": total_tests / total_execution_time if total_execution_time > 0 else 0
        }
        
        # Test suite performance
        for test_type, results in self.results.items():
            if isinstance(results, dict) and "execution_time" in results:
                report["test_suite_performance"][test_type] = {
                    "execution_time": results["execution_time"],
                    "tests_per_second": results.get("total_tests", 0) / results["execution_time"] if results["execution_time"] > 0 else 0,
                    "success_rate": results.get("passed", 0) / results.get("total_tests", 1) if results.get("total_tests", 0) > 0 else 0
                }
        
        # Generate recommendations
        if report["overall_metrics"]["success_rate"] < 0.9:
            report["recommendations"].append("Improve test success rate - target 90%+")
        
        if report["overall_metrics"]["tests_per_second"] < 10:
            report["recommendations"].append("Optimize test execution speed - target 10+ tests/second")
        
        for test_type, perf in report["test_suite_performance"].items():
            if perf["success_rate"] < 0.8:
                report["recommendations"].append(f"Focus on improving {test_type} success rate")
        
        self.results["performance_metrics"] = report
        return report
    
    def save_results(self, filename: str = "optimized_test_results.json"):
        """Save test results to file."""
        results_file = Path(filename)
        
        # Add timestamp and summary
        self.results["timestamp"] = time.time()
        self.results["total_execution_time"] = time.time() - self.start_time
        
        # Save to file
        with open(results_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {results_file}")
        return results_file
    
    def print_summary(self) -> Any:
        """Print test execution summary."""
        print("\n" + "="*60)
        print("🎯 OPTIMIZED TEST EXECUTION SUMMARY")
        print("="*60)
        
        total_time = time.time() - self.start_time
        
        # Overall summary
        total_tests = sum(
            results.get("total_tests", 0) 
            for results in self.results.values() 
            if isinstance(results, dict)
        )
        total_passed = sum(
            results.get("passed", 0) 
            for results in self.results.values() 
            if isinstance(results, dict)
        )
        total_failed = sum(
            results.get("failed", 0) 
            for results in self.results.values() 
            if isinstance(results, dict)
        )
        
        print(f"⏱️  Total Execution Time: {total_time:.2f}s")
        print(f"🧪 Total Tests: {total_tests}")
        print(f"✅ Passed: {total_passed}")
        print(f"❌ Failed: {total_failed}")
        print(f"📊 Success Rate: {(total_passed/total_tests*100):.1f}%" if total_tests > 0 else "N/A")
        print(f"🚀 Tests/Second: {total_tests/total_time:.1f}" if total_time > 0 else "N/A")
        
        # Test suite breakdown
        print("\n📋 Test Suite Breakdown:")
        for test_type, results in self.results.items():
            if isinstance(results, dict) and "total_tests" in results:
                success_rate = results.get("passed", 0) / results.get("total_tests", 1) * 100
                exec_time = results.get("execution_time", 0)
                print(f"  {test_type.replace('_', ' ').title()}: "
                      f"{results.get('passed', 0)}/{results.get('total_tests', 0)} passed "
                      f"({success_rate:.1f}%) in {exec_time:.2f}s")
        
        # Performance recommendations
        if "performance_metrics" in self.results:
            recommendations = self.results["performance_metrics"].get("recommendations", [])
            if recommendations:
                print("\n💡 Recommendations:")
                for rec in recommendations:
                    print(f"  • {rec}")
        
        print("="*60)


async def main():
    """Main test execution function."""
    print("🚀 Starting Optimized Test Suite...")
    print("="*60)
    
    # Create test runner
    runner = OptimizedTestRunner()
    
    try:
        # Run all test suites
        await runner.run_unit_tests()
        await runner.run_integration_tests()
        await runner.run_load_tests()
        await runner.run_debug_tests()
        
        # Generate performance report
        runner.generate_performance_report()
        
        # Save results
        runner.save_results()
        
        # Print summary
        runner.print_summary()
        
        # Exit with appropriate code
        total_failed = sum(
            results.get("failed", 0) 
            for results in runner.results.values() 
            if isinstance(results, dict)
        )
        
        if total_failed > 0:
            print(f"\n⚠️  {total_failed} tests failed!")
            sys.exit(1)
        else:
            print("\n🎉 All tests passed successfully!")
            sys.exit(0)
    
    except KeyboardInterrupt:
        print("\n⏹️  Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the test suite
    asyncio.run(main()) 