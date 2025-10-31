#!/usr/bin/env python3
"""
Advanced test runner for HeyGen AI system.
Integrates all advanced testing capabilities including parallel execution,
benchmarking, monitoring, and CI/CD integration.
"""

import sys
import time
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent
sys.path.insert(0, str(parent_dir))

# Import advanced testing modules
try:
    from tests.advanced.parallel_testing import LoadBalancedTestRunner, ParallelTestExecutor
    from tests.advanced.benchmark_suite import AdvancedBenchmarkSuite, BenchmarkConfig
    from tests.advanced.test_dashboard import TestDashboard
    from tests.advanced.ci_cd_integration import CICDManager, CIConfig
except ImportError as e:
    print(f"⚠️  Warning: Some advanced modules not available: {e}")
    print("   Running in basic mode...")

class AdvancedTestRunner:
    """Main advanced test runner integrating all capabilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.start_time = None
        self.end_time = None
        self.results = {}
        
        # Initialize components
        self.parallel_runner = None
        self.benchmark_suite = None
        self.dashboard = None
        self.cicd_manager = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all testing components."""
        try:
            # Initialize parallel test runner
            self.parallel_runner = LoadBalancedTestRunner(max_workers=4)
            
            # Initialize benchmark suite
            benchmark_config = BenchmarkConfig(
                iterations=50,
                warmup_iterations=5,
                memory_profiling=True,
                cpu_profiling=True,
                generate_plots=True
            )
            self.benchmark_suite = AdvancedBenchmarkSuite(benchmark_config)
            
            # Initialize test dashboard
            self.dashboard = TestDashboard()
            
            # Initialize CI/CD manager
            cicd_config = CIConfig(
                project_name="heygen-ai-advanced",
                python_version="3.11"
            )
            self.cicd_manager = CICDManager(cicd_config)
            
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize all components: {e}")
    
    def run_comprehensive_tests(self, test_types: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive test suite with all advanced features."""
        if test_types is None:
            test_types = ["basic", "parallel", "benchmark", "monitoring"]
        
        print("🚀 Starting Advanced Test Suite Execution")
        print("=" * 60)
        print(f"📋 Test Types: {', '.join(test_types)}")
        print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        self.start_time = time.time()
        
        # Run different types of tests
        if "basic" in test_types:
            self._run_basic_tests()
        
        if "parallel" in test_types and self.parallel_runner:
            self._run_parallel_tests()
        
        if "benchmark" in test_types and self.benchmark_suite:
            self._run_benchmark_tests()
        
        if "monitoring" in test_types and self.dashboard:
            self._run_monitoring_tests()
        
        self.end_time = time.time()
        
        # Generate comprehensive report
        report = self._generate_comprehensive_report()
        self._print_final_summary(report)
        
        return report
    
    def _run_basic_tests(self):
        """Run basic test suite."""
        print("\n📋 Running Basic Tests...")
        print("-" * 40)
        
        try:
            import subprocess
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "tests/test_refactored_simple.py", 
                "-v", "--tb=short"
            ], capture_output=True, text=True, timeout=300)
            
            self.results["basic_tests"] = {
                "status": "passed" if result.returncode == 0 else "failed",
                "duration": time.time() - self.start_time,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
            print(f"✅ Basic tests completed: {self.results['basic_tests']['status']}")
            
        except Exception as e:
            self.results["basic_tests"] = {
                "status": "error",
                "duration": 0,
                "error": str(e)
            }
            print(f"❌ Basic tests failed: {e}")
    
    def _run_parallel_tests(self):
        """Run parallel test suite."""
        print("\n🔄 Running Parallel Tests...")
        print("-" * 40)
        
        try:
            # Add some sample test functions
            def sample_test_1():
                time.sleep(0.1)
                return "test_1_result"
            
            def sample_test_2():
                time.sleep(0.2)
                return "test_2_result"
            
            def sample_test_3():
                time.sleep(0.15)
                return "test_3_result"
            
            # Add test suites
            self.parallel_runner.add_test_suite("Sample Tests 1", [
                sample_test_1, sample_test_2, sample_test_3
            ], priority=1)
            
            self.parallel_runner.add_test_suite("Sample Tests 2", [
                sample_test_1, sample_test_2, sample_test_3
            ], priority=2)
            
            # Run parallel tests
            parallel_results = self.parallel_runner.run_all_suites()
            
            self.results["parallel_tests"] = {
                "status": "completed",
                "duration": time.time() - self.start_time,
                "results": parallel_results
            }
            
            print(f"✅ Parallel tests completed")
            
        except Exception as e:
            self.results["parallel_tests"] = {
                "status": "error",
                "duration": 0,
                "error": str(e)
            }
            print(f"❌ Parallel tests failed: {e}")
    
    def _run_benchmark_tests(self):
        """Run benchmark test suite."""
        print("\n⚡ Running Benchmark Tests...")
        print("-" * 40)
        
        try:
            # Register benchmark functions
            def cpu_intensive_task(n: int = 100000):
                result = 0
                for i in range(n):
                    result += i ** 2
                return result
            
            def memory_intensive_task(size: int = 100000):
                data = [i for i in range(size)]
                return sum(data)
            
            def io_intensive_task():
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                    f.write("test data" * 1000)
                    temp_file = f.name
                
                with open(temp_file, 'r') as f:
                    content = f.read()
                
                Path(temp_file).unlink()
                return len(content)
            
            # Register benchmarks
            self.benchmark_suite.register_benchmark("CPU Intensive", cpu_intensive_task)
            self.benchmark_suite.register_benchmark("Memory Intensive", memory_intensive_task)
            self.benchmark_suite.register_benchmark("IO Intensive", io_intensive_task)
            
            # Run benchmarks
            benchmark_results = self.benchmark_suite.run_all_benchmarks()
            
            self.results["benchmark_tests"] = {
                "status": "completed",
                "duration": time.time() - self.start_time,
                "results": benchmark_results
            }
            
            print(f"✅ Benchmark tests completed")
            
        except Exception as e:
            self.results["benchmark_tests"] = {
                "status": "error",
                "duration": 0,
                "error": str(e)
            }
            print(f"❌ Benchmark tests failed: {e}")
    
    def _run_monitoring_tests(self):
        """Run monitoring and dashboard tests."""
        print("\n📊 Running Monitoring Tests...")
        print("-" * 40)
        
        try:
            # Start dashboard
            self.dashboard.start_dashboard()
            
            # Simulate some test events
            test_names = ["monitor_test_1", "monitor_test_2", "monitor_test_3"]
            
            for i in range(5):
                test_name = test_names[i % len(test_names)]
                
                # Log test start
                self.dashboard.log_test_start(test_name, {"iteration": i})
                
                # Simulate test execution
                time.sleep(0.1)
                
                # Log test completion
                if i % 3 == 0:
                    self.dashboard.log_test_failure(test_name, 0.1, f"Test failed at iteration {i}")
                else:
                    self.dashboard.log_test_completion(test_name, 0.1, f"Result {i}")
            
            # Get dashboard data
            dashboard_data = self.dashboard.get_dashboard_data()
            
            # Stop dashboard
            self.dashboard.stop_dashboard()
            
            self.results["monitoring_tests"] = {
                "status": "completed",
                "duration": time.time() - self.start_time,
                "dashboard_data": dashboard_data
            }
            
            print(f"✅ Monitoring tests completed")
            
        except Exception as e:
            self.results["monitoring_tests"] = {
                "status": "error",
                "duration": 0,
                "error": str(e)
            }
            print(f"❌ Monitoring tests failed: {e}")
    
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        # Calculate summary statistics
        total_tests = len(self.results)
        completed_tests = sum(1 for r in self.results.values() if r.get("status") == "completed")
        failed_tests = sum(1 for r in self.results.values() if r.get("status") in ["failed", "error"])
        
        return {
            "execution_summary": {
                "total_duration": total_duration,
                "start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
                "test_runner": "Advanced Test Suite v3.0"
            },
            "test_statistics": {
                "total_test_types": total_tests,
                "completed_test_types": completed_tests,
                "failed_test_types": failed_tests,
                "success_rate": (completed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            "test_results": self.results,
            "capabilities": {
                "parallel_testing": self.parallel_runner is not None,
                "benchmarking": self.benchmark_suite is not None,
                "monitoring": self.dashboard is not None,
                "ci_cd_integration": self.cicd_manager is not None
            }
        }
    
    def _print_final_summary(self, report: Dict[str, Any]):
        """Print final execution summary."""
        print("\n" + "=" * 60)
        print("📊 ADVANCED TEST SUITE EXECUTION SUMMARY")
        print("=" * 60)
        
        # Execution info
        exec_summary = report["execution_summary"]
        print(f"⏱️  Total Duration: {exec_summary['total_duration']:.2f} seconds")
        print(f"🚀 Test Runner: {exec_summary['test_runner']}")
        
        # Test statistics
        stats = report["test_statistics"]
        print(f"\n📈 Test Statistics:")
        print(f"   Test Types: {stats['completed_test_types']}/{stats['total_test_types']} completed ({stats['success_rate']:.1f}%)")
        
        # Capabilities
        capabilities = report["capabilities"]
        print(f"\n🔧 Available Capabilities:")
        for capability, available in capabilities.items():
            status_icon = "✅" if available else "❌"
            print(f"   {status_icon} {capability.replace('_', ' ').title()}")
        
        # Test results
        print(f"\n📋 Test Results:")
        for test_type, result in report["test_results"].items():
            status_icon = "✅" if result.get("status") == "completed" else "❌"
            duration = result.get("duration", 0)
            print(f"   {status_icon} {test_type.replace('_', ' ').title()}: {result.get('status', 'unknown')} ({duration:.2f}s)")
        
        # Overall result
        overall_success = stats["success_rate"] == 100
        result_icon = "🎉" if overall_success else "⚠️"
        result_text = "ALL TESTS COMPLETED" if overall_success else "SOME TESTS FAILED"
        
        print(f"\n{result_icon} OVERALL RESULT: {result_text}")
        print("=" * 60)
    
    def setup_cicd_pipeline(self):
        """Set up CI/CD pipeline."""
        if not self.cicd_manager:
            print("❌ CI/CD manager not available")
            return
        
        print("🚀 Setting up CI/CD Pipeline...")
        self.cicd_manager.setup_complete_cicd()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Advanced Test Runner for HeyGen AI")
    parser.add_argument("--test-types", nargs="+", 
                       choices=["basic", "parallel", "benchmark", "monitoring"],
                       default=["basic", "parallel", "benchmark", "monitoring"],
                       help="Types of tests to run")
    parser.add_argument("--setup-cicd", action="store_true",
                       help="Set up CI/CD pipeline")
    parser.add_argument("--config", type=str,
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    print("🔄 HeyGen AI - Advanced Test Suite Runner")
    print("Version: 3.0 | Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print()
    
    try:
        # Load configuration if provided
        config = {}
        if args.config and Path(args.config).exists():
            with open(args.config, 'r') as f:
                config = json.load(f)
        
        # Create advanced test runner
        runner = AdvancedTestRunner(config)
        
        if args.setup_cicd:
            # Set up CI/CD pipeline
            runner.setup_cicd_pipeline()
        else:
            # Run comprehensive tests
            report = runner.run_comprehensive_tests(args.test_types)
            
            # Save report
            report_dir = Path("test_reports")
            report_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = report_dir / f"advanced_test_report_{timestamp}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"\n💾 Report saved to: {report_file}")
            
            # Exit with appropriate code
            overall_success = report["test_statistics"]["success_rate"] == 100
            sys.exit(0 if overall_success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error running tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
