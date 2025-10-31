#!/usr/bin/env python3
"""
Advanced Test Runner for HeyGen AI
==================================

Comprehensive test runner with advanced features including benchmarking,
optimization, coverage analysis, and quality gates.
"""

import sys
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

class AdvancedTestRunner:
    """Advanced test runner with comprehensive testing capabilities"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.results = {
            "start_time": None,
            "end_time": None,
            "duration": 0,
            "test_results": {},
            "benchmark_results": {},
            "coverage_results": {},
            "quality_gate_results": {},
            "optimization_results": {}
        }
    
    def run_basic_tests(self) -> Dict[str, Any]:
        """Run basic test suite"""
        print("🧪 Running Basic Test Suite...")
        
        try:
            from run_tests import TestRunner
            runner = TestRunner()
            success = runner.run_pytest_tests()
            
            return {
                "success": success,
                "duration": runner.results.get("duration", 0),
                "status": "completed"
            }
        except Exception as e:
            print(f"  ❌ Error running basic tests: {e}")
            return {
                "success": False,
                "duration": 0,
                "status": "failed",
                "error": str(e)
            }
    
    def run_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks"""
        print("🚀 Running Performance Benchmarks...")
        
        try:
            from test_benchmark import PerformanceBenchmark
            benchmark = PerformanceBenchmark()
            benchmark.run_all_benchmarks()
            
            return {
                "success": True,
                "duration": sum(r.duration for r in benchmark.results),
                "benchmarks_run": len(benchmark.results),
                "status": "completed"
            }
        except Exception as e:
            print(f"  ❌ Error running benchmarks: {e}")
            return {
                "success": False,
                "duration": 0,
                "status": "failed",
                "error": str(e)
            }
    
    def run_coverage_analysis(self) -> Dict[str, Any]:
        """Run coverage analysis"""
        print("📊 Running Coverage Analysis...")
        
        try:
            from test_coverage_analyzer import CoverageAnalyzer
            analyzer = CoverageAnalyzer()
            report = analyzer.run_coverage_analysis()
            
            return {
                "success": True,
                "duration": report.test_duration,
                "total_coverage": report.total_percentage,
                "modules_analyzed": len(report.modules),
                "status": "completed"
            }
        except Exception as e:
            print(f"  ❌ Error running coverage analysis: {e}")
            return {
                "success": False,
                "duration": 0,
                "status": "failed",
                "error": str(e)
            }
    
    def run_quality_gate(self) -> Dict[str, Any]:
        """Run quality gate evaluation"""
        print("🚪 Running Quality Gate...")
        
        try:
            from test_quality_gate import QualityGate
            quality_gate = QualityGate()
            result = quality_gate.run_quality_gate()
            
            return {
                "success": result.overall_status.value not in ["failed", "poor"],
                "duration": 0,  # Quality gate doesn't track duration
                "overall_score": result.overall_score,
                "gates_passed": result.passed_gates,
                "total_gates": result.total_gates,
                "status": "completed"
            }
        except Exception as e:
            print(f"  ❌ Error running quality gate: {e}")
            return {
                "success": False,
                "duration": 0,
                "status": "failed",
                "error": str(e)
            }
    
    def run_optimization(self) -> Dict[str, Any]:
        """Run test optimization"""
        print("⚡ Running Test Optimization...")
        
        try:
            from test_optimizer import TestOptimizer
            optimizer = TestOptimizer()
            results = optimizer.optimize_all_tests()
            
            total_improvement = sum(r.improvement_percentage for r in results) / len(results) if results else 0
            
            return {
                "success": True,
                "duration": 0,  # Optimization doesn't track duration
                "tests_optimized": len(results),
                "average_improvement": total_improvement,
                "status": "completed"
            }
        except Exception as e:
            print(f"  ❌ Error running optimization: {e}")
            return {
                "success": False,
                "duration": 0,
                "status": "failed",
                "error": str(e)
            }
    
    def run_health_check(self) -> Dict[str, Any]:
        """Run health check"""
        print("🏥 Running Health Check...")
        
        try:
            from test_health_check import TestHealthChecker
            checker = TestHealthChecker()
            results = checker.run_full_health_check()
            
            return {
                "success": results["overall_health"] in ["excellent", "good"],
                "duration": results.get("timing", {}).get("duration", 0),
                "health_status": results["overall_health"],
                "status": "completed"
            }
        except Exception as e:
            print(f"  ❌ Error running health check: {e}")
            return {
                "success": False,
                "duration": 0,
                "status": "failed",
                "error": str(e)
            }
    
    def run_ci_tests(self) -> Dict[str, Any]:
        """Run CI test suite"""
        print("🔄 Running CI Test Suite...")
        
        try:
            from ci_test_runner import CITestRunner
            runner = CITestRunner(verbose=True, coverage=True)
            
            if not runner.check_environment():
                return {
                    "success": False,
                    "duration": 0,
                    "status": "failed",
                    "error": "Environment check failed"
                }
            
            results = runner.run_all_tests()
            
            return {
                "success": results["exit_code"] == 0,
                "duration": results["duration"],
                "status": "completed"
            }
        except Exception as e:
            print(f"  ❌ Error running CI tests: {e}")
            return {
                "success": False,
                "duration": 0,
                "status": "failed",
                "error": str(e)
            }
    
    def run_comprehensive_suite(self, include_benchmarks: bool = True, 
                               include_optimization: bool = True) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        print("🚀 Starting Comprehensive Test Suite")
        print("=" * 60)
        
        start_time = time.time()
        self.results["start_time"] = start_time
        
        # Run all test components
        self.results["test_results"]["health_check"] = self.run_health_check()
        self.results["test_results"]["basic_tests"] = self.run_basic_tests()
        self.results["test_results"]["ci_tests"] = self.run_ci_tests()
        self.results["test_results"]["coverage_analysis"] = self.run_coverage_analysis()
        self.results["test_results"]["quality_gate"] = self.run_quality_gate()
        
        if include_benchmarks:
            self.results["benchmark_results"] = self.run_benchmarks()
        
        if include_optimization:
            self.results["optimization_results"] = self.run_optimization()
        
        end_time = time.time()
        self.results["end_time"] = end_time
        self.results["duration"] = end_time - start_time
        
        return self.results
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive test report"""
        report = []
        report.append("🎯 HeyGen AI Comprehensive Test Report")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Duration: {self.results['duration']:.2f} seconds")
        report.append("")
        
        # Test results summary
        report.append("📊 Test Results Summary:")
        report.append("-" * 40)
        
        for test_name, result in self.results["test_results"].items():
            status = "✅ PASSED" if result["success"] else "❌ FAILED"
            duration = f"{result.get('duration', 0):.2f}s"
            report.append(f"  {test_name.replace('_', ' ').title()}: {status} ({duration})")
        
        # Benchmark results
        if self.results.get("benchmark_results"):
            benchmark = self.results["benchmark_results"]
            report.append(f"\n🚀 Benchmark Results:")
            report.append(f"  Benchmarks Run: {benchmark.get('benchmarks_run', 0)}")
            report.append(f"  Duration: {benchmark.get('duration', 0):.2f}s")
        
        # Coverage results
        if self.results["test_results"].get("coverage_analysis"):
            coverage = self.results["test_results"]["coverage_analysis"]
            report.append(f"\n📊 Coverage Results:")
            report.append(f"  Total Coverage: {coverage.get('total_coverage', 0):.1f}%")
            report.append(f"  Modules Analyzed: {coverage.get('modules_analyzed', 0)}")
        
        # Quality gate results
        if self.results["test_results"].get("quality_gate"):
            quality = self.results["test_results"]["quality_gate"]
            report.append(f"\n🚪 Quality Gate Results:")
            report.append(f"  Overall Score: {quality.get('overall_score', 0):.1f}/100")
            report.append(f"  Gates Passed: {quality.get('gates_passed', 0)}/{quality.get('total_gates', 0)}")
        
        # Optimization results
        if self.results.get("optimization_results"):
            optimization = self.results["optimization_results"]
            report.append(f"\n⚡ Optimization Results:")
            report.append(f"  Tests Optimized: {optimization.get('tests_optimized', 0)}")
            report.append(f"  Average Improvement: {optimization.get('average_improvement', 0):.1f}%")
        
        # Overall status
        all_tests_passed = all(
            result["success"] for result in self.results["test_results"].values()
        )
        
        report.append(f"\n🎯 Overall Status: {'✅ SUCCESS' if all_tests_passed else '❌ FAILURE'}")
        
        return "\n".join(report)
    
    def save_comprehensive_results(self, filename: str = "comprehensive_test_results.json"):
        """Save comprehensive test results to JSON file"""
        # Convert datetime objects to strings for JSON serialization
        json_results = self.results.copy()
        if json_results["start_time"]:
            json_results["start_time"] = datetime.fromtimestamp(json_results["start_time"]).isoformat()
        if json_results["end_time"]:
            json_results["end_time"] = datetime.fromtimestamp(json_results["end_time"]).isoformat()
        
        results_file = self.base_dir / filename
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Comprehensive test results saved to: {results_file}")

def main():
    """Main advanced test runner function"""
    parser = argparse.ArgumentParser(description="Advanced Test Runner for HeyGen AI")
    parser.add_argument("--benchmarks", action="store_true", help="Include performance benchmarks")
    parser.add_argument("--optimization", action="store_true", help="Include test optimization")
    parser.add_argument("--quick", action="store_true", help="Run quick test suite (no benchmarks/optimization)")
    parser.add_argument("--component", choices=["basic", "benchmark", "coverage", "quality", "optimization", "health", "ci"], 
                       help="Run specific component only")
    
    args = parser.parse_args()
    
    runner = AdvancedTestRunner()
    
    if args.component:
        # Run specific component
        if args.component == "basic":
            result = runner.run_basic_tests()
        elif args.component == "benchmark":
            result = runner.run_benchmarks()
        elif args.component == "coverage":
            result = runner.run_coverage_analysis()
        elif args.component == "quality":
            result = runner.run_quality_gate()
        elif args.component == "optimization":
            result = runner.run_optimization()
        elif args.component == "health":
            result = runner.run_health_check()
        elif args.component == "ci":
            result = runner.run_ci_tests()
        
        print(f"\nComponent Result: {'✅ SUCCESS' if result['success'] else '❌ FAILURE'}")
        return 0 if result['success'] else 1
    else:
        # Run comprehensive suite
        include_benchmarks = args.benchmarks or not args.quick
        include_optimization = args.optimization or not args.quick
        
        results = runner.run_comprehensive_suite(
            include_benchmarks=include_benchmarks,
            include_optimization=include_optimization
        )
        
        # Generate and display report
        report = runner.generate_comprehensive_report()
        print(f"\n{report}")
        
        # Save results
        runner.save_comprehensive_results()
        
        # Return appropriate exit code
        all_tests_passed = all(
            result["success"] for result in results["test_results"].values()
        )
        
        return 0 if all_tests_passed else 1

if __name__ == "__main__":
    sys.exit(main())





