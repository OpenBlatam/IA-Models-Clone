#!/usr/bin/env python3
"""
Test Optimizer for HeyGen AI
============================

Advanced test optimization and parallel execution system.
"""

import asyncio
import concurrent.futures
import multiprocessing
import time
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class TestOptimizationResult:
    """Result of test optimization"""
    test_file: str
    original_duration: float
    optimized_duration: float
    improvement_percentage: float
    parallel_execution: bool
    optimizations_applied: List[str]

class TestOptimizer:
    """Advanced test optimizer for HeyGen AI"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.optimization_results: List[TestOptimizationResult] = []
        self.test_files: List[Path] = []
        self.base_dir = Path(__file__).parent
        self.test_dir = self.base_dir / "tests"
    
    def discover_test_files(self) -> List[Path]:
        """Discover all test files in the test directory"""
        print("🔍 Discovering test files...")
        
        if not self.test_dir.exists():
            print(f"  ❌ Test directory not found: {self.test_dir}")
            return []
        
        test_files = list(self.test_dir.glob("test_*.py"))
        print(f"  ✅ Found {len(test_files)} test files")
        
        for test_file in test_files:
            print(f"    - {test_file.name}")
        
        self.test_files = test_files
        return test_files
    
    def analyze_test_file(self, test_file: Path) -> Dict[str, Any]:
        """Analyze a test file for optimization opportunities"""
        print(f"🔬 Analyzing {test_file.name}...")
        
        analysis = {
            "file": str(test_file),
            "size": test_file.stat().st_size,
            "lines": 0,
            "test_functions": 0,
            "async_tests": 0,
            "fixtures": 0,
            "imports": 0,
            "optimization_opportunities": []
        }
        
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                analysis["lines"] = len(lines)
                
                # Count test functions
                test_functions = [line for line in lines if line.strip().startswith('def test_')]
                analysis["test_functions"] = len(test_functions)
                
                # Count async tests
                async_tests = [line for line in lines if 'async def test_' in line]
                analysis["async_tests"] = len(async_tests)
                
                # Count fixtures
                fixtures = [line for line in lines if '@pytest.fixture' in line]
                analysis["fixtures"] = len(fixtures)
                
                # Count imports
                imports = [line for line in lines if line.strip().startswith('import ') or line.strip().startswith('from ')]
                analysis["imports"] = len(imports)
                
                # Identify optimization opportunities
                opportunities = []
                
                if analysis["async_tests"] > 0:
                    opportunities.append("async_optimization")
                
                if analysis["test_functions"] > 10:
                    opportunities.append("parallel_execution")
                
                if analysis["fixtures"] > 5:
                    opportunities.append("fixture_optimization")
                
                if analysis["lines"] > 1000:
                    opportunities.append("code_splitting")
                
                if "time.sleep" in content:
                    opportunities.append("sleep_optimization")
                
                if "requests.get" in content or "requests.post" in content:
                    opportunities.append("network_mocking")
                
                analysis["optimization_opportunities"] = opportunities
                
        except Exception as e:
            print(f"  ⚠️ Error analyzing {test_file.name}: {e}")
            analysis["error"] = str(e)
        
        return analysis
    
    def run_test_sequentially(self, test_file: Path) -> Tuple[float, bool]:
        """Run a test file sequentially and measure duration"""
        print(f"⏱️ Running {test_file.name} sequentially...")
        
        start_time = time.time()
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"
            ], capture_output=True, text=True, timeout=300)
            
            end_time = time.time()
            duration = end_time - start_time
            success = result.returncode == 0
            
            print(f"  {'✅' if success else '❌'} Sequential: {duration:.2f}s")
            return duration, success
            
        except subprocess.TimeoutExpired:
            print(f"  ⏰ Sequential: Timeout (300s)")
            return 300.0, False
        except Exception as e:
            print(f"  ❌ Sequential: Error - {e}")
            return 0.0, False
    
    def run_test_parallel(self, test_file: Path) -> Tuple[float, bool]:
        """Run a test file with parallel execution and measure duration"""
        print(f"⚡ Running {test_file.name} in parallel...")
        
        start_time = time.time()
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short", 
                "-n", str(self.max_workers)
            ], capture_output=True, text=True, timeout=300)
            
            end_time = time.time()
            duration = end_time - start_time
            success = result.returncode == 0
            
            print(f"  {'✅' if success else '❌'} Parallel: {duration:.2f}s")
            return duration, success
            
        except subprocess.TimeoutExpired:
            print(f"  ⏰ Parallel: Timeout (300s)")
            return 300.0, False
        except Exception as e:
            print(f"  ❌ Parallel: Error - {e}")
            return 0.0, False
    
    def optimize_test_file(self, test_file: Path) -> TestOptimizationResult:
        """Optimize a single test file"""
        print(f"\n🔧 Optimizing {test_file.name}...")
        
        # Analyze the test file
        analysis = self.analyze_test_file(test_file)
        
        # Run sequential test
        seq_duration, seq_success = self.run_test_sequentially(test_file)
        
        if not seq_success:
            print(f"  ⚠️ Sequential test failed, skipping optimization")
            return TestOptimizationResult(
                test_file=str(test_file),
                original_duration=seq_duration,
                optimized_duration=seq_duration,
                improvement_percentage=0.0,
                parallel_execution=False,
                optimizations_applied=["skipped_due_to_failure"]
            )
        
        # Try parallel execution if applicable
        parallel_duration = seq_duration
        parallel_success = seq_success
        optimizations_applied = []
        
        if "parallel_execution" in analysis["optimization_opportunities"]:
            parallel_duration, parallel_success = self.run_test_parallel(test_file)
            
            if parallel_success and parallel_duration < seq_duration:
                optimizations_applied.append("parallel_execution")
            else:
                parallel_duration = seq_duration
        
        # Calculate improvement
        improvement = ((seq_duration - parallel_duration) / seq_duration) * 100 if seq_duration > 0 else 0
        
        result = TestOptimizationResult(
            test_file=str(test_file),
            original_duration=seq_duration,
            optimized_duration=parallel_duration,
            improvement_percentage=improvement,
            parallel_execution=parallel_success and parallel_duration < seq_duration,
            optimizations_applied=optimizations_applied
        )
        
        self.optimization_results.append(result)
        return result
    
    def optimize_all_tests(self) -> List[TestOptimizationResult]:
        """Optimize all test files"""
        print("🚀 Starting Test Optimization")
        print("=" * 50)
        
        # Discover test files
        test_files = self.discover_test_files()
        
        if not test_files:
            print("❌ No test files found to optimize")
            return []
        
        # Optimize each test file
        for test_file in test_files:
            try:
                self.optimize_test_file(test_file)
            except Exception as e:
                print(f"❌ Error optimizing {test_file.name}: {e}")
        
        return self.optimization_results
    
    def generate_optimization_report(self) -> str:
        """Generate optimization report"""
        report = []
        report.append("⚡ HeyGen AI Test Optimization Report")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Tests Optimized: {len(self.optimization_results)}")
        report.append(f"Max Workers: {self.max_workers}")
        report.append("")
        
        if not self.optimization_results:
            report.append("No optimization results available.")
            return "\n".join(report)
        
        # Calculate overall statistics
        total_original = sum(r.original_duration for r in self.optimization_results)
        total_optimized = sum(r.optimized_duration for r in self.optimization_results)
        overall_improvement = ((total_original - total_optimized) / total_original) * 100 if total_original > 0 else 0
        
        report.append("📊 Overall Statistics:")
        report.append(f"  Total Original Duration: {total_original:.2f}s")
        report.append(f"  Total Optimized Duration: {total_optimized:.2f}s")
        report.append(f"  Overall Improvement: {overall_improvement:.1f}%")
        report.append(f"  Time Saved: {total_original - total_optimized:.2f}s")
        report.append("")
        
        # Individual test results
        report.append("📋 Individual Test Results:")
        report.append("-" * 50)
        
        for result in self.optimization_results:
            report.append(f"📁 {Path(result.test_file).name}")
            report.append(f"  Original: {result.original_duration:.2f}s")
            report.append(f"  Optimized: {result.optimized_duration:.2f}s")
            report.append(f"  Improvement: {result.improvement_percentage:.1f}%")
            report.append(f"  Parallel: {'✅' if result.parallel_execution else '❌'}")
            if result.optimizations_applied:
                report.append(f"  Optimizations: {', '.join(result.optimizations_applied)}")
            report.append("")
        
        # Recommendations
        report.append("💡 Optimization Recommendations:")
        report.append("-" * 40)
        
        # Find best improvements
        best_improvements = sorted(self.optimization_results, key=lambda x: x.improvement_percentage, reverse=True)
        
        if best_improvements and best_improvements[0].improvement_percentage > 0:
            report.append(f"  🏆 Best Improvement: {Path(best_improvements[0].test_file).name} ({best_improvements[0].improvement_percentage:.1f}%)")
        
        # Count parallel optimizations
        parallel_count = sum(1 for r in self.optimization_results if r.parallel_execution)
        report.append(f"  ⚡ Parallel Optimizations: {parallel_count}/{len(self.optimization_results)}")
        
        # Find tests that could benefit from more optimization
        no_improvement = [r for r in self.optimization_results if r.improvement_percentage == 0]
        if no_improvement:
            report.append(f"  🔍 Tests with no improvement: {len(no_improvement)}")
            report.append("    Consider:")
            report.append("    - Adding more async tests")
            report.append("    - Optimizing fixtures")
            report.append("    - Reducing I/O operations")
        
        return "\n".join(report)
    
    def save_optimization_results(self, filename: str = "optimization_results.json"):
        """Save optimization results to JSON file"""
        results_data = []
        for result in self.optimization_results:
            results_data.append({
                "test_file": result.test_file,
                "original_duration": result.original_duration,
                "optimized_duration": result.optimized_duration,
                "improvement_percentage": result.improvement_percentage,
                "parallel_execution": result.parallel_execution,
                "optimizations_applied": result.optimizations_applied
            })
        
        results_file = self.base_dir / filename
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Optimization results saved to: {results_file}")

def main():
    """Main optimization function"""
    optimizer = TestOptimizer()
    results = optimizer.optimize_all_tests()
    
    # Generate and display report
    report = optimizer.generate_optimization_report()
    print(f"\n{report}")
    
    # Save results
    optimizer.save_optimization_results()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())





