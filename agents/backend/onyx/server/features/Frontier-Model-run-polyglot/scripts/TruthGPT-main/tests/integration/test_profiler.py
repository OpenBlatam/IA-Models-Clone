"""
Test Profiler
Identifies slow tests and performance bottlenecks
"""

import sys
import os
import time
import unittest
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from run_unified_tests import UnifiedTestRunner
except ImportError:
    print("❌ Error: Could not import UnifiedTestRunner")
    sys.exit(1)


class TestProfiler:
    """Profile test execution times"""
    
    def __init__(self):
        self.test_times = {}
        self.category_times = defaultdict(float)
        self.slow_tests = []
        
    def profile_test_run(self, test_runner: UnifiedTestRunner) -> Dict:
        """Profile a test run"""
        print("⏱️  Profiling test execution...")
        print("=" * 60)
        
        # Custom test result collector
        class ProfilingTestResult(unittest.TestResult):
            def __init__(self, profiler):
                super().__init__()
                self.profiler = profiler
                self.test_start_times = {}
            
            def startTest(self, test):
                super().startTest(test)
                self.test_start_times[test] = time.time()
            
            def stopTest(self, test):
                super().stopTest(test)
                if test in self.test_start_times:
                    elapsed = time.time() - self.test_start_times[test]
                    test_name = str(test)
                    self.profiler.test_times[test_name] = elapsed
                    
                    # Extract category from test name
                    category = test_name.split('.')[1] if '.' in test_name else 'unknown'
                    self.profiler.category_times[category] += elapsed
        
        # Run tests with profiling
        profiler_result = ProfilingTestResult(self)
        test_runner.test_suite.run(profiler_result)
        
        # Identify slow tests
        sorted_tests = sorted(self.test_times.items(), key=lambda x: x[1], reverse=True)
        self.slow_tests = sorted_tests[:20]  # Top 20 slowest
        
        return {
            'total_tests': len(self.test_times),
            'total_time': sum(self.test_times.values()),
            'average_time': sum(self.test_times.values()) / len(self.test_times) if self.test_times else 0,
            'slow_tests': self.slow_tests,
            'category_times': dict(self.category_times)
        }
    
    def generate_report(self, profile_data: Dict) -> str:
        """Generate profiling report"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("TEST PROFILING REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary
        report_lines.append("📊 SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"Total Tests:       {profile_data['total_tests']}")
        report_lines.append(f"Total Time:        {profile_data['total_time']:.2f}s")
        report_lines.append(f"Average Time:      {profile_data['average_time']:.3f}s")
        report_lines.append("")
        
        # Category breakdown
        report_lines.append("📁 TIME BY CATEGORY")
        report_lines.append("-" * 80)
        sorted_categories = sorted(
            profile_data['category_times'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for category, time_taken in sorted_categories:
            percentage = (time_taken / profile_data['total_time'] * 100) if profile_data['total_time'] > 0 else 0
            report_lines.append(f"  {category:20s} {time_taken:8.2f}s ({percentage:5.1f}%)")
        report_lines.append("")
        
        # Slow tests
        report_lines.append("🐌 SLOWEST TESTS (Top 20)")
        report_lines.append("-" * 80)
        for i, (test_name, time_taken) in enumerate(profile_data['slow_tests'], 1):
            percentage = (time_taken / profile_data['total_time'] * 100) if profile_data['total_time'] > 0 else 0
            report_lines.append(f"  {i:2d}. {time_taken:6.2f}s ({percentage:5.1f}%) - {test_name}")
        report_lines.append("")
        
        # Recommendations
        report_lines.append("💡 RECOMMENDATIONS")
        report_lines.append("-" * 80)
        
        slowest = profile_data['slow_tests'][0] if profile_data['slow_tests'] else None
        if slowest and slowest[1] > 5.0:
            report_lines.append(f"  ⚠️  Slowest test takes {slowest[1]:.2f}s - consider optimization")
        
        avg_time = profile_data['average_time']
        if avg_time > 1.0:
            report_lines.append(f"  ⚠️  Average test time is {avg_time:.3f}s - tests may be too complex")
        elif avg_time < 0.1:
            report_lines.append(f"  ✅ Average test time is {avg_time:.3f}s - tests are fast")
        
        # Check for outliers
        if profile_data['slow_tests']:
            slowest_time = profile_data['slow_tests'][0][1]
            if slowest_time > avg_time * 10:
                report_lines.append(f"  ⚠️  Significant outliers detected (slowest is {slowest_time/avg_time:.1f}x average)")
        
        return "\n".join(report_lines)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Profile test execution')
    parser.add_argument('category', nargs='?', help='Test category to profile (optional)')
    
    args = parser.parse_args()
    
    profiler = TestProfiler()
    test_runner = UnifiedTestRunner()
    
    if args.category:
        test_runner.run_specific_test_category(args.category)
        # Need to rebuild suite for profiling
        test_runner = UnifiedTestRunner()
        test_runner.add_all_tests()
    else:
        test_runner.add_all_tests()
    
    # Profile tests
    profile_data = profiler.profile_test_run(test_runner)
    
    # Generate report
    report = profiler.generate_report(profile_data)
    print(report)
    
    # Save report
    report_file = project_root / 'test_profile_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Report saved to: {report_file}")


if __name__ == "__main__":
    main()







