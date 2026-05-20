"""
Advanced Test Runner
Enhanced test runner with coverage, HTML reports, metrics, and more
"""

import unittest
import sys
import os
import time
import logging
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import test utilities
try:
    from tests.html_report_generator import HTMLReportGenerator
    from tests.test_metrics import TestMetricsTracker
    from tests.test_profiler import TestProfiler
    from tests.test_coverage import get_coverage_report, generate_html_coverage, COVERAGE_AVAILABLE
except ImportError as e:
    print(f"⚠️  Warning: Some advanced features unavailable: {e}")

# Import all test modules
try:
    from tests.test_core import TestCoreComponents
    from tests.test_optimization import TestOptimizationEngine
    from tests.test_models import TestModelManager
    from tests.test_training import TestTrainingManager
    from tests.test_inference import TestInferenceEngine
    from tests.test_monitoring import TestMonitoringSystem
    from tests.test_integration import TestIntegration
    from tests.test_edge_cases import TestEdgeCases
    from tests.test_performance import TestPerformance
    from tests.test_security import TestSecurity
    from tests.test_compatibility import TestCompatibility
    from tests.test_regression import TestRegression
    from tests.test_validation import TestValidation
    from tests.test_benchmarks import TestBenchmarks
except ImportError as e:
    print(f"❌ Error importing test modules: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedTestRunner:
    """Advanced test runner with enhanced features"""
    
    def __init__(self, 
                 enable_coverage: bool = False,
                 enable_html_report: bool = False,
                 enable_metrics: bool = False,
                 enable_profiling: bool = False,
                 parallel: bool = False,
                 filter_tests: str = None):
        self.test_suite = unittest.TestSuite()
        self.results = {}
        self.start_time = None
        self.test_result = None
        
        # Feature flags
        self.enable_coverage = enable_coverage and COVERAGE_AVAILABLE
        self.enable_html_report = enable_html_report
        self.enable_metrics = enable_metrics
        self.enable_profiling = enable_profiling
        self.parallel = parallel
        self.filter_tests = filter_tests
        
        # Initialize tools
        if self.enable_metrics:
            self.metrics_tracker = TestMetricsTracker()
        if self.enable_profiling:
            self.profiler = TestProfiler()
        if self.enable_html_report:
            self.html_generator = HTMLReportGenerator()
        
        # Coverage setup
        if self.enable_coverage:
            try:
                import coverage
                self.cov = coverage.Coverage()
                self.cov.start()
            except Exception as e:
                logger.warning(f"Could not start coverage: {e}")
                self.enable_coverage = False
    
    def add_all_tests(self):
        """Add all test classes to the test suite"""
        logger.info("🧪 Adding all test classes to unified test suite...")
        
        loader = unittest.TestLoader()
        
        test_classes = [
            (TestCoreComponents, "core component tests"),
            (TestOptimizationEngine, "optimization tests"),
            (TestModelManager, "model management tests"),
            (TestTrainingManager, "training management tests"),
            (TestInferenceEngine, "inference engine tests"),
            (TestMonitoringSystem, "monitoring system tests"),
            (TestIntegration, "integration tests"),
            (TestEdgeCases, "edge cases and stress tests"),
            (TestPerformance, "performance and benchmark tests"),
            (TestSecurity, "security and validation tests"),
            (TestCompatibility, "compatibility tests"),
            (TestRegression, "regression tests"),
            (TestValidation, "validation tests"),
            (TestBenchmarks, "benchmark tests")
        ]
        
        for test_class, description in test_classes:
            suite = loader.loadTestsFromTestCase(test_class)
            
            # Apply filter if specified
            if self.filter_tests:
                filtered_suite = unittest.TestSuite()
                for test in suite:
                    if self.filter_tests.lower() in str(test).lower():
                        filtered_suite.addTest(test)
                suite = filtered_suite
            
            self.test_suite.addTest(suite)
            logger.info(f"✅ Added {description}")
        
        logger.info(f"📊 Total test classes added: {len(test_classes)}")
    
    def run_tests(self, verbose: bool = True):
        """Run all tests with enhanced features"""
        logger.info("🚀 Starting advanced test suite...")
        self.start_time = time.time()
        
        # Create test runner
        runner = unittest.TextTestRunner(
            verbosity=2 if verbose else 1,
            descriptions=True,
            failfast=False
        )
        
        # Run tests
        result = runner.run(self.test_suite)
        
        # Stop coverage if enabled
        if self.enable_coverage:
            try:
                self.cov.stop()
                self.cov.save()
            except Exception:
                pass
        
        # Calculate execution time
        execution_time = time.time() - self.start_time
        
        # Store detailed result
        self.test_result = result
        
        # Store results
        total_tests = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
        
        success_rate = ((total_tests - failures - errors) / total_tests * 100) if total_tests > 0 else 0.0
        tests_per_second = total_tests / execution_time if execution_time > 0 else 0.0
        
        self.results = {
            'total_tests': total_tests,
            'failures': failures,
            'errors': errors,
            'skipped': skipped,
            'success_rate': success_rate,
            'execution_time': execution_time,
            'tests_per_second': tests_per_second,
            'failures': result.failures,
            'errors': result.errors,
            'skipped': result.skipped if hasattr(result, 'skipped') else []
        }
        
        # Record metrics
        if self.enable_metrics:
            self.metrics_tracker.record_test_run(self.results)
        
        return result
    
    def generate_reports(self):
        """Generate all enabled reports"""
        reports = []
        
        # HTML Report
        if self.enable_html_report:
            try:
                html_file = self.html_generator.generate(self.results, "test_report.html")
                reports.append(f"📄 HTML report: {html_file}")
            except Exception as e:
                logger.warning(f"Could not generate HTML report: {e}")
        
        # Coverage Report
        if self.enable_coverage:
            try:
                coverage_report = get_coverage_report()
                reports.append(f"📊 Coverage report:\n{coverage_report}")
                
                success, message = generate_html_coverage()
                if success:
                    reports.append(f"📄 HTML coverage: {message}")
            except Exception as e:
                logger.warning(f"Could not generate coverage report: {e}")
        
        # Metrics Summary
        if self.enable_metrics:
            try:
                summary = self.metrics_tracker.get_summary()
                trends = self.metrics_tracker.get_trends()
                reports.append(f"📈 Metrics summary:\n{self._format_metrics(summary, trends)}")
            except Exception as e:
                logger.warning(f"Could not generate metrics: {e}")
        
        # Profiling Report
        if self.enable_profiling:
            try:
                profiler_report = self.profiler.generate_summary_report()
                reports.append(f"⏱️  Profiling report:\n{profiler_report}")
            except Exception as e:
                logger.warning(f"Could not generate profiling report: {e}")
        
        return reports
    
    def _format_metrics(self, summary: dict, trends: dict) -> str:
        """Format metrics summary"""
        lines = [
            f"Total runs: {summary.get('total_runs', 0)}",
            f"Latest success rate: {summary.get('latest_run', {}).get('success_rate', 0):.1f}%",
            f"Average success rate: {summary.get('average_success_rate', 0):.1f}%",
            f"Best success rate: {summary.get('best_success_rate', 0):.1f}%",
            f"Average execution time: {summary.get('average_execution_time', 0):.2f}s",
        ]
        if 'trends' in summary and summary['trends']:
            lines.append(f"Success rate trend: {summary['trends'].get('success_rate_trend', 'N/A')}")
        return "\n".join(lines)
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    TRUTHGPT ADVANCED TEST REPORT                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🎯 TEST SUMMARY                                                             ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  Total Tests Run:     {self.results['total_tests']:>6}                                           ║
║  Passed:              {self.results['total_tests'] - self.results['failures'] - self.results['errors']:>6}                                           ║
║  Failures:            {self.results['failures']:>6}                                           ║
║  Errors:              {self.results['errors']:>6}                                           ║
║  Skipped:             {self.results['skipped']:>6}                                           ║
║  Success Rate:        {self.results['success_rate']:>5.1f}%                                        ║
║                                                                              ║
║  ⏱️  PERFORMANCE                                                              ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  Execution Time:      {self.results['execution_time']:>6.2f}s                                        ║
║  Tests/Second:       {self.results['tests_per_second']:>6.1f}                                           ║
║                                                                              ║
║  🔧 FEATURES ENABLED                                                          ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  Coverage:            {'✅' if self.enable_coverage else '❌'}                                           ║
║  HTML Reports:        {'✅' if self.enable_html_report else '❌'}                                           ║
║  Metrics Tracking:    {'✅' if self.enable_metrics else '❌'}                                           ║
║  Profiling:           {'✅' if self.enable_profiling else '❌'}                                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        return report

def main():
    """Main test runner with CLI arguments"""
    parser = argparse.ArgumentParser(description='Advanced TruthGPT Test Runner')
    parser.add_argument('--coverage', action='store_true', help='Enable test coverage')
    parser.add_argument('--html', action='store_true', help='Generate HTML report')
    parser.add_argument('--metrics', action='store_true', help='Track test metrics')
    parser.add_argument('--profile', action='store_true', help='Profile test execution')
    parser.add_argument('--filter', type=str, help='Filter tests by name')
    parser.add_argument('--category', type=str, help='Run specific test category')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode')
    
    args = parser.parse_args()
    
    print("🧪 TruthGPT Advanced Test Runner")
    print("=" * 60)
    
    # Verify directory
    if not Path("core").exists() or not Path("tests").exists():
        print("❌ Error: core/ or tests/ directory not found")
        print(f"Current directory: {os.getcwd()}")
        sys.exit(1)
    
    # Create test runner
    test_runner = AdvancedTestRunner(
        enable_coverage=args.coverage,
        enable_html_report=args.html,
        enable_metrics=args.metrics,
        enable_profiling=args.profile,
        filter_tests=args.filter
    )
    
    # Handle category filter
    if args.category:
        # This would require category-specific implementation
        logger.info(f"Running category: {args.category}")
    
    # Add all tests
    test_runner.add_all_tests()
    
    # Run tests
    result = test_runner.run_tests(verbose=not args.quiet)
    
    # Generate reports
    print(test_runner.generate_summary_report())
    
    # Generate additional reports
    additional_reports = test_runner.generate_reports()
    if additional_reports:
        print("\n" + "=" * 60)
        print("ADDITIONAL REPORTS")
        print("=" * 60)
        for report in additional_reports:
            print(report)
    
    # Exit with appropriate code
    if result.failures or result.errors:
        logger.error("❌ Some tests failed!")
        sys.exit(1)
    else:
        logger.info("🎉 All tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()







