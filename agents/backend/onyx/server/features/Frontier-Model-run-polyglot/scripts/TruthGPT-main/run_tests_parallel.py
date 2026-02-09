"""
Parallel Test Runner
Runs tests in parallel for faster execution
"""

import unittest
import sys
import os
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

# Add the project root to the Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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

class ParallelTestRunner:
    """Runs tests in parallel"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(4, os.cpu_count() or 1)
        self.test_classes = [
            TestCoreComponents,
            TestOptimizationEngine,
            TestModelManager,
            TestTrainingManager,
            TestInferenceEngine,
            TestMonitoringSystem,
            TestIntegration,
            TestEdgeCases,
            TestPerformance,
            TestSecurity,
            TestCompatibility,
            TestRegression,
            TestValidation,
            TestBenchmarks
        ]
        self.results = {}
        self.start_time = None
    
    def run_test_class(self, test_class) -> Dict:
        """Run a single test class"""
        class_name = test_class.__name__
        logger.info(f"🧪 Running {class_name}...")
        
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(test_class)
        
        runner = unittest.TextTestRunner(
            verbosity=1,
            stream=open(os.devnull, 'w')  # Suppress output for parallel runs
        )
        
        start = time.time()
        result = runner.run(suite)
        duration = time.time() - start
        
        total = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
        passed = total - failures - errors - skipped
        
        return {
            'class_name': class_name,
            'total': total,
            'passed': passed,
            'failed': failures,
            'errors': errors,
            'skipped': skipped,
            'duration': duration,
            'result': result
        }
    
    def run_parallel(self) -> Dict:
        """Run all tests in parallel"""
        logger.info(f"🚀 Starting parallel test execution with {self.max_workers} workers...")
        self.start_time = time.time()
        
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all test classes
            future_to_class = {
                executor.submit(self.run_test_class, test_class): test_class
                for test_class in self.test_classes
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_class):
                test_class = future_to_class[future]
                try:
                    result = future.result()
                    results[result['class_name']] = result
                    logger.info(f"✅ {result['class_name']}: {result['passed']}/{result['total']} passed")
                except Exception as e:
                    logger.error(f"❌ {test_class.__name__} failed: {e}")
                    results[test_class.__name__] = {
                        'class_name': test_class.__name__,
                        'total': 0,
                        'passed': 0,
                        'failed': 0,
                        'errors': 1,
                        'skipped': 0,
                        'duration': 0,
                        'error': str(e)
                    }
        
        total_time = time.time() - self.start_time
        
        # Aggregate results
        aggregated = {
            'total_tests': sum(r['total'] for r in results.values()),
            'total_passed': sum(r['passed'] for r in results.values()),
            'total_failed': sum(r['failed'] for r in results.values()),
            'total_errors': sum(r['errors'] for r in results.values()),
            'total_skipped': sum(r['skipped'] for r in results.values()),
            'total_duration': total_time,
            'results': results
        }
        
        aggregated['success_rate'] = (
            (aggregated['total_passed'] / aggregated['total_tests'] * 100)
            if aggregated['total_tests'] > 0 else 0
        )
        
        return aggregated
    
    def generate_report(self, aggregated: Dict) -> str:
        """Generate test report"""
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    TRUTHGPT PARALLEL TEST REPORT                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🎯 TEST SUMMARY                                                             ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  Total Tests Run:     {aggregated['total_tests']:>6}                                           ║
║  Passed:              {aggregated['total_passed']:>6}                                           ║
║  Failed:              {aggregated['total_failed']:>6}                                           ║
║  Errors:              {aggregated['total_errors']:>6}                                           ║
║  Skipped:             {aggregated['total_skipped']:>6}                                           ║
║  Success Rate:        {aggregated['success_rate']:>5.1f}%                                        ║
║                                                                              ║
║  ⏱️  PERFORMANCE                                                              ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  Total Execution Time: {aggregated['total_duration']:>6.2f}s                                    ║
║  Workers Used:         {self.max_workers:>6}                                           ║
║  Tests/Second:        {(aggregated['total_tests'] / aggregated['total_duration'] if aggregated['total_duration'] > 0 else 0):>6.1f}                                           ║
║                                                                              ║
║  📊 BY TEST CLASS                                                             ║
║  ──────────────────────────────────────────────────────────────────────────  ║
"""
        
        for class_name, result in sorted(aggregated['results'].items()):
            status = "✅" if result['failed'] == 0 and result['errors'] == 0 else "❌"
            report += f"║  {status} {class_name:50} {result['passed']:>3}/{result['total']:<3} ({result['duration']:>5.2f}s)  ║\n"
        
        report += """║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        
        return report

def main():
    """Main function"""
    print("🧪 TruthGPT Parallel Test Runner")
    print("=" * 60)
    
    # Verify we're in the right directory
    if not Path("core").exists() or not Path("tests").exists():
        print("❌ Error: core/ or tests/ directory not found")
        print(f"Current directory: {os.getcwd()}")
        print("Please run from the TruthGPT-main directory")
        sys.exit(1)
    
    # Check for max_workers argument
    max_workers = None
    if len(sys.argv) > 1:
        try:
            max_workers = int(sys.argv[1])
        except ValueError:
            print(f"⚠️  Invalid max_workers value: {sys.argv[1]}, using default")
    
    # Create and run parallel test runner
    runner = ParallelTestRunner(max_workers=max_workers)
    aggregated = runner.run_parallel()
    
    # Generate and display report
    report = runner.generate_report(aggregated)
    print(report)
    
    # Exit with appropriate code
    if aggregated['total_failed'] > 0 or aggregated['total_errors'] > 0:
        logger.error("❌ Some tests failed!")
        sys.exit(1)
    else:
        logger.info("🎉 All tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()







