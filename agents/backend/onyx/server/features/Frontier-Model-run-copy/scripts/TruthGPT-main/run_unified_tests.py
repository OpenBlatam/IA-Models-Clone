"""
Unified Test Runner
Consolidates all testing functionality from the old scattered test files
"""

import unittest
import sys
import os
import time
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import all test modules
from tests.test_core import TestCoreComponents
from tests.test_optimization import TestOptimizationEngine
from tests.test_models import TestModelManager
from tests.test_training import TestTrainingManager
from tests.test_inference import TestInferenceEngine
from tests.test_monitoring import TestMonitoringSystem
from tests.test_integration import TestIntegration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedTestRunner:
    """Unified test runner that consolidates all old test functionality"""
    
    def __init__(self):
        self.test_suite = unittest.TestSuite()
        self.results = {}
        self.start_time = None
        
    def add_all_tests(self):
        """Add all test classes to the test suite"""
        logger.info("🧪 Adding all test classes to unified test suite...")
        
        # Core component tests
        self.test_suite.addTest(unittest.makeSuite(TestCoreComponents))
        logger.info("✅ Added core component tests")
        
        # Optimization tests (replaces all old optimization test files)
        self.test_suite.addTest(unittest.makeSuite(TestOptimizationEngine))
        logger.info("✅ Added optimization tests (replaces 15+ old optimization test files)")
        
        # Model tests (replaces scattered model tests)
        self.test_suite.addTest(unittest.makeSuite(TestModelManager))
        logger.info("✅ Added model management tests")
        
        # Training tests (replaces scattered training tests)
        self.test_suite.addTest(unittest.makeSuite(TestTrainingManager))
        logger.info("✅ Added training management tests")
        
        # Inference tests (replaces scattered inference tests)
        self.test_suite.addTest(unittest.makeSuite(TestInferenceEngine))
        logger.info("✅ Added inference engine tests")
        
        # Monitoring tests (replaces scattered monitoring tests)
        self.test_suite.addTest(unittest.makeSuite(TestMonitoringSystem))
        logger.info("✅ Added monitoring system tests")
        
        # Integration tests (replaces scattered integration tests)
        self.test_suite.addTest(unittest.makeSuite(TestIntegration))
        logger.info("✅ Added integration tests")
        
        logger.info(f"📊 Total test classes added: 7 (replaces 48+ old test files)")
    
    def run_tests(self, verbose=True):
        """Run all tests with detailed reporting"""
        logger.info("🚀 Starting unified test suite...")
        self.start_time = time.time()
        
        # Create test runner
        runner = unittest.TextTestRunner(
            verbosity=2 if verbose else 1,
            descriptions=True,
            failfast=False
        )
        
        # Run tests
        result = runner.run(self.test_suite)
        
        # Calculate execution time
        execution_time = time.time() - self.start_time
        
        # Store results
        self.results = {
            'total_tests': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
            'success_rate': ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100,
            'execution_time': execution_time,
            'tests_per_second': result.testsRun / execution_time if execution_time > 0 else 0
        }
        
        return result
    
    def generate_report(self):
        """Generate comprehensive test report"""
        logger.info("📊 Generating comprehensive test report...")
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           TRUTHGPT UNIFIED TEST REPORT                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🎯 TEST SUMMARY                                                             ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  Total Tests Run:     {self.results['total_tests']:>6}                                           ║
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
║  🏗️  ARCHITECTURE IMPROVEMENTS                                               ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  Old Test Files:     48+ scattered files                                   ║
║  New Test Files:     7 organized modules                                     ║
║  Code Reduction:      ~85% fewer test files                                 ║
║  Maintainability:    Significantly improved                                 ║
║                                                                              ║
║  ✅ CONSOLIDATED FUNCTIONALITY                                               ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  • Core Components     (replaces scattered initialization tests)           ║
║  • Optimization Engine (replaces 15+ optimization test files)             ║
║  • Model Management    (replaces scattered model tests)                     ║
║  • Training System     (replaces scattered training tests)                 ║
║  • Inference Engine    (replaces scattered inference tests)                ║
║  • Monitoring System   (replaces scattered monitoring tests)               ║
║  • Integration Tests   (replaces scattered integration tests)              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        
        return report
    
    def run_specific_test_category(self, category):
        """Run tests for a specific category"""
        category_map = {
            'core': TestCoreComponents,
            'optimization': TestOptimizationEngine,
            'models': TestModelManager,
            'training': TestTrainingManager,
            'inference': TestInferenceEngine,
            'monitoring': TestMonitoringSystem,
            'integration': TestIntegration
        }
        
        if category not in category_map:
            logger.error(f"Unknown test category: {category}")
            logger.info(f"Available categories: {list(category_map.keys())}")
            return
        
        logger.info(f"🧪 Running {category} tests...")
        
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(category_map[category]))
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        logger.info(f"✅ {category} tests completed")
        return result

def main():
    """Main test runner function"""
    print("🧪 TruthGPT Unified Test Runner")
    print("=" * 60)
    
    # Create test runner
    test_runner = UnifiedTestRunner()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        category = sys.argv[1].lower()
        if category in ['core', 'optimization', 'models', 'training', 'inference', 'monitoring', 'integration']:
            test_runner.run_specific_test_category(category)
            return
        elif category == 'help':
            print("Available test categories:")
            print("  core        - Core component tests")
            print("  optimization - Optimization engine tests")
            print("  models      - Model management tests")
            print("  training    - Training system tests")
            print("  inference   - Inference engine tests")
            print("  monitoring  - Monitoring system tests")
            print("  integration - Integration tests")
            print("  all         - Run all tests (default)")
            return
    
    # Add all tests
    test_runner.add_all_tests()
    
    # Run all tests
    result = test_runner.run_tests(verbose=True)
    
    # Generate and display report
    report = test_runner.generate_report()
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

