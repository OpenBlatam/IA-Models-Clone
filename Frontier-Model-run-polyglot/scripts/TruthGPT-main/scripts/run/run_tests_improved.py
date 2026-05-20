"""
Improved Test Runner
Enhanced test runner with more options and better reporting
"""

import unittest
import sys
import os
import time
import logging
import argparse
from pathlib import Path
from run_unified_tests import UnifiedTestRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='TruthGPT Unified Test Runner')
    parser.add_argument('category', nargs='?', default='all',
                       help='Test category to run (core, optimization, models, training, inference, monitoring, integration, edge, performance, all)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet output (minimal)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List all available test categories')
    parser.add_argument('--save-report', action='store_true',
                       help='Save detailed report to file')
    parser.add_argument('--failfast', action='store_true',
                       help='Stop on first failure')
    parser.add_argument('--coverage', action='store_true',
                       help='Run with coverage (requires pytest-cov)')
    
    args = parser.parse_args()
    
    # Verify we're in the right directory
    if not Path("core").exists() or not Path("tests").exists():
        print("❌ Error: core/ or tests/ directory not found")
        print(f"Current directory: {os.getcwd()}")
        print("Please run from the TruthGPT-main directory")
        sys.exit(1)
    
    # List categories
    if args.list:
        print("Available test categories:")
        categories = [
            ('core', 'Core component tests'),
            ('optimization', 'Optimization engine tests'),
            ('models', 'Model management tests'),
            ('training', 'Training system tests'),
            ('inference', 'Inference engine tests'),
            ('monitoring', 'Monitoring system tests'),
            ('integration', 'Integration tests'),
            ('edge', 'Edge cases and stress tests'),
            ('performance', 'Performance and benchmark tests'),
            ('all', 'Run all tests (default)')
        ]
        for cat, desc in categories:
            print(f"  {cat:15} - {desc}")
        return
    
    # Create test runner
    test_runner = UnifiedTestRunner()
    
    # Set verbosity
    verbose = args.verbose and not args.quiet
    
    # Run specific category or all
    if args.category.lower() == 'all':
        test_runner.add_all_tests()
        result = test_runner.run_tests(verbose=verbose)
    else:
        result = test_runner.run_specific_test_category(args.category.lower())
        if result is None:
            print(f"❌ Unknown category: {args.category}")
            print("Use --list to see available categories")
            sys.exit(1)
    
    # Generate report
    report = test_runner.generate_report()
    print(report)
    
    # Generate detailed report if needed
    if result and (result.failures or result.errors):
        detailed_report = test_runner.generate_detailed_report(result)
        print(detailed_report)
        
        if args.save_report:
            report_file = f"test_report_{int(time.time())}.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
                f.write(detailed_report)
            logger.info(f"📄 Detailed report saved to {report_file}")
    
    # Exit with appropriate code
    if result and (result.failures or result.errors):
        logger.error("❌ Some tests failed!")
        sys.exit(1)
    else:
        logger.info("🎉 All tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()








