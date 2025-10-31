#!/usr/bin/env python3
"""
Comprehensive test runner for the ADS system
"""

import sys
import os
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any

class TestRunner:
    """Comprehensive test runner for the ADS system"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
        # Test categories
        self.test_categories = {
            'unit': {
                'description': 'Unit Tests',
                'tests': [
                    'tests/unit/test_optimization_system.py',
                    'tests/unit/test_campaign_management.py',
                    'tests/unit/test_value_objects.py'
                ]
            },
            'integration': {
                'description': 'Integration Tests',
                'tests': [
                    'tests/integration/test_system_integration.py'
                ]
            },
            'basic': {
                'description': 'Basic Functionality Tests',
                'tests': [
                    'test_basic.py'
                ]
            }
        }
    
    def run_tests(self, categories: List[str] = None, verbose: bool = False, parallel: bool = False):
        """Run tests for specified categories"""
        if categories is None:
            categories = list(self.test_categories.keys())
        
        self.start_time = time.time()
        print("🧪 ADS System Test Runner")
        print("=" * 60)
        print(f"Starting tests at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Categories: {', '.join(categories)}")
        print(f"Verbose: {verbose}")
        print(f"Parallel: {parallel}")
        print("=" * 60)
        
        # Check if pytest is available
        if not self._check_pytest():
            print("❌ pytest not available. Installing...")
            self._install_pytest()
        
        # Run tests for each category
        for category in categories:
            if category in self.test_categories:
                self._run_category_tests(category, verbose, parallel)
            else:
                print(f"⚠️  Unknown test category: {category}")
        
        # Print summary
        self._print_summary()
        
        return self.failed_tests == 0
    
    def _check_pytest(self) -> bool:
        """Check if pytest is available"""
        try:
            import pytest
            return True
        except ImportError:
            return False
    
    def _install_pytest(self):
        """Install pytest if not available"""
        try:
            print("📦 Installing pytest...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'pytest'], 
                         check=True, capture_output=True)
            print("✅ pytest installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install pytest: {e}")
            sys.exit(1)
    
    def _run_category_tests(self, category: str, verbose: bool, parallel: bool):
        """Run tests for a specific category"""
        category_info = self.test_categories[category]
        print(f"\n🔍 Running {category_info['description']}...")
        print("-" * 40)
        
        category_start_time = time.time()
        category_passed = 0
        category_failed = 0
        
        for test_file in category_info['tests']:
            if os.path.exists(test_file):
                result = self._run_single_test(test_file, verbose, parallel)
                if result['success']:
                    category_passed += 1
                else:
                    category_failed += 1
            else:
                print(f"⚠️  Test file not found: {test_file}")
        
        category_time = time.time() - category_start_time
        
        print(f"\n📊 {category_info['description']} Results:")
        print(f"   ✅ Passed: {category_passed}")
        print(f"   ❌ Failed: {category_failed}")
        print(f"   ⏱️  Time: {category_time:.2f}s")
        
        # Store results
        self.test_results[category] = {
            'passed': category_passed,
            'failed': category_failed,
            'time': category_time
        }
        
        self.total_tests += category_passed + category_failed
        self.passed_tests += category_passed
        self.failed_tests += category_failed
    
    def _run_single_test(self, test_file: str, verbose: bool, parallel: bool) -> Dict[str, Any]:
        """Run a single test file"""
        print(f"  🧪 Running: {test_file}")
        
        # Build pytest command
        cmd = [sys.executable, '-m', 'pytest', test_file]
        
        if verbose:
            cmd.append('-v')
        
        if parallel:
            cmd.extend(['-n', 'auto'])
        
        # Add coverage if available
        try:
            import pytest_cov
            cmd.extend(['--cov=domain', '--cov=optimization', '--cov-report=term-missing'])
        except ImportError:
            pass
        
        start_time = time.time()
        
        try:
            # Run the test
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            test_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"    ✅ PASSED ({test_time:.2f}s)")
                return {
                    'success': True,
                    'time': test_time,
                    'output': result.stdout
                }
            else:
                print(f"    ❌ FAILED ({test_time:.2f}s)")
                if verbose:
                    print(f"      Error: {result.stderr}")
                return {
                    'success': False,
                    'time': test_time,
                    'output': result.stdout,
                    'error': result.stderr
                }
        
        except subprocess.TimeoutExpired:
            print(f"    ⏰ TIMEOUT (>300s)")
            return {
                'success': False,
                'time': 300,
                'error': 'Test timed out after 300 seconds'
            }
        except Exception as e:
            print(f"    💥 ERROR: {e}")
            return {
                'success': False,
                'time': 0,
                'error': str(e)
            }
    
    def _print_summary(self):
        """Print test execution summary"""
        total_time = time.time() - self.start_time
        
        print("\n" + "=" * 60)
        print("📊 TEST EXECUTION SUMMARY")
        print("=" * 60)
        
        # Category results
        for category, results in self.test_results.items():
            status = "✅ PASSED" if results['failed'] == 0 else "❌ FAILED"
            print(f"{category.upper():15} {status:10} "
                  f"Passed: {results['passed']:3} Failed: {results['failed']:3} "
                  f"Time: {results['time']:6.2f}s")
        
        print("-" * 60)
        
        # Overall results
        overall_status = "✅ ALL TESTS PASSED" if self.failed_tests == 0 else "❌ SOME TESTS FAILED"
        print(f"OVERALL STATUS: {overall_status}")
        print(f"TOTAL TESTS:   {self.total_tests}")
        print(f"PASSED:        {self.passed_tests}")
        print(f"FAILED:        {self.failed_tests}")
        print(f"SUCCESS RATE:  {(self.passed_tests / self.total_tests * 100):.1f}%" if self.total_tests > 0 else "N/A")
        print(f"TOTAL TIME:    {total_time:.2f}s")
        
        print("=" * 60)
        
        if self.failed_tests == 0:
            print("🎉 All tests completed successfully!")
        else:
            print(f"⚠️  {self.failed_tests} test(s) failed. Check the output above for details.")
    
    def run_simple_tests(self):
        """Run simple tests that don't require pytest"""
        print("\n🧪 Running Simple Tests...")
        print("-" * 40)
        
        simple_tests = [
            'test_basic.py'
        ]
        
        for test_file in simple_tests:
            if os.path.exists(test_file):
                print(f"  🧪 Running: {test_file}")
                try:
                    result = subprocess.run([sys.executable, test_file], 
                                         capture_output=True, text=True, timeout=60)
                    
                    if result.returncode == 0:
                        print(f"    ✅ PASSED")
                        self.passed_tests += 1
                    else:
                        print(f"    ❌ FAILED")
                        print(f"      Error: {result.stderr}")
                        self.failed_tests += 1
                    
                    self.total_tests += 1
                    
                except Exception as e:
                    print(f"    💥 ERROR: {e}")
                    self.failed_tests += 1
                    self.total_tests += 1
            else:
                print(f"  ⚠️  Test file not found: {test_file}")
    
    def check_system_health(self):
        """Check if the ADS system is healthy and can run tests"""
        print("\n🏥 Checking System Health...")
        print("-" * 40)
        
        health_checks = [
            ("Python version", self._check_python_version),
            ("Required modules", self._check_required_modules),
            ("Test files", self._check_test_files),
            ("System imports", self._check_system_imports)
        ]
        
        all_healthy = True
        
        for check_name, check_func in health_checks:
            try:
                result = check_func()
                if result:
                    print(f"  ✅ {check_name}: OK")
                else:
                    print(f"  ❌ {check_name}: FAILED")
                    all_healthy = False
            except Exception as e:
                print(f"  💥 {check_name}: ERROR - {e}")
                all_healthy = False
        
        if all_healthy:
            print("  🎉 System is healthy and ready for testing!")
        else:
            print("  ⚠️  System has some issues that may affect testing")
        
        return all_healthy
    
    def _check_python_version(self) -> bool:
        """Check Python version compatibility"""
        version = sys.version_info
        return version.major >= 3 and version.minor >= 8
    
    def _check_required_modules(self) -> bool:
        """Check if required modules are available"""
        required_modules = ['pytest', 'unittest', 'decimal', 'datetime']
        missing_modules = []
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            print(f"      Missing modules: {', '.join(missing_modules)}")
            return False
        
        return True
    
    def _check_test_files(self) -> bool:
        """Check if test files exist"""
        all_tests = []
        for category in self.test_categories.values():
            all_tests.extend(category['tests'])
        
        missing_files = [test for test in all_tests if not os.path.exists(test)]
        
        if missing_files:
            print(f"      Missing test files: {', '.join(missing_files)}")
            return False
        
        return True
    
    def _check_system_imports(self) -> bool:
        """Check if the ADS system can be imported"""
        try:
            # Try to import key modules
            sys.path.insert(0, '.')
            from domain.entities import Ad, AdCampaign, AdGroup
            from domain.value_objects import AdStatus, AdType, Platform
            from optimization.factory import OptimizationFactory
            return True
        except ImportError as e:
            print(f"      Import error: {e}")
            return False

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ADS System Test Runner')
    parser.add_argument('--categories', '-c', nargs='+', 
                       choices=['unit', 'integration', 'basic', 'all'],
                       default=['all'],
                       help='Test categories to run')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--parallel', '-p', action='store_true',
                       help='Run tests in parallel')
    parser.add_argument('--health-check', action='store_true',
                       help='Run system health check only')
    parser.add_argument('--simple', action='store_true',
                       help='Run simple tests only')
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    # Handle special cases
    if args.health_check:
        runner.check_system_health()
        return
    
    if args.simple:
        runner.run_simple_tests()
        return
    
    # Normal test execution
    if 'all' in args.categories:
        categories = list(runner.test_categories.keys())
    else:
        categories = args.categories
    
    # Check system health first
    if not runner.check_system_health():
        print("\n⚠️  System health check failed. Tests may not run correctly.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return
    
    # Run tests
    success = runner.run_tests(categories, args.verbose, args.parallel)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
