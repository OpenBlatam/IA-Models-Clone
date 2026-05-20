#!/usr/bin/env python3
"""
Comprehensive test runner for copywriting service.
"""
import sys
import os
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Any
import json
import time


class TestRunner:
    """Comprehensive test runner for copywriting service."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_dir = Path(__file__).parent
        self.results = {}
    
    def run_all_tests(self, verbose: bool = True, coverage: bool = False) -> Dict[str, Any]:
        """Run all tests and return results."""
        print("🚀 Running comprehensive copywriting service tests...")
        print("=" * 60)
        
        test_categories = [
            ("Unit Tests", "unit", "Fast, isolated component tests"),
            ("Integration Tests", "integration", "Component interaction tests"),
            ("API Tests", "api", "API endpoint tests"),
            ("Performance Tests", "performance", "Performance and load tests")
        ]
        
        total_passed = 0
        total_failed = 0
        total_skipped = 0
        
        for category_name, category_dir, description in test_categories:
            print(f"\n📁 {category_name} - {description}")
            print("-" * 40)
            
            category_results = self._run_test_category(category_dir, verbose, coverage)
            
            self.results[category_name] = category_results
            total_passed += category_results.get('passed', 0)
            total_failed += category_results.get('failed', 0)
            total_skipped += category_results.get('skipped', 0)
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"✅ Passed: {total_passed}")
        print(f"❌ Failed: {total_failed}")
        print(f"⏭️  Skipped: {total_skipped}")
        print(f"📈 Total: {total_passed + total_failed + total_skipped}")
        
        success_rate = (total_passed / (total_passed + total_failed)) * 100 if (total_passed + total_failed) > 0 else 0
        print(f"🎯 Success Rate: {success_rate:.1f}%")
        
        return {
            'total_passed': total_passed,
            'total_failed': total_failed,
            'total_skipped': total_skipped,
            'success_rate': success_rate,
            'categories': self.results
        }
    
    def _run_test_category(self, category: str, verbose: bool, coverage: bool) -> Dict[str, Any]:
        """Run tests for a specific category."""
        category_path = self.test_dir / category
        
        if not category_path.exists():
            print(f"⚠️  Category directory not found: {category_path}")
            return {'passed': 0, 'failed': 0, 'skipped': 0, 'error': 'Directory not found'}
        
        # Find test files
        test_files = list(category_path.glob("test_*.py"))
        
        if not test_files:
            print(f"⚠️  No test files found in {category_path}")
            return {'passed': 0, 'failed': 0, 'skipped': 0, 'error': 'No test files'}
        
        print(f"Found {len(test_files)} test files:")
        for test_file in test_files:
            print(f"  - {test_file.name}")
        
        # Build pytest command
        cmd = ["python", "-m", "pytest"]
        
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend(["--cov=agents.backend.onyx.server.features.copywriting", "--cov-report=html"])
        
        # Add test files
        cmd.extend([str(f) for f in test_files])
        
        # Add pytest options
        cmd.extend([
            "--tb=short",
            "--disable-warnings",
            "--color=yes",
            "--durations=10"
        ])
        
        print(f"\n🔧 Running: {' '.join(cmd)}")
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            end_time = time.time()
            
            duration = end_time - start_time
            
            # Parse results
            passed = result.stdout.count("PASSED")
            failed = result.stdout.count("FAILED")
            skipped = result.stdout.count("SKIPPED")
            
            print(f"⏱️  Duration: {duration:.2f}s")
            print(f"✅ Passed: {passed}")
            print(f"❌ Failed: {failed}")
            print(f"⏭️  Skipped: {skipped}")
            
            if result.returncode != 0:
                print(f"\n❌ Test failures detected:")
                print(result.stdout)
                if result.stderr:
                    print("STDERR:")
                    print(result.stderr)
            
            return {
                'passed': passed,
                'failed': failed,
                'skipped': skipped,
                'duration': duration,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return {'passed': 0, 'failed': 0, 'skipped': 0, 'error': str(e)}
    
    def run_specific_tests(self, test_pattern: str, verbose: bool = True) -> Dict[str, Any]:
        """Run specific tests matching a pattern."""
        print(f"🔍 Running tests matching: {test_pattern}")
        
        cmd = ["python", "-m", "pytest", "-k", test_pattern]
        
        if verbose:
            cmd.append("-v")
        
        cmd.extend([
            "--tb=short",
            "--disable-warnings",
            "--color=yes"
        ])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            passed = result.stdout.count("PASSED")
            failed = result.stdout.count("FAILED")
            skipped = result.stdout.count("SKIPPED")
            
            print(f"✅ Passed: {passed}")
            print(f"❌ Failed: {failed}")
            print(f"⏭️  Skipped: {skipped}")
            
            if result.returncode != 0:
                print(f"\n❌ Test failures:")
                print(result.stdout)
            
            return {
                'passed': passed,
                'failed': failed,
                'skipped': skipped,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return {'error': str(e)}
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests specifically."""
        print("⚡ Running performance tests...")
        
        cmd = [
            "python", "-m", "pytest",
            "tests/integration/test_performance.py",
            "-v",
            "--tb=short",
            "--disable-warnings",
            "-m", "performance"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            print("Performance test results:")
            print(result.stdout)
            
            return {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
        except Exception as e:
            print(f"❌ Error running performance tests: {e}")
            return {'error': str(e)}
    
    def generate_report(self, results: Dict[str, Any], output_file: str = "test_report.json"):
        """Generate a test report."""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_passed': results.get('total_passed', 0),
                'total_failed': results.get('total_failed', 0),
                'total_skipped': results.get('total_skipped', 0),
                'success_rate': results.get('success_rate', 0)
            },
            'categories': results.get('categories', {}),
            'environment': {
                'python_version': sys.version,
                'platform': os.name,
                'test_directory': str(self.test_dir)
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Test report saved to: {output_file}")
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed."""
        print("🔍 Checking dependencies...")
        
        required_packages = [
            'pytest',
            'pytest-asyncio',
            'pytest-cov',
            'fastapi',
            'httpx'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"✅ {package}")
            except ImportError:
                print(f"❌ {package} - MISSING")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
            print("Install with: pip install " + " ".join(missing_packages))
            return False
        
        print("✅ All dependencies available")
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Copywriting Service Test Runner")
    parser.add_argument("--category", choices=["unit", "integration", "api", "performance", "all"], 
                       default="all", help="Test category to run")
    parser.add_argument("--pattern", help="Test pattern to match")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies only")
    parser.add_argument("--report", help="Generate test report to file")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    if args.check_deps:
        success = runner.check_dependencies()
        sys.exit(0 if success else 1)
    
    if not runner.check_dependencies():
        print("❌ Missing dependencies. Run with --check-deps to see details.")
        sys.exit(1)
    
    if args.pattern:
        results = runner.run_specific_tests(args.pattern, args.verbose)
    elif args.category == "performance":
        results = runner.run_performance_tests()
    elif args.category == "all":
        results = runner.run_all_tests(args.verbose, args.coverage)
    else:
        results = runner._run_test_category(args.category, args.verbose, args.coverage)
    
    if args.report:
        runner.generate_report(results, args.report)
    
    # Exit with error code if tests failed
    total_failed = results.get('total_failed', 0)
    if isinstance(total_failed, int) and total_failed > 0:
        sys.exit(1)
    elif isinstance(results, dict) and results.get('returncode', 0) != 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()





