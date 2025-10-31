#!/usr/bin/env python3
"""
Test Runner for HeyGen AI
========================

Comprehensive test runner that can execute tests with different configurations
and provide detailed reporting.
"""

import sys
import os
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

class TestRunner:
    """Test runner for HeyGen AI test suite"""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent / "tests"
        self.results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None,
            "duration": 0
        }
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are available"""
        print("Checking dependencies...")
        
        try:
            import pytest
            print("✓ pytest is available")
        except ImportError:
            print("✗ pytest not found - will use validation script instead")
            return False
        
        try:
            import pytest_asyncio
            print("✓ pytest-asyncio is available")
        except ImportError:
            print("⚠ pytest-asyncio not found - async tests may not work")
        
        return True
    
    def run_validation_script(self) -> bool:
        """Run the validation script as fallback"""
        print("\nRunning validation script...")
        try:
            result = subprocess.run([
                sys.executable, "validate_tests.py"
            ], capture_output=True, text=True, cwd=Path(__file__).parent)
            
            print(result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
            
            return result.returncode == 0
        except Exception as e:
            print(f"Failed to run validation script: {e}")
            return False
    
    def run_pytest_tests(self, test_pattern: str = None, verbose: bool = True) -> bool:
        """Run tests using pytest"""
        print(f"\nRunning pytest tests...")
        
        cmd = [sys.executable, "-m", "pytest"]
        
        if verbose:
            cmd.append("-v")
        
        if test_pattern:
            cmd.append(test_pattern)
        else:
            cmd.append(str(self.test_dir))
        
        # Add additional pytest options
        cmd.extend([
            "--tb=short",
            "--strict-markers",
            "--disable-warnings"
        ])
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, cwd=Path(__file__).parent)
            end_time = time.time()
            
            self.results["start_time"] = start_time
            self.results["end_time"] = end_time
            self.results["duration"] = end_time - start_time
            
            return result.returncode == 0
        except Exception as e:
            print(f"Failed to run pytest: {e}")
            return False
    
    def run_specific_tests(self, test_files: List[str]) -> bool:
        """Run specific test files"""
        print(f"\nRunning specific test files: {test_files}")
        
        all_passed = True
        for test_file in test_files:
            test_path = self.test_dir / test_file
            if test_path.exists():
                print(f"\nRunning {test_file}...")
                success = self.run_pytest_tests(str(test_path))
                if not success:
                    all_passed = False
            else:
                print(f"Test file not found: {test_file}")
                all_passed = False
        
        return all_passed
    
    def run_test_categories(self) -> Dict[str, bool]:
        """Run tests by category"""
        categories = {
            "unit": ["test_core_structures.py", "test_enterprise_features.py"],
            "integration": ["test_advanced_integration.py", "test_enhanced_system.py"],
            "basic": ["test_basic_imports.py", "test_simple.py"]
        }
        
        results = {}
        
        for category, test_files in categories.items():
            print(f"\n{'='*50}")
            print(f"Running {category.upper()} tests")
            print(f"{'='*50}")
            
            results[category] = self.run_specific_tests(test_files)
        
        return results
    
    def generate_report(self) -> str:
        """Generate a test report"""
        report = []
        report.append("HeyGen AI Test Report")
        report.append("=" * 50)
        report.append(f"Start Time: {self.results.get('start_time', 'N/A')}")
        report.append(f"End Time: {self.results.get('end_time', 'N/A')}")
        report.append(f"Duration: {self.results.get('duration', 0):.2f} seconds")
        report.append("")
        
        if self.results["total_tests"] > 0:
            report.append(f"Total Tests: {self.results['total_tests']}")
            report.append(f"Passed: {self.results['passed']}")
            report.append(f"Failed: {self.results['failed']}")
            report.append(f"Skipped: {self.results['skipped']}")
            report.append(f"Errors: {self.results['errors']}")
            report.append("")
            
            success_rate = (self.results['passed'] / self.results['total_tests']) * 100
            report.append(f"Success Rate: {success_rate:.1f}%")
        
        return "\n".join(report)
    
    def main(self):
        """Main test runner function"""
        print("HeyGen AI Test Runner")
        print("=" * 50)
        
        # Check dependencies
        has_pytest = self.check_dependencies()
        
        if has_pytest:
            print("\nRunning full test suite with pytest...")
            success = self.run_pytest_tests()
        else:
            print("\nFalling back to validation script...")
            success = self.run_validation_script()
        
        # Generate and display report
        report = self.generate_report()
        print(f"\n{report}")
        
        if success:
            print("\n✅ All tests completed successfully!")
            return 0
        else:
            print("\n❌ Some tests failed. Please check the output above.")
            return 1

def main():
    """Entry point for the test runner"""
    runner = TestRunner()
    return runner.main()

if __name__ == "__main__":
    sys.exit(main())





