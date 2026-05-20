from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import sys
import os
import subprocess
import argparse
import time
from datetime import datetime
from pathlib import Path
import json
import asyncio
from tests.debug_tools import APIDebugger, print_debug_info
            from tests.load_test import run_comprehensive_load_test
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Test Runner for LinkedIn Posts API
==================================

Comprehensive test runner with multiple configurations and reporting.
"""


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))



class TestRunner:
    """Comprehensive test runner with multiple configurations."""
    
    def __init__(self) -> Any:
        self.debugger = APIDebugger()
        self.test_results = {}
        self.start_time = None
        
    def run_unit_tests(self, verbose: bool = False) -> dict:
        """Run unit tests."""
        print("🧪 Running Unit Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            "tests/unit/",
            "-m", "unit",
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        return {
            "type": "unit",
            "success": result.returncode == 0,
            "duration": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
    
    def run_integration_tests(self, verbose: bool = False) -> dict:
        """Run integration tests."""
        print("🔗 Running Integration Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            "tests/integration/",
            "-m", "integration",
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        return {
            "type": "integration",
            "success": result.returncode == 0,
            "duration": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
    
    def run_performance_tests(self, verbose: bool = False) -> dict:
        """Run performance tests."""
        print("⚡ Running Performance Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            "tests/",
            "-m", "performance",
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        return {
            "type": "performance",
            "success": result.returncode == 0,
            "duration": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
    
    async def run_load_tests(self) -> dict:
        """Run load tests."""
        print("🚀 Running Load Tests...")
        
        try:
            start_time = time.time()
            report = await run_comprehensive_load_test()
            end_time = time.time()
            
            return {
                "type": "load",
                "success": True,
                "duration": end_time - start_time,
                "report": report
            }
        except Exception as e:
            return {
                "type": "load",
                "success": False,
                "duration": 0,
                "error": str(e)
            }
    
    def run_coverage_tests(self, verbose: bool = False) -> dict:
        """Run tests with coverage."""
        print("📊 Running Coverage Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            "tests/",
            "--cov=linkedin_posts",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--cov-report=xml",
            "--cov-fail-under=80"
        ]
        
        if verbose:
            cmd.append("-v")
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        return {
            "type": "coverage",
            "success": result.returncode == 0,
            "duration": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
    
    def run_debug_tests(self, verbose: bool = False) -> dict:
        """Run debug tests."""
        print("🐛 Running Debug Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            "tests/",
            "-m", "debug",
            "--tb=long"
        ]
        
        if verbose:
            cmd.append("-v")
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        return {
            "type": "debug",
            "success": result.returncode == 0,
            "duration": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
    
    def run_all_tests(self, verbose: bool = False, include_load: bool = True) -> dict:
        """Run all tests."""
        print("🎯 Running All Tests...")
        
        self.start_time = time.time()
        
        # Run different test types
        test_suites = [
            ("unit", lambda: self.run_unit_tests(verbose)),
            ("integration", lambda: self.run_integration_tests(verbose)),
            ("performance", lambda: self.run_performance_tests(verbose)),
            ("coverage", lambda: self.run_coverage_tests(verbose)),
            ("debug", lambda: self.run_debug_tests(verbose))
        ]
        
        if include_load:
            test_suites.append(("load", lambda: asyncio.run(self.run_load_tests())))
        
        for test_type, test_func in test_suites:
            try:
                result = test_func()
                self.test_results[test_type] = result
                
                status = "✅ PASSED" if result["success"] else "❌ FAILED"
                print(f"  {status} {test_type.upper()} tests ({result['duration']:.2f}s)")
                
            except Exception as e:
                self.test_results[test_type] = {
                    "type": test_type,
                    "success": False,
                    "duration": 0,
                    "error": str(e)
                }
                print(f"  ❌ ERROR {test_type.upper()} tests: {e}")
        
        total_time = time.time() - self.start_time
        
        return {
            "total_duration": total_time,
            "results": self.test_results,
            "summary": self.generate_summary()
        }
    
    def generate_summary(self) -> dict:
        """Generate test summary."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r["success"])
        failed_tests = total_tests - passed_tests
        
        total_duration = sum(r.get("duration", 0) for r in self.test_results.values())
        
        return {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "total_duration": total_duration
        }
    
    def print_summary(self) -> Any:
        """Print test summary."""
        summary = self.generate_summary()
        
        print("\n" + "="*60)
        print("📋 TEST SUMMARY")
        print("="*60)
        print(f"Total Test Suites: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Total Duration: {summary['total_duration']:.2f}s")
        
        if self.start_time:
            total_time = time.time() - self.start_time
            print(f"Total Time: {total_time:.2f}s")
        
        print("\nDetailed Results:")
        for test_type, result in self.test_results.items():
            status = "✅ PASSED" if result["success"] else "❌ FAILED"
            duration = result.get("duration", 0)
            print(f"  {test_type.upper()}: {status} ({duration:.2f}s)")
            
            if not result["success"] and "error" in result:
                print(f"    Error: {result['error']}")
        
        print("="*60)
    
    def save_results(self, filename: str = None) -> str:
        """Save test results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"
        
        # Prepare results for JSON serialization
        serializable_results = {}
        for test_type, result in self.test_results.items():
            serializable_results[test_type] = {
                "type": result["type"],
                "success": result["success"],
                "duration": result.get("duration", 0),
                "return_code": result.get("return_code"),
                "error": result.get("error"),
                "timestamp": datetime.now().isoformat()
            }
        
        data = {
            "summary": self.generate_summary(),
            "results": serializable_results,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(data, f, indent=2)
        
        print(f"📄 Test results saved to: {filename}")
        return filename


def main():
    """Main function for test runner."""
    parser = argparse.ArgumentParser(description="LinkedIn Posts API Test Runner")
    parser.add_argument(
        "--test-type",
        choices=["unit", "integration", "performance", "load", "coverage", "debug", "all"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--no-load",
        action="store_true",
        help="Skip load tests"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save results to file"
    )
    parser.add_argument(
        "--output-file",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    try:
        if args.test_type == "all":
            results = runner.run_all_tests(
                verbose=args.verbose,
                include_load=not args.no_load
            )
        elif args.test_type == "unit":
            results = {"results": {"unit": runner.run_unit_tests(args.verbose)}}
        elif args.test_type == "integration":
            results = {"results": {"integration": runner.run_integration_tests(args.verbose)}}
        elif args.test_type == "performance":
            results = {"results": {"performance": runner.run_performance_tests(args.verbose)}}
        elif args.test_type == "load":
            results = {"results": {"load": asyncio.run(runner.run_load_tests())}}
        elif args.test_type == "coverage":
            results = {"results": {"coverage": runner.run_coverage_tests(args.verbose)}}
        elif args.test_type == "debug":
            results = {"results": {"debug": runner.run_debug_tests(args.verbose)}}
        
        # Print summary
        runner.print_summary()
        
        # Save results if requested
        if args.save_results:
            output_file = args.output_file or f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            runner.save_results(output_file)
        
        # Exit with appropriate code
        summary = runner.generate_summary()
        if summary["failed"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n⚠️ Test run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test run failed: {e}")
        sys.exit(1)


match __name__:
    case "__main__":
    main() 