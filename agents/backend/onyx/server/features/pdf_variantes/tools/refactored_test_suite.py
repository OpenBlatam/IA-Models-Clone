"""
Refactored Test Suite
=====================
Test suite using base classes and improved structure.
"""

from typing import Dict, Any, List
from pathlib import Path
import time
from .base import BaseAPITool, ToolResult
from .config import get_config
from .utils import print_success, print_error, print_info


class TestSuite(BaseAPITool):
    """Refactored test suite."""
    
    def test_endpoint(
        self,
        name: str,
        method: str,
        endpoint: str,
        expected_status: int = 200,
        **kwargs
    ) -> Dict[str, Any]:
        """Test a single endpoint."""
        start_time = time.time()
        
        try:
            response = self.make_request(method, endpoint, **kwargs)
            duration = time.time() - start_time
            
            success = response.status_code == expected_status
            
            return {
                "name": name,
                "status": "passed" if success else "failed",
                "duration": duration,
                "status_code": response.status_code,
                "expected_status": expected_status,
                "error": None if success else f"Expected {expected_status}, got {response.status_code}"
            }
        except Exception as e:
            return {
                "name": name,
                "status": "failed",
                "duration": time.time() - start_time,
                "status_code": 0,
                "expected_status": expected_status,
                "error": str(e)
            }
    
    def run(self, tests: List[Dict[str, Any]] = None, **kwargs) -> ToolResult:
        """Run test suite."""
        if tests is None:
            tests = [
                {"name": "Health Check", "method": "GET", "endpoint": "/health", "expected_status": 200}
            ]
        
        print_info(f"Running test suite with {len(tests)} tests...")
        
        results = []
        for test_config in tests:
            result = self.test_endpoint(**test_config)
            results.append(result)
            self.results.append(result)
            
            if result["status"] == "passed":
                print_success(f"{result['name']}: passed ({result['duration']:.3f}s)")
            else:
                print_error(f"{result['name']}: failed - {result.get('error', 'Unknown error')}")
        
        passed = sum(1 for r in results if r["status"] == "passed")
        failed = len(results) - passed
        
        success = failed == 0
        message = f"Tests: {passed} passed, {failed} failed out of {len(results)}"
        
        return ToolResult(
            success=success,
            message=message,
            data={
                "total": len(results),
                "passed": passed,
                "failed": failed,
                "results": results
            }
        )


def main():
    """Main entry point."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Test Suite (Refactored)")
    parser.add_argument("--url", help="API base URL")
    parser.add_argument("--suite", help="Test suite JSON file")
    parser.add_argument("--export", help="Export results")
    
    args = parser.parse_args()
    
    config = get_config()
    base_url = args.url or config.base_url
    
    suite = TestSuite(base_url=base_url)
    
    tests = None
    if args.suite:
        with open(args.suite, "r") as f:
            suite_data = json.load(f)
            tests = suite_data.get("tests", [])
    
    result = suite.run(tests=tests)
    
    if args.export:
        suite.export_results(args.export)
    
    return 0 if result.success else 1


if __name__ == "__main__":
    exit(main())



