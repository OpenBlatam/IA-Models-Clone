#!/usr/bin/env python3
"""
API Test Suite
==============
Automated test suite for API endpoints.

⚠️ DEPRECATED: This file is deprecated. Use `tools.refactored_test_suite.TestSuite` instead.

For new code, use:
    from tools.refactored_test_suite import TestSuite
    # or
    from tools.manager import ToolManager
    manager = ToolManager()
    result = manager.run_tool("test")
"""
import warnings

warnings.warn(
    "api_test_suite.py is deprecated. Use 'tools.refactored_test_suite.TestSuite' instead.",
    DeprecationWarning,
    stacklevel=2
)

import json
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class TestResult:
    """Test result."""
    name: str
    status: str  # passed, failed, skipped
    duration: float
    error: Optional[str] = None
    response: Optional[Dict[str, Any]] = None


class APITestSuite:
    """Automated API test suite."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": "Bearer test_token_123"
        })
        self.results: List[TestResult] = []
    
    def test_health(self) -> TestResult:
        """Test health endpoint."""
        import time
        start = time.time()
        
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            duration = time.time() - start
            
            if response.status_code == 200:
                return TestResult(
                    name="Health Check",
                    status="passed",
                    duration=duration,
                    response={"status": response.status_code, "data": response.json()}
                )
            else:
                return TestResult(
                    name="Health Check",
                    status="failed",
                    duration=duration,
                    error=f"Expected 200, got {response.status_code}"
                )
        except Exception as e:
            return TestResult(
                name="Health Check",
                status="failed",
                duration=time.time() - start,
                error=str(e)
            )
    
    def test_upload(self, file_path: Path) -> TestResult:
        """Test file upload."""
        import time
        start = time.time()
        
        if not file_path.exists():
            return TestResult(
                name="File Upload",
                status="skipped",
                duration=0,
                error=f"File not found: {file_path}"
            )
        
        try:
            with open(file_path, "rb") as f:
                files = {"file": (file_path.name, f, "application/pdf")}
                response = self.session.post(f"{self.base_url}/pdf/upload", files=files, timeout=30)
                duration = time.time() - start
            
            if response.status_code in [200, 201]:
                data = response.json()
                return TestResult(
                    name="File Upload",
                    status="passed",
                    duration=duration,
                    response={"status": response.status_code, "data": data}
                )
            else:
                return TestResult(
                    name="File Upload",
                    status="failed",
                    duration=duration,
                    error=f"Expected 200/201, got {response.status_code}: {response.text[:200]}"
                )
        except Exception as e:
            return TestResult(
                name="File Upload",
                status="failed",
                duration=time.time() - start,
                error=str(e)
            )
    
    def test_endpoint(self, name: str, method: str, endpoint: str, 
                     expected_status: int = 200, **kwargs) -> TestResult:
        """Test a generic endpoint."""
        import time
        start = time.time()
        
        try:
            url = f"{self.base_url}{endpoint}"
            
            if method.upper() == "GET":
                response = self.session.get(url, timeout=10, **kwargs)
            elif method.upper() == "POST":
                response = self.session.post(url, timeout=10, **kwargs)
            elif method.upper() == "PUT":
                response = self.session.put(url, timeout=10, **kwargs)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, timeout=10, **kwargs)
            else:
                return TestResult(
                    name=name,
                    status="failed",
                    duration=time.time() - start,
                    error=f"Unsupported method: {method}"
                )
            
            duration = time.time() - start
            
            if response.status_code == expected_status:
                try:
                    data = response.json()
                except:
                    data = response.text[:500]
                
                return TestResult(
                    name=name,
                    status="passed",
                    duration=duration,
                    response={"status": response.status_code, "data": data}
                )
            else:
                return TestResult(
                    name=name,
                    status="failed",
                    duration=duration,
                    error=f"Expected {expected_status}, got {response.status_code}: {response.text[:200]}"
                )
        except Exception as e:
            return TestResult(
                name=name,
                status="failed",
                duration=time.time() - start,
                error=str(e)
            )
    
    def run_suite(self, tests: List[Dict[str, Any]]) -> List[TestResult]:
        """Run a test suite."""
        print("🧪 Running API Test Suite")
        print("=" * 60)
        
        for test_config in tests:
            test_type = test_config.get("type")
            
            if test_type == "health":
                result = self.test_health()
            elif test_type == "upload":
                file_path = Path(test_config.get("file"))
                result = self.test_upload(file_path)
            elif test_type == "endpoint":
                result = self.test_endpoint(
                    test_config.get("name", "Test"),
                    test_config.get("method", "GET"),
                    test_config.get("endpoint"),
                    test_config.get("expected_status", 200),
                    **test_config.get("kwargs", {})
                )
            else:
                result = TestResult(
                    name=test_config.get("name", "Unknown"),
                    status="skipped",
                    duration=0,
                    error=f"Unknown test type: {test_type}"
                )
            
            self.results.append(result)
            
            # Print result
            status_emoji = "✅" if result.status == "passed" else "❌" if result.status == "failed" else "⏭️"
            print(f"{status_emoji} {result.name}: {result.status} ({result.duration:.3f}s)")
            if result.error:
                print(f"   Error: {result.error}")
        
        return self.results
    
    def print_summary(self):
        """Print test summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == "passed")
        failed = sum(1 for r in self.results if r.status == "failed")
        skipped = sum(1 for r in self.results if r.status == "skipped")
        
        total_duration = sum(r.duration for r in self.results)
        
        print("\n" + "=" * 60)
        print("📊 Test Summary")
        print("=" * 60)
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⏭️  Skipped: {skipped}")
        print(f"⏱️  Total Duration: {total_duration:.3f}s")
        print(f"📈 Success Rate: {(passed/total*100) if total > 0 else 0:.1f}%")
        print("=" * 60)
    
    def export_results(self, file_path: Path):
        """Export test results."""
        data = {
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results if r.status == "passed"),
                "failed": sum(1 for r in self.results if r.status == "failed"),
                "skipped": sum(1 for r in self.results if r.status == "skipped"),
            },
            "results": [asdict(r) for r in self.results],
            "exported_at": datetime.now().isoformat()
        }
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Test results exported to {file_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="API Test Suite")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--suite", help="Test suite JSON file")
    parser.add_argument("--export", help="Export results to file")
    
    args = parser.parse_args()
    
    suite = APITestSuite(base_url=args.url)
    
    if args.suite:
        # Load test suite from file
        with open(args.suite, "r") as f:
            tests = json.load(f)
        suite.run_suite(tests)
    else:
        # Run default test suite
        default_tests = [
            {"type": "health"},
            {"type": "endpoint", "name": "Root Endpoint", "method": "GET", "endpoint": "/"},
        ]
        suite.run_suite(default_tests)
    
    suite.print_summary()
    
    if args.export:
        suite.export_results(Path(args.export))


if __name__ == "__main__":
    main()



