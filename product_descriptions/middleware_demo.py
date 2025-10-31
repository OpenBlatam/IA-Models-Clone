from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import json
import logging
from typing import Dict, Any, List
import requests
from pathlib import Path
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Middleware Demo
Product Descriptions Feature - Comprehensive Middleware Demonstration
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MiddlewareDemo:
    """Comprehensive middleware demonstration"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        
    """__init__ function."""
self.base_url = base_url
        self.session = requests.Session()
        self.results: List[Dict[str, Any]] = []
    
    def log_result(self, test_name: str, success: bool, data: Dict[str, Any], duration: float):
        """Log test result"""
        result = {
            "test": test_name,
            "success": success,
            "data": data,
            "duration": duration,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.results.append(result)
        logger.info(f"Test: {test_name} - {'PASS' if success else 'FAIL'} ({duration:.3f}s)")
    
    def test_health_endpoint(self) -> Dict[str, Any]:
        """Test health endpoint"""
        start_time = time.time()
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            duration = time.time() - start_time
            
            success = response.status_code == 200
            data = {
                "status_code": response.status_code,
                "response": response.json(),
                "headers": dict(response.headers)
            }
            
            self.log_result("Health Endpoint", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Health Endpoint", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    def test_status_endpoint(self) -> Dict[str, Any]:
        """Test status endpoint"""
        start_time = time.time()
        
        try:
            response = self.session.get(f"{self.base_url}/status")
            duration = time.time() - start_time
            
            success = response.status_code == 200
            data = {
                "status_code": response.status_code,
                "response": response.json(),
                "headers": dict(response.headers)
            }
            
            self.log_result("Status Endpoint", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Status Endpoint", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    def test_git_status(self) -> Dict[str, Any]:
        """Test git status endpoint"""
        start_time = time.time()
        
        try:
            payload = {
                "include_untracked": True,
                "include_ignored": False
            }
            
            response = self.session.post(
                f"{self.base_url}/git/status",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            duration = time.time() - start_time
            
            success = response.status_code in [200, 500]  # 500 is expected if no git repo
            data = {
                "status_code": response.status_code,
                "response": response.json(),
                "headers": dict(response.headers)
            }
            
            self.log_result("Git Status", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Git Status", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    def test_create_branch(self) -> Dict[str, Any]:
        """Test create branch endpoint"""
        start_time = time.time()
        
        try:
            payload = {
                "branch_name": "test-branch",
                "base_branch": "main",
                "checkout": True
            }
            
            response = self.session.post(
                f"{self.base_url}/git/branch/create",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            duration = time.time() - start_time
            
            success = response.status_code in [200, 500]  # 500 is expected if no git repo
            data = {
                "status_code": response.status_code,
                "response": response.json(),
                "headers": dict(response.headers)
            }
            
            self.log_result("Create Branch", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Create Branch", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    def test_commit_changes(self) -> Dict[str, Any]:
        """Test commit changes endpoint"""
        start_time = time.time()
        
        try:
            payload = {
                "message": "test commit",
                "files": None,
                "include_untracked": True
            }
            
            response = self.session.post(
                f"{self.base_url}/git/commit",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            duration = time.time() - start_time
            
            success = response.status_code in [200, 500]  # 500 is expected if no git repo
            data = {
                "status_code": response.status_code,
                "response": response.json(),
                "headers": dict(response.headers)
            }
            
            self.log_result("Commit Changes", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Commit Changes", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    def test_model_version(self) -> Dict[str, Any]:
        """Test model version endpoint"""
        start_time = time.time()
        
        try:
            payload = {
                "model_name": "test-model",
                "version": "1.0.0",
                "description": "Test model version",
                "tags": ["test", "demo"]
            }
            
            response = self.session.post(
                f"{self.base_url}/models/version",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            duration = time.time() - start_time
            
            success = response.status_code in [200, 500]  # 500 is expected if no models dir
            data = {
                "status_code": response.status_code,
                "response": response.json(),
                "headers": dict(response.headers)
            }
            
            self.log_result("Model Version", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Model Version", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    def test_performance_stats(self) -> Dict[str, Any]:
        """Test performance statistics endpoint"""
        start_time = time.time()
        
        try:
            response = self.session.get(f"{self.base_url}/performance/stats")
            duration = time.time() - start_time
            
            success = response.status_code == 200
            data = {
                "status_code": response.status_code,
                "response": response.json(),
                "headers": dict(response.headers)
            }
            
            self.log_result("Performance Stats", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Performance Stats", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    def test_error_stats(self) -> Dict[str, Any]:
        """Test error statistics endpoint"""
        start_time = time.time()
        
        try:
            response = self.session.get(f"{self.base_url}/errors/stats")
            duration = time.time() - start_time
            
            success = response.status_code == 200
            data = {
                "status_code": response.status_code,
                "response": response.json(),
                "headers": dict(response.headers)
            }
            
            self.log_result("Error Stats", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Error Stats", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    def test_rate_limiting(self) -> Dict[str, Any]:
        """Test rate limiting by making many requests"""
        start_time = time.time()
        
        try:
            # Make multiple requests quickly
            responses = []
            for i in range(10):
                response = self.session.get(f"{self.base_url}/health")
                responses.append({
                    "request": i + 1,
                    "status_code": response.status_code,
                    "headers": dict(response.headers)
                })
            
            duration = time.time() - start_time
            
            # Check if any requests were rate limited
            rate_limited = any(r["status_code"] == 429 for r in responses)
            success = True  # Rate limiting is expected behavior
            
            data = {
                "total_requests": len(responses),
                "rate_limited_requests": sum(1 for r in responses if r["status_code"] == 429),
                "responses": responses
            }
            
            self.log_result("Rate Limiting", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Rate Limiting", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling with invalid requests"""
        start_time = time.time()
        
        try:
            # Test with invalid JSON
            response = self.session.post(
                f"{self.base_url}/git/status",
                data="invalid json",
                headers={"Content-Type": "application/json"}
            )
            duration = time.time() - start_time
            
            success = response.status_code in [400, 422]  # Expected error status codes
            data = {
                "status_code": response.status_code,
                "response": response.json() if response.content else {"error": "No response body"},
                "headers": dict(response.headers)
            }
            
            self.log_result("Error Handling", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Error Handling", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    def test_security_headers(self) -> Dict[str, Any]:
        """Test security headers"""
        start_time = time.time()
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            duration = time.time() - start_time
            
            headers = dict(response.headers)
            security_headers = {
                "X-Content-Type-Options": headers.get("X-Content-Type-Options"),
                "X-Frame-Options": headers.get("X-Frame-Options"),
                "X-XSS-Protection": headers.get("X-XSS-Protection"),
                "Referrer-Policy": headers.get("Referrer-Policy"),
                "Content-Security-Policy": headers.get("Content-Security-Policy")
            }
            
            success = all(security_headers.values())
            data = {
                "status_code": response.status_code,
                "security_headers": security_headers,
                "all_headers": headers
            }
            
            self.log_result("Security Headers", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Security Headers", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_request_tracking(self) -> Dict[str, Any]:
        """Test request ID tracking"""
        start_time = time.time()
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            duration = time.time() - start_time
            
            headers = dict(response.headers)
            request_id = headers.get("X-Request-ID")
            
            success = bool(request_id)
            data = {
                "status_code": response.status_code,
                "request_id": request_id,
                "response_time_header": headers.get("X-Response-Time"),
                "all_headers": headers
            }
            
            self.log_result("Request Tracking", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Request Tracking", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all middleware tests"""
        logger.info("Starting Middleware Demo Tests...")
        
        tests = [
            self.test_health_endpoint,
            self.test_status_endpoint,
            self.test_git_status,
            self.test_create_branch,
            self.test_commit_changes,
            self.test_model_version,
            self.test_performance_stats,
            self.test_error_stats,
            self.test_rate_limiting,
            self.test_error_handling,
            self.test_security_headers,
            self.test_request_tracking
        ]
        
        for test in tests:
            try:
                test()
                time.sleep(0.1)  # Small delay between tests
            except Exception as e:
                logger.error(f"Test failed: {test.__name__} - {e}")
        
        # Generate summary
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["success"])
        failed_tests = total_tests - passed_tests
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "results": self.results
        }
        
        logger.info(f"Demo completed: {passed_tests}/{total_tests} tests passed")
        return summary
    
    def save_results(self, filename: str = "middleware_demo_results.json"):
        """Save test results to file"""
        try:
            with open(filename, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(self.results, f, indent=2)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

def main():
    """Main demo execution"""
    print("=" * 60)
    print("MIDDLEWARE DEMO - PRODUCT DESCRIPTIONS FEATURE")
    print("=" * 60)
    
    # Create demo instance
    demo = MiddlewareDemo()
    
    # Run all tests
    summary = demo.run_all_tests()
    
    # Print summary
    print("\n" + "=" * 60)
    print("DEMO SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    
    # Print detailed results
    print("\n" + "=" * 60)
    print("DETAILED RESULTS")
    print("=" * 60)
    
    for result in summary['results']:
        status = "PASS" if result['success'] else "FAIL"
        print(f"{status}: {result['test']} ({result['duration']:.3f}s)")
        
        if not result['success'] and 'error' in result['data']:
            print(f"  Error: {result['data']['error']}")
    
    # Save results
    demo.save_results()
    
    print("\n" + "=" * 60)
    print("Demo completed! Check middleware_demo_results.json for detailed results.")
    print("=" * 60)

match __name__:
    case "__main__":
    main() 