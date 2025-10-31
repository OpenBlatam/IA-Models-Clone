from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import time
import json
import logging
import requests
import random
from typing import Dict, Any, List
from datetime import datetime, timedelta
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Error Handling Middleware Demo
Product Descriptions Feature - Comprehensive Error Handling, Logging, and Monitoring Demonstration
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorHandlingDemo:
    """Comprehensive error handling middleware demonstration"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        
    """__init__ function."""
self.base_url = base_url
        self.session = requests.Session()
        self.results: List[Dict[str, Any]] = []
        self.error_monitor_stats: Dict[str, Any] = {}
    
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
    
    async def test_unexpected_errors(self) -> Dict[str, Any]:
        """Test handling of unexpected errors"""
        start_time = time.time()
        
        try:
            # Test with malformed JSON to trigger unexpected error
            response = self.session.post(
                f"{self.base_url}/git/status",
                data="invalid json data",
                headers={"Content-Type": "application/json"}
            )
            
            duration = time.time() - start_time
            
            success = response.status_code == 500
            data = {
                "status_code": response.status_code,
                "response": response.json() if response.content else {"error": "No response body"},
                "unexpected_error_handled": success,
                "error_type": "UNEXPECTED"
            }
            
            self.log_result("Unexpected Errors", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Unexpected Errors", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_validation_errors(self) -> Dict[str, Any]:
        """Test validation error handling"""
        start_time = time.time()
        
        try:
            # Test with invalid data to trigger validation errors
            payload = {
                "branch_name": "",  # Invalid empty name
                "base_branch": "main",
                "checkout": True
            }
            
            response = self.session.post(
                f"{self.base_url}/git/branch/create",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            duration = time.time() - start_time
            
            success = response.status_code == 400
            data = {
                "status_code": response.status_code,
                "response": response.json(),
                "validation_error_handled": success,
                "error_type": "VALIDATION"
            }
            
            self.log_result("Validation Errors", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Validation Errors", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_git_operation_errors(self) -> Dict[str, Any]:
        """Test git operation error handling"""
        start_time = time.time()
        
        try:
            # Test git status with non-existent repository
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
            
            success = response.status_code == 500
            data = {
                "status_code": response.status_code,
                "response": response.json(),
                "git_error_handled": success,
                "error_type": "GIT_OPERATION"
            }
            
            self.log_result("Git Operation Errors", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Git Operation Errors", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_model_version_errors(self) -> Dict[str, Any]:
        """Test model versioning error handling"""
        start_time = time.time()
        
        try:
            # Test model versioning with invalid data
            payload = {
                "model_name": "",  # Invalid empty name
                "version": "",     # Invalid empty version
                "description": "Test model",
                "tags": ["test"]
            }
            
            response = self.session.post(
                f"{self.base_url}/models/version",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            duration = time.time() - start_time
            
            success = response.status_code == 400
            data = {
                "status_code": response.status_code,
                "response": response.json(),
                "model_error_handled": success,
                "error_type": "VALIDATION"
            }
            
            self.log_result("Model Version Errors", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Model Version Errors", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_batch_processing_errors(self) -> Dict[str, Any]:
        """Test batch processing error handling"""
        start_time = time.time()
        
        try:
            # Test batch processing with invalid operation
            payload = {
                "items": [1, 2, 3],
                "operation": "invalid_operation",  # Invalid operation
                "batch_size": 10
            }
            
            response = self.session.post(
                f"{self.base_url}/batch/process",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            duration = time.time() - start_time
            
            success = response.status_code == 400
            data = {
                "status_code": response.status_code,
                "response": response.json(),
                "batch_error_handled": success,
                "error_type": "VALIDATION"
            }
            
            self.log_result("Batch Processing Errors", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Batch Processing Errors", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_error_monitoring(self) -> Dict[str, Any]:
        """Test error monitoring functionality"""
        start_time = time.time()
        
        try:
            # Get error monitoring data
            response = self.session.get(
                f"{self.base_url}/error/monitoring",
                headers={"Content-Type": "application/json"}
            )
            
            duration = time.time() - start_time
            
            success = response.status_code == 200
            response_data = response.json() if response.content else {}
            
            data = {
                "status_code": response.status_code,
                "response": response_data,
                "monitoring_available": success,
                "error_stats": response_data.get("data", {}).get("error_stats", {}),
                "circuit_breaker_status": response_data.get("data", {}).get("circuit_breaker_status", {}),
                "recent_alerts": response_data.get("data", {}).get("recent_alerts", 0)
            }
            
            # Store error monitor stats for later use
            self.error_monitor_stats = response_data.get("data", {})
            
            self.log_result("Error Monitoring", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Error Monitoring", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_error_statistics(self) -> Dict[str, Any]:
        """Test error statistics endpoint"""
        start_time = time.time()
        
        try:
            # Get error statistics
            response = self.session.get(
                f"{self.base_url}/errors/stats",
                headers={"Content-Type": "application/json"}
            )
            
            duration = time.time() - start_time
            
            success = response.status_code == 200
            response_data = response.json() if response.content else {}
            
            data = {
                "status_code": response.status_code,
                "response": response_data,
                "stats_available": success,
                "total_errors": response_data.get("data", {}).get("total_errors", 0),
                "error_rate": response_data.get("data", {}).get("error_rate", 0),
                "errors_by_type": response_data.get("data", {}).get("errors_by_type", {}),
                "errors_by_severity": response_data.get("data", {}).get("errors_by_severity", {})
            }
            
            self.log_result("Error Statistics", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Error Statistics", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_error_response_headers(self) -> Dict[str, Any]:
        """Test error response headers"""
        start_time = time.time()
        
        try:
            # Trigger a validation error to check headers
            payload = {
                "branch_name": "",
                "base_branch": "main",
                "checkout": True
            }
            
            response = self.session.post(
                f"{self.base_url}/git/branch/create",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            duration = time.time() - start_time
            
            headers = dict(response.headers)
            
            success = response.status_code == 400
            data = {
                "status_code": response.status_code,
                "headers": headers,
                "content_type": headers.get("content-type"),
                "has_request_id": "X-Request-ID" in headers,
                "has_error_id": "X-Error-ID" in headers,
                "headers_valid": success
            }
            
            self.log_result("Error Response Headers", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Error Response Headers", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_circuit_breaker_functionality(self) -> Dict[str, Any]:
        """Test circuit breaker functionality"""
        start_time = time.time()
        
        try:
            # Make multiple requests to trigger circuit breaker
            responses = []
            for i in range(5):
                try:
                    response = self.session.post(
                        f"{self.base_url}/git/status",
                        json={"include_untracked": True, "include_ignored": False},
                        headers={"Content-Type": "application/json"},
                        timeout=1
                    )
                    responses.append(response.status_code)
                except requests.exceptions.RequestException:
                    responses.append("timeout")
                
                await asyncio.sleep(0.1)
            
            duration = time.time() - start_time
            
            # Check if circuit breaker was triggered
            circuit_breaker_status = self.error_monitor_stats.get("circuit_breaker_status", {})
            
            success = "circuit_breaker_status" in self.error_monitor_stats
            data = {
                "responses": responses,
                "circuit_breaker_open": circuit_breaker_status.get("open", False),
                "failure_count": circuit_breaker_status.get("failure_count", 0),
                "threshold": circuit_breaker_status.get("threshold", 0),
                "circuit_breaker_functioning": success
            }
            
            self.log_result("Circuit Breaker Functionality", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Circuit Breaker Functionality", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async async def test_slow_request_detection(self) -> Dict[str, Any]:
        """Test slow request detection"""
        start_time = time.time()
        
        try:
            # Make a request that might be slow
            response = self.session.get(
                f"{self.base_url}/error/monitoring",
                headers={"Content-Type": "application/json"}
            )
            
            duration = time.time() - start_time
            
            success = response.status_code == 200
            data = {
                "status_code": response.status_code,
                "duration": duration,
                "slow_request_threshold": 1.0,  # 1 second
                "is_slow": duration > 1.0,
                "slow_request_detected": success
            }
            
            self.log_result("Slow Request Detection", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Slow Request Detection", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_error_logging(self) -> Dict[str, Any]:
        """Test error logging functionality"""
        start_time = time.time()
        
        try:
            # Trigger multiple errors to test logging
            errors = []
            
            # Validation error
            response1 = self.session.post(
                f"{self.base_url}/git/branch/create",
                json={"branch_name": "", "base_branch": "main", "checkout": True},
                headers={"Content-Type": "application/json"}
            )
            errors.append({
                "type": "validation",
                "status_code": response1.status_code,
                "error_code": response1.json().get("error_code") if response1.content else None
            })
            
            # Git operation error
            response2 = self.session.post(
                f"{self.base_url}/git/status",
                json={"include_untracked": True, "include_ignored": False},
                headers={"Content-Type": "application/json"}
            )
            errors.append({
                "type": "git_operation",
                "status_code": response2.status_code,
                "error_code": response2.json().get("error_code") if response2.content else None
            })
            
            # Unexpected error
            response3 = self.session.post(
                f"{self.base_url}/git/status",
                data="invalid json",
                headers={"Content-Type": "application/json"}
            )
            errors.append({
                "type": "unexpected",
                "status_code": response3.status_code,
                "error_code": response3.json().get("error_code") if response3.content else None
            })
            
            duration = time.time() - start_time
            
            success = all(error["status_code"] in [400, 500] for error in errors)
            data = {
                "errors_triggered": len(errors),
                "errors": errors,
                "all_errors_logged": success,
                "logging_functional": success
            }
            
            self.log_result("Error Logging", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Error Logging", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_error_context_tracking(self) -> Dict[str, Any]:
        """Test error context tracking"""
        start_time = time.time()
        
        try:
            # Make a request with custom headers to test context tracking
            headers = {
                "Content-Type": "application/json",
                "X-Correlation-ID": "test-correlation-123",
                "User-Agent": "ErrorHandlingDemo/1.0"
            }
            
            response = self.session.post(
                f"{self.base_url}/git/branch/create",
                json={"branch_name": "", "base_branch": "main", "checkout": True},
                headers=headers
            )
            
            duration = time.time() - start_time
            
            success = response.status_code == 400
            response_data = response.json() if response.content else {}
            
            data = {
                "status_code": response.status_code,
                "response": response_data,
                "has_request_id": "request_id" in response_data,
                "has_correlation_id": "correlation_id" in response_data,
                "correlation_id_match": response_data.get("correlation_id") == "test-correlation-123",
                "context_tracking_working": success
            }
            
            self.log_result("Error Context Tracking", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Error Context Tracking", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_error_cleanup(self) -> Dict[str, Any]:
        """Test error cleanup functionality"""
        start_time = time.time()
        
        try:
            # Clear old error records
            response = self.session.post(
                f"{self.base_url}/error/clear",
                headers={"Content-Type": "application/json"}
            )
            
            duration = time.time() - start_time
            
            success = response.status_code == 200
            response_data = response.json() if response.content else {}
            
            data = {
                "status_code": response.status_code,
                "response": response_data,
                "cleanup_successful": success,
                "cleared_count": response_data.get("data", {}).get("cleared_count", 0),
                "cleanup_functional": success
            }
            
            self.log_result("Error Cleanup", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Error Cleanup", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all error handling middleware tests"""
        logger.info("Starting Error Handling Middleware Demo Tests...")
        
        tests = [
            self.test_unexpected_errors,
            self.test_validation_errors,
            self.test_git_operation_errors,
            self.test_model_version_errors,
            self.test_batch_processing_errors,
            self.test_error_monitoring,
            self.test_error_statistics,
            self.test_error_response_headers,
            self.test_circuit_breaker_functionality,
            self.test_slow_request_detection,
            self.test_error_logging,
            self.test_error_context_tracking,
            self.test_error_cleanup
        ]
        
        for test in tests:
            try:
                await test()
                await asyncio.sleep(0.2)  # Small delay between tests
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
            "results": self.results,
            "error_monitor_summary": {
                "total_errors": self.error_monitor_stats.get("error_stats", {}).get("total_errors", 0),
                "error_rate": self.error_monitor_stats.get("error_stats", {}).get("error_rate", 0),
                "circuit_breaker_open": self.error_monitor_stats.get("circuit_breaker_status", {}).get("open", False),
                "recent_alerts": self.error_monitor_stats.get("recent_alerts", 0)
            }
        }
        
        logger.info(f"Error Handling Demo completed: {passed_tests}/{total_tests} tests passed")
        return summary
    
    def save_results(self, filename: str = "error_handling_demo_results.json"):
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

async def main():
    """Main demo execution"""
    print("=" * 70)
    print("ERROR HANDLING MIDDLEWARE DEMO - PRODUCT DESCRIPTIONS FEATURE")
    print("=" * 70)
    
    # Create demo instance
    demo = ErrorHandlingDemo()
    
    # Run all tests
    summary = await demo.run_all_tests()
    
    # Print summary
    print("\n" + "=" * 70)
    print("DEMO SUMMARY")
    print("=" * 70)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    
    # Print error monitor summary
    error_summary = summary.get("error_monitor_summary", {})
    print(f"\nError Monitor Summary:")
    print(f"  Total Errors: {error_summary.get('total_errors', 0)}")
    print(f"  Error Rate: {error_summary.get('error_rate', 0)} errors/minute")
    print(f"  Circuit Breaker Open: {error_summary.get('circuit_breaker_open', False)}")
    print(f"  Recent Alerts: {error_summary.get('recent_alerts', 0)}")
    
    # Print detailed results
    print("\n" + "=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)
    
    for result in summary['results']:
        status = "PASS" if result['success'] else "FAIL"
        print(f"{status}: {result['test']} ({result['duration']:.3f}s)")
        
        if not result['success'] and 'error' in result['data']:
            print(f"  Error: {result['data']['error']}")
    
    # Save results
    demo.save_results()
    
    print("\n" + "=" * 70)
    print("Demo completed! Check error_handling_demo_results.json for detailed results.")
    print("=" * 70)

match __name__:
    case "__main__":
    asyncio.run(main()) 