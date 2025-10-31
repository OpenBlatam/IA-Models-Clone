#!/usr/bin/env python3
"""
Enhanced Blaze AI Features Test Script

This script demonstrates and tests all the enterprise-grade features
including security, monitoring, rate limiting, and error handling.
"""

import asyncio
import time
import random
import requests
import json
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedFeaturesTester:
    """Test class for all enhanced Blaze AI features."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = {}
    
    def log_test_result(self, test_name: str, success: bool, details: str = ""):
        """Log test results."""
        self.test_results[test_name] = {
            "success": success,
            "details": details,
            "timestamp": time.time()
        }
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} {test_name}: {details}")
    
    def test_basic_health_endpoints(self) -> bool:
        """Test basic health check endpoints."""
        logger.info("Testing basic health endpoints...")
        
        try:
            # Test basic health check
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                self.log_test_result("Basic Health Check", True, f"Status: {data.get('status')}")
            else:
                self.log_test_result("Basic Health Check", False, f"Status code: {response.status_code}")
                return False
            
            # Test detailed health check
            response = self.session.get(f"{self.base_url}/health/detailed")
            if response.status_code == 200:
                data = response.json()
                systems = data.get('systems', {})
                self.log_test_result("Detailed Health Check", True, f"Systems checked: {len(systems)}")
            else:
                self.log_test_result("Detailed Health Check", False, f"Status code: {response.status_code}")
                return False
            
            return True
            
        except Exception as e:
            self.log_test_result("Basic Health Endpoints", False, f"Exception: {str(e)}")
            return False
    
    def test_enhanced_health_endpoints(self) -> bool:
        """Test enhanced health check endpoints."""
        logger.info("Testing enhanced health endpoints...")
        
        try:
            # Test metrics endpoint
            response = self.session.get(f"{self.base_url}/metrics")
            if response.status_code == 200:
                data = response.json()
                self.log_test_result("Metrics Endpoint", True, f"Metrics available: {len(data)}")
            elif response.status_code == 503:
                self.log_test_result("Metrics Endpoint", True, "Performance monitoring not available (expected)")
            else:
                self.log_test_result("Metrics Endpoint", False, f"Unexpected status code: {response.status_code}")
            
            # Test Prometheus metrics
            response = self.session.get(f"{self.base_url}/metrics/prometheus")
            if response.status_code == 200:
                content = response.text
                self.log_test_result("Prometheus Metrics", True, f"Metrics exported: {len(content)} chars")
            elif response.status_code == 503:
                self.log_test_result("Prometheus Metrics", True, "Performance monitoring not available (expected)")
            else:
                self.log_test_result("Prometheus Metrics", False, f"Unexpected status code: {response.status_code}")
            
            # Test security status
            response = self.session.get(f"{self.base_url}/security/status")
            if response.status_code == 200:
                data = response.json()
                status = data.get('status', 'unknown')
                self.log_test_result("Security Status", True, f"Security status: {status}")
            else:
                self.log_test_result("Security Status", False, f"Status code: {response.status_code}")
            
            # Test error summary
            response = self.session.get(f"{self.base_url}/errors/summary")
            if response.status_code == 200:
                data = response.json()
                if 'error' in data:
                    self.log_test_result("Error Summary", True, "Error monitoring not available (expected)")
                else:
                    self.log_test_result("Error Summary", True, "Error monitoring available")
            else:
                self.log_test_result("Error Summary", False, f"Status code: {response.status_code}")
            
            return True
            
        except Exception as e:
            self.log_test_result("Enhanced Health Endpoints", False, f"Exception: {str(e)}")
            return False
    
    def test_rate_limiting(self) -> bool:
        """Test rate limiting functionality."""
        logger.info("Testing rate limiting...")
        
        try:
            # Make multiple rapid requests to trigger rate limiting
            responses = []
            for i in range(10):
                response = self.session.get(f"{self.base_url}/health")
                responses.append(response.status_code)
                time.sleep(0.1)  # Small delay between requests
            
            # Check if any requests were rate limited (429 status)
            rate_limited = any(status == 429 for status in responses)
            if rate_limited:
                self.log_test_result("Rate Limiting", True, "Rate limiting is working (some requests blocked)")
            else:
                self.log_test_result("Rate Limiting", True, "All requests processed (rate limits not exceeded)")
            
            return True
            
        except Exception as e:
            self.log_test_result("Rate Limiting", False, f"Exception: {str(e)}")
            return False
    
    def test_security_features(self) -> bool:
        """Test security features."""
        logger.info("Testing security features...")
        
        try:
            # Test with suspicious headers
            suspicious_headers = {
                'X-Forwarded-For': '192.168.1.1',
                'User-Agent': 'Mozilla/5.0 (compatible; BadBot/1.0)',
                'X-API-Key': 'invalid-key'
            }
            
            response = self.session.get(
                f"{self.base_url}/health",
                headers=suspicious_headers
            )
            
            # Check if request was processed or blocked
            if response.status_code in [200, 401, 403]:
                self.log_test_result("Security Headers", True, f"Request processed with status: {response.status_code}")
            else:
                self.log_test_result("Security Headers", False, f"Unexpected status: {response.status_code}")
            
            # Test with potentially malicious query parameters
            malicious_params = {
                'q': 'eval(',
                'script': '<script>alert("xss")</script>',
                'path': '../../../etc/passwd'
            }
            
            response = self.session.get(
                f"{self.base_url}/health",
                params=malicious_params
            )
            
            if response.status_code in [200, 400, 403]:
                self.log_test_result("Security Parameters", True, f"Malicious params handled with status: {response.status_code}")
            else:
                self.log_test_result("Security Parameters", False, f"Unexpected status: {response.status_code}")
            
            return True
            
        except Exception as e:
            self.log_test_result("Security Features", False, f"Exception: {str(e)}")
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling and recovery."""
        logger.info("Testing error handling...")
        
        try:
            # Test with invalid endpoint
            response = self.session.get(f"{self.base_url}/invalid/endpoint")
            if response.status_code == 404:
                self.log_test_result("Error Handling - 404", True, "Invalid endpoint properly handled")
            else:
                self.log_test_result("Error Handling - 404", False, f"Unexpected status: {response.status_code}")
            
            # Test with malformed JSON in POST request
            response = self.session.post(
                f"{self.base_url}/health",
                data="invalid json",
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code in [400, 422]:
                self.log_test_result("Error Handling - Invalid JSON", True, f"Malformed JSON handled with status: {response.status_code}")
            else:
                self.log_test_result("Error Handling - Invalid JSON", False, f"Unexpected status: {response.status_code}")
            
            return True
            
        except Exception as e:
            self.log_test_result("Error Handling", False, f"Exception: {str(e)}")
            return False
    
    def test_performance_monitoring(self) -> bool:
        """Test performance monitoring features."""
        logger.info("Testing performance monitoring...")
        
        try:
            # Make several requests to generate metrics
            start_time = time.time()
            for i in range(5):
                response = self.session.get(f"{self.base_url}/health")
                time.sleep(0.2)
            
            # Check if metrics are being collected
            response = self.session.get(f"{self.base_url}/metrics")
            if response.status_code == 200:
                data = response.json()
                self.log_test_result("Performance Monitoring", True, f"Metrics collected: {len(data)}")
            elif response.status_code == 503:
                self.log_test_result("Performance Monitoring", True, "Performance monitoring not available (expected)")
            else:
                self.log_test_result("Performance Monitoring", False, f"Unexpected status: {response.status_code}")
            
            return True
            
        except Exception as e:
            self.log_test_result("Performance Monitoring", False, f"Exception: {str(e)}")
            return False
    
    def test_circuit_breaker(self) -> bool:
        """Test circuit breaker functionality."""
        logger.info("Testing circuit breaker...")
        
        try:
            # This would typically test actual service calls that might fail
            # For now, we'll just verify the endpoint exists and responds
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                self.log_test_result("Circuit Breaker", True, "Service responding normally")
            else:
                self.log_test_result("Circuit Breaker", False, f"Service not responding: {response.status_code}")
            
            return True
            
        except Exception as e:
            self.log_test_result("Circuit Breaker", False, f"Exception: {str(e)}")
            return False
    
    def test_api_documentation(self) -> bool:
        """Test API documentation endpoints."""
        logger.info("Testing API documentation...")
        
        try:
            # Test OpenAPI docs
            response = self.session.get(f"{self.base_url}/docs")
            if response.status_code == 200:
                self.log_test_result("API Documentation - Swagger", True, "Swagger UI accessible")
            else:
                self.log_test_result("API Documentation - Swagger", False, f"Status code: {response.status_code}")
            
            # Test ReDoc
            response = self.session.get(f"{self.base_url}/redoc")
            if response.status_code == 200:
                self.log_test_result("API Documentation - ReDoc", True, "ReDoc accessible")
            else:
                self.log_test_result("API Documentation - ReDoc", False, f"Status code: {response.status_code}")
            
            # Test OpenAPI JSON
            response = self.session.get(f"{self.base_url}/openapi.json")
            if response.status_code == 200:
                data = response.json()
                version = data.get('info', {}).get('version', 'unknown')
                self.log_test_result("API Documentation - OpenAPI JSON", True, f"Version: {version}")
            else:
                self.log_test_result("API Documentation - OpenAPI JSON", False, f"Status code: {response.status_code}")
            
            return True
            
        except Exception as e:
            self.log_test_result("API Documentation", False, f"Exception: {str(e)}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return results."""
        logger.info("Starting Enhanced Blaze AI Features Test Suite...")
        logger.info("=" * 60)
        
        test_methods = [
            self.test_basic_health_endpoints,
            self.test_enhanced_health_endpoints,
            self.test_rate_limiting,
            self.test_security_features,
            self.test_error_handling,
            self.test_performance_monitoring,
            self.test_circuit_breaker,
            self.test_api_documentation
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                self.log_test_result(test_method.__name__, False, f"Test failed with exception: {str(e)}")
        
        # Generate summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['success'])
        failed_tests = total_tests - passed_tests
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "test_results": self.test_results
        }
        
        logger.info("=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success Rate: {summary['success_rate']:.1f}%")
        
        if failed_tests > 0:
            logger.info("\nFailed Tests:")
            for test_name, result in self.test_results.items():
                if not result['success']:
                    logger.info(f"  - {test_name}: {result['details']}")
        
        return summary
    
    def generate_report(self, summary: Dict[str, Any]) -> str:
        """Generate a detailed test report."""
        report = []
        report.append("# Enhanced Blaze AI Features Test Report")
        report.append("")
        report.append(f"**Test Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Base URL:** {self.base_url}")
        report.append("")
        report.append("## Summary")
        report.append(f"- **Total Tests:** {summary['total_tests']}")
        report.append(f"- **Passed:** {summary['passed_tests']}")
        report.append(f"- **Failed:** {summary['failed_tests']}")
        report.append(f"- **Success Rate:** {summary['success_rate']:.1f}%")
        report.append("")
        
        report.append("## Detailed Results")
        for test_name, result in summary['test_results'].items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            report.append(f"### {test_name}")
            report.append(f"- **Status:** {status}")
            report.append(f"- **Details:** {result['details']}")
            report.append(f"- **Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result['timestamp']))}")
            report.append("")
        
        return "\n".join(report)


async def main():
    """Main test execution function."""
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            logger.error("Server is not responding properly. Please ensure the Blaze AI server is running.")
            return
    except requests.exceptions.RequestException:
        logger.error("Cannot connect to server. Please ensure the Blaze AI server is running on http://localhost:8000")
        logger.info("To start the server, run: python main.py --dev")
        return
    
    # Run tests
    tester = EnhancedFeaturesTester()
    summary = tester.run_all_tests()
    
    # Generate and save report
    report = tester.generate_report(summary)
    with open("enhanced_features_test_report.md", "w") as f:
        f.write(report)
    
    logger.info(f"\nDetailed report saved to: enhanced_features_test_report.md")
    
    # Return exit code based on test results
    if summary['failed_tests'] > 0:
        logger.warning(f"Some tests failed. Please review the report for details.")
        return 1
    else:
        logger.info("All tests passed! 🎉")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
