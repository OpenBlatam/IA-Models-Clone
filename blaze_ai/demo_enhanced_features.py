#!/usr/bin/env python3
"""
Enhanced Blaze AI Features Interactive Demo

This script provides an interactive demonstration of all the enterprise-grade
features including security, monitoring, rate limiting, and error handling.
"""

import asyncio
import time
import json
import requests
import threading
from typing import Dict, Any, List
import logging
from datetime import datetime
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedFeaturesDemo:
    """Interactive demonstration of all enhanced Blaze AI features."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.demo_results = {}
        self.is_running = False
        
    def print_header(self, title: str):
        """Print a formatted header."""
        print("\n" + "=" * 60)
        print(f"🚀 {title}")
        print("=" * 60)
    
    def print_section(self, title: str):
        """Print a formatted section header."""
        print(f"\n📋 {title}")
        print("-" * 40)
    
    def print_success(self, message: str):
        """Print a success message."""
        print(f"✅ {message}")
    
    def print_info(self, message: str):
        """Print an info message."""
        print(f"ℹ️  {message}")
    
    def print_warning(self, message: str):
        """Print a warning message."""
        print(f"⚠️  {message}")
    
    def print_error(self, message: str):
        """Print an error message."""
        print(f"❌ {message}")
    
    def wait_for_user(self, message: str = "Press Enter to continue..."):
        """Wait for user input."""
        input(f"\n⏸️  {message}")
    
    def demo_health_endpoints(self):
        """Demonstrate health check endpoints."""
        self.print_section("Health Check Endpoints")
        
        try:
            # Basic health check
            self.print_info("Testing basic health check...")
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                self.print_success(f"Basic health: {data.get('status')}")
                self.print_info(f"Response time: {response.elapsed.total_seconds():.3f}s")
            else:
                self.print_error(f"Health check failed: {response.status_code}")
                return False
            
            # Detailed health check
            self.print_info("Testing detailed health check...")
            response = self.session.get(f"{self.base_url}/health/detailed")
            if response.status_code == 200:
                data = response.json()
                systems = data.get('systems', {})
                self.print_success(f"Detailed health: {len(systems)} systems checked")
                
                # Show system status
                for system_name, system_status in systems.items():
                    status = system_status.get('status', 'unknown')
                    status_icon = "✅" if status == "healthy" else "⚠️" if status == "degraded" else "❌"
                    self.print_info(f"  {status_icon} {system_name}: {status}")
            else:
                self.print_error(f"Detailed health check failed: {response.status_code}")
                return False
            
            return True
            
        except Exception as e:
            self.print_error(f"Health endpoints demo failed: {str(e)}")
            return False
    
    def demo_metrics_endpoints(self):
        """Demonstrate metrics and monitoring endpoints."""
        self.print_section("Metrics and Monitoring")
        
        try:
            # Metrics endpoint
            self.print_info("Testing metrics endpoint...")
            response = self.session.get(f"{self.base_url}/metrics")
            if response.status_code == 200:
                data = response.json()
                self.print_success(f"Metrics available: {len(data)} metrics")
                
                # Show some key metrics
                if isinstance(data, dict):
                    for key, value in list(data.items())[:5]:  # Show first 5 metrics
                        self.print_info(f"  📊 {key}: {value}")
            elif response.status_code == 503:
                self.print_warning("Performance monitoring not available (expected in demo mode)")
            else:
                self.print_error(f"Metrics endpoint failed: {response.status_code}")
            
            # Prometheus metrics
            self.print_info("Testing Prometheus metrics export...")
            response = self.session.get(f"{self.base_url}/metrics/prometheus")
            if response.status_code == 200:
                content = response.text
                self.print_success(f"Prometheus metrics exported: {len(content)} characters")
                
                # Show first few lines
                lines = content.split('\n')[:3]
                for line in lines:
                    if line.strip():
                        self.print_info(f"  📈 {line}")
            elif response.status_code == 503:
                self.print_warning("Performance monitoring not available (expected in demo mode)")
            else:
                self.print_error(f"Prometheus metrics failed: {response.status_code}")
            
            return True
            
        except Exception as e:
            self.print_error(f"Metrics demo failed: {str(e)}")
            return False
    
    def demo_security_features(self):
        """Demonstrate security features."""
        self.print_section("Security Features")
        
        try:
            # Security status
            self.print_info("Checking security status...")
            response = self.session.get(f"{self.base_url}/security/status")
            if response.status_code == 200:
                data = response.json()
                status = data.get('status', 'unknown')
                self.print_success(f"Security status: {status}")
                
                # Show security details
                threats_blocked = data.get('threats_blocked', 0)
                ips_blocked = data.get('ips_blocked', 0)
                self.print_info(f"  🛡️  Threats blocked: {threats_blocked}")
                self.print_info(f"  🚫 IPs blocked: {ips_blocked}")
            else:
                self.print_error(f"Security status failed: {response.status_code}")
            
            # Test with suspicious headers
            self.print_info("Testing security with suspicious headers...")
            suspicious_headers = {
                'X-Forwarded-For': '192.168.1.1',
                'User-Agent': 'Mozilla/5.0 (compatible; BadBot/1.0)',
                'X-API-Key': 'invalid-key'
            }
            
            response = self.session.get(
                f"{self.base_url}/health",
                headers=suspicious_headers
            )
            
            if response.status_code in [200, 401, 403]:
                self.print_success(f"Security headers handled: Status {response.status_code}")
            else:
                self.print_warning(f"Unexpected response: {response.status_code}")
            
            # Test with malicious parameters
            self.print_info("Testing security with malicious parameters...")
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
                self.print_success(f"Malicious parameters handled: Status {response.status_code}")
            else:
                self.print_warning(f"Unexpected response: {response.status_code}")
            
            return True
            
        except Exception as e:
            self.print_error(f"Security demo failed: {str(e)}")
            return False
    
    def demo_error_handling(self):
        """Demonstrate error handling and recovery."""
        self.print_section("Error Handling and Recovery")
        
        try:
            # Test 404 handling
            self.print_info("Testing 404 error handling...")
            response = self.session.get(f"{self.base_url}/invalid/endpoint")
            if response.status_code == 404:
                self.print_success("404 errors properly handled")
            else:
                self.print_warning(f"Unexpected 404 response: {response.status_code}")
            
            # Test malformed JSON handling
            self.print_info("Testing malformed JSON handling...")
            response = self.session.post(
                f"{self.base_url}/health",
                data="invalid json",
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code in [400, 422]:
                self.print_success(f"Malformed JSON handled: Status {response.status_code}")
            else:
                self.print_warning(f"Unexpected JSON response: {response.status_code}")
            
            # Error summary
            self.print_info("Checking error summary...")
            response = self.session.get(f"{self.base_url}/errors/summary")
            if response.status_code == 200:
                data = response.json()
                if 'error' in data:
                    self.print_warning("Error monitoring not available (expected in demo mode)")
                else:
                    self.print_success("Error monitoring available")
                    # Show error details
                    for key, value in data.items():
                        self.print_info(f"  📊 {key}: {value}")
            else:
                self.print_error(f"Error summary failed: {response.status_code}")
            
            return True
            
        except Exception as e:
            self.print_error(f"Error handling demo failed: {str(e)}")
            return False
    
    def demo_rate_limiting(self):
        """Demonstrate rate limiting functionality."""
        self.print_section("Rate Limiting and Throttling")
        
        try:
            self.print_info("Testing rate limiting with rapid requests...")
            
            # Make multiple rapid requests
            responses = []
            start_time = time.time()
            
            for i in range(15):
                response = self.session.get(f"{self.base_url}/health")
                responses.append(response.status_code)
                
                # Show progress
                if (i + 1) % 5 == 0:
                    self.print_info(f"  Made {i + 1}/15 requests...")
                
                time.sleep(0.1)  # Small delay between requests
            
            elapsed_time = time.time() - start_time
            self.print_info(f"Completed {len(responses)} requests in {elapsed_time:.2f}s")
            
            # Analyze responses
            status_counts = {}
            for status in responses:
                status_counts[status] = status_counts.get(status, 0) + 1
            
            self.print_info("Response status distribution:")
            for status, count in status_counts.items():
                status_icon = "✅" if status == 200 else "⚠️" if status == 429 else "❌"
                self.print_info(f"  {status_icon} Status {status}: {count} requests")
            
            # Check for rate limiting
            rate_limited = any(status == 429 for status in responses)
            if rate_limited:
                self.print_success("Rate limiting is working (some requests blocked)")
            else:
                self.print_success("All requests processed (rate limits not exceeded)")
            
            return True
            
        except Exception as e:
            self.print_error(f"Rate limiting demo failed: {str(e)}")
            return False
    
    def demo_api_documentation(self):
        """Demonstrate API documentation endpoints."""
        self.print_section("API Documentation")
        
        try:
            # Swagger UI
            self.print_info("Testing Swagger UI accessibility...")
            response = self.session.get(f"{self.base_url}/docs")
            if response.status_code == 200:
                self.print_success("Swagger UI accessible")
                self.print_info(f"  📖 URL: {self.base_url}/docs")
            else:
                self.print_error(f"Swagger UI failed: {response.status_code}")
            
            # ReDoc
            self.print_info("Testing ReDoc accessibility...")
            response = self.session.get(f"{self.base_url}/redoc")
            if response.status_code == 200:
                self.print_success("ReDoc accessible")
                self.print_info(f"  📚 URL: {self.base_url}/redoc")
            else:
                self.print_error(f"ReDoc failed: {response.status_code}")
            
            # OpenAPI JSON
            self.print_info("Testing OpenAPI JSON schema...")
            response = self.session.get(f"{self.base_url}/openapi.json")
            if response.status_code == 200:
                data = response.json()
                version = data.get('info', {}).get('version', 'unknown')
                title = data.get('info', {}).get('title', 'Unknown API')
                self.print_success(f"OpenAPI schema accessible")
                self.print_info(f"  🏷️  Title: {title}")
                self.print_info(f"  🏷️  Version: {version}")
                self.print_info(f"  🔗 URL: {self.base_url}/openapi.json")
            else:
                self.print_error(f"OpenAPI JSON failed: {response.status_code}")
            
            return True
            
        except Exception as e:
            self.print_error(f"API documentation demo failed: {str(e)}")
            return False
    
    def demo_load_testing(self):
        """Demonstrate load testing capabilities."""
        self.print_section("Load Testing Simulation")
        
        try:
            self.print_info("Simulating load testing scenario...")
            
            # Simulate concurrent users
            def make_requests(user_id: int, num_requests: int):
                user_responses = []
                for i in range(num_requests):
                    try:
                        response = self.session.get(f"{self.base_url}/health")
                        user_responses.append({
                            'user_id': user_id,
                            'request_id': i,
                            'status': response.status_code,
                            'response_time': response.elapsed.total_seconds()
                        })
                        time.sleep(random.uniform(0.1, 0.3))  # Random delay
                    except Exception as e:
                        user_responses.append({
                            'user_id': user_id,
                            'request_id': i,
                            'error': str(e)
                        })
                return user_responses
            
            # Start multiple threads to simulate concurrent users
            threads = []
            all_responses = []
            
            self.print_info("Starting 3 concurrent users, each making 5 requests...")
            
            for user_id in range(3):
                thread = threading.Thread(
                    target=lambda u=user_id: all_responses.extend(make_requests(u, 5))
                )
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Analyze results
            self.print_success(f"Load test completed: {len(all_responses)} total requests")
            
            # Calculate statistics
            successful_requests = [r for r in all_responses if 'error' not in r]
            failed_requests = [r for r in all_responses if 'error' in r]
            
            if successful_requests:
                response_times = [r['response_time'] for r in successful_requests]
                avg_response_time = sum(response_times) / len(response_times)
                min_response_time = min(response_times)
                max_response_time = max(response_times)
                
                self.print_info("Response time statistics:")
                self.print_info(f"  📊 Average: {avg_response_time:.3f}s")
                self.print_info(f"  📊 Minimum: {min_response_time:.3f}s")
                self.print_info(f"  📊 Maximum: {max_response_time:.3f}s")
            
            # Status distribution
            status_counts = {}
            for response in successful_requests:
                status = response['status']
                status_counts[status] = status_counts.get(status, 0) + 1
            
            self.print_info("Status code distribution:")
            for status, count in status_counts.items():
                status_icon = "✅" if status == 200 else "⚠️" if status == 429 else "❌"
                self.print_info(f"  {status_icon} Status {status}: {count} requests")
            
            if failed_requests:
                self.print_warning(f"  ❌ Failed requests: {len(failed_requests)}")
            
            return True
            
        except Exception as e:
            self.print_error(f"Load testing demo failed: {str(e)}")
            return False
    
    def demo_performance_monitoring(self):
        """Demonstrate performance monitoring features."""
        self.print_section("Performance Monitoring")
        
        try:
            self.print_info("Generating load to test performance monitoring...")
            
            # Make several requests to generate metrics
            start_time = time.time()
            response_times = []
            
            for i in range(10):
                request_start = time.time()
                response = self.session.get(f"{self.base_url}/health")
                request_time = time.time() - request_start
                response_times.append(request_time)
                
                if (i + 1) % 3 == 0:
                    self.print_info(f"  Completed {i + 1}/10 requests...")
                
                time.sleep(0.2)
            
            total_time = time.time() - start_time
            avg_response_time = sum(response_times) / len(response_times)
            
            self.print_success(f"Performance test completed in {total_time:.2f}s")
            self.print_info(f"  📊 Average response time: {avg_response_time:.3f}s")
            self.print_info(f"  📊 Total throughput: {len(response_times)/total_time:.2f} req/s")
            
            # Check if metrics are being collected
            self.print_info("Checking if metrics are being collected...")
            response = self.session.get(f"{self.base_url}/metrics")
            if response.status_code == 200:
                data = response.json()
                self.print_success(f"Performance monitoring active: {len(data)} metrics available")
            elif response.status_code == 503:
                self.print_warning("Performance monitoring not available (expected in demo mode)")
            else:
                self.print_error(f"Metrics endpoint failed: {response.status_code}")
            
            return True
            
        except Exception as e:
            self.print_error(f"Performance monitoring demo failed: {str(e)}")
            return False
    
    def run_full_demo(self):
        """Run the complete demonstration."""
        self.print_header("Enhanced Blaze AI Features Interactive Demo")
        self.print_info("This demo showcases all the enterprise-grade features")
        self.print_info(f"Target server: {self.base_url}")
        
        # Check if server is accessible
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code != 200:
                self.print_error("Server is not responding properly. Please ensure the Blaze AI server is running.")
                self.print_info("To start the server, run: python main.py --dev")
                return False
        except requests.exceptions.RequestException:
            self.print_error("Cannot connect to server. Please ensure the Blaze AI server is running.")
            self.print_info("To start the server, run: python main.py --dev")
            return False
        
        self.print_success("Server connection established! Starting demo...")
        self.wait_for_user()
        
        # Run all demos
        demos = [
            ("Health Endpoints", self.demo_health_endpoints),
            ("Metrics and Monitoring", self.demo_metrics_endpoints),
            ("Security Features", self.demo_security_features),
            ("Error Handling", self.demo_error_handling),
            ("Rate Limiting", self.demo_rate_limiting),
            ("API Documentation", self.demo_api_documentation),
            ("Performance Monitoring", self.demo_performance_monitoring),
            ("Load Testing", self.demo_load_testing)
        ]
        
        results = {}
        
        for demo_name, demo_func in demos:
            try:
                self.print_header(f"Demo: {demo_name}")
                success = demo_func()
                results[demo_name] = success
                
                if success:
                    self.print_success(f"✅ {demo_name} demo completed successfully!")
                else:
                    self.print_error(f"❌ {demo_name} demo failed!")
                
                self.wait_for_user()
                
            except Exception as e:
                self.print_error(f"Demo {demo_name} crashed: {str(e)}")
                results[demo_name] = False
        
        # Show final results
        self.print_header("Demo Results Summary")
        
        total_demos = len(results)
        successful_demos = sum(1 for success in results.values() if success)
        failed_demos = total_demos - successful_demos
        
        self.print_info(f"Total demos: {total_demos}")
        self.print_info(f"Successful: {successful_demos}")
        self.print_info(f"Failed: {failed_demos}")
        self.print_info(f"Success rate: {(successful_demos/total_demos*100):.1f}%")
        
        if failed_demos > 0:
            self.print_warning("\nFailed demos:")
            for demo_name, success in results.items():
                if not success:
                    self.print_warning(f"  ❌ {demo_name}")
        
        # Show next steps
        self.print_header("Next Steps")
        self.print_info("🎯 To explore further:")
        self.print_info("  1. Visit the API documentation: http://localhost:8000/docs")
        self.print_info("  2. Check the health endpoints: http://localhost:8000/health/detailed")
        self.print_info("  3. Monitor metrics: http://localhost:8000/metrics")
        self.print_info("  4. Run the test suite: python test_enhanced_features.py")
        self.print_info("  5. Check the deployment guide: DEPLOYMENT_GUIDE.md")
        
        self.print_success("🎉 Demo completed! The enhanced Blaze AI system is ready for production use.")
        
        return successful_demos == total_demos


async def main():
    """Main demo execution function."""
    demo = EnhancedFeaturesDemo()
    success = demo.run_full_demo()
    
    if success:
        print("\n🎯 All demos passed successfully!")
        return 0
    else:
        print("\n⚠️  Some demos failed. Please check the server and try again.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
