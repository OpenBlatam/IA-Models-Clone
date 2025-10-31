"""
BUL System Performance Tests
===========================

Locust performance tests for the BUL API endpoints.
"""

from locust import HttpUser, task, between
import json
import random
from datetime import datetime

class BULUser(HttpUser):
    """Locust user class for BUL API performance testing"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a user starts"""
        self.client.verify = False  # Disable SSL verification for testing
    
    @task(3)
    def health_check(self):
        """Test health endpoint (high frequency)"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed with status {response.status_code}")
    
    @task(2)
    def get_agents(self):
        """Test get agents endpoint"""
        with self.client.get("/agents", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "agents" in data:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"Get agents failed with status {response.status_code}")
    
    @task(1)
    def get_stats(self):
        """Test get stats endpoint"""
        with self.client.get("/stats", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "documents_generated" in data:
                    response.success()
                else:
                    response.failure("Invalid stats response format")
            else:
                response.failure(f"Get stats failed with status {response.status_code}")
    
    @task(5)
    def generate_document(self):
        """Test document generation endpoint (main functionality)"""
        # Sample requests for different business areas and document types
        sample_requests = [
            {
                "query": "Create a comprehensive marketing plan for launching a new mobile app in the competitive fintech market",
                "business_area": "marketing",
                "document_type": "plan",
                "language": "es",
                "format": "markdown"
            },
            {
                "query": "Generate a detailed sales proposal for enterprise software solution targeting mid-size companies",
                "business_area": "sales",
                "document_type": "proposal",
                "language": "en",
                "format": "html"
            },
            {
                "query": "Create a financial report analyzing Q3 performance and forecasting Q4 projections",
                "business_area": "finance",
                "document_type": "report",
                "language": "es",
                "format": "markdown"
            },
            {
                "query": "Draft an employee handbook section covering remote work policies and procedures",
                "business_area": "hr",
                "document_type": "manual",
                "language": "es",
                "format": "markdown"
            },
            {
                "query": "Generate a service level agreement template for cloud infrastructure services",
                "business_area": "legal",
                "document_type": "contract",
                "language": "en",
                "format": "markdown"
            }
        ]
        
        # Select random request
        request_data = random.choice(sample_requests)
        
        with self.client.post("/generate", 
                            json=request_data,
                            catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "success" in data:
                    if data["success"]:
                        response.success()
                    else:
                        response.failure(f"Document generation failed: {data.get('error_message', 'Unknown error')}")
                else:
                    response.failure("Invalid response format")
            elif response.status_code == 429:
                response.failure("Rate limited")
            else:
                response.failure(f"Generate document failed with status {response.status_code}")
    
    @task(1)
    def generate_document_with_context(self):
        """Test document generation with additional context"""
        request_data = {
            "query": "Create a technical specification document for a new API integration",
            "business_area": "it",
            "document_type": "manual",
            "language": "en",
            "format": "markdown",
            "context": "The API should support RESTful operations with OAuth2 authentication",
            "requirements": [
                "Must support CRUD operations",
                "Should include rate limiting",
                "Must be documented with OpenAPI 3.0",
                "Should include error handling"
            ]
        }
        
        with self.client.post("/generate",
                            json=request_data,
                            catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    response.success()
                else:
                    response.failure(f"Document generation with context failed: {data.get('error_message', 'Unknown error')}")
            else:
                response.failure(f"Generate document with context failed with status {response.status_code}")
    
    @task(1)
    def generate_document_large_request(self):
        """Test document generation with large request"""
        large_request = {
            "query": "Create a comprehensive business plan for a startup in the AI/ML space" + " with detailed market analysis" * 10,
            "business_area": "strategy",
            "document_type": "plan",
            "language": "es",
            "format": "markdown",
            "context": "This is a detailed context for the business plan. " * 50,
            "requirements": [f"Requirement {i}: Detailed requirement description" for i in range(20)]
        }
        
        with self.client.post("/generate",
                            json=large_request,
                            catch_response=True) as response:
            if response.status_code in [200, 413]:  # 413 if payload too large
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        response.success()
                    else:
                        response.failure(f"Large document generation failed: {data.get('error_message', 'Unknown error')}")
                else:
                    response.failure("Request payload too large")
            else:
                response.failure(f"Large document generation failed with status {response.status_code}")

class BULHeavyUser(HttpUser):
    """Heavy user class for stress testing"""
    
    wait_time = between(0.5, 1.5)  # Faster requests for stress testing
    weight = 1  # Lower weight (fewer heavy users)
    
    def on_start(self):
        """Called when a heavy user starts"""
        self.client.verify = False
    
    @task(10)
    def rapid_document_generation(self):
        """Rapid document generation for stress testing"""
        quick_requests = [
            {
                "query": "Create a quick email template for customer follow-up",
                "business_area": "marketing",
                "document_type": "email",
                "language": "es",
                "format": "markdown"
            },
            {
                "query": "Generate a brief project status report",
                "business_area": "operations",
                "document_type": "report",
                "language": "en",
                "format": "markdown"
            },
            {
                "query": "Create a simple user guide for new employees",
                "business_area": "hr",
                "document_type": "manual",
                "language": "es",
                "format": "markdown"
            }
        ]
        
        request_data = random.choice(quick_requests)
        
        with self.client.post("/generate",
                            json=request_data,
                            catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    response.success()
                else:
                    response.failure(f"Rapid generation failed: {data.get('error_message', 'Unknown error')}")
            elif response.status_code == 429:
                response.failure("Rate limited during stress test")
            else:
                response.failure(f"Rapid generation failed with status {response.status_code}")

class BULReadOnlyUser(HttpUser):
    """Read-only user class for testing read operations"""
    
    wait_time = between(0.5, 2)
    weight = 2  # More read-only users
    
    def on_start(self):
        """Called when a read-only user starts"""
        self.client.verify = False
    
    @task(5)
    def health_check(self):
        """Frequent health checks"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed with status {response.status_code}")
    
    @task(3)
    def get_agents(self):
        """Get agents information"""
        with self.client.get("/agents", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Get agents failed with status {response.status_code}")
    
    @task(2)
    def get_stats(self):
        """Get system statistics"""
        with self.client.get("/stats", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Get stats failed with status {response.status_code}")
    
    @task(1)
    def get_docs(self):
        """Access API documentation"""
        with self.client.get("/docs", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Get docs failed with status {response.status_code}")

# Custom metrics for monitoring
class BULMetrics:
    """Custom metrics for BUL system monitoring"""
    
    def __init__(self):
        self.document_generation_times = []
        self.error_rates = {}
        self.response_times = {}
    
    def record_document_generation_time(self, time_ms):
        """Record document generation time"""
        self.document_generation_times.append(time_ms)
    
    def record_error(self, endpoint, error_type):
        """Record error occurrence"""
        key = f"{endpoint}_{error_type}"
        self.error_rates[key] = self.error_rates.get(key, 0) + 1
    
    def record_response_time(self, endpoint, time_ms):
        """Record response time for endpoint"""
        if endpoint not in self.response_times:
            self.response_times[endpoint] = []
        self.response_times[endpoint].append(time_ms)
    
    def get_average_generation_time(self):
        """Get average document generation time"""
        if not self.document_generation_times:
            return 0
        return sum(self.document_generation_times) / len(self.document_generation_times)
    
    def get_error_rate(self, endpoint):
        """Get error rate for specific endpoint"""
        total_errors = sum(count for key, count in self.error_rates.items() if key.startswith(endpoint))
        return total_errors
    
    def get_average_response_time(self, endpoint):
        """Get average response time for endpoint"""
        if endpoint not in self.response_times or not self.response_times[endpoint]:
            return 0
        return sum(self.response_times[endpoint]) / len(self.response_times[endpoint])

# Global metrics instance
metrics = BULMetrics()

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "health_check_max_time": 100,  # 100ms
    "document_generation_max_time": 15000,  # 15 seconds
    "agents_endpoint_max_time": 500,  # 500ms
    "stats_endpoint_max_time": 1000,  # 1 second
    "max_error_rate": 0.05,  # 5%
    "max_response_time_p95": 10000,  # 10 seconds
}

def check_performance_thresholds():
    """Check if performance meets thresholds"""
    violations = []
    
    # Check document generation time
    avg_gen_time = metrics.get_average_generation_time()
    if avg_gen_time > PERFORMANCE_THRESHOLDS["document_generation_max_time"]:
        violations.append(f"Document generation time {avg_gen_time}ms exceeds threshold {PERFORMANCE_THRESHOLDS['document_generation_max_time']}ms")
    
    # Check error rates
    for endpoint in ["/generate", "/health", "/agents", "/stats"]:
        error_rate = metrics.get_error_rate(endpoint)
        if error_rate > PERFORMANCE_THRESHOLDS["max_error_rate"]:
            violations.append(f"Error rate for {endpoint} {error_rate} exceeds threshold {PERFORMANCE_THRESHOLDS['max_error_rate']}")
    
    return violations