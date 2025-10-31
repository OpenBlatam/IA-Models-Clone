"""
Advanced test fixtures and data generators for HeyGen AI.
Provides comprehensive test data and fixtures for consistent testing.
"""

import pytest
import json
import time
import random
import string
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Generator
from dataclasses import dataclass, field
from enum import Enum
import asyncio

# Test data generators
class TestDataGenerator:
    """Advanced test data generator."""
    
    @staticmethod
    def generate_string(length: int = 10, prefix: str = "") -> str:
        """Generate random string."""
        chars = string.ascii_letters + string.digits
        random_str = ''.join(random.choice(chars) for _ in range(length))
        return f"{prefix}{random_str}"
    
    @staticmethod
    def generate_email() -> str:
        """Generate random email."""
        username = TestDataGenerator.generate_string(8)
        domain = TestDataGenerator.generate_string(6)
        return f"{username}@{domain}.com"
    
    @staticmethod
    def generate_phone() -> str:
        """Generate random phone number."""
        return f"+1{random.randint(1000000000, 9999999999)}"
    
    @staticmethod
    def generate_date_range(days: int = 30) -> tuple:
        """Generate date range."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        return start_date, end_date
    
    @staticmethod
    def generate_json_data(complexity: int = 3) -> Dict[str, Any]:
        """Generate complex JSON test data."""
        data = {
            "id": random.randint(1, 10000),
            "name": TestDataGenerator.generate_string(10),
            "email": TestDataGenerator.generate_email(),
            "phone": TestDataGenerator.generate_phone(),
            "created_at": datetime.now().isoformat(),
            "active": random.choice([True, False]),
            "metadata": {
                "version": f"v{random.randint(1, 10)}.{random.randint(0, 9)}",
                "tags": [TestDataGenerator.generate_string(5) for _ in range(3)],
                "settings": {
                    "notifications": random.choice([True, False]),
                    "theme": random.choice(["light", "dark", "auto"]),
                    "language": random.choice(["en", "es", "fr", "de"])
                }
            }
        }
        
        if complexity > 1:
            data["nested_data"] = {
                "items": [
                    {
                        "id": i,
                        "value": TestDataGenerator.generate_string(8),
                        "score": random.uniform(0, 100)
                    }
                    for i in range(random.randint(1, 5))
                ],
                "statistics": {
                    "total": random.randint(100, 10000),
                    "average": random.uniform(0, 100),
                    "trend": random.choice(["up", "down", "stable"])
                }
            }
        
        if complexity > 2:
            data["relationships"] = {
                "parent_id": random.randint(1, 1000) if random.choice([True, False]) else None,
                "children": [random.randint(1, 1000) for _ in range(random.randint(0, 3))],
                "dependencies": {
                    "required": [TestDataGenerator.generate_string(6) for _ in range(2)],
                    "optional": [TestDataGenerator.generate_string(6) for _ in range(3)]
                }
            }
        
        return data

# Test fixtures
@pytest.fixture
def sample_user_data():
    """Fixture providing sample user data."""
    return {
        "id": 1,
        "username": "test_user",
        "email": "test@example.com",
        "full_name": "Test User",
        "created_at": datetime.now().isoformat(),
        "is_active": True,
        "permissions": ["read", "write", "admin"],
        "profile": {
            "bio": "Test user profile",
            "avatar_url": "https://example.com/avatar.jpg",
            "timezone": "UTC"
        }
    }

@pytest.fixture
def sample_service_data():
    """Fixture providing sample service data."""
    return {
        "name": "test-service",
        "version": "1.0.0",
        "status": "running",
        "port": 8080,
        "dependencies": ["database", "cache", "queue"],
        "config": {
            "max_connections": 100,
            "timeout": 30,
            "retry_attempts": 3
        },
        "metrics": {
            "cpu_usage": 45.2,
            "memory_usage": 67.8,
            "request_count": 1234
        }
    }

@pytest.fixture
def sample_api_request():
    """Fixture providing sample API request data."""
    return {
        "method": "POST",
        "url": "/api/v1/test",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer test_token",
            "User-Agent": "TestClient/1.0"
        },
        "body": {
            "action": "create",
            "data": TestDataGenerator.generate_json_data(2)
        },
        "timestamp": datetime.now().isoformat()
    }

@pytest.fixture
def sample_error_data():
    """Fixture providing sample error data."""
    return {
        "error_code": "TEST_ERROR_001",
        "error_message": "Test error occurred",
        "error_type": "ValidationError",
        "stack_trace": [
            "File 'test.py', line 10, in test_function",
            "File 'test.py', line 5, in helper_function"
        ],
        "context": {
            "user_id": 123,
            "request_id": "req_456",
            "timestamp": datetime.now().isoformat()
        },
        "severity": "medium"
    }

# Performance test fixtures
@pytest.fixture
def performance_test_data():
    """Fixture providing data for performance tests."""
    return {
        "small_dataset": [TestDataGenerator.generate_string(10) for _ in range(100)],
        "medium_dataset": [TestDataGenerator.generate_string(20) for _ in range(1000)],
        "large_dataset": [TestDataGenerator.generate_string(50) for _ in range(10000)],
        "json_objects": [TestDataGenerator.generate_json_data(2) for _ in range(100)],
        "nested_objects": [TestDataGenerator.generate_json_data(3) for _ in range(50)]
    }

# Async test fixtures
@pytest.fixture
def async_test_data():
    """Fixture providing data for async tests."""
    return {
        "async_operations": [
            {"name": "fetch_data", "delay": 0.01},
            {"name": "process_data", "delay": 0.02},
            {"name": "save_data", "delay": 0.01}
        ],
        "concurrent_requests": 10,
        "timeout": 5.0
    }

# Mock fixtures
@pytest.fixture
def mock_database():
    """Fixture providing mock database."""
    class MockDatabase:
        def __init__(self):
            self.data = {}
            self.queries = []
        
        def insert(self, table: str, data: Dict[str, Any]) -> str:
            record_id = TestDataGenerator.generate_string(8)
            self.data[record_id] = {"table": table, "data": data}
            self.queries.append(f"INSERT INTO {table}")
            return record_id
        
        def select(self, table: str, where: Dict[str, Any] = None) -> List[Dict[str, Any]]:
            self.queries.append(f"SELECT FROM {table}")
            return [record for record in self.data.values() 
                   if record["table"] == table and 
                   (not where or all(record["data"].get(k) == v for k, v in where.items()))]
        
        def update(self, table: str, data: Dict[str, Any], where: Dict[str, Any]) -> int:
            self.queries.append(f"UPDATE {table}")
            updated = 0
            for record in self.data.values():
                if record["table"] == table and all(record["data"].get(k) == v for k, v in where.items()):
                    record["data"].update(data)
                    updated += 1
            return updated
        
        def delete(self, table: str, where: Dict[str, Any]) -> int:
            self.queries.append(f"DELETE FROM {table}")
            deleted = 0
            to_delete = []
            for record_id, record in self.data.items():
                if record["table"] == table and all(record["data"].get(k) == v for k, v in where.items()):
                    to_delete.append(record_id)
                    deleted += 1
            for record_id in to_delete:
                del self.data[record_id]
            return deleted
    
    return MockDatabase()

@pytest.fixture
def mock_api_client():
    """Fixture providing mock API client."""
    class MockAPIClient:
        def __init__(self):
            self.responses = {}
            self.requests = []
        
        def set_response(self, endpoint: str, response: Dict[str, Any], status_code: int = 200):
            self.responses[endpoint] = {"data": response, "status_code": status_code}
        
        async def get(self, endpoint: str) -> Dict[str, Any]:
            self.requests.append({"method": "GET", "endpoint": endpoint})
            await asyncio.sleep(0.01)  # Simulate network delay
            return self.responses.get(endpoint, {"data": {}, "status_code": 404})
        
        async def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
            self.requests.append({"method": "POST", "endpoint": endpoint, "data": data})
            await asyncio.sleep(0.01)  # Simulate network delay
            return self.responses.get(endpoint, {"data": {}, "status_code": 404})
        
        def get_request_count(self) -> int:
            return len(self.requests)
        
        def get_requests(self) -> List[Dict[str, Any]]:
            return self.requests.copy()
    
    return MockAPIClient()

# Test configuration fixtures
@pytest.fixture
def test_config():
    """Fixture providing test configuration."""
    return {
        "test_mode": True,
        "debug": True,
        "timeout": 30,
        "retry_attempts": 3,
        "log_level": "DEBUG",
        "database_url": "sqlite:///:memory:",
        "api_base_url": "http://localhost:8000",
        "test_data_size": "small"
    }

@pytest.fixture
def test_environment():
    """Fixture setting up test environment."""
    import os
    
    # Set test environment variables
    os.environ["TEST_MODE"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"
    
    yield
    
    # Cleanup
    os.environ.pop("TEST_MODE", None)
    os.environ.pop("LOG_LEVEL", None)
    os.environ.pop("DATABASE_URL", None)

# Parametrized fixtures
@pytest.fixture(params=[1, 10, 100, 1000])
def dataset_size(request):
    """Parametrized fixture for different dataset sizes."""
    return request.param

@pytest.fixture(params=["small", "medium", "large"])
def data_complexity(request):
    """Parametrized fixture for different data complexities."""
    return request.param

@pytest.fixture(params=[0.01, 0.1, 1.0])
def async_delay(request):
    """Parametrized fixture for different async delays."""
    return request.param

# Test data factories
class TestDataFactory:
    """Factory for creating test data."""
    
    @staticmethod
    def create_user(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create user test data."""
        data = {
            "id": random.randint(1, 10000),
            "username": TestDataGenerator.generate_string(8),
            "email": TestDataGenerator.generate_email(),
            "full_name": TestDataGenerator.generate_string(15),
            "created_at": datetime.now().isoformat(),
            "is_active": True,
            "permissions": ["read"],
            "profile": {
                "bio": TestDataGenerator.generate_string(50),
                "avatar_url": f"https://example.com/{TestDataGenerator.generate_string(10)}.jpg",
                "timezone": random.choice(["UTC", "EST", "PST", "GMT"])
            }
        }
        
        if overrides:
            data.update(overrides)
        
        return data
    
    @staticmethod
    def create_service(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create service test data."""
        data = {
            "name": f"test-service-{TestDataGenerator.generate_string(6)}",
            "version": f"{random.randint(1, 10)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
            "status": random.choice(["running", "stopped", "starting", "error"]),
            "port": random.randint(8000, 9000),
            "dependencies": [TestDataGenerator.generate_string(8) for _ in range(random.randint(1, 5))],
            "config": {
                "max_connections": random.randint(10, 1000),
                "timeout": random.randint(5, 300),
                "retry_attempts": random.randint(1, 10)
            },
            "metrics": {
                "cpu_usage": random.uniform(0, 100),
                "memory_usage": random.uniform(0, 100),
                "request_count": random.randint(0, 100000)
            }
        }
        
        if overrides:
            data.update(overrides)
        
        return data
    
    @staticmethod
    def create_batch_data(count: int, data_type: str = "user") -> List[Dict[str, Any]]:
        """Create batch test data."""
        factory_methods = {
            "user": TestDataFactory.create_user,
            "service": TestDataFactory.create_service
        }
        
        factory_method = factory_methods.get(data_type, TestDataFactory.create_user)
        return [factory_method() for _ in range(count)]

# Test utilities
class TestUtilities:
    """Utility functions for tests."""
    
    @staticmethod
    def assert_dict_contains(actual: Dict[str, Any], expected: Dict[str, Any]):
        """Assert that actual dict contains all expected keys and values."""
        for key, value in expected.items():
            assert key in actual, f"Key '{key}' not found in actual dict"
            assert actual[key] == value, f"Value for key '{key}' doesn't match. Expected: {value}, Actual: {actual[key]}"
    
    @staticmethod
    def assert_list_contains(list_data: List[Any], expected_item: Any):
        """Assert that list contains expected item."""
        assert expected_item in list_data, f"Expected item {expected_item} not found in list"
    
    @staticmethod
    def assert_performance(execution_time: float, max_time: float):
        """Assert that execution time is within acceptable limits."""
        assert execution_time <= max_time, f"Execution time {execution_time:.3f}s exceeds maximum {max_time:.3f}s"
    
    @staticmethod
    def measure_time(func):
        """Decorator to measure function execution time."""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Function {func.__name__} executed in {execution_time:.3f}s")
            return result, execution_time
        return wrapper

# Test markers
pytestmark = pytest.mark.usefixtures("test_environment")

# Example test using fixtures
def test_user_creation(sample_user_data):
    """Test user creation with fixture data."""
    assert sample_user_data["id"] == 1
    assert sample_user_data["username"] == "test_user"
    assert sample_user_data["is_active"] is True
    assert "read" in sample_user_data["permissions"]

def test_service_configuration(sample_service_data):
    """Test service configuration with fixture data."""
    assert sample_service_data["name"] == "test-service"
    assert sample_service_data["status"] == "running"
    assert sample_service_data["port"] == 8080
    assert len(sample_service_data["dependencies"]) == 3

def test_api_request_processing(sample_api_request):
    """Test API request processing with fixture data."""
    assert sample_api_request["method"] == "POST"
    assert sample_api_request["url"] == "/api/v1/test"
    assert "Authorization" in sample_api_request["headers"]
    assert "action" in sample_api_request["body"]

def test_error_handling(sample_error_data):
    """Test error handling with fixture data."""
    assert sample_error_data["error_code"] == "TEST_ERROR_001"
    assert sample_error_data["error_type"] == "ValidationError"
    assert sample_error_data["severity"] == "medium"
    assert len(sample_error_data["stack_trace"]) == 2

def test_performance_with_data(performance_test_data):
    """Test performance with different data sizes."""
    small_data = performance_test_data["small_dataset"]
    medium_data = performance_test_data["medium_dataset"]
    large_data = performance_test_data["large_dataset"]
    
    # Test small dataset performance
    start_time = time.time()
    sorted(small_data)
    small_time = time.time() - start_time
    assert small_time < 0.1, f"Small dataset sorting too slow: {small_time:.3f}s"
    
    # Test medium dataset performance
    start_time = time.time()
    sorted(medium_data)
    medium_time = time.time() - start_time
    assert medium_time < 1.0, f"Medium dataset sorting too slow: {medium_time:.3f}s"
    
    # Test large dataset performance
    start_time = time.time()
    sorted(large_data)
    large_time = time.time() - start_time
    assert large_time < 5.0, f"Large dataset sorting too slow: {large_time:.3f}s"

@pytest.mark.asyncio
async def test_async_operations(async_test_data, mock_api_client):
    """Test async operations with fixtures."""
    # Setup mock responses
    mock_api_client.set_response("/api/data", {"result": "success"})
    mock_api_client.set_response("/api/process", {"processed": True})
    
    # Test async operations
    response1 = await mock_api_client.get("/api/data")
    response2 = await mock_api_client.post("/api/process", {"data": "test"})
    
    assert response1["data"]["result"] == "success"
    assert response2["data"]["processed"] is True
    assert mock_api_client.get_request_count() == 2

def test_database_operations(mock_database):
    """Test database operations with mock."""
    # Insert test data
    user_id = mock_database.insert("users", {"name": "Test User", "email": "test@example.com"})
    assert user_id is not None
    
    # Query test data
    users = mock_database.select("users", {"name": "Test User"})
    assert len(users) == 1
    assert users[0]["data"]["email"] == "test@example.com"
    
    # Update test data
    updated = mock_database.update("users", {"email": "updated@example.com"}, {"name": "Test User"})
    assert updated == 1
    
    # Verify update
    users = mock_database.select("users", {"name": "Test User"})
    assert users[0]["data"]["email"] == "updated@example.com"

def test_data_factory():
    """Test data factory functionality."""
    # Test user creation
    user = TestDataFactory.create_user({"username": "custom_user"})
    assert user["username"] == "custom_user"
    assert "email" in user
    assert "id" in user
    
    # Test service creation
    service = TestDataFactory.create_service({"name": "custom_service"})
    assert service["name"] == "custom_service"
    assert "version" in service
    assert "status" in service
    
    # Test batch creation
    users = TestDataFactory.create_batch_data(5, "user")
    assert len(users) == 5
    assert all("username" in user for user in users)

def test_utilities():
    """Test utility functions."""
    # Test dict contains
    actual = {"a": 1, "b": 2, "c": 3}
    expected = {"a": 1, "b": 2}
    TestUtilities.assert_dict_contains(actual, expected)
    
    # Test list contains
    data = [1, 2, 3, 4, 5]
    TestUtilities.assert_list_contains(data, 3)
    
    # Test performance assertion
    TestUtilities.assert_performance(0.05, 0.1)

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
