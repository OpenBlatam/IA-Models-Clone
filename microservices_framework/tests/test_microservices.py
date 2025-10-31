"""
Comprehensive Test Suite for Microservices Framework
Tests: Service discovery, circuit breakers, caching, security, observability
"""

import asyncio
import pytest
import json
import time
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch

# Test imports
from httpx import AsyncClient
from fastapi.testclient import TestClient

# Framework imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.core.service_registry import ServiceRegistry, ServiceInstance, ServiceType, ServiceStatus
from shared.core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError
from shared.caching.cache_manager import CacheManager, CacheConfig, CacheLevel, InMemoryCacheBackend
from shared.security.security_manager import SecurityManager, SecurityConfig, User
from shared.messaging.message_broker import MessageBrokerFactory, BrokerConfig, MessageBrokerType, Message
from shared.monitoring.observability import ObservabilityManager, TracingConfig, MetricsConfig

# Service imports
from services.user_service.main import app as user_service_app

class TestServiceRegistry:
    """Test service registry functionality"""
    
    @pytest.fixture
    async def service_registry(self):
        """Create service registry for testing"""
        registry = ServiceRegistry("redis://localhost:6379")
        # Mock Redis for testing
        registry.redis_client = AsyncMock()
        registry.redis_client.ping = AsyncMock(return_value=True)
        registry.redis_client.setex = AsyncMock(return_value=True)
        registry.redis_client.sadd = AsyncMock(return_value=True)
        registry.redis_client.smembers = AsyncMock(return_value=set())
        registry.redis_client.get = AsyncMock(return_value=None)
        return registry
    
    @pytest.mark.asyncio
    async def test_service_registration(self, service_registry):
        """Test service registration"""
        service_instance = ServiceInstance(
            service_id="test-service-1",
            service_name="test-service",
            service_type=ServiceType.API,
            host="localhost",
            port=8000,
            version="1.0.0",
            status=ServiceStatus.HEALTHY,
            health_check_url="http://localhost:8000/health",
            metadata={"test": "data"},
            last_heartbeat=time.time(),
            registered_at=time.time()
        )
        
        result = await service_registry.register_service(service_instance)
        assert result is True
        
        # Verify Redis calls
        service_registry.redis_client.setex.assert_called_once()
        service_registry.redis_client.sadd.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_service_discovery(self, service_registry):
        """Test service discovery"""
        # Mock Redis response
        service_data = {
            "service_id": "test-service-1",
            "service_name": "test-service",
            "service_type": "api",
            "host": "localhost",
            "port": 8000,
            "version": "1.0.0",
            "status": "healthy",
            "health_check_url": "http://localhost:8000/health",
            "metadata": {"test": "data"},
            "last_heartbeat": time.time(),
            "registered_at": time.time()
        }
        
        service_registry.redis_client.smembers = AsyncMock(return_value={b"test-service-1"})
        service_registry.redis_client.get = AsyncMock(return_value=json.dumps(service_data).encode())
        
        services = await service_registry.discover_services("test-service")
        assert len(services) == 1
        assert services[0].service_name == "test-service"
        assert services[0].status == ServiceStatus.HEALTHY

class TestCircuitBreaker:
    """Test circuit breaker functionality"""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker for testing"""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=5.0,
            timeout=1.0,
            max_retries=2
        )
        return CircuitBreaker("test-breaker", config)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_success(self, circuit_breaker):
        """Test successful circuit breaker operation"""
        async def success_func():
            return "success"
        
        result = await circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.state.value == "closed"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure(self, circuit_breaker):
        """Test circuit breaker failure handling"""
        async def failure_func():
            raise Exception("Test failure")
        
        # Should fail and retry
        with pytest.raises(Exception):
            await circuit_breaker.call(failure_func)
        
        # After enough failures, circuit should open
        for _ in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failure_func)
        
        assert circuit_breaker.state.value == "open"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, circuit_breaker):
        """Test circuit breaker recovery"""
        # Open the circuit
        async def failure_func():
            raise Exception("Test failure")
        
        for _ in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failure_func)
        
        assert circuit_breaker.state.value == "open"
        
        # Wait for recovery timeout
        await asyncio.sleep(6)
        
        # Should be in half-open state
        assert circuit_breaker.state.value == "half_open"
        
        # Successful call should close circuit
        async def success_func():
            return "success"
        
        result = await circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.state.value == "closed"

class TestCacheManager:
    """Test cache manager functionality"""
    
    @pytest.fixture
    def cache_manager(self):
        """Create cache manager for testing"""
        config = CacheConfig(
            strategy=CacheStrategy.TTL,
            default_ttl=3600,
            max_size=100
        )
        manager = CacheManager(config)
        
        # Add in-memory backend
        backend = InMemoryCacheBackend(config)
        manager.add_backend(CacheLevel.L1, backend)
        
        return manager
    
    @pytest.mark.asyncio
    async def test_cache_set_get(self, cache_manager):
        """Test cache set and get operations"""
        key = "test-key"
        value = {"test": "data", "number": 123}
        
        # Set value
        result = await cache_manager.set(key, value)
        assert result is True
        
        # Get value
        cached_value = await cache_manager.get(key)
        assert cached_value == value
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self, cache_manager):
        """Test cache expiration"""
        key = "test-key"
        value = "test-value"
        
        # Set with short TTL
        await cache_manager.set(key, value, ttl=1)
        
        # Should be available immediately
        cached_value = await cache_manager.get(key)
        assert cached_value == value
        
        # Wait for expiration
        await asyncio.sleep(2)
        
        # Should be expired
        cached_value = await cache_manager.get(key)
        assert cached_value is None
    
    @pytest.mark.asyncio
    async def test_cache_delete(self, cache_manager):
        """Test cache delete operation"""
        key = "test-key"
        value = "test-value"
        
        # Set value
        await cache_manager.set(key, value)
        
        # Verify it exists
        exists = await cache_manager.exists(key)
        assert exists is True
        
        # Delete value
        result = await cache_manager.delete(key)
        assert result is True
        
        # Verify it's gone
        exists = await cache_manager.exists(key)
        assert exists is False

class TestSecurityManager:
    """Test security manager functionality"""
    
    @pytest.fixture
    def security_config(self):
        """Create security configuration for testing"""
        return SecurityConfig(
            jwt_secret="test-secret-key",
            jwt_algorithm="HS256",
            jwt_expiration=3600,
            rate_limit_enabled=True,
            max_requests_per_minute=10
        )
    
    @pytest.fixture
    def security_manager(self, security_config):
        """Create security manager for testing"""
        return SecurityManager(security_config)
    
    def test_jwt_token_creation(self, security_manager):
        """Test JWT token creation and verification"""
        user = User(
            id="test-user-1",
            username="testuser",
            email="test@example.com",
            roles=["user"],
            permissions=["read"]
        )
        
        # Create token
        token = security_manager.jwt_manager.create_access_token(user)
        assert token is not None
        
        # Verify token
        payload = security_manager.jwt_manager.verify_token(token)
        assert payload["sub"] == user.id
        assert payload["username"] == user.username
        assert payload["roles"] == user.roles
    
    def test_password_hashing(self, security_manager):
        """Test password hashing and verification"""
        password = "test-password"
        
        # Hash password
        hashed = security_manager.jwt_manager.hash_password(password)
        assert hashed != password
        assert len(hashed) > 0
        
        # Verify password
        is_valid = security_manager.jwt_manager.verify_password(password, hashed)
        assert is_valid is True
        
        # Test wrong password
        is_valid = security_manager.jwt_manager.verify_password("wrong-password", hashed)
        assert is_valid is False
    
    def test_input_validation(self, security_manager):
        """Test input validation"""
        # Test valid input
        is_valid, error = security_manager.input_validator.validate_input("normal text")
        assert is_valid is True
        assert error == ""
        
        # Test SQL injection attempt
        is_valid, error = security_manager.input_validator.validate_input("'; DROP TABLE users; --")
        assert is_valid is False
        assert "SQL injection" in error
        
        # Test XSS attempt
        is_valid, error = security_manager.input_validator.validate_input("<script>alert('xss')</script>")
        assert is_valid is False
        assert "XSS" in error

class TestMessageBroker:
    """Test message broker functionality"""
    
    @pytest.fixture
    def broker_config(self):
        """Create broker configuration for testing"""
        return BrokerConfig(
            broker_type=MessageBrokerType.REDIS,
            host="localhost",
            port=6379
        )
    
    @pytest.mark.asyncio
    async def test_message_creation(self):
        """Test message creation"""
        message = Message(
            id="test-message-1",
            topic="test-topic",
            payload={"test": "data"},
            headers={"content-type": "application/json"},
            priority=MessagePriority.NORMAL
        )
        
        assert message.id == "test-message-1"
        assert message.topic == "test-topic"
        assert message.payload == {"test": "data"}
        assert message.priority == MessagePriority.NORMAL

class TestObservability:
    """Test observability functionality"""
    
    @pytest.fixture
    async def observability_manager(self):
        """Create observability manager for testing"""
        config = ObservabilityManager(
            tracing_config=TracingConfig(
                service_name="test-service",
                enabled=False  # Disable for testing
            ),
            metrics_config=MetricsConfig(
                enabled=False  # Disable for testing
            )
        )
        return config
    
    @pytest.mark.asyncio
    async def test_health_checks(self, observability_manager):
        """Test health check functionality"""
        health_status = await observability_manager.get_health_status()
        assert "status" in health_status
        assert "timestamp" in health_status

class TestUserService:
    """Test user service endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(user_service_app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_create_user(self, client):
        """Test user creation endpoint"""
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "full_name": "Test User",
            "age": 25
        }
        
        response = client.post("/users", json=user_data)
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == user_data["username"]
        assert data["email"] == user_data["email"]
        assert "id" in data
    
    def test_get_user(self, client):
        """Test get user endpoint"""
        # First create a user
        user_data = {
            "username": "testuser2",
            "email": "test2@example.com",
            "full_name": "Test User 2",
            "age": 30
        }
        
        create_response = client.post("/users", json=user_data)
        user_id = create_response.json()["id"]
        
        # Then get the user
        response = client.get(f"/users/{user_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == user_id
        assert data["username"] == user_data["username"]
    
    def test_list_users(self, client):
        """Test list users endpoint"""
        response = client.get("/users")
        assert response.status_code == 200
        data = response.json()
        assert "users" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data

class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete microservices workflow"""
        # This would test the complete flow:
        # 1. Service registration
        # 2. Service discovery
        # 3. API Gateway routing
        # 4. Circuit breaker protection
        # 5. Caching
        # 6. Security validation
        # 7. Observability tracking
        
        # For now, this is a placeholder for comprehensive integration tests
        assert True

# Performance tests
class TestPerformance:
    """Performance tests for the microservices framework"""
    
    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test cache performance"""
        config = CacheConfig(max_size=1000)
        manager = CacheManager(config)
        backend = InMemoryCacheBackend(config)
        manager.add_backend(CacheLevel.L1, backend)
        
        # Test bulk operations
        start_time = time.time()
        
        for i in range(1000):
            await manager.set(f"key-{i}", f"value-{i}")
        
        for i in range(1000):
            await manager.get(f"key-{i}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time
        assert duration < 5.0  # 5 seconds for 2000 operations
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_performance(self):
        """Test circuit breaker performance"""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            recovery_timeout=1.0,
            timeout=0.1
        )
        breaker = CircuitBreaker("perf-test", config)
        
        async def fast_func():
            return "success"
        
        # Test many concurrent calls
        start_time = time.time()
        
        tasks = [breaker.call(fast_func) for _ in range(100)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert all(result == "success" for result in results)
        assert duration < 2.0  # Should be fast

# Fixtures for test setup
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
async def cleanup():
    """Cleanup after each test"""
    yield
    # Add any cleanup logic here

# Test configuration
@pytest.fixture(scope="session")
def test_config():
    """Test configuration"""
    return {
        "redis_url": "redis://localhost:6379",
        "jaeger_endpoint": "localhost:14268",
        "test_timeout": 30
    }

# Markers for different test types
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.microservices
]

# Test discovery
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])






























