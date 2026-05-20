#!/usr/bin/env python3
"""
Comprehensive test suite for OS Content System
Includes unit tests, integration tests, and performance tests
"""

import pytest
import asyncio
import time
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch, AsyncMock
import json
import yaml

# Import the modules to test
from config import OSContentConfig, get_config
from logger import OSContentLogger, get_logger, get_performance_logger
from integrated_app import app
from refactored_architecture import RefactoredOSContentApplication

class TestConfig:
    """Test configuration management"""
    
    def test_database_config(self):
        """Test database configuration"""
        config = OSContentConfig()
        assert config.database.host == "localhost"
        assert config.database.port == 5432
        assert config.database.connection_string.startswith("postgresql://")
    
    def test_redis_config(self):
        """Test Redis configuration"""
        config = OSContentConfig()
        assert config.redis.host == "localhost"
        assert config.redis.port == 6379
        assert config.redis.connection_string.startswith("redis://")
    
    def test_api_config(self):
        """Test API configuration"""
        config = OSContentConfig()
        assert config.api.host == "0.0.0.0"
        assert config.api.port == 8000
        assert config.api.workers == 1
    
    def test_security_config_validation(self):
        """Test security configuration validation"""
        with pytest.raises(ValueError):
            OSContentConfig(security={"secret_key": "short"})
    
    def test_config_to_dict(self):
        """Test configuration serialization"""
        config = OSContentConfig()
        config_dict = config.to_dict()
        assert "environment" in config_dict
        assert "database" in config_dict
        assert "api" in config_dict
    
    def test_config_save_load(self):
        """Test configuration save and load"""
        config = OSContentConfig()
        
        # Test YAML
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            yaml_path = f.name
        
        try:
            config.save_to_file(yaml_path, "yaml")
            loaded_config = OSContentConfig.load_from_file(yaml_path)
            assert loaded_config.environment == config.environment
        finally:
            Path(yaml_path).unlink(missing_ok=True)
        
        # Test JSON
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name
        
        try:
            config.save_to_file(json_path, "json")
            loaded_config = OSContentConfig.load_from_file(json_path)
            assert loaded_config.environment == config.environment
        finally:
            Path(json_path).unlink(missing_ok=True)

class TestLogger:
    """Test logging system"""
    
    def test_logger_creation(self):
        """Test logger creation"""
        logger = OSContentLogger("test_logger")
        assert logger.name == "test_logger"
        assert logger.logger is not None
    
    def test_performance_logger(self):
        """Test performance logger"""
        logger = OSContentLogger("test_perf")
        perf_logger = logger.get_performance_logger()
        
        perf_logger.start_timer("test_operation")
        time.sleep(0.1)  # Simulate some work
        perf_logger.end_timer("test_operation")
    
    def test_log_with_context(self):
        """Test logging with context"""
        logger = OSContentLogger("test_context")
        logger.log_with_context("INFO", "Test message", user_id=123, action="test")
    
    def test_api_request_logging(self):
        """Test API request logging"""
        logger = OSContentLogger("test_api")
        logger.log_api_request("GET", "/api/test", 200, 0.15, user_id=123)
    
    def test_database_operation_logging(self):
        """Test database operation logging"""
        logger = OSContentLogger("test_db")
        logger.log_database_operation("SELECT", "users", 0.05, rows_returned=10)
    
    def test_ml_operation_logging(self):
        """Test ML operation logging"""
        logger = OSContentLogger("test_ml")
        logger.log_ml_operation("inference", "bert-base", 1.25, input_size=512)

class TestRefactoredArchitecture:
    """Test refactored architecture"""
    
    @pytest.mark.asyncio
    async def test_application_initialization(self):
        """Test application initialization"""
        app = RefactoredOSContentApplication()
        await app.initialize()
        assert app is not None
    
    @pytest.mark.asyncio
    async def test_application_shutdown(self):
        """Test application shutdown"""
        app = RefactoredOSContentApplication()
        await app.initialize()
        await app.shutdown()

class TestFastAPIApp:
    """Test FastAPI application"""
    
    def test_app_creation(self):
        """Test FastAPI app creation"""
        assert app is not None
        assert hasattr(app, 'routes')
    
    def test_app_info(self):
        """Test app information"""
        assert app.title == "OS Content System"
        assert app.version == "1.0.0"

class TestPerformance:
    """Performance tests"""
    
    def test_config_loading_performance(self):
        """Test configuration loading performance"""
        start_time = time.time()
        for _ in range(100):
            config = OSContentConfig()
        end_time = time.time()
        
        # Should load 100 configs in less than 1 second
        assert (end_time - start_time) < 1.0
    
    def test_logger_performance(self):
        """Test logger performance"""
        logger = OSContentLogger("perf_test")
        
        start_time = time.time()
        for i in range(1000):
            logger.logger.info(f"Test message {i}")
        end_time = time.time()
        
        # Should log 1000 messages in less than 2 seconds
        assert (end_time - start_time) < 2.0

class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_full_system_integration(self):
        """Test full system integration"""
        # Create configuration
        config = OSContentConfig()
        
        # Create logger
        logger = OSContentLogger("integration_test")
        
        # Create application
        app = RefactoredOSContentApplication()
        
        # Initialize system
        await app.initialize()
        
        # Test logging
        logger.logger.info("Integration test successful")
        
        # Shutdown system
        await app.shutdown()
        
        assert True  # If we get here, integration test passed

class TestErrorHandling:
    """Test error handling"""
    
    def test_invalid_config_handling(self):
        """Test handling of invalid configuration"""
        with pytest.raises(ValueError):
            OSContentConfig(monitoring={"log_level": "INVALID"})
    
    def test_logger_error_handling(self):
        """Test logger error handling"""
        logger = OSContentLogger("error_test")
        
        # Test logging with invalid level
        logger.log_with_context("INVALID_LEVEL", "Test message")
        # Should not raise exception, just log as INFO
    
    @pytest.mark.asyncio
    async def test_application_error_handling(self):
        """Test application error handling"""
        with pytest.raises(Exception):
            # This should raise an exception due to missing dependencies
            app = RefactoredOSContentApplication()
            await app.initialize()

# Test fixtures
@pytest.fixture
def temp_config():
    """Temporary configuration for testing"""
    return OSContentConfig()

@pytest.fixture
def temp_logger():
    """Temporary logger for testing"""
    return OSContentLogger("temp_test")

@pytest.fixture
def temp_dir():
    """Temporary directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

# Performance benchmarks
class TestBenchmarks:
    """Performance benchmarks"""
    
    def benchmark_config_creation(self, benchmark):
        """Benchmark configuration creation"""
        def create_config():
            return OSContentConfig()
        
        benchmark(create_config)
    
    def benchmark_logger_creation(self, benchmark):
        """Benchmark logger creation"""
        def create_logger():
            return OSContentLogger("benchmark")
        
        benchmark(create_logger)
    
    def benchmark_logging_operations(self, benchmark):
        """Benchmark logging operations"""
        logger = OSContentLogger("benchmark")
        
        def log_operations():
            for i in range(100):
                logger.logger.info(f"Benchmark message {i}")
        
        benchmark(log_operations)

# Test configuration
pytest_plugins = ["pytest_asyncio"]

def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        if "test_performance" in item.name or "test_benchmark" in item.name:
            item.add_marker(pytest.mark.slow)
        if "test_integration" in item.name:
            item.add_marker(pytest.mark.integration)

if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=.",
        "--cov-report=html",
        "--cov-report=term-missing"
    ])
