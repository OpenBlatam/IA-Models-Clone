"""
Simple test configuration and settings for copywriting service tests.
"""
import os
import pytest
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import get_settings


class TestEnvironment(Enum):
    """Test environment types."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    MONITORING = "monitoring"
    LOAD = "load"


class TestCategory(Enum):
    """Test categories."""
    FAST = "fast"
    SLOW = "slow"
    CRITICAL = "critical"
    OPTIONAL = "optional"


@dataclass
class PerformanceThresholds:
    """Performance thresholds for testing."""
    max_response_time: float = 1.0  # seconds
    max_memory_usage: int = 100 * 1024 * 1024  # 100MB
    max_cpu_usage: float = 80.0  # percentage
    min_throughput: int = 10  # requests per second
    max_error_rate: float = 0.01  # 1%


@dataclass
class TestConfig:
    """Test configuration."""
    environment: TestEnvironment = TestEnvironment.UNIT
    category: TestCategory = TestCategory.FAST
    performance_thresholds: PerformanceThresholds = None
    mock_external_services: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"
    parallel_execution: bool = False
    max_workers: int = 4
    timeout: int = 30
    retry_count: int = 3
    
    def __post_init__(self):
        if self.performance_thresholds is None:
            self.performance_thresholds = PerformanceThresholds()


class TestConfigManager:
    """Manages test configuration."""
    
    def __init__(self):
        self.config = TestConfig()
        self._load_environment_config()
    
    def _load_environment_config(self):
        """Load configuration from environment variables."""
        # Load environment
        env = os.getenv("TEST_ENVIRONMENT", "unit")
        if env in [e.value for e in TestEnvironment]:
            self.config.environment = TestEnvironment(env)
        
        # Load category
        category = os.getenv("TEST_CATEGORY", "fast")
        if category in [c.value for c in TestCategory]:
            self.config.category = TestCategory(category)
        
        # Load performance thresholds
        self.config.performance_thresholds.max_response_time = float(
            os.getenv("TEST_MAX_RESPONSE_TIME", "1.0")
        )
        self.config.performance_thresholds.max_memory_usage = int(
            os.getenv("TEST_MAX_MEMORY_USAGE", str(100 * 1024 * 1024))
        )
        self.config.performance_thresholds.max_cpu_usage = float(
            os.getenv("TEST_MAX_CPU_USAGE", "80.0")
        )
        self.config.performance_thresholds.min_throughput = int(
            os.getenv("TEST_MIN_THROUGHPUT", "10")
        )
        self.config.performance_thresholds.max_error_rate = float(
            os.getenv("TEST_MAX_ERROR_RATE", "0.01")
        )
        
        # Load other settings
        self.config.mock_external_services = os.getenv("TEST_MOCK_SERVICES", "true").lower() == "true"
        self.config.enable_logging = os.getenv("TEST_ENABLE_LOGGING", "true").lower() == "true"
        self.config.log_level = os.getenv("TEST_LOG_LEVEL", "INFO")
        self.config.parallel_execution = os.getenv("TEST_PARALLEL", "false").lower() == "true"
        self.config.max_workers = int(os.getenv("TEST_MAX_WORKERS", "4"))
        self.config.timeout = int(os.getenv("TEST_TIMEOUT", "30"))
        self.config.retry_count = int(os.getenv("TEST_RETRY_COUNT", "3"))
    
    def get_config(self) -> TestConfig:
        """Get current test configuration."""
        return self.config
    
    def update_config(self, **kwargs):
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def reset_config(self):
        """Reset configuration to defaults."""
        self.config = TestConfig()
        self._load_environment_config()


class TestConfigTests:
    """Test cases for test configuration."""
    
    @pytest.fixture
    def config_manager(self):
        """Create a test configuration manager."""
        return TestConfigManager()
    
    def test_config_initialization(self, config_manager):
        """Test configuration initialization."""
        config = config_manager.get_config()
        
        assert config is not None
        assert config.environment == TestEnvironment.UNIT
        assert config.category == TestCategory.FAST
        assert config.performance_thresholds is not None
        assert config.mock_external_services is True
        assert config.enable_logging is True
        assert config.log_level == "INFO"
        assert config.parallel_execution is False
        assert config.max_workers == 4
        assert config.timeout == 30
        assert config.retry_count == 3
    
    def test_performance_thresholds(self, config_manager):
        """Test performance thresholds configuration."""
        config = config_manager.get_config()
        thresholds = config.performance_thresholds
        
        assert thresholds.max_response_time == 1.0
        assert thresholds.max_memory_usage == 100 * 1024 * 1024
        assert thresholds.max_cpu_usage == 80.0
        assert thresholds.min_throughput == 10
        assert thresholds.max_error_rate == 0.01
    
    def test_environment_enum(self):
        """Test environment enum values."""
        assert TestEnvironment.UNIT.value == "unit"
        assert TestEnvironment.INTEGRATION.value == "integration"
        assert TestEnvironment.PERFORMANCE.value == "performance"
        assert TestEnvironment.SECURITY.value == "security"
        assert TestEnvironment.MONITORING.value == "monitoring"
        assert TestEnvironment.LOAD.value == "load"
    
    def test_category_enum(self):
        """Test category enum values."""
        assert TestCategory.FAST.value == "fast"
        assert TestCategory.SLOW.value == "slow"
        assert TestCategory.CRITICAL.value == "critical"
        assert TestCategory.OPTIONAL.value == "optional"
    
    def test_config_update(self, config_manager):
        """Test configuration updates."""
        config_manager.update_config(
            environment=TestEnvironment.INTEGRATION,
            category=TestCategory.SLOW,
            timeout=60
        )
        
        config = config_manager.get_config()
        assert config.environment == TestEnvironment.INTEGRATION
        assert config.category == TestCategory.SLOW
        assert config.timeout == 60
    
    def test_config_reset(self, config_manager):
        """Test configuration reset."""
        # Update config
        config_manager.update_config(
            environment=TestEnvironment.PERFORMANCE,
            timeout=120
        )
        
        # Reset config
        config_manager.reset_config()
        
        config = config_manager.get_config()
        assert config.environment == TestEnvironment.UNIT
        assert config.timeout == 30
    
    def test_performance_thresholds_validation(self, config_manager):
        """Test performance thresholds validation."""
        config = config_manager.get_config()
        thresholds = config.performance_thresholds
        
        # Test threshold values are reasonable
        assert thresholds.max_response_time > 0
        assert thresholds.max_memory_usage > 0
        assert 0 <= thresholds.max_cpu_usage <= 100
        assert thresholds.min_throughput > 0
        assert 0 <= thresholds.max_error_rate <= 1
    
    def test_environment_variable_loading(self, config_manager):
        """Test environment variable loading."""
        # Set environment variables
        os.environ["TEST_ENVIRONMENT"] = "integration"
        os.environ["TEST_CATEGORY"] = "slow"
        os.environ["TEST_MAX_RESPONSE_TIME"] = "2.0"
        os.environ["TEST_MOCK_SERVICES"] = "false"
        
        # Create new config manager to load env vars
        new_config_manager = TestConfigManager()
        config = new_config_manager.get_config()
        
        assert config.environment == TestEnvironment.INTEGRATION
        assert config.category == TestCategory.SLOW
        assert config.performance_thresholds.max_response_time == 2.0
        assert config.mock_external_services is False
        
        # Clean up environment variables
        del os.environ["TEST_ENVIRONMENT"]
        del os.environ["TEST_CATEGORY"]
        del os.environ["TEST_MAX_RESPONSE_TIME"]
        del os.environ["TEST_MOCK_SERVICES"]
    
    def test_settings_integration(self, config_manager):
        """Test integration with application settings."""
        # Test that we can get application settings
        settings = get_settings()
        assert settings is not None
        
        # Test that config manager works with settings
        config = config_manager.get_config()
        assert config is not None
        assert config.timeout > 0
    
    def test_config_serialization(self, config_manager):
        """Test configuration serialization."""
        config = config_manager.get_config()
        
        # Test that config can be converted to dict
        config_dict = {
            "environment": config.environment.value,
            "category": config.category.value,
            "mock_external_services": config.mock_external_services,
            "enable_logging": config.enable_logging,
            "log_level": config.log_level,
            "parallel_execution": config.parallel_execution,
            "max_workers": config.max_workers,
            "timeout": config.timeout,
            "retry_count": config.retry_count
        }
        
        assert isinstance(config_dict, dict)
        assert config_dict["environment"] == "unit"
        assert config_dict["category"] == "fast"
        assert config_dict["timeout"] == 30
    
    def test_config_validation(self, config_manager):
        """Test configuration validation."""
        config = config_manager.get_config()
        
        # Test that all required fields are present
        assert hasattr(config, "environment")
        assert hasattr(config, "category")
        assert hasattr(config, "performance_thresholds")
        assert hasattr(config, "mock_external_services")
        assert hasattr(config, "enable_logging")
        assert hasattr(config, "log_level")
        assert hasattr(config, "parallel_execution")
        assert hasattr(config, "max_workers")
        assert hasattr(config, "timeout")
        assert hasattr(config, "retry_count")
        
        # Test that values are of correct types
        assert isinstance(config.environment, TestEnvironment)
        assert isinstance(config.category, TestCategory)
        assert isinstance(config.performance_thresholds, PerformanceThresholds)
        assert isinstance(config.mock_external_services, bool)
        assert isinstance(config.enable_logging, bool)
        assert isinstance(config.log_level, str)
        assert isinstance(config.parallel_execution, bool)
        assert isinstance(config.max_workers, int)
        assert isinstance(config.timeout, int)
        assert isinstance(config.retry_count, int)
    
    def test_performance_thresholds_creation(self):
        """Test performance thresholds creation."""
        thresholds = PerformanceThresholds()
        
        assert thresholds.max_response_time == 1.0
        assert thresholds.max_memory_usage == 100 * 1024 * 1024
        assert thresholds.max_cpu_usage == 80.0
        assert thresholds.min_throughput == 10
        assert thresholds.max_error_rate == 0.01
        
        # Test custom thresholds
        custom_thresholds = PerformanceThresholds(
            max_response_time=2.0,
            max_memory_usage=200 * 1024 * 1024,
            max_cpu_usage=90.0,
            min_throughput=20,
            max_error_rate=0.005
        )
        
        assert custom_thresholds.max_response_time == 2.0
        assert custom_thresholds.max_memory_usage == 200 * 1024 * 1024
        assert custom_thresholds.max_cpu_usage == 90.0
        assert custom_thresholds.min_throughput == 20
        assert custom_thresholds.max_error_rate == 0.005
    
    def test_config_manager_methods(self, config_manager):
        """Test configuration manager methods."""
        # Test get_config
        config = config_manager.get_config()
        assert config is not None
        
        # Test update_config
        original_timeout = config.timeout
        config_manager.update_config(timeout=60)
        updated_config = config_manager.get_config()
        assert updated_config.timeout == 60
        
        # Test reset_config
        config_manager.reset_config()
        reset_config = config_manager.get_config()
        assert reset_config.timeout == original_timeout
    
    def test_config_consistency(self, config_manager):
        """Test configuration consistency."""
        config1 = config_manager.get_config()
        config2 = config_manager.get_config()
        
        # Configs should be the same object
        assert config1 is config2
        
        # Values should be consistent
        assert config1.environment == config2.environment
        assert config1.category == config2.category
        assert config1.timeout == config2.timeout
        assert config1.performance_thresholds.max_response_time == config2.performance_thresholds.max_response_time
