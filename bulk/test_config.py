#!/usr/bin/env python3
"""
Test Configuration for TruthGPT System
======================================

Configuración centralizada para todos los tests del sistema TruthGPT.
Incluye configuraciones para diferentes entornos, timeouts, y parámetros de test.

Características:
- Configuración centralizada
- Diferentes entornos (dev, staging, prod)
- Timeouts configurables
- Parámetros de test personalizables
- Configuración de logging
- Configuración de base de datos de test
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class TestConfig:
    """Configuration for TruthGPT tests."""
    
    # Environment settings
    environment: str = "development"
    base_url: str = "http://localhost:8000"
    api_timeout: int = 30
    
    # Test execution settings
    parallel_execution: bool = False
    max_concurrent_tests: int = 5
    test_timeout: int = 300
    
    # Performance test settings
    performance_test_duration: int = 60
    load_test_concurrent_users: int = 10
    load_test_requests_per_user: int = 5
    stress_test_max_users: int = 50
    stress_test_duration: int = 120
    
    # ML test settings
    ml_test_sample_size: int = 1000
    ml_test_cross_validation_folds: int = 5
    ml_test_random_state: int = 42
    
    # API test settings
    api_test_retry_count: int = 3
    api_test_retry_delay: float = 1.0
    api_test_rate_limit_requests: int = 10
    
    # Database test settings
    test_database_url: Optional[str] = None
    test_database_cleanup: bool = True
    
    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[str] = None
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Report settings
    generate_html_report: bool = True
    generate_json_report: bool = True
    generate_markdown_report: bool = True
    report_output_dir: str = "test_reports"
    
    # Test data settings
    test_data_dir: str = "test_data"
    use_mock_data: bool = False
    mock_data_size: int = 100
    
    # External service settings
    openrouter_api_key: Optional[str] = None
    redis_url: Optional[str] = None
    elasticsearch_url: Optional[str] = None
    
    # Feature flags
    enable_ml_tests: bool = True
    enable_performance_tests: bool = True
    enable_integration_tests: bool = True
    enable_api_tests: bool = True
    enable_load_tests: bool = True
    
    # Test categories
    test_categories: Dict[str, bool] = field(default_factory=lambda: {
        "unit": True,
        "integration": True,
        "performance": True,
        "load": True,
        "api": True,
        "ml": True,
        "end_to_end": True
    })
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Load from environment variables
        self._load_from_env()
        
        # Setup logging
        self._setup_logging()
        
        # Create directories
        self._create_directories()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Environment
        self.environment = os.getenv("TEST_ENVIRONMENT", self.environment)
        self.base_url = os.getenv("TEST_BASE_URL", self.base_url)
        
        # API settings
        self.api_timeout = int(os.getenv("TEST_API_TIMEOUT", self.api_timeout))
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", self.openrouter_api_key)
        
        # Database settings
        self.test_database_url = os.getenv("TEST_DATABASE_URL", self.test_database_url)
        
        # Redis settings
        self.redis_url = os.getenv("REDIS_URL", self.redis_url)
        
        # Elasticsearch settings
        self.elasticsearch_url = os.getenv("ELASTICSEARCH_URL", self.elasticsearch_url)
        
        # Logging settings
        self.log_level = os.getenv("TEST_LOG_LEVEL", self.log_level)
        self.log_file = os.getenv("TEST_LOG_FILE", self.log_file)
        
        # Report settings
        self.report_output_dir = os.getenv("TEST_REPORT_DIR", self.report_output_dir)
        
        # Feature flags
        self.enable_ml_tests = os.getenv("ENABLE_ML_TESTS", "true").lower() == "true"
        self.enable_performance_tests = os.getenv("ENABLE_PERFORMANCE_TESTS", "true").lower() == "true"
        self.enable_integration_tests = os.getenv("ENABLE_INTEGRATION_TESTS", "true").lower() == "true"
        self.enable_api_tests = os.getenv("ENABLE_API_TESTS", "true").lower() == "true"
        self.enable_load_tests = os.getenv("ENABLE_LOAD_TESTS", "true").lower() == "true"
        
        # Test execution
        self.parallel_execution = os.getenv("PARALLEL_TESTS", "false").lower() == "true"
        self.max_concurrent_tests = int(os.getenv("MAX_CONCURRENT_TESTS", self.max_concurrent_tests))
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Create formatter
        formatter = logging.Formatter(self.log_format)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, self.log_level.upper()))
        
        # Setup file handler if specified
        handlers = [console_handler]
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(getattr(logging, self.log_level.upper()))
            handlers.append(file_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            handlers=handlers,
            force=True
        )
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            self.report_output_dir,
            self.test_data_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_test_suites(self) -> Dict[str, Dict[str, Any]]:
        """Get test suite configurations."""
        return {
            "basic_system": {
                "enabled": True,
                "timeout": self.test_timeout,
                "retry_count": self.api_test_retry_count,
                "retry_delay": self.api_test_retry_delay
            },
            "ultimate_system": {
                "enabled": True,
                "timeout": self.test_timeout,
                "retry_count": self.api_test_retry_count,
                "retry_delay": self.api_test_retry_delay
            },
            "advanced_ml": {
                "enabled": self.enable_ml_tests,
                "timeout": self.test_timeout * 2,  # ML tests take longer
                "sample_size": self.ml_test_sample_size,
                "cross_validation_folds": self.ml_test_cross_validation_folds,
                "random_state": self.ml_test_random_state
            },
            "performance_load": {
                "enabled": self.enable_performance_tests,
                "timeout": self.test_timeout * 3,  # Performance tests take longer
                "duration": self.performance_test_duration,
                "concurrent_users": self.load_test_concurrent_users,
                "requests_per_user": self.load_test_requests_per_user,
                "stress_max_users": self.stress_test_max_users,
                "stress_duration": self.stress_test_duration
            },
            "integration": {
                "enabled": self.enable_integration_tests,
                "timeout": self.test_timeout,
                "retry_count": self.api_test_retry_count,
                "retry_delay": self.api_test_retry_delay
            },
            "api_comprehensive": {
                "enabled": self.enable_api_tests,
                "timeout": self.api_timeout,
                "retry_count": self.api_test_retry_count,
                "retry_delay": self.api_test_retry_delay,
                "rate_limit_requests": self.api_test_rate_limit_requests
            }
        }
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration."""
        configs = {
            "development": {
                "base_url": "http://localhost:8000",
                "api_timeout": 30,
                "test_timeout": 300,
                "log_level": "DEBUG",
                "enable_all_tests": True
            },
            "staging": {
                "base_url": "https://staging.truthgpt.com",
                "api_timeout": 60,
                "test_timeout": 600,
                "log_level": "INFO",
                "enable_all_tests": True
            },
            "production": {
                "base_url": "https://api.truthgpt.com",
                "api_timeout": 120,
                "test_timeout": 1200,
                "log_level": "WARNING",
                "enable_all_tests": False,
                "enable_load_tests": False,
                "enable_performance_tests": False
            }
        }
        
        return configs.get(self.environment, configs["development"])
    
    def validate(self) -> bool:
        """Validate configuration."""
        errors = []
        
        # Check required settings
        if not self.base_url:
            errors.append("Base URL is required")
        
        if self.api_timeout <= 0:
            errors.append("API timeout must be positive")
        
        if self.test_timeout <= 0:
            errors.append("Test timeout must be positive")
        
        # Check environment-specific requirements
        if self.environment == "production":
            if self.enable_load_tests:
                errors.append("Load tests should not be enabled in production")
            
            if self.enable_performance_tests:
                errors.append("Performance tests should not be enabled in production")
        
        # Check external service requirements
        if self.enable_ml_tests and not self.openrouter_api_key:
            errors.append("OpenRouter API key is required for ML tests")
        
        if errors:
            for error in errors:
                logging.error(f"Configuration error: {error}")
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "environment": self.environment,
            "base_url": self.base_url,
            "api_timeout": self.api_timeout,
            "parallel_execution": self.parallel_execution,
            "max_concurrent_tests": self.max_concurrent_tests,
            "test_timeout": self.test_timeout,
            "performance_test_duration": self.performance_test_duration,
            "load_test_concurrent_users": self.load_test_concurrent_users,
            "load_test_requests_per_user": self.load_test_requests_per_user,
            "stress_test_max_users": self.stress_test_max_users,
            "stress_test_duration": self.stress_test_duration,
            "ml_test_sample_size": self.ml_test_sample_size,
            "ml_test_cross_validation_folds": self.ml_test_cross_validation_folds,
            "ml_test_random_state": self.ml_test_random_state,
            "api_test_retry_count": self.api_test_retry_count,
            "api_test_retry_delay": self.api_test_retry_delay,
            "api_test_rate_limit_requests": self.api_test_rate_limit_requests,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "generate_html_report": self.generate_html_report,
            "generate_json_report": self.generate_json_report,
            "generate_markdown_report": self.generate_markdown_report,
            "report_output_dir": self.report_output_dir,
            "test_data_dir": self.test_data_dir,
            "use_mock_data": self.use_mock_data,
            "mock_data_size": self.mock_data_size,
            "enable_ml_tests": self.enable_ml_tests,
            "enable_performance_tests": self.enable_performance_tests,
            "enable_integration_tests": self.enable_integration_tests,
            "enable_api_tests": self.enable_api_tests,
            "enable_load_tests": self.enable_load_tests,
            "test_categories": self.test_categories
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TestConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, config_file: str) -> "TestConfig":
        """Load configuration from file."""
        import json
        
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def save_to_file(self, config_file: str):
        """Save configuration to file."""
        import json
        
        with open(config_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

# Global configuration instance
config = TestConfig()

# Environment-specific configurations
DEVELOPMENT_CONFIG = TestConfig(
    environment="development",
    base_url="http://localhost:8000",
    api_timeout=30,
    test_timeout=300,
    log_level="DEBUG",
    enable_all_tests=True
)

STAGING_CONFIG = TestConfig(
    environment="staging",
    base_url="https://staging.truthgpt.com",
    api_timeout=60,
    test_timeout=600,
    log_level="INFO",
    enable_all_tests=True
)

PRODUCTION_CONFIG = TestConfig(
    environment="production",
    base_url="https://api.truthgpt.com",
    api_timeout=120,
    test_timeout=1200,
    log_level="WARNING",
    enable_all_tests=False,
    enable_load_tests=False,
    enable_performance_tests=False
)

def get_config(environment: str = None) -> TestConfig:
    """Get configuration for specific environment."""
    if environment is None:
        environment = os.getenv("TEST_ENVIRONMENT", "development")
    
    configs = {
        "development": DEVELOPMENT_CONFIG,
        "staging": STAGING_CONFIG,
        "production": PRODUCTION_CONFIG
    }
    
    return configs.get(environment, DEVELOPMENT_CONFIG)

def setup_test_environment(environment: str = None):
    """Setup test environment with appropriate configuration."""
    test_config = get_config(environment)
    
    # Validate configuration
    if not test_config.validate():
        raise ValueError("Invalid test configuration")
    
    # Setup logging
    test_config._setup_logging()
    
    # Create directories
    test_config._create_directories()
    
    return test_config

if __name__ == "__main__":
    # Example usage
    config = setup_test_environment("development")
    
    print("Test Configuration:")
    print(f"Environment: {config.environment}")
    print(f"Base URL: {config.base_url}")
    print(f"API Timeout: {config.api_timeout}")
    print(f"Test Timeout: {config.test_timeout}")
    print(f"Parallel Execution: {config.parallel_execution}")
    print(f"Enable ML Tests: {config.enable_ml_tests}")
    print(f"Enable Performance Tests: {config.enable_performance_tests}")
    print(f"Enable Integration Tests: {config.enable_integration_tests}")
    print(f"Enable API Tests: {config.enable_api_tests}")
    print(f"Enable Load Tests: {config.enable_load_tests}")
    
    # Save configuration
    config.save_to_file("test_config.json")
    print("\nConfiguration saved to test_config.json")
























