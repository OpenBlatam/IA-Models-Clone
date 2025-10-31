"""
Centralized test configuration for HeyGen AI system.
Refactored for better maintainability and flexibility.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

class TestEnvironment(Enum):
    """Test environment types."""
    LOCAL = "local"
    CI = "ci"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(Enum):
    """Log level types."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class DatabaseConfig:
    """Database configuration for tests."""
    url: str = "sqlite:///:memory:"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10
    timeout: int = 30

@dataclass
class APIConfig:
    """API configuration for tests."""
    base_url: str = "http://localhost:8000"
    timeout: int = 30
    retry_attempts: int = 3
    headers: Dict[str, str] = field(default_factory=dict)

@dataclass
class PerformanceConfig:
    """Performance test configuration."""
    timeout_threshold: float = 1.0
    memory_threshold: float = 100.0  # MB
    cpu_threshold: float = 80.0  # %
    throughput_threshold: float = 100.0  # ops/s
    max_iterations: int = 1000

@dataclass
class CoverageConfig:
    """Coverage test configuration."""
    min_coverage: float = 80.0  # %
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "*/tests/*",
        "*/test_*",
        "*/__pycache__/*",
        "*/migrations/*"
    ])
    include_patterns: List[str] = field(default_factory=lambda: [
        "*.py"
    ])

@dataclass
class TestConfig:
    """Main test configuration."""
    environment: TestEnvironment = TestEnvironment.LOCAL
    log_level: LogLevel = LogLevel.INFO
    debug_mode: bool = False
    parallel_workers: int = 1
    test_timeout: float = 300.0  # seconds
    retry_failed_tests: bool = True
    max_retries: int = 3
    
    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    coverage: CoverageConfig = field(default_factory=CoverageConfig)
    
    # Test data configuration
    test_data_size: str = "small"  # small, medium, large
    generate_fake_data: bool = True
    data_cleanup: bool = True
    
    # Reporting configuration
    generate_reports: bool = True
    report_formats: List[str] = field(default_factory=lambda: ["json", "html"])
    report_output_dir: str = "test_reports"
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Set environment-specific defaults
        if self.environment == TestEnvironment.CI:
            self.parallel_workers = 4
            self.test_timeout = 600.0
            self.log_level = LogLevel.INFO
        elif self.environment == TestEnvironment.PRODUCTION:
            self.debug_mode = False
            self.log_level = LogLevel.ERROR
            self.generate_fake_data = False

class TestConfigManager:
    """Manages test configuration loading and validation."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or self._find_config_file()
        self.config: Optional[TestConfig] = None
        self._load_config()
    
    def _find_config_file(self) -> str:
        """Find test configuration file."""
        possible_paths = [
            "tests/config/test_config.json",
            "test_config.json",
            "tests/test_config.json",
            ".test_config.json"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
        
        return "tests/config/test_config.json"
    
    def _load_config(self):
        """Load configuration from file or create default."""
        if self.config_file and Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                self.config = self._dict_to_config(config_data)
            except Exception as e:
                print(f"Warning: Could not load config from {self.config_file}: {e}")
                self.config = TestConfig()
        else:
            self.config = TestConfig()
        
        # Override with environment variables
        self._apply_env_overrides()
    
    def _dict_to_config(self, data: Dict[str, Any]) -> TestConfig:
        """Convert dictionary to TestConfig object."""
        # Handle environment
        environment = TestEnvironment(data.get("environment", "local"))
        
        # Handle log level
        log_level = LogLevel(data.get("log_level", "INFO"))
        
        # Create sub-configurations
        database = DatabaseConfig(**data.get("database", {}))
        api = APIConfig(**data.get("api", {}))
        performance = PerformanceConfig(**data.get("performance", {}))
        coverage = CoverageConfig(**data.get("coverage", {}))
        
        # Create main config
        config = TestConfig(
            environment=environment,
            log_level=log_level,
            debug_mode=data.get("debug_mode", False),
            parallel_workers=data.get("parallel_workers", 1),
            test_timeout=data.get("test_timeout", 300.0),
            retry_failed_tests=data.get("retry_failed_tests", True),
            max_retries=data.get("max_retries", 3),
            database=database,
            api=api,
            performance=performance,
            coverage=coverage,
            test_data_size=data.get("test_data_size", "small"),
            generate_fake_data=data.get("generate_fake_data", True),
            data_cleanup=data.get("data_cleanup", True),
            generate_reports=data.get("generate_reports", True),
            report_formats=data.get("report_formats", ["json", "html"]),
            report_output_dir=data.get("report_output_dir", "test_reports")
        )
        
        return config
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        if not self.config:
            return
        
        # Environment
        if os.getenv("TEST_ENVIRONMENT"):
            self.config.environment = TestEnvironment(os.getenv("TEST_ENVIRONMENT"))
        
        # Log level
        if os.getenv("TEST_LOG_LEVEL"):
            self.config.log_level = LogLevel(os.getenv("TEST_LOG_LEVEL"))
        
        # Debug mode
        if os.getenv("TEST_DEBUG"):
            self.config.debug_mode = os.getenv("TEST_DEBUG").lower() == "true"
        
        # Parallel workers
        if os.getenv("TEST_PARALLEL_WORKERS"):
            self.config.parallel_workers = int(os.getenv("TEST_PARALLEL_WORKERS"))
        
        # Test timeout
        if os.getenv("TEST_TIMEOUT"):
            self.config.test_timeout = float(os.getenv("TEST_TIMEOUT"))
        
        # Database URL
        if os.getenv("TEST_DATABASE_URL"):
            self.config.database.url = os.getenv("TEST_DATABASE_URL")
        
        # API base URL
        if os.getenv("TEST_API_BASE_URL"):
            self.config.api.base_url = os.getenv("TEST_API_BASE_URL")
        
        # Performance thresholds
        if os.getenv("TEST_PERFORMANCE_TIMEOUT"):
            self.config.performance.timeout_threshold = float(os.getenv("TEST_PERFORMANCE_TIMEOUT"))
        
        if os.getenv("TEST_MEMORY_THRESHOLD"):
            self.config.performance.memory_threshold = float(os.getenv("TEST_MEMORY_THRESHOLD"))
        
        # Coverage threshold
        if os.getenv("TEST_COVERAGE_THRESHOLD"):
            self.config.coverage.min_coverage = float(os.getenv("TEST_COVERAGE_THRESHOLD"))
    
    def get_config(self) -> TestConfig:
        """Get current configuration."""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        if not self.config:
            return
        
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            elif hasattr(self.config.database, key):
                setattr(self.config.database, key, value)
            elif hasattr(self.config.api, key):
                setattr(self.config.api, key, value)
            elif hasattr(self.config.performance, key):
                setattr(self.config.performance, key, value)
            elif hasattr(self.config.coverage, key):
                setattr(self.config.coverage, key, value)
    
    def save_config(self, file_path: Optional[str] = None):
        """Save current configuration to file."""
        if not self.config:
            return
        
        save_path = file_path or self.config_file
        config_dict = self._config_to_dict(self.config)
        
        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def _config_to_dict(self, config: TestConfig) -> Dict[str, Any]:
        """Convert TestConfig object to dictionary."""
        return {
            "environment": config.environment.value,
            "log_level": config.log_level.value,
            "debug_mode": config.debug_mode,
            "parallel_workers": config.parallel_workers,
            "test_timeout": config.test_timeout,
            "retry_failed_tests": config.retry_failed_tests,
            "max_retries": config.max_retries,
            "database": {
                "url": config.database.url,
                "echo": config.database.echo,
                "pool_size": config.database.pool_size,
                "max_overflow": config.database.max_overflow,
                "timeout": config.database.timeout
            },
            "api": {
                "base_url": config.api.base_url,
                "timeout": config.api.timeout,
                "retry_attempts": config.api.retry_attempts,
                "headers": config.api.headers
            },
            "performance": {
                "timeout_threshold": config.performance.timeout_threshold,
                "memory_threshold": config.performance.memory_threshold,
                "cpu_threshold": config.performance.cpu_threshold,
                "throughput_threshold": config.performance.throughput_threshold,
                "max_iterations": config.performance.max_iterations
            },
            "coverage": {
                "min_coverage": config.coverage.min_coverage,
                "exclude_patterns": config.coverage.exclude_patterns,
                "include_patterns": config.coverage.include_patterns
            },
            "test_data_size": config.test_data_size,
            "generate_fake_data": config.generate_fake_data,
            "data_cleanup": config.data_cleanup,
            "generate_reports": config.generate_reports,
            "report_formats": config.report_formats,
            "report_output_dir": config.report_output_dir
        }
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any errors."""
        errors = []
        
        if not self.config:
            errors.append("Configuration is None")
            return errors
        
        # Validate environment
        if not isinstance(self.config.environment, TestEnvironment):
            errors.append("Invalid environment type")
        
        # Validate log level
        if not isinstance(self.config.log_level, LogLevel):
            errors.append("Invalid log level type")
        
        # Validate numeric values
        if self.config.parallel_workers < 1:
            errors.append("Parallel workers must be at least 1")
        
        if self.config.test_timeout <= 0:
            errors.append("Test timeout must be positive")
        
        if self.config.max_retries < 0:
            errors.append("Max retries must be non-negative")
        
        # Validate performance thresholds
        if self.config.performance.timeout_threshold <= 0:
            errors.append("Performance timeout threshold must be positive")
        
        if self.config.performance.memory_threshold <= 0:
            errors.append("Memory threshold must be positive")
        
        if self.config.performance.cpu_threshold <= 0 or self.config.performance.cpu_threshold > 100:
            errors.append("CPU threshold must be between 0 and 100")
        
        # Validate coverage threshold
        if self.config.coverage.min_coverage < 0 or self.config.coverage.min_coverage > 100:
            errors.append("Coverage threshold must be between 0 and 100")
        
        return errors

# Global configuration manager instance
config_manager = TestConfigManager()

def get_test_config() -> TestConfig:
    """Get the global test configuration."""
    return config_manager.get_config()

def update_test_config(updates: Dict[str, Any]):
    """Update the global test configuration."""
    config_manager.update_config(updates)

def save_test_config(file_path: Optional[str] = None):
    """Save the global test configuration."""
    config_manager.save_config(file_path)

# Default configuration file
DEFAULT_CONFIG = {
    "environment": "local",
    "log_level": "INFO",
    "debug_mode": False,
    "parallel_workers": 1,
    "test_timeout": 300.0,
    "retry_failed_tests": True,
    "max_retries": 3,
    "database": {
        "url": "sqlite:///:memory:",
        "echo": False,
        "pool_size": 5,
        "max_overflow": 10,
        "timeout": 30
    },
    "api": {
        "base_url": "http://localhost:8000",
        "timeout": 30,
        "retry_attempts": 3,
        "headers": {}
    },
    "performance": {
        "timeout_threshold": 1.0,
        "memory_threshold": 100.0,
        "cpu_threshold": 80.0,
        "throughput_threshold": 100.0,
        "max_iterations": 1000
    },
    "coverage": {
        "min_coverage": 80.0,
        "exclude_patterns": [
            "*/tests/*",
            "*/test_*",
            "*/__pycache__/*",
            "*/migrations/*"
        ],
        "include_patterns": ["*.py"]
    },
    "test_data_size": "small",
    "generate_fake_data": True,
    "data_cleanup": True,
    "generate_reports": True,
    "report_formats": ["json", "html"],
    "report_output_dir": "test_reports"
}

if __name__ == "__main__":
    # Create default config file if it doesn't exist
    config_file = "tests/config/test_config.json"
    if not Path(config_file).exists():
        Path(config_file).parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        print(f"Created default config file: {config_file}")
    
    # Validate configuration
    errors = config_manager.validate_config()
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration is valid")
