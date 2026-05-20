"""
Test configuration and settings for copywriting service tests.
"""
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


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
    single_request_max_time: float = 1.0
    batch_request_max_time: float = 5.0
    concurrent_request_max_time: float = 10.0
    load_test_max_time: float = 60.0
    memory_max_increase_mb: float = 200.0
    response_time_std_dev_ratio: float = 0.5
    min_success_rate: float = 0.95
    max_concurrent_requests: int = 100
    max_batch_size: int = 20


@dataclass
class CoverageThresholds:
    """Coverage thresholds for testing."""
    min_line_coverage: float = 90.0
    min_branch_coverage: float = 85.0
    min_function_coverage: float = 95.0
    min_error_scenario_coverage: float = 100.0


@dataclass
class SecurityThresholds:
    """Security testing thresholds."""
    max_input_length: int = 1000
    max_malicious_inputs: int = 50
    min_security_test_coverage: float = 100.0
    max_error_exposure: int = 0  # No sensitive data in errors


@dataclass
class TestConfig:
    """Main test configuration."""
    environment: TestEnvironment = TestEnvironment.UNIT
    category: TestCategory = TestCategory.FAST
    performance: PerformanceThresholds = PerformanceThresholds()
    coverage: CoverageThresholds = CoverageThresholds()
    security: SecurityThresholds = SecurityThresholds()
    
    # Test data settings
    test_data_cache_size: int = 1000
    test_data_cleanup_interval: int = 100
    
    # Mock settings
    mock_ai_delay: float = 0.1
    mock_ai_failure_rate: float = 0.0
    
    # Logging settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Parallel execution settings
    max_parallel_tests: int = 4
    parallel_timeout: int = 300
    
    # Database settings
    test_database_url: str = "sqlite:///:memory:"
    test_database_cleanup: bool = True
    
    # Cache settings
    test_cache_backend: str = "memory"
    test_cache_ttl: int = 300


class TestConfigManager:
    """Manages test configuration."""
    
    def __init__(self):
        self._config = TestConfig()
        self._environment_overrides = {}
        self._load_environment_variables()
    
    def _load_environment_variables(self):
        """Load configuration from environment variables."""
        env_mappings = {
            'TEST_ENVIRONMENT': ('environment', TestEnvironment),
            'TEST_CATEGORY': ('category', TestCategory),
            'TEST_LOG_LEVEL': ('log_level', str),
            'TEST_MAX_PARALLEL': ('max_parallel_tests', int),
            'TEST_DATABASE_URL': ('test_database_url', str),
            'TEST_CACHE_BACKEND': ('test_cache_backend', str),
        }
        
        for env_var, (attr_name, attr_type) in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                if attr_type == TestEnvironment:
                    value = TestEnvironment(value)
                elif attr_type == TestCategory:
                    value = TestCategory(value)
                elif attr_type == int:
                    value = int(value)
                elif attr_type == float:
                    value = float(value)
                
                setattr(self._config, attr_name, value)
    
    def get_config(self) -> TestConfig:
        """Get current test configuration."""
        return self._config
    
    def update_config(self, **updates):
        """Update test configuration."""
        for key, value in updates.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
    
    def get_performance_thresholds(self) -> PerformanceThresholds:
        """Get performance thresholds."""
        return self._config.performance
    
    def get_coverage_thresholds(self) -> CoverageThresholds:
        """Get coverage thresholds."""
        return self._config.coverage
    
    def get_security_thresholds(self) -> SecurityThresholds:
        """Get security thresholds."""
        return self._config.security
    
    def is_environment(self, environment: TestEnvironment) -> bool:
        """Check if current environment matches."""
        return self._config.environment == environment
    
    def is_category(self, category: TestCategory) -> bool:
        """Check if current category matches."""
        return self._config.category == category
    
    def should_run_slow_tests(self) -> bool:
        """Check if slow tests should run."""
        return os.getenv('RUN_SLOW_TESTS', 'false').lower() == 'true'
    
    def should_run_performance_tests(self) -> bool:
        """Check if performance tests should run."""
        return os.getenv('RUN_PERFORMANCE_TESTS', 'false').lower() == 'true'
    
    def should_run_load_tests(self) -> bool:
        """Check if load tests should run."""
        return os.getenv('RUN_LOAD_TESTS', 'false').lower() == 'true'


# Global test configuration manager
test_config_manager = TestConfigManager()


class TestDataConfig:
    """Configuration for test data generation."""
    
    def __init__(self):
        self.sample_products = [
            "Zapatos deportivos de alta gama",
            "Smartphone de última generación",
            "Laptop gaming profesional",
            "Reloj inteligente para fitness",
            "Auriculares inalámbricos premium"
        ]
        
        self.sample_platforms = [
            "Instagram", "Facebook", "Twitter", "LinkedIn", "TikTok"
        ]
        
        self.sample_tones = [
            "inspirational", "informative", "playful", "professional", "urgent"
        ]
        
        self.sample_audiences = [
            "Jóvenes activos", "Profesionales", "Familias", "Tech enthusiasts", "Fitness lovers"
        ]
        
        self.sample_key_points = [
            ["Comodidad", "Estilo", "Durabilidad"],
            ["Innovación", "Rendimiento", "Diseño"],
            ["Calidad", "Precio", "Garantía"],
            ["Tecnología", "Conectividad", "Batería"],
            ["Salud", "Bienestar", "Monitoreo"]
        ]
        
        self.sample_instructions = [
            "Enfatiza la innovación",
            "Destaca la calidad",
            "Resalta el valor",
            "Menciona la garantía",
            "Enfócate en los beneficios"
        ]
        
        self.sample_restrictions = [
            ["no mencionar precio"],
            ["evitar términos técnicos"],
            ["no usar superlativos"],
            ["mantener tono profesional"],
            ["incluir call to action"]
        ]
    
    def get_random_product(self) -> str:
        """Get a random product description."""
        import random
        return random.choice(self.sample_products)
    
    def get_random_platform(self) -> str:
        """Get a random platform."""
        import random
        return random.choice(self.sample_platforms)
    
    def get_random_tone(self) -> str:
        """Get a random tone."""
        import random
        return random.choice(self.sample_tones)
    
    def get_random_audience(self) -> str:
        """Get a random audience."""
        import random
        return random.choice(self.sample_audiences)
    
    def get_random_key_points(self) -> List[str]:
        """Get random key points."""
        import random
        return random.choice(self.sample_key_points)
    
    def get_random_instructions(self) -> str:
        """Get random instructions."""
        import random
        return random.choice(self.sample_instructions)
    
    def get_random_restrictions(self) -> List[str]:
        """Get random restrictions."""
        import random
        return random.choice(self.sample_restrictions)


# Global test data configuration
test_data_config = TestDataConfig()


class MockConfig:
    """Configuration for mock services."""
    
    def __init__(self):
        self.ai_service_delay = 0.1
        self.ai_service_failure_rate = 0.0
        self.database_delay = 0.05
        self.cache_delay = 0.01
        self.network_delay = 0.1
        self.network_failure_rate = 0.0
    
    def get_ai_service_config(self) -> Dict[str, Any]:
        """Get AI service mock configuration."""
        return {
            "delay": self.ai_service_delay,
            "failure_rate": self.ai_service_failure_rate,
            "response_template": {
                "variants": [{"headline": "Mock Headline", "primary_text": "Mock Content"}],
                "model_used": "gpt-3.5-turbo",
                "generation_time": self.ai_service_delay,
                "extra_metadata": {"tokens_used": 50}
            }
        }
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database mock configuration."""
        return {
            "delay": self.database_delay,
            "connection_string": "sqlite:///:memory:",
            "cleanup_after_test": True
        }
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache mock configuration."""
        return {
            "delay": self.cache_delay,
            "backend": "memory",
            "ttl": 300
        }


# Global mock configuration
mock_config = MockConfig()


class TestMarkers:
    """Test markers configuration."""
    
    MARKERS = {
        "slow": "marks tests as slow (deselect with '-m \"not slow\"')",
        "integration": "marks tests as integration tests",
        "performance": "marks tests as performance tests",
        "security": "marks tests as security tests",
        "monitoring": "marks tests as monitoring tests",
        "load": "marks tests as load tests",
        "unit": "marks tests as unit tests",
        "benchmark": "marks tests as benchmark tests",
        "example": "marks tests as example tests",
        "critical": "marks tests as critical",
        "optional": "marks tests as optional"
    }
    
    @classmethod
    def get_marker_config(cls) -> Dict[str, str]:
        """Get marker configuration for pytest."""
        return cls.MARKERS
    
    @classmethod
    def get_marker_filter(cls, environment: TestEnvironment) -> List[str]:
        """Get marker filter based on environment."""
        filters = {
            TestEnvironment.UNIT: ["unit"],
            TestEnvironment.INTEGRATION: ["integration"],
            TestEnvironment.PERFORMANCE: ["performance", "benchmark"],
            TestEnvironment.SECURITY: ["security"],
            TestEnvironment.MONITORING: ["monitoring"],
            TestEnvironment.LOAD: ["load"]
        }
        return filters.get(environment, [])


class TestFixtures:
    """Test fixtures configuration."""
    
    @staticmethod
    def get_fixture_scope(test_type: str) -> str:
        """Get fixture scope based on test type."""
        scopes = {
            "unit": "function",
            "integration": "class",
            "performance": "module",
            "security": "function",
            "monitoring": "class"
        }
        return scopes.get(test_type, "function")
    
    @staticmethod
    def get_fixture_autouse(test_type: str) -> bool:
        """Get fixture autouse setting based on test type."""
        autouse_types = ["performance", "monitoring"]
        return test_type in autouse_types


class TestReporting:
    """Test reporting configuration."""
    
    def __init__(self):
        self.html_report = True
        self.xml_report = True
        self.json_report = True
        self.coverage_report = True
        self.performance_report = True
        self.security_report = True
    
    def get_report_config(self) -> Dict[str, Any]:
        """Get report configuration."""
        return {
            "html_report": self.html_report,
            "xml_report": self.xml_report,
            "json_report": self.json_report,
            "coverage_report": self.coverage_report,
            "performance_report": self.performance_report,
            "security_report": self.security_report
        }
    
    def get_report_paths(self) -> Dict[str, str]:
        """Get report file paths."""
        return {
            "html": "reports/test_report.html",
            "xml": "reports/test_report.xml",
            "json": "reports/test_report.json",
            "coverage": "reports/coverage_report.html",
            "performance": "reports/performance_report.json",
            "security": "reports/security_report.json"
        }


# Global test reporting configuration
test_reporting = TestReporting()
